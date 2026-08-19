use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::filter::is_codec_drain_boundary,
    error::Result,
    pad::SrcPad,
    platform::cuda::CudaDevice,
    pool::UnboundObjectPool,
};

/// Errors specific to `CudaScaler`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaScalerError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error(
        "filter `{0}` not found in this ffmpeg build — a CUDA-enabled build \
         (`--enable-ffnvcodec` plus `--enable-cuda-nvcc` or `--enable-cuda-llvm`) is required"
    )]
    FilterNotFound(&'static str),

    #[error("CudaScaler only scales CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error("CudaScaler only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// The frame carries no hardware frames context, so it did not come from
    /// a CUDA producer at all.
    #[error("CUDA frame has no hardware frames context")]
    MissingFramesContext,

    /// The frame was allocated against a different CUDA context than this
    /// element's. Its device pointers mean nothing here.
    #[error("CUDA frame belongs to a different CUDA context than this CudaScaler")]
    ForeignContext,

    #[error("CudaScaler only scales NV12 surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    #[error("failed to allocate the buffer source parameters")]
    BufferSrcParamsAlloc,

    #[error("failed to configure the buffer source (code {0})")]
    BufferSrcConfig(i32),

    #[error("the filter graph rejected a frame (code {0})")]
    BufferSrcPush(i32),

    #[error("failed to read a scaled frame out of the filter graph (code {0})")]
    BufferSinkPull(i32),
}

/// Interpolation `scale_cuda` uses when resampling — the CUDA counterpart to
/// the `libswscale` flags [`crate::elements::SwScaler`] takes, exposed for the
/// same reason: downscaling 1080p to 720p looks visibly different between
/// nearest and lanczos, and the caller is the one who knows whether this
/// scaler sits in a quality-sensitive path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaScalerInterp {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos,
}

impl CudaScalerInterp {
    /// `scale_cuda`'s own `interp_algo` option values.
    fn algo(self) -> u32 {
        match self {
            Self::Nearest => 1,
            Self::Bilinear => 2,
            Self::Bicubic => 3,
            Self::Lanczos => 4,
        }
    }
}

/// Resizes GPU-resident `Pixel::CUDA` NV12 `Video` frames without ever
/// touching the CPU, through libavfilter's `scale_cuda`. A `Filter`:
/// receives via `Sink`, pushes the scaled frame into its own single src pad.
///
/// This is what makes a hardware transcode at a different resolution
/// possible at all — `CudaDecoder -> CudaScaler -> CudaEncoder` stays on the
/// GPU end to end, where the only alternative was
/// [`crate::elements::CudaDownload`] plus [`crate::elements::SwScaler`] plus
/// [`crate::elements::CudaUpload`], i.e. two PCIe crossings and a
/// `libswscale` pass per frame.
///
/// # NV12 only, resize only
///
/// Everything on this crate's CUDA path speaks NV12 (see
/// [`crate::elements::CudaUpload`]'s own note), so this neither takes nor
/// produces anything else. `scale_cuda` can convert *within* the YUV family,
/// but not between RGB and YUV in either direction: its PTX module carries
/// no kernel for those pairs, so the conversion fails as
/// `Unsupported conversion: bgr0 -> nv12` rather than falling back to
/// anything.
///
/// That is a narrower limit than it sounds. NVENC ingests RGB surfaces
/// directly and converts them in hardware, and `scale_cuda` resizes an RGB
/// surface fine as long as it stays RGB — so a BGRA capture path never needs
/// the conversion this filter refuses. What does need NV12 is
/// [`crate::elements::CudaRenderer`], whose trait hands the graphics side
/// separate luma/chroma pointers.
///
/// # The graph is built from the first frame
///
/// libavfilter has to be told the *input* surface pool, not just its size,
/// before it can configure a hardware filter — and that pool belongs to
/// whichever element upstream allocated the frame. So the graph is built
/// lazily from the first frame that arrives and rebuilt if a later frame's
/// dimensions or pool change, exactly as [`crate::elements::SwScaler`] rebuilds
/// its `libswscale` context. The output size is fixed at construction.
pub struct CudaScaler {
    pp_log: PpLog,
    name: Arc<str>,
    /// This element's own reference to the shared context, released in
    /// `Drop`. Held for `device_ctx`'s sake: the pointer below stays valid
    /// only as long as this reference does.
    hw_device_ctx: *mut ffi::AVBufferRef,
    /// Captured at construction so an incoming frame can be checked against
    /// this element's own CUDA context. Only ever compared, same as
    /// [`crate::elements::CudaEncoder`]'s.
    device_ctx: *const ffi::AVHWDeviceContext,
    width: u32,
    height: u32,
    interp: CudaScalerInterp,
    /// `None` until the first frame arrives — see this type's own docs.
    graph: Option<GraphState>,
    pad: SrcPad,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the scaled surface
    /// itself comes from `scale_cuda`'s own output pool. Same split as
    /// [`crate::elements::CudaUpload`].
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

/// One configured `buffer -> scale_cuda -> buffersink` graph, plus what it
/// was configured *for*, so a changed input can be detected.
struct GraphState {
    /// Owns the filter contexts below; freeing it frees them, so it must
    /// outlive them — which it does, both being dropped together with the
    /// element.
    _graph: ffmpeg::filter::Graph,
    source: ffmpeg::filter::Context,
    sink: ffmpeg::filter::Context,
    input_width: u32,
    input_height: u32,
    /// The exact input pool this graph was configured against. A same-sized
    /// frame from a *different* pool — an upstream decoder rebuilt after a
    /// seek, say — still needs a rebuild, and comparing sizes alone would
    /// miss it.
    input_frames_ctx: *mut ffi::AVBufferRef,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no thread
// affinity of its own, `device_ctx` is only ever compared, and a filter
// graph and its contexts likewise have no thread affinity — ffmpeg-next
// already marks `filter::Graph` itself `Send`. `&mut self` on every method
// that touches them rules out concurrent access. Same reasoning as
// `CudaUpload`.
unsafe impl Send for CudaScaler {}

impl CudaScaler {
    /// `device` must be the same [`CudaDevice`] the upstream CUDA elements
    /// were built from — a frame allocated against another context is
    /// rejected rather than scaled from meaningless pointers. This element
    /// takes its own FFmpeg reference, so `device` itself need not outlive
    /// the call.
    ///
    /// `width`/`height` are what every output frame will be; the input side
    /// is learned from whatever frames actually arrive, so this needs no
    /// input dimensions up front (see this type's own docs).
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        width: u32,
        height: u32,
        interp: CudaScalerInterp,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaScaler, &name, None);

        let hw_device_ctx = unsafe { ffi::av_buffer_ref(device.as_ptr()) };
        let device_ctx = unsafe { (*hw_device_ctx).data as *const ffi::AVHWDeviceContext };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "created: dst={width}x{height}, interp={interp:?}");
        Self {
            name,
            pp_log,
            hw_device_ctx,
            device_ctx,
            width,
            height,
            interp,
            graph: None,
            pad,
            pool,
        }
    }

    /// Rejects anything that is not a CUDA NV12 frame from this element's own
    /// device, so a bad frame fails here rather than as an opaque libavfilter
    /// error code — or, worse, as device pointers read against the wrong
    /// context. Returns the frame's own frames context, which is what the
    /// graph has to be configured against.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<*mut ffi::AVBufferRef, CudaScalerError> {
        if frame.format() != ffmpeg::format::Pixel::CUDA {
            return Err(CudaScalerError::UnsupportedFormat(frame.format()));
        }
        unsafe {
            let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() {
                return Err(CudaScalerError::MissingFramesContext);
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            if !std::ptr::eq((*frames_ctx).device_ctx, self.device_ctx) {
                return Err(CudaScalerError::ForeignContext);
            }
            let sw_format = (*frames_ctx).sw_format;
            if sw_format != ffi::AVPixelFormat::AV_PIX_FMT_NV12 {
                return Err(CudaScalerError::UnsupportedSurfaceFormat(
                    ffmpeg::format::Pixel::from(sw_format),
                ));
            }
            Ok(frames_ref)
        }
    }

    /// Whether the configured graph (if any) already matches `frame`'s own
    /// dimensions and surface pool.
    fn graph_matches(
        &self,
        frame: &ffmpeg::frame::Video,
        frames_ctx: *mut ffi::AVBufferRef,
    ) -> bool {
        match &self.graph {
            Some(state) => {
                state.input_width == frame.width()
                    && state.input_height == frame.height()
                    && std::ptr::eq(state.input_frames_ctx, frames_ctx)
            }
            None => false,
        }
    }

    fn build_graph(
        &mut self,
        frame: &ffmpeg::frame::Video,
        frames_ctx: *mut ffi::AVBufferRef,
    ) -> std::result::Result<(), CudaScalerError> {
        let mut graph = ffmpeg::filter::Graph::new();

        let buffer =
            ffmpeg::filter::find("buffer").ok_or(CudaScalerError::FilterNotFound("buffer"))?;
        let scale = ffmpeg::filter::find("scale_cuda")
            .ok_or(CudaScalerError::FilterNotFound("scale_cuda"))?;
        let buffersink = ffmpeg::filter::find("buffersink")
            .ok_or(CudaScalerError::FilterNotFound("buffersink"))?;

        // `scale_cuda` neither reorders nor retimes, and nothing between the
        // source and the sink rescales timestamps, so a nominal 1/1 time base
        // carries each frame's own pts through numerically unchanged. This
        // element is not the place that decides what a pts *means*.
        let args = format!(
            "video_size={}x{}:pix_fmt={}:time_base=1/1:pixel_aspect=1/1",
            frame.width(),
            frame.height(),
            ffi::AVPixelFormat::AV_PIX_FMT_CUDA as i32,
        );
        let mut source = graph.add(&buffer, "in", &args)?;

        // The size and format above describe the frames; *this* is what tells
        // libavfilter which CUDA pool they come from, and without it
        // `scale_cuda` has no device to configure itself on.
        unsafe {
            let params = ffi::av_buffersrc_parameters_alloc();
            if params.is_null() {
                return Err(CudaScalerError::BufferSrcParamsAlloc);
            }
            (*params).hw_frames_ctx = frames_ctx;
            let code = ffi::av_buffersrc_parameters_set(source.as_mut_ptr(), params);
            ffi::av_free(params.cast());
            if code < 0 {
                return Err(CudaScalerError::BufferSrcConfig(code));
            }
        }

        let mut scale = graph.add(
            &scale,
            "scale",
            &format!(
                "w={}:h={}:interp_algo={}",
                self.width,
                self.height,
                self.interp.algo()
            ),
        )?;
        let mut sink = graph.add(&buffersink, "out", "")?;

        source.link(0, &mut scale, 0);
        scale.link(0, &mut sink, 0);
        graph.validate()?;

        self.graph = Some(GraphState {
            _graph: graph,
            source,
            sink,
            input_width: frame.width(),
            input_height: frame.height(),
            input_frames_ctx: frames_ctx,
        });
        Ok(())
    }

    fn scale(&mut self, frame: &ffmpeg::frame::Video) -> Result<()> {
        let frames_ctx = self
            .validate(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        if !self.graph_matches(frame, frames_ctx) {
            if self.graph.is_some() {
                // A live source can renegotiate mid-stream, and a decoder
                // rebuilt after a seek hands out a new pool — same reason
                // `SwScaler` rebuilds its own context, and worth seeing in a
                // log when output suddenly looks stretched.
                pp_info!(
                    self,
                    "input changed to {}x{}, rebuilding the filter graph",
                    frame.width(),
                    frame.height()
                );
            } else {
                pp_debug!(
                    self,
                    "input is {}x{}, building the filter graph",
                    frame.width(),
                    frame.height()
                );
            }
            self.build_graph(frame, frames_ctx).inspect_err(|error| {
                pp_error!(self, "failed to build the filter graph: {error}")
            })?;
        }

        let source = unsafe {
            self.graph
                .as_mut()
                .expect("built or confirmed matching above")
                .source
                .as_mut_ptr()
        };
        // `av_buffersrc_write_frame`, not `add_frame`: the latter takes over
        // the caller's reference and resets the frame, and this frame is
        // shared — downstream `Arc` clones and the upstream pool both still
        // expect it intact.
        let code = unsafe { ffi::av_buffersrc_write_frame(source, frame.as_ptr()) };
        if code < 0 {
            pp_error!(self, "av_buffersrc_write_frame failed: {code}");
            return Err(CudaScalerError::BufferSrcPush(code).into());
        }
        self.drain()
    }

    /// Pulls everything the graph is willing to produce right now. One frame
    /// in means one frame out for `scale_cuda`, but a graph is allowed to
    /// hold or emit more than that, so this drains rather than reading once.
    fn drain(&mut self) -> Result<()> {
        let sink = unsafe {
            self.graph
                .as_mut()
                .expect("only called with a configured graph")
                .sink
                .as_mut_ptr()
        };
        loop {
            let mut output = self.pool.get();
            let code = unsafe {
                let ptr = output.as_mut_ptr();
                // The pooled wrapper may still reference the previous frame's
                // surface; releasing it here is what returns that surface to
                // the filter's pool rather than leaking it.
                ffi::av_frame_unref(ptr);
                ffi::av_buffersink_get_frame(sink, ptr)
            };
            if code < 0 {
                if is_codec_drain_boundary(&ffmpeg::Error::from(code)) {
                    return Ok(());
                }
                pp_error!(self, "av_buffersink_get_frame failed: {code}");
                return Err(CudaScalerError::BufferSinkPull(code).into());
            }
            self.pad.push(MediaBuffer::Video(Arc::new(output)))?;
        }
    }
}

impl Element for CudaScaler {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaScaler
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaScaler {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaScaler {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.scale(&frame),
            MediaBuffer::Eos => {
                // `scale_cuda` holds nothing back, but a graph that never
                // sees EOF never reports it either, and this is the one
                // place anything buffered inside it can still come out.
                if self.graph.is_some() {
                    let source = unsafe {
                        self.graph
                            .as_mut()
                            .expect("checked above")
                            .source
                            .as_mut_ptr()
                    };
                    let code = unsafe {
                        ffi::av_buffersrc_add_frame_flags(source, std::ptr::null_mut(), 0)
                    };
                    if code < 0 {
                        pp_error!(self, "failed to flush the filter graph: {code}");
                        return Err(CudaScalerError::BufferSrcPush(code).into());
                    }
                    self.drain()?;
                }
                self.pad.push(MediaBuffer::Eos)
            }
            other => Err(CudaScalerError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame spatial transform,
        // same reasoning as `SwScaler::control`.
        self.pad.control(msg)
    }
}

impl Drop for CudaScaler {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw_device_ctx");
        unsafe { ffi::av_buffer_unref(&mut self.hw_device_ctx) };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        elements::{CudaDecoder, CudaDownload, CudaUpload},
        test_support::try_test_video,
    };

    fn try_cuda_device() -> Option<CudaDevice> {
        match CudaDevice::new() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: no usable CUDA device on this machine ({error})");
                None
            }
        }
    }

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    fn pooled(frame: ffmpeg::frame::Video) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        MediaBuffer::Video(Arc::new(slot))
    }

    /// A flat mid-grey NV12 frame: after any interpolation every output pixel
    /// must still be that same value, so a scaled result can be checked
    /// exactly rather than "looks plausible".
    fn nv12_flat(width: u32, height: u32, luma: u8, pts: i64) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        frame.set_pts(Some(pts));
        let y_stride = frame.stride(0);
        let y = frame.data_mut(0);
        for row in 0..height as usize {
            y[row * y_stride..row * y_stride + width as usize].fill(luma);
        }
        let uv_stride = frame.stride(1);
        let uv = frame.data_mut(1);
        for row in 0..(height as usize / 2) {
            uv[row * uv_stride..row * uv_stride + width as usize].fill(128);
        }
        frame
    }

    /// Uploads one synthetic frame and returns it as a CUDA `MediaBuffer`,
    /// so a test can feed the scaler without a decoder.
    fn cuda_frame(
        device: &CudaDevice,
        width: u32,
        height: u32,
        luma: u8,
        pts: i64,
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) = CudaUpload::new("upload", device, width, height) else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        upload
            .consume(pooled(nv12_flat(width, height, luma, pts)))
            .expect("upload");
        let frame = uploaded.lock().unwrap().remove(0);
        Some(frame)
    }

    /// The contract: the frame comes out at the configured size, still on the
    /// GPU, still carrying its timestamp — and the pixels survive the resize.
    /// The `CudaDownload` on the tail is what makes the last part checkable at
    /// all, since a CUDA surface has no readable planes.
    #[test]
    fn a_cuda_frame_is_resized_on_the_gpu_and_keeps_its_pts() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let Some(frame) = cuda_frame(&device, 128, 128, 200, 777) else {
            return;
        };

        let mut scaler = CudaScaler::new("scaler", &device, 64, 64, CudaScalerInterp::Bilinear);
        let mut download = CudaDownload::new("download", &device, 64, 64);
        let received = capture(&mut download);
        scaler.src_pads()[0].link(Box::new(download));

        scaler.consume(frame).expect("scale");
        scaler.consume(MediaBuffer::Eos).expect("eos");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(scaled) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(scaled.width(), 64);
        assert_eq!(scaled.height(), 64);
        assert_eq!(scaled.pts(), Some(777), "the scaler dropped the pts");
        let stride = scaled.stride(0);
        for row in 0..64usize {
            for column in 0..64usize {
                let luma = scaled.data(0)[row * stride + column];
                assert_eq!(
                    luma, 200,
                    "a flat frame must stay flat through the resize (row {row}, column {column})"
                );
            }
        }
        assert!(
            received.last().is_some_and(MediaBuffer::is_eos),
            "Eos was not forwarded"
        );
    }

    /// The point of the element: real NVDEC output is resized without ever
    /// leaving the GPU, which is what a hardware transcode needs.
    #[test]
    fn decoded_nvdec_frames_are_resized_without_leaving_the_gpu() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let Some(path) = try_test_video() else {
            return;
        };
        let frames = decode_some_frames(&device, &path);
        if frames.is_empty() {
            eprintln!("skipping: NVDEC produced no frames for the configured fixture");
            return;
        }
        let expected_pts: Vec<_> = frames
            .iter()
            .map(|buf| match buf {
                MediaBuffer::Video(frame) => frame.pts(),
                other => panic!("the decoder emitted a {}", other.kind()),
            })
            .collect();

        let mut scaler = CudaScaler::new("scaler", &device, 320, 180, CudaScalerInterp::Bicubic);
        let received = capture(&mut scaler);
        for frame in frames {
            scaler.consume(frame).expect("scale failed");
        }

        let received = received.lock().unwrap();
        assert_eq!(received.len(), expected_pts.len());
        for (buf, pts) in received.iter().zip(expected_pts) {
            let MediaBuffer::Video(frame) = buf else {
                panic!("expected a Video buffer, got {}", buf.kind());
            };
            assert_eq!(
                frame.format(),
                ffmpeg::format::Pixel::CUDA,
                "the scaled frame left the GPU"
            );
            assert_eq!(frame.width(), 320);
            assert_eq!(frame.height(), 180);
            assert_eq!(frame.pts(), pts, "the scaler dropped the pts");
        }
    }

    /// A mid-stream input change must rebuild the graph rather than fail or
    /// keep scaling from a stale one — same contract `SwScaler` has for its
    /// `libswscale` context.
    #[test]
    fn rebuilds_its_graph_when_the_input_changes_mid_stream() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let Some(first) = cuda_frame(&device, 128, 128, 100, 0) else {
            return;
        };
        let Some(second) = cuda_frame(&device, 256, 64, 100, 1) else {
            return;
        };

        let mut scaler = CudaScaler::new("scaler", &device, 64, 64, CudaScalerInterp::Bilinear);
        let received = capture(&mut scaler);
        scaler.consume(first).expect("first frame must scale");
        scaler
            .consume(second)
            .expect("a differently-shaped second frame must still scale");

        let received = received.lock().unwrap();
        assert_eq!(received.len(), 2);
        for buf in received.iter() {
            let MediaBuffer::Video(frame) = buf else {
                panic!("expected a Video buffer, got {}", buf.kind());
            };
            assert_eq!(frame.width(), 64);
            assert_eq!(frame.height(), 64);
        }
    }

    /// A CPU frame has no device pointers, and one from another context has
    /// pointers that mean nothing here — both must be refused before
    /// libavfilter ever sees them.
    #[test]
    fn a_cpu_frame_and_a_foreign_context_frame_are_typed_errors() {
        let Some(device) = try_cuda_device() else {
            return;
        };
        let mut scaler = CudaScaler::new("scaler", &device, 64, 64, CudaScalerInterp::Bilinear);
        let _received = capture(&mut scaler);

        let error = scaler
            .consume(pooled(nv12_flat(128, 128, 100, 0)))
            .expect_err("a CPU frame must not be scaled");
        assert!(
            error.to_string().contains("only scales CUDA frames"),
            "expected UnsupportedFormat, got {error}"
        );

        let Some(other_device) = try_cuda_device() else {
            return;
        };
        let Some(foreign) = cuda_frame(&other_device, 128, 128, 100, 0) else {
            return;
        };
        let error = scaler
            .consume(foreign)
            .expect_err("a frame from a foreign CUDA context must not be scaled");
        assert!(
            error.to_string().contains("different CUDA context"),
            "expected ForeignContext, got {error}"
        );
    }

    /// Decodes a handful of real frames, so the test above asserts against
    /// genuine NVDEC output rather than a hand-built `AVFrame` — same helper
    /// shape as `cuda_download`'s own tests.
    fn decode_some_frames(device: &CudaDevice, path: &str) -> Vec<MediaBuffer> {
        let mut input = ffmpeg::format::input(path).expect("failed to open the test video");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("the test video has no video stream");
        let stream_index = stream.index();
        let params = stream.parameters();

        let mut decoder =
            CudaDecoder::new("cuda-decoder", params, device, 4).expect("failed to open NVDEC");
        let received = capture(&mut decoder);

        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("decode failed");
            if received.lock().unwrap().len() >= 3 {
                break;
            }
        }
        std::mem::take(&mut *received.lock().unwrap())
    }
}
