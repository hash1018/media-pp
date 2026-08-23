use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::scale_graph::CudaScaleGraph;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::{
        cuda::{CudaDevice, CudaFrameFormat},
        ffmpeg::AvBufferRef,
    },
};

/// Errors specific to `CudaScaler`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaScalerError {
    /// FFmpeg rejected filter-graph or frame processing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The linked FFmpeg build does not provide a required CUDA filter.
    #[error(
        "filter `{0}` not found in this ffmpeg build — a CUDA-enabled build \
         (`--enable-ffnvcodec` plus `--enable-cuda-nvcc` or `--enable-cuda-llvm`) is required"
    )]
    FilterNotFound(&'static str),

    /// The input frame is not backed by CUDA hardware surfaces.
    #[error("CudaScaler only scales CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The sink received a buffer other than decoded video or end-of-stream.
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

    /// The CUDA surface layout cannot be processed by `scale_cuda`.
    #[error("CudaScaler cannot scale {0:?} surfaces")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    /// FFmpeg could not allocate the filter graph's buffer source.
    #[error("failed to allocate the buffer source filter")]
    BufferSrcAlloc,

    /// FFmpeg could not allocate buffer-source parameters.
    #[error("failed to allocate the buffer source parameters")]
    BufferSrcParamsAlloc,

    /// FFmpeg rejected the buffer-source parameters.
    #[error("failed to configure the buffer source (code {0})")]
    BufferSrcConfig(i32),

    /// FFmpeg could not initialize the configured buffer source.
    #[error("failed to initialize the buffer source (code {0})")]
    BufferSrcInit(i32),

    /// FFmpeg rejected an input frame pushed into the graph.
    #[error("the filter graph rejected a frame (code {0})")]
    BufferSrcPush(i32),

    /// FFmpeg failed to return a scaled frame from the graph.
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
    /// Select the nearest input texel without interpolation.
    Nearest,
    /// Interpolate from the nearest four texels.
    Bilinear,
    /// Use cubic interpolation for smoother scaling.
    Bicubic,
    /// Use Lanczos interpolation for sharper high-quality scaling.
    Lanczos,
}

impl CudaScalerInterp {
    /// `scale_cuda`'s own `interp_algo` option values.
    pub(crate) fn algo(self) -> u32 {
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
/// # Resize only
///
/// The output surface holds whatever the input did — every
/// [`CudaFrameFormat`] goes through, and none of them turns into another.
/// `scale_cuda` cannot convert between RGB and YUV in either direction: its
/// PTX module carries no kernel for those pairs, so such a conversion fails
/// as `Unsupported conversion: bgr0 -> nv12` rather than falling back to
/// anything.
///
/// That is a narrower limit than it sounds, because nothing downstream
/// needs the conversion. NVENC ingests `Bgra` as directly as `Nv12`,
/// converting in hardware, so a capture recorded through
/// `CudaUpload -> CudaScaler -> CudaEncoder` stays BGRA end to end. The one
/// element that does require `Nv12` is [`crate::elements::CudaRenderer`],
/// whose trait hands the graphics side separate luma/chroma pointers.
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
    _hw_device_ctx: Arc<AvBufferRef>,
    /// Captured at construction so an incoming frame can be checked against
    /// this element's own CUDA context. Only ever compared, same as
    /// [`crate::elements::CudaEncoder`]'s.
    device_ctx: *const ffi::AVHWDeviceContext,
    width: u32,
    height: u32,
    /// Built from the first frame and rebuilt when the input changes — see
    /// this type's own docs.
    graph: CudaScaleGraph,
    pad: SrcPad,
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

        let hw_device_ctx = device.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition. Only the pointer's identity is kept, to compare against an
        // incoming frame's; the reference held alongside it is what keeps that
        // identity from being reused by a different context.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *const ffi::AVHWDeviceContext };

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::Cuda,
            )),
        );
        pp_info!(pp_log: &pp_log, "created: dst={width}x{height}, interp={interp:?}");
        Self {
            name,
            pp_log,
            _hw_device_ctx: hw_device_ctx,
            device_ctx,
            width,
            height,
            graph: CudaScaleGraph::new(interp),
            pad,
        }
    }

    /// Rejects anything that is not a CUDA NV12 frame from this element's own
    /// device, so a bad frame fails here rather than as an opaque libavfilter
    /// error code — or, worse, as device pointers read against the wrong
    /// context. Returns the frame's own frames context, which is what the
    /// graph has to be configured against.
    pub(crate) fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<*mut ffi::AVBufferRef, CudaScalerError> {
        if frame.format() != ffmpeg::format::Pixel::CUDA {
            return Err(CudaScalerError::UnsupportedFormat(frame.format()));
        }
        // SAFETY: `frame` is a live `frame::Video` already confirmed to be
        // `Pixel::CUDA`, so `as_ptr` yields an initialized `AVFrame` and a hardware
        // frame's `hw_frames_ctx` is either null — rejected here — or an
        // `AVBufferRef` whose `data` is an `AVHWFramesContext`. Only pointer
        // identity is compared, never dereferenced past that.
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
            if CudaFrameFormat::from_sw_format(sw_format).is_none() {
                return Err(CudaScalerError::UnsupportedSurfaceFormat(
                    ffmpeg::format::Pixel::from(sw_format),
                ));
            }
            Ok(frames_ref)
        }
    }

    fn scale(&mut self, frame: &ffmpeg::frame::Video) -> Result<()> {
        let frames_ctx = self
            .validate(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        if !self
            .graph
            .matches(frame, frames_ctx, self.width, self.height)
        {
            // A live source can renegotiate mid-stream, and a decoder rebuilt
            // after a seek hands out a new pool — same reason `SwScaler`
            // rebuilds its own context, and worth seeing in a log when output
            // suddenly looks stretched.
            pp_debug!(
                self,
                "input is {}x{}, building the filter graph",
                frame.width(),
                frame.height()
            );
        }
        let scaled = self
            .graph
            .scale(frame, frames_ctx, self.width, self.height)
            .inspect_err(|error| pp_error!(self, "scale failed: {error}"))?;
        for output in scaled {
            self.pad.push(MediaBuffer::Video(Arc::new(output)))?;
        }
        Ok(())
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
    /// Resizes on the device; a system-memory frame belongs in SwScaler.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::Cuda,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.scale(&frame),
            MediaBuffer::Eos => {
                let drained = self
                    .graph
                    .flush()
                    .inspect_err(|error| pp_error!(self, "failed to flush the graph: {error}"))?;
                for output in drained {
                    self.pad.push(MediaBuffer::Video(Arc::new(output)))?;
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
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{
        elements::{CudaDecoder, CudaDownload, CudaUpload},
        pool::UnboundObjectPool,
        test_support::{try_cuda_device, try_test_video},
    };

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
        let Ok(mut upload) =
            CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
        else {
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

    /// libavfilter keeps the colorimetry its link was configured with and
    /// reports every frame that disagrees as "video frame properties changing
    /// on the fly" — once per frame, for the whole run. The link is built
    /// from the first frame's own tags, so a change in them is a rebuild like
    /// a change in size.
    #[test]
    fn the_filter_link_follows_the_frames_colorimetry() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(tagged) = cuda_frame(&device, 128, 128, 200, 0) else {
            return;
        };
        let MediaBuffer::Video(frame) = &tagged else {
            panic!("the upload produces a Video buffer");
        };
        // SAFETY: `frame` is a live CUDA frame from the upload above, so
        // `hw_frames_ctx` is its own pool. Only the pointer is taken, to hand the
        // graph the same pool the frames come from.
        let frames_ctx = unsafe { (*frame.as_ptr()).hw_frames_ctx };
        // What a converted capture carries.
        let mut bt709 = ffmpeg::frame::Video::empty();
        // SAFETY: `frame` is a live frame from the upload above, and the target is
        // the empty local on the line before, so the two cannot alias.
        unsafe { ffi::av_frame_ref(bt709.as_mut_ptr(), frame.as_ptr()) };
        bt709.set_color_space(ffmpeg::color::Space::BT709);
        bt709.set_color_range(ffmpeg::color::Range::MPEG);

        let mut graph = CudaScaleGraph::new(CudaScalerInterp::Bilinear);
        graph
            .scale(&bt709, frames_ctx, 64, 64)
            .expect("the first frame builds the graph");
        assert!(
            graph.matches(&bt709, frames_ctx, 64, 64),
            "the same frame must not rebuild the graph"
        );

        // Same pool, same size, different tags: the link would keep reporting
        // a mismatch for every frame, so it has to be rebuilt.
        let mut untagged = ffmpeg::frame::Video::empty();
        // SAFETY: `frame` is a live frame from the upload above, and the target is
        // the empty local on the line before, so the two cannot alias.
        unsafe { ffi::av_frame_ref(untagged.as_mut_ptr(), frame.as_ptr()) };
        untagged.set_color_space(ffmpeg::color::Space::Unspecified);
        untagged.set_color_range(ffmpeg::color::Range::Unspecified);
        assert!(
            !graph.matches(&untagged, frames_ctx, 64, 64),
            "a frame tagged differently needs its own link"
        );
    }

    /// The contract: the frame comes out at the configured size, still on the
    /// GPU, still carrying its timestamp — and the pixels survive the resize.
    /// The `CudaDownload` on the tail is what makes the last part checkable at
    /// all, since a CUDA surface has no readable planes.
    #[test]
    fn a_cuda_frame_is_resized_on_the_gpu_and_keeps_its_pts() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(frame) = cuda_frame(&device, 128, 128, 200, 777) else {
            return;
        };

        let mut scaler = CudaScaler::new("scaler", &device, 64, 64, CudaScalerInterp::Bilinear);
        let mut download = CudaDownload::new("download", &device, CudaFrameFormat::Nv12, 64, 64);
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
        let Some((device, _cuda_lock)) = try_cuda_device() else {
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

    /// The capture path's format. `scale_cuda` resizes an RGB surface fine as
    /// long as it stays RGB, and the output must not have quietly become
    /// something else — nothing downstream could convert it back.
    #[test]
    fn a_bgra_surface_is_resized_and_stays_bgra() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Ok(mut upload) = CudaUpload::new("upload", &device, CudaFrameFormat::Bgra, 128, 64)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return;
        };
        let uploaded = capture(&mut upload);
        let mut bgra = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 128, 64);
        bgra.set_pts(Some(3));
        upload.consume(pooled(bgra)).expect("upload");
        let frame = uploaded.lock().unwrap().remove(0);

        let mut scaler = CudaScaler::new("scaler", &device, 64, 32, CudaScalerInterp::Bilinear);
        let received = capture(&mut scaler);
        scaler.consume(frame).expect("bgra scale");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(scaled) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(scaled.width(), 64);
        assert_eq!(scaled.height(), 32);
        assert_eq!(scaled.pts(), Some(3));
        // SAFETY: the assertions above have established this is a live CUDA frame,
        // so its `hw_frames_ctx` is set and its `data` is an `AVHWFramesContext`.
        let sw_format = unsafe {
            let frames_ref = (*scaled.as_ptr()).hw_frames_ctx;
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            ffmpeg::format::Pixel::from((*frames_ctx).sw_format)
        };
        assert_eq!(
            sw_format,
            ffmpeg::format::Pixel::BGRA,
            "the scaler changed the surface layout"
        );
    }

    /// A mid-stream input change must rebuild the graph rather than fail or
    /// keep scaling from a stale one — same contract `SwScaler` has for its
    /// `libswscale` context.
    #[test]
    fn rebuilds_its_graph_when_the_input_changes_mid_stream() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
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
        let Some((device, _cuda_lock)) = try_cuda_device() else {
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

        // Directly, not `try_cuda_device` again: the lock it returns is
        // already held for this test and does not nest.
        let other_device = CudaDevice::new().expect("a second CUDA device");
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
