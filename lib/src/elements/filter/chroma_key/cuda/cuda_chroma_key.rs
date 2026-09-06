use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::super::handle::{ChromaKeyControl, ChromaKeyHandle};
use super::super::options::{ChromaKeyOptions, feather_band};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::{CudaDriverError, CudaUploadError},
    error::Result,
    pad::SrcPad,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        driver::{BgraSurface, CudaDriver},
        frame::create_hw_frames_ctx,
    },
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
};

/// Errors specific to `CudaChromaKey`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaChromaKeyError {
    /// FFmpeg could not take a second reference to the keyed frame already
    /// in hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous keyed frame (code {0})")]
    FrameRef(i32),

    /// The frame is not a CUDA surface at all.
    #[error("CudaChromaKey keys CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The sink received something other than a decoded video frame.
    #[error("CudaChromaKey only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// A CUDA frame arrived without the frames context that describes it.
    #[error("the frame carries no CUDA frames context")]
    MissingFramesContext,

    /// The surface belongs to another device, so this element's context
    /// cannot read it.
    #[error("the frame belongs to a different CUDA device than this CudaChromaKey")]
    ForeignContext,

    /// The surface is CUDA-resident but not in the layout this keys.
    #[error("CudaChromaKey keys BGRA surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    /// The frame is not the size this element allocated its pool for.
    #[error(
        "CudaChromaKey was built for {expected_width}x{expected_height}, \
         got {actual_width}x{actual_height}"
    )]
    DimensionMismatch {
        /// Width the frame actually carries.
        actual_width: u32,
        /// Height the frame actually carries.
        actual_height: u32,
        /// Width this element was constructed for.
        expected_width: u32,
        /// Height this element was constructed for.
        expected_height: u32,
    },

    /// A surface arrived with no device pointer to read.
    #[error("a surface arrived with no device pointer")]
    MissingSurface,

    /// Allocating this element's own frames context failed.
    #[error(transparent)]
    Frames(#[from] CudaUploadError),

    /// The CUDA driver rejected the keying kernel or its launch.
    #[error(transparent)]
    Driver(#[from] CudaDriverError),
}

/// Keys a solid background colour out of a CUDA-resident BGRA surface into
/// alpha, on the GPU.
///
/// The CUDA counterpart of `D3d11ChromaKey` (not linked: that type only
/// exists in a Windows build with `d3d11`, and this one compiles without
/// it), and the reason a
/// Linux graph can key at all: a CUDA compositor cannot accept a D3D11
/// texture, and the software element in between would cost a
/// [`CudaDownload`](crate::elements::CudaDownload) and a
/// [`CudaUpload`](crate::elements::CudaUpload) — two PCIe crossings per
/// frame — around it.
///
/// # BGRA in, BGRA out
///
/// Input must be a CUDA surface whose `sw_format` is BGRA, the same
/// constraint both other backends have and for the same reason: keying
/// compares a pixel's colour against the key, and a YUV round trip would
/// quantize the backdrop away from the colour being keyed. The typical
/// placement is right after a [`CudaUpload`](crate::elements::CudaUpload)
/// carrying BGRA, with the result going straight into a compositor layer.
///
/// Only alpha is written; the colour passes through untouched, as do PTS,
/// duration, and the colour-space tags — this makes no new timeline and no
/// new colour.
///
/// # Tuning it while it runs
///
/// [`CudaChromaKey::new`] returns a [`ChromaKeyHandle`] beside the element.
/// Keying is tuned by eye, and rebuilding the element for each nudge would
/// mean reopening whatever produces its frames.
///
/// # The kernel
///
/// Hand-written PTX, JIT-compiled by the driver when the module loads, so
/// nothing here needs a CUDA toolkit installed — only a driver. It computes
/// the same normalized BGR distance and the same feather ramp
/// [`SwChromaKey`](crate::elements::SwChromaKey) does, from the same
/// resolved band the D3D11 shader reads.
pub struct CudaChromaKey {
    pp_log: PpLog,
    name: Arc<str>,
    /// This element's own reference to the shared context, released in
    /// `Drop`.
    _hw_device_ctx: Arc<AvBufferRef>,
    /// The last keyed frame and the surface it was made from, so a producer
    /// that re-emits an unchanged picture is answered with it instead of
    /// another kernel over every pixel — see [`RepeatedOutput`].
    ///
    /// The input is not the whole of what the result depends on: the key
    /// settings can change under a [`ChromaKeyHandle`], and a surface keyed
    /// green is not an answer to the same surface once the key turned blue.
    /// `refresh_options` forgets what is held here when that happens.
    repeated: RepeatedOutput,
    /// The pool keyed frames are allocated from.
    hw_frames_ctx: AvBufferRef,
    /// The device context incoming frames must belong to, compared by
    /// pointer — a surface from another device would be read against the
    /// wrong context.
    device_ctx: *mut ffi::AVHWDeviceContext,
    driver: CudaDriver,
    width: u32,
    height: u32,
    /// What this element is keying by right now, refreshed from `control`
    /// once per frame. Kept alongside it so a change can be recognized —
    /// see `refresh_options`.
    options: ChromaKeyOptions,
    control: Arc<ChromaKeyControl>,
    pad: SrcPad,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the CUDA surface
    /// itself comes from `hw_frames_ctx`'s own pool. Same split as
    /// `CudaUpload`.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the buffers are heap-allocated FFmpeg buffers with no thread
// affinity of their own, `device_ctx` only ever has its address compared, and
// `&mut self` on every method that touches them rules out concurrent access.
// Same reasoning as `CudaConverter`.
unsafe impl Send for CudaChromaKey {}

impl CudaChromaKey {
    /// `device` must be the same [`CudaDevice`] every other CUDA element in
    /// this pipeline was built from. This element takes its own FFmpeg
    /// reference, so `device` itself need not outlive the call.
    ///
    /// Output dimensions are not a parameter: keying is a per-pixel
    /// transform, so every frame comes out the size it went in. Unlike
    /// [`CudaConverter`](crate::elements::CudaConverter), odd dimensions are
    /// fine — BGRA has no subsampled plane to halve.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        width: u32,
        height: u32,
        options: ChromaKeyOptions,
    ) -> std::result::Result<(Self, ChromaKeyHandle), CudaChromaKeyError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaChromaKey, &name, None);

        let driver = CudaDriver::retain_primary()?;
        let hw_device_ctx = device.retain();
        let hw_frames_ctx =
            // SAFETY: `create_hw_frames_ctx`'s contract is a live device context, which
            // is what the owned `AvBufferRef` beside it is.
            unsafe { create_hw_frames_ctx(&hw_device_ctx, CudaFrameFormat::Bgra, width, height) }
                .map_err(CudaUploadError::from)?;
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition. Only the pointer's identity is kept, to compare against an
        // incoming frame's; the reference held alongside it is what keeps that
        // identity from being reused by a different context.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *mut ffi::AVHWDeviceContext };

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::Cuda,
            )),
        );
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let key_color = options.method.key_color();
        pp_info!(
            pp_log: &pp_log,
            "opened: {width}x{height} BGRA, key_color={key_color:?}, threshold={}, smoothing={}",
            options.threshold,
            options.smoothing
        );

        let control = Arc::new(ChromaKeyControl::new(options));
        let handle = ChromaKeyHandle::new(control.clone());
        let element = Self {
            name,
            pp_log,
            _hw_device_ctx: hw_device_ctx,
            hw_frames_ctx,
            device_ctx,
            driver,
            width,
            height,
            options,
            control,
            pad,
            repeated: RepeatedOutput::new(),
            pool,
        };
        Ok((element, handle))
    }

    /// Picks up whatever the handle has been set to, and forgets the cached
    /// keyed surface if it was made with something else.
    ///
    /// Called once per frame, before keying. Reading the settings again
    /// mid-frame could pick up a change made in between, which is the one
    /// thing `ChromaKeyControl`'s single lock exists to prevent.
    fn refresh_options(&mut self) {
        let options = self.control.get();
        if options != self.options {
            pp_debug!(
                self,
                "retuned: key_color={:?}, threshold={}, smoothing={}",
                options.method.key_color(),
                options.threshold,
                options.smoothing
            );
            self.options = options;
            self.repeated.clear();
        }
    }

    /// Rejects anything this cannot key before a device pointer is read out
    /// of it: the format, the frames context and its device, the surface
    /// layout, and the size this element was built for.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<(), CudaChromaKeyError> {
        if frame.format() != ffmpeg::format::Pixel::CUDA {
            return Err(CudaChromaKeyError::UnsupportedFormat(frame.format()));
        }
        if frame.width() != self.width || frame.height() != self.height {
            return Err(CudaChromaKeyError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        // SAFETY: `frame` is a live `frame::Video` already confirmed to be
        // `Pixel::CUDA`, so `as_ptr` yields an initialized `AVFrame` and a hardware
        // frame's `hw_frames_ctx` is either null — rejected here — or an
        // `AVBufferRef` whose `data` is an `AVHWFramesContext`. Only pointer
        // identity is compared, never dereferenced past that.
        unsafe {
            let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() {
                return Err(CudaChromaKeyError::MissingFramesContext);
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            if !std::ptr::eq((*frames_ctx).device_ctx, self.device_ctx) {
                return Err(CudaChromaKeyError::ForeignContext);
            }
            let sw_format = (*frames_ctx).sw_format;
            if CudaFrameFormat::from_sw_format(sw_format) != Some(CudaFrameFormat::Bgra) {
                return Err(CudaChromaKeyError::UnsupportedSurfaceFormat(
                    ffmpeg::format::Pixel::from(sw_format),
                ));
            }
        }
        Ok(())
    }

    fn key(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.validate(source)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        let mut destination = self.pool.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, and the unref before
        // the allocation is what hands its previous surface back — see the comment
        // beside it. The frames context is this element's own, held for its life.
        unsafe {
            let dst = destination.as_mut_ptr();
            // The pooled wrapper may still reference the previous frame's
            // surface; releasing it here is what returns that surface to the
            // frames pool rather than leaking it for the element's lifetime.
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), dst, 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_get_buffer failed: {code}");
                return Err(CudaChromaKeyError::Frames(CudaUploadError::HwFrameGet(code)).into());
            }
        }

        (|| -> std::result::Result<(), CudaChromaKeyError> {
            let source_surface =
                BgraSurface::from_frame(source).ok_or(CudaChromaKeyError::MissingSurface)?;
            let destination_surface =
                BgraSurface::from_frame(&destination).ok_or(CudaChromaKeyError::MissingSurface)?;
            let (band_low, inv_band_width) =
                feather_band(self.options.threshold, self.options.smoothing);
            self.driver.key_bgra(
                source_surface,
                destination_surface,
                self.width,
                self.height,
                self.options.method.key_color(),
                band_low,
                inv_band_width,
            )?;
            // The kernel is issued on this driver's own context, while
            // whatever reads the result next — a compositor, a download —
            // issues its work through FFmpeg's stream. One synchronize per
            // frame is what makes the keyed surface visible to both, the
            // same point `CudaConverter` synchronizes at.
            self.driver.synchronize()?;
            Ok(())
        })()
        .inspect_err(|error| pp_error!(self, "{error}"))?;

        // SAFETY: both frames are live and distinct — `destination` came from the
        // pool, `source` is the caller's — so `av_frame_copy_props` reads one and
        // writes the other with no aliasing.
        unsafe {
            // PTS, duration and the colour tags are all part of the buffer
            // contract. Unlike `CudaConverter`, this keeps the colour ones:
            // keying writes alpha and leaves BGR alone, so the result is the
            // same colour the input was.
            ffi::av_frame_copy_props(destination.as_mut_ptr(), source.as_ptr());
        }
        Ok(destination)
    }
}

impl PerFrameTransform for CudaChromaKey {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        CudaChromaKeyError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.key(source)
    }
}

impl Element for CudaChromaKey {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaChromaKey
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaChromaKey {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaChromaKey {
    /// Writes alpha into a device-resident frame; the surface layout it
    /// requires is a runtime value, not part of this.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::Cuda,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            // The same surface as last time is the same pixels as last time,
            // and keying them again produces the frame already in hand —
            // see [`PerFrameTransform`].
            MediaBuffer::Video(frame) => {
                self.refresh_options();
                let keyed = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(keyed))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => Err(CudaChromaKeyError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to beyond the cached keyed frame — a pure
        // per-frame transform, same reasoning as `CudaConverter::control`.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

impl Drop for CudaChromaKey {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::super::super::options::ChromaKeyMethod;
    use super::*;
    use crate::{
        elements::{CudaDownload, CudaUpload},
        test_support::try_cuda_device,
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

    fn default_options() -> ChromaKeyOptions {
        ChromaKeyOptions {
            method: ChromaKeyMethod::Green,
            threshold: 0.15,
            smoothing: 0.1,
        }
    }

    /// Uploads one BGRA frame built by `pixel`, so a test starts from a
    /// CUDA-resident source with content it can predict.
    fn cuda_bgra(
        device: &CudaDevice,
        width: u32,
        height: u32,
        pts: i64,
        pixel: impl Fn(u32, u32) -> [u8; 4],
    ) -> Option<MediaBuffer> {
        let Ok(mut upload) =
            CudaUpload::new("upload", device, CudaFrameFormat::Bgra, width, height)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
        frame.set_pts(Some(pts));
        let stride = frame.stride(0);
        for y in 0..height {
            let row = &mut frame.data_mut(0)[y as usize * stride..];
            for x in 0..width {
                row[x as usize * 4..x as usize * 4 + 4].copy_from_slice(&pixel(x, y));
            }
        }
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    fn download_bgra(
        device: &CudaDevice,
        buffer: MediaBuffer,
        width: u32,
        height: u32,
    ) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut download =
            CudaDownload::new("download", device, CudaFrameFormat::Bgra, width, height);
        let received = capture(&mut download);
        download.consume(buffer).expect("download");
        match received.lock().unwrap().remove(0) {
            MediaBuffer::Video(frame) => frame,
            other => panic!("expected a Video buffer, got {}", other.kind()),
        }
    }

    fn key_element(
        device: &CudaDevice,
        width: u32,
        height: u32,
    ) -> Option<(CudaChromaKey, ChromaKeyHandle)> {
        match CudaChromaKey::new("key", device, width, height, default_options()) {
            Ok(pair) => Some(pair),
            Err(error) => {
                eprintln!("skipping: no usable CUDA keying here ({error})");
                None
            }
        }
    }

    /// Another `AVFrame` over the same surface, with its own timestamp —
    /// what a screen capture with nothing new to show hands over each tick.
    fn repeat_of(buffer: &MediaBuffer, pts: i64) -> MediaBuffer {
        let MediaBuffer::Video(frame) = buffer else {
            panic!("expected a Video buffer");
        };
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        // SAFETY: both are live `AVFrame`s and distinct — the slot is the
        // empty one just taken from the pool — so this references the
        // surface rather than copying it.
        unsafe {
            assert!(ffi::av_frame_ref(slot.as_mut_ptr(), frame.as_ptr()) >= 0);
        }
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    /// The whole contract in one pass: the key colour goes transparent, a
    /// clearly different colour keeps both its alpha and its RGB, and the
    /// timestamp survives.
    #[test]
    fn the_key_colour_goes_transparent_and_the_rest_comes_through() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(source) = cuda_bgra(&device, 2, 1, 42, |x, _| {
            if x == 0 {
                [0, 255, 0, 255]
            } else {
                [0, 0, 255, 255]
            }
        }) else {
            return;
        };
        let Some((mut key, _handle)) = key_element(&device, 2, 1) else {
            return;
        };
        let keyed = capture(&mut key);

        key.consume(source).expect("key the frame");

        let out = keyed.lock().unwrap().remove(0);
        let MediaBuffer::Video(frame) = &out else {
            panic!("expected a Video buffer");
        };
        assert_eq!(
            frame.pts(),
            Some(42),
            "the timestamp is part of the contract"
        );

        let downloaded = download_bgra(&device, out.clone(), 2, 1);
        let row = downloaded.data(0);
        assert_eq!(row[3], 0, "the key colour must key out completely");
        assert_eq!(
            &row[4..8],
            &[0, 0, 255, 255],
            "a clearly different colour keeps its alpha and its RGB"
        );
    }

    /// A producer repeating an unchanged image is answered out of the last
    /// keyed surface rather than keyed again — the kernel is the expensive
    /// part, and it would write the pixels already there.
    #[test]
    fn an_unchanged_surface_is_keyed_once() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(source) = cuda_bgra(&device, 4, 2, 100, |_, _| [0, 255, 0, 255]) else {
            return;
        };
        let repeat = repeat_of(&source, 200);
        let Some((mut key, _handle)) = key_element(&device, 4, 2) else {
            return;
        };
        let keyed = capture(&mut key);

        key.consume(source).expect("key the first frame");
        key.consume(repeat).expect("key the repeat");

        let received = keyed.lock().unwrap();
        assert_eq!(received.len(), 2, "every tick still produces a frame");
        let (MediaBuffer::Video(first), MediaBuffer::Video(second)) = (&received[0], &received[1])
        else {
            panic!("expected Video buffers");
        };
        assert_eq!(
            crate::buffer::picture_id(second),
            crate::buffer::picture_id(first),
            "an unchanged surface was keyed a second time"
        );
        assert_eq!(
            second.pts(),
            Some(200),
            "a repeat carries its own timestamp, not the one it points at"
        );
    }

    /// The same hazard the other two backends have: a cache filled under the
    /// old key is not an answer to a repeat that arrives after the key
    /// moved, and a still capture would otherwise stay frozen at the old
    /// settings with nothing to ever dislodge it.
    #[test]
    fn a_repeat_is_keyed_again_after_a_retune() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some(source) = cuda_bgra(&device, 4, 2, 100, |_, _| [0, 255, 0, 255]) else {
            return;
        };
        let repeat = repeat_of(&source, 200);
        let Some((mut key, handle)) = key_element(&device, 4, 2) else {
            return;
        };
        let keyed = capture(&mut key);

        key.consume(source).expect("key the first frame");

        let mut retuned = handle.options();
        retuned.method = ChromaKeyMethod::Blue;
        handle.set_options(retuned);

        key.consume(repeat).expect("key the repeat");

        let (first, second) = {
            let received = keyed.lock().unwrap();
            let (MediaBuffer::Video(first), MediaBuffer::Video(second)) =
                (&received[0], &received[1])
            else {
                panic!("expected Video buffers");
            };
            assert_ne!(
                crate::buffer::picture_id(second),
                crate::buffer::picture_id(first),
                "a retune must retire the surface the cache was holding"
            );
            (received[0].clone(), received[1].clone())
        };

        let alpha_of = |buffer: MediaBuffer| download_bgra(&device, buffer, 4, 2).data(0)[3];
        assert_eq!(alpha_of(first), 0, "green keys out against green");
        assert_eq!(
            alpha_of(second),
            255,
            "and stays once the key is blue instead"
        );
    }

    #[test]
    fn a_cpu_frame_is_a_typed_error_not_a_panic() {
        let Some((device, _cuda_lock)) = try_cuda_device() else {
            return;
        };
        let Some((mut key, _handle)) = key_element(&device, 4, 2) else {
            return;
        };
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 4, 2),
            |_| {},
        );
        let frame = MediaBuffer::Video(Arc::new(pool.get()));

        let error = key
            .consume(frame)
            .expect_err("a CPU frame must be rejected");
        assert!(matches!(
            error,
            crate::error::Error::CudaChromaKeyError(CudaChromaKeyError::UnsupportedFormat(
                ffmpeg::format::Pixel::BGRA
            ))
        ));
    }
}
