use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use super::super::handle::{VideoEffectControl, VideoEffectHandle};
use super::super::options::{EffectParams, VideoEffect};
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

/// Errors specific to `CudaVideoEffect`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum CudaVideoEffectError {
    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The frame is not a CUDA surface at all.
    #[error("CudaVideoEffect takes CUDA frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The sink received something other than a decoded video frame.
    #[error("CudaVideoEffect only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// A CUDA frame arrived without the frames context that describes it.
    #[error("the frame carries no CUDA frames context")]
    MissingFramesContext,

    /// The surface belongs to another device.
    #[error("the frame belongs to a different CUDA device than this CudaVideoEffect")]
    ForeignContext,

    /// The surface is CUDA-resident but not BGRA.
    #[error("CudaVideoEffect takes BGRA surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    /// The frame is not the size this element allocated its pool for.
    #[error(
        "CudaVideoEffect was built for {expected_width}x{expected_height}, \
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

    /// The CUDA driver rejected the kernel or its launch.
    #[error(transparent)]
    Driver(#[from] CudaDriverError),
}

/// Applies a [`VideoEffect`] — a colour correction or a luma key — to a
/// CUDA-resident BGRA surface, on the GPU.
///
/// The CUDA member of the family whose software member is
/// [`SwVideoEffect`](crate::elements::SwVideoEffect); all of them evaluate
/// one resolved set of numbers per pixel. The rest is `CudaChromaKey`'s,
/// for its reasons: BGRA in and out, a surface of the size this was built
/// for, from the same [`CudaDevice`] every other CUDA element in the
/// pipeline uses, and PTS, duration and colour tags carried through.
///
/// An effect that changes nothing, and an element that is turned off, hand
/// each frame straight through — the same surface, not a copy of it.
///
/// # The kernel
///
/// Hand-written PTX, JIT-compiled by the driver when it loads, so nothing
/// here needs a CUDA toolkit — only a driver. The exponent is `2^(e·log2 x)`
/// through the hardware's approximate `lg2`/`ex2`, which is why this agrees
/// with the software element to within one step rather than exactly
/// wherever a gamma is set.
pub struct CudaVideoEffect {
    pp_log: PpLog,
    name: Arc<str>,
    /// This element's own reference to the shared context, released in
    /// `Drop`.
    _hw_device_ctx: Arc<AvBufferRef>,
    /// The last output and the surface it was made from — see
    /// [`RepeatedOutput`]. Cleared when the effect changes.
    repeated: RepeatedOutput,
    /// The pool outputs are allocated from.
    hw_frames_ctx: AvBufferRef,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *mut ffi::AVHWDeviceContext,
    driver: CudaDriver,
    width: u32,
    height: u32,
    /// The effect in force and what it resolves to, refreshed from `control`
    /// once per frame.
    effect: VideoEffect,
    params: EffectParams,
    control: Arc<VideoEffectControl>,
    enabled: bool,

    pad: SrcPad,
    /// Reuses only the CPU-side `AVFrame` wrapper; each surface comes from
    /// `hw_frames_ctx`'s own pool.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the buffers are heap-allocated FFmpeg buffers with no thread
// affinity, `device_ctx` only ever has its address compared, and `&mut self`
// on every method that touches them rules out concurrent access — the
// reasoning `CudaChromaKey` gives.
unsafe impl Send for CudaVideoEffect {}

impl CudaVideoEffect {
    /// `device` must be the same [`CudaDevice`] every other CUDA element in
    /// this pipeline was built from; this takes its own reference, so
    /// `device` need not outlive the call.
    ///
    /// Output dimensions are the input's: an effect is per pixel. Odd sizes
    /// are fine — BGRA has no subsampled plane.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        width: u32,
        height: u32,
        effect: VideoEffect,
    ) -> std::result::Result<(Self, VideoEffectHandle), CudaVideoEffectError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaVideoEffect, &name, None);

        let driver = CudaDriver::retain_primary()?;
        let hw_device_ctx = device.retain();
        let hw_frames_ctx =
            // SAFETY: `create_hw_frames_ctx`'s contract is a live device context,
            // which is what the owned `AvBufferRef` beside it is.
            unsafe { create_hw_frames_ctx(&hw_device_ctx, CudaFrameFormat::Bgra, width, height) }
                .map_err(CudaUploadError::from)?;
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext`. Only the pointer's
        // identity is kept; the reference beside it keeps the identity from being
        // reused by another context.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *mut ffi::AVHWDeviceContext };

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::Cuda,
            )),
        );
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(
            pp_log: &pp_log,
            "opened: {width}x{height} BGRA, {} {effect:?}",
            effect.name()
        );

        let control = Arc::new(VideoEffectControl::new(effect));
        let handle = VideoEffectHandle::new(control.clone());
        let element = Self {
            name,
            pp_log,
            _hw_device_ctx: hw_device_ctx,
            hw_frames_ctx,
            device_ctx,
            driver,
            width,
            height,
            effect,
            params: effect.params(),
            control,
            enabled: true,
            pad,
            repeated: RepeatedOutput::new(),
            pool,
        };
        Ok((element, handle))
    }

    /// Picks up whatever the handle has been set to, once per frame.
    fn refresh(&mut self) {
        let effect = self.control.get();
        if effect != self.effect {
            pp_debug!(self, "retuned: {} {effect:?}", effect.name());
            self.effect = effect;
            self.params = effect.params();
            self.repeated.clear();
        }
        let enabled = self.control.enabled();
        if enabled != self.enabled {
            pp_debug!(self, "effect {}", if enabled { "on" } else { "off" });
            self.enabled = enabled;
            self.repeated.clear();
        }
    }

    /// Rejects anything this cannot read before a device pointer is taken
    /// out of it.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<(), CudaVideoEffectError> {
        if frame.format() != ffmpeg::format::Pixel::CUDA {
            return Err(CudaVideoEffectError::UnsupportedFormat(frame.format()));
        }
        if frame.width() != self.width || frame.height() != self.height {
            return Err(CudaVideoEffectError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            });
        }
        // SAFETY: `frame` is a live `Pixel::CUDA` frame, so its `hw_frames_ctx` is
        // either null — refused here — or an `AVBufferRef` whose `data` is an
        // `AVHWFramesContext`. Only pointer identity is compared.
        unsafe {
            let frames_ref = (*frame.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() {
                return Err(CudaVideoEffectError::MissingFramesContext);
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            if !std::ptr::eq((*frames_ctx).device_ctx, self.device_ctx) {
                return Err(CudaVideoEffectError::ForeignContext);
            }
            let sw_format = (*frames_ctx).sw_format;
            if CudaFrameFormat::from_sw_format(sw_format) != Some(CudaFrameFormat::Bgra) {
                return Err(CudaVideoEffectError::UnsupportedSurfaceFormat(
                    ffmpeg::format::Pixel::from(sw_format),
                ));
            }
        }
        Ok(())
    }

    fn run(
        &mut self,
        source: &ffmpeg::frame::Video,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.validate(source)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        let mut destination = self.pool.get();
        // SAFETY: `dst` is the pooled wrapper's own `AVFrame`; unreferencing it
        // first hands its previous surface back to the frames pool. The frames
        // context is this element's own, held for its life.
        unsafe {
            let dst = destination.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), dst, 0);
            if code < 0 {
                pp_error!(self, "av_hwframe_get_buffer failed: {code}");
                return Err(CudaVideoEffectError::Frames(CudaUploadError::HwFrameGet(code)).into());
            }
        }

        (|| -> std::result::Result<(), CudaVideoEffectError> {
            let source_surface =
                BgraSurface::from_frame(source).ok_or(CudaVideoEffectError::MissingSurface)?;
            let destination_surface = BgraSurface::from_frame(&destination)
                .ok_or(CudaVideoEffectError::MissingSurface)?;
            let params = &self.params;
            self.driver.effect_bgra(
                source_surface,
                destination_surface,
                self.width,
                self.height,
                &params.rows,
                params.exponent,
                params.opacity,
                [
                    params.luma_low,
                    params.luma_low_inv,
                    params.luma_high,
                    params.luma_high_inv,
                ],
            )?;
            // The kernel runs on this driver's context while whatever reads
            // the result next goes through FFmpeg's stream: one synchronize
            // per frame makes it visible to both, as `CudaChromaKey` does.
            self.driver.synchronize()?;
            Ok(())
        })()
        .inspect_err(|error| pp_error!(self, "{error}"))?;

        // SAFETY: both frames are live and distinct — `destination` came from
        // the pool, `source` is the caller's.
        unsafe {
            ffi::av_frame_copy_props(destination.as_mut_ptr(), source.as_ptr());
        }
        Ok(destination)
    }
}

impl PerFrameTransform for CudaVideoEffect {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        CudaVideoEffectError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.run(source)
    }
}

impl Element for CudaVideoEffect {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaVideoEffect
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for CudaVideoEffect {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for CudaVideoEffect {
    /// Device-resident frames; the surface layout it needs is a runtime
    /// value, not part of this.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::Cuda,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                self.refresh();
                if !self.enabled || self.params.is_identity() {
                    // Straight through: the same picture, not a copy of it.
                    return self.pad.push(MediaBuffer::Video(frame));
                }
                let output = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(output))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(CudaVideoEffectError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

impl Drop for CudaVideoEffect {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::super::super::options::apply;
    use super::*;
    use crate::{
        elements::{ColorCorrection, CudaDownload, CudaUpload, LumaKey},
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

    /// Uploads one BGRA frame whose pixel at `x` is `pixels[x]`, a row of
    /// them, so one kernel launch covers every sample a test asks about.
    fn cuda_row(device: &CudaDevice, pixels: &[[u8; 4]]) -> Option<MediaBuffer> {
        let width = pixels.len() as u32;
        let Ok(mut upload) = CudaUpload::new("upload", device, CudaFrameFormat::Bgra, width, 1)
        else {
            eprintln!("skipping: this machine has no usable CUDA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, 1);
        frame.set_pts(Some(7));
        for (x, pixel) in pixels.iter().enumerate() {
            frame.data_mut(0)[x * 4..x * 4 + 4].copy_from_slice(pixel);
        }
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        Some(uploaded.lock().unwrap().remove(0))
    }

    fn download_row(device: &CudaDevice, buffer: MediaBuffer, width: u32) -> Vec<[u8; 4]> {
        let mut download = CudaDownload::new("download", device, CudaFrameFormat::Bgra, width, 1);
        let received = capture(&mut download);
        download.consume(buffer).expect("download");
        let MediaBuffer::Video(frame) = received.lock().unwrap().remove(0) else {
            panic!("expected a Video buffer");
        };
        (0..width as usize)
            .map(|x| frame.data(0)[x * 4..x * 4 + 4].try_into().unwrap())
            .collect()
    }

    const PIXELS: [[u8; 4]; 6] = [
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [30, 140, 220, 255],
        [200, 60, 10, 160],
        [100, 110, 120, 255],
        [1, 254, 77, 200],
    ];

    fn effects() -> Vec<VideoEffect> {
        vec![
            VideoEffect::ColorCorrection(ColorCorrection {
                brightness: 0.1,
                contrast: 1.4,
                saturation: 0.6,
                hue_degrees: 35.0,
                gamma: 1.6,
                opacity: 0.8,
            }),
            VideoEffect::ColorCorrection(ColorCorrection {
                saturation: 0.0,
                ..ColorCorrection::default()
            }),
            VideoEffect::LumaKey(LumaKey {
                min: 0.25,
                min_smoothing: 0.1,
                max: 0.75,
                max_smoothing: 0.1,
            }),
        ]
    }

    /// The kernel is the shared definition: every effect, on a spread of
    /// pixels, comes out within one step of the CPU's answer — the room the
    /// hardware's approximate `lg2`/`ex2` needs where a gamma is set.
    #[test]
    fn every_effect_computes_what_the_shared_definition_says() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let width = PIXELS.len() as u32;
        for effect in effects() {
            let Ok((mut element, _)) = CudaVideoEffect::new("effect", &device, width, 1, effect)
            else {
                eprintln!("skipping: no usable CUDA video effect here");
                return;
            };
            let received = capture(&mut element);
            let Some(source) = cuda_row(&device, &PIXELS) else {
                return;
            };
            element.consume(source).expect("run");
            let output = received.lock().unwrap().remove(0);
            let drawn = download_row(&device, output, width);

            for (pixel, drawn) in PIXELS.iter().zip(drawn) {
                let expected = apply(&effect.params(), *pixel);
                for channel in 0..4 {
                    assert!(
                        drawn[channel].abs_diff(expected[channel]) <= 1,
                        "{effect:?} on {pixel:?}: got {drawn:?}, expected {expected:?}"
                    );
                }
            }
        }
    }

    /// Without a gamma nothing is approximate, so every byte is exact.
    #[test]
    fn a_linear_effect_is_byte_exact() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let width = PIXELS.len() as u32;
        let effect = VideoEffect::ColorCorrection(ColorCorrection {
            brightness: 0.1,
            opacity: 0.5,
            ..ColorCorrection::default()
        });
        let Ok((mut element, _)) = CudaVideoEffect::new("effect", &device, width, 1, effect) else {
            return;
        };
        let received = capture(&mut element);
        let Some(source) = cuda_row(&device, &PIXELS) else {
            return;
        };
        element.consume(source).expect("run");
        let output = received.lock().unwrap().remove(0);
        let drawn = download_row(&device, output, width);
        let expected: Vec<[u8; 4]> = PIXELS
            .iter()
            .map(|pixel| apply(&effect.params(), *pixel))
            .collect();
        assert_eq!(drawn, expected);
    }

    /// A neutral effect hands on the surface that arrived; a retune reaches
    /// the next frame, and a surface of the wrong size is refused by name.
    #[test]
    fn neutral_passes_through_and_retuning_and_refusals_behave() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let Ok((mut element, handle)) = CudaVideoEffect::new(
            "effect",
            &device,
            2,
            1,
            VideoEffect::LumaKey(LumaKey::default()),
        ) else {
            return;
        };
        let received = capture(&mut element);
        let Some(source) = cuda_row(&device, &[[128, 128, 128, 255]; 2]) else {
            return;
        };
        let MediaBuffer::Video(input) = &source else {
            unreachable!();
        };
        let input_id = crate::buffer::picture_id(input);

        element.consume(source.clone()).expect("neutral");
        handle.set_effect(VideoEffect::LumaKey(LumaKey {
            max: 0.4,
            ..LumaKey::default()
        }));
        element.consume(source).expect("keyed");

        let mut received = received.lock().unwrap();
        let MediaBuffer::Video(passed) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(crate::buffer::picture_id(passed), input_id);
        let keyed = received.remove(1);
        assert_eq!(
            download_row(&device, keyed, 2)[0][3],
            0,
            "mid grey is above 0.4"
        );
        drop(received);

        let Some(wrong) = cuda_row(&device, &[[0, 0, 0, 255]; 3]) else {
            return;
        };
        assert!(matches!(
            element.consume(wrong),
            Err(crate::error::Error::CudaVideoEffectError(
                CudaVideoEffectError::DimensionMismatch {
                    actual_width: 3,
                    ..
                }
            ))
        ));
    }
}
