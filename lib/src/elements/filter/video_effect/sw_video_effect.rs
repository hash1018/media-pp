use std::sync::Arc;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::handle::{VideoEffectControl, VideoEffectHandle};
use super::options::{EffectParams, VideoEffect, apply};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
};

/// Output frames allocated once the input size is known — the same count
/// and the same reason as `SwChromaKey`'s.
const POOL_SIZE: usize = 4;

/// Errors specific to `SwVideoEffect`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwVideoEffectError {
    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input picture is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),

    /// The input is not BGRA.
    #[error(
        "SwVideoEffect only takes BGRA frames (place it after a Scaler converting to BGRA), \
         got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error(
        "SwVideoEffect only processes decoded Video frames, got a {0}; \
         connect it after a decoder or scaler, not a demuxer"
    )]
    UnsupportedBuffer(&'static str),
}

/// Applies a [`VideoEffect`] — a colour correction or a luma key — to a
/// decoded BGRA frame, on the CPU. A `Filter`: receives via `Sink`, pushes
/// the result on through its own single src pad.
///
/// The software member of a family whose GPU members are `D3d11VideoEffect`
/// and `CudaVideoEffect`. All three evaluate one resolved set of numbers
/// per pixel, so an effect means the same thing on each.
///
/// BGRA in and BGRA out, for the reason `SwChromaKey` gives: a luma key's
/// result is an alpha channel, and a colour correction is defined on RGB.
/// PTS and the colour tags pass through.
///
/// An effect that changes nothing — [`ColorCorrection::default`], or a
/// [`LumaKey`] whose range is everything — and an element that is turned
/// off both hand each frame straight through, the same picture rather than
/// a copy of it.
///
/// [`ColorCorrection::default`]: super::ColorCorrection::default
/// [`LumaKey`]: super::LumaKey
pub struct SwVideoEffect {
    pp_log: PpLog,
    name: Arc<str>,
    /// The effect in force and what it resolves to, refreshed from `control`
    /// once per frame.
    effect: VideoEffect,
    params: EffectParams,
    control: Arc<VideoEffectControl>,
    enabled: bool,

    dims: Option<(u32, u32)>,
    pool: Option<UnboundObjectPool<ffmpeg::frame::Video>>,
    /// The last output and the picture it was made from — see
    /// [`RepeatedOutput`]. Cleared when the effect changes, since a picture
    /// corrected one way is not an answer to the same picture corrected
    /// another.
    repeated: RepeatedOutput,
    pad: SrcPad,
}

impl SwVideoEffect {
    /// Creates a software video-effect filter, and the handle that retunes
    /// it while it runs.
    pub fn new(name: impl Into<String>, effect: VideoEffect) -> (Self, VideoEffectHandle) {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwVideoEffect, &name, None);
        pp_info!(pp_log: &pp_log, "created: {} {effect:?}", effect.name());
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::System,
            )),
        );
        let control = Arc::new(VideoEffectControl::new(effect));
        let handle = VideoEffectHandle::new(control.clone());
        let element = Self {
            name,
            pp_log,
            effect,
            params: effect.params(),
            control,
            enabled: true,
            dims: None,
            pool: None,
            repeated: RepeatedOutput::new(),
            pad,
        };
        (element, handle)
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
            // The pooled output the cache holds, which an element that is
            // off cannot use — the reason `SwChromaKey` clears it too.
            self.repeated.clear();
        }
    }

    fn ensure_pool(&mut self, width: u32, height: u32) {
        if self.dims == Some((width, height)) {
            return;
        }
        pp_debug!(self, "output is {width}x{height} BGRA, (re)building pool");
        self.pool = Some(UnboundObjectPool::new(
            POOL_SIZE,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        ));
        self.dims = Some((width, height));
    }
}

impl Element for SwVideoEffect {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwVideoEffect
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for SwVideoEffect {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwVideoEffect {
    /// Works pixel by pixel on the CPU; the GPU counterparts take frames
    /// resident on their own device.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
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
                Err(SwVideoEffectError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // A pure per-pixel transform: nothing buffered beyond the cached
        // output.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

impl PerFrameTransform for SwVideoEffect {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        SwVideoEffectError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        if frame.format() != ffmpeg::format::Pixel::BGRA {
            pp_error!(self, "unsupported pixel format: {:?}", frame.format());
            return Err(SwVideoEffectError::UnsupportedFormat(frame.format()).into());
        }
        self.ensure_pool(frame.width(), frame.height());
        let mut output = self
            .pool
            .as_ref()
            .expect("built above for this exact size")
            .get();
        apply_to_frame(&self.params, frame, &mut output);
        output.set_pts(frame.pts());
        output.set_color_space(frame.color_space());
        output.set_color_range(frame.color_range());
        Ok(output)
    }
}

/// Every pixel of `source` through `params` into `destination`. Both are
/// BGRA at the same size — `ensure_pool` sees to that.
fn apply_to_frame(
    params: &EffectParams,
    source: &ffmpeg::frame::Video,
    destination: &mut ffmpeg::frame::Video,
) {
    let width = source.width() as usize;
    let height = source.height() as usize;
    let source_stride = source.stride(0);
    let destination_stride = destination.stride(0);
    let source_data = source.data(0);
    let destination_data = destination.data_mut(0);

    for row in 0..height {
        let source_row = &source_data[row * source_stride..row * source_stride + width * 4];
        let destination_row =
            &mut destination_data[row * destination_stride..row * destination_stride + width * 4];
        for (src, dst) in source_row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(destination_row.as_chunks_mut::<4>().0)
        {
            *dst = apply(params, *src);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::elements::{ColorCorrection, LumaKey};

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

    fn new_effect(
        effect: VideoEffect,
    ) -> (
        SwVideoEffect,
        VideoEffectHandle,
        Arc<Mutex<Vec<MediaBuffer>>>,
    ) {
        let (mut element, handle) = SwVideoEffect::new("effect", effect);
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        (element, handle, received)
    }

    fn bgra_frame(width: u32, height: u32, pixel: [u8; 4]) -> MediaBuffer {
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        );
        let mut frame = pool.get();
        let stride = frame.stride(0);
        {
            let data = frame.data_mut(0);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let offset = y * stride + x * 4;
                    data[offset..offset + 4].copy_from_slice(&pixel);
                }
            }
        }
        frame.set_pts(Some(7));
        MediaBuffer::Video(Arc::new(frame))
    }

    fn first_pixel(buffer: &MediaBuffer) -> [u8; 4] {
        let MediaBuffer::Video(frame) = buffer else {
            panic!("expected a Video buffer");
        };
        frame.data(0)[0..4].try_into().unwrap()
    }

    fn brighter() -> VideoEffect {
        VideoEffect::ColorCorrection(ColorCorrection {
            brightness: 0.2,
            ..ColorCorrection::default()
        })
    }

    #[test]
    fn every_pixel_is_what_the_shared_definition_says() {
        let (mut element, _handle, received) = new_effect(brighter());
        element
            .consume(bgra_frame(3, 2, [10, 20, 30, 255]))
            .expect("apply");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        let expected = apply(&brighter().params(), [10, 20, 30, 255]);
        for y in 0..2 {
            for x in 0..3 {
                let offset = y * frame.stride(0) + x * 4;
                assert_eq!(frame.data(0)[offset..offset + 4], expected);
            }
        }
        assert_eq!(frame.pts(), Some(7), "the pts is carried through");
    }

    /// A neutral effect and a disabled one both forward the picture that
    /// arrived — no pool slot spent, no pass over any pixel.
    #[test]
    fn nothing_to_do_hands_the_same_picture_through() {
        for (effect, enabled) in [
            (
                VideoEffect::ColorCorrection(ColorCorrection::default()),
                true,
            ),
            (VideoEffect::LumaKey(LumaKey::default()), true),
            (brighter(), false),
        ] {
            let (mut element, handle, received) = new_effect(effect);
            handle.set_enabled(enabled);
            let source = bgra_frame(2, 2, [10, 20, 30, 255]);
            let MediaBuffer::Video(input) = &source else {
                unreachable!();
            };
            let id = crate::buffer::picture_id(input);

            element.consume(source.clone()).expect("pass through");

            let received = received.lock().unwrap();
            let MediaBuffer::Video(output) = &received[0] else {
                panic!("expected a Video buffer");
            };
            assert_eq!(
                crate::buffer::picture_id(output),
                id,
                "{effect:?}, {enabled}"
            );
        }
    }

    /// Retuning reaches the pixels — including a switch to the other kind
    /// of effect — and is not answered out of a cache made the old way.
    #[test]
    fn retuning_changes_what_comes_out_even_for_a_repeat() {
        let (mut element, handle, received) = new_effect(brighter());
        let source = bgra_frame(2, 2, [128, 128, 128, 255]);

        element.consume(source.clone()).expect("first");
        handle.set_effect(VideoEffect::LumaKey(LumaKey {
            max: 0.4,
            ..LumaKey::default()
        }));
        element.consume(source).expect("same picture, new effect");

        let received = received.lock().unwrap();
        assert_eq!(first_pixel(&received[0]), [179, 179, 179, 255]);
        assert_eq!(first_pixel(&received[1])[3], 0, "mid grey is above 0.4");
    }

    #[test]
    fn a_non_bgra_frame_is_a_typed_error() {
        let (mut element, _handle, _received) = new_effect(brighter());
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 4, 4),
            |_| {},
        );
        let error = element
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("YUV420P must be refused");
        assert!(matches!(
            error,
            crate::error::Error::SwVideoEffectError(SwVideoEffectError::UnsupportedFormat(
                ffmpeg::format::Pixel::YUV420P
            ))
        ));
    }

    #[test]
    fn audio_is_refused_and_eos_forwarded() {
        let (mut element, _handle, received) = new_effect(brighter());
        let error = element
            .consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())))
            .expect_err("audio must be refused");
        assert!(matches!(
            error,
            crate::error::Error::SwVideoEffectError(SwVideoEffectError::UnsupportedBuffer("Audio"))
        ));
        element.consume(MediaBuffer::Eos).expect("eos");
        assert!(received.lock().unwrap()[0].is_eos());
    }
}
