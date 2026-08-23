use std::sync::Arc;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::options::ChromaKeyOptions;
use crate::{
    buffer::MediaBuffer,
    color::Color,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// How many output frames [`SwChromaKey`] pre-allocates once it learns its
/// input dimensions — see [`SwScaler`](super::super::scaler::SwScaler)'s
/// identical constant for the reasoning; this one just can't be sized at
/// construction time since, unlike a scaler's fixed `dst_width`/
/// `dst_height`, output dimensions here are whatever the input turns out
/// to be.
const POOL_SIZE: usize = 4;

/// Errors specific to `SwChromaKey`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum SwChromaKeyError {
    /// The input video is not in the BGRA pixel format required by this filter.
    #[error(
        "SwChromaKey only keys BGRA frames (place it after a Scaler converting to BGRA), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error(
        "SwChromaKey only processes decoded Video frames, got a {0}; \
         connect it after a decoder/scaler, not a demuxer"
    )]
    UnsupportedBuffer(&'static str),
}

/// Keys a solid background color out of a decoded BGRA frame into alpha —
/// the software half of this crate's chroma-key support, the GPU-resident
/// other half being `D3d11ChromaKey`. A `Filter`:
/// receives via `Sink`, pushes the keyed frame on through its own (single)
/// src pad.
///
/// Expects `BGRA` input — the same format [`crate::elements::SwVideoCompositor`]/
/// `D3d11VideoCompositor` layers already use, so the
/// typical placement is right between a `Scaler` (converting a decoder's
/// YUV output to BGRA) and a compositor's `add_source`. RGB channels pass
/// through unchanged; only alpha is written.
pub struct SwChromaKey {
    pp_log: PpLog,
    name: Arc<str>,
    key_color: Color,
    threshold: f32,
    smoothing: f32,
    /// `None` until the first frame arrives, then rebuilt only if a later
    /// frame's dimensions differ — mirrors `SwScaler`'s lazily-built
    /// scaling `context`, except what's cached here is just the output
    /// pool's frame size, since keying itself needs no persistent
    /// per-resolution state.
    dims: Option<(u32, u32)>,
    pool: Option<UnboundObjectPool<ffmpeg::frame::Video>>,
    pad: SrcPad,
}

impl SwChromaKey {
    /// Creates a software chroma-key filter with fixed keying parameters.
    pub fn new(name: impl Into<String>, options: ChromaKeyOptions) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwChromaKey, &name, None);
        let key_color = options.method.key_color();
        pp_info!(
            pp_log: &pp_log,
            "created: key_color={key_color:?}, threshold={}, smoothing={}",
            options.threshold,
            options.smoothing
        );
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(
                PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::System),
            ),
        );
        Self {
            name,
            pp_log,
            key_color,
            threshold: options.threshold,
            smoothing: options.smoothing,
            dims: None,
            pool: None,
            pad,
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

impl Element for SwChromaKey {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwChromaKey
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for SwChromaKey {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for SwChromaKey {
    /// Keys pixel by pixel on the CPU; the GPU counterpart is D3d11ChromaKey.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::System),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if frame.format() != ffmpeg::format::Pixel::BGRA {
                    pp_error!(self, "unsupported pixel format: {:?}", frame.format());
                    return Err(SwChromaKeyError::UnsupportedFormat(frame.format()).into());
                }
                self.ensure_pool(frame.width(), frame.height());
                let mut output = self
                    .pool
                    .as_ref()
                    .expect("built above for this exact size")
                    .get();
                key_bgra(
                    &frame,
                    &mut output,
                    self.key_color,
                    self.threshold,
                    self.smoothing,
                );
                output.set_pts(frame.pts());
                self.pad.push(MediaBuffer::Video(Arc::new(output)))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => {
                pp_error!(self, "unsupported buffer: Packet");
                Err(SwChromaKeyError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(SwChromaKeyError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // A pure per-pixel transform, same as `SwScaler` — nothing local
        // buffered or ordered to flush on any `ControlMsg`.
        self.pad.control(msg)
    }
}

/// Writes `destination`'s alpha channel from `source`'s BGR distance to
/// `key_color`, copying RGB through unchanged. Both frames must already be
/// `BGRA` at the same dimensions — the caller (`SwChromaKey::consume`)
/// guarantees that via `ensure_pool`.
fn key_bgra(
    source: &ffmpeg::frame::Video,
    destination: &mut ffmpeg::frame::Video,
    key_color: Color,
    threshold: f32,
    smoothing: f32,
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
            let [b, g, r, _a] = *src;
            let distance = color_distance(b, g, r, key_color);
            *dst = [b, g, r, alpha_for(distance, threshold, smoothing)];
        }
    }
}

/// Normalized Euclidean BGR distance to `key`, scaled so the maximum
/// possible distance (opposite corners of the color cube) is `1.0` —
/// keeps `threshold`/`smoothing` in an intuitive `0.0..=1.0` range
/// regardless of channel depth.
fn color_distance(b: u8, g: u8, r: u8, key: Color) -> f32 {
    let db = (f32::from(b) - f32::from(key.blue)) / 255.0;
    let dg = (f32::from(g) - f32::from(key.green)) / 255.0;
    let dr = (f32::from(r) - f32::from(key.red)) / 255.0;
    (db * db + dg * dg + dr * dr).sqrt() / 3f32.sqrt()
}

/// Linear ramp from fully transparent (`distance <= threshold - smoothing
/// / 2`) to fully opaque (`distance >= threshold + smoothing / 2`).
/// `smoothing <= 0.0` collapses this to a hard step at `threshold`.
fn alpha_for(distance: f32, threshold: f32, smoothing: f32) -> u8 {
    let half = smoothing.max(0.0) / 2.0;
    let low = threshold - half;
    let high = threshold + half;
    let t = if high > low {
        ((distance - low) / (high - low)).clamp(0.0, 1.0)
    } else if distance <= threshold {
        0.0
    } else {
        1.0
    };
    (t * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{super::options::ChromaKeyMethod, *};

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

    fn new_chroma_key(options: ChromaKeyOptions) -> (SwChromaKey, Arc<Mutex<Vec<MediaBuffer>>>) {
        let mut key = SwChromaKey::new("key", options);
        let received = Arc::new(Mutex::new(Vec::new()));
        key.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        (key, received)
    }

    /// Builds a `BGRA` frame of `width`x`height` whose pixel at `(x, y)` is
    /// whatever `fill(x, y)` (as `[b, g, r, a]`) returns.
    fn bgra_frame(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> MediaBuffer {
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        );
        let mut frame = pool.get();
        let stride = frame.stride(0);
        {
            let data = frame.data_mut(0);
            for y in 0..height {
                for x in 0..width {
                    let offset = y as usize * stride + x as usize * 4;
                    data[offset..offset + 4].copy_from_slice(&fill(x, y));
                }
            }
        }
        frame.set_pts(Some(7));
        MediaBuffer::Video(Arc::new(frame))
    }

    fn pixel(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> [u8; 4] {
        let offset = y * frame.stride(0) + x * 4;
        frame.data(0)[offset..offset + 4].try_into().unwrap()
    }

    fn default_options() -> ChromaKeyOptions {
        ChromaKeyOptions {
            method: ChromaKeyMethod::Green,
            threshold: 0.15,
            smoothing: 0.1,
        }
    }

    #[test]
    fn a_pixel_matching_the_key_color_becomes_fully_transparent() {
        let (mut key, received) = new_chroma_key(default_options());
        let green = [0, 255, 0, 255]; // BGRA: pure green
        key.consume(bgra_frame(2, 2, |_, _| green))
            .expect("keying must succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(pixel(frame, 0, 0), [0, 255, 0, 0]);
    }

    #[test]
    fn a_clearly_different_pixel_stays_fully_opaque_and_unchanged() {
        let (mut key, received) = new_chroma_key(default_options());
        let red = [0, 0, 255, 255]; // BGRA: pure red, far from the green key
        key.consume(bgra_frame(2, 2, |_, _| red))
            .expect("keying must succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(pixel(frame, 0, 0), [0, 0, 255, 255]);
    }

    #[test]
    fn a_pixel_inside_the_smoothing_band_gets_a_partial_alpha() {
        let (mut key, received) = new_chroma_key(default_options());
        // A slightly-off green sitting inside the feather band around
        // threshold=0.15 (half-width 0.05, so the band is 0.10..0.20):
        // only the red channel differs, by 60/255, so distance =
        // (60/255) / sqrt(3) ~= 0.136.
        let almost_green = [0, 255, 60, 255];
        key.consume(bgra_frame(1, 1, |_, _| almost_green))
            .expect("keying must succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        let alpha = pixel(frame, 0, 0)[3];
        assert!(
            alpha > 0 && alpha < 255,
            "expected a feathered mid-range alpha, got {alpha}"
        );
    }

    #[test]
    fn a_custom_key_color_replaces_the_green_default() {
        let (mut key, received) = new_chroma_key(ChromaKeyOptions {
            method: ChromaKeyMethod::Custom(Color::new(10, 20, 30)),
            threshold: 0.05,
            smoothing: 0.0,
        });
        key.consume(bgra_frame(1, 1, |_, _| [30, 20, 10, 255]))
            .expect("keying must succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(pixel(frame, 0, 0)[3], 0);
    }

    #[test]
    fn pts_is_carried_through() {
        let (mut key, received) = new_chroma_key(default_options());
        key.consume(bgra_frame(1, 1, |_, _| [0, 255, 0, 255]))
            .expect("keying must succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(frame.pts(), Some(7));
    }

    #[test]
    fn non_bgra_input_is_a_typed_error_not_a_panic() {
        let (mut key, _received) = new_chroma_key(default_options());
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 4, 4),
            |_| {},
        );
        let frame = MediaBuffer::Video(Arc::new(pool.get()));

        let error = key.consume(frame).expect_err("YUV420P must be rejected");
        assert!(matches!(
            error,
            crate::error::Error::SwChromaKeyError(SwChromaKeyError::UnsupportedFormat(
                ffmpeg::format::Pixel::YUV420P
            ))
        ));
    }

    #[test]
    fn packet_and_audio_buffers_are_rejected_not_silently_dropped() {
        let (mut key, _received) = new_chroma_key(default_options());

        let error = key
            .consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())))
            .expect_err("an Audio buffer must be rejected");
        assert!(matches!(
            error,
            crate::error::Error::SwChromaKeyError(SwChromaKeyError::UnsupportedBuffer("Audio"))
        ));
    }

    #[test]
    fn eos_is_forwarded() {
        let (mut key, received) = new_chroma_key(default_options());
        key.consume(MediaBuffer::Eos).expect("Eos must forward");

        let received = received.lock().unwrap();
        assert!(matches!(received[0], MediaBuffer::Eos));
    }
}
