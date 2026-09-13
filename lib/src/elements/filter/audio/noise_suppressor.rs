use std::{collections::VecDeque, sync::Arc};

use crate::pp_log::{PpLog, pp_info};
use ffmpeg_next as ffmpeg;
use nnnoiseless::DenoiseState;
use thiserror::Error as ThisError;

use super::audio_f32::{self, Unreadable};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
};

/// The one rate RNNoise was trained at, and so the one it takes.
pub const NOISE_SUPPRESSOR_SAMPLE_RATE: u32 = 48_000;

/// Samples RNNoise takes and gives back at a time: 10 ms.
const BLOCK: usize = DenoiseState::FRAME_SIZE;

/// RNNoise scales its input like 16-bit samples, not like `f32` audio.
const I16_SCALE: f32 = 32_768.0;

/// Errors specific to [`NoiseSuppressor`].
#[derive(Debug, ThisError, PartialEq)]
pub enum NoiseSuppressorError {
    /// The input is not at 48 kHz, the only rate RNNoise works at. An
    /// [`AudioResampler`](crate::elements::AudioResampler) in front converts
    /// it.
    #[error("NoiseSuppressor takes 48000 Hz audio, got {0} Hz; put an AudioResampler in front")]
    UnsupportedSampleRate(u32),

    /// The input frame does not describe any audio channels.
    #[error("audio frame has no channel layout")]
    MissingChannels,

    /// The input is not `f32`.
    #[error(
        "NoiseSuppressor takes f32 audio, packed or planar, got {0:?}; put an AudioResampler in front"
    )]
    UnsupportedSampleFormat(ffmpeg::format::Sample),

    /// A packed frame's plane is shorter than its declared samples need.
    #[error(
        "audio plane is too small for its declared format: need {required} bytes, got {actual}"
    )]
    InvalidPlaneSize {
        /// Bytes the declared samples need.
        required: usize,
        /// Bytes the plane has.
        actual: usize,
    },

    /// The sink received a buffer other than decoded audio or end-of-stream.
    #[error("NoiseSuppressor only processes decoded Audio frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

impl From<Unreadable> for NoiseSuppressorError {
    fn from(unreadable: Unreadable) -> Self {
        match unreadable {
            Unreadable::Format(format) => Self::UnsupportedSampleFormat(format),
            Unreadable::NoChannels => Self::MissingChannels,
            Unreadable::TooShort { required, actual } => {
                Self::InvalidPlaneSize { required, actual }
            }
        }
    }
}

/// One channel's denoiser and what is waiting on either side of it.
struct Channel {
    denoise: Box<DenoiseState<'static>>,
    /// Input not yet a whole block.
    pending: Vec<f32>,
    /// Denoised samples not yet handed on, already lined up with the input
    /// they came from.
    ready: VecDeque<f32>,
}

impl Channel {
    fn new() -> Self {
        Self {
            denoise: DenoiseState::new(),
            pending: Vec::with_capacity(BLOCK),
            ready: VecDeque::new(),
        }
    }
}

/// Takes background noise out of speech — fans, keyboards, a room — with
/// RNNoise, the recurrent network a streaming application's noise
/// suppression uses. Behind the `rnnoise` feature.
///
/// Takes decoded `f32` audio at 48 kHz, packed or planar, any number of
/// channels, each denoised on its own. What comes out is frame for frame
/// what went in — the same sample counts, format, layout and timestamps —
/// with the samples replaced.
///
/// # Delay
///
/// RNNoise works on 10 ms blocks and gives back each block one block late,
/// so a frame is handed on only once the input after it has arrived: 10 ms
/// after it, plus whatever it takes a frame not a multiple of 10 ms long to
/// complete a block. The frame keeps its own timestamps, and its samples
/// are the denoised samples *of that frame* — the block RNNoise holds back
/// is taken off the front rather than left to shift the sound late against
/// its own clock. So the delay is in when a frame arrives, never in what
/// its timestamps say it is.
///
/// # EOS, control and errors
///
/// `Eos` drains what is still held back by feeding RNNoise silence behind
/// it, then passes on. `Flush` and `Stop` drop it and start RNNoise afresh,
/// since what follows is a different stretch of sound. A frame whose
/// channel count differs from the last drains what the old layout held
/// first, as `Eos` would, and starts again with the new one. A frame at the
/// wrong rate or in the wrong format is an error for that frame and changes
/// nothing.
///
/// The network's weights are compiled into the crate, so nothing is loaded
/// from disk. Each channel keeps one RNNoise state, about 100 KB.
pub struct NoiseSuppressor {
    pp_log: PpLog,
    name: Arc<str>,
    channels: Vec<Channel>,
    /// Input frames not yet handed on, oldest first. Each goes out as it came
    /// in, with its samples replaced, once enough is ready.
    waiting: VecDeque<Arc<ffmpeg::frame::Audio>>,
    /// Denoised samples still to be thrown away: the block RNNoise returns
    /// before it has heard anything.
    lead_in: usize,
    pad: SrcPad,
}

impl NoiseSuppressor {
    /// Creates a suppressor. It takes its channel count from the first frame.
    pub fn new(name: impl Into<String>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::NoiseSuppressor, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name: name.clone(),
            pp_log,
            channels: Vec::new(),
            waiting: VecDeque::new(),
            lead_in: BLOCK,
            pad: SrcPad::with_contract(
                format!("{name}_src"),
                OutputContract::Fixed(PortContract::frame(
                    MediaKind::AudioFrame,
                    MemoryDomain::System,
                )),
            ),
        }
    }

    /// Starts again with `channels` and nothing held.
    fn restart(&mut self, channels: usize) {
        self.channels = (0..channels).map(|_| Channel::new()).collect();
        self.waiting.clear();
        self.lead_in = BLOCK;
    }

    /// Denoises every whole block the channels hold, throwing away what is
    /// left of the lead-in.
    fn denoise_blocks(&mut self) {
        // Every channel is fed the same samples, so all hold the same number
        // of blocks and pass the lead-in together.
        let blocks = self
            .channels
            .first()
            .map_or(0, |channel| channel.pending.len() / BLOCK);
        let skip = self.lead_in.min(blocks * BLOCK);
        let mut input = [0.0; BLOCK];
        let mut output = [0.0; BLOCK];
        for channel in &mut self.channels {
            let mut skip = skip;
            for _ in 0..blocks {
                for (scaled, sample) in input.iter_mut().zip(channel.pending.drain(..BLOCK)) {
                    *scaled = sample * I16_SCALE;
                }
                channel.denoise.process_frame(&mut output, &input);
                let from = skip.min(BLOCK);
                skip -= from;
                channel
                    .ready
                    .extend(output[from..].iter().map(|sample| sample / I16_SCALE));
            }
        }
        self.lead_in -= skip;
    }

    /// Hands on every waiting frame whose denoised samples are all ready.
    fn release_ready(&mut self) -> Result<()> {
        while let Some(front) = self.waiting.front() {
            let samples = front.samples();
            if self
                .channels
                .iter()
                .any(|channel| channel.ready.len() < samples)
            {
                break;
            }
            let frame = self.waiting.pop_front().expect("front was just seen");
            let denoised: Vec<Vec<f32>> = self
                .channels
                .iter_mut()
                .map(|channel| channel.ready.drain(..samples).collect())
                .collect();
            // Copied only when another branch still holds it — see
            // `AudioVolume`.
            let mut frame = Arc::try_unwrap(frame).unwrap_or_else(|shared| shared.as_ref().clone());
            audio_f32::write(&mut frame, &denoised);
            self.pad.push(MediaBuffer::Audio(Arc::new(frame)))?;
        }
        Ok(())
    }

    /// Feeds silence behind what is held until every waiting frame has gone
    /// out — at most two blocks: the one completing the last partial block,
    /// and the one RNNoise needs to give that back.
    fn drain(&mut self) -> Result<()> {
        while !self.waiting.is_empty() {
            for channel in &mut self.channels {
                let short = BLOCK - channel.pending.len() % BLOCK;
                channel.pending.extend(std::iter::repeat_n(0.0, short));
            }
            self.denoise_blocks();
            self.release_ready()?;
        }
        // The silence fed in was never anyone's input, and RNNoise now holds
        // some of it: whatever comes next starts from nothing, lead-in and
        // all, rather than behind what is left of it.
        let channels = self.channels.len();
        self.restart(channels);
        Ok(())
    }

    fn process(&mut self, frame: Arc<ffmpeg::frame::Audio>) -> Result<()> {
        let rate = frame.rate();
        if rate != NOISE_SUPPRESSOR_SAMPLE_RATE {
            return Err(NoiseSuppressorError::UnsupportedSampleRate(rate).into());
        }
        let samples = audio_f32::read(&frame).map_err(NoiseSuppressorError::from)?;
        if samples.len() != self.channels.len() {
            // A different layout is a different signal: what the old one
            // held goes out first, and the new one starts from nothing.
            if !self.channels.is_empty() {
                self.drain()?;
            }
            self.restart(samples.len());
        }
        for (channel, samples) in self.channels.iter_mut().zip(samples) {
            channel.pending.extend(samples);
        }
        self.waiting.push_back(frame);
        self.denoise_blocks();
        self.release_ready()
    }
}

impl Element for NoiseSuppressor {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::NoiseSuppressor
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for NoiseSuppressor {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for NoiseSuppressor {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => self.process(frame),
            MediaBuffer::Eos => {
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            MediaBuffer::Packet(_) => Err(NoiseSuppressorError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(NoiseSuppressorError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            let channels = self.channels.len();
            self.restart(channels);
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ffmpeg::format::sample::Type;

    use super::*;
    use crate::elements::filter::audio::audio_f32::tests::frame;

    const RATE: u32 = NOISE_SUPPRESSOR_SAMPLE_RATE;

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

    fn suppressor() -> (NoiseSuppressor, Arc<Mutex<Vec<MediaBuffer>>>) {
        let mut suppressor = NoiseSuppressor::new("denoise");
        let received = Arc::new(Mutex::new(Vec::new()));
        suppressor.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        (suppressor, received)
    }

    /// Feeds `signal`, one channel, in frames of `size` samples each
    /// stamped with its first sample's index, then `Eos`.
    fn feed(suppressor: &mut NoiseSuppressor, signal: &[f32], size: usize) {
        for (index, chunk) in signal.chunks(size).enumerate() {
            let mut frame = Arc::try_unwrap(frame(&[chunk.to_vec()], RATE, Type::Packed)).unwrap();
            frame.set_pts(Some((index * size) as i64));
            suppressor
                .consume(MediaBuffer::Audio(Arc::new(frame)))
                .unwrap();
        }
        suppressor.consume(MediaBuffer::Eos).unwrap();
    }

    /// Every audio frame that came out, as (pts, first channel).
    fn frames_out(received: &Arc<Mutex<Vec<MediaBuffer>>>) -> Vec<(Option<i64>, Vec<f32>)> {
        received
            .lock()
            .unwrap()
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Audio(frame) => {
                    Some((frame.pts(), audio_f32::read(frame).unwrap().remove(0)))
                }
                _ => None,
            })
            .collect()
    }

    fn energy(samples: &[f32]) -> f32 {
        samples.iter().map(|sample| sample * sample).sum()
    }

    /// A deterministic white noise — the same every run, so the test is
    /// about RNNoise and not about the draw.
    fn noise(amplitude: f32, samples: usize) -> Vec<f32> {
        let mut state = 0x2545_f491_u32;
        (0..samples)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * amplitude
            })
            .collect()
    }

    /// The block RNNoise holds back is taken off the front, so what comes out
    /// under a timestamp is what went in under it. Near-silence shows it:
    /// RNNoise applies no gain to a block it hears as silent, only its
    /// overlap-add and a high-pass for DC, which give a 380 Hz tone back
    /// within a few percent. The same output measured against the input one
    /// block later — where it would sit had the block not been taken off —
    /// is nothing like it, which is what makes the first comparison mean
    /// something.
    #[test]
    fn what_comes_out_under_a_timestamp_is_what_went_in_under_it() {
        let (mut suppressor, received) = suppressor();
        let input: Vec<f32> = (0..RATE as usize / 5)
            .map(|index| 1e-7 * (index as f32 * 0.05).sin())
            .collect();
        feed(&mut suppressor, &input, 480);
        let output: Vec<f32> = frames_out(&received)
            .into_iter()
            .flat_map(|(_, samples)| samples)
            .collect();
        assert_eq!(output.len(), input.len());

        // Past the first two blocks, while RNNoise's window and its
        // high-pass settle.
        let from = 2 * BLOCK;
        let off_by = |lag: usize| {
            let error: Vec<f32> = output[from..]
                .iter()
                .zip(&input[from - lag..])
                .map(|(out, into)| out - into)
                .collect();
            (energy(&error) / energy(&input[from..])).sqrt()
        };
        assert!(off_by(0) < 0.1, "aligned: {:.3} of the signal", off_by(0));
        assert!(
            off_by(BLOCK) > 0.5,
            "a block late would not be: {:.3}",
            off_by(BLOCK)
        );
    }

    /// The point of it: steady broadband noise, the fan and the air
    /// conditioner, is taken most of the way out once RNNoise has heard a
    /// little of it.
    #[test]
    fn steady_noise_is_taken_down_by_more_than_ten_decibels() {
        let (mut suppressor, received) = suppressor();
        let input = noise(0.03, RATE as usize * 2);
        feed(&mut suppressor, &input, 480);
        let output: Vec<f32> = frames_out(&received)
            .into_iter()
            .flat_map(|(_, samples)| samples)
            .collect();
        let settled = RATE as usize / 2;
        let before = energy(&input[settled..]);
        let after = energy(&output[settled..]);
        assert!(
            after < before / 10.0,
            "{:.1} dB",
            10.0 * (after / before).log10()
        );
    }

    /// Frame for frame: whatever sizes arrive — not multiples of RNNoise's
    /// block, a capture's odd lengths — each goes out once, with its own
    /// timestamp and sample count, in order, and the last of them at `Eos`.
    #[test]
    fn every_frame_goes_out_once_with_its_own_timestamp_whatever_its_size() {
        let (mut suppressor, received) = suppressor();
        let sizes = [441, 1024, 100, 480, 7, 960, 333];
        let mut pts = 0;
        let mut expected = Vec::new();
        for (index, size) in sizes.into_iter().enumerate() {
            let mut frame =
                Arc::try_unwrap(frame(&[noise(0.01, size)], RATE, Type::Planar)).unwrap();
            frame.set_pts(Some(pts));
            expected.push((Some(pts), size));
            pts += size as i64;
            suppressor
                .consume(MediaBuffer::Audio(Arc::new(frame)))
                .unwrap();
            // Held back until RNNoise has given back what they need.
            assert!(received.lock().unwrap().len() <= index);
        }
        suppressor.consume(MediaBuffer::Eos).unwrap();

        let out: Vec<_> = frames_out(&received)
            .into_iter()
            .map(|(pts, samples)| (pts, samples.len()))
            .collect();
        assert_eq!(out, expected);
        assert!(matches!(
            received.lock().unwrap().last(),
            Some(MediaBuffer::Eos)
        ));
    }

    /// A different channel count is a different signal: what the old one
    /// held goes out first, then the new one starts from nothing.
    #[test]
    fn a_new_channel_count_drains_the_old_one_first() {
        let (mut suppressor, received) = suppressor();
        suppressor
            .consume(MediaBuffer::Audio(frame(
                &[noise(0.01, 480)],
                RATE,
                Type::Packed,
            )))
            .unwrap();
        assert!(received.lock().unwrap().is_empty(), "held back a block");
        let stereo = vec![noise(0.01, 480), noise(0.01, 480)];
        suppressor
            .consume(MediaBuffer::Audio(frame(&stereo, RATE, Type::Packed)))
            .unwrap();
        let received = received.lock().unwrap();
        let MediaBuffer::Audio(first) = &received[0] else {
            panic!("audio")
        };
        assert_eq!(first.channels(), 1, "the mono frame, drained");
        assert_eq!(received.len(), 1, "the stereo one waits for its own block");
    }

    /// A `Flush` drops what was held rather than handing on sound from
    /// before a seek, and the next frame is held back as the first one was.
    #[test]
    fn a_flush_drops_what_was_held() {
        let (mut suppressor, received) = suppressor();
        suppressor
            .consume(MediaBuffer::Audio(frame(
                &[noise(0.01, 480)],
                RATE,
                Type::Packed,
            )))
            .unwrap();
        suppressor.control(ControlMsg::Flush).unwrap();
        suppressor.consume(MediaBuffer::Eos).unwrap();
        assert!(matches!(
            received.lock().unwrap().as_slice(),
            [MediaBuffer::Eos]
        ));
    }

    #[test]
    fn a_frame_at_another_rate_or_format_is_an_error_and_changes_nothing() {
        let (mut suppressor, received) = suppressor();
        let error = suppressor
            .consume(MediaBuffer::Audio(frame(
                &[vec![0.0; 441]],
                44_100,
                Type::Packed,
            )))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::NoiseSuppressorError(NoiseSuppressorError::UnsupportedSampleRate(44_100))
        ));
        let mut integers = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::I16(Type::Packed),
            480,
            ffmpeg::ChannelLayout::MONO,
        );
        integers.set_rate(RATE);
        let error = suppressor
            .consume(MediaBuffer::Audio(Arc::new(integers)))
            .unwrap_err();
        assert!(matches!(
            error,
            crate::Error::NoiseSuppressorError(NoiseSuppressorError::UnsupportedSampleFormat(_))
        ));
        suppressor.consume(MediaBuffer::Eos).unwrap();
        assert!(matches!(
            received.lock().unwrap().as_slice(),
            [MediaBuffer::Eos]
        ));
    }
}
