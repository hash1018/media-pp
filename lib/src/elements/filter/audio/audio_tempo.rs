use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, Rescale};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Context, Element, ElementType, Sink, Source, element_pp_log},
    elements::AudioFormat,
    error::Result,
    playback_clock::PlaybackClock,
    playback_state::PlaybackState,
    pp_log::PpLog,
};

use super::stretcher::{Piece, Stretcher};

/// Errors specific to [`AudioTempo`].
#[derive(Debug, ThisError)]
pub enum AudioTempoError {
    /// FFmpeg could not stretch the sound.
    #[error("stretching the sound failed: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The input frame says nothing a stretch could be set up from: no
    /// sample rate, or no channels.
    #[error("audio frame has no sample rate or no channels")]
    UnknownFormat,

    /// It was handed something other than decoded audio or the end of the
    /// stream.
    #[error("AudioTempo only takes decoded Audio frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Stretches sound to the pipeline's playback rate without changing its
/// pitch: at twice the rate, what a pacer hands on in a second is two
/// seconds of sound, and this makes it one again.
///
/// What it is for is sound going somewhere that takes it at the wall
/// clock's pace rather than the pipeline's: an [`crate::elements::AudioMixer`]
/// input, an encoder. It goes after the [`crate::elements::Pacer`] that
/// times the sound, and reads the rate from the pipeline's playback clock
/// as each frame comes, so [`crate::pipeline::Pipeline::set_rate`] reaches
/// it with nothing of its own to set. The audio renderers stretch their
/// sound themselves, and need none.
///
/// At the file's own speed a frame goes on as it came, and nothing else is
/// done. Stretched, a frame's `pts` is where its first sample is in the
/// media, in its own sample rate: each sample out stands for `rate` of the
/// media. Playing backwards no sound reaches it; a rate's direction does
/// not change how much is stretched. The format — sample format, rate and
/// channel layout — goes on as it came; a change of it partway ends the
/// stretch of the old and begins one of the new.
///
/// A `Flush` forgets what a stretch holds, and the end of the stream hands
/// it on before the `Eos`. While a preroll runs a frame goes on as it came:
/// what a preroll asks of the terminal after this is one sample, and a
/// stretch hands on nothing until it has been given more than that.
pub struct AudioTempo {
    pp_log: PpLog,
    name: Arc<str>,
    pad: crate::pad::SrcPad,
    /// The pipeline's; `None` for one no pipeline wired, which passes
    /// everything as it came.
    playback_clock: Option<Arc<PlaybackClock>>,
    /// The pipeline's, for whether a preroll is running.
    state: Option<Arc<PlaybackState>>,
    /// What stretches, and the format and layout it was made for.
    stretcher: Option<(AudioFormat, u64, Stretcher)>,
}

impl AudioTempo {
    /// One that stretches to the rate of the pipeline it is wired into, and
    /// passes everything as it came in none.
    pub fn new(name: impl Into<String>) -> Self {
        let name: Arc<str> = name.into().into();
        Self {
            pp_log: element_pp_log(ElementType::AudioTempo, &name, None),
            pad: crate::pad::SrcPad::with_contract(
                format!("{name}_src"),
                OutputContract::Fixed(PortContract::frame(
                    MediaKind::AudioFrame,
                    MemoryDomain::System,
                )),
            ),
            name,
            playback_clock: None,
            state: None,
            stretcher: None,
        }
    }

    /// How much sound is stretched now: the pipeline's rate, whichever way.
    fn rate(&self) -> f64 {
        self.playback_clock
            .as_ref()
            .map_or(1.0, |clock| clock.rate().abs())
    }

    fn stretch(&mut self, frame: Arc<ffmpeg::frame::Audio>) -> Result<()> {
        if self
            .state
            .as_ref()
            .is_some_and(|state| state.is_prerolling())
        {
            return self.pad.push(MediaBuffer::Audio(frame));
        }
        let rate = self.rate();
        if self
            .stretcher
            .as_ref()
            .is_none_or(|(_, _, stretcher)| stretcher.passes(rate))
            && rate == 1.0
        {
            return self.pad.push(MediaBuffer::Audio(frame));
        }
        let (sample_rate, channels) = (frame.rate(), frame.channels());
        if sample_rate == 0 || channels == 0 {
            return Err(AudioTempoError::UnknownFormat.into());
        }
        let format = AudioFormat::new(frame.format(), sample_rate, channels);
        let channel_layout = frame.channel_layout();
        let layout = channel_layout.bits();
        if self
            .stretcher
            .as_ref()
            .is_some_and(|(made, made_layout, _)| *made != format || *made_layout != layout)
        {
            self.finish()?;
        }
        let (_, _, stretcher) = self.stretcher.get_or_insert_with(|| {
            (
                format,
                layout,
                Stretcher::with_layout(format, channel_layout),
            )
        });
        let nanos = ffmpeg::Rational::new(1, 1_000_000_000);
        let media_ns = frame
            .pts()
            .zip(crate::buffer::time_base(&frame))
            .map_or(0, |(pts, base)| pts.rescale(base, nanos));
        let pieces = stretcher
            .stretch(&frame, media_ns, rate)
            .map_err(AudioTempoError::from)?;
        self.hand_on(pieces, Some(frame))
    }

    /// Hands on what the stretch put out, `frame` for a piece that is the
    /// frame as it came.
    fn hand_on(
        &mut self,
        pieces: Vec<Piece>,
        frame: Option<Arc<ffmpeg::frame::Audio>>,
    ) -> Result<()> {
        let mut frame = frame;
        for piece in pieces {
            match piece {
                Piece::AsIs => {
                    if let Some(frame) = frame.take() {
                        self.pad.push(MediaBuffer::Audio(frame))?;
                    }
                }
                Piece::Stretched {
                    mut frame,
                    media_ns,
                    ..
                } => {
                    let base = ffmpeg::Rational::new(1, frame.rate() as i32);
                    frame.set_pts(Some(
                        media_ns.rescale(ffmpeg::Rational::new(1, 1_000_000_000), base),
                    ));
                    crate::buffer::set_time_base(&mut frame, base);
                    self.pad.push(MediaBuffer::Audio(Arc::new(frame)))?;
                }
            }
        }
        Ok(())
    }

    /// Hands on everything a stretch holds, and forgets it.
    fn finish(&mut self) -> Result<()> {
        let Some((_, _, mut stretcher)) = self.stretcher.take() else {
            return Ok(());
        };
        let pieces = stretcher.finish().map_err(AudioTempoError::from)?;
        self.hand_on(pieces, None)
    }
}

impl Element for AudioTempo {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AudioTempo
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        self.playback_clock = Some(Arc::clone(&context.playback_clock));
        self.state = Some(Arc::clone(&context.state));
    }
}

impl Source for AudioTempo {
    fn src_pads(&mut self) -> &mut [crate::pad::SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for AudioTempo {
    /// Decoded sound, in any format; it goes on in the one it came in.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => self.stretch(frame),
            MediaBuffer::Eos => {
                self.finish()?;
                self.pad.push(MediaBuffer::Eos)
            }
            MediaBuffer::Packet(_) => Err(AudioTempoError::UnsupportedBuffer("Packet").into()),
            MediaBuffer::Video(_) => Err(AudioTempoError::UnsupportedBuffer("Video").into()),
        }
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop)
            && let Some((_, _, stretcher)) = &mut self.stretcher
        {
            stretcher.reset();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::clock::Clock;
    use crate::elements::AppSink;

    const RATE: u32 = 48_000;

    /// A tempo wired to a context of its own, and what it hands on.
    fn tempo() -> (AudioTempo, Arc<PlaybackClock>, Arc<Mutex<Vec<MediaBuffer>>>) {
        let context = Arc::new(Context::for_test_with_clock(
            crate::bus::Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::new(Clock::new()),
        ));
        let mut tempo = AudioTempo::new("tempo");
        tempo.attach_context(&context);
        let out = Arc::new(Mutex::new(Vec::new()));
        let collected = Arc::clone(&out);
        tempo.src_pads()[0].link(Box::new(AppSink::new("out", move |buf| {
            collected.lock().unwrap().push(buf);
            Ok(())
        })));
        (tempo, Arc::clone(&context.playback_clock), out)
    }

    /// A tenth of a second of planar stereo, `at` tenths in, as a decoder
    /// hands it on.
    fn tenth(at: i64) -> Arc<ffmpeg::frame::Audio> {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
            RATE as usize / 10,
            ffmpeg::ChannelLayout::default(2),
        );
        frame.set_rate(RATE);
        frame.set_pts(Some(at * i64::from(RATE) / 10));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, RATE as i32));
        Arc::new(frame)
    }

    fn samples(out: &[MediaBuffer]) -> usize {
        out.iter()
            .map(|buf| match buf {
                MediaBuffer::Audio(frame) => frame.samples(),
                _ => 0,
            })
            .sum()
    }

    /// At the file's own speed each frame goes on as it came: the same one.
    #[test]
    fn at_one_a_frame_goes_on_as_it_came() {
        let (mut tempo, _clock, out) = tempo();
        let frame = tenth(0);
        tempo
            .consume(MediaBuffer::Audio(Arc::clone(&frame)))
            .expect("consume");
        let out = out.lock().unwrap();
        let [MediaBuffer::Audio(went)] = out.as_slice() else {
            panic!("one frame on: {}", out.len());
        };
        assert!(Arc::ptr_eq(went, &frame));
    }

    /// At twice the rate two seconds of sound go on as one, in the format
    /// they came in, the first at where the sound began in the media; the
    /// end of the stream hands on what the stretch held before it.
    #[test]
    fn at_twice_the_rate_sound_goes_on_half_as_long() {
        let (mut tempo, clock, out) = tempo();
        clock.set_rate(2.0);
        for at in 10..30 {
            tempo
                .consume(MediaBuffer::Audio(tenth(at)))
                .expect("consume");
        }
        tempo.consume(MediaBuffer::Eos).expect("eos");
        let out = out.lock().unwrap();
        assert!(matches!(out.last(), Some(MediaBuffer::Eos)), "the end last");
        let got = samples(&out) as f64;
        let expected = f64::from(RATE);
        assert!(
            (got - expected).abs() < expected * 0.03,
            "{got} samples, not about {expected}"
        );
        let MediaBuffer::Audio(first) = &out[0] else {
            panic!("sound first");
        };
        assert_eq!(first.format(), tenth(0).format(), "planar still");
        assert_eq!(first.pts(), Some(i64::from(RATE)), "a second in");
        assert_eq!(
            crate::buffer::time_base(first),
            Some(ffmpeg::Rational::new(1, RATE as i32))
        );
    }

    /// A flush forgets what a stretch held: nothing of it goes on after.
    #[test]
    fn a_flush_forgets_what_was_held() {
        let (mut tempo, clock, out) = tempo();
        clock.set_rate(2.0);
        tempo
            .consume(MediaBuffer::Audio(tenth(0)))
            .expect("consume");
        tempo.control(&ControlMsg::Flush).expect("flush");
        let before = out.lock().unwrap().len();
        tempo.consume(MediaBuffer::Eos).expect("eos");
        let out = out.lock().unwrap();
        assert_eq!(out.len(), before + 1, "only the end");
    }
}
