//! [`SubtitleEncoder`]: what a transcriber heard, as a subtitle track.

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use crate::{
    buffer::{MediaBuffer, Metadata},
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Attached, ElementType, Identity},
    elements::TrackFormat,
    error::Result,
    subtitle,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::meta::Transcript;

/// The unit a [`SubtitleEncoder`]'s packets are timed in: milliseconds,
/// which is what an [`Utterance`](super::meta::Utterance) says its times in,
/// so a line's place in the track is its own.
const TIME_BASE: ffmpeg::Rational = ffmpeg::Rational(1, 1000);

/// Why a [`SubtitleEncoder`] refused a buffer.
#[derive(Debug, ThisError)]
pub enum SubtitleEncoderError {
    /// It was handed something other than decoded sound.
    #[error("SubtitleEncoder takes decoded sound carrying a Transcript, got {0}")]
    UnsupportedBuffer(&'static str),
}

/// Turns the lines a transcriber heard into subtitle packets, for a file's
/// text track or a subtitle file of its own.
///
/// It goes after a whisper transcriber, such as `SwWhisperTranscriber`
/// (feature `whisper`), and reads
/// the [`Transcript`] on the sound it hands on: each line becomes one packet
/// in the [`subtitle::Codec`] it was made for — `MovText` for an MP4's track,
/// `SubRip` for an `.srt`, `WebVtt` for a `.vtt` — timed where the line was
/// said. The sound itself goes no further. Register its track with
/// [`FileMuxer::add_stream`](crate::elements::FileMuxer::add_stream)`(name,
/// &encoder)`, which takes the codec and time base from it. Needs no
/// feature.
///
/// # Example
///
/// The lines heard in a file, beside its picture and sound in a copy of it —
/// the sound split by a [`Tee`](crate::elements::Tee), one branch copied as
/// it is and one transcribed:
///
/// ```no_run
/// # #[cfg(not(feature = "whisper"))]
/// # fn main() {}
/// # #[cfg(feature = "whisper")]
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{
///     AudioFormat, AudioResampler, FileDemuxer, FileMuxer, Mp4Layout, SubtitleEncoder,
///     SwDecoder, SwWhisperTranscriber, WHISPER_SAMPLE_RATE, WhisperOptions,
/// };
/// use media_pp::{bus::BusEvent, ffmpeg, pipeline::Pipeline, subtitle};
///
/// let (source, _) = FileDemuxer::open("demux", "talk.mp4")?;
/// let audio = source.best(ffmpeg::media::Type::Audio)?;
/// let decoder = SwDecoder::new("decoder", audio.parameters.clone())?;
/// let mono_f32 = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
/// let resampler = AudioResampler::new("resampler", AudioFormat::new(mono_f32, WHISPER_SAMPLE_RATE, 1));
/// let transcriber = SwWhisperTranscriber::new("whisper", "ggml-base.bin", WhisperOptions::default())?;
/// let encoder = SubtitleEncoder::new("captions", subtitle::Codec::MovText);
///
/// let mut muxer = FileMuxer::create("captioned.mp4", Mp4Layout::Standard)?;
/// let sound = muxer.add_stream("sound", &audio)?;
/// let text = muxer.add_stream("text", &encoder)?;
/// let mut sinks = muxer.open()?;
/// let (sound, text) = (sinks.take(sound)?, sinks.take(text)?);
///
/// let (pipeline, ()) = Pipeline::new("caption", source, |source, ctx| {
///     let copy = ctx.branch().to(sound)?;
///     let listen = ctx
///         .branch()
///         .pipe(decoder)
///         .pipe(resampler)
///         .queue("speech", 64)
///         .pipe(transcriber)
///         .pipe(encoder)
///         .to(text)?;
///     let tee = ctx.tee("sound").branch(copy).branch(listen).build()?;
///     ctx.attach(source, audio.index, tee)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// for event in pipeline.bus().iter() {
///     if matches!(event, BusEvent::Finished) {
///         pipeline.stop();
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// A transcriber reports each line once, in order, never overlapping the one
/// before — the shape a text track needs, since an MP4's cannot hold
/// overlap. A line the model gave no length would be on screen for no time
/// at all, and stretching it would overlap whatever starts where it does,
/// so it is left out of the track and said in the log at debug level.
///
/// It holds nothing: a line goes on as soon as the sound carrying it
/// arrives, and the end of the stream is the track's end.
pub struct SubtitleEncoder(FilterStage<Encoding>);

filter_stage!(SubtitleEncoder);

/// What a [`SubtitleEncoder`] does with each buffer of sound.
pub(crate) struct Encoding {
    pp_log: PpLog,
    codec: subtitle::Codec,
}

impl SubtitleEncoder {
    /// A subtitle encoder writing `codec`.
    pub fn new(name: impl Into<String>, codec: subtitle::Codec) -> Self {
        let identity = Identity::new(ElementType::SubtitleEncoder, name);
        let pp_log = identity.pp_log().clone();
        pp_info!(pp_log: &pp_log, "opened: codec={codec:?}");
        Self(FilterStage::new(identity, Encoding { pp_log, codec }))
    }

    /// The codec its packets are in.
    pub fn codec(&self) -> subtitle::Codec {
        self.0.inner().codec
    }
}

/// The track a [`SubtitleEncoder`]'s packets go in: its codec, timed in
/// milliseconds.
impl From<&SubtitleEncoder> for TrackFormat {
    fn from(encoder: &SubtitleEncoder) -> Self {
        TrackFormat::new(encoder.codec().parameters(), TIME_BASE)
    }
}

impl Filter for Encoding {
    /// Decoded sound, as a transcriber hands it on.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(PortContract::packet(MediaKind::SubtitlePacket))
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        if !matches!(buf, MediaBuffer::Audio(_)) {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(SubtitleEncoderError::UnsupportedBuffer(kind).into());
        }
        let Some(transcript) = buf.metadata().and_then(Metadata::get::<Transcript>) else {
            return Ok(());
        };
        for line in &transcript.utterances {
            let duration = line.end_ms - line.start_ms;
            if duration <= 0 {
                pp_debug!(self, "a line with no length is left out: {:?}", line.text);
                continue;
            }
            out.push(
                self.codec
                    .packet_in(&line.text, line.start_ms, duration, TIME_BASE),
            );
        }
        Ok(())
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::elements::speech::meta::Utterance;

    /// What `encoder` makes of `buf`.
    fn encode(encoder: &mut SubtitleEncoder, buf: MediaBuffer) -> Vec<MediaBuffer> {
        let mut out = Output::default();
        encoder
            .0
            .inner_mut()
            .transform(buf, &mut out)
            .expect("encoded");
        out.into_buffers()
    }

    fn speech(lines: &[(i64, i64, &str)]) -> MediaBuffer {
        let frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            16,
            ffmpeg::channel_layout::ChannelLayout::MONO,
        );
        let buf = MediaBuffer::Audio(Arc::new(frame).into());
        let utterances = lines
            .iter()
            .map(|&(start_ms, end_ms, text)| Utterance {
                start_ms,
                end_ms,
                text: text.to_owned(),
                words: Vec::new(),
            })
            .collect();
        buf.with_metadata(Metadata::default().with(Transcript { utterances }))
    }

    /// Each line becomes one packet, timed in milliseconds where it was
    /// said, and a line with no length none.
    #[test]
    fn each_line_is_one_packet_where_it_was_said() {
        let mut encoder = SubtitleEncoder::new("srt", subtitle::Codec::SubRip);
        let packets = encode(
            &mut encoder,
            speech(&[(1_000, 2_500, "안녕하세요"), (2_500, 2_500, "음")]),
        );
        assert_eq!(packets.len(), 1, "the line with no length is left out");
        let MediaBuffer::Packet(packet) = &packets[0] else {
            panic!("a packet");
        };
        assert_eq!(packet.pts(), Some(1_000));
        assert_eq!(packet.duration(), 1_500);
        assert_eq!(packets[0].time_base(), Some(TIME_BASE));
        assert_eq!(packet.data(), Some("안녕하세요".as_bytes()));
    }

    /// Sound with nothing heard in it makes nothing, and is not handed on.
    #[test]
    fn sound_without_a_transcript_makes_nothing() {
        let mut encoder = SubtitleEncoder::new("srt", subtitle::Codec::SubRip);
        let frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            16,
            ffmpeg::channel_layout::ChannelLayout::MONO,
        );
        let packets = encode(&mut encoder, MediaBuffer::Audio(Arc::new(frame).into()));
        assert!(packets.is_empty());
    }

    /// Its track takes the codec and time base from it.
    #[test]
    fn its_track_is_its_codec_in_milliseconds() {
        let encoder = SubtitleEncoder::new("vtt", subtitle::Codec::WebVtt);
        let format = TrackFormat::from(&encoder);
        assert_eq!(format.time_base, TIME_BASE);
        assert_eq!(format.parameters.id(), ffmpeg::codec::Id::WEBVTT);
    }
}
