//! Sound at a playback rate, for the audio renderers: stretched to the rate
//! without its pitch changing, and a record of which media each sample
//! handed to a device stands for.
//!
//! [`Pipeline::set_rate`](crate::pipeline::Pipeline::set_rate) changes how
//! fast the playback clock covers media. A renderer is where that clock meets
//! the device, which plays samples at its own rate whatever the pipeline
//! says: so it is the renderer that has to hand the device the media sped up
//! or slowed down, and that has to say where playback is from what the
//! device has played — each sample of stretched sound stands for `rate`
//! samples' worth of media. Both renderers do it with these two, so neither
//! backend has a rate of its own to get right.

// Built for its own tests too, where no audio renderer uses all of it.
#![cfg_attr(
    not(any(
        all(target_os = "windows", feature = "wasapi-renderer"),
        all(target_os = "linux", feature = "pipewire-audio-renderer")
    )),
    allow(dead_code)
)]

use std::collections::VecDeque;

use ffmpeg_next as ffmpeg;

use crate::elements::AudioFormat;

/// Stretches packed audio in a renderer's own format to a playback rate
/// without changing its pitch — FFmpeg's `atempo`, one graph per rate.
///
/// At 1.0 nothing is stretched and nothing is copied: the renderer hands the
/// device its frame as it came. A rate change drains the graph for the old
/// rate first, so the sound it still held is played before the sound after
/// it, at the rate it was stretched to.
pub(crate) struct Stretcher {
    format: AudioFormat,
    graph: Option<Stretching>,
}

/// One rate's graph, and where in the media its next output starts.
struct Stretching {
    graph: ffmpeg::filter::Graph,
    rate: f64,
    /// Samples fed in, which is what each frame's `pts` counts.
    fed: i64,
    /// Where the next sample out of the graph is, in the media.
    media_ns: i64,
}

/// What a renderer hands its device next, in order.
pub(crate) enum Piece {
    /// The frame it was given, as it came: nothing is being stretched.
    AsIs,
    /// Sound stretched to `rate`, the first sample of which is at `media_ns`.
    Stretched {
        frame: ffmpeg::frame::Audio,
        media_ns: i64,
        rate: f64,
    },
}

impl Stretcher {
    pub(crate) fn new(format: AudioFormat) -> Self {
        Self {
            format,
            graph: None,
        }
    }

    /// Whether a frame at `rate` goes to the device as it is, with nothing
    /// held from before to go ahead of it — what a renderer checks first, so
    /// playing at the file's own speed costs nothing.
    pub(crate) fn passes(&self, rate: f64) -> bool {
        rate == 1.0 && self.graph.is_none()
    }

    /// `frame`, whose first sample is at `media_ns`, at `rate`: what to hand
    /// the device for it, in order.
    pub(crate) fn stretch(
        &mut self,
        frame: &ffmpeg::frame::Audio,
        media_ns: i64,
        rate: f64,
    ) -> Result<Vec<Piece>, ffmpeg::Error> {
        let mut pieces = Vec::new();
        if self.graph.as_ref().is_some_and(|graph| graph.rate != rate) {
            pieces.extend(self.finish()?);
        }
        if rate == 1.0 {
            pieces.push(Piece::AsIs);
            return Ok(pieces);
        }
        if self.graph.is_none() {
            self.graph = Some(Stretching::open(self.format, rate, media_ns)?);
        }
        let format = self.format;
        let stretching = self.graph.as_mut().expect("opened above");
        let mut input =
            ffmpeg::frame::Audio::new(format.sample_format, frame.samples(), layout(format));
        input.set_rate(format.sample_rate);
        let bytes = frame.samples() * bytes_per_frame(format);
        input.data_mut(0)[..bytes].copy_from_slice(&frame.data(0)[..bytes]);
        input.set_pts(Some(stretching.fed));
        stretching.fed += frame.samples() as i64;
        stretching
            .graph
            .get("in")
            .expect("the graph has its source")
            .source()
            .add(&input)?;
        stretching.take(format, &mut pieces)?;
        Ok(pieces)
    }

    /// Everything the graph still holds, stretched, and the graph closed:
    /// for the end of the stream, and for a rate change.
    pub(crate) fn finish(&mut self) -> Result<Vec<Piece>, ffmpeg::Error> {
        let mut pieces = Vec::new();
        if let Some(mut stretching) = self.graph.take() {
            stretching
                .graph
                .get("in")
                .expect("the graph has its source")
                .source()
                .flush()?;
            stretching.take(self.format, &mut pieces)?;
        }
        Ok(pieces)
    }

    /// Forgets what the graph holds: for a seek, and for a stop.
    pub(crate) fn reset(&mut self) {
        self.graph = None;
    }
}

impl Stretching {
    fn open(format: AudioFormat, rate: f64, media_ns: i64) -> Result<Self, ffmpeg::Error> {
        let mut graph = ffmpeg::filter::Graph::new();
        let layout = layout(format);
        let source_args = format!(
            "time_base=1/{rate}:sample_rate={rate}:sample_fmt={format}:channel_layout=0x{mask:x}",
            rate = format.sample_rate,
            format = format.sample_format.name(),
            mask = layout.bits(),
        );
        let abuffer = ffmpeg::filter::find("abuffer").ok_or(ffmpeg::Error::FilterNotFound)?;
        let abuffersink =
            ffmpeg::filter::find("abuffersink").ok_or(ffmpeg::Error::FilterNotFound)?;
        graph.add(&abuffer, "in", &source_args)?;
        graph.add(&abuffersink, "out", "")?;
        // `atempo` takes 0.5 to 100 at once, and sounds best within 0.5 to
        // 2: further than that is several of it. `aformat` holds the output
        // to the renderer's own format, which is what it hands its device.
        let mut chain = Vec::new();
        let mut left = rate;
        while left > 2.0 {
            chain.push("atempo=2".to_owned());
            left /= 2.0;
        }
        while left < 0.5 {
            chain.push("atempo=0.5".to_owned());
            left /= 0.5;
        }
        chain.push(format!("atempo={left}"));
        chain.push(format!(
            "aformat=sample_fmts={}:sample_rates={}:channel_layouts=0x{:x}",
            format.sample_format.name(),
            format.sample_rate,
            layout.bits()
        ));
        graph
            .output("in", 0)?
            .input("out", 0)?
            .parse(&chain.join(","))?;
        graph.validate()?;
        Ok(Self {
            graph,
            rate,
            fed: 0,
            media_ns,
        })
    }

    /// Takes every frame the graph has ready into `pieces`.
    fn take(&mut self, format: AudioFormat, pieces: &mut Vec<Piece>) -> Result<(), ffmpeg::Error> {
        let mut sink = self.graph.get("out").expect("the graph has its sink");
        loop {
            let mut frame = ffmpeg::frame::Audio::empty();
            match sink.sink().frame(&mut frame) {
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    return Ok(());
                }
                Err(error) => return Err(error),
            }
            if frame.samples() == 0 {
                continue;
            }
            let media_ns = self.media_ns;
            // Each sample stretched stands for `rate` samples of the media.
            self.media_ns = media_ns.saturating_add(
                (frame.samples() as f64 * self.rate * 1e9 / f64::from(format.sample_rate)) as i64,
            );
            pieces.push(Piece::Stretched {
                frame,
                media_ns,
                rate: self.rate,
            });
        }
    }
}

fn layout(format: AudioFormat) -> ffmpeg::ChannelLayout {
    ffmpeg::ChannelLayout::default(i32::from(format.channels))
}

fn bytes_per_frame(format: AudioFormat) -> usize {
    format.sample_format.bytes() * usize::from(format.channels)
}

/// Which media the samples handed to a device stand for, from the first on:
/// each span played at a rate covers that many times its own length of
/// media. Where playback is, is where the samples the device has played
/// reach.
///
/// Kept in samples, not nanoseconds, so a long stretch at one rate adds up
/// exactly. Spans the device has played through are forgotten as a
/// position is read past them.
pub(crate) struct PlayedMedia {
    sample_rate: u32,
    /// `(first sample, media there, rate)`, oldest first.
    spans: VecDeque<(u64, i64, f64)>,
    /// Samples handed over so far.
    handed: u64,
}

impl PlayedMedia {
    /// Nothing handed over yet, the first sample to be at `media_ns`.
    pub(crate) fn new(sample_rate: u32, media_ns: i64) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            spans: VecDeque::from([(0, media_ns, 1.0)]),
            handed: 0,
        }
    }

    /// `samples` more handed over, played at `rate`.
    pub(crate) fn push(&mut self, samples: u64, rate: f64) {
        let (_, _, last_rate) = *self.spans.back().expect("never empty");
        if last_rate != rate {
            let media_ns = self.media_at(self.handed);
            self.spans.push_back((self.handed, media_ns, rate));
        }
        self.handed += samples;
    }

    /// The media reached once the device has played `played` samples, at
    /// most as far as what it has been handed.
    pub(crate) fn media_at(&self, played: u64) -> i64 {
        let played = played.min(self.handed);
        let (start, media_ns, rate) = self
            .spans
            .iter()
            .rev()
            .find(|(start, _, _)| *start <= played)
            .copied()
            .expect("the first span starts at zero");
        let ns = (played - start) as f64 * 1e9 / f64::from(self.sample_rate);
        media_ns.saturating_add((ns * rate) as i64)
    }

    /// As [`Self::media_at`], forgetting the spans played through: what a
    /// renderer calls as it publishes where playback is.
    pub(crate) fn played(&mut self, played: u64) -> i64 {
        while self.spans.len() > 1 && self.spans[1].0 <= played {
            self.spans.pop_front();
        }
        self.media_at(played)
    }

    /// How far the media handed over reaches.
    pub(crate) fn handed_until(&self) -> i64 {
        self.media_at(self.handed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn format() -> AudioFormat {
        AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            RATE,
            2,
        )
    }

    /// A second of a 440 Hz tone, stereo.
    fn tone(samples: usize) -> ffmpeg::frame::Audio {
        let format = format();
        let mut frame = ffmpeg::frame::Audio::new(format.sample_format, samples, layout(format));
        frame.set_rate(RATE);
        let data = frame.plane_mut::<(f32, f32)>(0);
        for (index, sample) in data.iter_mut().enumerate() {
            let value = (index as f32 * 440.0 * std::f32::consts::TAU / RATE as f32).sin() * 0.5;
            *sample = (value, value);
        }
        frame
    }

    fn samples_of(pieces: &[Piece]) -> usize {
        pieces
            .iter()
            .map(|piece| match piece {
                Piece::AsIs => 0,
                Piece::Stretched { frame, .. } => frame.samples(),
            })
            .sum()
    }

    /// At twice the rate a second of sound becomes half a second, and at
    /// half the rate two, each stretched sample standing for `rate` of the
    /// media from where the sound began.
    #[test]
    fn sound_is_stretched_to_the_rate_and_says_what_media_it_stands_for() {
        for rate in [2.0, 0.5, 4.0, 0.25] {
            let mut stretcher = Stretcher::new(format());
            let mut pieces = Vec::new();
            for second in 0..4 {
                let frame = tone(RATE as usize / 10);
                for _ in 0..10 {
                    pieces.extend(
                        stretcher
                            .stretch(&frame, 1_000_000_000 * second, rate)
                            .expect("stretch"),
                    );
                }
            }
            pieces.extend(stretcher.finish().expect("finish"));
            let expected = 4.0 * f64::from(RATE) / rate;
            let got = samples_of(&pieces) as f64;
            assert!(
                (got - expected).abs() < expected * 0.03,
                "at {rate}, 4 s of sound came out as {got} samples, not about {expected}"
            );
            let Some(Piece::Stretched { media_ns, .. }) = pieces.first() else {
                panic!("stretched sound first");
            };
            assert_eq!(*media_ns, 0, "from where the sound began");
            for piece in &pieces {
                let Piece::Stretched { frame, .. } = piece else {
                    panic!("nothing as it is at {rate}");
                };
                assert_eq!(frame.format(), format().sample_format);
                assert_eq!(frame.rate(), RATE);
            }
        }
    }

    /// At the file's own speed nothing is stretched; a change from another
    /// rate first hands on what that rate's graph still held.
    #[test]
    fn at_one_nothing_is_stretched_and_a_change_plays_out_what_was_held() {
        let mut stretcher = Stretcher::new(format());
        assert!(stretcher.passes(1.0));
        let frame = tone(4_800);
        let pieces = stretcher.stretch(&frame, 0, 2.0).expect("stretch");
        assert!(!stretcher.passes(1.0), "a graph holds sound now");
        let before = samples_of(&pieces);
        let pieces = stretcher.stretch(&frame, 100_000_000, 1.0).expect("back");
        assert!(
            matches!(pieces.last(), Some(Piece::AsIs)),
            "then the frame itself"
        );
        assert!(
            before + samples_of(&pieces) > 2_000,
            "what the old graph held came out first"
        );
        assert!(stretcher.passes(1.0));
    }

    /// Where playback is, from what the device has played: at the rate each
    /// span was handed over at.
    #[test]
    fn played_samples_reach_the_media_their_rate_says() {
        let mut map = PlayedMedia::new(RATE, 5_000_000_000);
        map.push(u64::from(RATE), 1.0);
        map.push(u64::from(RATE), 2.0);
        map.push(u64::from(RATE) / 2, 0.5);
        assert_eq!(map.media_at(0), 5_000_000_000);
        assert_eq!(map.media_at(u64::from(RATE)), 6_000_000_000);
        assert_eq!(map.media_at(u64::from(RATE) * 3 / 2), 7_000_000_000);
        assert_eq!(map.handed_until(), 8_250_000_000);
        assert_eq!(
            map.played(u64::from(RATE) * 10),
            8_250_000_000,
            "no further than what was handed over"
        );
        assert_eq!(map.spans.len(), 1, "what was played through is forgotten");
    }
}
