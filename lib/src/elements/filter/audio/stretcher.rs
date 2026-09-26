//! Sound stretched to a playback rate without its pitch changing — what
//! [`crate::elements::AudioTempo`] and the audio renderers share.
//!
//! [`Pipeline::set_rate`](crate::pipeline::Pipeline::set_rate) changes how
//! fast the playback clock covers media, and a pacer hands the sound on that
//! much faster or slower. Whatever takes it at the wall clock's pace — a
//! device, a mixer — needs it stretched back to its own length: each sample
//! out standing for `rate` samples of the media.

use ffmpeg_next::{self as ffmpeg, ffi};

use crate::elements::AudioFormat;

/// Stretches audio to a playback rate without changing its pitch — FFmpeg's
/// `atempo`, one graph per rate.
///
/// At 1.0 nothing is stretched and nothing is copied: the frame goes on as
/// it came. A rate change drains the graph for the old rate first, so the
/// sound it still held goes before the sound after it, at the rate it was
/// stretched to.
pub(crate) struct Stretcher {
    format: AudioFormat,
    /// The channel layout, as the mask the graph is told it by: a mask
    /// rather than a `ChannelLayout`, which holds a pointer and is not `Send`.
    layout: u64,
    graph: Option<Stretching>,
}

// `Send`, as every renderer holding one has to be: a field that is not
// broke the Linux renderer while the Windows one, which says it is `Send`
// for reasons of its own, still built. Checked here, on every platform.
const _: fn() = || {
    fn send<T: Send>() {}
    send::<Stretcher>();
};

/// One rate's graph, and where in the media its next output starts.
struct Stretching {
    graph: ffmpeg::filter::Graph,
    rate: f64,
    /// Samples fed in, which is what each frame's `pts` counts.
    fed: i64,
    /// Where the next sample out of the graph is, in the media.
    media_ns: i64,
}

/// What goes on next, in order.
pub(crate) enum Piece {
    /// The frame it was given, as it came: nothing is being stretched.
    AsIs,
    /// Sound stretched to `rate`, the first sample of which is at `media_ns`.
    Stretched {
        frame: ffmpeg::frame::Audio,
        media_ns: i64,
        /// Read by the renderers, which say where playback is from it.
        #[cfg_attr(
            not(any(
                all(target_os = "windows", feature = "wasapi-renderer"),
                all(target_os = "linux", feature = "pipewire-audio-renderer")
            )),
            allow(dead_code)
        )]
        rate: f64,
    },
}

impl Stretcher {
    #[cfg_attr(
        not(any(
            all(target_os = "windows", feature = "wasapi-renderer"),
            all(target_os = "linux", feature = "pipewire-audio-renderer")
        )),
        allow(dead_code)
    )]
    /// For frames in `format`, laid out as its channel count says by
    /// default — a renderer's own format.
    pub(crate) fn new(format: AudioFormat) -> Self {
        Self::with_layout(
            format,
            ffmpeg::ChannelLayout::default(i32::from(format.channels)),
        )
    }

    /// For frames in `format` whose channels are laid out as `layout`.
    pub(crate) fn with_layout(format: AudioFormat, layout: ffmpeg::ChannelLayout) -> Self {
        Self {
            format,
            layout: layout.bits(),
            graph: None,
        }
    }

    /// Whether a frame at `rate` goes on as it is, with nothing held from
    /// before to go ahead of it — what a caller checks first, so playing at
    /// the file's own speed costs nothing.
    pub(crate) fn passes(&self, rate: f64) -> bool {
        rate == 1.0 && self.graph.is_none()
    }

    /// `frame`, whose first sample is at `media_ns`, at `rate`: what goes on
    /// for it, in order. `frame` is referenced, not copied, and not changed.
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
            self.graph = Some(Stretching::open(self.format, self.layout, rate, media_ns)?);
        }
        let stretching = self.graph.as_mut().expect("opened above");
        // A reference of its own, which the graph takes the buffers of and
        // which carries the graph's count as its `pts`: `frame` itself may
        // be shared, and is left as it is.
        let mut input = ffmpeg::frame::Audio::empty();
        // SAFETY: both frames are live; a reference to a refcounted frame
        // shares its buffers, and one to any other copies them.
        let code = unsafe { ffi::av_frame_ref(input.as_mut_ptr(), frame.as_ptr()) };
        if code < 0 {
            return Err(ffmpeg::Error::from(code));
        }
        input.set_pts(Some(stretching.fed));
        stretching.fed += frame.samples() as i64;
        stretching
            .graph
            .get("in")
            .expect("the graph has its source")
            .source()
            .add(&input)?;
        stretching.take(self.format, &mut pieces)?;
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
    fn open(
        format: AudioFormat,
        layout: u64,
        rate: f64,
        media_ns: i64,
    ) -> Result<Self, ffmpeg::Error> {
        let mut graph = ffmpeg::filter::Graph::new();
        let source_args = format!(
            "time_base=1/{rate}:sample_rate={rate}:sample_fmt={format}:channel_layout=0x{mask:x}",
            rate = format.sample_rate,
            format = format.sample_format.name(),
            mask = layout,
        );
        let abuffer = ffmpeg::filter::find("abuffer").ok_or(ffmpeg::Error::FilterNotFound)?;
        let abuffersink =
            ffmpeg::filter::find("abuffersink").ok_or(ffmpeg::Error::FilterNotFound)?;
        graph.add(&abuffer, "in", &source_args)?;
        graph.add(&abuffersink, "out", "")?;
        // `atempo` takes 0.5 to 100 at once, and sounds best within 0.5 to
        // 2: further than that is several of it. `aformat` holds the output
        // to the format it came in, which is what goes on.
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
            layout
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

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    fn packed() -> AudioFormat {
        AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            RATE,
            2,
        )
    }

    /// `samples` of a 440 Hz tone, stereo, packed.
    fn tone(samples: usize) -> ffmpeg::frame::Audio {
        let format = packed();
        let mut frame = ffmpeg::frame::Audio::new(
            format.sample_format,
            samples,
            ffmpeg::ChannelLayout::default(2),
        );
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
            let mut stretcher = Stretcher::new(packed());
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
                assert_eq!(frame.format(), packed().sample_format);
                assert_eq!(frame.rate(), RATE);
            }
        }
    }

    /// Planar sound, as a decoder hands it on, stretches the same, and
    /// stays planar.
    #[test]
    fn planar_sound_is_stretched_as_it_comes() {
        let planar = AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
            RATE,
            2,
        );
        let mut stretcher = Stretcher::new(planar);
        let mut pieces = Vec::new();
        for _ in 0..20 {
            let mut frame = ffmpeg::frame::Audio::new(
                planar.sample_format,
                RATE as usize / 10,
                ffmpeg::ChannelLayout::default(2),
            );
            frame.set_rate(RATE);
            pieces.extend(stretcher.stretch(&frame, 0, 2.0).expect("stretch"));
        }
        pieces.extend(stretcher.finish().expect("finish"));
        let expected = 2.0 * f64::from(RATE) / 2.0;
        let got = samples_of(&pieces) as f64;
        assert!((got - expected).abs() < expected * 0.03, "{got} samples");
        for piece in &pieces {
            let Piece::Stretched { frame, .. } = piece else {
                panic!("stretched");
            };
            assert_eq!(frame.format(), planar.sample_format);
        }
    }

    /// At the file's own speed nothing is stretched; a change from another
    /// rate first hands on what that rate's graph still held.
    #[test]
    fn at_one_nothing_is_stretched_and_a_change_plays_out_what_was_held() {
        let mut stretcher = Stretcher::new(packed());
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
}
