//! Control conformance: random sequences of pause, resume, seek, frame step,
//! rate — backwards too — looping, finish and stop, run against every
//! shape of pipeline this crate is used in, and judged against what every
//! terminal was handed.
//!
//! # Why random, and why judged afterwards
//!
//! The control bugs this crate has had were not in one message but in an
//! order of them: a seek the moment a file ended, a pause in the last second
//! of it, a seek while paused at its end. Each was found by a user or a slow
//! CI runner and then pinned by a test of that one order. What they share is
//! a small set of promises a pipeline makes whatever the order, so these
//! tests make the orders and check the promises:
//!
//! - every call returns — a control call that does not, within
//!   [`OP_TIMEOUT`], fails the sequence with its history — and so does one
//!   a `stop` cuts short;
//! - once a pause has returned nothing new reaches a terminal, except the
//!   one picture a seek's preroll asks for and the pictures a step does;
//! - after a seek, what a terminal is handed is from the new position on,
//!   and goes forward — nothing from before the seek arrives after it —
//!   with nothing missing in between: no step between two samples much
//!   wider than the stream's own spacing, where nothing drops by design;
//! - playing backwards, the same going down from the new position, and no
//!   sound at all;
//! - resumed away from the end, it flows again — and a looping file has no
//!   end to be near, forwards;
//! - played to the end, the pipeline says [`BusEvent::Finished`], unless it
//!   loops, when it goes on into the next lap instead;
//! - finished, every terminal's last word is an `Eos`;
//! - every buffer a terminal is handed belongs to the segment it was handed
//!   last, and a segment on a new timeline follows a flush — see
//!   [`crate::stream`];
//! - and nothing reports an error.
//!
//! # Every shape, not a chosen few
//!
//! A file pipeline is built from one choice on each of six axes — how the
//! picture fans out, what decodes it, what filters it, what paces it, what
//! the sound goes through, how deep the queues are — see [`FileShape`].
//! [`matrix`] covers every pair of choices on any two axes at least once,
//! starting from the shapes these tests had by hand, so an element that
//! misbehaves only next to another is met next to it. `MEDIA_PP_CONTROL_FULL=1`
//! runs every combination instead.
//!
//! # Judged by what was asked, not by how it was carried
//!
//! A terminal records what it is handed, and the harness writes each call it
//! makes on the pipeline into every terminal's record as it begins and as it
//! returns — so the record interleaves the data with the calls, and "paused"
//! means "after `pause` returned", which is the promise a caller relies on.
//! Nothing here reads the control messages a terminal is handed, except
//! [`Recorder`]'s one line that turns the start of a new timeline into
//! [`Entry::Timeline`]: that is the line to change when the way a timeline is
//! announced does, and the promises stay as they are — which is what lets
//! these tests stand behind a change to how control travels.
//!
//! Nothing here times a picture against a clock, so a slow machine makes a
//! run slower rather than wrong — which matters, because a slow machine is
//! what these are for.
//!
//! # Running them harder
//!
//! By default each shape runs a few fixed sequences, so an ordinary test run
//! is deterministic. `MEDIA_PP_CONTROL_ITERS` runs more,
//! `MEDIA_PP_CONTROL_RANDOM=1` seeds them from the clock instead,
//! `MEDIA_PP_CONTROL_SEED` replays one seed a failure printed —
//! `MEDIA_PP_CONTROL_SHAPE` with the shape it printed runs that shape alone —
//! and `MEDIA_PP_CONTROL_TRACE=<directory>` writes this crate's log there at
//! `Trace`, every control message at every element, for reading what the
//! replay did. The races these look for show when threads are short of
//! cores, so CI runs them pinned to two: `taskset -c 0,1` on Linux, a
//! two-core affinity mask on Windows. That is how the seek-at-the-end
//! deadlock of 208af56 was reproduced, where a free machine never showed it.

use std::collections::HashSet;

use super::*;

use crate::buffer::time_base;
use crate::control::ControlMsg;
use crate::elements::{
    AudioFormat, AudioResampler, AudioTempo, ColorCorrection, DecodeTarget, FileDemuxerHandle,
    Rack, SwScaler, SwVideoEffect, VideoDecodeBin, VideoEffect,
};

/// How long any one control call may take before the sequence is failed as
/// hung. Generous, so a loaded two-core runner is never mistaken for a
/// deadlock: a real one does not return at all.
const OP_TIMEOUT: Duration = Duration::from_secs(20);

/// How far before a seek's target a sample may start and still be the one
/// covering it: a frame or an audio packet's worth, and some.
const LANDING_TOLERANCE: Duration = Duration::from_millis(100);

/// How far before the end of the file its last picture or sound may start.
/// An accurate seek at or past the end shows the last of the file rather than
/// nothing — see the decoders' preroll gate — so that is where such a seek's
/// data may begin.
const LAST_SAMPLE: Duration = Duration::from_millis(200);

/// How long a resumed pipeline, away from the end, may take to hand every
/// terminal something new.
const FLOW_TIMEOUT: Duration = Duration::from_secs(8);

/// How far from a lap's join a gap between two samples is the join's: a lap
/// is as long as its furthest packet, so the stream that ends first has the
/// difference missing at every join, by design.
const LAP_JOIN: Duration = Duration::from_millis(300);

/// What one terminal was handed, in the order it was handed it — and the
/// harness's calls, in the order they began and returned.
#[derive(Debug, Clone)]
enum Entry {
    Data {
        pts: Option<Duration>,
    },
    Eos,
    /// The stream moved to a new timeline, beginning at this position.
    Timeline(Duration),
    /// The stream began a segment — see [`crate::stream`].
    Segment {
        id: u64,
        flushed: bool,
    },
    /// Something was handed outside the segment it belongs to, or a segment
    /// came out of turn: what, for the failure to say.
    Outside(String),
    /// A call on the pipeline began, or returned — written by the harness
    /// into every terminal's record. What it was is `calls[index]`.
    Call {
        index: usize,
        returned: bool,
    },
}

/// A terminal that records rather than presents.
struct Recorder {
    pp_log: PpLog,
    name: Arc<str>,
    /// Used where a frame does not say its own time base.
    fallback: ffmpeg::Rational,
    log: Arc<Mutex<Vec<Entry>>>,
    /// How long it takes over each buffer — a slower terminal than its
    /// siblings, so one branch prerolls after the other.
    delay: Duration,
    /// The timeline of the segment it was handed last.
    segment: Option<u64>,
}

impl Recorder {
    fn new(name: &str, fallback: ffmpeg::Rational) -> (Self, Arc<Mutex<Vec<Entry>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                pp_log: element_pp_log(ElementType::Other, name, None),
                name: name.into(),
                fallback,
                log: Arc::clone(&log),
                delay: Duration::ZERO,
                segment: None,
            },
            log,
        )
    }

    fn slowed(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn pts(&self, frame: &ffmpeg::Frame) -> Option<Duration> {
        let base = time_base(frame).unwrap_or(self.fallback);
        let pts = frame.pts()?;
        Some(Duration::from_nanos(
            pts.rescale(base, ffmpeg::Rational::new(1, 1_000_000_000))
                .max(0) as u64,
        ))
    }
}

impl Element for Recorder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
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

impl Sink for Recorder {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        // What it is handed is on the timeline of the thread handing it —
        // which has to be the one the last segment opened.
        let number = crate::timeline::current();
        if number != crate::timeline::UNNUMBERED && self.segment != Some(number) {
            self.log.lock().unwrap().push(Entry::Outside(format!(
                "{} on timeline {number}, in the segment of {:?}",
                buf.kind(),
                self.segment
            )));
        }
        thread::sleep(self.delay);
        let entry = match &buf {
            MediaBuffer::Video(frame) => Entry::Data {
                pts: self.pts(frame),
            },
            MediaBuffer::Audio(frame) => Entry::Data {
                pts: self.pts(frame),
            },
            MediaBuffer::Eos => Entry::Eos,
            _ => Entry::Data { pts: None },
        };
        self.log.lock().unwrap().push(entry);
        Ok(())
    }

    /// The one place these tests read how control is carried: where a new
    /// timeline is announced to a terminal, today by a `Seek` reaching it.
    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        if let ControlMsg::Seek(target) = msg {
            self.log.lock().unwrap().push(Entry::Timeline(*target));
        }
        Ok(())
    }

    /// Each segment once, each on a later timeline than the one before it,
    /// and — but for the stream's first — after a flush.
    fn stream_event(&mut self, event: crate::stream::Event<'_>) -> Result<()> {
        let crate::stream::StreamEvent::Segment(segment) = event.0;
        let mut log = self.log.lock().unwrap();
        if let Some(before) = self.segment
            && (segment.id <= before || !segment.flushed)
        {
            log.push(Entry::Outside(format!(
                "segment on timeline {} after one on {before}, flushed: {}",
                segment.id, segment.flushed
            )));
        }
        self.segment = Some(segment.id);
        log.push(Entry::Segment {
            id: segment.id,
            flushed: segment.flushed,
        });
        Ok(())
    }
}

/// How a file's picture reaches its terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branches {
    /// One terminal.
    One,
    /// Decoded once, then fanned out to two paced branches.
    PictureTee,
    /// Fanned out as packets, each branch decoding its own, one slower to
    /// take a picture than the other — so one prerolls while the other is
    /// still catching up.
    PacketTee,
}

/// What decodes the picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Decode {
    Decoder,
    /// A `VideoDecodeBin`, which routes control into a line of its own.
    Bin,
}

/// What the decoded picture goes through before it is paced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Filter {
    None,
    /// A scaler, which answers a repeated picture with what it made.
    Scaled,
    /// To BGRA and through a `Rack` holding an effect — a Source's filters.
    Racked,
}

/// What times the picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pacing {
    Pacer,
    /// A `VideoSynchronizer`, which drops what is late by design.
    Synchronizer,
    /// Nothing: handed on as fast as it decodes.
    Unpaced,
}

/// What the file's sound goes through, if it is played at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sound {
    None,
    /// Paced, then stretched to the rate — how a mixer input is fed.
    Stretched,
    /// Paced, then resampled to another rate and layout.
    Resampled,
}

/// How deep the queues are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Depth {
    Normal,
    /// One deep, so backpressure is felt at every step — where a thread
    /// blocked handing data on meets control.
    Tight,
}

const BRANCHES: [Branches; 3] = [Branches::One, Branches::PictureTee, Branches::PacketTee];
const DECODE: [Decode; 2] = [Decode::Decoder, Decode::Bin];
const FILTER: [Filter; 3] = [Filter::None, Filter::Scaled, Filter::Racked];
const PACING: [Pacing; 3] = [Pacing::Pacer, Pacing::Synchronizer, Pacing::Unpaced];
const SOUND: [Sound; 3] = [Sound::None, Sound::Stretched, Sound::Resampled];
const DEPTH: [Depth; 2] = [Depth::Normal, Depth::Tight];
/// How many choices each axis has, in [`FileShape::axes`]' order.
const AXES: [usize; 6] = [3, 2, 3, 3, 3, 2];

/// A file pipeline, one choice per axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileShape {
    branches: Branches,
    decode: Decode,
    filter: Filter,
    pacing: Pacing,
    sound: Sound,
    depth: Depth,
}

impl FileShape {
    const fn new(branches: Branches, pacing: Pacing, sound: Sound, depth: Depth) -> Self {
        Self {
            branches,
            decode: Decode::Decoder,
            filter: Filter::None,
            pacing,
            sound,
            depth,
        }
    }

    /// Which choice this is on each axis.
    fn axes(self) -> [usize; 6] {
        let at = |found: Option<usize>| found.expect("every choice is listed");
        [
            at(BRANCHES.iter().position(|b| *b == self.branches)),
            at(DECODE.iter().position(|d| *d == self.decode)),
            at(FILTER.iter().position(|f| *f == self.filter)),
            at(PACING.iter().position(|p| *p == self.pacing)),
            at(SOUND.iter().position(|s| *s == self.sound)),
            at(DEPTH.iter().position(|d| *d == self.depth)),
        ]
    }

    fn from_axes(axes: [usize; 6]) -> Self {
        Self {
            branches: BRANCHES[axes[0]],
            decode: DECODE[axes[1]],
            filter: FILTER[axes[2]],
            pacing: PACING[axes[3]],
            sound: SOUND[axes[4]],
            depth: DEPTH[axes[5]],
        }
    }

    /// Every pair of choices it makes, on two different axes.
    fn pairs(self) -> Vec<(usize, usize, usize, usize)> {
        let axes = self.axes();
        let mut pairs = Vec::new();
        for a in 0..axes.len() {
            for b in a + 1..axes.len() {
                pairs.push((a, axes[a], b, axes[b]));
            }
        }
        pairs
    }

    fn label(self) -> String {
        format!(
            "{:?}-{:?}-{:?}-{:?}-{:?}-{:?}",
            self.branches, self.decode, self.filter, self.pacing, self.sound, self.depth
        )
        .to_lowercase()
    }
}

/// The shapes these tests had by hand, each still in the matrix as it was:
/// a player's picture, a whole player, a picture fanned out, one-deep
/// queues, a synchronized picture, and packets fanned out before decoding.
const NAMED: [FileShape; 6] = [
    FileShape::new(Branches::One, Pacing::Pacer, Sound::None, Depth::Normal),
    FileShape::new(
        Branches::One,
        Pacing::Pacer,
        Sound::Stretched,
        Depth::Normal,
    ),
    FileShape::new(
        Branches::PictureTee,
        Pacing::Pacer,
        Sound::None,
        Depth::Normal,
    ),
    FileShape::new(Branches::One, Pacing::Pacer, Sound::None, Depth::Tight),
    FileShape::new(
        Branches::One,
        Pacing::Synchronizer,
        Sound::None,
        Depth::Normal,
    ),
    FileShape::new(
        Branches::PacketTee,
        Pacing::Pacer,
        Sound::None,
        Depth::Normal,
    ),
];

/// Every combination of choices.
fn every_shape() -> Vec<FileShape> {
    let mut all = Vec::new();
    let mut axes = [0usize; 6];
    loop {
        all.push(FileShape::from_axes(axes));
        let mut axis = axes.len();
        loop {
            if axis == 0 {
                return all;
            }
            axis -= 1;
            axes[axis] += 1;
            if axes[axis] < AXES[axis] {
                break;
            }
            axes[axis] = 0;
        }
    }
}

/// One choice on one axis, as [`KNOWN_BROKEN`] names it.
// No known bug names a choice right now; the next one the matrix finds
// does, and until then nothing makes one.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum Choice {
    Branches(Branches),
}

impl Choice {
    fn made_by(self, shape: FileShape) -> bool {
        match self {
            Choice::Branches(branches) => shape.branches == branches,
        }
    }
}

/// What a known bug breaks, and the test that reproduces it — which is
/// where the bug is described.
struct KnownBroken {
    /// A shape making this choice is never played backwards: it runs, and
    /// turns round no more until the bug is fixed.
    backwards: Choice,
    test: &'static str,
}

/// What known bugs break, left out of the sequences until each is fixed.
/// Remove an entry with the `#[ignore]` on its test, once the test passes.
const KNOWN_BROKEN: [KnownBroken; 0] = [];

/// Whether `shape` may be played backwards: no known bug breaks it there.
fn turns_round(shape: FileShape) -> bool {
    !KNOWN_BROKEN
        .iter()
        .any(|known| known.backwards.made_by(shape))
}

/// The shapes a run covers: [`NAMED`], then — greedily, and the same every
/// time — whichever shape meets most pairs of choices no shape so far has,
/// until every pair has been met. Every combination instead with
/// `MEDIA_PP_CONTROL_FULL=1`.
fn matrix() -> Vec<FileShape> {
    let all = every_shape();
    if std::env::var_os("MEDIA_PP_CONTROL_FULL").is_some() {
        return all;
    }
    let mut uncovered: HashSet<_> = all.iter().flat_map(|shape| shape.pairs()).collect();
    let mut chosen = Vec::new();
    for shape in NAMED {
        for pair in shape.pairs() {
            uncovered.remove(&pair);
        }
        chosen.push(shape);
    }
    while !uncovered.is_empty() {
        let mut best = (0, all[0]);
        for shape in &all {
            let meets = shape
                .pairs()
                .iter()
                .filter(|pair| uncovered.contains(pair))
                .count();
            if meets > best.0 {
                best = (meets, *shape);
            }
        }
        for pair in best.1.pairs() {
            uncovered.remove(&pair);
        }
        chosen.push(best.1);
    }
    chosen
}

/// The shapes of pipeline the sequences run against.
#[derive(Debug, Clone, Copy)]
enum Shape {
    /// A file, played — see [`FileShape`].
    File(FileShape),
    /// A live source, which cannot be sought and never ends.
    Live,
    /// An offline compositor fed by a live source through a pipeline of its
    /// own — a render waiting on its input, holding that pipeline back a
    /// frame ahead of what it has drawn, while its own is paused, resumed
    /// and stopped. Not sought: it is not seekable.
    Offline,
    /// The same for sound: an offline mixer fed by a live tone.
    OfflineMix,
}

impl Shape {
    /// Whether every sample a terminal is handed after a seek must follow
    /// the one before it. Not for a `VideoSynchronizer`, which drops a
    /// picture that is late by design, nor a live source, which skips the
    /// ticks a starved thread missed.
    fn lossless(self) -> bool {
        match self {
            Shape::File(shape) => shape.pacing != Pacing::Synchronizer,
            Shape::Live => false,
            Shape::Offline | Shape::OfflineMix => true,
        }
    }

    fn label(self) -> String {
        match self {
            Shape::File(shape) => shape.label(),
            other => format!("{other:?}").to_lowercase(),
        }
    }
}

/// A built pipeline and what its terminals recorded.
struct Rig {
    pipeline: Arc<Pipeline>,
    duration: Option<Duration>,
    terminals: Vec<(&'static str, Arc<Mutex<Vec<Entry>>>)>,
    /// A file's looping, where the source is one.
    looping: Option<FileDemuxerHandle>,
    /// Pipelines feeding the one under test, kept running as long as it is.
    _feeds: Vec<Arc<Pipeline>>,
}

/// The picture decoded, and queued for what follows.
fn decoded(
    shape: FileShape,
    chain: ChainBuilder,
    parameters: &ffmpeg::codec::Parameters,
    name: &str,
    frames: usize,
) -> Result<ChainBuilder> {
    let chain = match shape.decode {
        Decode::Decoder => chain.pipe(SwDecoder::new(
            format!("{name}-decoder"),
            parameters.clone(),
        )?),
        Decode::Bin => chain.pipe(VideoDecodeBin::open(
            format!("{name}-decoder"),
            parameters.clone(),
            DecodeTarget::System,
            None,
        )?),
    };
    Ok(chain.queue(format!("{name}-frames"), frames))
}

/// The decoded picture through the shape's filter.
fn filtered(shape: FileShape, chain: ChainBuilder, name: &str) -> ChainBuilder {
    let bilinear = ffmpeg::software::scaling::Flags::BILINEAR;
    match shape.filter {
        Filter::None => chain,
        Filter::Scaled => chain.pipe(SwScaler::new(
            format!("{name}-scaler"),
            ffmpeg::format::Pixel::YUV420P,
            160,
            120,
            bilinear,
        )),
        Filter::Racked => {
            let frame = || PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System);
            let (rack, handle) = Rack::new(
                format!("{name}-rack"),
                InputContract::Fixed(frame()),
                OutputContract::Fixed(frame()),
            );
            let (effect, _) = SwVideoEffect::new(
                format!("{name}-effect"),
                VideoEffect::ColorCorrection(ColorCorrection::default()),
            );
            handle
                .replace(vec![Box::new(effect)])
                .expect("fill the rack");
            chain
                .pipe(SwScaler::new(
                    format!("{name}-to-bgra"),
                    ffmpeg::format::Pixel::BGRA,
                    160,
                    120,
                    bilinear,
                ))
                .pipe(rack)
        }
    }
}

/// The picture timed as the shape says.
fn paced(shape: FileShape, chain: ChainBuilder, name: &str) -> ChainBuilder {
    match shape.pacing {
        Pacing::Pacer => chain.pipe(Pacer::new(format!("{name}-pacer"))),
        Pacing::Synchronizer => chain.pipe(VideoSynchronizer::new(format!("{name}-sync"))),
        Pacing::Unpaced => chain,
    }
}

impl Rig {
    fn build(shape: Shape) -> Option<Self> {
        match shape {
            Shape::File(shape) => Self::build_file(shape),
            Shape::Live => Some(Self::build_live()),
            Shape::Offline => Some(Self::build_offline()),
            Shape::OfflineMix => Some(Self::build_offline_mix()),
        }
    }

    fn build_live() -> Self {
        let (recorder, log) = Recorder::new("screen", ffmpeg::Rational::new(1, 30));
        let source = TestVideoSource::new(
            "live",
            TestVideoOptions {
                width: 64,
                height: 48,
                frame_rate: ffmpeg::Rational::new(30, 1),
            },
        );
        let (pipeline, ()) = Pipeline::new("conformance-live", source, |source, ctx| {
            let branch = ctx.branch().queue("frames", 4).to(recorder)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the live pipeline");
        Self {
            pipeline,
            duration: None,
            terminals: vec![("screen", log)],
            looping: None,
            _feeds: Vec::new(),
        }
    }

    fn build_offline_mix() -> Self {
        let (recorder, log) = Recorder::new("speakers", ffmpeg::Rational::new(1, 48_000));
        let (mixer, handle) = crate::elements::AudioMixer::new(
            "offline-mix",
            crate::elements::AudioMixerOptions {
                sample_rate: 48_000,
                channels: 2,
                mode: crate::elements::RenderMode::Offline { end: None },
            },
        );
        let input = handle.add_source("tone").expect("add its input");
        let source = crate::elements::TestAudioSource::new(
            "tone",
            crate::elements::TestAudioOptions::default(),
        );
        let (feed, ()) = Pipeline::new("conformance-offline-mix-feed", source, |source, ctx| {
            let branch = ctx.branch().queue("tone-frames", 4).to(input)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the feed");
        feed.run().expect("run the feed");
        let (pipeline, ()) = Pipeline::new("conformance-offline-mix", mixer, |source, ctx| {
            let branch = ctx.branch().queue("mixed", 4).to(recorder)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the offline mix");
        Self {
            pipeline,
            duration: None,
            terminals: vec![("speakers", log)],
            looping: None,
            _feeds: vec![feed],
        }
    }

    fn build_offline() -> Self {
        let (recorder, log) = Recorder::new("screen", ffmpeg::Rational::new(1, 30));
        let (compositor, handle) = crate::elements::SwVideoCompositor::new(
            "offline",
            crate::elements::VideoCompositorOptions {
                width: 64,
                height: 48,
                frame_rate: ffmpeg::Rational::new(30, 1),
                mode: crate::elements::RenderMode::Offline { end: None },
                ..Default::default()
            },
        )
        .expect("make the offline compositor");
        let input = handle
            .add_source(
                "camera",
                crate::elements::VideoLayer::new(crate::elements::VideoRect::new(0, 0, 64, 48)),
            )
            .expect("add its input");
        let source = TestVideoSource::new(
            "live",
            TestVideoOptions {
                width: 64,
                height: 48,
                frame_rate: ffmpeg::Rational::new(30, 1),
            },
        );
        let (feed, ()) = Pipeline::new("conformance-offline-feed", source, |source, ctx| {
            let branch = ctx.branch().queue("camera-frames", 4).to(input.sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the feed");
        feed.run().expect("run the feed");
        let (pipeline, ()) = Pipeline::new("conformance-offline", compositor, |source, ctx| {
            let branch = ctx.branch().queue("composed", 4).to(recorder)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the offline pipeline");
        Self {
            pipeline,
            duration: None,
            terminals: vec![("screen", log)],
            looping: None,
            _feeds: vec![feed],
        }
    }

    fn build_file(shape: FileShape) -> Option<Self> {
        let path = try_test_video()?;
        let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
        let duration = source.duration().expect("the fixture says how long it is");
        let looping = source.looping_handle();
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("the fixture has a picture")
            .clone();
        let audio = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
            .cloned();
        let (packets, frames) = match shape.depth {
            Depth::Tight => (1, 1),
            Depth::Normal => (8, 4),
        };
        let mut terminals = Vec::new();
        let name = format!("conformance-{}", shape.label());
        let (pipeline, ()) = Pipeline::new(name, source, |source, ctx| {
            let picture = match shape.branches {
                Branches::PacketTee => {
                    let mut tee = ctx.tee("tee");
                    for (name, delay) in [("screen", 0), ("second-screen", 20)] {
                        let (recorder, log) = Recorder::new(name, video.time_base);
                        terminals.push((name, log));
                        let chain = ctx.branch().queue(format!("{name}-packets"), packets);
                        let chain = decoded(shape, chain, &video.parameters, name, frames)?;
                        let chain = filtered(shape, chain, name);
                        tee = tee.branch(
                            paced(shape, chain, name)
                                .to(recorder.slowed(Duration::from_millis(delay)))?,
                        );
                    }
                    tee.build()?
                }
                Branches::PictureTee => {
                    let chain = ctx.branch().queue("video-packets", packets);
                    let chain = decoded(shape, chain, &video.parameters, "video", frames)?;
                    let chain = filtered(shape, chain, "video");
                    let mut tee = ctx.tee("tee");
                    for name in ["screen", "second-screen"] {
                        let (recorder, log) = Recorder::new(name, video.time_base);
                        terminals.push((name, log));
                        let branch = ctx.branch().queue(format!("{name}-frames"), frames.min(2));
                        tee = tee.branch(paced(shape, branch, name).to(recorder)?);
                    }
                    chain.to_branch(tee.build()?)?
                }
                Branches::One => {
                    let (recorder, log) = Recorder::new("screen", video.time_base);
                    terminals.push(("screen", log));
                    let chain = ctx.branch().queue("video-packets", packets);
                    let chain = decoded(shape, chain, &video.parameters, "video", frames)?;
                    let chain = filtered(shape, chain, "video");
                    paced(shape, chain, "video").to(recorder)?
                }
            };
            ctx.attach(source, video.index, picture)?;

            if let Some(audio) = &audio
                && shape.sound != Sound::None
            {
                let (recorder, log) = Recorder::new("speakers", audio.time_base);
                terminals.push(("speakers", log));
                let sound = ctx
                    .branch()
                    .queue("audio-packets", packets)
                    .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
                    .queue("audio-frames", frames);
                let sound = match shape.pacing {
                    Pacing::Unpaced => sound,
                    Pacing::Pacer | Pacing::Synchronizer => sound.pipe(Pacer::new("audio-pacer")),
                };
                let sound = match shape.sound {
                    // As a mixer input is fed: stretched to the rate after its
                    // pacer, which holds sound until it has enough to stretch.
                    Sound::Stretched => sound.pipe(AudioTempo::new("audio-tempo")),
                    Sound::Resampled => sound.pipe(AudioResampler::new(
                        "audio-resampler",
                        AudioFormat::new(
                            ffmpeg::format::Sample::I16(ffmpeg::format::sample::Type::Packed),
                            44_100,
                            2,
                        ),
                    )),
                    Sound::None => unreachable!("no sound branch without sound"),
                };
                ctx.attach(source, audio.index, sound.to(recorder)?)?;
            }
            Ok(())
        })
        .expect("wire the file pipeline");
        Some(Self {
            pipeline,
            duration: Some(duration),
            terminals,
            looping: Some(looping),
            _feeds: Vec::new(),
        })
    }

    /// Writes into every terminal's record that call `index` began, or
    /// returned.
    fn mark(&self, index: usize, returned: bool) {
        for (_, log) in &self.terminals {
            log.lock().unwrap().push(Entry::Call { index, returned });
        }
    }

    /// Where everything stands, for a failure to show: each element's
    /// state, what it has taken, and how full its queue is, then what each
    /// terminal was last handed. A stall is found from this without a
    /// trace, which slows a run down enough to hide the race behind it.
    fn describe(&self) -> String {
        let stats = self.pipeline.stats();
        let mut out = format!("  paused: {}\n", stats.paused);
        for element in &stats.elements {
            out += &format!(
                "  {:<22} {:?} in={} eos={} idle={:?}{}\n",
                element.name,
                element.state,
                element.buffers_in,
                element.eos,
                element.idle_for,
                element
                    .queue
                    .as_ref()
                    .map(|queue| format!(" queue={}/{}", queue.len, queue.capacity))
                    .unwrap_or_default()
            );
        }
        for (name, log) in &self.terminals {
            out += &format!("  {name} ended {:?}\n", tail(&log.lock().unwrap()));
        }
        out
    }

    fn data_counts(&self) -> Vec<usize> {
        self.terminals
            .iter()
            .map(|(_, log)| {
                log.lock()
                    .unwrap()
                    .iter()
                    .filter(|entry| matches!(entry, Entry::Data { .. }))
                    .count()
            })
            .collect()
    }

    /// The furthest any picture terminal has been handed.
    fn furthest_picture(&self) -> Option<Duration> {
        self.terminals
            .iter()
            .filter(|(name, _)| *name != "speakers")
            .filter_map(|(_, log)| {
                log.lock()
                    .unwrap()
                    .iter()
                    .filter_map(|entry| match entry {
                        Entry::Data { pts } => *pts,
                        _ => None,
                    })
                    .max()
            })
            .max()
    }
}

/// One step of a sequence.
#[derive(Debug, Clone, Copy)]
enum Op {
    Pause,
    Resume,
    /// To this position, in this mode.
    Seek(Duration, SeekMode),
    /// Let it run, or stay paused, this long.
    Wait(Duration),
    /// To just short of the end, playing, and on until it finishes — or,
    /// looping, into the next lap.
    PlayToEnd,
    /// The picture this many pictures on, or back.
    Step(i64),
    /// Playing on at this rate, backwards at [`Pipeline::REVERSE_RATE`].
    Rate(f64),
    /// A file's looping turned on or off.
    Loop(bool),
}

/// A call the harness made on the pipeline, as the judge needs to know it.
#[derive(Debug, Clone, Copy)]
struct Call {
    name: &'static str,
    /// How many samples it may hand each terminal while it runs, or `None`
    /// for as many as playing hands it: a call made while playing, or one
    /// that plays.
    allowed: Option<usize>,
    /// Whether playback is paused once it has returned.
    ends_paused: bool,
    /// A frame step, which drops the sound it passes over.
    step: bool,
    /// A stop: the record means nothing after it.
    stop: bool,
}

impl Call {
    fn new(name: &'static str, allowed: Option<usize>, ends_paused: bool) -> Self {
        Self {
            name,
            allowed,
            ends_paused,
            step: false,
            stop: false,
        }
    }
}

/// A small, seedable generator — enough to spread sequences over the
/// orders that matter, and to replay one exactly.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// The next step of a sequence — never a turn backwards where `turns` is
/// false, a known bug breaking it there.
fn op(rng: &mut Rng, duration: Option<Duration>, turns: bool) -> Op {
    let Some(duration) = duration else {
        return match rng.below(6) {
            0 => Op::Pause,
            1 => Op::Resume,
            2 => Op::Seek(Duration::from_secs(1), SeekMode::Accurate),
            3 => Op::Step(1),
            4 => Op::Rate([2.0, Pipeline::REVERSE_RATE][rng.below(2) as usize]),
            _ => Op::Wait(Duration::from_millis(rng.below(300))),
        };
    };
    let mode = if rng.below(3) == 0 {
        SeekMode::Keyframe
    } else {
        SeekMode::Accurate
    };
    match rng.below(14) {
        0 | 1 => Op::Pause,
        2 | 3 => Op::Resume,
        4 => {
            // Anywhere in the file.
            let at = duration.mul_f64(rng.below(1_000) as f64 / 1_000.0);
            Op::Seek(at, mode)
        }
        5 => {
            // Near the end, where the source reaches its end a queue's depth
            // before the picture does.
            let back = Duration::from_millis(rng.below(1_500));
            Op::Seek(duration.saturating_sub(back), mode)
        }
        6 => {
            // At the start, or past the end.
            if rng.below(2) == 0 {
                Op::Seek(Duration::ZERO, mode)
            } else {
                Op::Seek(duration + Duration::from_millis(500), mode)
            }
        }
        7 => Op::PlayToEnd,
        10 | 11 => Op::Step(match rng.below(4) {
            0 => 1,
            1 => 4,
            2 => -1,
            _ => -3,
        }),
        // Not four times: the model counts only the waits as playing, and
        // what the other calls take is played too, four times as far.
        12 => {
            let rate = [0.5, 1.0, 2.0, -0.5, -1.0, -2.0][rng.below(6) as usize];
            Op::Rate(if turns { rate } else { rate.abs() })
        }
        13 => Op::Loop(rng.below(2) == 0),
        _ => Op::Wait(Duration::from_millis(rng.below(400))),
    }
}

/// Runs `call` on a thread of its own and waits at most [`OP_TIMEOUT`] for
/// it: a call that never returns is the failure this is looking for, and it
/// cannot be interrupted, only abandoned.
fn within<T: Send + 'static>(call: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        let _ = done.send(call());
    });
    finished.recv_timeout(OP_TIMEOUT).ok()
}

/// Everything the bus says while a sequence runs, collected off it as it
/// comes so that `Finished` can be waited on and errors are all seen.
struct BusLog {
    events: Arc<Mutex<Vec<BusEvent>>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl BusLog {
    fn start(pipeline: &Arc<Pipeline>) -> Self {
        let events = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let pipeline = Arc::clone(pipeline);
            let events = Arc::clone(&events);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    match pipeline.bus().recv_timeout(Duration::from_millis(20)) {
                        Ok(event) => events.lock().unwrap().push(event),
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                }
            })
        };
        Self {
            events,
            stop,
            thread: Some(thread),
        }
    }

    fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    fn finished_since(&self, from: usize) -> bool {
        self.events.lock().unwrap()[from..]
            .iter()
            .any(|event| matches!(event, BusEvent::Finished))
    }

    fn landings(&self) -> Vec<Duration> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                BusEvent::Seeked { landed, .. } => Some(*landed),
                _ => None,
            })
            .collect()
    }

    fn errors(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .map(|event| format!("{event:?}"))
            .collect()
    }
}

impl Drop for BusLog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// What the harness knows of where playback is, to decide which promises
/// apply after each step.
#[derive(Default)]
struct Model {
    paused: bool,
    /// Whether playback is far enough from the end that a resume must be
    /// seen to flow — cleared once it has played long enough to be near it.
    far_from_end: bool,
    /// How long it has played since the last seek.
    played: Duration,
    /// The seeks that succeeded, in order: what each terminal's new
    /// timelines are matched against. `None` for one whose target the
    /// pipeline works out — a step back's, a turn's, and the one a resume
    /// after a step makes to line everything up to the picture — and whether
    /// it plays backwards from there.
    seeks: Vec<(Option<Duration>, SeekMode, bool)>,
    /// Whether the picture has been stepped since the last seek, so the next
    /// resume seeks to it first.
    stepped: bool,
    /// Where the last step left the picture.
    picture: Option<Duration>,
    /// The rate it plays at.
    rate: f64,
    /// Whether the file loops, and whether it ever has in this sequence —
    /// which lets a gap at a lap's join through.
    looping: bool,
    looped: bool,
    /// Whether anything paces playback. Unpaced it plays as fast as it
    /// decodes, so the end is near as soon as it has played at all.
    paced: bool,
}

impl Model {
    fn backwards(&self) -> bool {
        self.rate < 0.0
    }

    /// Whether it plays on for ever the way it is going: forwards on a
    /// looping file. Backwards the start of the first lap still ends it.
    fn endless(&self) -> bool {
        self.looping && !self.backwards()
    }

    /// A seek succeeded, to `target` or where the pipeline works out.
    fn seeked(&mut self, target: Option<Duration>, mode: SeekMode) {
        self.seeks.push((target, mode, self.backwards()));
    }

    fn sought(&mut self, target: Duration, duration: Option<Duration>) {
        self.played = Duration::ZERO;
        // The end it plays towards: the start of the file, backwards.
        self.far_from_end = if self.backwards() {
            target > Duration::from_secs(3)
        } else {
            self.looping
                || duration.is_some_and(|duration| target + Duration::from_secs(3) < duration)
        };
    }

    fn waited(&mut self, wait: Duration) {
        if !self.paused {
            self.played += wait.mul_f64(self.rate.abs());
            if self.played > Duration::from_millis(1_500) && !self.endless() {
                self.far_from_end = false;
            }
        }
    }
}

/// What the harness has asked of the pipeline so far, and the rig whose
/// records it marks.
struct Calls<'a> {
    rig: &'a Rig,
    made: Vec<Call>,
}

impl Calls<'_> {
    /// Marks `call` as begun, and answers its index for [`Self::returned`].
    fn begin(&mut self, call: Call) -> usize {
        let index = self.made.len();
        self.made.push(call);
        self.rig.mark(index, false);
        index
    }

    /// Marks call `index` as returned, playback paused afterwards or not.
    fn returned(&mut self, index: usize, paused: bool) {
        self.made[index].ends_paused = paused;
        self.rig.mark(index, true);
    }
}

/// Runs one sequence against a fresh pipeline of `shape`, and says what it
/// broke, if anything — turning round only where no known bug breaks it.
fn run_sequence(shape: Shape, seed: u64, steps: usize) -> std::result::Result<(), String> {
    let turns = match shape {
        Shape::File(shape) => turns_round(shape),
        Shape::Live | Shape::Offline | Shape::OfflineMix => true,
    };
    run_sequence_turning(shape, seed, steps, turns)
}

/// The same, turning round where `turns` says — which a known bug's own
/// test sets to reproduce it.
fn run_sequence_turning(
    shape: Shape,
    seed: u64,
    steps: usize,
    turns: bool,
) -> std::result::Result<(), String> {
    // Here rather than only in `conform`, so a test replaying one sequence
    // is traced as well.
    trace_if_asked();
    let Some(rig) = Rig::build(shape) else {
        return Ok(());
    };
    // A decoder whose pictures come late leaves some out by design, and a
    // runner pinned to two cores is slow enough, playing backwards, for that
    // to happen. What these sequences check is that control loses nothing.
    rig.pipeline.state.decode_everything();
    let rig = Arc::new(rig);
    let mut rng = Rng(seed.max(1));
    let mut history: Vec<String> = Vec::new();
    let label = shape.label();
    let fail = |history: &[String], why: String| {
        Err(format!(
            "{label} seed {seed}: {why}\n  after: {}\n  replay: MEDIA_PP_CONTROL_SHAPE={label} MEDIA_PP_CONTROL_SEED={seed}",
            history.join(" → ")
        ))
    };

    let bus = BusLog::start(&rig.pipeline);
    let mut calls = Calls {
        rig: &rig,
        made: Vec::new(),
    };
    let mut model = Model {
        far_from_end: rig.duration.is_some(),
        rate: 1.0,
        paced: !matches!(
            shape,
            Shape::File(FileShape {
                pacing: Pacing::Unpaced,
                ..
            })
        ),
        ..Model::default()
    };
    model.sought(Duration::ZERO, rig.duration);
    if rig.duration.is_none() {
        model.far_from_end = true;
    }
    {
        let index = calls.begin(Call::new("run", None, false));
        let pipeline = Arc::clone(&rig.pipeline);
        match within(move || pipeline.run()) {
            Some(Ok(())) => {}
            Some(Err(error)) => return fail(&history, format!("run failed: {error}")),
            None => return fail(&history, "run did not return".into()),
        }
        calls.returned(index, false);
    }

    for _ in 0..steps {
        // Unpaced, whatever has played since the last call has run to
        // the end — unless the end goes round.
        if !model.paced && !model.paused && !model.endless() {
            model.far_from_end = false;
        }
        let step = op(&mut rng, rig.duration, turns);
        history.push(format!("{step:?}"));
        let before = rig.data_counts();
        // What a call made while paused may hand a terminal; one made
        // playing may hand it anything, until it returns.
        let while_paused = |samples: usize| model.paused.then_some(samples);
        match step {
            Op::Pause => {
                let index = calls.begin(Call::new("pause", while_paused(0), true));
                let pipeline = Arc::clone(&rig.pipeline);
                if within(move || pipeline.pause()).is_none() {
                    return fail(&history, "pause did not return".into());
                }
                model.paused = true;
                calls.returned(index, true);
            }
            Op::Resume => {
                let index = calls.begin(Call::new("resume", None, false));
                let pipeline = Arc::clone(&rig.pipeline);
                if within(move || pipeline.resume()).is_none() {
                    return fail(&history, "resume did not return".into());
                }
                model.paused = false;
                calls.returned(index, false);
                if std::mem::take(&mut model.stepped)
                    && let Some(picture) = model.picture
                {
                    // Lined up to the picture before playing on.
                    model.seeked(None, SeekMode::Accurate);
                    model.sought(picture, rig.duration);
                }
            }
            Op::Rate(rate) => {
                let turned = (rate < 0.0) != model.backwards();
                let allowed = while_paused(usize::from(turned));
                let index = calls.begin(Call::new("set_rate", allowed, model.paused));
                let pipeline = Arc::clone(&rig.pipeline);
                match within(move || pipeline.set_rate(rate)) {
                    None => return fail(&history, "set_rate did not return".into()),
                    Some(Ok(())) => {
                        model.rate = rate;
                        if turned {
                            // Repositioned to the picture shown, which the
                            // model does not know: whether that is far from
                            // the end it now plays towards neither.
                            model.seeked(None, SeekMode::Accurate);
                            model.stepped = false;
                            model.played = Duration::ZERO;
                            model.far_from_end = model.endless();
                        }
                    }
                    Some(Err(crate::Error::SeekError(_))) if rig.duration.is_none() => {}
                    Some(Err(error)) => {
                        return fail(
                            &history,
                            format!(
                                "set_rate failed: {error}
{}",
                                rig.describe()
                            ),
                        );
                    }
                }
                calls.returned(index, model.paused);
            }
            Op::Step(frames) => {
                // A step against the way it plays is a seek, and prerolls
                // one picture; with it, as many as it steps.
                let against = (frames < 0) != model.backwards();
                let allowed = while_paused(if against {
                    1
                } else {
                    frames.unsigned_abs() as usize
                });
                let mut call = Call::new("step", allowed, true);
                call.step = true;
                let index = calls.begin(call);
                let pipeline = Arc::clone(&rig.pipeline);
                match within(move || pipeline.step(frames)) {
                    None => return fail(&history, "step did not return".into()),
                    Some(Ok(at)) => {
                        model.paused = true;
                        model.stepped = true;
                        model.picture = Some(at);
                        if against {
                            model.seeked(None, SeekMode::Accurate);
                            model.sought(at, rig.duration);
                        }
                    }
                    // Nothing shown yet to step from, and nothing changed.
                    Some(Err(crate::Error::PipelineError(PipelineError::NoPicture))) => {}
                    Some(Err(error)) if rig.duration.is_none() => {
                        if !matches!(error, crate::Error::SeekError(_)) {
                            return fail(&history, format!("live step refused oddly: {error}"));
                        }
                    }
                    Some(Err(error)) => {
                        return fail(
                            &history,
                            format!(
                                "step failed: {error}
{}",
                                rig.describe()
                            ),
                        );
                    }
                }
                calls.returned(index, model.paused);
            }
            Op::Seek(target, mode) => {
                let index = calls.begin(Call::new("seek", while_paused(1), model.paused));
                let pipeline = Arc::clone(&rig.pipeline);
                match within(move || pipeline.seek(target, mode)) {
                    None => return fail(&history, "seek did not return".into()),
                    Some(Ok(())) => {
                        model.seeked(Some(target), mode);
                        model.stepped = false;
                        model.sought(target, rig.duration);
                    }
                    Some(Err(error)) if rig.duration.is_none() => {
                        // A live source refuses, and nothing else changes.
                        if !matches!(error, crate::Error::SeekError(_)) {
                            return fail(&history, format!("live seek refused oddly: {error}"));
                        }
                    }
                    Some(Err(error)) => {
                        return fail(
                            &history,
                            format!(
                                "seek failed: {error}
{}",
                                rig.describe()
                            ),
                        );
                    }
                }
                calls.returned(index, model.paused);
            }
            Op::Wait(wait) => {
                thread::sleep(wait);
                model.waited(wait);
            }
            Op::Loop(on) => {
                let Some(looping) = &rig.looping else {
                    continue;
                };
                looping.set_looping(on);
                model.looping = on;
                model.looped |= on;
                // Turned off, the lap it is on ends it; where that is, the
                // model does not know.
                if !on {
                    model.far_from_end = false;
                }
            }
            Op::PlayToEnd => {
                let Some(duration) = rig.duration else {
                    continue;
                };
                // Just short of the start, backwards.
                let target = if model.backwards() {
                    Duration::from_millis(400)
                } else {
                    duration.saturating_sub(Duration::from_millis(400))
                };
                let from = bus.len();
                let index = calls.begin(Call::new("seek", while_paused(1), model.paused));
                let pipeline = Arc::clone(&rig.pipeline);
                match within(move || pipeline.seek(target, SeekMode::Accurate)) {
                    None => return fail(&history, "seek to the end did not return".into()),
                    Some(Err(error)) => {
                        return fail(
                            &history,
                            format!(
                                "seek to the end failed: {error}
{}",
                                rig.describe()
                            ),
                        );
                    }
                    Some(Ok(())) => {
                        model.seeked(Some(target), SeekMode::Accurate);
                        model.stepped = false;
                    }
                }
                calls.returned(index, model.paused);
                let index = calls.begin(Call::new("resume", None, false));
                let pipeline = Arc::clone(&rig.pipeline);
                if within(move || pipeline.resume()).is_none() {
                    return fail(&history, "resume did not return".into());
                }
                calls.returned(index, false);
                model.paused = false;
                model.sought(target, rig.duration);
                // Backwards, a looping file ends at the start of its first
                // lap, from whichever lap the source was on: as long as the
                // furthest picture shown, played at the rate — for however
                // many laps it went on while the calls before ran.
                let back_over_laps = (model.looped && model.backwards())
                    .then(|| rig.furthest_picture())
                    .flatten()
                    .map_or(Duration::ZERO, |furthest| {
                        furthest.div_f64(model.rate.abs().max(0.1))
                    });
                let deadline = Instant::now() + OP_TIMEOUT + back_over_laps;
                if model.endless() {
                    // Into the next lap. Not "and never `Finished`": one an
                    // earlier seek past the end posted can reach this log
                    // after `from` was read — the log is taken off the bus
                    // on a thread of its own — and one that ended it here
                    // leaves the picture short of the next lap anyway.
                    let next_lap = duration + Duration::from_millis(100);
                    while rig.furthest_picture().is_none_or(|at| at < next_lap) {
                        if Instant::now() > deadline {
                            return fail(
                                &history,
                                format!(
                                    "a looping file never went on past its end\n{}",
                                    rig.describe()
                                ),
                            );
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                } else {
                    while !bus.finished_since(from) {
                        if Instant::now() > deadline {
                            return fail(&history, "played to the end, never Finished".into());
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    model.far_from_end = false;
                }
            }
        }
        // Resumed away from the end: every terminal must be handed something
        // new, or playback has stalled — every one that plays, backwards only
        // the picture.
        let resumed = matches!(step, Op::Resume) || (matches!(step, Op::Seek(..)) && !model.paused);
        if resumed && model.far_from_end {
            let deadline = Instant::now() + FLOW_TIMEOUT;
            let playing: Vec<bool> = rig
                .terminals
                .iter()
                .map(|(name, _)| !model.backwards() || *name != "speakers")
                .collect();
            loop {
                let now = rig.data_counts();
                if now
                    .iter()
                    .zip(&before)
                    .zip(&playing)
                    .all(|((now, before), playing)| !playing || now > before)
                {
                    break;
                }
                if Instant::now() > deadline {
                    return fail(
                        &history,
                        format!(
                            "resumed away from the end, but nothing flowed: {before:?} → {now:?}\n{}",
                            rig.describe()
                        ),
                    );
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    // Ended each way a caller ends one: finished in order — which has to
    // leave every terminal with an `Eos` after its last data — abandoned, or
    // abandoned while a call is still under way, which both have to return.
    match rng.below(3) {
        0 => {
            history.push("Finish".into());
            let index = calls.begin(Call::new("finish", None, false));
            let pipeline = Arc::clone(&rig.pipeline);
            if within(move || pipeline.finish()).is_none() {
                return fail(&history, "finish did not return".into());
            }
            calls.returned(index, false);
            for (name, log) in &rig.terminals {
                let ended = log
                    .lock()
                    .unwrap()
                    .iter()
                    .rev()
                    .find(|entry| matches!(entry, Entry::Data { .. } | Entry::Eos))
                    .is_none_or(|entry| matches!(entry, Entry::Eos));
                if !ended {
                    return fail(
                        &history,
                        format!(
                            "{name}: finished without an Eos after its data; it ended {:?}",
                            tail(&log.lock().unwrap())
                        ),
                    );
                }
            }
        }
        1 => {
            history.push("Stop".into());
            let mut call = Call::new("stop", None, false);
            call.stop = true;
            calls.begin(call);
            let pipeline = Arc::clone(&rig.pipeline);
            if within(move || pipeline.stop()).is_none() {
                return fail(&history, "stop did not return".into());
            }
        }
        _ => {
            let under_way = match rig.duration {
                Some(duration) => match rng.below(4) {
                    0 => Op::Seek(duration.mul_f64(0.5), SeekMode::Accurate),
                    1 => Op::Step(1),
                    2 if turns => Op::Rate(Pipeline::REVERSE_RATE),
                    _ => Op::Resume,
                },
                None => Op::Resume,
            };
            history.push(format!("Stop during {under_way:?}"));
            // The timeline it starts, if it gets that far before the stop:
            // recorded now, since its answer comes too late to say. One it
            // never starts is simply never matched.
            match under_way {
                Op::Seek(target, mode) => model.seeked(Some(target), mode),
                Op::Step(frames) if (frames < 0) != model.backwards() => {
                    model.seeked(None, SeekMode::Accurate);
                }
                Op::Rate(rate) if (rate < 0.0) != model.backwards() => {
                    model.rate = rate;
                    model.seeked(None, SeekMode::Accurate);
                }
                Op::Resume if model.stepped && model.picture.is_some() => {
                    model.seeked(None, SeekMode::Accurate);
                }
                _ => {}
            }
            calls.begin(Call::new("under way", None, false));
            let pipeline = Arc::clone(&rig.pipeline);
            let (done, returned) = mpsc::channel();
            thread::spawn(move || {
                let outcome = match under_way {
                    Op::Seek(target, mode) => pipeline.seek(target, mode).map(|_| ()),
                    Op::Step(frames) => pipeline.step(frames).map(|_| ()),
                    Op::Rate(rate) => pipeline.set_rate(rate),
                    _ => {
                        pipeline.resume();
                        Ok(())
                    }
                };
                let _ = done.send(outcome.map_err(|error| error.to_string()));
            });
            thread::sleep(Duration::from_millis(rng.below(40)));
            let mut call = Call::new("stop", None, false);
            call.stop = true;
            calls.begin(call);
            let pipeline = Arc::clone(&rig.pipeline);
            if within(move || pipeline.stop()).is_none() {
                return fail(
                    &history,
                    format!("stop did not return during {under_way:?}"),
                );
            }
            if returned.recv_timeout(OP_TIMEOUT).is_err() {
                return fail(
                    &history,
                    format!(
                        "{under_way:?} did not return once stopped\n{}",
                        rig.describe()
                    ),
                );
            }
        }
    }
    let errors = bus.errors();
    if !errors.is_empty() {
        return fail(&history, format!("errors on the bus: {errors:?}"));
    }
    let landings = bus.landings();
    let lap = rig.duration.filter(|_| model.looped);
    for (name, log) in &rig.terminals {
        let log = log.lock().unwrap();
        // Whatever the calls did: past a stop as much as before it, which is
        // where the promises about data end.
        if let Some(what) = log.iter().find_map(|entry| match entry {
            Entry::Outside(what) => Some(what),
            _ => None,
        }) {
            let handed = sketch(&log, &calls.made);
            return fail(
                &history,
                format!("{name}: {what}\n  it was handed: {handed}"),
            );
        }
        if let Err(why) = judge(
            &log,
            &calls.made,
            &model.seeks,
            &landings,
            Judged {
                duration: rig.duration,
                lap,
                lossless: shape.lossless(),
                shows_pictures: *name != "speakers",
            },
        ) {
            let handed = sketch(&log, &calls.made);
            return fail(
                &history,
                format!("{name}: {why}\n  it was handed: {handed}"),
            );
        }
    }
    Ok(())
}

/// Everything a terminal was handed, one short word each, for a failure
/// the data promises found to show: a sample as the millisecond it starts
/// at, a new timeline by where it starts, a call by its name — `[seek` as it
/// began, `seek]` as it returned.
fn sketch(log: &[Entry], calls: &[Call]) -> String {
    let words: Vec<String> = log
        .iter()
        .map(|entry| match entry {
            Entry::Data { pts: Some(pts) } => pts.as_millis().to_string(),
            Entry::Data { pts: None } => "?".into(),
            Entry::Eos => "Eos".into(),
            Entry::Timeline(target) => format!("Timeline({})", target.as_millis()),
            Entry::Segment { id, flushed } => {
                format!("Segment({id}{})", if *flushed { "f" } else { "" })
            }
            Entry::Outside(what) => format!("<{what}>"),
            Entry::Call {
                index,
                returned: false,
            } => format!("[{}", calls[*index].name),
            Entry::Call {
                index,
                returned: true,
            } => format!("{}]", calls[*index].name),
        })
        .collect();
    words.join(" ")
}

/// The last few things a terminal was handed, for a failure to show.
fn tail(log: &[Entry]) -> &[Entry] {
    &log[log.len().saturating_sub(8)..]
}

/// What a terminal is judged against, besides its record.
struct Judged {
    duration: Option<Duration>,
    /// How long a lap is, where the file looped in this sequence.
    lap: Option<Duration>,
    lossless: bool,
    shows_pictures: bool,
}

/// Checks one terminal's record against the promises about data.
fn judge(
    log: &[Entry],
    calls: &[Call],
    seeks: &[(Option<Duration>, SeekMode, bool)],
    landings: &[Duration],
    judged: Judged,
) -> std::result::Result<(), String> {
    let Judged {
        duration,
        lap,
        lossless,
        shows_pictures,
    } = judged;
    // The widest step allowed between two samples in a row: a few of the
    // stream's own, so a picture or a packet of sound gone missing shows,
    // and one a timestamp rounded differently does not.
    let widest = typical_spacing(log)
        .map(|spacing| (spacing * 5 / 2).max(spacing + Duration::from_millis(20)));
    // The call under way, and how many samples it may still hand this
    // terminal — `None` for as many as playing does.
    let mut under_way: Option<(usize, Option<usize>)> = None;
    // Whether playback is paused, between calls.
    let mut paused = false;
    let mut seek = 0usize;
    let mut floor: Option<Duration> = None;
    // Backwards the seek's target is the most a sample may start at, and
    // they go down from it.
    let mut ceiling: Option<Duration> = None;
    let mut last: Option<Duration> = None;
    // Whether what comes next may start further on than the stream's own
    // spacing allows: a step drops the sound, and unless a seek lines it
    // up again — a `finish` does not — it goes on from further on.
    let mut skipped = false;
    // Whether the last new timeline has yet to say which lap it is on. A
    // looping file's seek is to a position in the file, on the lap playback
    // is on — so its samples start a whole number of laps past the position
    // the timeline was announced at, and the first of them says how many.
    let mut unlapped = false;
    // An accurate seek's target past the end of the file — see where it
    // is set.
    let mut past_end: Option<Duration> = None;
    for (at, entry) in log.iter().enumerate() {
        match entry {
            Entry::Call {
                index,
                returned: false,
            } => {
                let call = calls[*index];
                if call.stop {
                    break;
                }
                under_way = Some((*index, call.allowed));
            }
            Entry::Call {
                index,
                returned: true,
            } => {
                under_way = None;
                paused = calls[*index].ends_paused;
                // Once the step is done: what it was handed while it was
                // still pausing was played in the ordinary way.
                skipped |= calls[*index].step && !shows_pictures;
            }
            Entry::Timeline(target) => {
                last = None;
                let (asked, mode, backwards) =
                    seeks
                        .get(seek)
                        .copied()
                        .unwrap_or((Some(*target), SeekMode::Accurate, false));
                let asked = asked.unwrap_or(*target);
                if asked != *target {
                    return Err(format!(
                        "entry {at}: seek {seek} was for {asked:?}, this terminal was told {target:?}"
                    ));
                }
                (floor, ceiling) = if backwards {
                    (None, Some(*target))
                } else {
                    let floor = match mode {
                        SeekMode::Accurate => duration.map_or(*target, |duration| {
                            (*target).min(duration.saturating_sub(LAST_SAMPLE))
                        }),
                        SeekMode::Keyframe => landings.get(seek).copied().unwrap_or(Duration::ZERO),
                    };
                    (Some(floor), None)
                };
                // Past the end of the file, a looping one's source reads a
                // target as a place on the timeline once it has looped — a
                // step back on a later lap goes there — and as the end of
                // the file before; the first sample says which.
                past_end = (!backwards
                    && mode == SeekMode::Accurate
                    && duration.is_some_and(|duration| *target > duration))
                .then_some(*target);
                seek += 1;
                unlapped = lap.is_some();
            }
            Entry::Segment { .. } => {}
            Entry::Outside(what) => return Err(format!("entry {at}: {what}")),
            Entry::Eos => {}
            Entry::Data { pts } => {
                match &mut under_way {
                    Some((index, Some(left))) => {
                        if *left == 0 && !ends_the_stream(&log[at..]) {
                            return Err(format!(
                                "entry {at}: data while paused, during {}, {pts:?}",
                                calls[*index].name
                            ));
                        }
                        *left = left.saturating_sub(1);
                    }
                    Some((_, None)) => {}
                    None => {
                        if paused && !ends_the_stream(&log[at..]) {
                            return Err(format!("entry {at}: data while paused, {pts:?}"));
                        }
                    }
                }
                if ceiling.is_some() && !shows_pictures {
                    return Err(format!("entry {at}: sound played backwards, {pts:?}"));
                }
                let Some(pts) = *pts else { continue };
                if let Some(target) = past_end.take()
                    && pts + LANDING_TOLERANCE >= target
                {
                    // Read as a place on the timeline: shown where it is.
                    floor = Some(target);
                    unlapped = false;
                }
                if std::mem::take(&mut unlapped)
                    && let (Some(lap), Some(reference)) = (lap, floor.or(ceiling))
                    && pts > reference
                    && !lap.is_zero()
                {
                    let laps = ((pts - reference).as_nanos() + lap.as_nanos() / 2) / lap.as_nanos();
                    let shift = Duration::from_nanos((laps * lap.as_nanos()) as u64);
                    floor = floor.map(|floor| floor + shift);
                    ceiling = ceiling.map(|ceiling| ceiling + shift);
                }
                let gap_allowed = |last: Duration| crosses_lap(last, pts, lap);
                if let Some(ceiling) = ceiling {
                    if pts > ceiling + LANDING_TOLERANCE {
                        return Err(format!(
                            "entry {at}: {pts:?} after a seek backwards to {ceiling:?} — from after it"
                        ));
                    }
                    if let Some(last) = last
                        && pts > last
                    {
                        return Err(format!(
                            "entry {at}: {pts:?} after {last:?}, backwards — going forwards"
                        ));
                    }
                    if let (true, Some(last), Some(widest)) = (lossless, last, widest)
                        && last - pts > widest
                        && !gap_allowed(last)
                    {
                        return Err(format!(
                            "entry {at}: {pts:?} after {last:?}, backwards — {:?} missing between them",
                            last - pts
                        ));
                    }
                    last = Some(pts);
                    continue;
                }
                if let Some(floor) = floor
                    && pts + LANDING_TOLERANCE < floor
                {
                    return Err(format!(
                        "entry {at}: {pts:?} after a seek to {floor:?} — from before it"
                    ));
                }
                if let Some(last) = last
                    && pts < last
                {
                    return Err(format!(
                        "entry {at}: {pts:?} after {last:?} — going backwards"
                    ));
                }
                if let (true, false, Some(last), Some(widest)) =
                    (lossless, std::mem::take(&mut skipped), last, widest)
                    && pts - last > widest
                    && !gap_allowed(last)
                {
                    return Err(format!(
                        "entry {at}: {pts:?} after {last:?} — {:?} missing between them",
                        pts - last
                    ));
                }
                last = Some(pts);
            }
        }
    }
    Ok(())
}

/// Whether two samples in a row lie either side of a lap's join — within
/// [`LAP_JOIN`] of it — where what is missing between them is the join's.
fn crosses_lap(a: Duration, b: Duration, lap: Option<Duration>) -> bool {
    let Some(lap) = lap.filter(|lap| !lap.is_zero()) else {
        return false;
    };
    let (low, high) = (a.min(b).saturating_sub(LAP_JOIN), a.max(b) + LAP_JOIN);
    let first = low.as_nanos() / lap.as_nanos() + 1;
    first * lap.as_nanos() <= high.as_nanos()
}

/// Whether `rest` is samples and then the end of the stream, nothing
/// between them but calls: what a `Pacer` kept goes on with the `Eos`
/// behind it, preroll or not, since nothing would come to hand it on after
/// — see `Pacer::consume`. A step it had kept a picture for then shows the
/// last of the file rather than the one picture it asked for.
fn ends_the_stream(rest: &[Entry]) -> bool {
    rest.iter()
        .find(|entry| {
            !matches!(
                entry,
                Entry::Data { .. } | Entry::Call { .. } | Entry::Segment { .. }
            )
        })
        .is_some_and(|entry| matches!(entry, Entry::Eos))
}

/// How far apart a terminal's samples usually are: the median step
/// between two in a row, either way, not counting across a new timeline.
/// `None` for a record too short to say.
fn typical_spacing(log: &[Entry]) -> Option<Duration> {
    let mut steps = Vec::new();
    let mut last: Option<Duration> = None;
    for entry in log {
        match entry {
            Entry::Timeline(_) => last = None,
            Entry::Data { pts: Some(pts) } => {
                if let Some(last) = last
                    && *pts != last
                {
                    steps.push(pts.abs_diff(last));
                }
                last = Some(*pts);
            }
            _ => {}
        }
    }
    if steps.len() < 5 {
        return None;
    }
    steps.sort();
    Some(steps[steps.len() / 2])
}

/// Writes the crate's log where `MEDIA_PP_CONTROL_TRACE` says, once for
/// the whole test binary — the logger can only be installed once.
fn trace_if_asked() {
    static GUARD: std::sync::OnceLock<Option<crate::log::LogGuard>> = std::sync::OnceLock::new();
    GUARD.get_or_init(|| {
        let directory = std::env::var("MEDIA_PP_CONTROL_TRACE").ok()?;
        crate::log::init("conformance", directory, crate::log::Level::Trace, 7).ok()
    });
}

/// The sequences to run for the shape numbered `key`: fixed ones by
/// default, so an ordinary run is the same every time; more, or
/// clock-seeded ones, when asked.
fn seeds(key: u64) -> Vec<u64> {
    if let Ok(seed) = std::env::var("MEDIA_PP_CONTROL_SEED") {
        return vec![seed.parse().expect("MEDIA_PP_CONTROL_SEED is a number")];
    }
    let iterations: u64 = std::env::var("MEDIA_PP_CONTROL_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let base = if std::env::var_os("MEDIA_PP_CONTROL_RANDOM").is_some() {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |since| since.as_nanos() as u64)
            .wrapping_add(key)
    } else {
        (key + 1) * 1_000
    };
    (0..iterations)
        .map(|index| base.wrapping_add(index))
        .collect()
}

/// Runs `shape`'s sequences, unless `MEDIA_PP_CONTROL_SHAPE` names another.
fn conform(shape: Shape, key: u64) {
    if let Ok(only) = std::env::var("MEDIA_PP_CONTROL_SHAPE")
        && only != shape.label()
    {
        return;
    }
    trace_if_asked();
    let steps = std::env::var("MEDIA_PP_CONTROL_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12);
    for seed in seeds(key) {
        if let Err(failure) = run_sequence(shape, seed, steps) {
            panic!("{failure}");
        }
    }
}

/// How many tests the matrix is spread over, so the harness runs its
/// shapes side by side.
const SHARDS: usize = 6;

fn conform_matrix(shard: usize) {
    for (key, shape) in matrix().into_iter().enumerate() {
        if key % SHARDS == shard {
            conform(Shape::File(shape), key as u64);
        }
    }
}

/// The matrix meets every pair of choices, keeps the shapes these tests
/// had by hand, and names every shape differently — a replay finds its
/// shape by name.
#[test]
fn the_matrix_meets_every_pair_of_choices() {
    let matrix = matrix();
    for known in &KNOWN_BROKEN {
        eprintln!(
            "never backwards: {:?} — see {}",
            known.backwards, known.test
        );
    }
    eprintln!(
        "{} shapes: {}",
        matrix.len(),
        matrix
            .iter()
            .map(|shape| shape.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let met: HashSet<_> = matrix.iter().flat_map(|shape| shape.pairs()).collect();
    let every: HashSet<_> = every_shape()
        .iter()
        .flat_map(|shape| shape.pairs())
        .collect();
    assert_eq!(met, every, "a pair of choices no shape makes");
    for named in NAMED {
        assert!(matrix.contains(&named), "{} left out", named.label());
    }
    let labels: HashSet<_> = matrix.iter().map(|shape| shape.label()).collect();
    assert_eq!(labels.len(), matrix.len());
}

/// On a looping file, an accurate seek made on a lap after the first shows
/// the picture at its target, on that lap — not the keyframe before it.
///
/// Found by the matrix, the first time it looped. A caller's seek is a
/// position in the file, and the source keeps to the lap playback is on, so
/// the pictures come stamped a lap further on; the decoders' preroll gate
/// held the target as the position in the file, found the first picture
/// past it, and showed it. The segment the seek begins now says where that
/// position is on the timeline — see `crate::stream` — and the gate
/// compares like with like.
#[test]
fn an_accurate_seek_on_a_later_lap_shows_its_target() {
    let shape = FileShape::new(Branches::One, Pacing::Pacer, Sound::None, Depth::Normal);
    let Some(rig) = Rig::build_file(shape) else {
        return;
    };
    let duration = rig.duration.expect("a file says how long it is");
    let looping = rig.looping.clone().expect("a file loops");
    looping.set_looping(true);
    rig.pipeline.run().expect("run");
    rig.pipeline.pause();
    // Past the end is on the next lap, at once.
    rig.pipeline
        .seek(duration + Duration::from_millis(500), SeekMode::Accurate)
        .expect("seek onto the second lap");
    let lap = looping.lap_offset();
    assert!(
        !lap.is_zero(),
        "the seek past the end moved to the next lap"
    );
    let target = duration.mul_f64(0.9);
    rig.pipeline
        .seek(target, SeekMode::Accurate)
        .expect("seek on the second lap");
    let (_, log) = &rig.terminals[0];
    let shown = log
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find_map(|entry| match entry {
            Entry::Data { pts } => *pts,
            _ => None,
        })
        .expect("the seek showed a picture");
    rig.pipeline.stop();
    assert!(
        shown + LANDING_TOLERANCE >= target + lap,
        "sought to {target:?} on the lap starting at {lap:?}, shown {shown:?}"
    );
}

/// A paused seek to the last of a file, with one-deep queues and its sound
/// played, prerolls every terminal — the sound's as much as the picture's.
///
/// Found by the matrix. The preroll reads to the end of the file, and the
/// source handed every pad its `Eos` in turn with a push that waits for
/// room — first the picture's, whose branch had its sample and took
/// nothing more, so the one-deep queue in front of it stayed full. The
/// sound's `Eos`, which answers a seek past its last sample with the sample
/// before it, waited behind that push on the source's one thread, and the
/// seek timed out. Now each pad is handed its end as it can take one — see
/// `FileDemuxer::end_every_pad`.
///
/// A race on where the fixture's last packets fall, so it replays the
/// sequence that found it rather than build one by hand: about one run in
/// three failed, and a hand-built one near the end was not seen to.
#[test]
fn a_paused_seek_to_the_end_prerolls_the_sound_through_one_deep_queues() {
    // The sequence the matrix found it with: paused, played to the end, a
    // picture back, and two accurate seeks near the end.
    let shape = FileShape::new(Branches::One, Pacing::Pacer, Sound::Resampled, Depth::Tight);
    if let Err(failure) = run_sequence(Shape::File(shape), 1_790_490_323_966_967_315, 12) {
        panic!("{failure}");
    }
}

/// Packets fanned out to two branches through one-deep queues, played
/// backwards, step on together — the faster branch's as much as the slower
/// one's.
///
/// Found by the matrix. Backwards a decoder hands a stretch's pictures on
/// once the stretch is whole, which takes the rest of it read. The slower
/// branch, which had its step's picture and took nothing more, filled its
/// queues, and the source waited inside its push to that branch through the
/// `Tee` — where nothing let it go: the `Tee` keeps what comes for a branch
/// that has its sample, but only what comes after it has. The faster
/// branch, a stretch short, showed nothing, and the step timed out. The
/// source now holds a packet read backwards until its pad can take it, as
/// forwards it parks one, and the `Tee` answers ready once the branch has
/// its sample.
#[test]
fn packets_fanned_out_backwards_step_on_in_every_branch() {
    // The sequence the matrix found it with: backwards, played to the start,
    // and a step on from a keyframe seek.
    let shape = FileShape {
        filter: Filter::Scaled,
        ..FileShape::new(
            Branches::PacketTee,
            Pacing::Unpaced,
            Sound::None,
            Depth::Tight,
        )
    };
    if let Err(failure) =
        run_sequence_turning(Shape::File(shape), 1_790_491_301_682_327_112, 12, true)
    {
        panic!("{failure}");
    }
}

/// A file played backwards finishes. `Finish` reads the stretch under way
/// to its end, and a packet read backwards that its pad cannot take yet is
/// held until it can — but not there: the pipeline's interrupt is out for
/// as long as the request is, no queue takes anything meanwhile, and the
/// source waited for good on a queue that would never have room.
#[test]
fn a_file_played_backwards_finishes() {
    // The sequence the matrix found it with: backwards, twice as fast, a
    // keyframe seek, a step, and the finish.
    let shape = FileShape::new(Branches::One, Pacing::Pacer, Sound::None, Depth::Tight);
    if let Err(failure) = run_sequence(Shape::File(shape), 1_790_501_120_723_546_607, 12) {
        panic!("{failure}");
    }
}

#[test]
fn the_matrix_keeps_its_control_promises_0() {
    conform_matrix(0);
}

#[test]
fn the_matrix_keeps_its_control_promises_1() {
    conform_matrix(1);
}

#[test]
fn the_matrix_keeps_its_control_promises_2() {
    conform_matrix(2);
}

#[test]
fn the_matrix_keeps_its_control_promises_3() {
    conform_matrix(3);
}

#[test]
fn the_matrix_keeps_its_control_promises_4() {
    conform_matrix(4);
}

#[test]
fn the_matrix_keeps_its_control_promises_5() {
    conform_matrix(5);
}

#[test]
fn a_live_source_keeps_its_control_promises() {
    conform(Shape::Live, 100);
}

#[test]
fn an_offline_render_keeps_its_control_promises() {
    conform(Shape::Offline, 101);
}

#[test]
fn an_offline_mix_keeps_its_control_promises() {
    conform(Shape::OfflineMix, 102);
}
