//! Control conformance: random sequences of pause, resume, seek, frame step,
//! rate — backwards too — and stop,
//! run against the shapes of pipeline this crate is used in, and judged
//! against what every terminal was handed.
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
//!   [`OP_TIMEOUT`], fails the sequence with its history;
//! - paused means nothing new reaches a terminal, except the one picture a
//!   seek's preroll asks for and the pictures a step does;
//! - after a seek, what a terminal is handed is from the new position on,
//!   and goes forward — nothing from before the seek arrives after it —
//!   with nothing missing in between: no step between two samples much
//!   wider than the stream's own spacing, where nothing drops by design;
//! - playing backwards, the same going down from the new position, and no
//!   sound at all;
//! - resumed away from the end, it flows again;
//! - played to the end, the pipeline says [`BusEvent::Finished`];
//! - and nothing reports an error.
//!
//! The terminal records every buffer and control message it is handed, in
//! order, and the promises about data are judged from that record once the
//! sequence is over. Nothing here times a picture against a clock, so a slow
//! machine makes a run slower rather than wrong — which matters, because a
//! slow machine is what these are for.
//!
//! # Running them harder
//!
//! By default each shape runs a few fixed sequences, so an ordinary test run
//! is deterministic. `MEDIA_PP_CONTROL_ITERS` runs more,
//! `MEDIA_PP_CONTROL_RANDOM=1` seeds them from the clock instead, and
//! `MEDIA_PP_CONTROL_SEED` replays one seed a failure printed, and
//! `MEDIA_PP_CONTROL_TRACE=<directory>` writes this crate's log there at
//! `Trace`, every control message at every element, for reading what the
//! replay did. The races
//! these look for show when threads are short of cores, so CI runs them
//! pinned to two: `taskset -c 0,1` on Linux, a two-core affinity mask on
//! Windows. That is how the seek-at-the-end deadlock of 208af56 was
//! reproduced, where a free machine never showed it.

use super::*;

use crate::buffer::time_base;
use crate::control::ControlMsg;

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

/// What one terminal was handed, in the order it was handed it.
#[derive(Debug, Clone)]
enum Entry {
    Data { pts: Option<Duration> },
    Eos,
    Control(ControlMsg),
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

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        self.log.lock().unwrap().push(Entry::Control(msg.clone()));
        Ok(())
    }
}

/// The shapes of pipeline the sequences run against.
#[derive(Debug, Clone, Copy)]
enum Shape {
    /// A file's picture, decoded, queued and paced — a player's video half.
    Video,
    /// Picture and sound off one demuxer, each paced — a whole player,
    /// the sound stretched to the rate after its pacer as a mixer input is.
    AudioVideo,
    /// One decoded picture fanned out to two paced branches.
    Tee,
    /// The picture with every queue one deep, so backpressure is felt at
    /// every step — where a thread blocked handing data on meets control.
    Tight,
    /// A live source, which cannot be sought and never ends.
    Live,
    /// The picture timed by a `VideoSynchronizer` rather than a `Pacer` —
    /// how a player shows it, dropping what is late.
    Synchronized,
    /// The file's packets fanned out before they are decoded, to two
    /// branches of their own, one slower to take a picture than the other
    /// — so one prerolls while the other is still catching up.
    TeeOfPackets,
    /// An offline compositor fed by a live source through a pipeline of its
    /// own — a render waiting on its input, holding that pipeline back a
    /// frame ahead of what it has drawn, while its own is paused, resumed
    /// and stopped. Not sought: it is not seekable.
    Offline,
}

impl Shape {
    /// Whether every sample a terminal is handed after a seek must follow
    /// the one before it. Not for a `VideoSynchronizer`, which drops a
    /// picture that is late by design, nor a live source, which skips the
    /// ticks a starved thread missed.
    fn lossless(self) -> bool {
        !matches!(self, Shape::Synchronized | Shape::Live)
    }
}

/// A built pipeline and what its terminals recorded.
struct Rig {
    pipeline: Arc<Pipeline>,
    duration: Option<Duration>,
    terminals: Vec<(&'static str, Arc<Mutex<Vec<Entry>>>)>,
    /// Pipelines feeding the one under test, kept running as long as it is.
    _feeds: Vec<Arc<Pipeline>>,
}

impl Rig {
    fn build(shape: Shape) -> Option<Self> {
        if let Shape::Live = shape {
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
            return Some(Self {
                pipeline,
                duration: None,
                terminals: vec![("screen", log)],
                _feeds: Vec::new(),
            });
        }

        if let Shape::Offline = shape {
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
            return Some(Self {
                pipeline,
                duration: None,
                terminals: vec![("screen", log)],
                _feeds: vec![feed],
            });
        }

        let path = try_test_video()?;
        let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
        let duration = source.duration().expect("the fixture says how long it is");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("the fixture has a picture")
            .clone();
        let audio = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
            .cloned();
        let (packets, frames) = match shape {
            Shape::Tight => (1, 1),
            _ => (8, 4),
        };
        let mut terminals = Vec::new();
        let name = format!("conformance-{shape:?}").to_lowercase();
        let (pipeline, ()) = Pipeline::new(name, source, |source, ctx| {
            if let Shape::TeeOfPackets = shape {
                let mut tee = ctx.tee("tee");
                for (name, delay) in [("screen", 0), ("second-screen", 20)] {
                    let (recorder, log) = Recorder::new(name, video.time_base);
                    terminals.push((name, log));
                    tee = tee.branch(
                        ctx.branch()
                            .queue(format!("{name}-packets"), packets)
                            .pipe(SwDecoder::new(
                                format!("{name}-decoder"),
                                video.parameters.clone(),
                            )?)
                            .queue(format!("{name}-frames"), frames)
                            .pipe(Pacer::new(format!("{name}-pacer")))
                            .to(recorder.slowed(Duration::from_millis(delay)))?,
                    );
                }
                ctx.attach(source, video.index, tee.build()?)?;
                return Ok(());
            }
            let decoder = SwDecoder::new("video-decoder", video.parameters.clone())?;
            let picture = ctx
                .branch()
                .queue("video-packets", packets)
                .pipe(decoder)
                .queue("video-frames", frames);
            let picture = match shape {
                Shape::Tee => {
                    let mut branches = Vec::new();
                    for name in ["screen", "second-screen"] {
                        let (recorder, log) = Recorder::new(name, video.time_base);
                        terminals.push((name, log));
                        branches.push(
                            ctx.branch()
                                .queue(format!("{name}-frames"), 2)
                                .pipe(Pacer::new(format!("{name}-pacer")))
                                .to(recorder)?,
                        );
                    }
                    let mut tee = ctx.tee("tee");
                    for branch in branches {
                        tee = tee.branch(branch);
                    }
                    picture.to_branch(tee.build()?)?
                }
                Shape::Synchronized => {
                    let (recorder, log) = Recorder::new("screen", video.time_base);
                    terminals.push(("screen", log));
                    picture
                        .pipe(VideoSynchronizer::new("video-sync"))
                        .to(recorder)?
                }
                _ => {
                    let (recorder, log) = Recorder::new("screen", video.time_base);
                    terminals.push(("screen", log));
                    picture.pipe(Pacer::new("video-pacer")).to(recorder)?
                }
            };
            ctx.attach(source, video.index, picture)?;

            if let (Shape::AudioVideo, Some(audio)) = (shape, &audio) {
                let (recorder, log) = Recorder::new("speakers", audio.time_base);
                terminals.push(("speakers", log));
                let sound = ctx
                    .branch()
                    .queue("audio-packets", packets)
                    .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
                    .queue("audio-frames", frames)
                    .pipe(Pacer::new("audio-pacer"))
                    // As a mixer input is fed: stretched to the rate after its
                    // pacer, which holds sound until it has enough to stretch.
                    .pipe(crate::elements::AudioTempo::new("audio-tempo"))
                    .to(recorder)?;
                ctx.attach(source, audio.index, sound)?;
            }
            Ok(())
        })
        .expect("wire the file pipeline");
        Some(Self {
            pipeline,
            duration: Some(duration),
            terminals,
            _feeds: Vec::new(),
        })
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
    /// To just short of the end, playing, and on until it finishes.
    PlayToEnd,
    /// The picture this many pictures on, or back.
    Step(i64),
    /// Playing on at this rate, backwards at [`Pipeline::REVERSE_RATE`].
    Rate(f64),
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

fn op(rng: &mut Rng, duration: Option<Duration>) -> Op {
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
    match rng.below(13) {
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
        12 => Op::Rate([0.5, 1.0, 2.0, -0.5, -1.0, -2.0][rng.below(6) as usize]),
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
    /// The seeks that succeeded, in order: what each terminal's `Seek`
    /// controls are matched against. `None` for one whose target the
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
}

impl Model {
    fn backwards(&self) -> bool {
        self.rate < 0.0
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
            duration.is_some_and(|duration| target + Duration::from_secs(3) < duration)
        };
    }

    fn waited(&mut self, wait: Duration) {
        if !self.paused {
            self.played += wait.mul_f64(self.rate.abs());
            if self.played > Duration::from_millis(1_500) {
                self.far_from_end = false;
            }
        }
    }
}

/// Runs one sequence against a fresh pipeline of `shape`, and says what it
/// broke, if anything.
fn run_sequence(shape: Shape, seed: u64, steps: usize) -> std::result::Result<(), String> {
    let Some(rig) = Rig::build(shape) else {
        return Ok(());
    };
    let rig = Arc::new(rig);
    let mut rng = Rng(seed.max(1));
    let mut history: Vec<String> = Vec::new();
    let fail = |history: &[String], why: String| {
        Err(format!(
            "{shape:?} seed {seed}: {why}\n  after: {}",
            history.join(" → ")
        ))
    };

    let bus = BusLog::start(&rig.pipeline);
    let mut model = Model {
        far_from_end: rig.duration.is_some(),
        rate: 1.0,
        ..Model::default()
    };
    model.sought(Duration::ZERO, rig.duration);
    if rig.duration.is_none() {
        model.far_from_end = true;
    }
    {
        let pipeline = Arc::clone(&rig.pipeline);
        match within(move || pipeline.run()) {
            Some(Ok(())) => {}
            Some(Err(error)) => return fail(&history, format!("run failed: {error}")),
            None => return fail(&history, "run did not return".into()),
        }
    }

    for _ in 0..steps {
        let step = op(&mut rng, rig.duration);
        history.push(format!("{step:?}"));
        let before = rig.data_counts();
        match step {
            Op::Pause => {
                let pipeline = Arc::clone(&rig.pipeline);
                if within(move || pipeline.pause()).is_none() {
                    return fail(&history, "pause did not return".into());
                }
                model.paused = true;
            }
            Op::Resume => {
                let pipeline = Arc::clone(&rig.pipeline);
                if within(move || pipeline.resume()).is_none() {
                    return fail(&history, "resume did not return".into());
                }
                model.paused = false;
                if std::mem::take(&mut model.stepped)
                    && let Some(picture) = model.picture
                {
                    // Lined up to the picture before playing on.
                    model.seeked(None, SeekMode::Accurate);
                    model.sought(picture, rig.duration);
                }
            }
            Op::Rate(rate) => {
                let pipeline = Arc::clone(&rig.pipeline);
                match within(move || pipeline.set_rate(rate)) {
                    None => return fail(&history, "set_rate did not return".into()),
                    Some(Ok(())) => {
                        let turned = (rate < 0.0) != model.backwards();
                        model.rate = rate;
                        if turned {
                            // Repositioned to the picture shown, which the
                            // model does not know: whether that is far from
                            // the end it now plays towards neither.
                            model.seeked(None, SeekMode::Accurate);
                            model.stepped = false;
                            model.played = Duration::ZERO;
                            model.far_from_end = false;
                        }
                    }
                    Some(Err(crate::Error::SeekError(_))) if rig.duration.is_none() => {}
                    Some(Err(error)) => return fail(&history, format!("set_rate failed: {error}")),
                }
            }
            Op::Step(frames) => {
                let pipeline = Arc::clone(&rig.pipeline);
                match within(move || pipeline.step(frames)) {
                    None => return fail(&history, "step did not return".into()),
                    Some(Ok(at)) => {
                        model.paused = true;
                        model.stepped = true;
                        model.picture = Some(at);
                        // A step against the way it plays is a seek.
                        if (frames < 0) != model.backwards() {
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
                    Some(Err(error)) => return fail(&history, format!("step failed: {error}")),
                }
            }
            Op::Seek(target, mode) => {
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
                    Some(Err(error)) => return fail(&history, format!("seek failed: {error}")),
                }
            }
            Op::Wait(wait) => {
                thread::sleep(wait);
                model.waited(wait);
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
                let pipeline = Arc::clone(&rig.pipeline);
                match within(move || pipeline.seek(target, SeekMode::Accurate)) {
                    None => return fail(&history, "seek to the end did not return".into()),
                    Some(Err(error)) => {
                        return fail(&history, format!("seek to the end failed: {error}"));
                    }
                    Some(Ok(())) => {
                        model.seeked(Some(target), SeekMode::Accurate);
                        model.stepped = false;
                    }
                }
                let pipeline = Arc::clone(&rig.pipeline);
                if within(move || pipeline.resume()).is_none() {
                    return fail(&history, "resume did not return".into());
                }
                model.paused = false;
                model.sought(target, rig.duration);
                let deadline = Instant::now() + OP_TIMEOUT;
                while !bus.finished_since(from) {
                    if Instant::now() > deadline {
                        return fail(&history, "played to the end, never Finished".into());
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                model.far_from_end = false;
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

    // Ended either way a caller ends one: abandoned, or finished in order —
    // which has to leave every terminal with an `Eos` after its last data.
    let finishing = rng.below(2) == 0;
    let pipeline = Arc::clone(&rig.pipeline);
    if finishing {
        history.push("Finish".into());
        if within(move || pipeline.finish()).is_none() {
            return fail(&history, "finish did not return".into());
        }
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
    } else {
        history.push("Stop".into());
        if within(move || pipeline.stop()).is_none() {
            return fail(&history, "stop did not return".into());
        }
    }
    let errors = bus.errors();
    if !errors.is_empty() {
        return fail(&history, format!("errors on the bus: {errors:?}"));
    }
    let landings = bus.landings();
    for (name, log) in &rig.terminals {
        let log = log.lock().unwrap();
        if let Err(why) = judge(
            &log,
            &model.seeks,
            &landings,
            rig.duration,
            shape.lossless(),
            *name != "speakers",
        ) {
            let handed = sketch(&log);
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
/// at, a control message by its name.
fn sketch(log: &[Entry]) -> String {
    let words: Vec<String> = log
        .iter()
        .map(|entry| match entry {
            Entry::Data { pts: Some(pts) } => pts.as_millis().to_string(),
            Entry::Data { pts: None } => "?".into(),
            Entry::Eos => "Eos".into(),
            Entry::Control(ControlMsg::Seek(target)) => format!("Seek({})", target.as_millis()),
            Entry::Control(ControlMsg::Preroll(context)) => {
                format!("Preroll({})", context.samples())
            }
            Entry::Control(message) => format!("{message:?}"),
        })
        .collect();
    words.join(" ")
}

/// The last few things a terminal was handed, for a failure to show.
fn tail(log: &[Entry]) -> &[Entry] {
    &log[log.len().saturating_sub(8)..]
}

/// Checks one terminal's record against the promises about data.
fn judge(
    log: &[Entry],
    seeks: &[(Option<Duration>, SeekMode, bool)],
    landings: &[Duration],
    duration: Option<Duration>,
    lossless: bool,
    shows_pictures: bool,
) -> std::result::Result<(), String> {
    // The widest step allowed between two samples in a row: a few of the
    // stream's own, so a picture or a packet of sound gone missing shows,
    // and one a timestamp rounded differently does not.
    let widest = typical_spacing(log)
        .map(|spacing| (spacing * 5 / 2).max(spacing + Duration::from_millis(20)));
    let mut paused = false;
    // How many samples a preroll has let through while paused, not yet
    // taken: what each `Preroll` asks each terminal for — one for a seek's,
    // the pictures asked for in a step's.
    let mut prerolls = 0usize;
    let mut flushed = false;
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
    for (at, entry) in log.iter().enumerate() {
        match entry {
            Entry::Control(ControlMsg::Pause) => {
                paused = true;
                prerolls = 0;
            }
            Entry::Control(ControlMsg::Resume) => paused = false,
            Entry::Control(ControlMsg::Preroll(context)) if paused => {
                prerolls += context.samples();
                skipped |= context.is_step() && !shows_pictures;
            }
            Entry::Control(ControlMsg::Flush) => flushed = true,
            Entry::Control(ControlMsg::Seek(target)) => {
                flushed = false;
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
                seek += 1;
            }
            Entry::Control(ControlMsg::Stop) => break,
            Entry::Control(_) | Entry::Eos => {}
            Entry::Data { pts } => {
                if paused {
                    if prerolls == 0 && !ends_the_stream(&log[at..]) {
                        return Err(format!("entry {at}: data while paused, {pts:?}"));
                    }
                    prerolls = prerolls.saturating_sub(1);
                }
                if flushed {
                    return Err(format!(
                        "entry {at}: data between a flush and its seek, {pts:?}"
                    ));
                }
                if ceiling.is_some() && !shows_pictures {
                    return Err(format!("entry {at}: sound played backwards, {pts:?}"));
                }
                let Some(pts) = *pts else { continue };
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

/// Whether `rest` is samples and then the end of the stream, nothing
/// between them: what a `Pacer` kept goes on with the `Eos` behind it,
/// preroll or not, since nothing would come to hand it on after — see
/// `Pacer::consume`. A step it had kept a picture for then shows the last
/// of the file rather than the one picture it asked for.
fn ends_the_stream(rest: &[Entry]) -> bool {
    rest.iter()
        .find(|entry| !matches!(entry, Entry::Data { .. }))
        .is_some_and(|entry| matches!(entry, Entry::Eos))
}

/// How far apart a terminal's samples usually are: the median step
/// between two in a row, either way, not counting across a seek. `None` for a record
/// too short to say.
fn typical_spacing(log: &[Entry]) -> Option<Duration> {
    let mut steps = Vec::new();
    let mut last: Option<Duration> = None;
    for entry in log {
        match entry {
            Entry::Control(ControlMsg::Seek(_)) => last = None,
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

/// The sequences to run: fixed ones by default, so an ordinary run is the
/// same every time; more, or clock-seeded ones, when asked.
fn seeds(shape: Shape) -> Vec<u64> {
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
    } else {
        (shape as u64 + 1) * 1_000
    };
    (0..iterations)
        .map(|index| base.wrapping_add(index))
        .collect()
}

fn conform(shape: Shape) {
    trace_if_asked();
    let steps = std::env::var("MEDIA_PP_CONTROL_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12);
    for seed in seeds(shape) {
        if let Err(failure) = run_sequence(shape, seed, steps) {
            panic!("{failure}");
        }
    }
}

#[test]
fn a_file_picture_keeps_its_control_promises() {
    conform(Shape::Video);
}

#[test]
fn a_file_with_sound_keeps_its_control_promises() {
    conform(Shape::AudioVideo);
}

#[test]
fn a_fanned_out_picture_keeps_its_control_promises() {
    conform(Shape::Tee);
}

#[test]
fn one_deep_queues_keep_their_control_promises() {
    conform(Shape::Tight);
}

#[test]
fn a_live_source_keeps_its_control_promises() {
    conform(Shape::Live);
}

#[test]
fn an_offline_render_keeps_its_control_promises() {
    conform(Shape::Offline);
}

#[test]
fn a_synchronized_picture_keeps_its_control_promises() {
    conform(Shape::Synchronized);
}

#[test]
fn packets_fanned_out_before_decoding_keep_their_control_promises() {
    conform(Shape::TeeOfPackets);
}
