//! Control conformance for driven pipelines: random orders of pause,
//! resume and waits, ended by a stop, a finish or a drop, against every
//! shape a driven pipeline comes in, polled every way one is — and judged,
//! as the threaded sequences are, against what every terminal was handed.
//!
//! A driven pipeline cannot be sought and has no queue, so its sequences
//! are shorter than a file's; what they are for is the other half of the
//! risk. Its control is applied by whatever polls it, which a request
//! wakes from another thread; it is polled on threads it shares, by a
//! runner, an executor that knows nothing of it, or its owner's own loop;
//! and three of them run side by side in every sequence. The promises:
//!
//! - every request is taken: a pause reaches every terminal, a resume
//!   brings data again, a stop or a finish ends the pipeline — each within
//!   [`TAKEN`] of the call, with nothing else to wake the poll;
//! - nothing reaches a terminal between its pause and its resume;
//! - a pause costs the source's schedule nothing: what comes after one
//!   follows what came before it, rather than the pause's worth later;
//! - every stream begins with a segment, before any data;
//! - stopped or dropped, a terminal's last word is its stop; finished, its
//!   end, and the bus says the pipeline has played everything;
//! - every branch of a fan-out is handed the same;
//! - and nothing reports an error.
//!
//! Run as the threaded ones are — `MEDIA_PP_CONTROL_ITERS`,
//! `MEDIA_PP_CONTROL_RANDOM`, `MEDIA_PP_CONTROL_SEED` with
//! `MEDIA_PP_CONTROL_SHAPE`, `MEDIA_PP_CONTROL_STEPS`,
//! `MEDIA_PP_CONTROL_LOAD`, pinned to two cores — see the module above.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context as TaskContext, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

use ffmpeg_next::{self as ffmpeg, Rescale};

use super::{Rng, load_if_asked, seeds, trace_if_asked};
use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    element::{Element, ElementType, Filter, IntoSource, Output, RawSink, element_pp_log},
    elements::{TestAudioOptions, TestAudioSource, TestVideoOptions, TestVideoSource},
    pipeline::{Driven, DrivenPipeline, DrivenPipelineHandle, DrivenRunner},
    pp_log::PpLog,
    stream::StreamEvent,
};

/// How long a request may take to be taken, and a resumed pipeline to
/// hand on something new. Generous, for a loaded two-core runner: a
/// request that is never taken never is.
const TAKEN: Duration = Duration::from_secs(10);

/// How far apart, in the stream, what came last before a pause and first
/// after it may be: a picture or a tick of sound, and some — not the
/// pause's length.
const ACROSS_A_PAUSE: Duration = Duration::from_millis(200);

/// How many pipelines run side by side in a sequence.
const SIDE_BY_SIDE: usize = 3;

#[derive(Debug, Clone, Copy)]
enum Makes {
    Pictures,
    Sound,
}

#[derive(Debug, Clone, Copy)]
enum After {
    /// The terminal straight behind the source.
    Nothing,
    /// A filter of its own in between.
    Filter,
    /// A fan-out to two terminals.
    FanOut,
}

#[derive(Debug, Clone, Copy)]
enum Poller {
    /// A [`DrivenRunner`] of two threads, all three on it.
    Runner,
    /// Each as a [`Future`](std::future::Future) on an executor of its own
    /// that knows nothing of it but its waker.
    Future,
    /// Each by its owner's own loop, keeping the time it asks for.
    ByHand,
}

#[derive(Debug, Clone, Copy)]
struct DrivenShape {
    makes: Makes,
    after: After,
    poller: Poller,
}

impl DrivenShape {
    fn label(self) -> String {
        format!("driven-{:?}-{:?}-{:?}", self.makes, self.after, self.poller).to_lowercase()
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Pause,
    Resume,
    Wait(Duration),
}

#[derive(Debug, Clone, Copy)]
enum Ending {
    Stop,
    Finish,
    Drop,
}

/// What a terminal was handed and told, and the harness's calls, in order.
#[derive(Debug, Clone, PartialEq)]
enum Entry {
    Segment,
    Data(Duration),
    Paused,
    Resumed,
    Stopped,
    Eos,
    Called(&'static str),
}

#[derive(Clone, Default)]
struct Record(Arc<Mutex<Vec<Entry>>>);

impl Record {
    fn push(&self, entry: Entry) {
        self.0.lock().unwrap().push(entry);
    }

    fn entries(&self) -> Vec<Entry> {
        self.0.lock().unwrap().clone()
    }

    /// Whether `wanted` came after the last time `call` was written.
    fn since_call(&self, call: &'static str, wanted: impl Fn(&Entry) -> bool) -> bool {
        let entries = self.entries();
        let from = entries
            .iter()
            .rposition(|entry| *entry == Entry::Called(call))
            .map_or(0, |at| at + 1);
        entries[from..].iter().any(wanted)
    }
}

struct Recorder {
    pp_log: PpLog,
    name: Arc<str>,
    record: Record,
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

impl RawSink for Recorder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        let at = buf
            .pts()
            .zip(buf.time_base())
            .map(|(pts, unit)| pts.rescale(unit, ffmpeg::Rational(1, 1_000_000_000)))
            .unwrap_or(-1);
        self.record
            .push(Entry::Data(Duration::from_nanos(at.max(0) as u64)));
        Ok(())
    }

    fn stream_event(&mut self, event: &StreamEvent) -> crate::error::Result<()> {
        self.record.push(match event {
            StreamEvent::Segment(_) => Entry::Segment,
            StreamEvent::Eos => Entry::Eos,
        });
        Ok(())
    }

    fn pausing(&mut self) -> crate::error::Result<()> {
        self.record.push(Entry::Paused);
        Ok(())
    }

    fn resuming(&mut self) -> crate::error::Result<()> {
        self.record.push(Entry::Resumed);
        Ok(())
    }

    fn stopping(&mut self) -> crate::error::Result<()> {
        self.record.push(Entry::Stopped);
        Ok(())
    }
}

/// Hands on what it is handed — a filter between the source and the
/// terminal, as a server's are.
struct Through;

impl Filter for Through {
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> crate::error::Result<()> {
        out.push(buf);
        Ok(())
    }
}

/// One pipeline of a sequence: its terminals' records and its handle.
struct Under {
    records: Vec<Record>,
    handle: DrivenPipelineHandle,
    paused: bool,
}

impl Under {
    fn call(&self, call: &'static str) {
        for record in &self.records {
            record.push(Entry::Called(call));
        }
    }

    fn wait_for(&self, what: &str, done: impl Fn(&Record) -> bool) -> Result<(), String> {
        let deadline = Instant::now() + TAKEN;
        for record in &self.records {
            while !done(record) {
                if Instant::now() >= deadline {
                    return Err(format!("{what} within {TAKEN:?}"));
                }
                thread::sleep(Duration::from_millis(2));
            }
        }
        Ok(())
    }
}

fn build(shape: DrivenShape, index: usize) -> (DrivenPipeline, Under) {
    let terminals = if matches!(shape.after, After::FanOut) {
        2
    } else {
        1
    };
    let records: Vec<Record> = (0..terminals).map(|_| Record::default()).collect();
    let sinks: Vec<Recorder> = records
        .iter()
        .enumerate()
        .map(|(branch, record)| {
            let name: Arc<str> = format!("terminal-{branch}").into();
            Recorder {
                pp_log: element_pp_log(ElementType::Other, &name, None),
                name,
                record: record.clone(),
            }
        })
        .collect();
    let id = format!("{}-{index}", shape.label());
    let pipeline = match shape.makes {
        Makes::Pictures => wired(
            id,
            TestVideoSource::new(
                "camera",
                TestVideoOptions {
                    width: 32,
                    height: 24,
                    ..TestVideoOptions::default()
                },
            ),
            shape.after,
            sinks,
        ),
        Makes::Sound => wired(
            id,
            TestAudioSource::new("tone", TestAudioOptions::default()),
            shape.after,
            sinks,
        ),
    };
    let handle = pipeline.handle();
    (
        pipeline,
        Under {
            records,
            handle,
            paused: false,
        },
    )
}

fn wired<M, I: IntoSource<M>>(
    id: String,
    source: I,
    after: After,
    mut sinks: Vec<Recorder>,
) -> DrivenPipeline {
    let (pipeline, ()) = DrivenPipeline::new(id, source, |source, ctx| {
        let branch = match after {
            After::Nothing => ctx.branch().to(sinks.remove(0))?,
            After::Filter => ctx
                .branch()
                .pipe(Through.named("through"))
                .to(sinks.remove(0))?,
            After::FanOut => {
                let first = ctx.branch().to(sinks.remove(0))?;
                let second = ctx.branch().to(sinks.remove(0))?;
                ctx.tee("tee").branch(first).branch(second).build()?
            }
        };
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("a driven shape wires");
    pipeline
}

struct Unpark(thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// What polls one sequence's pipelines, and how a drop reaches them.
enum Polling {
    Runner(Option<DrivenRunner>),
    Threads(Vec<(Arc<AtomicBool>, thread::JoinHandle<()>)>),
}

impl Polling {
    fn start(poller: Poller, pipelines: Vec<DrivenPipeline>) -> Self {
        match poller {
            Poller::Runner => {
                let runner = DrivenRunner::new(2).expect("a runner starts");
                for pipeline in pipelines {
                    runner.spawn(pipeline);
                }
                Self::Runner(Some(runner))
            }
            Poller::Future | Poller::ByHand => Self::Threads(
                pipelines
                    .into_iter()
                    .map(|pipeline| {
                        let dropped = Arc::new(AtomicBool::new(false));
                        let thread = thread::spawn({
                            let dropped = Arc::clone(&dropped);
                            move || poll_on_this_thread(pipeline, poller, &dropped)
                        });
                        (dropped, thread)
                    })
                    .collect(),
            ),
        }
    }

    /// Drops every pipeline still being polled, where it is polled.
    fn drop_all(&mut self) {
        match self {
            Self::Runner(runner) => drop(runner.take()),
            Self::Threads(threads) => {
                for (dropped, thread) in threads.drain(..) {
                    dropped.store(true, Ordering::Release);
                    thread.thread().unpark();
                    let _ = thread.join();
                }
            }
        }
    }
}

fn poll_on_this_thread(mut pipeline: DrivenPipeline, poller: Poller, dropped: &AtomicBool) {
    let waker = Waker::from(Arc::new(Unpark(thread::current())));
    let mut cx = TaskContext::from_waker(&waker);
    while !dropped.load(Ordering::Acquire) {
        match poller {
            Poller::Future => {
                if std::pin::Pin::new(&mut pipeline).poll(&mut cx).is_ready() {
                    return;
                }
                thread::park();
            }
            _ => match pipeline.poll(&mut cx) {
                Driven::Finished => return,
                Driven::Pending { at: None } => thread::park(),
                Driven::Pending { at: Some(at) } => {
                    thread::park_timeout(at.saturating_duration_since(Instant::now()));
                }
            },
        }
    }
}

fn run_sequence(shape: DrivenShape, seed: u64, steps: usize) -> Result<(), String> {
    let mut rng = Rng(seed.max(1));
    let (pipelines, mut under): (Vec<_>, Vec<_>) =
        (0..SIDE_BY_SIDE).map(|index| build(shape, index)).unzip();
    let mut polling = Polling::start(shape.poller, pipelines);
    let mut history = Vec::new();
    let outcome = (|| {
        for _ in 0..steps {
            let which = rng.below(SIDE_BY_SIDE as u64) as usize;
            let op = match rng.below(5) {
                0 | 1 => Op::Pause,
                2 => Op::Resume,
                _ => Op::Wait(Duration::from_millis(rng.below(150))),
            };
            history.push(format!("{op:?} #{which}"));
            let under = &mut under[which];
            match op {
                Op::Pause => {
                    under.call("pause");
                    under.handle.pause();
                    if !under.paused {
                        under.paused = true;
                        under.wait_for("the pause reaches every terminal", |record| {
                            record.since_call("pause", |entry| *entry == Entry::Paused)
                        })?;
                    }
                }
                Op::Resume => {
                    under.call("resume");
                    under.handle.resume();
                    if under.paused {
                        under.paused = false;
                        under.wait_for("data flows again after a resume", |record| {
                            record.since_call("resume", |entry| matches!(entry, Entry::Data(_)))
                        })?;
                    }
                }
                Op::Wait(long) => thread::sleep(long),
            }
        }
        let ending = match rng.below(3) {
            0 => Ending::Stop,
            1 => Ending::Finish,
            _ => Ending::Drop,
        };
        history.push(format!("{ending:?}"));
        match ending {
            Ending::Stop | Ending::Finish => {
                for under in &under {
                    if matches!(ending, Ending::Stop) {
                        under.call("stop");
                        under.handle.stop();
                    } else {
                        under.call("finish");
                        under.handle.finish();
                    }
                }
                for under in &under {
                    let deadline = Instant::now() + TAKEN;
                    while !under.handle.is_finished() {
                        if Instant::now() >= deadline {
                            return Err(format!("{ending:?} did not end it within {TAKEN:?}"));
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                }
                polling.drop_all();
            }
            Ending::Drop => {
                for under in &under {
                    under.call("drop");
                }
                polling.drop_all();
            }
        }
        for (index, under) in under.iter().enumerate() {
            judge(under, ending).map_err(|why| format!("pipeline #{index}: {why}"))?;
        }
        Ok(())
    })();
    polling.drop_all();
    outcome.map_err(|why| {
        format!(
            "{} seed {seed}: {why}\n  after: {}\n  replay: MEDIA_PP_CONTROL_SHAPE={} MEDIA_PP_CONTROL_SEED={seed}",
            shape.label(),
            history.join(", "),
            shape.label()
        )
    })
}

/// Holds one pipeline's records to the promises, once it has ended.
fn judge(under: &Under, ending: Ending) -> Result<(), String> {
    let mut handed = Vec::new();
    for record in &under.records {
        let entries = record.entries();
        let heard: Vec<&Entry> = entries
            .iter()
            .filter(|entry| !matches!(entry, Entry::Called(_)))
            .collect();
        if heard.first() != Some(&&Entry::Segment) {
            return Err(format!(
                "its stream did not begin with a segment: {heard:?}"
            ));
        }
        let mut paused = false;
        let mut before_pause = None;
        let mut last = None;
        for entry in &heard {
            match entry {
                Entry::Data(at) => {
                    if paused {
                        return Err(format!("data at {at:?} while paused"));
                    }
                    if let Some(before) = before_pause.take()
                        && at.saturating_sub(before) > ACROSS_A_PAUSE
                    {
                        return Err(format!(
                            "{before:?} before a pause, and {at:?} after it: the pause cost the schedule"
                        ));
                    }
                    last = Some(*at);
                }
                Entry::Paused => {
                    paused = true;
                    before_pause = last;
                }
                Entry::Resumed => paused = false,
                _ => {}
            }
        }
        let wanted = match ending {
            Ending::Stop | Ending::Drop => Entry::Stopped,
            Ending::Finish => Entry::Eos,
        };
        if heard.last() != Some(&&wanted) {
            return Err(format!(
                "{ending:?}, and its last word was {:?}, not {wanted:?}",
                heard.last()
            ));
        }
        if heard
            .iter()
            .filter(|entry| ***entry == Entry::Segment)
            .count()
            != 1
        {
            return Err("more than one segment, with nothing to begin one".to_owned());
        }
        handed.push(
            heard
                .iter()
                .filter_map(|entry| match entry {
                    Entry::Data(at) => Some(*at),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        );
    }
    if handed.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err("the branches of its fan-out were handed different data".to_owned());
    }
    let events: Vec<BusEvent> = std::iter::from_fn(|| under.handle.bus().try_recv()).collect();
    if let Some(error) = events
        .iter()
        .find(|event| matches!(event, BusEvent::Error { .. }))
    {
        return Err(format!("an error: {error}"));
    }
    if matches!(ending, Ending::Finish)
        && !events
            .iter()
            .any(|event| matches!(event, BusEvent::Finished))
    {
        return Err(format!("finished, and the bus did not say so: {events:?}"));
    }
    Ok(())
}

fn conform(poller: Poller) {
    trace_if_asked();
    load_if_asked();
    let steps = std::env::var("MEDIA_PP_CONTROL_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);
    let shapes: Vec<DrivenShape> = [Makes::Pictures, Makes::Sound]
        .into_iter()
        .flat_map(|makes| {
            [After::Nothing, After::Filter, After::FanOut]
                .into_iter()
                .map(move |after| DrivenShape {
                    makes,
                    after,
                    poller,
                })
        })
        .filter(|shape| {
            std::env::var("MEDIA_PP_CONTROL_SHAPE").map_or(true, |only| only == shape.label())
        })
        .collect();
    // Side by side, as a server's pipelines are, and so that they take no
    // longer than one shape would.
    let failures: Vec<String> = thread::scope(|scope| {
        let runs: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(key, shape)| {
                let shape = *shape;
                let key = 10_000 + key as u64 + 100 * poller as u64;
                scope.spawn(move || {
                    seeds(key)
                        .into_iter()
                        .filter_map(|seed| run_sequence(shape, seed, steps).err())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        runs.into_iter()
            .flat_map(|run| run.join().expect("a sequence's thread does not panic"))
            .collect()
    });
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn driven_on_a_runner() {
    conform(Poller::Runner);
}

#[test]
fn driven_as_a_future() {
    conform(Poller::Future);
}

#[test]
fn driven_by_hand() {
    conform(Poller::ByHand);
}
