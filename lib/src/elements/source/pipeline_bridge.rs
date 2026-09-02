use std::{
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use crate::pp_log::{PpLog, pp_info, pp_warn};
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{InputContract, OutputContract},
    control::{ControlMsg, ControlReceiver, drain_control},
    element::{Element, ElementType, Sink, Source, SourceElement, element_pp_log},
    error::Result,
    pad::SrcPad,
    queue::OverflowPolicy,
};

/// How long [`PipelineBridge::run`] waits for a buffer before looking at its
/// own control channel again.
///
/// The reason this is a poll rather than a blocking receive: a bridge with no
/// input is the ordinary state, not a fault, and a `Stop` sent to it has to
/// arrive while it is in exactly that state. Short enough that stopping feels
/// immediate, long enough that an idle bridge is not a spin.
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Errors specific to [`PipelineBridge`].
#[derive(Debug, ThisError)]
pub enum PipelineBridgeError {
    /// This sink was replaced by a later [`PipelineBridgeHandle::connect`],
    /// or had already reported its own end.
    ///
    /// Reported rather than ignored so the pipeline still pushing into it
    /// hears about it: a `Queue` posts it to its own bus and stops, which is
    /// how a producer nobody is reading learns to end.
    #[error("this bridge input was superseded by a later connection")]
    Superseded,

    /// The buffer could not be handed over before the configured
    /// [`OverflowPolicy::Block`] timeout elapsed.
    #[error("the bridge did not take a buffer within {0:?}")]
    SendTimedOut(Duration),

    /// Seeking was asked of a bridge. The timeline belongs to whatever feeds
    /// it, in a pipeline this one has no authority over.
    #[error("a PipelineBridge cannot seek what another pipeline is producing")]
    SeekUnsupported,

    /// The bridge's own pipeline has finished, so there is nothing on the
    /// other side any more.
    ///
    /// Distinct from [`PipelineBridgeError::Superseded`] because the feeding
    /// side answers them differently: a superseded input can connect again,
    /// and this one has nowhere left to connect to.
    #[error("the pipeline on the other side of this bridge has finished")]
    Disconnected,
}

/// Construction-time options for [`PipelineBridge::new`].
#[derive(Debug, Clone, Copy)]
pub struct PipelineBridgeOptions {
    /// How many buffers may sit between the two pipelines.
    ///
    /// The same trade a [`crate::queue::Queue`] makes: room to absorb one
    /// side being briefly busy, at the cost of that many buffers' worth of
    /// latency and of whatever they hold open.
    pub depth: usize,
    /// What happens when that room runs out — see [`OverflowPolicy`].
    pub policy: OverflowPolicy,
}

impl Default for PipelineBridgeOptions {
    fn default() -> Self {
        Self {
            depth: 8,
            policy: OverflowPolicy::default(),
        }
    }
}

/// Carries buffers from one [`crate::pipeline::Pipeline`] into another, so
/// the two can start, end and fail independently.
///
/// # What it is for
///
/// A pipeline is one-shot: a source that dies is not restarted, it is
/// replaced, and replacing it means building a new pipeline. Everything
/// downstream of it would go with it — unless the boundary falls between
/// them. This is that boundary, in the general case.
///
/// The general case is what was missing. Crossing from one pipeline into
/// another was already possible through [`crate::elements::AudioMixer`] or a
/// video compositor, and an application whose graph meets at one of those
/// needs nothing here. But both of them decide what they carry: a media
/// kind, a format, and a rate of their own. Packets, or frames that are not
/// to be composited, or anything else that only needs to *cross*, had
/// nowhere to do it.
///
/// # Shape
///
/// ```text
///  pipeline "up"                        pipeline "down"
///  source ─ … ─ [ PipelineBridgeSink ]  [ PipelineBridge ] ─ … ─ sink
///                        └──────── one bounded queue ───────┘
/// ```
///
/// The downstream half is this element, driven as its pipeline's own
/// `SourceElement`. The upstream half is a [`Sink`] from
/// [`PipelineBridgeHandle::connect`], which whichever pipeline is feeding it
/// terminates at.
///
/// # One input at a time
///
/// Deliberately, and unlike a mixer: with no way to combine buffers, several
/// inputs would only interleave in whatever order they arrived. A second
/// [`PipelineBridgeHandle::connect`] *replaces* the first, which is the
/// reconnection path — the old [`Sink`] then refuses with
/// [`PipelineBridgeError::Superseded`] rather than feeding its own
/// replacement.
///
/// # What an empty bridge does
///
/// Nothing, and its pipeline stays alive doing it. A mixer with no input
/// emits silence and a compositor re-emits its last picture, because each has
/// a rate of its own; a bridge has only what it is given. So a downstream
/// pipeline with no upstream is idle rather than finished, and picks up again
/// when something connects.
///
/// # Ends
///
/// An input's `Eos` ends *the input*, not the bridge — otherwise the first
/// disconnection would tear down the very pipeline this exists to keep
/// running. [`PipelineBridgeHandle::input_ended`] reports it and the next
/// `connect` starts another.
///
/// What ends the bridge is [`PipelineBridgeHandle::finish`], which sends
/// `Eos` downstream and returns from `run`. A muxer down there writes its
/// trailer on that; stopping the pipeline instead tells it to abandon the
/// file. The downstream pipeline's own `finish` does the same thing from the
/// other side — two doors into one room, which is right when the two halves
/// have different owners.
///
/// # Both sides can still be controlled
///
/// Each pipeline keeps its own `pause`, `resume`, `finish` and `stop`, and
/// they mean what they always did. Pausing the *downstream* one stops the
/// bridge emitting, which fills the queue between them, which is felt on the
/// feeding side as ordinary backpressure — blocking or dropping according to
/// [`PipelineBridgeOptions::policy`]. That is a queue behaving like a queue
/// rather than anything the bridge decides.
///
/// What does not cross is control itself, with one exception: see
/// [`Sink::control`] on the input this hands out. Seeking is refused here
/// outright ([`SourceElement::is_seekable`] is `false`) — the timeline
/// belongs to whatever feeds the bridge, and an application holding both
/// pipelines seeks the one that owns it.
///
/// # Timestamps cross unchanged
///
/// The two pipelines have their own [`crate::clock::Clock`] and
/// [`crate::playback_clock::PlaybackClock`], and this does not re-time what
/// passes through it — it cannot, not knowing what that is. A mixer re-times
/// to its own tick and a compositor to its own rate; a bridge hands over the
/// timestamps it was given.
///
/// So downstream must be somewhere those still mean something: a muxer,
/// which writes what it is handed, or anything preceded by
/// [`crate::elements::TimestampOrigin`] to re-base them onto the clock that
/// is actually going to be measured against. What must *not* follow a bridge
/// unguarded is a [`crate::elements::Pacer`] — it would be pacing one
/// pipeline's timestamps against another pipeline's clock, which is a stream
/// released all at once or one that never arrives.
pub struct PipelineBridge {
    pp_log: PpLog,
    name: Arc<str>,
    shared: Arc<BridgeShared>,
    pad: SrcPad,
}

/// Shared between the bridge and every handle and sink derived from it.
struct BridgeShared {
    /// The buffers in flight, and which connection put them there.
    ///
    /// One lock for the queue and the connection identity together: a sink
    /// has to check that it is still the live one *and* push under the same
    /// lock, or a replacement racing with it could have its own first buffer
    /// overtaken by the old input's last.
    state: Mutex<BridgeState>,
    /// Woken by a push, by a connection change, and by `finish`.
    changed: std::sync::Condvar,
    /// Issues a distinct identity for every `connect`, including one
    /// replacing another.
    next_connection: AtomicU64,
    /// Buffers `OverflowPolicy::DropNewest` has thrown away.
    ///
    /// Counted here rather than reported where it happens, because where it
    /// happens is a `Sink` on the feeding pipeline's thread and the bus that
    /// should hear about it belongs to this one. The run loop posts what it
    /// sees this move by — same visibility a `Queue` gives its own drops,
    /// from the side that has a bus to say it on.
    dropped: AtomicU64,
    options: PipelineBridgeOptions,
}

struct BridgeState {
    buffers: std::collections::VecDeque<MediaBuffer>,
    /// Which connection may push. `None` before the first `connect` and
    /// after an input ends.
    connection: Option<u64>,
    /// Set by an input's `Eos`, cleared by the next `connect`.
    input_ended: bool,
    /// Set by `finish`: the bridge sends `Eos` downstream and returns.
    finished: bool,
    /// Set by a `Flush` from the feeding side, cleared once the bridge has
    /// passed it on. A flag rather than a queued marker because a flush that
    /// waited its turn behind the buffers it invalidates would be no flush.
    flush: bool,
}

impl PipelineBridge {
    /// Creates one, and the handle the feeding side connects through.
    pub fn new(
        name: impl Into<String>,
        options: PipelineBridgeOptions,
    ) -> (Self, PipelineBridgeHandle) {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::Other, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "created: depth={}, policy={:?}",
            options.depth,
            options.policy
        );
        let shared = Arc::new(BridgeShared {
            state: Mutex::new(BridgeState {
                buffers: std::collections::VecDeque::new(),
                connection: None,
                input_ended: false,
                finished: false,
                flush: false,
            }),
            changed: std::sync::Condvar::new(),
            next_connection: AtomicU64::new(1),
            dropped: AtomicU64::new(0),
            options,
        });
        let handle = PipelineBridgeHandle {
            shared: Arc::downgrade(&shared),
            name: name.clone(),
        };
        let pad = SrcPad::with_contract(format!("{name}_src"), OutputContract::Passthrough);
        (
            Self {
                pp_log,
                name,
                shared,
                pad,
            },
            handle,
        )
    }
}

/// A cheaply-cloneable way to connect a pipeline to a [`PipelineBridge`], and
/// to end it.
///
/// Holds only a [`Weak`] reference, for the same reason
/// [`crate::elements::MixerHandle`] does: keeping one after the bridge's own
/// pipeline has finished must not keep its buffers alive forever, and every
/// operation becomes a harmless `None` once it is gone.
#[derive(Clone)]
pub struct PipelineBridgeHandle {
    shared: Weak<BridgeShared>,
    name: Arc<str>,
}

impl PipelineBridgeHandle {
    /// A [`Sink`] for the feeding pipeline to terminate at, replacing
    /// whatever was connected before.
    ///
    /// `None` once the bridge's own pipeline has finished.
    pub fn connect(&self) -> Option<Box<dyn Sink>> {
        let shared = self.shared.upgrade()?;
        let id = shared.next_connection.fetch_add(1, Ordering::Relaxed);
        {
            let mut state = shared.state.lock().unwrap();
            // What a *live* input left behind goes with it: cutting one off
            // mid-stream abandons its timeline, and handing on the tail of it
            // would put buffers from a stream nobody is producing any more in
            // front of the new input's own.
            //
            // An input that reached its own end is the opposite case. It
            // finished; what it handed over is complete, and dropping it here
            // would lose data a producer successfully delivered — the tail of
            // a clip, or of the connection a reconnection is replacing.
            if !state.input_ended {
                state.buffers.clear();
            }
            state.connection = Some(id);
            state.input_ended = false;
        }
        shared.changed.notify_all();
        Some(Box::new(PipelineBridgeSink {
            pp_log: element_pp_log(ElementType::Other, &self.name, None),
            name: self.name.clone(),
            id,
            shared: self.shared.clone(),
        }))
    }

    /// Whether the connected input has reached its own end.
    ///
    /// `false` before anything has connected: nothing has ended, it simply
    /// has not begun. What this is for is noticing that a source finished so
    /// another can take its place — see [`PipelineBridgeHandle::connect`].
    pub fn input_ended(&self) -> bool {
        self.shared
            .upgrade()
            .is_some_and(|shared| shared.state.lock().unwrap().input_ended)
    }

    /// Ends the bridge: `Eos` goes downstream and its `run` returns.
    ///
    /// This is the ordered ending, and the difference from stopping the
    /// downstream pipeline is what a muxer down there does about it — writes
    /// its trailer, rather than abandoning the file.
    pub fn finish(&self) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        shared.state.lock().unwrap().finished = true;
        shared.changed.notify_all();
    }
}

/// The upstream half: what a feeding pipeline's branch ends at.
struct PipelineBridgeSink {
    pp_log: PpLog,
    name: Arc<str>,
    id: u64,
    shared: Weak<BridgeShared>,
}

impl Element for PipelineBridgeSink {
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

impl Sink for PipelineBridgeSink {
    /// Anything, because a bridge is defined by not caring: it exists for
    /// what a mixer and a compositor cannot carry. Paired with the
    /// `Passthrough` on the other half, so whatever contract arrives keeps
    /// propagating past the boundary rather than stopping at it.
    fn input_contract(&self) -> InputContract {
        InputContract::Any
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let Some(shared) = self.shared.upgrade() else {
            // The bridge's own pipeline has finished. Reported rather than
            // swallowed, for the reason a superseded input is: a producer
            // nobody reads has no other way to learn it should stop, and
            // without this the whole feeding pipeline would go on reading,
            // decoding and handing buffers to nothing at all.
            return Err(PipelineBridgeError::Disconnected.into());
        };
        let deadline = std::time::Instant::now();
        let mut state = shared.state.lock().unwrap();
        loop {
            if state.connection != Some(self.id) {
                return Err(PipelineBridgeError::Superseded.into());
            }
            if let MediaBuffer::Eos = buf {
                // The input's end, not the bridge's — see the type docs.
                state.input_ended = true;
                state.connection = None;
                drop(state);
                shared.changed.notify_all();
                pp_info!(self, "input ended");
                return Ok(());
            }
            if state.buffers.len() < shared.options.depth {
                state.buffers.push_back(buf);
                drop(state);
                shared.changed.notify_all();
                return Ok(());
            }
            match shared.options.policy {
                OverflowPolicy::DropNewest => {
                    drop(state);
                    // Silent to the caller, as this policy is: falling behind
                    // is what it exists to absorb. The bridge's own run loop
                    // is what says so, on the bus that belongs to it.
                    shared.dropped.fetch_add(1, Ordering::Relaxed);
                    return Ok(());
                }
                OverflowPolicy::Block(timeout) => {
                    let left = timeout.saturating_sub(deadline.elapsed());
                    if left.is_zero() {
                        return Err(PipelineBridgeError::SendTimedOut(timeout).into());
                    }
                    let (guard, _) = shared
                        .changed
                        .wait_timeout(state, left.min(CONTROL_POLL_INTERVAL))
                        .unwrap();
                    state = guard;
                }
            }
        }
    }

    /// `Flush` crosses; nothing else does.
    ///
    /// The test is whether the message means something inside an element or
    /// something about the pipeline that sent it. `Flush` is the first: drop
    /// what you are holding, which downstream *must* hear after a seek or it
    /// keeps frames belonging to a timeline that has been left. Ordering is
    /// not a difficulty for this one message, because arriving ahead of the
    /// buffers it invalidates is exactly what it is for.
    ///
    /// The rest name the sender's own clock. Injecting `Pause` here would
    /// leave this side's elements believing they are paused while this side's
    /// [`crate::clock::Clock`] — the one a `Pacer` and the playback clock
    /// actually read — keeps running. Two authorities over one timeline is
    /// the defect, not the missing feature; the downstream pipeline has its
    /// own `pause`, `finish` and `stop` for what its owner wants of it.
    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        let Some(shared) = self.shared.upgrade() else {
            return Ok(());
        };
        if matches!(msg, ControlMsg::Flush) {
            {
                let mut state = shared.state.lock().unwrap();
                state.buffers.clear();
                state.flush = true;
            }
            shared.changed.notify_all();
        }
        Ok(())
    }
}

impl Element for PipelineBridge {
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

impl Source for PipelineBridge {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for PipelineBridge {
    /// Live, because it cannot be asked for a buffer it has not been given.
    /// Whether what feeds it is live is the other pipeline's business and not
    /// something this can see.
    fn is_live(&self) -> bool {
        true
    }

    /// The timeline belongs to whatever is upstream, in a pipeline this one
    /// has no authority over.
    fn is_seekable(&self) -> bool {
        false
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(PipelineBridgeError::SeekUnsupported.into())
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        let mut reported_drops = 0;
        loop {
            let dropped = self.shared.dropped.load(Ordering::Relaxed);
            if dropped > reported_drops {
                reported_drops = dropped;
                bus.post(
                    &self.pp_log,
                    BusEvent::Dropped {
                        element_type: self.element_type(),
                        name: self.name.clone(),
                    },
                );
            }
            let outcome = drain_control(control, self, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }
            // Waited on with a timeout rather than blocked on, because having
            // nothing to carry is this element's ordinary state and a `Stop`
            // has to arrive while it is in it.
            let taken = {
                let shared = &self.shared;
                let state = shared.state.lock().unwrap();
                let (mut state, _) = shared
                    .changed
                    .wait_timeout(state, CONTROL_POLL_INTERVAL)
                    .unwrap();
                if state.finished {
                    drop(state);
                    pp_info!(self, "finished");
                    if let Err(error) = self.pad.push(MediaBuffer::Eos) {
                        pp_warn!(self, "end of stream was not delivered: {error}");
                    }
                    return Ok(());
                }
                let flush = std::mem::take(&mut state.flush);
                (state.buffers.pop_front(), flush)
            };
            let (taken, flush) = taken;
            if flush {
                self.pad.control(ControlMsg::Flush)?;
            }
            if let Some(buffer) = taken {
                self.shared.changed.notify_all();
                // One buffer's failure is not this bridge's end, the same way
                // a `Queue` reports a failing downstream and keeps its worker.
                if let Err(error) = self.pad.push(buffer) {
                    bus.post(
                        &self.pp_log,
                        BusEvent::Error {
                            element_type: self.element_type(),
                            name: self.name.clone(),
                            error,
                        },
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex as StdMutex, atomic::AtomicUsize};

    use super::*;
    use crate::{elements::AppSink, pipeline::Pipeline};

    /// What crossed the bridge, in order.
    /// A sink, the timestamps that reached it, and how many ends of stream it
    /// saw.
    type Watched = (
        Box<dyn Sink>,
        Arc<StdMutex<Vec<Option<i64>>>>,
        Arc<AtomicUsize>,
    );

    fn watched() -> Watched {
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let ends = Arc::new(AtomicUsize::new(0));
        let sink = AppSink::new("watcher", {
            let seen = Arc::clone(&seen);
            let ends = Arc::clone(&ends);
            move |buffer| {
                match &buffer {
                    MediaBuffer::Packet(packet) => seen.lock().unwrap().push(packet.pts()),
                    MediaBuffer::Eos => {
                        ends.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {}
                }
                Ok(())
            }
        });
        (Box::new(sink), seen, ends)
    }

    /// Counts flushes and keeps what arrived between them.
    struct FlushWatcher {
        pp_log: PpLog,
        flushed: Arc<AtomicUsize>,
        seen: Arc<StdMutex<Vec<Option<i64>>>>,
    }

    impl Element for FlushWatcher {
        fn name(&self) -> Arc<str> {
            "flush-watcher".into()
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

    impl Sink for FlushWatcher {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Packet(packet) = &buf {
                self.seen.lock().unwrap().push(packet.pts());
            }
            Ok(())
        }

        fn control(&mut self, msg: ControlMsg) -> Result<()> {
            if matches!(msg, ControlMsg::Flush) {
                self.flushed.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        }
    }

    fn packet(pts: i64) -> MediaBuffer {
        let mut packet = ffmpeg_next::Packet::empty();
        packet.set_pts(Some(pts));
        MediaBuffer::Packet(Arc::new(packet))
    }

    /// The downstream pipeline, running, with the bridge as its source.
    fn downstream(bridge: PipelineBridge, sink: Box<dyn Sink>) -> std::sync::Arc<Pipeline> {
        let pipeline = Pipeline::new("down", bridge, move |source, context| {
            let branch = context.branch().to(sink)?;
            context.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the downstream pipeline");
        pipeline.run().expect("run the downstream pipeline");
        pipeline
    }

    fn wait_until(mut ready: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !ready() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The whole point, in one line: what goes in one pipeline comes out of
    /// the other.
    #[test]
    fn a_buffer_put_in_one_end_comes_out_of_the_other() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        let (sink, seen, _) = watched();
        let pipeline = downstream(bridge, sink);

        let mut input = handle.connect().expect("the bridge is running");
        input.consume(packet(7)).expect("hand it over");

        wait_until(|| !seen.lock().unwrap().is_empty());
        pipeline.stop();
        assert_eq!(*seen.lock().unwrap(), vec![Some(7)]);
    }

    /// An input ending is not the bridge ending. If it were, the first
    /// disconnection would take down the pipeline this exists to keep
    /// running — and a second input could never replace the first.
    #[test]
    fn an_input_ending_leaves_the_bridge_running_for_the_next_one() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        let (sink, seen, ends) = watched();
        let pipeline = downstream(bridge, sink);

        let mut first = handle.connect().expect("the bridge is running");
        first.consume(packet(1)).expect("hand it over");
        first.consume(MediaBuffer::Eos).expect("end the input");
        wait_until(|| handle.input_ended());

        assert!(handle.input_ended(), "the handle reports the input's end");
        assert_eq!(
            ends.load(Ordering::Relaxed),
            0,
            "an input's end must not reach downstream as the bridge's"
        );
        assert!(pipeline.is_running(), "nor end the pipeline it drives");

        let mut second = handle.connect().expect("still running");
        assert!(!handle.input_ended(), "a new input is not an ended one");
        second.consume(packet(2)).expect("hand it over");

        wait_until(|| seen.lock().unwrap().len() == 2);
        pipeline.stop();
        assert_eq!(*seen.lock().unwrap(), vec![Some(1), Some(2)]);
    }

    /// A replaced input must not be able to feed its own replacement, and
    /// must be told rather than ignored: the pipeline still pushing into it
    /// is how a producer nobody reads learns to stop.
    #[test]
    fn a_superseded_input_is_refused_rather_than_swallowed() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        let (sink, seen, _) = watched();
        let pipeline = downstream(bridge, sink);

        let mut first = handle.connect().expect("the bridge is running");
        let mut second = handle.connect().expect("replacing the first");

        let refused = first.consume(packet(1));
        second
            .consume(packet(2))
            .expect("the live input still works");

        wait_until(|| !seen.lock().unwrap().is_empty());
        pipeline.stop();

        assert!(
            matches!(
                refused,
                Err(crate::error::Error::PipelineBridgeError(
                    PipelineBridgeError::Superseded
                ))
            ),
            "the replaced input has to hear about it, got {refused:?}"
        );
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Some(2)],
            "and must not have put anything in front of its replacement"
        );
    }

    /// A bridge with nothing to carry is the ordinary state, not a fault —
    /// and a `Stop` has to arrive while it is in it. Blocking on the queue
    /// instead of polling would make an idle bridge an unstoppable pipeline.
    #[test]
    fn a_starved_bridge_still_stops() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        let (sink, _, _) = watched();
        let pipeline = downstream(bridge, sink);
        // Nothing ever connects.
        drop(handle);

        let started = std::time::Instant::now();
        pipeline.stop();
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "stopping an idle bridge took {:?}",
            started.elapsed()
        );
        assert!(!pipeline.is_running());
    }

    /// `finish` is the ordered ending, and the difference from stopping is
    /// what a muxer downstream does about it: writes its trailer rather than
    /// abandoning the file.
    #[test]
    fn finish_sends_end_of_stream_downstream() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        let (sink, _, ends) = watched();
        let pipeline = downstream(bridge, sink);

        handle.finish();
        wait_until(|| ends.load(Ordering::Relaxed) > 0);

        assert_eq!(
            ends.load(Ordering::Relaxed),
            1,
            "downstream has to see exactly one end of stream"
        );
        wait_until(|| !pipeline.is_running());
        assert!(!pipeline.is_running(), "and the bridge's own run returns");
    }

    /// A seek on the feeding side leaves this side holding frames from a
    /// timeline that has been left. `Flush` is the one message that crosses,
    /// and it has to arrive ahead of them rather than behind.
    #[test]
    fn a_flush_crosses_and_takes_the_queued_buffers_with_it() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        let flushed = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = FlushWatcher {
            pp_log: element_pp_log(ElementType::Other, "flush-watcher", None),
            flushed: Arc::clone(&flushed),
            seen: Arc::clone(&seen),
        };
        let pipeline = downstream(bridge, Box::new(sink));
        let mut input = handle.connect().expect("the bridge is running");

        input.consume(packet(1)).expect("queued");
        input
            .control(ControlMsg::Flush)
            .expect("a flush from the feeding pipeline");
        input.consume(packet(2)).expect("after the flush");

        wait_until(|| flushed.load(Ordering::Relaxed) > 0 && !seen.lock().unwrap().is_empty());
        pipeline.stop();

        assert_eq!(flushed.load(Ordering::Relaxed), 1, "the flush crossed");
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Some(2)],
            "and what belonged to the timeline it left did not"
        );
    }

    /// A downstream pipeline that has ended is not a reason for the feeding
    /// one to keep reading, decoding and handing buffers to nothing.
    #[test]
    fn feeding_a_bridge_whose_pipeline_has_gone_is_refused() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        let mut input = handle.connect().expect("the bridge is alive");
        drop(bridge);

        assert!(
            matches!(
                input.consume(packet(1)),
                Err(crate::error::Error::PipelineBridgeError(
                    PipelineBridgeError::Disconnected
                ))
            ),
            "the feeding side has to hear that there is nothing on the other side"
        );
    }

    /// The other half of that rule: an input cut off while it was still
    /// running leaves nothing behind, because what it had queued belongs to a
    /// stream that is no longer being produced.
    #[test]
    fn superseding_a_live_input_discards_what_it_had_queued() {
        let (bridge, handle) = PipelineBridge::new("bridge", PipelineBridgeOptions::default());
        // No pipeline, so nothing drains and what is queued stays queued.
        let shared = Arc::clone(&bridge.shared);
        let mut first = handle.connect().expect("the bridge is alive");

        first.consume(packet(1)).expect("queued behind nothing");
        assert_eq!(shared.state.lock().unwrap().buffers.len(), 1);

        let _second = handle.connect().expect("replacing a live input");

        assert!(
            shared.state.lock().unwrap().buffers.is_empty(),
            "an abandoned timeline's buffers must not reach the new input's reader"
        );
        drop(bridge);
    }

    /// Falling behind is what `DropNewest` exists to absorb, so the feeding
    /// side is not told — but something has to say it happened, and the bus
    /// that can is the bridge's own.
    #[test]
    fn dropping_under_the_newest_policy_is_reported_on_the_bridges_own_bus() {
        let (bridge, handle) = PipelineBridge::new(
            "bridge",
            PipelineBridgeOptions {
                depth: 1,
                policy: OverflowPolicy::DropNewest,
            },
        );
        // No pipeline: nothing drains, so the second buffer has nowhere to go.
        let shared = Arc::clone(&bridge.shared);
        let mut input = handle.connect().expect("the bridge is alive");

        input.consume(packet(1)).expect("fills the one slot");
        input
            .consume(packet(2))
            .expect("dropped, and not an error here");

        assert_eq!(
            shared.dropped.load(Ordering::Relaxed),
            1,
            "the drop is counted for the run loop to report"
        );
        drop(bridge);
    }
}
