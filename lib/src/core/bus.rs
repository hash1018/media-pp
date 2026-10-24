//! Out-of-band reporting for what cannot be returned to the caller.
//!
//! Inside a single thread a failing stage propagates with `?`. Once a buffer
//! has crossed a [`Queue`](crate::queue::Queue) there is no longer a caller
//! to return to, so the element posts a [`BusEvent`] here instead and keeps
//! running. Whoever owns the pipeline observes those events through
//! [`BusReceiver`].
//!
//! Every event is delivered as a [`BusMessage`] carrying the stable graph
//! identity of the element that posted it, so a report can be attributed to
//! one branch even when several elements share a name.

use std::{
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::RecvTimeoutError,
    },
    task::{Poll, Waker},
    time::Duration,
};

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::{element::ElementType, error::Error, graph::ElementId};

/// Something that happened in a running pipeline: an element ended, failed,
/// dropped a buffer or was sought, or the whole pipeline finished.
///
/// Read them from [`Pipeline::bus`](crate::pipeline::Pipeline::bus). Each
/// prints as one line naming the element. A pipeline that has played to
/// its end posts [`BusEvent::Finished`]; that is where to stop it.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::bus::BusEvent;
/// use media_pp::elements::{FrameCounter, TestVideoOptions, TestVideoSource};
/// use media_pp::pipeline::Pipeline;
///
/// let source = TestVideoSource::new("source", TestVideoOptions::default());
/// let (counter, _pictures) = FrameCounter::new("counter");
/// let (pipeline, ()) = Pipeline::new("watch", source, |source, ctx| {
///     ctx.attach(source, 0, ctx.branch().to(counter)?)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// for event in pipeline.bus().iter() {
///     match event {
///         BusEvent::Error { name, error, .. } => eprintln!("{name} failed: {error}"),
///         BusEvent::Finished => pipeline.stop(),
///         other => println!("{other}"),
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Non-exhaustive
///
/// Marked `#[non_exhaustive]`: this grows as elements gain conditions worth
/// reporting, and a caller acting on `Eos`/`Error` should not stop compiling
/// because some unrelated element learned to report a stall. Within this crate
/// the attribute has no effect, so [`Bus::post`] and
/// [`BusReceiver::log_events`] still fail to compile until they handle a new
/// variant — the completeness check stays where it belongs.
#[derive(Debug)]
#[non_exhaustive]
pub enum BusEvent {
    /// An element completed ordered end-of-stream processing.
    ///
    /// Every element that does says so — a `Queue` as well as the muxer
    /// after it — so one of these is not the pipeline ending. Watch
    /// [`BusEvent::Finished`] for that.
    Eos {
        /// Built-in kind of the element that completed.
        element_type: ElementType,
        /// Caller-selected instance name of the element that completed.
        name: Arc<str>,
    },

    /// An element encountered a failure that could not be returned through
    /// the synchronous call stack.
    Error {
        /// Built-in kind of the element reporting the failure.
        element_type: ElementType,
        /// Caller-selected instance name of the element reporting the failure.
        name: Arc<str>,
        /// Typed crate or component error reported by the element.
        error: Error,
    },
    /// A `Queue` with `OverflowPolicy::DropNewest` dropped a buffer
    /// because it was full.
    Dropped {
        /// Built-in kind of the queue that dropped the buffer.
        element_type: ElementType,
        /// Caller-selected instance name of the queue that dropped the buffer.
        name: Arc<str>,
    },
    /// Posted by the source's thread once
    /// [`crate::element::SeekableSource::seek`] returns — `requested` is
    /// whatever [`crate::pipeline::Pipeline::seek`] was called with;
    /// `landed` is where the source actually ended up, which the source
    /// itself has to resolve (e.g. `FileDemuxer` can only reposition to a
    /// keyframe at or before `requested`, never exactly on top of an
    /// arbitrary timestamp — see its `seek` impl). Watch this instead of
    /// assuming `requested` took effect verbatim.
    Seeked {
        /// Built-in kind of the source that performed the seek.
        element_type: ElementType,
        /// Caller-selected instance name of the source that performed the seek.
        name: Arc<str>,
        /// Absolute media position requested by [`crate::pipeline::Pipeline::seek`].
        requested: Duration,
        /// Absolute media position at which the source actually resumed.
        landed: Duration,
    },

    /// Every terminal sink in the pipeline has accepted end-of-stream: the
    /// muxers have written their trailers, the renderers have taken their
    /// last frame. Posted once per stream, after the last terminal's own
    /// [`BusEvent::Eos`], and posted again only after a seek has sent a new
    /// stream through — so this, not the first `Eos`, is when a pipeline
    /// played to its end can be stopped.
    ///
    /// Terminals are counted as the graph stands when each one ends: a
    /// branch a `Tee` detaches no longer has to, and one it finishes counts
    /// once its own `Eos` arrives. It is not posted for a terminal that
    /// failed its `Eos` — that failure is a [`BusEvent::Error`] — nor for a
    /// source that stopped on an error, whose branches are stopped rather
    /// than ended. Posted by the pipeline, not an element, so its
    /// [`BusMessage::element_id`] is `None`.
    ///
    /// It follows the terminals' own `Eos`, not every element's: a `Queue`
    /// says it has ended once the terminal behind it has taken its `Eos`,
    /// which is also when this is posted, so a queue's `Eos` often arrives
    /// just after it. Stopping on this, as a pipeline played to its end
    /// should be, means those later `Eos` may never be read — which is why
    /// what has ended is this, and not a count of `Eos`.
    Finished,
}

/// One line saying what happened and to which element: `[name] eos`,
/// `[name] error: ...`, `[name] dropped a buffer (queue full)`,
/// `[name] seeked: requested ..., landed ...`, and `finished` for the
/// pipeline itself. What a program that only reports its bus prints —
/// `println!("{event}")` — and what [`BusReceiver::log_events`] prints.
impl fmt::Display for BusEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BusEvent::Eos { name, .. } => write!(f, "[{name}] eos"),
            BusEvent::Error { name, error, .. } => write!(f, "[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => write!(f, "[{name}] dropped a buffer (queue full)"),
            BusEvent::Seeked {
                name,
                requested,
                landed,
                ..
            } => write!(
                f,
                "[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"
            ),
            BusEvent::Finished => write!(f, "finished"),
        }
    }
}

/// Logs a failure under the identity of the element that failed — the one
/// the event names — rather than only the one posting it. A queue, a tee or
/// a demuxer posts what went wrong after it, and a record filed under the
/// poster alone named the queue where the bus named the sink. The poster is
/// kept in the message, as the place the failure stopped being returnable.
fn log_error(pp_log: &PpLog, element_type: ElementType, name: &str, error: &Error) {
    if !crate::log::enabled(crate::log::Level::Error) {
        return;
    }
    let element = format!("{element_type:?}");
    if pp_log.element() == element && pp_log.name() == name {
        pp_error!(pp_log: pp_log, "{error}");
    } else {
        let failed = PpLog::new(&element, name, pp_log.pipeline_id());
        pp_error!(
            pp_log: &failed,
            "{error} (reported by {}({}))",
            pp_log.element(),
            pp_log.name()
        );
    }
}

/// The sending half of a pipeline's event channel, which elements post
/// [`BusEvent`]s on.
///
/// A program reads the other half, [`BusReceiver`], from
/// [`Pipeline::bus`](crate::pipeline::Pipeline::bus); an element of its own
/// takes this one from the pipeline's [`Context`](crate::element::Context).
///
/// Cross-thread event channel. Once a buffer crosses a `Queue` boundary,
/// errors can no longer be propagated up the call stack with `?` — they're
/// posted here instead so the owner of the `Pipeline` can observe them.
#[derive(Clone)]
pub struct Bus {
    tx: Sender<BusMessage>,
    element_id: Option<ElementId>,
    /// After `tx`, so it drops after it: the last sender wakes an async
    /// wait only once the channel already reads as disconnected.
    sender: Sending,
}

/// The receiving half of a [`Bus`], held by whoever owns the pipeline.
///
/// Draining it blocks until every `Bus` sender has been dropped — once the
/// pipeline has stopped, or once every source has ended of itself. A source
/// that can be sought back does not end at the end of its media: a
/// `FileDemuxer` stays there until stopped. What says a pipeline has played
/// everything is [`BusEvent::Finished`], and the usual loop stops it there.
pub struct BusReceiver {
    rx: Receiver<BusMessage>,
    waiting: Arc<Waiting>,
}

/// What a [`BusReceiver::recv_async`] waits on: the wakers of those waiting,
/// and how many [`Bus`] senders are left to wake them.
///
/// crossbeam's channel has no async side, and says nothing when its last
/// sender goes, so the senders are counted here beside it.
#[derive(Default)]
struct Waiting {
    wakers: Mutex<Vec<Waker>>,
    senders: AtomicUsize,
}

impl Waiting {
    fn wake(&self) {
        let wakers = std::mem::take(&mut *self.wakers.lock().unwrap_or_else(|p| p.into_inner()));
        for waker in wakers {
            waker.wake();
        }
    }
}

/// One [`Bus`] sender's part in [`Waiting::senders`]: counted as it is
/// made or cloned, and waking every waiter as the last one goes, since
/// nothing will be posted after it.
struct Sending(Arc<Waiting>);

impl Sending {
    fn new(waiting: Arc<Waiting>) -> Self {
        waiting.senders.fetch_add(1, Ordering::AcqRel);
        Self(waiting)
    }
}

impl Clone for Sending {
    fn clone(&self) -> Self {
        Self::new(Arc::clone(&self.0))
    }
}

impl Drop for Sending {
    fn drop(&mut self) {
        if self.0.senders.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.wake();
        }
    }
}

/// One bus event together with the stable graph identity of the element
/// that posted it. Drivers and standalone elements that do not belong to a
/// `PipelineGraph` use `None`.
#[derive(Debug)]
pub struct BusMessage {
    /// Stable graph identity of the posting element, or `None` for a driver or
    /// standalone element outside a pipeline graph.
    pub element_id: Option<ElementId>,

    /// Event payload posted by the element.
    pub event: BusEvent,
}

impl Bus {
    /// Creates an unbounded event channel with no graph element identity.
    ///
    /// Pipeline construction derives element-specific senders internally.
    /// Standalone elements and drivers can use the returned sender directly;
    /// their [`BusMessage::element_id`] remains `None`.
    pub fn new() -> (Bus, BusReceiver) {
        let (tx, rx) = unbounded();
        let waiting = Arc::new(Waiting::default());
        (
            Bus {
                tx,
                element_id: None,
                sender: Sending::new(Arc::clone(&waiting)),
            },
            BusReceiver { rx, waiting },
        )
    }

    /// The same channel, posting as the pipeline rather than as any one
    /// element of it.
    pub(crate) fn for_pipeline(&self) -> Bus {
        Bus {
            element_id: None,
            ..self.clone()
        }
    }

    pub(crate) fn for_element(&self, element_id: ElementId) -> Bus {
        Bus {
            element_id: Some(element_id),
            ..self.clone()
        }
    }

    /// Posts a failure that arrived from somewhere downstream, attributed
    /// to whichever element raised it.
    ///
    /// `element_type`/`name` are the reporter's own and are used only when
    /// the error carries no origin — which is when it never crossed a chain
    /// stage, so there is nobody else to name.
    ///
    /// Every element that pushes into a pad and reports what comes back
    /// should use this rather than [`Bus::post`] with its own identity. A
    /// queue, a demuxer, a bridge and a WebRTC track are all only the place
    /// a failure stopped being returnable; none of them is what failed, and
    /// saying so loses the one thing an observer needs — see
    /// [`Error::traced_at`](crate::error::Error::traced_at).
    pub fn post_downstream_error(
        &self,
        pp_log: &PpLog,
        element_type: ElementType,
        name: Arc<str>,
        error: Error,
    ) {
        let (element_type, name) = match error.origin() {
            Some(origin) => (origin.element_type, origin.name.clone()),
            None => (element_type, name),
        };
        self.post(
            pp_log,
            BusEvent::Error {
                element_type,
                name,
                error,
            },
        );
    }

    /// Puts `event` on the bus for whoever reads it, and records it in the
    /// log under the identity in `pp_log` — the posting element's own, or,
    /// for an error the event says another element raised, that element's,
    /// so the log names what failed as the bus does.
    ///
    /// Never blocks on the receiver. If the receiving half has already been
    /// dropped, the event is discarded; posting never turns pipeline
    /// teardown into another error.
    ///
    /// `pp_log` — [`crate::element::Element::pp_log`], or
    /// [`Attached::pp_log`](crate::element::Attached::pp_log) for a `Source`,
    /// `Filter` or `Sink` of your own — is used instead of `event`'s own `name`
    /// so the element's full identity, pipeline id included, reaches the log
    /// record rather than just the name carried in the event.
    pub fn post(&self, pp_log: &PpLog, event: BusEvent) {
        // Each `pp_*` macro checks `crate::log::enabled` before evaluating its
        // arguments, so posting to a bus nobody is logging costs no `format!`
        // — no hand-rolled check needed here.
        match &event {
            BusEvent::Eos { .. } => {
                pp_info!(pp_log: pp_log, "event=eos phase=reported")
            }
            BusEvent::Error {
                element_type,
                name,
                error,
            } => log_error(pp_log, *element_type, name, error),
            BusEvent::Dropped { .. } => {
                pp_warn!(pp_log: pp_log, "dropped a buffer (queue full)")
            }
            BusEvent::Seeked {
                requested, landed, ..
            } => pp_info!(pp_log: pp_log, "seeked: requested {requested:.2?}, landed {landed:.2?}"),
            BusEvent::Finished => pp_info!(pp_log: pp_log, "event=finished"),
        }
        // Nothing to do if the receiving end is gone (pipeline dropped).
        let _ = self.tx.send(BusMessage {
            element_id: self.element_id,
            event,
        });
        self.sender.0.wake();
    }
}

impl BusReceiver {
    /// The channel itself, for a caller inside this crate that waits on it
    /// beside another — `Player`, on its window's events and its bus at once.
    ///
    /// Crate-private because it is crossbeam's type: public, the channel
    /// crate would become part of this one's API. A public form would be a
    /// wait on the bus and something else together, which nothing outside
    /// has asked for.
    #[cfg(any(
        all(
            target_os = "windows",
            any(feature = "d3d11", feature = "d3d12"),
            feature = "wasapi-renderer"
        ),
        all(
            target_os = "linux",
            feature = "vulkan",
            feature = "pipewire-audio-renderer"
        ),
        all(target_os = "macos", feature = "metal", feature = "coreaudio-renderer")
    ))]
    pub(crate) fn receiver(&self) -> &Receiver<BusMessage> {
        &self.rx
    }

    /// Blocks until the next event arrives.
    ///
    /// Returns `None` only after every corresponding [`Bus`] sender has been
    /// dropped and all already-queued events have been received. This discards
    /// the posting element's stable graph ID; use [`Self::recv_message`] when
    /// duplicate element names must be distinguished.
    pub fn recv(&self) -> Option<BusEvent> {
        self.recv_message().map(|message| message.event)
    }

    /// Receives one currently queued event without blocking.
    ///
    /// Returns `None` both when the channel is currently empty and when every
    /// sender has disconnected. Use [`Self::try_recv_message`] to retain the
    /// posting element's stable graph ID.
    pub fn try_recv(&self) -> Option<BusEvent> {
        self.try_recv_message().map(|message| message.event)
    }

    /// Waits up to `timeout` for the next event.
    ///
    /// For a loop that has something else to watch as well — a channel of
    /// its own, a deadline — and so can neither block in [`Self::recv`] nor
    /// spin on [`Self::try_recv`]. Unlike those two, it says why it came back
    /// empty: [`RecvTimeoutError::Timeout`] when nothing was posted in time,
    /// [`RecvTimeoutError::Disconnected`] once every sender has dropped and
    /// everything already posted has been received — the pipeline is over
    /// and nothing more will come.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<BusEvent, RecvTimeoutError> {
        self.rx
            .recv_timeout(timeout)
            .map(|message| message.event)
            .map_err(|error| match error {
                crossbeam_channel::RecvTimeoutError::Timeout => RecvTimeoutError::Timeout,
                crossbeam_channel::RecvTimeoutError::Disconnected => RecvTimeoutError::Disconnected,
            })
    }

    /// Iterates over events until every corresponding [`Bus`] sender drops.
    ///
    /// The iterator blocks while the channel is still connected but empty.
    /// It discards stable graph IDs; use [`Self::iter_with_ids`] when duplicate
    /// element names must be distinguished.
    pub fn iter(&self) -> impl Iterator<Item = BusEvent> + '_ {
        self.iter_with_ids().map(|message| message.event)
    }

    /// Blocks until the next event and its stable posting-element ID arrive.
    ///
    /// Returns `None` after the channel disconnects and its queued messages
    /// have been drained.
    pub fn recv_message(&self) -> Option<BusMessage> {
        self.rx.recv().ok()
    }

    /// Waits for the next event without holding a thread, for a caller on an
    /// async runtime — any: nothing here is tied to one.
    ///
    /// `None` once every [`Bus`] sender has gone and what was already posted
    /// has been received, as [`Self::recv`] says it: which a pipeline's bus
    /// does once [`Pipeline::stop`](crate::pipeline::Pipeline::stop) has
    /// returned, so a task waiting here hears of a stop made elsewhere by
    /// the bus ending. Dropping the future gives up the wait and loses no
    /// event.
    pub async fn recv_async(&self) -> Option<BusEvent> {
        std::future::poll_fn(|cx| {
            if let Some(ready) = self.ready() {
                return Poll::Ready(ready);
            }
            self.waiting
                .wakers
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(cx.waker().clone());
            // Looked at again with the waker in place: a post or the last
            // sender going between the first look and now woke nothing.
            match self.ready() {
                Some(ready) => Poll::Ready(ready),
                None => Poll::Pending,
            }
        })
        .await
        .map(|message| message.event)
    }

    /// What a wait has to answer now, if anything: a message, or `None`
    /// for the channel having ended.
    fn ready(&self) -> Option<Option<BusMessage>> {
        match self.rx.try_recv() {
            Ok(message) => Some(Some(message)),
            Err(crossbeam_channel::TryRecvError::Disconnected) => Some(None),
            Err(crossbeam_channel::TryRecvError::Empty) => None,
        }
    }

    /// Receives one currently queued message without blocking.
    ///
    /// Returns `None` for both an empty connected channel and a disconnected
    /// channel.
    pub fn try_recv_message(&self) -> Option<BusMessage> {
        self.rx.try_recv().ok()
    }

    /// Iterates over messages, preserving stable graph element IDs, until all
    /// corresponding [`Bus`] senders have been dropped.
    ///
    /// The iterator blocks while the channel remains connected but empty.
    pub fn iter_with_ids(&self) -> impl Iterator<Item = BusMessage> + '_ {
        self.rx.iter()
    }

    /// Blocks and prints each event as its [`Display`](fmt::Display) says —
    /// errors and dropped buffers to stderr, the rest to stdout — until every
    /// corresponding [`Bus`] sender has been dropped. This consumes both events already
    /// queued and events posted while the call is waiting; use
    /// [`BusReceiver::try_recv`] to drain only what is currently available.
    ///
    /// Convenience for examples and smoke tests; anything that needs to
    /// act on specific events — e.g. deciding whether an `Error` warrants
    /// a [`crate::pipeline::Pipeline::stop`] — should match on `iter()`
    /// directly instead, where `error`'s concrete variant (see
    /// [`crate::error::Error`]) is still available, not just its
    /// `Display` text.
    pub fn log_events(&self) {
        for event in self.iter() {
            match event {
                BusEvent::Error { .. } | BusEvent::Dropped { .. } => eprintln!("{event}"),
                BusEvent::Eos { .. } | BusEvent::Seeked { .. } | BusEvent::Finished => {
                    println!("{event}")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line an example prints for each event is the one `log_events`
    /// always printed, so switching to `println!("{event}")` changes nothing
    /// anyone reads.
    #[test]
    fn each_event_displays_as_one_line_naming_its_element() {
        let name: Arc<str> = "muxer".into();
        let cases = [
            (
                BusEvent::Eos {
                    element_type: ElementType::FileMuxer,
                    name: name.clone(),
                },
                "[muxer] eos",
            ),
            (
                BusEvent::Error {
                    element_type: ElementType::FileMuxer,
                    name: name.clone(),
                    error: Error::Other("disk full".into()),
                },
                "[muxer] error: disk full",
            ),
            (
                BusEvent::Dropped {
                    element_type: ElementType::Queue,
                    name: name.clone(),
                },
                "[muxer] dropped a buffer (queue full)",
            ),
            (
                BusEvent::Seeked {
                    element_type: ElementType::FileDemuxer,
                    name,
                    requested: Duration::from_millis(1500),
                    landed: Duration::from_millis(1000),
                },
                "[muxer] seeked: requested 1.50s, landed 1.00s",
            ),
            (BusEvent::Finished, "finished"),
        ];
        for (event, line) in cases {
            assert_eq!(event.to_string(), line);
        }
    }

    /// `recv_timeout` tells the three outcomes apart: an event, nothing yet,
    /// and nothing ever again — the last only once what was already posted
    /// has been received.
    #[test]
    fn recv_timeout_says_whether_it_timed_out_or_the_bus_ended() {
        let (bus, rx) = Bus::new();
        let pp_log = PpLog::new("Test", "test", None);
        let wait = Duration::from_millis(20);

        assert_eq!(rx.recv_timeout(wait).err(), Some(RecvTimeoutError::Timeout));

        bus.post(&pp_log, BusEvent::Finished);
        drop(bus);
        assert!(matches!(rx.recv_timeout(wait), Ok(BusEvent::Finished)));
        assert_eq!(
            rx.recv_timeout(wait).err(),
            Some(RecvTimeoutError::Disconnected)
        );
    }

    use crate::test_support::block_on;

    /// An async wait is woken by a post from another thread, and by the
    /// last sender going — once what was posted before it has been
    /// received — so a task that waits on a pipeline's bus hears of a stop.
    #[test]
    fn recv_async_is_woken_by_a_post_and_by_the_last_sender_going() {
        let (bus, rx) = Bus::new();
        let pp_log = PpLog::new("Test", "test", None);
        let element = bus.for_element(ElementId::for_test(1));

        let posting = std::thread::spawn({
            let pp_log = pp_log.clone();
            move || {
                std::thread::sleep(Duration::from_millis(50));
                element.post(&pp_log, BusEvent::Finished);
                std::thread::sleep(Duration::from_millis(50));
                // The last senders go: this one, and `bus` below.
                drop(element);
            }
        });
        assert!(matches!(
            block_on(rx.recv_async()),
            Some(BusEvent::Finished)
        ));
        drop(bus);
        posting.join().unwrap();
        assert!(block_on(rx.recv_async()).is_none(), "the bus ended");
    }

    /// What was posted before the last sender went is still received by an
    /// async wait, ahead of the end.
    #[test]
    fn recv_async_hands_over_what_was_posted_before_the_end() {
        let (bus, rx) = Bus::new();
        let pp_log = PpLog::new("Test", "test", None);
        bus.post(&pp_log, BusEvent::Finished);
        drop(bus);
        assert!(matches!(
            block_on(rx.recv_async()),
            Some(BusEvent::Finished)
        ));
        assert!(block_on(rx.recv_async()).is_none());
    }

    /// A pipeline stopped from another thread ends the bus a task waits on:
    /// what a task watching a live pipeline relies on to hear of a stop it
    /// did not make itself.
    #[test]
    fn recv_async_ends_once_the_pipeline_is_stopped() {
        use crate::elements::{FrameCounter, TestVideoOptions, TestVideoSource};
        use crate::pipeline::Pipeline;

        let source = TestVideoSource::new("source", TestVideoOptions::default());
        let (counter, _pictures) = FrameCounter::new("counter");
        let (pipeline, ()) = Pipeline::new("watched", source, |source, ctx| {
            ctx.attach(source, 0, ctx.branch().to(counter)?)?;
            Ok(())
        })
        .unwrap();
        pipeline.run().unwrap();
        let stopping = std::thread::spawn({
            let pipeline = Arc::clone(&pipeline);
            move || {
                std::thread::sleep(Duration::from_millis(100));
                pipeline.stop();
            }
        });
        // Returns at all only because the stop ended the bus.
        block_on(async { while pipeline.bus().recv_async().await.is_some() {} });
        stopping.join().unwrap();
    }
}
