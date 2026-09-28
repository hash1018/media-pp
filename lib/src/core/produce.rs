//! Writing a source by what it makes alone — see [`Produce`].
//!
//! A source written directly, as a [`SourceElement`], runs its own loop:
//! it takes its control between buffers and hands each request on the one
//! way every source must, keeps a pause out of the schedule it makes things
//! on, and ends its stream. Seventeen did, each a little differently —
//! docs/stream-events.md lists what that cost. A [`Produce`] makes the next
//! thing when asked; the framework runs the loop.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::OutputContract,
    control::{ChannelGone, ControlReceiver, handle_request},
    element::{Context, Element, ElementType, Source, SourceElement},
    error::Result,
    graph::ElementId,
    pad::SrcPad,
    pp_log::{PpLog, pp_info},
};

/// A source written as what it makes alone: asked for the next thing, it
/// makes it.
///
/// The framework does the rest — the pad, the loop that asks, what the
/// pipeline asks of the thread in between (a pause, a stop, the end of the
/// stream it asks for), keeping a pause out of the time the source makes
/// things on, and the end of the stream once the source has made its last.
/// A source that has to wait — for its next tick, for a device — waits
/// through [`Wait`], which lets go as soon as the pipeline has something
/// for the thread, and is asked again after.
///
/// Put in a pipeline as any source is, with
/// [`Pipeline::new`](crate::pipeline::Pipeline::new) or
/// [`PipelineBuilder::add_source`](crate::pipeline::PipelineBuilder::add_source),
/// which take a `Produce` as they take a [`SourceElement`] (see
/// [`IntoSource`]). One output, no seeking: a source with several outputs
/// or one that can be sought is written the direct way.
pub trait Produce: Element {
    /// Whether it makes what it makes at a rate of its own — a camera, a
    /// clock of its own — rather than as fast as it is asked: see
    /// [`SourceElement::is_live`].
    fn is_live(&self) -> bool;

    /// The next thing it makes. Blocks, where it has to, only through
    /// `wait`; let go, it answers [`Produced::Nothing`] and is asked again.
    ///
    /// An `Err` ends the source, as one [`SourceElement::run`] returns
    /// does. What goes wrong with one part of what it makes and not the
    /// rest — a compositor's layer it cannot draw — is posted on the bus
    /// its [`Context`] carries, kept from
    /// [`attach_context`](Element::attach_context), and it goes on.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced>;

    /// What it hands on — see [`SrcPad::with_contract`]. Nothing said by
    /// default.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Unknown
    }

    /// Stops what it reads from while the pipeline is paused — see
    /// [`SourceElement::pausing`]. Nothing by default.
    fn pausing(&mut self) -> Result<()> {
        Ok(())
    }

    /// Starts it again once playback goes on — see
    /// [`SourceElement::resuming`]. Nothing by default.
    fn resuming(&mut self) -> Result<()> {
        Ok(())
    }

    /// Sets up what has to live on the source's own thread — an apartment
    /// joined, a device started — there, before it is asked for anything.
    /// An `Err` ends the source before it began, and [`Self::stopping`] is
    /// not called for it. Nothing by default.
    fn starting(&mut self) -> Result<()> {
        Ok(())
    }

    /// Lets go, on the same thread, of what [`Self::starting`] set up, once
    /// the source will be asked for nothing more — stopped, at its end, or
    /// failed. What fails here is the source's own to log. Nothing by
    /// default.
    fn stopping(&mut self) {}
}

/// What a [`Produce`] made when asked.
pub enum Produced {
    /// A buffer, to go on.
    Buffer(MediaBuffer),
    /// Nothing this time — a wait let go, or nothing was due. Asked again
    /// once the pipeline has had the thread.
    Nothing,
    /// The end of its stream: the framework hands the end on, and the
    /// source is not asked again.
    End,
}

/// How a [`Produce`] waits: on a clock that stands still while the
/// pipeline is paused, and letting go as soon as the pipeline has something
/// for the thread.
///
/// The clock is what makes a pause cost a source nothing: something made
/// every so often keeps its schedule on [`Self::now`], and after a pause
/// the schedule is where it was, rather than owing the pause's worth all
/// at once.
pub struct Wait<'a> {
    /// Where the pipeline's requests come — `None` once every sender is
    /// gone, a source driven by hand with nothing left to ask it anything,
    /// which a wait then no longer lets go for.
    control: Option<&'a ControlReceiver>,
    paused: Duration,
}

impl Wait<'_> {
    /// Now, on a clock that stands still while the pipeline is paused.
    pub fn now(&self) -> Instant {
        let now = Instant::now();
        now.checked_sub(self.paused).unwrap_or(now)
    }

    /// Waits until `at`, on the clock [`Self::now`] reads. `true` once it
    /// is there; `false`, let go before, where the pipeline has something
    /// for this thread — answer [`Produced::Nothing`] then, and be asked
    /// again.
    pub fn until(&mut self, at: Instant) -> bool {
        let Some(control) = self.control else {
            std::thread::sleep(at.saturating_duration_since(self.now()));
            return true;
        };
        let mut ready = crossbeam_channel::Select::new();
        ready.recv(&control.rx);
        loop {
            let left = at.saturating_duration_since(self.now());
            if left.is_zero() {
                return true;
            }
            if ready.ready_timeout(left).is_ok() {
                return false;
            }
        }
    }

    /// Waits for `rx`'s next message the same way: [`Received::Got`] it,
    /// [`Received::LetGo`] where the pipeline has something for this thread
    /// first, [`Received::Gone`] once every sender has been dropped.
    ///
    /// The pipeline's request is only looked at, never taken: taking it is
    /// the framework's, once this has let go.
    pub(crate) fn recv<T>(&mut self, rx: &crossbeam_channel::Receiver<T>) -> Received<T> {
        match self.receive(rx, None) {
            Some(received) => received,
            None => unreachable!("a receive with no deadline ends only with an answer"),
        }
    }

    /// The same, giving up at `at` on [`Self::now`]'s clock: `None` once
    /// it is there with nothing received. A message already waiting when
    /// it comes is still received.
    #[cfg(all(target_os = "windows", feature = "wgc-capture"))]
    pub(crate) fn recv_until<T>(
        &mut self,
        rx: &crossbeam_channel::Receiver<T>,
        at: Instant,
    ) -> Option<Received<T>> {
        self.receive(rx, Some(at))
    }

    fn receive<T>(
        &mut self,
        rx: &crossbeam_channel::Receiver<T>,
        at: Option<Instant>,
    ) -> Option<Received<T>> {
        use crossbeam_channel::{RecvTimeoutError, TryRecvError};

        let Some(requests) = self.control else {
            let got = match at {
                None => rx.recv().ok(),
                Some(at) => match rx.recv_timeout(at.saturating_duration_since(self.now())) {
                    Ok(got) => Some(got),
                    Err(RecvTimeoutError::Timeout) => return None,
                    Err(RecvTimeoutError::Disconnected) => None,
                },
            };
            return Some(got.map_or(Received::Gone, Received::Got));
        };
        let mut ready = crossbeam_channel::Select::new();
        let control = ready.recv(&requests.rx);
        ready.recv(rx);
        loop {
            let index = match at {
                None => ready.ready(),
                Some(at) => match ready.ready_timeout(at.saturating_duration_since(self.now())) {
                    Ok(index) => index,
                    Err(_) if self.now() >= at => return None,
                    Err(_) => continue,
                },
            };
            if index == control {
                return Some(Received::LetGo);
            }
            match rx.try_recv() {
                Ok(got) => return Some(Received::Got(got)),
                Err(TryRecvError::Disconnected) => return Some(Received::Gone),
                // Taken by another receiver in between, or woken for
                // nothing: look again.
                Err(TryRecvError::Empty) => {}
            }
        }
    }
}

/// What [`Wait::recv`] answers.
pub(crate) enum Received<T> {
    Got(T),
    LetGo,
    Gone,
}

/// What a [`Produce`] is in a pipeline: the source the framework makes of
/// it, with the pad and the loop. Nothing to call on it — it is what
/// [`IntoSource`] hands a pipeline, and what the pipeline's wiring is given.
pub struct ProducingSource<P> {
    inner: P,
    pad: SrcPad,
    /// Time spent paused so far, which [`Wait::now`] leaves out.
    paused: Duration,
}

impl<P: Produce> ProducingSource<P> {
    pub(crate) fn new(inner: P) -> Self {
        let pad = SrcPad::with_contract(format!("{}_src", inner.name()), inner.output_contract());
        Self {
            inner,
            pad,
            paused: Duration::ZERO,
        }
    }

    /// The source itself — for the element of this crate that is a newtype
    /// over what the framework makes of it, to reach what it is made of.
    pub(crate) fn inner(&self) -> &P {
        &self.inner
    }

    /// The same, to change — for a test that drives what it is made of by
    /// hand.
    #[cfg(test)]
    pub(crate) fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    fn report(&self, bus: &Bus, error: crate::error::Error) {
        bus.post(
            self.inner.pp_log(),
            BusEvent::Error {
                element_type: self.inner.element_type(),
                name: self.inner.name(),
                error,
            },
        );
    }
}

impl<P: Produce> Element for ProducingSource<P> {
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        self.inner.element_type()
    }

    fn graph_id(&self) -> Option<ElementId> {
        self.inner.graph_id()
    }

    fn pp_log(&self) -> &PpLog {
        self.inner.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.inner.pp_log_mut()
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        self.inner.attach_context(context);
    }
}

impl<P: Produce> Source for ProducingSource<P> {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl<P: Produce> SourceElement for ProducingSource<P> {
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }

    /// Asks for the next thing, and hands on what it is, until the stream
    /// ends or the pipeline stops it — taking what the pipeline asks
    /// between one and the next, the one way every source does
    /// (`handle_request`). A buffer the pad refuses is reported, and the
    /// source goes on; a failure to make one ends it. What it sets up on
    /// this thread first it lets go of here last, however the loop ended
    /// ([`Produce::starting`], [`Produce::stopping`]).
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(pp_log: self.inner.pp_log(), "started");
        self.inner.starting()?;
        let ended = self.ask(control, bus);
        self.inner.stopping();
        ended
    }

    fn pausing(&mut self) -> Result<()> {
        self.inner.pausing()
    }

    fn resuming(&mut self) -> Result<()> {
        self.inner.resuming()
    }
}

impl<P: Produce> ProducingSource<P> {
    /// The loop itself: [`SourceElement::run`] between the source's own
    /// setting up and letting go.
    fn ask(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            // What the pipeline asks first, each request the one way every
            // source takes it (`handle_request`) — and whether anything is
            // left to ask, which a wait needs to know not to let go for a
            // channel nobody sends on.
            let mut gone = false;
            loop {
                let (request, ack) = match control.try_take() {
                    Ok(Some(request)) => request,
                    Ok(None) => break,
                    Err(ChannelGone) => {
                        gone = true;
                        break;
                    }
                };
                let outcome = handle_request(control, self, bus, request, ack)?;
                self.paused += outcome.paused_for;
                if outcome.stopped {
                    pp_info!(pp_log: self.inner.pp_log(), "stopped");
                    return Ok(());
                }
            }
            let mut wait = Wait {
                control: (!gone).then_some(control),
                paused: self.paused,
            };
            match self.inner.produce(&mut wait)? {
                Produced::Buffer(buf) => {
                    if let Err(error) = self.pad.push(buf) {
                        self.report(bus, error);
                    }
                }
                Produced::Nothing => {}
                Produced::End => {
                    pp_info!(
                        pp_log: self.inner.pp_log(),
                        "event=eos phase=source_completed outcome=ok"
                    );
                    let pp_log = self.inner.pp_log().clone();
                    if let Err(error) = self.pad.push_eos(&pp_log) {
                        self.report(bus, error);
                    }
                    return Ok(());
                }
            }
        }
    }
}

/// What can be put where a source is asked for: a [`SourceElement`]
/// itself, or a [`Produce`], which the framework makes one of.
///
/// `M` says which of the two a type is, and the compiler works it out:
/// nothing names it. A type that is both is refused as ambiguous —
/// implement one or the other.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is neither a `SourceElement` nor a `Produce`",
    note = "implement `Produce` for a source that makes one thing at a time on one output, or `SourceElement` for one that runs its own loop"
)]
pub trait IntoSource<M>: sealed::Sealed<M> {
    /// The source a pipeline is given — `Self` for a [`SourceElement`], a
    /// [`ProducingSource`] for a [`Produce`] — and what its wiring is
    /// handed.
    type Source: SourceElement + 'static;

    /// The source this is.
    fn into_source(self) -> Self::Source;
}

/// Says a type goes in as the [`SourceElement`] it is — see [`IntoSource`].
pub enum AsSourceElement {}

/// Says a type goes in as the [`Produce`] it is — see [`IntoSource`].
pub enum AsProduce {}

impl<S: SourceElement + 'static> IntoSource<AsSourceElement> for S {
    type Source = S;

    fn into_source(self) -> S {
        self
    }
}

impl<P: Produce + 'static> IntoSource<AsProduce> for P {
    type Source = ProducingSource<P>;

    fn into_source(self) -> ProducingSource<P> {
        ProducingSource::new(self)
    }
}

mod sealed {
    use super::{AsProduce, AsSourceElement, Produce, SourceElement};

    /// Keeps [`super::IntoSource`] to the two ways in it has.
    pub trait Sealed<M> {}

    impl<S: SourceElement + 'static> Sealed<AsSourceElement> for S {}
    impl<P: Produce + 'static> Sealed<AsProduce> for P {}
}

/// Makes `$name`, a newtype over `ProducingSource<_>`, the source it is —
/// every method its own. For an element of this crate that keeps the public
/// name and constructors it always had while its work moves into a
/// [`Produce`].
macro_rules! produce_source {
    ($name:ident) => {
        impl $crate::element::Element for $name {
            fn name(&self) -> ::std::sync::Arc<str> {
                self.0.name()
            }
            fn element_type(&self) -> $crate::element::ElementType {
                self.0.element_type()
            }
            fn graph_id(&self) -> Option<$crate::graph::ElementId> {
                self.0.graph_id()
            }
            fn pp_log(&self) -> &$crate::pp_log::PpLog {
                self.0.pp_log()
            }
            fn pp_log_mut(&mut self) -> &mut $crate::pp_log::PpLog {
                self.0.pp_log_mut()
            }
            fn attach_context(&mut self, context: &::std::sync::Arc<$crate::element::Context>) {
                self.0.attach_context(context);
            }
        }

        impl $crate::element::Source for $name {
            fn src_pads(&mut self) -> &mut [$crate::pad::SrcPad] {
                self.0.src_pads()
            }
        }

        impl $crate::element::SourceElement for $name {
            fn is_live(&self) -> bool {
                self.0.is_live()
            }
            fn run(
                &mut self,
                control: &$crate::control::ControlReceiver,
                bus: &$crate::bus::Bus,
            ) -> $crate::error::Result<()> {
                self.0.run(control, bus)
            }
            fn pausing(&mut self) -> $crate::error::Result<()> {
                self.0.pausing()
            }
            fn resuming(&mut self) -> $crate::error::Result<()> {
                self.0.resuming()
            }
        }
    };
}

pub(crate) use produce_source;

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Mutex,
            atomic::{AtomicBool, Ordering},
        },
        thread,
    };

    use ffmpeg_next as ffmpeg;

    use super::*;
    use crate::element::{Sink, element_pp_log};
    use crate::pipeline::Pipeline;

    /// Makes `left` packets, one every `every` on the clock a pause does
    /// not move, then ends.
    struct Ticks {
        pp_log: PpLog,
        every: Duration,
        left: usize,
        next: Option<Instant>,
    }

    impl Ticks {
        fn new(every: Duration, left: usize) -> Self {
            Self {
                pp_log: element_pp_log(ElementType::Other, "ticks", None),
                every,
                left,
                next: None,
            }
        }
    }

    impl Element for Ticks {
        fn name(&self) -> Arc<str> {
            "ticks".into()
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

    impl Produce for Ticks {
        fn is_live(&self) -> bool {
            true
        }

        fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
            if self.left == 0 {
                return Ok(Produced::End);
            }
            let every = self.every;
            let next = *self.next.get_or_insert_with(|| wait.now() + every);
            if !wait.until(next) {
                return Ok(Produced::Nothing);
            }
            self.next = Some(next + every);
            self.left -= 1;
            Ok(Produced::Buffer(MediaBuffer::Packet(Arc::new(
                ffmpeg::Packet::copy(&[self.left as u8]),
            ))))
        }
    }

    /// When each buffer arrived, and whether the end did.
    #[derive(Default)]
    struct Arrivals {
        at: Mutex<Vec<Instant>>,
        ended: AtomicBool,
    }

    struct Arriving(PpLog, Arc<Arrivals>);

    impl Element for Arriving {
        fn name(&self) -> Arc<str> {
            "arriving".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.0
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.0
        }
    }

    impl Sink for Arriving {
        fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
            self.1.at.lock().unwrap().push(Instant::now());
            Ok(())
        }
        fn stream_event(&mut self, event: &crate::stream::StreamEvent) -> Result<()> {
            if let crate::stream::StreamEvent::Eos = event {
                self.1.ended.store(true, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    fn running(ticks: Ticks) -> (Arc<Pipeline>, Arc<Arrivals>) {
        let arrivals = Arc::new(Arrivals::default());
        let sink = Arriving(
            element_pp_log(ElementType::Other, "arriving", None),
            Arc::clone(&arrivals),
        );
        let (pipeline, ()) = Pipeline::new("produce", ticks, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring succeeds");
        pipeline.run().expect("run");
        (pipeline, arrivals)
    }

    fn wait_for(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "{what}");
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// A `Produce` goes where a source does, and the framework runs it:
    /// everything it makes goes on, then the end of its stream.
    #[test]
    fn a_produce_is_a_source_the_framework_runs() {
        let (pipeline, arrivals) = running(Ticks::new(Duration::from_millis(5), 4));
        wait_for("the stream did not end", || {
            arrivals.ended.load(Ordering::SeqCst)
        });
        assert_eq!(arrivals.at.lock().unwrap().len(), 4);
        pipeline.stop();
        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
    }

    /// A pause reaches a source in the middle of a long wait at once — the
    /// wait lets go — and costs its schedule nothing: the next buffer comes
    /// as long after the last as it would have without the pause, on top
    /// of the pause, not the moment playback goes on.
    #[test]
    fn a_pause_lets_a_wait_go_and_costs_the_schedule_nothing() {
        let every = Duration::from_millis(300);
        let (pipeline, arrivals) = running(Ticks::new(every, 3));
        wait_for("nothing arrived", || {
            !arrivals.at.lock().unwrap().is_empty()
        });

        let pausing = Instant::now();
        pipeline.pause();
        assert!(
            pausing.elapsed() < Duration::from_millis(200),
            "the pause waited out the source's wait: {:?}",
            pausing.elapsed()
        );
        let paused = Duration::from_millis(500);
        thread::sleep(paused);
        pipeline.resume();
        wait_for("nothing arrived after the pause", || {
            arrivals.at.lock().unwrap().len() >= 2
        });
        pipeline.stop();

        let at = arrivals.at.lock().unwrap();
        let gap = at[1].duration_since(at[0]);
        assert!(
            gap >= paused + every / 2,
            "the next came {gap:?} after the last: owed at once once playback went on"
        );
    }

    /// What a `Hooked` does once it has started.
    #[derive(Clone, Copy, Debug)]
    enum Then {
        End,
        Fail,
        WaitForStop,
    }

    /// Notes, with the thread it was on, each hook and each asking.
    struct Hooked {
        pp_log: PpLog,
        cannot_start: bool,
        then: Then,
        seen: Arc<Mutex<Vec<(&'static str, thread::ThreadId)>>>,
    }

    impl Hooked {
        fn note(&self, what: &'static str) {
            self.seen
                .lock()
                .unwrap()
                .push((what, thread::current().id()));
        }
    }

    impl Element for Hooked {
        fn name(&self) -> Arc<str> {
            "hooked".into()
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

    impl Produce for Hooked {
        fn is_live(&self) -> bool {
            true
        }

        fn starting(&mut self) -> Result<()> {
            self.note("starting");
            if self.cannot_start {
                return Err(crate::error::Error::Other("no device".into()));
            }
            Ok(())
        }

        fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
            self.note("produce");
            match self.then {
                Then::End => Ok(Produced::End),
                Then::Fail => Err(crate::error::Error::Other("device lost".into())),
                Then::WaitForStop => {
                    let later = wait.now() + Duration::from_secs(60);
                    wait.until(later);
                    Ok(Produced::Nothing)
                }
            }
        }

        fn stopping(&mut self) {
            self.note("stopping");
        }
    }

    /// Runs a `Hooked` until it has let go of what it set up, or — where it
    /// could not start — until its failure is reported, and answers what it
    /// saw.
    fn hooked(cannot_start: bool, then: Then) -> Vec<(&'static str, thread::ThreadId)> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let source = Hooked {
            pp_log: element_pp_log(ElementType::Other, "hooked", None),
            cannot_start,
            then,
            seen: Arc::clone(&seen),
        };
        let sink = Arriving(
            element_pp_log(ElementType::Other, "arriving", None),
            Arc::new(Arrivals::default()),
        );
        let (pipeline, ()) = Pipeline::new("hooks", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring succeeds");
        pipeline.run().expect("run");
        let noted = |what| seen.lock().unwrap().iter().any(|(seen, _)| *seen == what);
        if cannot_start {
            wait_for("a failed start was not reported", || {
                pipeline
                    .bus()
                    .try_recv()
                    .is_some_and(|event| matches!(event, BusEvent::Error { .. }))
            });
        } else {
            wait_for("it was never asked", || noted("produce"));
        }
        pipeline.stop();
        seen.lock().unwrap().clone()
    }

    /// What a source sets up on its own thread it lets go of there, once,
    /// after everything it was asked — however the asking ended; and one
    /// that could not start is neither asked nor let go of.
    #[test]
    fn a_source_starts_and_stops_on_its_own_thread_around_everything_it_is_asked() {
        for then in [Then::End, Then::Fail, Then::WaitForStop] {
            let seen = hooked(false, then);
            let what: Vec<_> = seen.iter().map(|(what, _)| *what).collect();
            assert_eq!(what.first(), Some(&"starting"), "{then:?}: {what:?}");
            assert_eq!(what.last(), Some(&"stopping"), "{then:?}: {what:?}");
            assert_eq!(
                what.iter().filter(|what| **what == "stopping").count(),
                1,
                "{then:?}: {what:?}"
            );
            let (_, thread) = seen[0];
            assert_ne!(thread, thread::current().id(), "{then:?}");
            assert!(
                seen.iter().all(|(_, on)| *on == thread),
                "{then:?}: not all on the source's thread"
            );
        }
        let what: Vec<_> = hooked(true, Then::End)
            .into_iter()
            .map(|(what, _)| what)
            .collect();
        assert_eq!(what, ["starting"]);
    }

    /// A receive with a deadline gives up there, takes a message already
    /// waiting even once the deadline has come, and lets go for a request
    /// the pipeline sends before either.
    #[test]
    #[cfg(all(target_os = "windows", feature = "wgc-capture"))]
    fn a_receive_gives_up_at_its_deadline_and_lets_go_for_a_request() {
        let (control_tx, control) = crate::control::channel();
        let (tx, rx) = crossbeam_channel::unbounded::<u8>();
        let mut wait = Wait {
            control: Some(&control),
            paused: Duration::ZERO,
        };

        let asked = Instant::now();
        assert!(
            wait.recv_until(&rx, wait.now() + Duration::from_millis(30))
                .is_none()
        );
        assert!(asked.elapsed() >= Duration::from_millis(25));

        tx.send(7).unwrap();
        let past = wait.now();
        assert!(matches!(wait.recv_until(&rx, past), Some(Received::Got(7))));

        let _acknowledged = control_tx.enqueue(crate::control::ControlMsg::Pause);
        assert!(matches!(
            wait.recv_until(&rx, wait.now() + Duration::from_secs(60)),
            Some(Received::LetGo)
        ));
    }
}
