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
    control::{ChannelGone, ControlMsg, ControlReceiver, RequestKind, handle_request},
    element::{
        Context, Element, ElementType, ReversibleSource, SeekableSource, Source, SourceElement,
    },
    error::Result,
    graph::ElementId,
    pad::SrcPad,
    parking::Parking,
    playback_state::PlaybackState,
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
/// [`IntoSource`]). One output by default; a source with several — a
/// stream each, as a demuxer has — says what they are in
/// [`Self::outputs`] and which each buffer is for in [`Produced::On`].
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

    /// The pads it hands on through, in the order [`Produced::On`] numbers
    /// them — asked once, as the framework makes it a source. One by
    /// default, named for the source and declaring
    /// [`Self::output_contract`].
    fn outputs(&self) -> Vec<SrcPad> {
        vec![SrcPad::with_contract(
            format!("{}_src", self.name()),
            self.output_contract(),
        )]
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

    /// This source as a [`SeekableSource`], where it is one — asked as it is
    /// wired, and for each seek, which the framework takes on the source's
    /// thread between one thing made and the next; see
    /// [`SourceElement::as_seekable`]. A source that can be sought also stays
    /// at its end, rather than ending its thread, until it is stopped or
    /// sought back into what it reads. `None` by default.
    fn as_seekable(&mut self) -> Option<&mut dyn SeekableSource> {
        None
    }

    /// This source as a [`ReversibleSource`], where it is one — see
    /// [`SourceElement::as_reversible`]. `None` by default.
    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleSource> {
        None
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
    /// A buffer, to go on through the first output — the one a source with
    /// one output has.
    Buffer(MediaBuffer),
    /// A buffer, to go on through the output of this number — see
    /// [`Produce::outputs`].
    On(usize, MediaBuffer),
    /// A new run of what it makes begins here, through every output ahead
    /// of what it makes next — a [`Segment`](crate::stream::Segment) on the
    /// timeline its pipeline is on: another lap of a file, another input's
    /// stream. `position` is where the run begins in its media and `start`
    /// where on the timeline its buffers are stamped from. `flushed`: what
    /// it handed on before is to be let go of, as when what feeds it began
    /// again elsewhere, and a flush goes on ahead of the segment. Only in a
    /// pipeline: a source driven by hand begins nothing.
    Segment {
        /// Whether what went on before is to be let go of.
        flushed: bool,
        /// Where the run begins in the source's media.
        position: Duration,
        /// Where on the timeline its buffers are stamped from.
        start: Duration,
    },
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
    /// Which of the source's outputs something is wired to.
    linked: &'a [bool],
}

impl Wait<'_> {
    /// Which of the source's outputs something is wired to, by number —
    /// for a source of this crate that leaves out of its own reckoning what
    /// nothing reads. Empty for a source driven by hand, which has not
    /// been run.
    pub(crate) fn linked(&self) -> &[bool] {
        self.linked
    }

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

/// Errors the framework running a [`Produce`] finds in what it was handed.
#[derive(Debug, thiserror::Error)]
pub enum ProduceError {
    /// [`Produced::On`] named an output the source does not have — see
    /// [`Produce::outputs`].
    #[error("a buffer was made for output {index}, and there are {outputs}")]
    NoOutput {
        /// The output it was made for.
        index: usize,
        /// How many there are.
        outputs: usize,
    },
}

/// What a [`Produce`] is in a pipeline: the source the framework makes of
/// it, with the pads and the loop. Nothing to call on it — it is what
/// [`IntoSource`] hands a pipeline, and what the pipeline's wiring is given.
pub struct ProducingSource<P> {
    inner: P,
    pads: Vec<SrcPad>,
    /// Time spent paused so far, which [`Wait::now`] leaves out.
    paused: Duration,
    /// Its pipeline's, for the timeline a segment it begins is on and
    /// whether a preroll runs — `None` driven by hand.
    state: Option<Arc<PlaybackState>>,
    /// Which outputs something is wired to, as its thread starts.
    linked: Vec<bool>,
    /// What was made for an output that could not take it yet, with
    /// several outputs — see [`Parking`].
    parking: Parking,
    /// A segment the source began while what it made before was still held
    /// back, to go out behind it: `(position, start)`.
    owed_segment: Option<(Duration, Duration)>,
    /// Whether a `Seek` has repositioned the source since it last ended its
    /// outputs — which sends it back to making things.
    sought: bool,
}

/// How handing every output its end went.
enum Ending {
    /// Every output has its end.
    Ended,
    /// Stopped, or finished, on the way.
    Stopped,
    /// A seek moved the source back into what it reads first.
    Sought,
}

/// How long a source whose outputs cannot take what it holds for them
/// waits before looking again — for an output that frees up without saying
/// so. A request from the pipeline ends the wait at once.
const HELD_POLL: Duration = Duration::from_millis(1);

impl<P: Produce> ProducingSource<P> {
    pub(crate) fn new(inner: P) -> Self {
        let pads = inner.outputs();
        Self {
            parking: Parking::new(pads.len()),
            inner,
            pads,
            paused: Duration::ZERO,
            state: None,
            linked: Vec::new(),
            owed_segment: None,
            sought: false,
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

    /// Whether a preroll is running — including the pause that ends one,
    /// until it reaches this source: what was held back for it stays held
    /// (see `crate::playback_state`). Never, driven by hand.
    fn prerolling(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|state| state.preroll().is_some())
    }
}

/// Reports what went wrong downstream of `inner` on `bus`, as its own.
fn reporter<'a, P: Produce>(inner: &'a P, bus: &'a Bus) -> impl FnMut(crate::error::Error) + 'a {
    move |error| {
        bus.post(
            inner.pp_log(),
            BusEvent::Error {
                element_type: inner.element_type(),
                name: inner.name(),
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
        self.state = Some(Arc::clone(&context.state));
        self.inner.attach_context(context);
    }
}

impl<P: Produce> Source for ProducingSource<P> {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl<P: Produce> SourceElement for ProducingSource<P> {
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }

    fn as_seekable(&mut self) -> Option<&mut dyn SeekableSource> {
        self.inner.as_seekable()
    }

    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleSource> {
        self.inner.as_reversible()
    }

    /// Asks for the next thing, and hands on what it is, until the stream
    /// ends or the pipeline stops it — taking what the pipeline asks
    /// between one and the next, the one way every source does
    /// (`handle_request`). A buffer a pad refuses is reported, and the
    /// source goes on; a failure to make one ends it. A source that can be
    /// sought stays at its end until it is stopped, or sought back into
    /// what it reads. What it sets up on this thread first it lets go of
    /// here last, however the loop ended ([`Produce::starting`],
    /// [`Produce::stopping`]).
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(pp_log: self.inner.pp_log(), "started");
        self.linked = self.pads.iter().map(SrcPad::is_linked).collect();
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

    fn on_control(&mut self, msg: &ControlMsg) {
        match msg {
            // What is held back was made from the timeline being left — see
            // `Parking::forget` — and so is a segment still owed behind it.
            ControlMsg::Flush => {
                self.parking.forget();
                self.owed_segment = None;
            }
            ControlMsg::Seek(_) => self.sought = true,
            ControlMsg::Pause | ControlMsg::Resume | ControlMsg::Preroll(_) | ControlMsg::Stop => {}
        }
    }

    /// Hands on everything held back, which a finish plays on to — and,
    /// playing backwards, the rest of the stretch under way, which is
    /// played from its end: one cut short would lose its later pictures.
    /// Pushed without waiting for room, which the pipeline's interrupt
    /// makes where a queue is full.
    fn finishing(&mut self, backwards: bool, bus: &Bus) -> Result<()> {
        self.parking
            .deliver_all(&mut self.pads, &mut reporter(&self.inner, bus));
        if !backwards {
            return Ok(());
        }
        let mut wait = Wait {
            control: None,
            paused: self.paused,
            linked: &self.linked,
        };
        while self
            .inner
            .as_reversible()
            .is_some_and(|source| !source.stretch_complete())
        {
            let (index, buf) = match self.inner.produce(&mut wait)? {
                Produced::Buffer(buf) => (0, buf),
                Produced::On(index, buf) => (index, buf),
                Produced::Segment { .. } | Produced::Nothing => continue,
                Produced::End => break,
            };
            let outputs = self.pads.len();
            let pad = self
                .pads
                .get_mut(index)
                .ok_or(ProduceError::NoOutput { index, outputs })?;
            if let Err(error) = pad.push(buf) {
                reporter(&self.inner, bus)(error);
            }
        }
        Ok(())
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
            let control = (!gone).then_some(control);
            let prerolling = self.prerolling();
            // What is held back goes on first, as its outputs take it; a
            // segment begun behind it goes once all of it has.
            self.parking
                .drain(&mut self.pads, &mut reporter(&self.inner, bus));
            if let Some((position, start)) = self.owed_segment {
                if !self.parking.is_empty() {
                    self.wait_a_moment(control);
                    continue;
                }
                self.owed_segment = None;
                self.begin_segment(false, position, start, bus);
            }
            // Made more of, what is held back would only grow: reading
            // waits for an output to take it.
            if self.parking.blocked(&mut self.pads, prerolling) {
                self.wait_a_moment(control);
                continue;
            }
            let mut wait = Wait {
                control,
                paused: self.paused,
                linked: &self.linked,
            };
            match self.inner.produce(&mut wait)? {
                Produced::Buffer(buf) => self.hand_on(0, buf, prerolling, bus)?,
                Produced::On(index, buf) => self.hand_on(index, buf, prerolling, bus)?,
                Produced::Segment {
                    flushed: true,
                    position,
                    start,
                } => {
                    self.parking.forget();
                    self.begin_segment(true, position, start, bus);
                }
                Produced::Segment {
                    flushed: false,
                    position,
                    start,
                } => {
                    if self.parking.is_empty() {
                        self.begin_segment(false, position, start, bus);
                    } else {
                        self.owed_segment = Some((position, start));
                    }
                }
                Produced::Nothing => {}
                Produced::End => match self.end_every_output(control, bus)? {
                    Ending::Stopped => return Ok(()),
                    // Back into what it reads before every output had its
                    // end: what is still owed belongs to the timeline just
                    // left.
                    Ending::Sought => {}
                    Ending::Ended => {
                        pp_info!(
                            pp_log: self.inner.pp_log(),
                            "event=eos phase=source_completed outcome=ok"
                        );
                        // The end of what it reads is not the end of
                        // playback for a source that can be sought: a seek
                        // from there reads on from where it lands. So it
                        // stays, passing control on, until it is stopped.
                        let lingers = self.inner.as_seekable().is_some();
                        match control {
                            Some(control) if lingers => {
                                if !self.linger(control, bus)? {
                                    return Ok(());
                                }
                            }
                            _ => return Ok(()),
                        }
                    }
                },
            }
        }
    }

    /// Waits [`HELD_POLL`] on the clock a pause does not move, letting go
    /// as soon as the pipeline asks something.
    fn wait_a_moment(&self, control: Option<&ControlReceiver>) {
        let mut wait = Wait {
            control,
            paused: self.paused,
            linked: &[],
        };
        let at = wait.now() + HELD_POLL;
        wait.until(at);
    }

    /// Hands `buf` on through output `index` — held back for it, with
    /// several outputs, where it cannot take it yet (see [`Parking`]) —
    /// reporting what refuses it: one buffer's failure after this is not
    /// the source's end. An output it does not have is the source's own
    /// mistake, and ends it.
    fn hand_on(
        &mut self,
        index: usize,
        buf: MediaBuffer,
        prerolling: bool,
        bus: &Bus,
    ) -> Result<()> {
        let outputs = self.pads.len();
        if index >= outputs {
            return Err(ProduceError::NoOutput { index, outputs }.into());
        }
        let mut report = reporter(&self.inner, bus);
        if outputs > 1 {
            self.parking
                .hand_on(&mut self.pads, index, buf, prerolling, &mut report);
        } else if let Err(error) = self.pads[index].push(buf) {
            report(error);
        }
        Ok(())
    }

    /// Hands every output its end, each as soon as it owes nothing held
    /// back and can take one, answering the pipeline between tries — and
    /// says how it ended: every output ended, the source stopped, or a
    /// seek moved it back into what it reads first.
    ///
    /// Not each in turn with a push that waits. In a preroll a branch that
    /// has its sample takes nothing more, so the queue in front of it stays
    /// full; the output after it, whose branch may need its end to preroll
    /// at all — a seek past the last of the sound is answered by the sample
    /// before it, handed on at the end of the stream — waited behind that
    /// push on this one thread until the preroll timed out. Found by the
    /// conformance matrix, with one-deep queues and sound.
    ///
    /// An output that owes nothing has its end as soon as it can take it,
    /// whatever another still owes. Waited for until every output's held
    /// buffers had gone, the end was kept from a branch that needed it to
    /// preroll, behind a sibling that had its sample and took nothing more:
    /// a seek to the last picture, past the last of the sound, timed out
    /// once in a few hundred of the matrix's sequences.
    ///
    /// A finish that arrives meanwhile ends in order: what is held back
    /// and the ends still owed go out, waited for, since a finish plays on
    /// to them — and no output is handed a second end.
    fn end_every_output(&mut self, control: Option<&ControlReceiver>, bus: &Bus) -> Result<Ending> {
        self.sought = false;
        let pp_log = self.inner.pp_log().clone();
        let mut owed = vec![true; self.pads.len()];
        loop {
            {
                let mut report = reporter(&self.inner, bus);
                self.parking.drain(&mut self.pads, &mut report);
                for (index, pad) in self.pads.iter_mut().enumerate() {
                    if owed[index]
                        && !self.parking.owes(index)
                        && (!pad.is_linked() || pad.ready_consume())
                    {
                        if let Err(error) = pad.push_eos(&pp_log) {
                            report(error);
                        }
                        owed[index] = false;
                    }
                }
            }
            if !owed.contains(&true) {
                return Ok(Ending::Ended);
            }
            if let Some(control) = control {
                while let Ok(Some((request, ack))) = control.try_take() {
                    if matches!(request, RequestKind::Finish) {
                        let mut report = reporter(&self.inner, bus);
                        self.parking.deliver_all(&mut self.pads, &mut report);
                        for (index, pad) in self.pads.iter_mut().enumerate() {
                            if owed[index]
                                && let Err(error) = pad.push_eos(&pp_log)
                            {
                                report(error);
                            }
                        }
                        let _ = ack.send(());
                        return Ok(Ending::Stopped);
                    }
                    let outcome = handle_request(control, self, bus, request, ack)?;
                    self.paused += outcome.paused_for;
                    if outcome.stopped {
                        return Ok(Ending::Stopped);
                    }
                    if self.sought {
                        return Ok(Ending::Sought);
                    }
                }
            }
            std::thread::sleep(HELD_POLL);
        }
    }

    /// Waits at the end of what it reads, passing every request on to the
    /// branches — pausing and resuming them as the pipeline does — until it
    /// is stopped or finished, `Ok(false)`, or a seek moves it back into
    /// what it reads, `Ok(true)`, to read on from there.
    fn linger(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<bool> {
        self.sought = false;
        loop {
            let Some((request, ack)) = control.recv() else {
                return Ok(false);
            };
            if matches!(request, RequestKind::Finish) {
                // A graceful finish: the end it would hand on is already out.
                let _ = ack.send(());
                return Ok(false);
            }
            // Handled as a source handles one anywhere, a `Pause` included:
            // nothing is read until a `Resume`, or a seek's `Preroll` asking
            // for its picture. Passed on without pausing, as it once was at
            // a demuxer's end, a seek from the end read on into the paused
            // queue after it and never took the `Preroll` it then waited on.
            let outcome = handle_request(control, self, bus, request, ack)?;
            self.paused += outcome.paused_for;
            if outcome.stopped {
                return Ok(false);
            }
            if self.sought {
                return Ok(true);
            }
        }
    }

    /// Begins a segment through every output — see [`Produced::Segment`] —
    /// behind a flush where it is `flushed`. The flush is not the
    /// pipeline's, so every queue after this carries it on.
    fn begin_segment(&mut self, flushed: bool, position: Duration, start: Duration, bus: &Bus) {
        let Some(state) = &self.state else {
            return;
        };
        let segment = crate::stream::Segment {
            id: state.timeline(),
            flushed,
            position,
            start,
            show_from: None,
            backwards: state.backwards(),
        };
        let mut report = reporter(&self.inner, bus);
        if flushed {
            for pad in &mut self.pads {
                if let Err(error) = pad.control(&ControlMsg::Flush) {
                    report(error);
                }
            }
        }
        let pp_log = self.inner.pp_log().clone();
        if let Err(error) = crate::stream::begin_segment(&mut self.pads, segment, &pp_log) {
            report(error);
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
            fn as_seekable(&mut self) -> Option<&mut dyn $crate::element::SeekableSource> {
                self.0.as_seekable()
            }
            fn as_reversible(&mut self) -> Option<&mut dyn $crate::element::ReversibleSource> {
                self.0.as_reversible()
            }
            fn on_control(&mut self, msg: &$crate::control::ControlMsg) {
                self.0.on_control(msg)
            }
            fn finishing(
                &mut self,
                backwards: bool,
                bus: &$crate::bus::Bus,
            ) -> $crate::error::Result<()> {
                self.0.finishing(backwards, bus)
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

    /// Makes what it is told to, in order, on two outputs, then ends.
    struct Scripted {
        pp_log: PpLog,
        script: std::collections::VecDeque<Produced>,
    }

    impl Element for Scripted {
        fn name(&self) -> Arc<str> {
            "scripted".into()
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

    impl Produce for Scripted {
        fn is_live(&self) -> bool {
            true
        }
        fn outputs(&self) -> Vec<SrcPad> {
            vec![SrcPad::new("src_0"), SrcPad::new("src_1")]
        }
        fn produce(&mut self, _wait: &mut Wait<'_>) -> Result<Produced> {
            Ok(self.script.pop_front().unwrap_or(Produced::End))
        }
    }

    /// What a [`Logged`] wrote down, in order.
    type Log = Arc<Mutex<Vec<String>>>;

    /// Writes down everything it is handed: a packet as its one byte, the
    /// stream's events and a flush as words.
    struct Logged(PpLog, Log);

    impl Element for Logged {
        fn name(&self) -> Arc<str> {
            "logged".into()
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

    impl Sink for Logged {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Packet(packet) = buf {
                let byte = packet.data().map_or(0, |data| data[0]);
                self.1.lock().unwrap().push(byte.to_string());
            }
            Ok(())
        }
        fn stream_event(&mut self, event: &crate::stream::StreamEvent) -> Result<()> {
            let word = match event {
                crate::stream::StreamEvent::Segment(segment) if segment.flushed => "flushed",
                crate::stream::StreamEvent::Segment(_) => "segment",
                crate::stream::StreamEvent::Eos => "eos",
            };
            self.1.lock().unwrap().push(word.into());
            Ok(())
        }
        fn flow(&mut self, crate::element::Flow(msg): crate::element::Flow<'_>) -> Result<()> {
            if matches!(msg, ControlMsg::Flush) {
                self.1.lock().unwrap().push("flush".into());
            }
            Ok(())
        }
    }

    fn packet(byte: u8) -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg::Packet::copy(&[byte])))
    }

    /// Runs `script` to its end with each output logged, answering both
    /// logs and the pipeline.
    fn scripted(script: Vec<Produced>) -> (Arc<Pipeline>, [Log; 2]) {
        let logs = [Arc::default(), Arc::default()];
        let source = Scripted {
            pp_log: element_pp_log(ElementType::Other, "scripted", None),
            script: script.into(),
        };
        let (pipeline, ()) = Pipeline::new("scripted", source, |source, ctx| {
            for (output, log) in logs.iter().enumerate() {
                let sink = Logged(
                    element_pp_log(ElementType::Other, "logged", None),
                    Arc::clone(log),
                );
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, output, branch)?;
            }
            Ok(())
        })
        .expect("wiring succeeds");
        pipeline.run().expect("run");
        (pipeline, logs)
    }

    fn ended(log: &Mutex<Vec<String>>) -> bool {
        log.lock().unwrap().last().is_some_and(|word| word == "eos")
    }

    /// A source with several outputs hands each buffer to the one it was
    /// made for — the first, where it does not say — and ends every one.
    #[test]
    fn each_output_is_handed_what_was_made_for_it_and_its_end() {
        let (pipeline, logs) = scripted(vec![
            Produced::On(0, packet(1)),
            Produced::On(1, packet(2)),
            Produced::Buffer(packet(3)),
            Produced::End,
        ]);
        wait_for("every output ends", || logs.iter().all(|log| ended(log)));
        pipeline.stop();
        assert_eq!(*logs[0].lock().unwrap(), ["segment", "1", "3", "eos"]);
        assert_eq!(*logs[1].lock().unwrap(), ["segment", "2", "eos"]);
    }

    /// A segment a source begins goes through every output, in order with
    /// what it made; a flushed one behind a flush, which what it made
    /// afterwards follows.
    #[test]
    fn a_segment_a_source_begins_reaches_every_output_in_order() {
        let begins = |flushed| Produced::Segment {
            flushed,
            position: Duration::ZERO,
            start: Duration::ZERO,
        };
        let (pipeline, logs) = scripted(vec![
            Produced::On(0, packet(1)),
            begins(false),
            Produced::On(0, packet(2)),
            begins(true),
            Produced::On(1, packet(3)),
            Produced::End,
        ]);
        wait_for("every output ends", || logs.iter().all(|log| ended(log)));
        pipeline.stop();
        assert_eq!(
            *logs[0].lock().unwrap(),
            ["segment", "1", "segment", "2", "flush", "flushed", "eos"]
        );
        assert_eq!(
            *logs[1].lock().unwrap(),
            ["segment", "segment", "flush", "flushed", "3", "eos"]
        );
    }

    /// A buffer made for an output the source does not have is the
    /// source's own mistake: it ends, saying which, rather than dropping
    /// the buffer without a word.
    #[test]
    fn a_buffer_for_an_output_there_is_not_ends_the_source() {
        let (pipeline, _logs) = scripted(vec![Produced::On(2, packet(1))]);
        let error = pipeline
            .bus()
            .iter()
            .find_map(|event| match event {
                BusEvent::Error { error, .. } => Some(error),
                _ => None,
            })
            .expect("the source says why it ended");
        pipeline.stop();
        assert!(
            matches!(
                error,
                crate::error::Error::ProduceError(ProduceError::NoOutput {
                    index: 2,
                    outputs: 2
                })
            ),
            "{error}"
        );
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
            linked: &[],
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
