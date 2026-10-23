//! Writing a source by what it makes alone — see [`Source`].
//!
//! A source written directly, as a [`RawSource`], runs its own loop:
//! it takes its control between buffers and hands each request on the one
//! way every source must, keeps a pause out of the schedule it makes things
//! on, and ends its stream. Seventeen did, each a little differently —
//! docs/stream-events.md lists what that cost. A [`Source`] makes the next
//! thing when asked; the framework runs the loop.

use crate::core::bell::Bell;
use std::{
    sync::Arc,
    task::Waker,
    time::{Duration, Instant},
};

use crate::{
    buffer::MediaBuffer,
    bus::Bus,
    contract::OutputContract,
    control::{
        ChannelGone, ControlMsg, ControlReceiver, RequestKind, apply_finish, apply_one,
        apply_one_unacked, handle_request,
    },
    element::{
        Attached, Context, Element, ElementType, Identity, Named, RawSource, ReversibleSource,
        SeekableSource, SrcPads,
    },
    error::Result,
    pad::SrcPad,
    parking::Parking,
    playback_state::PlaybackState,
    pp_log::{PpLog, pp_info},
};

/// A source of your own, written as what it makes alone: asked for the next
/// thing, it makes it.
///
/// Implement [`Self::produce`], waiting only through the [`Wait`] it is
/// handed, and [`Self::is_live`]. Give it to
/// [`Pipeline::new`](crate::pipeline::Pipeline::new) as any built-in source.
/// For buffers another thread of the application hands in, use
/// [`AppSource`](crate::elements::AppSource) instead.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use std::time::Duration;
///
/// use media_pp::buffer::{MediaBuffer, set_time_base};
/// use media_pp::element::{Produced, Source, Wait};
/// use media_pp::elements::FrameCounter;
/// use media_pp::ffmpeg::{self, format::Pixel, frame::Video};
/// use media_pp::pipeline::Pipeline;
///
/// /// Ten black pictures, one every 40 ms.
/// struct Black(i64);
///
/// impl Source for Black {
///     fn is_live(&self) -> bool {
///         true
///     }
///
///     fn produce(&mut self, wait: &mut Wait<'_>) -> media_pp::Result<Produced> {
///         if self.0 == 10 {
///             return Ok(Produced::End);
///         }
///         if !wait.until(wait.now() + Duration::from_millis(40)) {
///             return Ok(Produced::Nothing); // the pipeline needs the thread
///         }
///         let mut picture = Video::new(Pixel::GRAY8, 64, 64);
///         picture.set_pts(Some(self.0));
///         set_time_base(&mut picture, ffmpeg::Rational(1, 25));
///         self.0 += 1;
///         Ok(Produced::Buffer(MediaBuffer::video(picture)))
///     }
/// }
///
/// let (counter, _pictures) = FrameCounter::new("counter");
/// let (pipeline, ()) = Pipeline::new("black", Black(0).named("black"), |source, ctx| {
///     ctx.attach(source, 0, ctx.branch().to(counter)?)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
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
/// which take a `Source` as they take a [`RawSource`] (see
/// [`IntoSource`]). One output by default; a source with several — a
/// stream each, as a demuxer has — says what they are in
/// [`Self::outputs`] and which each buffer is for in [`Produced::On`].
///
/// It goes by its type's name, or by the one [`Self::named`] gives it, and is
/// told who it is through [`Self::attached`]. Its log records and bus events
/// give its kind as [`ElementType::Other`](crate::element::ElementType::Other).
pub trait Source: Send {
    /// Whether it makes what it makes at a rate of its own — a camera, a
    /// clock of its own — rather than as fast as it is asked: see
    /// [`RawSource::is_live`].
    fn is_live(&self) -> bool;

    /// Makes the next thing. Blocks, where it has to, only through
    /// `wait`; let go, it answers [`Produced::Nothing`] and is asked again.
    ///
    /// An `Err` ends the source, as one [`RawSource::run`] returns
    /// does. What goes wrong with one part of what it makes and not the
    /// rest — a compositor's layer it cannot draw — is posted on the bus
    /// its [`Context`] carries, kept from [`Self::attached`], and it goes
    /// on.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced>;

    /// What it hands on — see [`SrcPad::with_contract`]. Nothing said by
    /// default.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Unknown
    }

    /// The pads it hands on through, in the order [`Produced::On`] numbers
    /// them — asked once, as the framework makes it a source. `None`, the
    /// default, is one, named for the source and declaring
    /// [`Self::output_contract`].
    fn outputs(&self) -> Option<Vec<SrcPad>> {
        None
    }

    /// Stops what it reads from while the pipeline is paused — see
    /// [`RawSource::pausing`]. Nothing by default.
    fn pausing(&mut self) -> Result<()> {
        Ok(())
    }

    /// Starts it again once playback goes on — see
    /// [`RawSource::resuming`]. Nothing by default.
    fn resuming(&mut self) -> Result<()> {
        Ok(())
    }

    /// This source as a [`SeekableSource`], where it is one — asked as it is
    /// wired, and for each seek, which the framework takes on the source's
    /// thread between one thing made and the next; see
    /// [`RawSource::as_seekable`]. A source that can be sought also stays
    /// at its end, rather than ending its thread, until it is stopped or
    /// sought back into what it reads. `None` by default.
    fn as_seekable(&mut self) -> Option<&mut dyn SeekableSource> {
        None
    }

    /// This source as a [`ReversibleSource`], where it is one — see
    /// [`RawSource::as_reversible`]. `None` by default.
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

    /// Whether it can be driven — run by a
    /// [`DrivenPipeline`](crate::pipeline::DrivenPipeline) on whatever
    /// thread polls it rather than on one of its own. It can where it waits
    /// only through [`Wait::until`], or on something it hands
    /// [`Wait::waker`] to, and where [`Self::starting`] sets up nothing
    /// that has to stay on one thread: a driven pipeline may be polled on
    /// another thread each time. `false` by default, and a driven pipeline
    /// refuses a source that says so.
    ///
    /// Driven, a wait never blocks: [`Wait::until`] answers `false` before
    /// its time, and the source is asked again once it has come. So the
    /// time a drivable source waits for is one it keeps until it is there
    /// — `wait.until(wait.now() + tick)`, worked out anew each time it is
    /// asked, is never reached.
    fn drivable(&self) -> bool {
        false
    }

    /// Called as it is added to a pipeline, before it is asked for anything.
    ///
    /// It is told who it is in the pipeline it is being added to, and that
    /// pipeline. Keep [`Attached::pp_log`] to log; take from
    /// [`Attached::context`] what is needed of the pipeline — its bus, to post
    /// what fails for one part of what it makes. Nothing by default.
    fn attached(&mut self, _cx: &Attached<'_>) {}

    /// Gives this source a name for the log and the bus.
    ///
    /// This source, going by `name` — to tell two of it apart in the log
    /// and on the bus. One not named goes by its type's name.
    fn named(self, name: impl Into<String>) -> Named<Self>
    where
        Self: Sized,
    {
        Named::new(name, self)
    }
}

/// What a [`Source`] made when asked.
pub enum Produced {
    /// A buffer, to go on through the first output — the one a source with
    /// one output has.
    Buffer(MediaBuffer),
    /// A buffer, to go on through the output of this number — see
    /// [`Source::outputs`].
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

/// How a [`Source`] waits: on a clock that stands still while the
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
    /// Driven, what the poll it runs in has to know of its waits — and
    /// then a wait never blocks.
    driving: Option<&'a mut Driving>,
}

/// What a wait in a driven poll leaves for the poll: when the source wants
/// to be asked again, and whether it took the poll's waker to be woken by
/// something else.
pub(crate) struct Driving {
    waker: Waker,
    /// The earliest time a wait was let go before, on the real clock.
    deadline: Option<Instant>,
    /// Whether the source took [`Self::waker`].
    took_waker: bool,
}

impl Driving {
    fn new(waker: Waker) -> Self {
        Self {
            waker,
            deadline: None,
            took_waker: false,
        }
    }

    /// Asks again by `at`, on the real clock.
    fn by(&mut self, at: Instant) {
        self.deadline = Some(self.deadline.map_or(at, |deadline| deadline.min(at)));
    }
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
    ///
    /// The only wait a source of your own has: blocking on anything else —
    /// a channel, a device call with no timeout — keeps a pause or a stop
    /// waiting for it. A source fed from another thread is
    /// [`AppSource`](crate::elements::AppSource)'s job; one that must look
    /// at something of its own waits a few milliseconds at a time here and
    /// looks again.
    ///
    /// In a [`DrivenPipeline`](crate::pipeline::DrivenPipeline) this never
    /// blocks: before `at`, it answers `false` at once, and the source is
    /// asked again once `at` has come.
    pub fn until(&mut self, at: Instant) -> bool {
        let now = self.now();
        if let Some(driving) = self.driving.as_deref_mut() {
            if now >= at {
                return true;
            }
            // `at` is on the clock a pause stands still on; the poll's
            // owner keeps the real one.
            driving.by(at + self.paused);
            return false;
        }
        let Some(control) = self.control else {
            std::thread::sleep(at.saturating_duration_since(self.now()));
            return true;
        };
        // Parked, and woken by the pipeline's bell: see `crate::core::bell`
        // on why not in a crossbeam `Select`.
        control.bell().listen();
        loop {
            if control.has_something() {
                return false;
            }
            let left = at.saturating_duration_since(self.now());
            if left.is_zero() {
                return true;
            }
            std::thread::park_timeout(left);
        }
    }

    /// The waker of the poll this runs in, in a
    /// [`DrivenPipeline`](crate::pipeline::DrivenPipeline) — `None` on a
    /// thread of the source's own.
    ///
    /// For a source that waits on something that can wake it: hand the
    /// waker to it, answer [`Produced::Nothing`], and the pipeline is polled
    /// again once it is woken. A source that waits on anything else is not
    /// [`Source::drivable`].
    pub fn waker(&mut self) -> Option<&Waker> {
        let driving = self.driving.as_deref_mut()?;
        driving.took_waker = true;
        Some(&driving.waker)
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
    pub(crate) fn recv_until<T>(
        &mut self,
        rx: &crossbeam_channel::Receiver<T>,
        at: Instant,
    ) -> Option<Received<T>> {
        self.receive(rx, Some(at))
    }

    /// [`Self::recv`] of a channel whose senders ring `bell` with each
    /// message, and when they go: parked until one does, or the pipeline
    /// has something — see [`crate::core::bell`] on why not in a crossbeam
    /// `Select`.
    pub(crate) fn recv_rung<T>(
        &mut self,
        rx: &crossbeam_channel::Receiver<T>,
        bell: &Bell,
    ) -> Received<T> {
        match self.receive_rung(rx, bell, None) {
            Some(received) => received,
            None => unreachable!("a receive with no deadline ends only with an answer"),
        }
    }

    /// Waits for `bell` to be rung, or the pipeline to have something:
    /// [`Received::Got`] for a ring — which may have nothing behind it, so
    /// the caller looks again at what rang it — and [`Received::LetGo`]
    /// for the pipeline. For a source whose input has no channel of its
    /// own: a broadcast's reader.
    pub(crate) fn rung(&mut self, bell: &Bell) -> Received<()> {
        if self.driving.is_some() {
            return Received::LetGo;
        }
        let Some(control) = self.control else {
            bell.listen();
            std::thread::park();
            return Received::Got(());
        };
        control.bell().listen();
        bell.listen();
        if control.has_something() {
            return Received::LetGo;
        }
        std::thread::park();
        if control.has_something() {
            Received::LetGo
        } else {
            Received::Got(())
        }
    }

    fn receive_rung<T>(
        &mut self,
        rx: &crossbeam_channel::Receiver<T>,
        bell: &Bell,
        at: Option<Instant>,
    ) -> Option<Received<T>> {
        use crossbeam_channel::TryRecvError;

        if self.driving.is_some() || self.control.is_none() {
            return self.receive(rx, at);
        }
        let control = self.control?;
        control.bell().listen();
        bell.listen();
        loop {
            if control.has_something() {
                return Some(Received::LetGo);
            }
            match rx.try_recv() {
                Ok(got) => return Some(Received::Got(got)),
                Err(TryRecvError::Disconnected) => return Some(Received::Gone),
                Err(TryRecvError::Empty) => {}
            }
            match at {
                None => std::thread::park(),
                Some(at) => {
                    let left = at.saturating_duration_since(self.now());
                    if left.is_zero() {
                        return None;
                    }
                    std::thread::park_timeout(left);
                }
            }
        }
    }

    fn receive<T>(
        &mut self,
        rx: &crossbeam_channel::Receiver<T>,
        at: Option<Instant>,
    ) -> Option<Received<T>> {
        use crossbeam_channel::{RecvTimeoutError, TryRecvError};

        // Driven, what is there now, or let go: a channel cannot wake the
        // poll, so a source waiting on one is not drivable — see
        // `Source::drivable` — and this only keeps it from blocking.
        let now = self.now();
        let paused = self.paused;
        if let Some(driving) = self.driving.as_deref_mut() {
            return match rx.try_recv() {
                Ok(got) => Some(Received::Got(got)),
                Err(TryRecvError::Disconnected) => Some(Received::Gone),
                Err(TryRecvError::Empty) => match at {
                    Some(at) if now >= at => None,
                    Some(at) => {
                        driving.by(at + paused);
                        Some(Received::LetGo)
                    }
                    None => Some(Received::LetGo),
                },
            };
        }
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

/// Errors the framework running a [`Source`] finds in what it was handed.
#[derive(Debug, thiserror::Error)]
pub enum ProduceError {
    /// [`Produced::On`] named an output the source does not have — see
    /// [`Source::outputs`].
    #[error("a buffer was made for output {index}, and there are {outputs}")]
    NoOutput {
        /// The output it was made for.
        index: usize,
        /// How many there are.
        outputs: usize,
    },
}

/// What a [`Source`] is in a pipeline: the source the framework makes of
/// it, with the pads and the loop.
///
/// Nothing to call on it — it is what
/// [`IntoSource`] hands a pipeline, and what the pipeline's wiring is given.
pub struct SourceStage<P> {
    /// Who it is — the source's name and log identity, which it does not
    /// keep itself.
    identity: Identity,
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
    /// Where the last poll left it, driven — see [`RawSource::drive`].
    driven: DriveState,
}

/// Where a driven source stands between one poll and the next: what its
/// thread's loop keeps in where it is, driven, it keeps here.
#[derive(Default)]
struct DriveState {
    /// Whether [`Source::starting`] has been called.
    started: bool,
    /// Since when it has been paused: a pause is a state, driven, rather
    /// than a wait.
    paused_since: Option<Instant>,
    /// At its end, the outputs still owed theirs.
    ending: Option<Vec<bool>>,
    /// Whether it has ended, and [`Source::stopping`] been called.
    ended: bool,
}

/// What one poll of a driven source hands it — see
/// [`DrivenPipeline`](crate::pipeline::DrivenPipeline). Only this crate
/// makes one.
#[doc(hidden)]
pub struct Drive<'a> {
    pub(crate) control: &'a ControlReceiver,
    pub(crate) bus: &'a Bus,
    pub(crate) waker: &'a Waker,
}

/// Where a poll left a driven source.
#[doc(hidden)]
pub enum Stepped {
    /// To be polled again when woken, or by the time given where there is
    /// one.
    Pending(Option<Instant>),
    /// Stopped, finished or at its end, and let go of.
    Ended,
}

/// How many things a driven source is asked for in one poll at most, so
/// that one pipeline does not keep the others on its thread waiting.
const DRIVEN_BUDGET: usize = 32;

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

impl<P: Source> SourceStage<P> {
    pub(crate) fn new(identity: Identity, inner: P) -> Self {
        let pads = inner.outputs().unwrap_or_else(|| {
            vec![SrcPad::with_contract(
                format!("{}_src", identity.name()),
                inner.output_contract(),
            )]
        });
        Self {
            parking: Parking::new(pads.len()),
            identity,
            inner,
            pads,
            paused: Duration::ZERO,
            state: None,
            linked: Vec::new(),
            owed_segment: None,
            sought: false,
            driven: DriveState::default(),
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

/// Reports on `bus` what went wrong downstream of the source `identity`
/// is: as the element it traces back to, where it does — a decoder's
/// refusal is the decoder's — and as the source's own where it does not.
fn reporter<'a>(identity: &'a Identity, bus: &'a Bus) -> impl FnMut(crate::error::Error) + 'a {
    move |error| {
        bus.post_downstream_error(
            identity.pp_log(),
            identity.element_type(),
            identity.name(),
            error,
        );
    }
}

impl<P: Source> Element for SourceStage<P> {
    fn name(&self) -> Arc<str> {
        self.identity.name()
    }

    fn element_type(&self) -> ElementType {
        self.identity.element_type()
    }

    fn pp_log(&self) -> &PpLog {
        self.identity.pp_log()
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.identity.pp_log_mut()
    }

    fn attach_context(&mut self, context: &Arc<Context>) {
        self.state = Some(Arc::clone(&context.state));
        self.inner
            .attached(&Attached::new(self.identity.pp_log(), context));
    }
}

impl<P: Source> SrcPads for SourceStage<P> {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl<P: Source> RawSource for SourceStage<P> {
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
    /// here last, however the loop ended ([`Source::starting`],
    /// [`Source::stopping`]).
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(pp_log: self.identity.pp_log(), "started");
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
            .deliver_all(&mut self.pads, &mut reporter(&self.identity, bus));
        if !backwards {
            return Ok(());
        }
        let mut wait = Wait {
            control: None,
            paused: self.paused,
            linked: &self.linked,
            driving: None,
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
                reporter(&self.identity, bus)(error);
            }
        }
        Ok(())
    }

    fn drivable(&self) -> bool {
        self.inner.drivable()
    }

    /// One poll of the source in a driven pipeline: what the thread's loop
    /// does in [`Self::run`], as far as it gets without waiting — the
    /// requests taken, the source asked until it has nothing now — and
    /// where it got to, for the next poll. [`Source::starting`] on the
    /// first, and [`Source::stopping`] once it has ended, however.
    fn drive(&mut self, drive: &mut Drive<'_>) -> Result<Stepped> {
        if self.driven.ended {
            return Ok(Stepped::Ended);
        }
        if !self.driven.started {
            pp_info!(pp_log: self.identity.pp_log(), "started");
            self.linked = self.pads.iter().map(SrcPad::is_linked).collect();
            self.driven.started = true;
            if let Err(error) = self.inner.starting() {
                self.driven.ended = true;
                return Err(error);
            }
        }
        let stepped = self.step(drive);
        if !matches!(stepped, Ok(Stepped::Pending(_))) {
            self.driven.ended = true;
            self.inner.stopping();
        }
        stepped
    }
}

impl<P: Source> SourceStage<P> {
    /// [`RawSource::drive`] between the source's own setting up and
    /// letting go.
    fn step(&mut self, drive: &mut Drive<'_>) -> Result<Stepped> {
        let (control, bus) = (drive.control, drive.bus);
        let state = Arc::clone(control.state());
        // What the pipeline asks first, as `handle_request` takes it — but
        // a pause, rather than waited out here, is where the next poll
        // finds the source.
        while let Ok(Some((request, ack))) = control.try_take() {
            let RequestKind::Control(msg) = request else {
                self.finish_driven(bus, &ack, state.backwards());
                pp_info!(pp_log: self.identity.pp_log(), "stopped");
                return Ok(Stepped::Ended);
            };
            if let Some(since) = self.driven.paused_since {
                if !state.holds() {
                    // Downstream goes on first, then the source itself, and
                    // only then is the request done — see `wait_out_pause`.
                    apply_one_unacked(self, bus, &msg, &state)?;
                    self.paused += since.elapsed();
                    self.driven.paused_since = None;
                    self.inner.resuming()?;
                    let _ = ack.send(());
                    continue;
                }
            } else if msg == ControlMsg::Pause {
                self.driven.paused_since = Some(Instant::now());
                self.inner.pausing()?;
            }
            if apply_one(self, bus, &msg, &ack, &state)? {
                if let Some(since) = self.driven.paused_since.take() {
                    self.paused += since.elapsed();
                }
                pp_info!(pp_log: self.identity.pp_log(), "stopped");
                return Ok(Stepped::Ended);
            }
        }
        if self.driven.paused_since.is_some() {
            return Ok(Stepped::Pending(None));
        }
        if self.driven.ending.is_some() {
            return Ok(self.end_driven(bus));
        }
        let mut driving = Driving::new(drive.waker.clone());
        for _ in 0..DRIVEN_BUDGET {
            let prerolling = self.prerolling();
            self.parking
                .drain(&mut self.pads, &mut reporter(&self.identity, bus));
            if let Some((position, start)) = self.owed_segment {
                if !self.parking.is_empty() {
                    return Ok(Stepped::Pending(Some(Instant::now() + HELD_POLL)));
                }
                self.owed_segment = None;
                self.begin_segment(false, position, start, bus);
            }
            if self.parking.blocked(&mut self.pads, prerolling) {
                return Ok(Stepped::Pending(Some(Instant::now() + HELD_POLL)));
            }
            driving.deadline = None;
            driving.took_waker = false;
            let mut wait = Wait {
                control: None,
                paused: self.paused,
                linked: &self.linked,
                driving: Some(&mut driving),
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
                // Let go of a wait: asked again by when it said, or once
                // what it handed the waker to wakes the poll. Neither, it
                // had nothing this time and is asked again at once, as its
                // thread's loop would.
                Produced::Nothing => {
                    if let Some(at) = driving.deadline {
                        return Ok(Stepped::Pending(Some(at)));
                    }
                    if driving.took_waker {
                        return Ok(Stepped::Pending(None));
                    }
                }
                // Nothing seeks a driven pipeline, so nothing brings the
                // source back from its end: it ends there.
                Produced::End => {
                    self.driven.ending = Some(vec![true; self.pads.len()]);
                    return Ok(self.end_driven(bus));
                }
            }
        }
        // More to do, and the rest of the thread's pipelines are owed a turn.
        Ok(Stepped::Pending(Some(Instant::now())))
    }

    /// Hands each output owed its end that end, as [`Self::end_every_output`]
    /// does, as far as it can without waiting.
    fn end_driven(&mut self, bus: &Bus) -> Stepped {
        let pp_log = self.identity.pp_log().clone();
        let Some(owed) = self.driven.ending.as_mut() else {
            return Stepped::Ended;
        };
        let mut report = reporter(&self.identity, bus);
        self.parking.drain(&mut self.pads, &mut report);
        for (index, pad) in self.pads.iter_mut().enumerate() {
            if owed[index] && !self.parking.owes(index) && (!pad.is_linked() || pad.ready_consume())
            {
                if let Err(error) = pad.push_eos(&pp_log) {
                    report(error);
                }
                owed[index] = false;
            }
        }
        if owed.contains(&true) {
            return Stepped::Pending(Some(Instant::now() + HELD_POLL));
        }
        pp_info!(
            pp_log: &pp_log,
            "event=eos phase=source_completed outcome=ok"
        );
        Stepped::Ended
    }

    /// A finish, driven: as [`apply_finish`] — or, at the end of its
    /// stream already, what is held back and the ends still owed, and no
    /// output a second end.
    fn finish_driven(&mut self, bus: &Bus, ack: &crossbeam_channel::Sender<()>, backwards: bool) {
        let Some(owed) = self.driven.ending.take() else {
            apply_finish(self, bus, ack, backwards);
            return;
        };
        let pp_log = self.identity.pp_log().clone();
        let mut report = reporter(&self.identity, bus);
        self.parking.deliver_all(&mut self.pads, &mut report);
        for (index, pad) in self.pads.iter_mut().enumerate() {
            if owed[index]
                && let Err(error) = pad.push_eos(&pp_log)
            {
                report(error);
            }
        }
        let _ = ack.send(());
    }

    /// The loop itself: [`RawSource::run`] between the source's own
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
                    pp_info!(pp_log: self.identity.pp_log(), "stopped");
                    return Ok(());
                }
            }
            let control = (!gone).then_some(control);
            let prerolling = self.prerolling();
            // What is held back goes on first, as its outputs take it; a
            // segment begun behind it goes once all of it has.
            self.parking
                .drain(&mut self.pads, &mut reporter(&self.identity, bus));
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
                driving: None,
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
                            pp_log: self.identity.pp_log(),
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
            driving: None,
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
        let mut report = reporter(&self.identity, bus);
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
        let pp_log = self.identity.pp_log().clone();
        let mut owed = vec![true; self.pads.len()];
        loop {
            {
                let mut report = reporter(&self.identity, bus);
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
                        let mut report = reporter(&self.identity, bus);
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
        let mut report = reporter(&self.identity, bus);
        if flushed {
            for pad in &mut self.pads {
                if let Err(error) = pad.control(&ControlMsg::Flush) {
                    report(error);
                }
            }
        }
        let pp_log = self.identity.pp_log().clone();
        if let Err(error) = crate::stream::begin_segment(&mut self.pads, segment, &pp_log) {
            report(error);
        }
    }
}

/// What can be put where a source is asked for: a [`RawSource`]
/// itself, or a [`Source`], which the framework makes one of — going by
/// its type's name, or by the one a [`Named`] gives it.
///
/// `M` says which a type is, and the compiler works it out:
/// nothing names it. A type that is both is refused as ambiguous —
/// implement one or the other.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is neither a `RawSource` nor a `Source`",
    note = "implement `Source` for a source of your own; `RawSource` is implemented only by this crate's sources"
)]
pub trait IntoSource<M>: sealed::Sealed<M> {
    /// The source a pipeline is given — `Self` for a [`RawSource`], a
    /// [`SourceStage`] for a [`Source`] — and what its wiring is
    /// handed.
    type Raw: RawSource + 'static;

    /// The source this is.
    fn into_source(self) -> Self::Raw;
}

/// Says a type goes in as the [`RawSource`] it is — see [`IntoSource`].
pub enum AsRawSource {}

/// Says a type goes in as the [`Source`] it is — see [`IntoSource`].
pub enum AsSource {}

/// Says a [`Named`] source goes in as the [`Source`] it holds, by its name
/// — see [`IntoSource`].
pub enum AsNamedSource {}

impl<S: RawSource + 'static> IntoSource<AsRawSource> for S {
    type Raw = S;

    fn into_source(self) -> S {
        self
    }
}

impl<P: Source + 'static> IntoSource<AsSource> for P {
    type Raw = SourceStage<P>;

    fn into_source(self) -> SourceStage<P> {
        SourceStage::new(Identity::of::<P>(), self)
    }
}

impl<P: Source + 'static> IntoSource<AsNamedSource> for Named<P> {
    type Raw = SourceStage<P>;

    fn into_source(self) -> SourceStage<P> {
        SourceStage::new(Identity::other(self.name), self.inner)
    }
}

mod sealed {
    use super::{AsNamedSource, AsRawSource, AsSource, Named, RawSource, Source};

    /// Keeps [`super::IntoSource`] to the three ways in it has.
    pub trait Sealed<M> {}

    impl<S: RawSource + 'static> Sealed<AsRawSource> for S {}
    impl<P: Source + 'static> Sealed<AsSource> for P {}
    impl<P: Source + 'static> Sealed<AsNamedSource> for Named<P> {}
}

/// Makes `$name`, a newtype over `SourceStage<_>`, the source it is —
/// every method its own. For an element of this crate that keeps the public
/// name and constructors it always had while its work moves into a
/// [`Source`].
macro_rules! source_stage {
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

        impl $crate::element::SrcPads for $name {
            fn src_pads(&mut self) -> &mut [$crate::pad::SrcPad] {
                self.0.src_pads()
            }
        }

        impl $crate::element::RawSource for $name {
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
            fn drivable(&self) -> bool {
                self.0.drivable()
            }
            fn drive(
                &mut self,
                drive: &mut $crate::element::Drive<'_>,
            ) -> $crate::error::Result<$crate::element::Stepped> {
                self.0.drive(drive)
            }
        }
    };
}

pub(crate) use source_stage;

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
    use crate::bus::BusEvent;
    use crate::element::{Filter, Output, RawSink, element_pp_log};
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

    impl Source for Ticks {
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
            Ok(Produced::Buffer(MediaBuffer::Packet(
                Arc::new(ffmpeg::Packet::copy(&[self.left as u8])).into(),
            )))
        }

        fn attached(&mut self, cx: &crate::element::Attached<'_>) {
            self.pp_log = cx.pp_log().clone();
        }
    }

    /// A source of its own keeps no name: it goes by its type's, or by the
    /// one it is given, and its one output is named for it.
    #[test]
    fn a_source_goes_by_its_type_s_name_or_the_one_it_is_given() {
        let mut unnamed = Ticks::new(Duration::from_millis(1), 1).into_source();
        assert_eq!(&*unnamed.name(), "Ticks");
        assert_eq!(unnamed.src_pads()[0].name(), "Ticks_src");
        let mut named = Ticks::new(Duration::from_millis(1), 1)
            .named("ticks")
            .into_source();
        assert_eq!(&*named.name(), "ticks");
        assert_eq!(named.src_pads()[0].name(), "ticks_src");
    }

    /// Refuses every buffer.
    struct Refusing;

    impl Filter for Refusing {
        fn transform(&mut self, _buf: MediaBuffer, _out: &mut Output) -> Result<()> {
            Err(crate::error::Error::Other("refused".into()))
        }
    }

    /// What fails downstream of a source is reported as the element that
    /// failed, not as the source that happened to be pushing to it.
    #[test]
    fn a_failure_downstream_is_reported_as_the_element_that_failed() {
        let (pipeline, ()) = Pipeline::new(
            "refusals",
            Ticks::new(Duration::from_millis(1), 1).named("ticks"),
            |ticks, ctx| {
                let branch = ctx
                    .branch()
                    .pipe(Refusing.named("refusing"))
                    .to(crate::elements::PacketCounter::new("counter").0)?;
                ctx.attach(ticks, 0, branch)?;
                Ok(())
            },
        )
        .expect("wiring succeeds");
        pipeline.run().expect("run");
        let failed = pipeline
            .bus()
            .iter()
            .find_map(|event| match event {
                BusEvent::Error { name, .. } => Some(name),
                _ => None,
            })
            .expect("the refusal is reported");
        pipeline.stop();
        assert_eq!(&*failed, "refusing");
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

    impl RawSink for Arriving {
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

    /// A `Source` goes where a source does, and the framework runs it:
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

    impl Source for Hooked {
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

        fn attached(&mut self, cx: &crate::element::Attached<'_>) {
            self.pp_log = cx.pp_log().clone();
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

    impl Source for Scripted {
        fn is_live(&self) -> bool {
            true
        }
        fn outputs(&self) -> Option<Vec<SrcPad>> {
            Some(vec![SrcPad::new("src_0"), SrcPad::new("src_1")])
        }
        fn produce(&mut self, _wait: &mut Wait<'_>) -> Result<Produced> {
            Ok(self.script.pop_front().unwrap_or(Produced::End))
        }

        fn attached(&mut self, cx: &crate::element::Attached<'_>) {
            self.pp_log = cx.pp_log().clone();
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

    impl RawSink for Logged {
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
        MediaBuffer::Packet(Arc::new(ffmpeg::Packet::copy(&[byte])).into())
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
            driving: None,
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
