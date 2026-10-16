//! Writing a terminal by what it does with each buffer — see [`Sink`].
//!
//! A terminal written directly, as a [`RawSink`], reacts to every control
//! message itself: a pause, a seek's flush, a stop, the end of its stream
//! among its buffers. Each did that for itself, and each a little
//! differently — docs/stream-events.md lists what that cost. A [`Sink`]
//! does what the terminal is for and nothing else; the framework does the
//! rest, once, the same way for every one.

use std::sync::Arc;

use crate::{
    buffer::MediaBuffer,
    contract::InputContract,
    control::ControlMsg,
    element::{Attached, Context, Element, ElementType, Flow, Identity, Named, RawSink},
    error::Result,
    pp_log::PpLog,
    stream::StreamEvent,
};

/// A sink of your own, written as what it does with each buffer: plays it,
/// shows it, hands it on out of the pipeline.
///
/// Implement [`Self::render`], and the other hooks only where the sink
/// holds something or drives a device. End a branch in it with
/// [`ChainBuilder::to`](crate::pipeline::ChainBuilder::to), as any built-in
/// sink.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::buffer::MediaBuffer;
/// use media_pp::element::Sink;
/// use media_pp::elements::{TestVideoOptions, TestVideoSource};
/// use media_pp::pipeline::Pipeline;
///
/// /// Prints the timestamp of each buffer.
/// struct PrintPts;
///
/// impl Sink for PrintPts {
///     fn render(&mut self, buf: MediaBuffer) -> media_pp::Result<()> {
///         println!("{:?}", buf.pts());
///         Ok(())
///     }
/// }
///
/// let source = TestVideoSource::new("source", TestVideoOptions::default());
/// let (pipeline, ()) = Pipeline::new("print", source, |source, ctx| {
///     let branch = ctx.branch().to(PrintPts.named("print"))?;
///     ctx.attach(source, 0, branch)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// The framework does the rest — `drain` before the end of the stream is
/// taken, `reset` when a seek leaves the timeline behind or the stream is
/// stopped, and a device's own `pausing` and `resuming` as playback pauses
/// and goes on. Whether it takes a buffer at all is the pipeline's to say:
/// while paused, and once a preroll has its sample, the terminal is not
/// handed one. The element never sees a control message.
///
/// Put in a pipeline as any terminal is, with
/// [`ChainBuilder::to`](crate::pipeline::ChainBuilder::to), which takes a
/// `Sink` as it takes a [`RawSink`] (see [`IntoTerminal`]). A terminal that
/// routes the stream itself — a muxer with several tracks, a compositor's
/// input — is written the direct way.
///
/// A terminal is not an [`Element`]: the stage the framework makes of it
/// keeps its name and log identity. It goes by its type's name, or by the
/// one [`Self::named`] gives it, and is told who it is through
/// [`Self::attached`]. Its log records and bus events give its kind as
/// [`ElementType::Other`](crate::element::ElementType::Other).
pub trait Sink: Send {
    /// Does with `buf` what this terminal is for. Never handed the end of
    /// the stream: that is `drain`'s.
    fn render(&mut self, buf: MediaBuffer) -> Result<()>;

    /// Finishes what it still holds before its end is taken — a device's
    /// buffer played out, a transcriber's last stretch. Nothing by default.
    fn drain(&mut self) -> Result<()> {
        Ok(())
    }

    /// Lets go of what belongs to the timeline a seek has left, or to a
    /// stream that has been stopped — a device's queued sound among it,
    /// which is why this, unlike a [`Filter`](crate::element::Filter)'s,
    /// can fail. Nothing by default.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    /// Lets go of what a stopped stream leaves, where that is more than a
    /// seek's flush does — a device's stream deactivated and the position it
    /// masters handed back, as well as its queued sound. What
    /// [`Self::reset`] does, by default.
    fn stopping(&mut self) -> Result<()> {
        self.reset()
    }

    /// Stops its device while the pipeline is paused. Nothing by default.
    fn pausing(&mut self) -> Result<()> {
        Ok(())
    }

    /// Starts it again once playback goes on. Nothing by default.
    fn resuming(&mut self) -> Result<()> {
        Ok(())
    }

    /// What it takes — see [`RawSink::input_contract`]. Nothing said by default.
    fn input_contract(&self) -> InputContract {
        InputContract::Unknown
    }

    /// Whether it can follow a seek — see [`RawSink::accepts_seek`]. Yes by
    /// default.
    fn accepts_seek(&self) -> bool {
        true
    }

    /// Called as it is wired into a pipeline, before any buffer.
    ///
    /// It is told who it is in the pipeline it is being wired into, and
    /// that pipeline — and again if it is wired anew. Keep
    /// [`Attached::pp_log`] to log; take from [`Attached::context`] what is
    /// needed of the pipeline. Nothing by default.
    fn attached(&mut self, _cx: &Attached<'_>) {}

    /// Gives this sink a name for the log and the bus.
    ///
    /// This terminal, going by `name` — to tell two of it apart in the log
    /// and on the bus. One not named goes by its type's name.
    fn named(self, name: impl Into<String>) -> Named<Self>
    where
        Self: Sized,
    {
        Named::new(name, self)
    }
}

/// What a [`Sink`] is in a pipeline: the terminal the framework makes of
/// it, keeping the rules it does not keep itself.
pub(crate) struct SinkStage<R> {
    /// Who it is — the terminal's name and log identity, which it does not
    /// keep itself.
    identity: Identity,
    /// The terminal itself — open to the element of this crate that is a
    /// newtype over its stage, to reach what it is made of.
    pub(crate) inner: R,
}

impl<R: Sink> SinkStage<R> {
    pub(crate) fn new(identity: Identity, inner: R) -> Self {
        Self { identity, inner }
    }
}

impl<R: Sink> Element for SinkStage<R> {
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
        self.inner
            .attached(&Attached::new(self.identity.pp_log(), context));
    }
}

impl<R: Sink> RawSink for SinkStage<R> {
    fn input_contract(&self) -> InputContract {
        self.inner.input_contract()
    }

    fn accepts_seek(&self) -> bool {
        self.inner.accepts_seek()
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.inner.render(buf)
    }

    /// The end is where it drains; a seek's flushed segment is handled with
    /// the `Flush` before it, in `flow`.
    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        match event {
            StreamEvent::Eos => self.inner.drain(),
            StreamEvent::Segment(_) => Ok(()),
        }
    }

    fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
        match msg {
            ControlMsg::Pause => self.inner.pausing(),
            ControlMsg::Resume => self.inner.resuming(),
            ControlMsg::Flush => self.inner.reset(),
            ControlMsg::Stop => self.inner.stopping(),
            ControlMsg::Seek(_) | ControlMsg::Preroll(_) => Ok(()),
        }
    }
}

/// What can be put where a terminal is asked for: a [`Sink`], which the
/// framework makes one of — going by its type's name, or by the one a
/// [`Named`] gives it — a [`RawSink`] itself, or a [`BoxSink`] holding
/// either.
///
/// `M` says which a type is, and the compiler works it out: nothing names
/// it. A type that is both a `Sink` and a `RawSink` is refused as
/// ambiguous — implement one or the other.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is neither a `RawSink` nor a `Sink`",
    note = "implement `Sink` for a terminal that does something with each buffer, or `RawSink` for one that routes the stream itself",
    note = "a chain ends at a `DetachedBranch` — a `Tee`'s fan-out — through `to_branch`, not `to`"
)]
pub trait IntoTerminal<M>: sealed::Sealed<M> {
    /// The terminal this is, whichever kind it is.
    fn into_terminal(self) -> BoxSink;
}

/// Says a type goes in as the [`RawSink`] it is — see [`IntoTerminal`].
pub enum AsRawSink {}

/// Says a type goes in as the [`Sink`] it is — see [`IntoTerminal`].
pub enum AsSink {}

/// Says a [`Named`] terminal goes in as the [`Sink`] it holds, by its name
/// — see [`IntoTerminal`].
pub enum AsNamedSink {}

/// Says a [`BoxSink`] goes in as the terminal it holds — see
/// [`IntoTerminal`].
pub enum AsBoxSink {}

impl<S: RawSink + 'static> IntoTerminal<AsRawSink> for S {
    fn into_terminal(self) -> BoxSink {
        BoxSink(Box::new(self))
    }
}

impl<R: Sink + 'static> IntoTerminal<AsSink> for R {
    fn into_terminal(self) -> BoxSink {
        BoxSink(Box::new(SinkStage::new(Identity::of::<R>(), self)))
    }
}

impl<R: Sink + 'static> IntoTerminal<AsNamedSink> for Named<R> {
    fn into_terminal(self) -> BoxSink {
        BoxSink(Box::new(SinkStage::new(
            Identity::other(self.name),
            self.inner,
        )))
    }
}

impl IntoTerminal<AsBoxSink> for BoxSink {
    fn into_terminal(self) -> BoxSink {
        self
    }
}

mod sealed {
    use super::{AsBoxSink, AsNamedSink, AsRawSink, AsSink, BoxSink, Named, RawSink, Sink};

    /// Keeps [`super::IntoTerminal`] to the four ways in it has.
    pub trait Sealed<M> {}

    impl<S: RawSink + 'static> Sealed<AsRawSink> for S {}
    impl<R: Sink + 'static> Sealed<AsSink> for R {}
    impl<R: Sink + 'static> Sealed<AsNamedSink> for Named<R> {}
    impl Sealed<AsBoxSink> for BoxSink {}
}

/// A boxed sink of any kind, for a terminal picked while the program runs.
///
/// A terminal of any kind, which kind forgotten: what to hold where the
/// terminal is picked as the program runs — a window on one machine, a
/// file on another — and what a muxer's track, a mixer's or a compositor's
/// input and a bridge's feeding end are handed out as.
/// [`ChainBuilder::to`](crate::pipeline::ChainBuilder::to) ends a branch in
/// one as in the terminal it holds.
///
/// Made with [`BoxSink::new`] from a [`Sink`] or a [`RawSink`] alike. It
/// derefs to the terminal inside, for what is asked of one directly — its
/// name, or a buffer handed to it by hand.
pub struct BoxSink(Box<dyn RawSink>);

impl BoxSink {
    /// Boxes `terminal`, whichever kind it is.
    pub fn new<M>(terminal: impl IntoTerminal<M>) -> Self {
        terminal.into_terminal()
    }

    /// A terminal this crate already holds boxed, as one.
    pub(crate) fn from_raw(terminal: Box<dyn RawSink>) -> Self {
        Self(terminal)
    }

    /// The terminal inside, as the framework drives it.
    pub(crate) fn into_raw(self) -> Box<dyn RawSink> {
        self.0
    }
}

impl std::ops::Deref for BoxSink {
    type Target = dyn RawSink;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl std::ops::DerefMut for BoxSink {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

impl std::fmt::Debug for BoxSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("BoxSink").field(&self.0.name()).finish()
    }
}

/// Makes `$name`, a newtype over `SinkStage<_>`, the terminal its stage
/// is — every method the stage's. For an element of this crate that keeps
/// the public name and constructors it always had while its work moves
/// into a [`Sink`]. A generic one names its parameters and their bounds:
/// `sink_stage!(Name<F> where F: Bound)`.
macro_rules! sink_stage {
    ($name:ident) => {
        $crate::render::sink_stage!(@impl [] $name [] []);
    };
    ($name:ident<$($param:ident),+> where $($bound:tt)+) => {
        $crate::render::sink_stage!(@impl [$($param),+] $name [$($param),+] [$($bound)+]);
    };
    (@impl [$($generic:ident),*] $name:ident [$($arg:ident),*] [$($bound:tt)*]) => {
        impl<$($generic),*> $crate::element::Element for $name<$($arg),*>
        where
            $($bound)*
        {
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

        impl<$($generic),*> $crate::element::RawSink for $name<$($arg),*>
        where
            $($bound)*
        {
            fn consume(&mut self, buf: $crate::buffer::MediaBuffer) -> $crate::error::Result<()> {
                self.0.consume(buf)
            }
            fn input_contract(&self) -> $crate::contract::InputContract {
                self.0.input_contract()
            }
            fn accepts_seek(&self) -> bool {
                self.0.accepts_seek()
            }
            fn flow(&mut self, flow: $crate::element::Flow<'_>) -> $crate::error::Result<()> {
                self.0.flow(flow)
            }
            fn stream_event(
                &mut self,
                event: &$crate::stream::StreamEvent,
            ) -> $crate::error::Result<()> {
                self.0.stream_event(event)
            }
        }
    };
}

pub(crate) use sink_stage;

#[cfg(test)]
mod tests {
    use std::{
        sync::Mutex,
        thread,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::{
        bus::BusEvent,
        element::element_pp_log,
        elements::{TestVideoOptions, TestVideoSource},
        pipeline::Pipeline,
    };

    /// Notes each call the framework makes of it.
    struct Noting {
        pp_log: PpLog,
        seen: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Noting {
        fn note(&self, what: &'static str) {
            let mut seen = self.seen.lock().unwrap();
            // A run of renders is one entry: how many is the source's.
            if !(what == "render" && seen.last() == Some(&"render")) {
                seen.push(what);
            }
        }
    }

    impl Sink for Noting {
        fn render(&mut self, _buf: MediaBuffer) -> Result<()> {
            self.note("render");
            Ok(())
        }

        fn drain(&mut self) -> Result<()> {
            self.note("drain");
            Ok(())
        }

        fn reset(&mut self) -> Result<()> {
            self.note("reset");
            Ok(())
        }

        fn pausing(&mut self) -> Result<()> {
            self.note("pausing");
            Ok(())
        }

        fn resuming(&mut self) -> Result<()> {
            self.note("resuming");
            Ok(())
        }

        fn attached(&mut self, cx: &crate::element::Attached<'_>) {
            self.pp_log = cx.pp_log().clone();
        }
    }

    /// Tells a stop from a seek's flush, as a device that deactivates for a
    /// stop does.
    struct Stopping(Noting);

    impl Sink for Stopping {
        fn render(&mut self, buf: MediaBuffer) -> Result<()> {
            self.0.render(buf)
        }
        fn reset(&mut self) -> Result<()> {
            self.0.reset()
        }
        fn stopping(&mut self) -> Result<()> {
            self.0.note("stopping");
            Ok(())
        }
    }

    /// A terminal of its own keeps no name: it goes by its type's, or by
    /// the one it is given where it is wired.
    #[test]
    fn a_terminal_goes_by_its_type_s_name_or_the_one_it_is_given() {
        let noting = || Noting {
            pp_log: element_pp_log(ElementType::Other, "noting", None),
            seen: Arc::default(),
        };
        let unnamed = noting().into_terminal();
        assert_eq!(&*unnamed.name(), "Noting");
        let named = noting().named("seen").into_terminal();
        assert_eq!(&*named.name(), "seen");
        assert_eq!(named.element_type(), ElementType::Other);
        assert_eq!(named.pp_log().name(), "seen");
    }

    /// A flush lets go of what a seek left and a stop of what a stopped
    /// stream leaves, which a terminal may tell apart: a stop is
    /// `stopping`, which is `reset` where it says nothing of its own.
    #[test]
    fn a_flush_resets_and_a_stop_stops() {
        use crate::{control::ControlMsg, element::RawSinkExt};

        let noting = |seen: &Arc<Mutex<Vec<&'static str>>>| Noting {
            pp_log: element_pp_log(ElementType::Other, "noting", None),
            seen: Arc::clone(seen),
        };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut telling = SinkStage::new(Identity::other("sink"), Stopping(noting(&seen)));
        telling.control(&ControlMsg::Flush).unwrap();
        telling.control(&ControlMsg::Stop).unwrap();
        assert_eq!(*seen.lock().unwrap(), ["reset", "stopping"]);

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut plain = SinkStage::new(Identity::other("sink"), noting(&seen));
        plain.control(&ControlMsg::Flush).unwrap();
        plain.control(&ControlMsg::Stop).unwrap();
        assert_eq!(*seen.lock().unwrap(), ["reset", "reset"]);
    }

    fn wait_for(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < deadline, "{what}");
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// A `Sink` goes where a terminal does, and the framework asks of it
    /// what a pause, playing on, the end of the stream and a stop each
    /// ask — never a control message of its own.
    #[test]
    fn a_render_is_a_terminal_the_framework_drives() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let noting = Noting {
            pp_log: element_pp_log(ElementType::Other, "noting", None),
            seen: Arc::clone(&seen),
        };
        let source = TestVideoSource::new(
            "frames",
            TestVideoOptions {
                width: 16,
                height: 16,
                frame_rate: ffmpeg_next::Rational::new(100, 1),
            },
        );
        let (pipeline, ()) = Pipeline::new("render", source, |source, ctx| {
            let branch = ctx.branch().to(noting)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring succeeds");
        pipeline.run().expect("run");
        let noted = |what| seen.lock().unwrap().contains(&what);
        wait_for("nothing was rendered", || noted("render"));
        pipeline.pause();
        wait_for("the pause was not seen", || noted("pausing"));
        pipeline.resume();
        wait_for("playing on was not seen", || noted("resuming"));
        pipeline.finish();
        wait_for("the end was not drained", || noted("drain"));
        pipeline.stop();

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen[..3], ["render", "pausing", "resuming"], "{seen:?}");
        let drained = seen.iter().position(|what| *what == "drain").unwrap();
        assert!(
            seen[drained + 1..].iter().all(|what| *what == "reset"),
            "nothing is rendered after the end: {seen:?}"
        );
        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
    }
}
