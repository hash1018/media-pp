//! Where a [`Pipeline`] is in its media and how that moves: seeking, and a
//! frame step, each made of the same stages — hold, reposition,
//! preroll and the end of the preroll.

use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

use crate::pp_log::{pp_trace, pp_warn};

use crate::{
    control::{ControlMsg, PrerollContext, PrerollError},
    error::Result,
    playback_state::Phase,
};

use super::{Pipeline, PipelineError};

/// The longest a seek or a step waits for its preroll.
const PREROLL_TIMEOUT: Duration = Duration::from_secs(5);

/// How [`Pipeline::seek`] chooses the sample shown at the requested position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SeekMode {
    /// Land at the preceding keyframe and preview the first decodable sample.
    Keyframe,
    /// Decode forward from the preceding keyframe and preview the sample that
    /// covers the requested timestamp.
    Accurate,
}

impl Pipeline {
    /// After a step: plays on from the picture shown, with everything else
    /// lined up to it first — an accurate seek to it, ended by playing on. A
    /// step drops the sound and leaves the clock where it was; see
    /// [`Self::step`]. `false`, having done nothing, where there is no
    /// picture to line up to.
    pub(super) fn play_on_from_the_picture(&self) -> bool {
        let (_, terminals) = self.picture_terminals();
        let Some((at, _)) = self.state.picture_at(&terminals) else {
            return false;
        };
        self.reposition(at);
        let preroll = self.preroll_for(|terminals| PrerollContext::for_seek(terminals, at));
        if let Err(error) = self.preroll(&preroll, PREROLL_TIMEOUT) {
            pp_warn!(
                pp_log: &self.pp_log,
                "playing on after a step, not everything lined up to the picture: {error}"
            );
        }
        self.end_preroll(true);
        true
    }

    /// Announces the preroll a seek is about to wait on, cancelling it
    /// immediately if the pipeline has already been abandoned.
    ///
    /// That second case is not hypothetical: `stop` runs before the operation
    /// lock, so it can arrive in the window between the seek repositioning its
    /// sources and reaching this call. One mutex covers both sides — either
    /// `stop` finds the preroll here, or this finds `stop`'s flag.
    fn publish_preroll(&self, preroll: &Arc<PrerollContext>) {
        let mut slot = self
            .preroll_slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if slot.abandoned {
            preroll.cancel();
            return;
        }
        slot.active = Some(Arc::clone(preroll));
    }

    fn retire_preroll(&self) {
        self.preroll_slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active = None;
    }

    /// Waits for `preroll`, rechecking the topology whenever it has not
    /// finished yet.
    ///
    /// The expected terminals are fixed when the seek starts; the graph is
    /// not. Detaching a `Tee` branch mid-seek removes its terminal without
    /// removing the obligation to hear from it, and nothing is left to report
    /// a sample for it. Rather than lock topology changes out for the whole
    /// wait, this simply stops expecting whoever has since left — which also
    /// covers any other way a terminal can disappear, not just that one.
    ///
    /// The graph snapshot only happens on a poll that found work still
    /// pending, so a preroll that completes promptly never takes one.
    fn await_preroll(
        &self,
        preroll: &PrerollContext,
        timeout: Duration,
    ) -> std::result::Result<(), PrerollError> {
        const TOPOLOGY_POLL_INTERVAL: Duration = Duration::from_millis(50);

        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let slice = remaining.min(TOPOLOGY_POLL_INTERVAL);
            match preroll.wait(slice) {
                Err(PrerollError::TimedOut { pending }) => {
                    let live = self.graph().terminal_ids();
                    for terminal in pending
                        .iter()
                        .map(|node| node.id)
                        .filter(|terminal| !live.contains(terminal))
                    {
                        pp_trace!(
                            pp_log: &self.pp_log,
                            "event=control control=Preroll phase=pending \
                             outcome=departed terminal={terminal:?}"
                        );
                        preroll.mark_departed(terminal);
                    }
                    if remaining <= TOPOLOGY_POLL_INTERVAL {
                        // Deadline reached; report what is still owed, minus
                        // anything the prune above just resolved.
                        return preroll.wait(Duration::ZERO);
                    }
                }
                outcome => return outcome,
            }
        }
    }

    /// Ends an in-flight seek's preroll wait and refuses any that starts
    /// afterwards. Safe with none in flight, and deliberately takes no other
    /// lock: the whole point is to run *before* the operation lock a seek is
    /// holding.
    pub(super) fn abandon_preroll(&self) {
        let preroll = {
            let mut slot = self
                .preroll_slot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            slot.abandoned = true;
            slot.active.take()
        };
        if let Some(preroll) = preroll {
            preroll.cancel();
        }
    }

    /// Jumps to an absolute position from the start of the media. The whole
    /// operation is serialized against lifecycle controls and internally runs
    /// `Pause -> Flush -> Seek -> Preroll -> Pause`, and `Resume` after that if it
    /// was playing — see its four stages below. Every source repositions (see
    /// [`crate::element::SeekableSource::seek`]) and every downstream element
    /// reacts before preroll begins. Once every terminal in the starting
    /// topology snapshot has
    /// accepted a first sample (or EOS), a paused pipeline remains paused and
    /// a playing pipeline resumes. Fails with [`PipelineError::NotRunning`]
    /// before [`Pipeline::run`] or once every source has stopped; a source
    /// parked at the end of its stream still counts as running.
    ///
    /// Raises an interrupt before starting the synchronous cascade so a
    /// `Pacer` in a long wait can return its worker promptly.
    /// The clock's playback anchor is still reset later, inside
    /// [`Sink::control`](crate::element::Sink::control) on `Pacer`, after
    /// that in-flight frame is
    /// out of the way.
    ///
    /// Before changing anything, [`Self::check_seek`] asks the graph whether
    /// every source and branch can follow: a live or non-seekable source, or
    /// a recording muxer, returns [`crate::control::SeekError`] without
    /// flushing the current timeline.
    ///
    /// `mode` chooses whether decoding stops at the preceding keyframe or
    /// advances to the sample covering `target`.
    ///
    /// Completion means every terminal accepted its first new-timeline sample
    /// according to [`Sink::consume`](crate::element::Sink::consume). For a
    /// video renderer that includes installing or submitting the preview
    /// frame, but not waiting for physical display scanout.
    /// Whether a [`Self::seek`] would be refused, and by what — without
    /// seeking, and without asking anything running.
    ///
    /// Answered from the graph as it stands: a source that is live or cannot
    /// reposition, and a sink that cannot follow a jump in the timeline (see
    /// [`Sink::accepts_seek`](crate::element::Sink::accepts_seek)), say so as
    /// they are wired. So this works before [`Self::run`] and after the
    /// sources have stopped, costs a lock rather than a round trip through
    /// every thread, and changes as branches come and go — a recording
    /// attached to a `Tee` refuses from the moment it is attached until it is
    /// detached. What a player needs to decide whether to offer a seek bar.
    pub fn check_seek(&self) -> std::result::Result<(), crate::control::SeekError> {
        crate::control::SeekError::from_rejections(self.graph.seek_rejections())
    }

    /// Says whether [`Self::set_rate`] with a negative rate would be
    /// refused, and by what, without changing anything — answered from the
    /// graph as [`Self::check_seek`] is. Playing backwards starts with a
    /// seek, so what refuses one refuses it too; besides, every source has
    /// to be a [`crate::element::ReversibleSource`], and every element that
    /// turns a picture's packets into pictures a
    /// [`crate::element::ReversibleDecoder`]. What a player needs to decide
    /// whether to offer it.
    pub fn check_reverse(&self) -> std::result::Result<(), crate::control::SeekError> {
        crate::control::SeekError::from_rejections(self.graph.reverse_rejections())
    }

    pub fn seek(&self, target: Duration, mode: SeekMode) -> Result<()> {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            return Err(PipelineError::NotRunning.into());
        }
        self.check_seek()?;
        // A seek lines everything up itself; nothing a step left out of line
        // is left for a resume to see to.
        self.stepped.store(false, Ordering::Release);
        let playing = !self.paused.load(Ordering::Acquire);

        // A seek is these four stages, each leaving the graph in a state
        // the next relies on — see each for which.
        if playing {
            self.pause_runtime();
        }
        self.reposition(target);
        let preroll = self.preroll_for(|terminals| match mode {
            SeekMode::Keyframe => PrerollContext::new(terminals),
            SeekMode::Accurate => PrerollContext::for_seek(terminals, target),
        });
        let prerolled = self.preroll(&preroll, PREROLL_TIMEOUT);
        self.end_preroll(playing);
        prerolled?;
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={:?} phase=completed outcome=ok",
            ControlMsg::Seek(target)
        );
        Ok(())
    }

    /// A seek's second stage, the graph paused: moves every source to
    /// `target` on a new timeline.
    ///
    /// After it, every source is at `target` or wherever it landed near it
    /// (see [`crate::bus::BusEvent::Seeked`]), nothing any element held from
    /// the old position is left — the `Flush` — and whatever of the old
    /// position reaches a queue later is dropped there, being on a timeline
    /// that is no longer current — see [`crate::timeline`]. Nothing has
    /// moved: every source is still paused.
    fn reposition(&self, target: Duration) {
        let msg = ControlMsg::Seek(target);
        pp_trace!(
            pp_log: &self.pp_log,
            "event=control control={msg:?} phase=requested"
        );
        self.state.interrupt();
        self.playback_clock.reset_for_seek();
        // Everything read from here on belongs to the new position, and a
        // queue drops whatever reaches it from the old one — what the
        // `Flush` below discards, and what it misses.
        self.state.set_backwards(self.rate() < 0.0);
        self.state.begin_timeline();
        self.broadcast(|control_tx| control_tx.enqueue(ControlMsg::Flush));
        self.broadcast(|control_tx| control_tx.enqueue(msg.clone()));
    }

    /// A preroll expecting every terminal the graph has now, as `make`
    /// sets it up, and naming them for a timeout to say which it waited on.
    ///
    /// Playing backwards, only those that show pictures: nothing reaches
    /// the sound in reverse, and waiting on it would be waiting out the
    /// timeout. What does reach another terminal is dropped, as in a step.
    fn preroll_for(
        &self,
        make: impl FnOnce(Vec<crate::graph::ElementId>) -> PrerollContext,
    ) -> Arc<PrerollContext> {
        if self.rate() < 0.0 {
            let (every, pictures) = self.picture_terminals();
            let quiet: Vec<_> = every
                .into_iter()
                .filter(|terminal| !pictures.contains(terminal))
                .collect();
            return self.preroll_expecting(pictures, |terminals| make(terminals).silencing(quiet));
        }
        self.preroll_expecting(self.graph().terminal_ids(), make)
    }

    /// A preroll expecting `terminals`, as `make` sets it up, and naming them
    /// for a timeout to say which it waited on.
    fn preroll_expecting(
        &self,
        terminals: Vec<crate::graph::ElementId>,
        make: impl FnOnce(Vec<crate::graph::ElementId>) -> PrerollContext,
    ) -> Arc<PrerollContext> {
        let graph = self.graph();
        let labels: Vec<_> = terminals
            .iter()
            .filter_map(|&id| graph.node(id).cloned())
            .collect();
        Arc::new(make(terminals).labelled(labels))
    }

    /// A seek's third stage: lets data through the paused graph until every
    /// terminal has taken its sample for `preroll` — or `timeout`, or a
    /// `stop`, ends the wait.
    ///
    /// Paused before and, as far as what flows is concerned, after: each
    /// terminal takes one sample and holds, and a branch with its sample is
    /// held while its siblings catch up — see `Tee`. What ends the preroll
    /// is [`Self::end_preroll`], which must follow whatever this answers.
    fn preroll(
        &self,
        preroll: &Arc<PrerollContext>,
        timeout: Duration,
    ) -> std::result::Result<(), PrerollError> {
        // Published before the wait, so `stop` can end it rather than queue
        // behind it; cleared whatever the wait answers, so no later `stop`
        // cancels a preroll that has already finished.
        self.publish_preroll(preroll);
        self.state.enter(Phase::Prerolling(Arc::clone(preroll)));
        self.broadcast(|control_tx| control_tx.enqueue(ControlMsg::Preroll(Arc::clone(preroll))));
        let prerolled = self.await_preroll(preroll, timeout);
        self.retire_preroll();
        prerolled
    }

    /// Every terminal, and those of them that show pictures: that decoded
    /// video is wired to reach, or — where the wiring does not say — that
    /// have taken some. The wiring first: a screen slower than its sibling
    /// may not have taken its first picture yet, and read as one that shows
    /// none, a step dropped its pictures as it drops the sound.
    fn picture_terminals(&self) -> (Vec<crate::graph::ElementId>, Vec<crate::graph::ElementId>) {
        let every = self.graph().terminal_ids();
        let shown = self.state.picture_terminals(&every);
        let pictures = every
            .iter()
            .copied()
            .filter(|&terminal| {
                self.graph.takes_pictures(terminal) == Some(true) || shown.contains(&terminal)
            })
            .collect();
        (every, pictures)
    }

    /// The slowest speed [`Self::set_rate`] takes, either way.
    pub const MIN_RATE: f64 = 0.25;
    /// The fastest speed [`Self::set_rate`] takes, either way.
    pub const MAX_RATE: f64 = 4.0;
    /// The media's own speed, backwards.
    pub const REVERSE_RATE: f64 = -1.0;

    /// Plays on at `rate` times its own speed, from where playback is: 2.0
    /// covers two seconds of media in each second, 0.5 half of one.
    ///
    /// Nothing is sought and nothing flushed. The playback clock takes the
    /// new rate from the position it has reached, so what paces the picture
    /// — a [`crate::elements::Pacer`], a
    /// [`crate::elements::VideoSynchronizer`] — shows it that much faster or
    /// slower from the next picture on. An audio renderer of this crate
    /// stretches its sound to the rate without changing its pitch, and says
    /// where playback is from what it has played, so the picture stays with
    /// the sound; a moment of sound already in the device plays out at the
    /// rate it was stretched to. Paused, it takes effect as playback goes on.
    /// A seek, a step and a pause leave it as it is.
    ///
    /// A negative rate plays backwards from the picture shown —
    /// [`Self::REVERSE_RATE`] at the media's own speed. Turning round is not
    /// a change of speed: the source goes back over its media a stretch at a
    /// time and the picture's decoder hands each stretch on last picture
    /// first, so everything is sought to the picture shown and prerolled
    /// there, as a seek is — paused or playing, as it was — and the same
    /// happens turning forwards again. Faster or slower the same way round
    /// is only a change of speed, as it is forwards. The sound is not
    /// played backwards: nothing reaches it, and the clock goes on the wall.
    /// Backwards the picture's decoder decodes each stretch from the
    /// keyframe before it, so a fast rate asks more of it than playing that
    /// fast forwards; what it cannot decode in time a pacer shows late.
    ///
    /// Refused where a seek is, before anything changes — a live source
    /// plays at the rate it arrives, and a recording has to be of the stream
    /// as it ran; see [`Self::check_seek`] — backwards also where
    /// [`Self::check_reverse`] says, and before [`Self::run`]; and with
    /// [`PipelineError::UnsupportedRate`] for a speed outside
    /// [`Self::MIN_RATE`]..=[`Self::MAX_RATE`], either way. A forward rate
    /// is taken before [`Self::run`] as well, for playback to start at it.
    pub fn set_rate(&self, rate: f64) -> Result<()> {
        if !(Self::MIN_RATE..=Self::MAX_RATE).contains(&rate.abs()) {
            return Err(PipelineError::UnsupportedRate.into());
        }
        let reverse = rate < 0.0;
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let running = self.running.load(Ordering::Acquire) > 0;
        if reverse {
            if !running {
                return Err(PipelineError::NotRunning.into());
            }
            self.check_reverse()?;
        } else {
            self.check_seek()?;
        }
        let turning = reverse != (self.rate() < 0.0);
        pp_trace!(
            pp_log: &self.pp_log,
            "event=rate rate={rate} turning={turning} phase=requested"
        );
        if !turning || !running {
            self.playback_clock.set_rate(rate);
            return Ok(());
        }
        // Turning around: the picture shown, played the other way — a seek
        // to it, with the rate set before the sources are repositioned, since
        // it is the sign a source reads to go back over its media.
        let (_, pictures) = self.picture_terminals();
        let at = self
            .state
            .picture_at(&pictures)
            .map(|(at, _)| at)
            .or_else(|| self.position())
            .unwrap_or_default();
        self.stepped.store(false, Ordering::Release);
        let playing = !self.paused.load(Ordering::Acquire);
        if playing {
            self.pause_runtime();
        }
        self.playback_clock.set_rate(rate);
        self.reposition(at);
        let preroll = self.preroll_for(|terminals| PrerollContext::for_seek(terminals, at));
        let prerolled = self.preroll(&preroll, PREROLL_TIMEOUT);
        self.end_preroll(playing);
        prerolled?;
        Ok(())
    }

    /// The rate playback goes at — 1.0 until [`Self::set_rate`] says
    /// otherwise.
    pub fn rate(&self) -> f64 {
        self.playback_clock.rate()
    }

    /// Moves the picture by `frames` and holds it there: forward by that many
    /// pictures, or back, and paused either way. Answers where the picture
    /// is now, in its media.
    ///
    /// Forward is a preroll that asks each terminal showing pictures for
    /// `frames` more, taken from wherever its decoder is: no seek, nothing
    /// decoded twice, and a picture decoded past the last one asked for waits
    /// to be the next step's first. Past the end there is nothing more to
    /// take, and the picture stays at the last. Back is an accurate seek to
    /// the instant just before the picture shown, which lands on the one
    /// before it however unevenly the pictures are spaced; each picture
    /// further back is counted in the spacing of the pictures shown, which
    /// is exact for a stream at a constant rate. A step back decodes again
    /// from the keyframe before, each time.
    ///
    /// Nothing but the picture moves. What reaches a terminal that does not
    /// show pictures — the sound — is dropped meanwhile, and the clock is
    /// let go of as a seek lets go of it: [`Self::position`] is `None` until
    /// playback moves on, and this answer is where the picture is.
    /// [`Self::resume`] after a step first seeks to the picture shown, which
    /// puts the sound and the clock back in line with it, and plays on from
    /// there.
    ///
    /// Refused where a seek is, before anything moves — a live source, a
    /// recording; see [`Self::check_seek`] — and with
    /// [`PipelineError::NoPicture`] where no terminal has taken a picture
    /// yet, and [`PipelineError::NotRunning`] as a seek is. A `frames` of zero
    /// pauses and answers where the picture is.
    pub fn step(&self, frames: i64) -> Result<Duration> {
        let _operation = self
            .operation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.running.load(Ordering::Acquire) == 0 {
            return Err(PipelineError::NotRunning.into());
        }
        self.check_seek()?;
        let (every, terminals) = self.picture_terminals();
        let Some((at, spacing)) = self.state.picture_at(&terminals) else {
            return Err(PipelineError::NoPicture.into());
        };
        // What shows no pictures — the sound — is not played meanwhile.
        let quiet: Vec<_> = every
            .into_iter()
            .filter(|terminal| !terminals.contains(terminal))
            .collect();
        // Stepping is something done paused, and it leaves the pipeline so.
        if !self.paused.swap(true, Ordering::AcqRel) {
            self.pause_runtime();
        }
        if frames == 0 {
            return Ok(at);
        }
        self.stepped.store(true, Ordering::Release);
        // Where the clock stood is not where the picture is going.
        self.playback_clock.reset_for_seek();
        // Playing backwards the stream goes down the media: what a step
        // forward takes from it is the picture before, and the picture after
        // is sought to.
        let reverse = self.rate() < 0.0;
        let along = if reverse { -frames } else { frames };
        let stepped = if along > 0 {
            // Only what has pictures left to take: one that has taken the
            // end of its stream never will, and waiting on it would be
            // waiting out the timeout.
            let taking: Vec<_> = terminals
                .iter()
                .copied()
                .filter(|&terminal| !self.completion.has_ended(terminal))
                .collect();
            if taking.is_empty() {
                return Ok(at);
            }
            let count = usize::try_from(along).unwrap_or(usize::MAX);
            let preroll = self.preroll_expecting(taking, |terminals| {
                PrerollContext::for_step(terminals, count).silencing(quiet)
            });
            let stepped = self.preroll(&preroll, PREROLL_TIMEOUT);
            self.end_preroll(false);
            stepped
        } else {
            // One back is the instant just before the picture shown, which
            // lands on the one before it however unevenly they are spaced.
            // Further back is counted in spacings, and aimed at the middle of
            // the picture wanted: positions are whole nanoseconds, and a count
            // of rounded spacings lands on a picture's very start as often as
            // not — which is the picture after.
            //
            // Playing backwards a seek shows the last picture at or before its
            // target, so a picture after the one shown is aimed at the middle
            // of it the same way.
            let back = u32::try_from(frames.unsigned_abs()).unwrap_or(u32::MAX);
            let target = match (reverse, spacing) {
                (false, Some(spacing)) if back > 1 => at
                    .saturating_sub(spacing.saturating_mul(back))
                    .saturating_add(spacing / 2),
                (false, _) => at.saturating_sub(Duration::from_nanos(1)),
                (true, Some(spacing)) => at
                    .saturating_add(spacing.saturating_mul(back))
                    .saturating_add(spacing / 2),
                (true, None) => at,
            };
            self.reposition(target);
            let preroll = self.preroll_expecting(terminals.clone(), |terminals| {
                PrerollContext::for_seek(terminals, target).silencing(quiet)
            });
            let stepped = self.preroll(&preroll, PREROLL_TIMEOUT);
            self.end_preroll(false);
            stepped
        };
        stepped?;
        Ok(self.state.picture_at(&terminals).map_or(at, |(at, _)| at))
    }

    /// A seek's last stage: out of the preroll by way of a pause, then on
    /// playing if `play_on`.
    ///
    /// Always the pause, even to play on. A preroll lets data through, and
    /// playing read off the state before its `Resume` had arrived let a
    /// queue hand a terminal data it had not yet been told to take. Paused,
    /// every source and queue waits for the `Resume` itself and passes it on
    /// before anything else — see `crate::playback_state`.
    fn end_preroll(&self, play_on: bool) {
        self.pause_runtime();
        if play_on {
            self.resume_runtime();
        }
    }
}

/// Seek's preroll wait, reachable without the operation lock.
///
/// `abandoned` is sticky because the calls that set it — `stop` and `finish` —
/// both end the pipeline for good. Once set, a seek that has not yet published
/// its preroll cancels it on arrival instead of waiting out a timeout nobody
/// is going to collect.
#[derive(Default)]
pub(super) struct PrerollSlot {
    active: Option<Arc<PrerollContext>>,
    abandoned: bool,
}
