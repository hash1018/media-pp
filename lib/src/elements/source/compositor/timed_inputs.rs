//! What an offline compositor keeps of each input it is fed through a sink:
//! that input's frames in time order, as many as it takes to know which one
//! the output time being made shows — see [`RenderMode::Offline`].
//!
//! Shared by every backend. They all hand frames around as the same pooled
//! reference, and what decides which frame is shown when depends on nothing
//! but timestamps.
//!
//! [`RenderMode::Offline`]: crate::elements::RenderMode::Offline

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard, Weak},
    time::Duration,
};

use ffmpeg_next as ffmpeg;

use crate::{
    buffer,
    bus::Bus,
    control::{ControlReceiver, drain_control},
    element::SourceElement,
    elements::source::render_mode::MediaTime,
    error::Result,
    playback_state::{Bell, PlaybackState},
    pool::UnboundObjectPoolRef,
    pp_log::pp_info,
};

pub(crate) type Picture = Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>;

/// Why a frame could not be placed in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Untimed {
    /// The frame has no `pts`.
    NoTimestamp,
    /// The frame says no time base, or one that is not positive — see
    /// [`buffer::time_base`].
    NoTimeBase,
}

/// What one input shows at one output time.
pub(crate) enum Shown {
    /// Not known yet: a frame that settles it has still to arrive.
    Pending,
    /// Nothing — the input has not begun, is between frames, or is over.
    Nothing,
    /// This frame.
    Frame(Picture),
}

/// One input's frames, from the sink that is given them to the compositor
/// that draws them.
pub(crate) struct TimedFeed {
    state: Mutex<FeedState>,
    /// The compositor's, rung whenever something arrives here.
    arrived: Bell,
    /// The pipeline feeding this input, rung as room is made — set as the
    /// sink is wired into it, and absent for a sink driven by hand, which
    /// nothing waits in front of.
    upstream: Mutex<Weak<PlaybackState>>,
}

struct FeedState {
    frames: VecDeque<Timed>,
    ended: bool,
    /// The output time the compositor wants next. Frames up to it are what
    /// it will draw; one past it is all the look-ahead that is needed.
    wanted: MediaTime,
}

struct Timed {
    picture: Picture,
    start: MediaTime,
    /// Where its duration runs out, where it says one.
    end: Option<MediaTime>,
}

impl TimedFeed {
    pub(crate) fn new(arrived: Bell, wanted: MediaTime) -> Self {
        Self {
            state: Mutex::new(FeedState {
                frames: VecDeque::new(),
                ended: false,
                wanted,
            }),
            arrived,
            upstream: Mutex::new(Weak::new()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, FeedState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Remembers the pipeline this input's sink was wired into, to ring as
    /// room is made.
    pub(crate) fn fed_by(&self, state: &Arc<PlaybackState>) {
        *self
            .upstream
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::downgrade(state);
    }

    /// Takes one frame, placed by its own timestamp in its own time base.
    ///
    /// Frames before the output time already reached are only the last of
    /// them worth keeping — every later output time is past them — so one
    /// that arrives there replaces them.
    pub(crate) fn push(&self, picture: Picture) -> std::result::Result<(), Untimed> {
        let pts = picture.pts().ok_or(Untimed::NoTimestamp)?;
        let base = buffer::time_base(&picture).ok_or(Untimed::NoTimeBase)?;
        let start = MediaTime::new(pts, base).ok_or(Untimed::NoTimeBase)?;
        // SAFETY: a plain field of a frame this holds a reference to.
        let duration = unsafe { (*picture.as_ptr()).duration };
        let end = (duration > 0).then(|| start.plus(duration));
        {
            let mut state = self.lock();
            let wanted = state.wanted;
            if start <= wanted {
                while state.frames.back().is_some_and(|last| last.start <= start) {
                    state.frames.pop_back();
                }
            }
            state.frames.push_back(Timed {
                picture,
                start,
                end,
            });
        }
        self.arrived.ring();
        Ok(())
    }

    /// Says there will be no more frames: what is held is shown to its end,
    /// and the input then stops being waited for.
    pub(crate) fn end(&self) {
        self.lock().ended = true;
        self.arrived.ring();
    }

    /// Drops every frame and the end with them — a flush.
    pub(crate) fn clear(&self) {
        let mut state = self.lock();
        state.frames.clear();
        state.ended = false;
    }

    /// Whether the sink can take another frame: not once it holds one that
    /// starts after the output time wanted next, which is all that has to
    /// be known about what comes after.
    pub(crate) fn wants_more(&self) -> bool {
        let state = self.lock();
        !state
            .frames
            .back()
            .is_some_and(|last| last.start > state.wanted)
    }

    /// What this input shows at `at`, an output time counted in output
    /// frames — one unit of its base is one output interval.
    ///
    /// A frame is shown from its start until its duration runs out or the
    /// next frame begins, whichever is first. The last frame of an input
    /// that has ended and says no duration is shown for the one output
    /// interval it starts in: nothing else says how long it lasts.
    pub(crate) fn shown_at(&self, at: MediaTime) -> Shown {
        let mut state = self.lock();
        // Superseded: the next frame has already begun.
        while state.frames.len() >= 2 && state.frames[1].start <= at {
            state.frames.pop_front();
        }
        let ended = state.ended;
        let Some(first) = state.frames.front() else {
            return if ended {
                Shown::Nothing
            } else {
                Shown::Pending
            };
        };
        if first.start > at {
            // Frames arrive in presentation order, so none earlier is coming.
            return Shown::Nothing;
        }
        let next_start = state.frames.get(1).map(|next| next.start);
        match first.end.or(next_start) {
            Some(end) if at < end => Shown::Frame(Arc::clone(&first.picture)),
            // A gap before the next frame, which starts later than `at`.
            Some(_) if next_start.is_some() || ended => Shown::Nothing,
            // Out before the next is known to have begun: it may yet start
            // at or before `at`.
            Some(_) => Shown::Pending,
            None if ended => {
                if first.start > at.plus(-1) {
                    Shown::Frame(Arc::clone(&first.picture))
                } else {
                    Shown::Nothing
                }
            }
            None => Shown::Pending,
        }
    }

    /// Whether this input has nothing left to show at `at` or after — `at`
    /// counted in output frames, as [`Self::shown_at`] takes it.
    pub(crate) fn over_at(&self, at: MediaTime) -> bool {
        let state = self.lock();
        if !state.ended {
            return false;
        }
        state.frames.iter().all(|frame| match frame.end {
            Some(end) => end <= at,
            // Shown for the one interval it starts in: over once `at` is
            // an interval or more past its start.
            None => frame.start <= at.plus(-1),
        })
    }

    /// Moves the output time wanted next on to `wanted`, and rings the
    /// pipeline in front if that made room.
    pub(crate) fn advance(&self, wanted: MediaTime) {
        let room = {
            let mut state = self.lock();
            state.wanted = wanted;
            !state.frames.back().is_some_and(|last| last.start > wanted)
        };
        if room
            && let Some(upstream) = self
                .upstream
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .upgrade()
        {
            upstream.wake();
        }
    }
}

/// One input of an offline compositor, as [`run_offline`] sees it.
pub(crate) trait TimedInput {
    /// Its frames, where it is fed through a sink.
    fn timed(&self) -> Option<&TimedFeed>;
    /// Makes `picture` what it draws in the next frame composed.
    fn show(&self, picture: Option<Picture>);
}

/// What [`run_offline`] needs of a compositor. The backends differ in how
/// they draw, and in nothing this loop does.
pub(crate) trait OfflineCompositor: SourceElement {
    type Input: TimedInput;
    /// Every input registered now.
    fn inputs(&self) -> Vec<Arc<Self::Input>>;
    /// The output frame to be made next.
    fn frame_index(&self) -> i64;
    /// Output frame `index` as a time, counted in output frames.
    fn output_time(&self, index: i64) -> MediaTime;
    /// Whether an input has ever been fed through a sink.
    fn fed(&self) -> bool;
    /// What the input sinks ring as anything arrives.
    fn arrived(&self) -> Bell;
    /// Says output frame `index` is the one being made next, which is where
    /// an input added from now on starts.
    fn making(&self, index: i64);
    /// Composes and pushes one frame from what the inputs show, moving
    /// [`Self::frame_index`] on by one.
    fn draw(&mut self, bus: &Bus) -> Result<()>;
    /// Pushes the `Eos` that ends the render.
    fn end_render(&mut self) -> Result<()>;
}

/// An offline compositor's run: each output time's frame once every input
/// has said what it shows then — see [`RenderMode::Offline`].
///
/// No schedule: what paces this is the inputs arriving and the pad taking
/// what is pushed. Each pass settles every input fed through a sink for the
/// output time [`OfflineCompositor::frame_index`] stands for, draws, and
/// moves those inputs on, which rings their pipelines to send the next.
///
/// [`RenderMode::Offline`]: crate::elements::RenderMode::Offline
pub(crate) fn run_offline<C: OfflineCompositor>(
    compositor: &mut C,
    control: &ControlReceiver,
    bus: &Bus,
    end: Option<Duration>,
) -> Result<()> {
    let end = end.map(MediaTime::from_duration);
    let arrived = compositor.arrived();
    loop {
        let outcome = drain_control(control, compositor, bus)?;
        if outcome.stopped {
            pp_info!(pp_log: compositor.pp_log(), "stopped");
            return Ok(());
        }

        let index = compositor.frame_index();
        let at = compositor.output_time(index);
        if end.is_some_and(|end| at >= end) {
            return finish(compositor, index);
        }
        let inputs = compositor.inputs();
        let mut settled = true;
        let mut over = true;
        for input in &inputs {
            let Some(timed) = input.timed() else {
                continue;
            };
            match timed.shown_at(at) {
                Shown::Pending => settled = false,
                Shown::Nothing => input.show(None),
                Shown::Frame(picture) => input.show(Some(picture)),
            }
            over &= timed.over_at(at);
        }
        // With no end of its own, the render is over once every input it was
        // fed has ended and shown its last frame — but not before one has
        // been added, when "every input" is none.
        let fed = compositor.fed();
        if end.is_none() && fed && over {
            return finish(compositor, index);
        }
        if !settled || (end.is_none() && !fed) {
            // Woken as anything arrives; the timeout is only so control is
            // looked at while nothing does.
            crossbeam_channel::select! {
                recv(arrived.rings()) -> _ => {}
                default(CONTROL_POLL_INTERVAL) => {}
            }
            continue;
        }

        compositor.draw(bus)?;
        let next = compositor.frame_index();
        compositor.making(next);
        let wanted = compositor.output_time(next);
        for input in &inputs {
            if let Some(timed) = input.timed() {
                timed.advance(wanted);
            }
        }
    }
}

/// Ends an offline render before output frame `index`: an `Eos` after the
/// last frame, and the source's run is over.
fn finish<C: OfflineCompositor>(compositor: &mut C, index: i64) -> Result<()> {
    pp_info!(pp_log: compositor.pp_log(), "finished: {index} frames");
    compositor.end_render()
}

/// How long the loop waits for an input before looking at control again.
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(5);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::UnboundObjectPool;

    const TENTHS: ffmpeg::Rational = ffmpeg::Rational(1, 10);

    fn frame(pts: i64, duration: i64) -> Picture {
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 2, 2),
            |_| {},
        );
        let mut frame = pool.get();
        frame.set_pts(Some(pts));
        buffer::set_time_base(&mut frame, TENTHS);
        // SAFETY: a plain field of a frame this test owns outright.
        unsafe { (*frame.as_mut_ptr()).duration = duration };
        Arc::new(frame)
    }

    /// Output times in tenths of a second, as a ten-frame output counts.
    fn at(tenths: i64) -> MediaTime {
        MediaTime::new(tenths, TENTHS).unwrap()
    }

    fn shows(feed: &TimedFeed, tenths: i64) -> Option<Option<i64>> {
        match feed.shown_at(at(tenths)) {
            Shown::Pending => None,
            Shown::Nothing => Some(None),
            Shown::Frame(picture) => Some(picture.pts()),
        }
    }

    /// A frame is shown until its duration runs out, and a gap after it is
    /// shown as nothing only once the next frame says the gap is real.
    #[test]
    fn a_gap_between_frames_is_nothing_once_it_is_known() {
        let feed = TimedFeed::new(Bell::new(), at(0));
        feed.push(frame(0, 2)).unwrap();
        assert_eq!(shows(&feed, 1), Some(Some(0)));
        assert_eq!(shows(&feed, 2), None, "the next frame may start at 2");
        feed.push(frame(4, 1)).unwrap();
        assert_eq!(shows(&feed, 2), Some(None));
        assert_eq!(shows(&feed, 4), Some(Some(4)));
    }

    /// The last frame of an ended input that says no duration lasts the one
    /// output interval it starts in; after that the input is over.
    #[test]
    fn a_last_frame_without_duration_lasts_one_interval() {
        let feed = TimedFeed::new(Bell::new(), at(0));
        feed.push(frame(3, 0)).unwrap();
        assert_eq!(shows(&feed, 3), None, "not ended: it may yet be followed");
        feed.end();
        assert_eq!(shows(&feed, 2), Some(None));
        assert_eq!(shows(&feed, 3), Some(Some(3)));
        assert!(!feed.over_at(at(3)));
        assert_eq!(shows(&feed, 4), Some(None));
        assert!(feed.over_at(at(4)));
    }

    /// The sink takes frames until it holds one past the output time wanted
    /// next, and is open again once the compositor moves past it.
    #[test]
    fn it_holds_back_one_frame_ahead_until_the_output_moves_on() {
        let feed = TimedFeed::new(Bell::new(), at(0));
        assert!(feed.wants_more());
        feed.push(frame(0, 1)).unwrap();
        assert!(feed.wants_more(), "a frame at the wanted time is not ahead");
        feed.push(frame(1, 1)).unwrap();
        assert!(
            !feed.wants_more(),
            "one past it is all the look-ahead needed"
        );
        feed.advance(at(1));
        assert!(feed.wants_more());
    }
}
