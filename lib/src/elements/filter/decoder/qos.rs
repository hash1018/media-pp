//! What a video decoder leaves undecoded while the pictures it hands on
//! arrive too late to be shown in time.
//!
//! A pacer or synchronizer that hands a picture on late says how late in
//! the pipeline's [`PlaybackState`]. A decoder reads it before each packet
//! and, while pictures keep coming late, decodes less: first no picture
//! that nothing refers to — the B-frames, commonly — then keyframes only.
//! Once pictures come on time again for a while, it goes back up a step.
//! What is shown stays at the right place in time; what is lost is the
//! pictures between.
//!
//! Playing backwards needs it most: each stretch is decoded from the
//! keyframe before it, so the work to show a second of it is as much as
//! the gap between keyframes. A preroll always decodes everything, since
//! what a seek or a step asks for is one particular picture.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use ffmpeg_next::{self as ffmpeg, ffi};

use crate::{
    playback_state::PlaybackState,
    pp_log::{PpLog, pp_debug},
};

/// How late a picture has to be before less is decoded.
const BEHIND: Duration = Duration::from_millis(150);

/// How early — on time — pictures have to come before more is decoded.
const CAUGHT_UP: Duration = Duration::from_millis(20);

/// How long a step down is given to show whether it is enough, before the
/// next.
const SETTLE: Duration = Duration::from_millis(500);

/// How long pictures have to come on time before a step back up is tried,
/// at first. Doubled each time a step up had to be taken back soon after,
/// so a decoder that cannot keep up at the step above stops trying it
/// every few seconds.
const RECOVER: Duration = Duration::from_secs(2);

/// The longest [`RECOVER`] grows to.
const RECOVER_AT_MOST: Duration = Duration::from_secs(16);

/// What a decoder decodes, most first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    All,
    References,
    Keyframes,
}

impl Level {
    fn discard(self) -> ffi::AVDiscard {
        match self {
            Level::All => ffi::AVDiscard::AVDISCARD_DEFAULT,
            Level::References => ffi::AVDiscard::AVDISCARD_NONREF,
            Level::Keyframes => ffi::AVDiscard::AVDISCARD_NONKEY,
        }
    }

    fn less(self) -> Self {
        match self {
            Level::All => Level::References,
            Level::References | Level::Keyframes => Level::Keyframes,
        }
    }

    fn more(self) -> Self {
        match self {
            Level::All | Level::References => Level::All,
            Level::Keyframes => Level::References,
        }
    }
}

/// One video decoder's side of it — see this module's docs.
pub(super) struct Qos {
    /// The pipeline's playback state; `None` for a decoder no pipeline
    /// wired, which decodes everything.
    state: Option<Arc<PlaybackState>>,
    level: Level,
    /// When the level last changed.
    since: Instant,
    /// Whether the last change was a step back up.
    stepped_up: bool,
    /// Since when pictures have come on time, uninterrupted.
    on_time_since: Option<Instant>,
    /// How long pictures have to come on time before the next step up.
    recover: Duration,
}

impl Default for Qos {
    fn default() -> Self {
        Self {
            state: None,
            level: Level::All,
            since: Instant::now(),
            stepped_up: false,
            on_time_since: None,
            recover: RECOVER,
        }
    }
}

impl Qos {
    pub(super) fn attach(&mut self, state: &Arc<PlaybackState>) {
        self.state = Some(Arc::clone(state));
    }

    /// Decides what to decode of the next packet from how late pictures
    /// are, and tells `decoder` if that changed.
    pub(super) fn follow(&mut self, decoder: &mut ffmpeg::decoder::Video, pp_log: &PpLog) {
        let wanted = match &self.state {
            None => Level::All,
            Some(state) if state.is_prerolling() => Level::All,
            Some(state) => {
                let late = state.picture_lateness();
                if late >= CAUGHT_UP {
                    self.on_time_since = None;
                } else if self.on_time_since.is_none() {
                    self.on_time_since = Some(Instant::now());
                }
                let settled = self.since.elapsed();
                if late > BEHIND && settled > SETTLE {
                    if self.stepped_up && settled < self.recover {
                        // Up too soon: wait longer before trying again.
                        self.recover = (self.recover * 2).min(RECOVER_AT_MOST);
                    }
                    self.level.less()
                } else if self
                    .on_time_since
                    .is_some_and(|since| since.elapsed() > self.recover)
                {
                    self.level.more()
                } else {
                    self.level
                }
            }
        };
        if wanted == self.level {
            return;
        }
        pp_debug!(
            pp_log: pp_log,
            "event=qos decode={wanted:?} was={:?} late={:?}",
            self.level,
            self.state
                .as_ref()
                .map_or(Duration::ZERO, |state| state.picture_lateness())
        );
        self.stepped_up = wanted < self.level;
        self.level = wanted;
        self.since = Instant::now();
        self.on_time_since = None;
        // SAFETY: `decoder` is an open codec context this decoder owns;
        // `skip_frame` is read by libavcodec before each frame it decodes,
        // and may be changed between packets.
        unsafe {
            (*decoder.as_mut_ptr()).skip_frame = wanted.discard();
        }
    }

    /// Everything again, for a new timeline: what made pictures late on
    /// the old one says nothing of the new.
    pub(super) fn reset(&mut self, decoder: &mut ffmpeg::decoder::Video) {
        self.level = Level::All;
        self.since = Instant::now();
        self.stepped_up = false;
        self.on_time_since = None;
        self.recover = RECOVER;
        // SAFETY: as in `follow`.
        unsafe {
            (*decoder.as_mut_ptr()).skip_frame = Level::All.discard();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::control::{ControlMsg, PrerollContext};
    use crate::element::{ElementType, element_pp_log};

    fn decoder() -> ffmpeg::decoder::Video {
        crate::ensure_ffmpeg();
        let codec = ffmpeg::decoder::find(ffmpeg::codec::Id::H264).expect("an H.264 decoder");
        ffmpeg::codec::context::Context::new_with_codec(codec)
            .decoder()
            .video()
            .expect("open it")
    }

    fn skipping(decoder: &ffmpeg::decoder::Video) -> ffi::AVDiscard {
        // SAFETY: a live, open codec context.
        unsafe { (*decoder.as_ptr()).skip_frame }
    }

    /// Late, it decodes less a step at a time, giving each step time to
    /// show whether it is enough; a preroll decodes everything whatever
    /// came before, and so does a new timeline.
    #[test]
    fn late_pictures_leave_less_decoded_a_step_at_a_time() {
        let log = element_pp_log(ElementType::Other, "qos", None);
        let state = PlaybackState::new();
        let mut qos = Qos::default();
        qos.attach(&state);
        let mut decoder = decoder();

        qos.follow(&mut decoder, &log);
        assert_eq!(skipping(&decoder), ffi::AVDiscard::AVDISCARD_DEFAULT);

        state.picture_late(Duration::from_millis(300));
        thread::sleep(SETTLE + Duration::from_millis(50));
        qos.follow(&mut decoder, &log);
        assert_eq!(skipping(&decoder), ffi::AVDiscard::AVDISCARD_NONREF);
        qos.follow(&mut decoder, &log);
        assert_eq!(
            skipping(&decoder),
            ffi::AVDiscard::AVDISCARD_NONREF,
            "one step, then time to show"
        );
        thread::sleep(SETTLE + Duration::from_millis(50));
        qos.follow(&mut decoder, &log);
        assert_eq!(skipping(&decoder), ffi::AVDiscard::AVDISCARD_NONKEY);

        let preroll = ControlMsg::Preroll(Arc::new(PrerollContext::for_seek(
            [],
            Duration::from_secs(1),
        )));
        state.observe(&preroll);
        qos.follow(&mut decoder, &log);
        assert_eq!(
            skipping(&decoder),
            ffi::AVDiscard::AVDISCARD_DEFAULT,
            "what a preroll asks for is one particular picture"
        );

        qos.reset(&mut decoder);
        assert_eq!(skipping(&decoder), ffi::AVDiscard::AVDISCARD_DEFAULT);
    }

    /// On time again, it goes back up a step only once pictures have come
    /// on time for a while.
    #[test]
    fn on_time_again_it_decodes_more_after_a_while() {
        let log = element_pp_log(ElementType::Other, "qos", None);
        let state = PlaybackState::new();
        let mut qos = Qos::default();
        qos.attach(&state);
        let mut decoder = decoder();
        state.picture_late(Duration::from_millis(300));
        thread::sleep(SETTLE + Duration::from_millis(50));
        qos.follow(&mut decoder, &log);
        assert_eq!(skipping(&decoder), ffi::AVDiscard::AVDISCARD_NONREF);

        state.picture_late(Duration::ZERO);
        qos.follow(&mut decoder, &log);
        thread::sleep(RECOVER / 2);
        qos.follow(&mut decoder, &log);
        assert_eq!(
            skipping(&decoder),
            ffi::AVDiscard::AVDISCARD_NONREF,
            "not at once"
        );
        thread::sleep(RECOVER / 2 + Duration::from_millis(100));
        qos.follow(&mut decoder, &log);
        assert_eq!(skipping(&decoder), ffi::AVDiscard::AVDISCARD_DEFAULT);
    }
}
