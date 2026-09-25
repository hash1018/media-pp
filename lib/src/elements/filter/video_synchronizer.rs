use std::{collections::VecDeque, sync::Arc, time::Duration};

use crate::pp_log::{PpLog, pp_debug, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    playback_clock::{PlaybackClock, PlaybackMaster},
    time::{MediaTimestamp, TimeBase},
};

const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const FALLBACK_FRAME_DURATION: Duration = Duration::from_millis(40);

fn nanoseconds() -> TimeBase {
    TimeBase::new_unchecked(ffmpeg::Rational::new(1, 1_000_000_000))
}

#[derive(Debug, ThisError)]
/// Input this element cannot schedule.
///
/// Scheduling needs a decoded video frame with a PTS and the time base it is
/// in; anything else is rejected rather than passed through unscheduled.
pub enum VideoSynchronizerError {
    /// A frame arrived with a `pts` but no unit to read it in.
    ///
    /// Every element in this crate that makes a frame says what unit its
    /// `pts` is in — see [`crate::buffer::time_base`] — so this is a frame
    /// made elsewhere. Stamp it with [`crate::buffer::set_time_base`].
    /// Refused rather than guessed at: a guessed unit plays at the wrong
    /// speed, or against the audio at the wrong offset, with no other sign.
    #[error(
        "a Video frame arrived with a pts but no time base to read it in; the element \
         that made it has to set one (media_pp::buffer::set_time_base)"
    )]
    NoTimeBase,

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("VideoSynchronizer only schedules decoded Video frames, got a {0}")]
    UnsupportedBuffer(&'static str),

    /// A decoded video frame has no presentation timestamp to schedule.
    #[error("VideoSynchronizer cannot schedule a video frame without a PTS")]
    MissingPts,
}

enum Decision {
    Render,
    Drop,
    Wait(Duration),
    Hold,
}

/// Schedules decoded video against the pipeline's current playback master.
///
/// In wall-master mode this replaces [`crate::elements::Pacer`]: the first
/// video PTS establishes the media origin and early frames wait. Once an
/// audio renderer registers and starts, the same instance automatically
/// compares video PTS with the played-audio position, waiting for early
/// frames and dropping frames more than one frame-duration late — each frame
/// due as much before its time as the renderer after it says it takes to put
/// a picture on the screen, so the two reach the viewer together. During
/// audio priming it holds the in-flight frame so the wall-to-audio handoff
/// cannot make the picture run ahead.
///
/// Do not put a `Pacer` in the same video branch; that would pace twice.
/// Put a [`crate::queue::Queue`] upstream so waits do not block demux/decode.
pub struct VideoSynchronizer {
    pp_log: PpLog,
    name: Arc<str>,
    /// The pipeline's, given by `Element::attach_context` — see there for
    /// why this is not something the caller supplies.
    playback_clock: Option<Arc<PlaybackClock>>,
    interrupt_epoch: u64,
    /// Preroll forwards frames without waiting on the paused playback clock.
    prerolling: bool,
    /// The last frame's presentation time, in nanoseconds — kept in one
    /// unit rather than the frame's own, so a stream whose unit changes
    /// still measures its frame spacing correctly.
    last_ns: Option<i64>,
    frame_duration: Duration,
    pending: VecDeque<MediaBuffer>,
    pad: SrcPad,
}

impl VideoSynchronizer {
    /// Creates a video scheduler. Each frame says what unit its own `pts` is
    /// in — see [`crate::buffer::time_base`] — so there is nothing here to
    /// be told, and nothing to be told wrong.
    pub fn new(name: impl Into<String>) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoSynchronizer, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name: name.clone(),
            pp_log,
            playback_clock: None,
            interrupt_epoch: 0,
            prerolling: false,
            last_ns: None,
            frame_duration: FALLBACK_FRAME_DURATION,
            pending: VecDeque::new(),
            pad: SrcPad::with_contract(format!("{name}_src"), OutputContract::Passthrough),
        }
    }

    /// When a frame is due, in nanoseconds of media time, read in the unit
    /// the frame itself carries.
    fn frame_ns(frame: &ffmpeg::frame::Video) -> Result<i64, VideoSynchronizerError> {
        let pts = frame.pts().ok_or(VideoSynchronizerError::MissingPts)?;
        let time_base =
            crate::buffer::time_base(frame).ok_or(VideoSynchronizerError::NoTimeBase)?;
        Ok(
            MediaTimestamp::new_unchecked(pts, TimeBase::new_unchecked(time_base))
                .rescale(nanoseconds()),
        )
    }

    fn observe_frame_duration(&mut self, frame_ns: i64) {
        if let Some(delta) = self.last_ns.and_then(|last| frame_ns.checked_sub(last))
            && delta > 0
        {
            self.frame_duration = ns_duration(delta);
        }
        self.last_ns = Some(frame_ns);
    }

    #[cfg(test)]
    fn decision(&mut self, frame_ns: i64) -> Decision {
        self.observe_frame_duration(frame_ns);
        self.decision_without_observing(frame_ns)
    }

    fn wait_for(&mut self, frame_ns: i64) -> WaitOutcome {
        if self.prerolling {
            return WaitOutcome::Render;
        }
        self.observe_frame_duration(frame_ns);
        // Unwired: nothing has given this one a pipeline, so there is no
        // position to schedule against and nothing to interrupt it either.
        // Rendering is the only answer that does not stall a branch that
        // was built wrong — and `attach_context` runs before any frame can
        // reach here, so it is unreachable through ordinary wiring.
        let Some(playback_clock) = self.playback_clock.clone() else {
            return WaitOutcome::Render;
        };
        loop {
            if playback_clock.interrupted_since(self.interrupt_epoch) {
                return WaitOutcome::Interrupted;
            }
            match self.decision_without_observing(frame_ns) {
                Decision::Render => return WaitOutcome::Render,
                Decision::Drop => return WaitOutcome::Drop,
                Decision::Wait(wait) => {
                    playback_clock.sleep_unless_interrupted(wait.min(INTERRUPT_POLL_INTERVAL));
                }
                Decision::Hold => playback_clock.sleep_unless_interrupted(INTERRUPT_POLL_INTERVAL),
            }
        }
    }

    fn decision_without_observing(&self, frame_ns: i64) -> Decision {
        let Some(playback_clock) = &self.playback_clock else {
            return Decision::Render;
        };
        let (master, position) = playback_clock.video_snapshot(frame_ns);
        match master {
            PlaybackMaster::Unavailable => Decision::Render,
            PlaybackMaster::AudioPriming => Decision::Hold,
            PlaybackMaster::Wall => match position {
                Some(position_ns) if frame_ns > position_ns => {
                    Decision::Wait(ns_duration(frame_ns.saturating_sub(position_ns)))
                }
                _ => Decision::Render,
            },
            PlaybackMaster::Audio => {
                let Some(position_ns) = position else {
                    return Decision::Hold;
                };
                // Handed over early by what the renderer takes to show it, so
                // it is on the screen when its sound is heard — the audio
                // position is already what the listener hears, device and all.
                let frame_ns =
                    frame_ns.saturating_sub(duration_ns(playback_clock.presentation_delay()));
                if frame_ns > position_ns {
                    Decision::Wait(ns_duration(frame_ns.saturating_sub(position_ns)))
                } else if position_ns.saturating_sub(frame_ns) > duration_ns(self.frame_duration) {
                    Decision::Drop
                } else {
                    Decision::Render
                }
            }
        }
    }
}

enum WaitOutcome {
    Render,
    Drop,
    Interrupted,
}

impl Element for VideoSynchronizer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoSynchronizer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        self.interrupt_epoch = context.playback_clock.interrupt_epoch();
        self.playback_clock = Some(Arc::clone(&context.playback_clock));
    }
}

impl Source for VideoSynchronizer {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VideoSynchronizer {
    /// Scheduling is a delay, not a transform: frames are held until the
    /// playback clock says they are due and forwarded unchanged. No
    /// memory-domain claim, because it never touches the pixels — it
    /// paces a system frame and a device texture alike.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::any_frame(MediaKind::VideoFrame))
    }

    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match &buf {
            MediaBuffer::Video(_) | MediaBuffer::Eos => {}
            other => return Err(VideoSynchronizerError::UnsupportedBuffer(other.kind()).into()),
        }

        self.pending.push_back(buf);
        while let Some(buf) = self.pending.pop_front() {
            let outcome = match &buf {
                MediaBuffer::Video(frame) => {
                    let frame_ns = Self::frame_ns(frame)?;
                    self.wait_for(frame_ns)
                }
                MediaBuffer::Eos => WaitOutcome::Render,
                _ => unreachable!("buffer kind validated before queueing"),
            };
            match outcome {
                WaitOutcome::Render => self.pad.push(buf)?,
                WaitOutcome::Drop => pp_debug!(self, "dropping late video frame"),
                WaitOutcome::Interrupted => {
                    self.pending.push_front(buf);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn control(&mut self, msg: &ControlMsg) -> crate::error::Result<()> {
        if let Some(playback_clock) = &self.playback_clock {
            self.interrupt_epoch = playback_clock.interrupt_epoch();
        }
        match msg {
            ControlMsg::Flush | ControlMsg::Stop => {
                self.pending.clear();
                self.last_ns = None;
                self.frame_duration = FALLBACK_FRAME_DURATION;
            }
            ControlMsg::Preroll(_) => {
                self.prerolling = true;
            }
            ControlMsg::Pause | ControlMsg::Resume => {
                self.prerolling = false;
            }
            ControlMsg::Seek(_) | ControlMsg::CheckSeek(_) => {}
        }
        Ok(())
    }
}

fn ns_duration(ns: i64) -> Duration {
    Duration::from_nanos(ns.max(0) as u64)
}

fn duration_ns(duration: Duration) -> i64 {
    duration.as_nanos().min(i64::MAX as u128) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        clock::Clock, control::PrerollContext, playback_clock::PlaybackClock,
        pool::UnboundObjectPool,
    };

    /// A synchronizer wired the way a pipeline wires one, and the playback
    /// clock it will schedule against.
    ///
    /// The clock comes back out of the context rather than going in: a test
    /// cannot hand one to the element any more, which is the point of the
    /// change these tests are checking.
    fn synchronizer() -> (VideoSynchronizer, Arc<PlaybackClock>) {
        let context = Arc::new(crate::element::Context::for_test_with_clock(
            crate::bus::Bus::new().0,
            "test",
            crate::graph::PipelineGraph::new(),
            crate::graph::ElementId::for_test(1),
            Arc::new(Clock::new()),
        ));
        let mut sync = VideoSynchronizer::new("sync");
        sync.attach_context(&context);
        (sync, Arc::clone(&context.playback_clock))
    }

    #[test]
    fn first_video_timestamp_establishes_wall_origin() {
        let (mut sync, playback) = synchronizer();
        assert!(matches!(sync.decision(ms(5_000)), Decision::Render));
        assert_eq!(playback.master(), PlaybackMaster::Wall);
        assert!(playback.position_ns().unwrap() >= 5_000_000_000);
    }

    #[test]
    fn audio_priming_holds_video_and_audio_master_drops_late_frames() {
        let (mut sync, playback) = synchronizer();
        let audio = playback.register_audio_master().unwrap();
        assert!(matches!(sync.decision(ms(1_000)), Decision::Hold));

        audio.publish(2_000_000_000, 3_000_000_000, false).unwrap();
        assert!(matches!(sync.decision(ms(1_000)), Decision::Drop));
        assert!(matches!(sync.decision(ms(2_010)), Decision::Wait(_)));
    }

    /// With a renderer that takes 20 ms to show a picture, a frame due 15 ms
    /// from now is handed over at once and one due 30 ms from now waits about
    /// 10: each is on the screen when its sound is heard.
    #[test]
    fn frames_are_handed_over_early_by_the_presentation_delay() {
        let (mut sync, playback) = synchronizer();
        let audio = playback.register_audio_master().unwrap();
        audio.publish(2_000_000_000, 3_000_000_000, false).unwrap();
        assert!(matches!(sync.decision(ms(2_015)), Decision::Wait(_)));

        let screen = playback.register_presenter();
        screen.publish(Duration::from_millis(20));
        assert!(matches!(sync.decision(ms(2_015)), Decision::Render));
        let Decision::Wait(wait) = sync.decision(ms(2_030)) else {
            panic!("a frame due after the delay still waits");
        };
        assert!(wait <= Duration::from_millis(10), "{wait:?}");
        drop(screen);
        assert!(matches!(sync.decision(ms(2_015)), Decision::Wait(_)));
    }

    /// `ms` milliseconds of media time, as the decisions below take it.
    fn ms(ms: i64) -> i64 {
        ms * 1_000_000
    }

    /// A frame due `pts` in `unit`, saying so.
    fn frame(pts: i64, unit: ffmpeg::Rational) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut frame = pool.get();
        frame.set_pts(Some(pts));
        crate::buffer::set_time_base(&mut frame, unit);
        MediaBuffer::Video(Arc::new(frame))
    }

    /// Each frame is read in its own unit: the same count in milliseconds
    /// and in seconds is two very different times.
    #[test]
    fn a_frame_is_read_in_the_unit_it_carries() {
        let in_ms = frame(1_500, ffmpeg::Rational::new(1, 1_000));
        let in_s = frame(1_500, ffmpeg::Rational::new(1, 1));
        let ns = |buffer: &MediaBuffer| match buffer {
            MediaBuffer::Video(frame) => VideoSynchronizer::frame_ns(frame).unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(ns(&in_ms), 1_500_000_000);
        assert_eq!(ns(&in_s), 1_500_000_000_000);
    }

    #[test]
    fn unschedulable_input_is_a_typed_error() {
        let (mut sync, _playback) = synchronizer();
        assert!(matches!(
            sync.consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty()))),
            Err(crate::error::Error::VideoSynchronizerError(_))
        ));

        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        assert!(matches!(
            sync.consume(MediaBuffer::Video(Arc::new(pool.get()))),
            Err(crate::error::Error::VideoSynchronizerError(
                VideoSynchronizerError::MissingPts
            ))
        ));

        // A pts with no unit is refused rather than guessed at, and leaves
        // nothing behind: the frame spacing is still unmeasured.
        let mut unitless = pool.get();
        unitless.set_pts(Some(10));
        assert!(matches!(
            sync.consume(MediaBuffer::Video(Arc::new(unitless))),
            Err(crate::error::Error::VideoSynchronizerError(
                VideoSynchronizerError::NoTimeBase
            ))
        ));
        assert!(sync.last_ns.is_none());
        assert!(sync.pending.is_empty());
    }

    /// Preroll has to outrun the clock this element schedules against —
    /// otherwise a paused pipeline could never deliver a preview frame, and
    /// an audio-mastered clock would hold it indefinitely. Suppressing
    /// pre-target media is not its job: that needs a time base, which the
    /// decoder has on every decoded branch and this element only has on the
    /// branches it happens to be on.
    #[test]
    fn preroll_bypasses_clock_scheduling_and_resume_restores_it() {
        let (mut sync, playback) = synchronizer();
        let _audio = playback.register_audio_master().unwrap();
        let context = Arc::new(PrerollContext::for_seek([], Duration::from_secs(2)));

        // Audio has primed nothing, so ordinary scheduling would hold here.
        assert!(matches!(sync.decision(ms(2_000)), Decision::Hold));

        sync.control(&ControlMsg::Preroll(context))
            .expect("preroll");
        assert!(matches!(sync.wait_for(ms(2_000)), WaitOutcome::Render));

        sync.control(&ControlMsg::Resume).expect("resume");
        assert!(matches!(sync.decision(ms(2_000)), Decision::Hold));
    }
}
