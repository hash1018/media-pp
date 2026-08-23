use std::{collections::VecDeque, sync::Arc, thread, time::Duration};

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
    time::{InvalidTimeBase, MediaTimestamp, TimeBase},
};

const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const FALLBACK_FRAME_DURATION: Duration = Duration::from_millis(40);

fn nanoseconds() -> TimeBase {
    TimeBase::new_unchecked(ffmpeg::Rational::new(1, 1_000_000_000))
}

#[derive(Debug, ThisError)]
/// Input this element cannot schedule.
///
/// Scheduling needs a decoded video frame with a PTS and a usable time base;
/// anything else is rejected rather than passed through unscheduled.
pub enum VideoSynchronizerError {
    /// The supplied input timestamp unit has a non-positive component.
    #[error(
        "invalid time base {numerator}/{denominator}: both numerator and denominator must be positive"
    )]
    InvalidTimeBase {
        /// Invalid rational numerator.
        numerator: i32,
        /// Invalid rational denominator.
        denominator: i32,
    },

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
/// frames and dropping frames more than one frame-duration late. During
/// audio priming it holds the in-flight frame so the wall-to-audio handoff
/// cannot make the picture run ahead.
///
/// Do not put a `Pacer` in the same video branch; that would pace twice.
/// Put a [`crate::queue::Queue`] upstream so waits do not block demux/decode.
pub struct VideoSynchronizer {
    pp_log: PpLog,
    name: Arc<str>,
    time_base: TimeBase,
    playback_clock: Arc<PlaybackClock>,
    interrupt_epoch: u64,
    last_pts: Option<i64>,
    frame_duration: Duration,
    pending: VecDeque<MediaBuffer>,
    pad: SrcPad,
}

impl VideoSynchronizer {
    /// Creates a video scheduler using `time_base` for incoming frame PTS values.
    pub fn new(
        name: impl Into<String>,
        time_base: ffmpeg::Rational,
        playback_clock: Arc<PlaybackClock>,
    ) -> Result<Self, VideoSynchronizerError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoSynchronizer, &name, None);
        let time_base = TimeBase::try_new(time_base).map_err(
            |InvalidTimeBase {
                 numerator,
                 denominator,
             }| VideoSynchronizerError::InvalidTimeBase {
                numerator,
                denominator,
            },
        )?;
        let interrupt_epoch = playback_clock.interrupt_epoch();
        pp_info!(pp_log: &pp_log, "created: time_base={time_base:?}");
        Ok(Self {
            name: name.clone(),
            pp_log,
            time_base,
            playback_clock,
            interrupt_epoch,
            last_pts: None,
            frame_duration: FALLBACK_FRAME_DURATION,
            pending: VecDeque::new(),
            pad: SrcPad::with_contract(format!("{name}_src"), OutputContract::Passthrough),
        })
    }

    fn timestamp_ns(&self, pts: i64) -> i64 {
        MediaTimestamp::new_unchecked(pts, self.time_base).rescale(nanoseconds())
    }

    fn observe_frame_duration(&mut self, pts: i64) {
        if let Some(delta) = self.last_pts.and_then(|last| pts.checked_sub(last))
            && delta > 0
        {
            let duration = Duration::from_nanos(self.timestamp_ns(delta).max(0) as u64);
            if !duration.is_zero() {
                self.frame_duration = duration;
            }
        }
        self.last_pts = Some(pts);
    }

    #[cfg(test)]
    fn decision(&mut self, pts: i64) -> Decision {
        self.observe_frame_duration(pts);
        self.decision_without_observing(pts)
    }

    fn wait_for(&mut self, pts: i64) -> WaitOutcome {
        self.observe_frame_duration(pts);
        loop {
            if self.playback_clock.interrupt_epoch() != self.interrupt_epoch {
                return WaitOutcome::Interrupted;
            }
            match self.decision_without_observing(pts) {
                Decision::Render => return WaitOutcome::Render,
                Decision::Drop => return WaitOutcome::Drop,
                Decision::Wait(wait) => thread::sleep(wait.min(INTERRUPT_POLL_INTERVAL)),
                Decision::Hold => thread::sleep(INTERRUPT_POLL_INTERVAL),
            }
        }
    }

    fn decision_without_observing(&self, pts: i64) -> Decision {
        let frame_ns = self.timestamp_ns(pts);
        let (master, position) = self.playback_clock.video_snapshot(frame_ns);
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
        InputContract::Fixed(PortContract::of(MediaKind::Video))
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
                    let pts = frame.pts().ok_or(VideoSynchronizerError::MissingPts)?;
                    self.wait_for(pts)
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

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        self.interrupt_epoch = self.playback_clock.interrupt_epoch();
        match msg {
            ControlMsg::Seek(_) | ControlMsg::Stop => {
                self.pending.clear();
                self.last_pts = None;
                self.frame_duration = FALLBACK_FRAME_DURATION;
            }
            ControlMsg::Pause | ControlMsg::Resume => {}
        }
        self.pad.control(msg)
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
    use crate::{clock::Clock, playback_clock::PlaybackClock, pool::UnboundObjectPool};

    fn synchronizer(clock: Arc<PlaybackClock>) -> VideoSynchronizer {
        VideoSynchronizer::new("sync", ffmpeg::Rational::new(1, 1_000), clock).unwrap()
    }

    #[test]
    fn first_video_timestamp_establishes_wall_origin() {
        let playback = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        let mut sync = synchronizer(playback.clone());
        assert!(matches!(sync.decision(5_000), Decision::Render));
        assert_eq!(playback.master(), PlaybackMaster::Wall);
        assert!(playback.position_ns().unwrap() >= 5_000_000_000);
    }

    #[test]
    fn audio_priming_holds_video_and_audio_master_drops_late_frames() {
        let playback = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        let mut sync = synchronizer(playback.clone());
        let audio = playback.register_audio_master().unwrap();
        assert!(matches!(sync.decision(1_000), Decision::Hold));

        audio.publish(2_000_000_000, 3_000_000_000, false).unwrap();
        assert!(matches!(sync.decision(1_000), Decision::Drop));
        assert!(matches!(sync.decision(2_010), Decision::Wait(_)));
    }

    #[test]
    fn invalid_time_base_and_non_video_input_are_typed_errors() {
        let playback = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        assert!(matches!(
            VideoSynchronizer::new("sync", ffmpeg::Rational::new(0, 1), playback.clone()),
            Err(VideoSynchronizerError::InvalidTimeBase { .. })
        ));
        let mut sync = synchronizer(playback);
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
    }
}
