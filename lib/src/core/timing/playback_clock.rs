//! Which stream currently defines the pipeline's media position.
//!
//! [`Clock`](crate::clock::Clock) stays the monotonic control and pause clock.
//! This module adds the *media* position on top of it, and the handover it
//! exists for: a pipeline starts on a wall-clock fallback and can pass the
//! position to one audio renderer once that renderer's endpoint is running,
//! without ever letting the position jump backwards. [`PlaybackMaster`] is
//! the state that handover is currently in.
//!
//! Only one audio master may hold the clock at a time, which is what
//! [`PlaybackClockError`] guards. No audio-backend type appears here: a
//! renderer publishes device-position snapshots through a private
//! registration, and video scheduling only ever reads the result.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use thiserror::Error as ThisError;

use crate::clock::Clock;

/// Which source currently defines the pipeline's media position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackMaster {
    /// No timestamped stream has established a position yet.
    Unavailable,
    /// Media position advances from the pipeline's pause-aware wall clock.
    Wall,
    /// An audio renderer owns the clock but has not started the endpoint yet.
    AudioPriming,
    /// An audio endpoint's played-sample position is the master clock.
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ThisError)]
/// Why an audio renderer could not take or keep the playback clock.
///
/// Both variants mean the caller's registration is not the live one — either
/// another master already holds the clock, or this registration was superseded.
/// Neither is fatal to playback: the position simply stays with whoever owns it.
pub enum PlaybackClockError {
    /// Another live audio renderer already owns the playback clock.
    #[error("this pipeline already has an audio playback-clock master")]
    AudioMasterAlreadyRegistered,

    /// The registration was released or superseded before this operation.
    #[error("the audio playback-clock registration is stale")]
    StaleAudioMaster,
}

/// Pipeline-wide media clock shared by audio output and video scheduling.
///
/// [`Clock`] remains the pipeline's monotonic control/pause clock. This
/// type adds the media-timeline position and can hand that position from a
/// wall-clock fallback to one audio renderer without letting the position
/// jump backwards. It deliberately contains no WASAPI types: an audio
/// backend publishes device-position snapshots through its private
/// registration, while video scheduling only reads the resulting position.
pub struct PlaybackClock {
    wall_clock: Arc<Clock>,
    state: Mutex<State>,
}

#[derive(Clone, Copy)]
enum State {
    Unavailable {
        next_registration: u64,
    },
    Wall {
        anchor_ns: i64,
        anchor_elapsed: Duration,
        next_registration: u64,
    },
    AudioPriming {
        registration: u64,
        held_ns: Option<i64>,
        next_registration: u64,
    },
    // Only an audio renderer moves the clock into these two, and the only one
    // in this crate is behind `wasapi-renderer`. They are dead in a build
    // without it, but they are the timeline contract `PlaybackClock` exists to
    // provide — gating them on a backend feature would invert that. See
    // `AudioMasterRegistration`.
    #[allow(dead_code)]
    Audio {
        registration: u64,
        position_ns: i64,
        sampled_elapsed: Duration,
        submitted_until_ns: i64,
        running: bool,
        next_registration: u64,
    },
    #[allow(dead_code)]
    AudioFallback {
        registration: u64,
        anchor_ns: i64,
        anchor_elapsed: Duration,
        next_registration: u64,
    },
}

impl PlaybackClock {
    pub(crate) fn new(wall_clock: Arc<Clock>) -> Self {
        Self {
            wall_clock,
            state: Mutex::new(State::Unavailable {
                next_registration: 1,
            }),
        }
    }

    /// Reports which timing source currently defines media position.
    ///
    /// This is a lock-protected snapshot; ownership may change immediately
    /// afterward as an audio renderer starts or stops.
    pub fn master(&self) -> PlaybackMaster {
        match *self.state.lock().unwrap() {
            State::Unavailable { .. } => PlaybackMaster::Unavailable,
            State::Wall { .. } | State::AudioFallback { .. } => PlaybackMaster::Wall,
            State::AudioPriming { .. } => PlaybackMaster::AudioPriming,
            State::Audio { .. } => PlaybackMaster::Audio,
        }
    }

    #[cfg(test)]
    pub(crate) fn position_ns(&self) -> Option<i64> {
        let state = self.state.lock().unwrap();
        position_at(*state, self.wall_clock.elapsed())
    }

    pub(crate) fn interrupt_epoch(&self) -> u64 {
        self.wall_clock.interrupt_epoch()
    }

    /// Establishes a wall-clock media origin if no stream owns one yet.
    /// Returns the current position after doing so.
    #[cfg(test)]
    pub(crate) fn ensure_wall_origin(&self, media_ns: i64) -> Option<i64> {
        let mut state = self.state.lock().unwrap();
        if let State::Unavailable { next_registration } = *state {
            self.wall_clock.start();
            let elapsed = self.wall_clock.elapsed();
            *state = State::Wall {
                anchor_ns: media_ns,
                anchor_elapsed: elapsed,
                next_registration,
            };
        }
        position_at(*state, self.wall_clock.elapsed())
    }

    /// How long from now until `media_ns` is its turn.
    ///
    /// The other direction of what [`PlaybackClock::video_snapshot`] reads:
    /// that answers *what media time is it*, this answers *when is this media
    /// time*. Both are the same relation between the media timeline and this
    /// pipeline's own, and keeping them in one place is what stops a paced
    /// element and a synchronized one from scheduling the same picture
    /// against two authorities that drift apart.
    ///
    /// A `Duration` rather than an `Instant`, because under an audio master
    /// there is no wall-clock moment to name: the position advances at the
    /// device's own rate, so the honest answer is how much is left *as of
    /// now*. A caller sleeps some of it and asks again, which is what
    /// [`crate::elements::Pacer`] already did with its own arithmetic.
    ///
    /// Establishes the origin from the first caller to ask, exactly as
    /// `video_snapshot` does — whichever of the two arrives first speaks for
    /// the pipeline, which is the point: a container's streams do not start
    /// at the same timestamp, and that offset is part of their sync.
    ///
    /// `Duration::ZERO` while an audio master is priming and has said
    /// nothing about where it is. Holding a caller there would be waiting on
    /// audio that has not started, which is the stall a renderer's deferred
    /// registration exists to avoid.
    pub(crate) fn remaining(&self, media_ns: i64) -> Duration {
        let mut state = self.state.lock().unwrap();
        if let State::Unavailable { next_registration } = *state {
            self.wall_clock.start();
            let elapsed = self.wall_clock.elapsed();
            *state = State::Wall {
                anchor_ns: media_ns,
                anchor_elapsed: elapsed,
                next_registration,
            };
        }
        let Some(position) = position_at(*state, self.wall_clock.elapsed()) else {
            return Duration::ZERO;
        };
        let ahead = media_ns.saturating_sub(position);
        if ahead <= 0 {
            return Duration::ZERO;
        }
        Duration::from_nanos(ahead as u64)
    }

    /// Moves the origin so that `media_ns` is due now.
    ///
    /// For a sender whose timeline restarts under the pipeline — a camera
    /// rebooting, an RTP timestamp base that wraps — where the timestamps
    /// that follow have no relation to the ones before them. Paced literally
    /// such a jump is a still picture for as long as it claims to be worth.
    ///
    /// A no-op while an audio master owns the position: what it is playing is
    /// what the pipeline is at, and a stream that jumped is the stream's
    /// problem to reconcile, not the clock's.
    pub(crate) fn re_anchor(&self, media_ns: i64) {
        let mut state = self.state.lock().unwrap();
        let next_registration = match *state {
            State::Unavailable {
                next_registration, ..
            }
            | State::Wall {
                next_registration, ..
            } => next_registration,
            State::AudioPriming { .. } | State::Audio { .. } | State::AudioFallback { .. } => {
                return;
            }
        };
        self.wall_clock.start();
        *state = State::Wall {
            anchor_ns: media_ns,
            anchor_elapsed: self.wall_clock.elapsed(),
            next_registration,
        };
    }

    pub(crate) fn video_snapshot(&self, media_ns: i64) -> (PlaybackMaster, Option<i64>) {
        let mut state = self.state.lock().unwrap();
        if let State::Unavailable { next_registration } = *state {
            self.wall_clock.start();
            let elapsed = self.wall_clock.elapsed();
            *state = State::Wall {
                anchor_ns: media_ns,
                anchor_elapsed: elapsed,
                next_registration,
            };
        }
        let elapsed = self.wall_clock.elapsed();
        let master = match *state {
            State::Unavailable { .. } => PlaybackMaster::Unavailable,
            State::Wall { .. } | State::AudioFallback { .. } => PlaybackMaster::Wall,
            State::AudioPriming { .. } => PlaybackMaster::AudioPriming,
            State::Audio { .. } => PlaybackMaster::Audio,
        };
        (master, position_at(*state, elapsed))
    }

    /// Claims the timeline for one audio renderer. Unused in a build without
    /// an audio renderer (see `AudioMasterRegistration`), hence the `allow`.
    #[allow(dead_code)]
    pub(crate) fn register_audio_master(
        self: &Arc<Self>,
    ) -> Result<AudioMasterRegistration, PlaybackClockError> {
        let mut state = self.state.lock().unwrap();
        let elapsed = self.wall_clock.elapsed();
        let (held_ns, registration, next_registration) = match *state {
            State::Unavailable { next_registration } => {
                (None, next_registration, next_registration.wrapping_add(1))
            }
            State::Wall {
                next_registration, ..
            } => (
                position_at(*state, elapsed),
                next_registration,
                next_registration.wrapping_add(1),
            ),
            State::AudioPriming { .. } | State::Audio { .. } | State::AudioFallback { .. } => {
                return Err(PlaybackClockError::AudioMasterAlreadyRegistered);
            }
        };
        *state = State::AudioPriming {
            registration,
            held_ns,
            next_registration,
        };
        Ok(AudioMasterRegistration {
            clock: self.clone(),
            registration,
        })
    }

    /// Resets media state for a seek while retaining the current audio
    /// renderer's ownership. The next timestamp/device sample establishes
    /// the post-seek position.
    pub(crate) fn reset_for_seek(&self) {
        let mut state = self.state.lock().unwrap();
        *state = match *state {
            State::Unavailable { next_registration }
            | State::Wall {
                next_registration, ..
            } => State::Unavailable { next_registration },
            State::AudioPriming {
                registration,
                next_registration,
                ..
            }
            | State::Audio {
                registration,
                next_registration,
                ..
            }
            | State::AudioFallback {
                registration,
                next_registration,
                ..
            } => State::AudioPriming {
                registration,
                held_ns: None,
                next_registration,
            },
        };
    }

    #[allow(dead_code)]
    fn release_audio_master(&self, registration: u64) {
        let mut state = self.state.lock().unwrap();
        let elapsed = self.wall_clock.elapsed();
        let (matches, next_registration) = match *state {
            State::AudioPriming {
                registration: current,
                next_registration,
                ..
            }
            | State::Audio {
                registration: current,
                next_registration,
                ..
            }
            | State::AudioFallback {
                registration: current,
                next_registration,
                ..
            } => (current == registration, next_registration),
            State::Unavailable { .. } | State::Wall { .. } => return,
        };
        if !matches {
            return;
        }
        *state = match position_at(*state, elapsed) {
            Some(anchor_ns) => State::Wall {
                anchor_ns,
                anchor_elapsed: elapsed,
                next_registration,
            },
            None => State::Unavailable { next_registration },
        };
    }
}

/// Exclusive, generation-checked writer owned by one audio renderer.
/// Dropping it hands the last known position back to the wall clock.
///
/// The only audio renderer in this crate is `WasapiRenderer`, behind the
/// `wasapi-renderer` feature, so a build without it constructs this nowhere and
/// every method below is dead. That is why the `allow`s here are deliberate
/// rather than a `cfg(feature = "wasapi-renderer")` gate: `PlaybackClock` is
/// the backend-independent timeline every renderer binds to, and teaching it
/// about one backend's Cargo feature would invert that relationship. The
/// crate's own tests exercise this path, so it is covered even when no shipped
/// element uses it.
#[allow(dead_code)]
pub(crate) struct AudioMasterRegistration {
    clock: Arc<PlaybackClock>,
    registration: u64,
}

#[allow(dead_code)]
impl AudioMasterRegistration {
    pub(crate) fn priming_target_ns(&self) -> Result<Option<i64>, PlaybackClockError> {
        match *self.clock.state.lock().unwrap() {
            State::AudioPriming {
                registration,
                held_ns,
                ..
            } if registration == self.registration => Ok(held_ns),
            State::Audio { registration, .. } if registration == self.registration => Ok(None),
            State::AudioFallback { registration, .. } if registration == self.registration => {
                Ok(None)
            }
            _ => Err(PlaybackClockError::StaleAudioMaster),
        }
    }

    pub(crate) fn publish(
        &self,
        position_ns: i64,
        submitted_until_ns: i64,
        running: bool,
    ) -> Result<(), PlaybackClockError> {
        let mut state = self.clock.state.lock().unwrap();
        self.clock.wall_clock.start();
        let elapsed = self.clock.wall_clock.elapsed();
        let (held_ns, next_registration) = match *state {
            State::AudioPriming {
                registration,
                held_ns,
                next_registration,
            } if registration == self.registration => (held_ns, next_registration),
            State::Audio {
                registration,
                next_registration,
                ..
            } if registration == self.registration => (None, next_registration),
            State::AudioFallback {
                registration,
                next_registration,
                ..
            } if registration == self.registration => (None, next_registration),
            _ => return Err(PlaybackClockError::StaleAudioMaster),
        };

        // A master handoff must never make video scheduling move backwards.
        let position_ns = held_ns.map_or(position_ns, |held| position_ns.max(held));
        let submitted_until_ns = submitted_until_ns.max(position_ns);
        *state = State::Audio {
            registration: self.registration,
            position_ns,
            sampled_elapsed: elapsed,
            submitted_until_ns,
            running,
            next_registration,
        };
        Ok(())
    }

    /// Audio ended before another stream: continue from its final played
    /// position using the wall clock while retaining this registration so
    /// a second renderer cannot race the still-attached one.
    pub(crate) fn finish(&self, position_ns: i64) -> Result<(), PlaybackClockError> {
        let mut state = self.clock.state.lock().unwrap();
        let elapsed = self.clock.wall_clock.elapsed();
        let next_registration = match *state {
            State::AudioPriming {
                registration,
                next_registration,
                ..
            }
            | State::Audio {
                registration,
                next_registration,
                ..
            } if registration == self.registration => next_registration,
            _ => return Err(PlaybackClockError::StaleAudioMaster),
        };
        *state = State::AudioFallback {
            registration: self.registration,
            anchor_ns: position_ns,
            anchor_elapsed: elapsed,
            next_registration,
        };
        Ok(())
    }

    pub(crate) fn reset_for_seek(&self) -> Result<(), PlaybackClockError> {
        let mut state = self.clock.state.lock().unwrap();
        let next_registration = match *state {
            State::AudioPriming {
                registration,
                next_registration,
                ..
            }
            | State::Audio {
                registration,
                next_registration,
                ..
            }
            | State::AudioFallback {
                registration,
                next_registration,
                ..
            } if registration == self.registration => next_registration,
            _ => return Err(PlaybackClockError::StaleAudioMaster),
        };
        *state = State::AudioPriming {
            registration: self.registration,
            held_ns: None,
            next_registration,
        };
        Ok(())
    }
}

impl Drop for AudioMasterRegistration {
    fn drop(&mut self) {
        self.clock.release_audio_master(self.registration);
    }
}

fn position_at(state: State, elapsed: Duration) -> Option<i64> {
    match state {
        State::Unavailable { .. } => None,
        State::Wall {
            anchor_ns,
            anchor_elapsed,
            ..
        } => Some(add_duration(
            anchor_ns,
            elapsed.saturating_sub(anchor_elapsed),
        )),
        State::AudioPriming { held_ns, .. } => held_ns,
        State::Audio {
            position_ns,
            sampled_elapsed,
            submitted_until_ns,
            running,
            ..
        } => {
            let projected = if running {
                add_duration(position_ns, elapsed.saturating_sub(sampled_elapsed))
            } else {
                position_ns
            };
            Some(projected.min(submitted_until_ns))
        }
        State::AudioFallback {
            anchor_ns,
            anchor_elapsed,
            ..
        } => Some(add_duration(
            anchor_ns,
            elapsed.saturating_sub(anchor_elapsed),
        )),
    }
}

fn add_duration(value_ns: i64, duration: Duration) -> i64 {
    let delta = duration.as_nanos().min(i64::MAX as u128) as i64;
    value_ns.saturating_add(delta)
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;

    #[test]
    fn wall_origin_advances_and_freezes_with_pipeline_clock() {
        let wall = Arc::new(Clock::new());
        let playback = PlaybackClock::new(wall.clone());
        assert!(playback.ensure_wall_origin(1_000).unwrap() >= 1_000);
        thread::sleep(Duration::from_millis(20));
        assert!(playback.position_ns().unwrap() >= 10_000_000);

        wall.pause();
        let paused = playback.position_ns().unwrap();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(playback.position_ns(), Some(paused));
    }

    #[test]
    fn audio_handoff_never_moves_backwards_and_release_continues_on_wall() {
        let wall = Arc::new(Clock::new());
        let playback = Arc::new(PlaybackClock::new(wall));
        playback.ensure_wall_origin(50_000_000);
        let audio = playback.register_audio_master().unwrap();
        let held = audio.priming_target_ns().unwrap().unwrap();

        audio
            .publish(held - 10_000_000, held + 100_000_000, true)
            .unwrap();
        assert!(playback.position_ns().unwrap() >= held);
        drop(audio);
        let released = playback.position_ns().unwrap();
        thread::sleep(Duration::from_millis(10));
        assert!(playback.position_ns().unwrap() >= released);
        assert_eq!(playback.master(), PlaybackMaster::Wall);
    }

    #[test]
    fn only_one_audio_master_can_publish_and_seek_retains_its_generation() {
        let wall = Arc::new(Clock::new());
        let playback = Arc::new(PlaybackClock::new(wall));
        let audio = playback.register_audio_master().unwrap();
        assert!(matches!(
            playback.register_audio_master(),
            Err(PlaybackClockError::AudioMasterAlreadyRegistered)
        ));

        playback.reset_for_seek();
        audio.publish(2_000, 3_000, true).unwrap();
        assert_eq!(playback.master(), PlaybackMaster::Audio);
    }

    #[test]
    fn audio_projection_is_capped_at_submitted_media() {
        let wall = Arc::new(Clock::new());
        let playback = Arc::new(PlaybackClock::new(wall));
        let audio = playback.register_audio_master().unwrap();
        audio.publish(10, 1_000_000, true).unwrap();
        thread::sleep(Duration::from_millis(5));
        assert_eq!(playback.position_ns(), Some(1_000_000));
    }

    #[test]
    fn finished_audio_continues_on_wall_and_can_reset_for_seek() {
        let wall = Arc::new(Clock::new());
        let playback = Arc::new(PlaybackClock::new(wall));
        let audio = playback.register_audio_master().unwrap();
        audio.publish(1_000, 2_000, false).unwrap();
        audio.finish(2_000).unwrap();
        assert_eq!(playback.master(), PlaybackMaster::Wall);
        thread::sleep(Duration::from_millis(5));
        assert!(playback.position_ns().unwrap() > 2_000);

        audio.reset_for_seek().unwrap();
        assert_eq!(playback.master(), PlaybackMaster::AudioPriming);
        assert_eq!(playback.position_ns(), None);
    }
}
