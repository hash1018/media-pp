//! Writing one track of a branch into a broadcast: [`BroadcastSink`].

use std::sync::Arc;

use crate::pp_log::{PpLog, pp_debug, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    element::{Attached, ElementType, Identity, Sink},
    elements::StreamInfo,
    error::Result,
    render::{SinkStage, sink_stage},
};

use super::state::Refused;
use super::{BroadcastError, Shared};

/// The end of the branch carrying one track into a [`Broadcast`](super::Broadcast).
///
/// Handed out by [`Broadcast::publish`](super::Broadcast::publish), one
/// for each track, and not made any other way: a sink belongs to the one
/// publication that made it. It takes that track's encoded packets and
/// never waits — what it is handed is kept and read by others, however
/// slowly they read.
///
/// The end of its stream ends its track, and the run once every track has
/// ended; so does a stop, or dropping it. A seek's flush begins a run of the
/// same tracks. Once the broadcast has been published again, it is refused
/// with [`BroadcastError::Superseded`], and once it is closed with
/// [`BroadcastError::Closed`] — each posted to the bus by the pipeline as a
/// buffer it could not take.
pub struct BroadcastSink(SinkStage<Writing>);

sink_stage!(BroadcastSink);

/// What a [`BroadcastSink`] does with each packet: keeps it in the
/// broadcast.
pub(crate) struct Writing {
    shared: Arc<Shared>,
    publication: u64,
    track: usize,
    kind: MediaKind,
    /// Whether its track has ended — the end of its stream, a stop.
    ended: bool,
    /// Whether it has said that what it was handed went over the bound on
    /// what a broadcast holds, which it says once.
    over_said: bool,
    pp_log: PpLog,
}

impl BroadcastSink {
    pub(super) fn new(shared: Arc<Shared>, publication: u64, track: &StreamInfo) -> Self {
        let identity = Identity::new(
            ElementType::BroadcastSink,
            format!("{}-{}", shared.name, track.index()),
        );
        let pp_log = identity.pp_log().clone();
        pp_info!(
            pp_log: &pp_log,
            "created: track={} kind={:?} codec={:?}",
            track.index(),
            track.kind(),
            track.parameters().id()
        );
        let kind = MediaKind::packet_for(track.kind()).unwrap_or(MediaKind::VideoPacket);
        Self(SinkStage::new(
            identity,
            Writing {
                shared,
                publication,
                track: track.index(),
                kind,
                ended: false,
                over_said: false,
                pp_log,
            },
        ))
    }
}

impl Writing {
    /// Ends its track, once.
    fn end(&mut self, why: &str) {
        if self.ended {
            return;
        }
        self.ended = true;
        let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        let run = state.run;
        if state.end_track(self.publication, self.track) {
            pp_info!(pp_log: &self.pp_log, "event=eos phase=run_ended run={run} by={why}");
        } else {
            pp_debug!(pp_log: &self.pp_log, "event=eos phase=track_ended by={why}");
        }
        self.shared.wake(&mut state);
    }
}

impl Sink for Writing {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::packet(self.kind))
    }

    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Packet(packet) = buf else {
            return Err(BroadcastError::UnsupportedBuffer(buf.kind()).into());
        };
        let retain = self.shared.options.retain;
        let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        let written = state.write(self.publication, self.track, packet, retain);
        let over = match written {
            Ok(over) => over,
            Err(Refused::Closed) => {
                return Err(BroadcastError::Closed(Arc::clone(&self.shared.name)).into());
            }
            Err(Refused::Superseded) => {
                return Err(BroadcastError::Superseded(Arc::clone(&self.shared.name)).into());
            }
        };
        self.shared.wake(&mut state);
        drop(state);
        if over && !self.over_said {
            self.over_said = true;
            pp_warn!(
                pp_log: &self.pp_log,
                "the broadcast went over {} MiB within its retain; letting go of the oldest packets",
                super::state::MAX_BYTES / (1024 * 1024)
            );
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<()> {
        self.end("eos");
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        let mut state = self.shared.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.flush(self.publication, self.track) {
            pp_info!(pp_log: &self.pp_log, "event=flush phase=run_began run={}", state.run);
            self.shared.wake(&mut state);
        } else {
            pp_debug!(pp_log: &self.pp_log, "event=flush phase=joined_run run={}", state.run);
        }
        self.ended = false;
        Ok(())
    }

    fn stopping(&mut self) -> Result<()> {
        self.end("stop");
        Ok(())
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

impl Drop for Writing {
    fn drop(&mut self) {
        self.end("drop");
    }
}
