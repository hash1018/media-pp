//! Reading a broadcast into a pipeline of its own: [`BroadcastSource`].

use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use crossbeam_channel::Receiver;

use crate::pp_log::{PpLog, pp_debug, pp_info, pp_warn};
use crate::{
    buffer::{MediaBuffer, PacketBuffer},
    contract::{MediaKind, OutputContract, PortContract},
    element::{Attached, ElementType, Identity, Produced, Source, SourceStage, Wait},
    elements::StreamInfo,
    error::Result,
    pad::SrcPad,
    produce::{Received, source_stage},
};

use super::state::time_base;
use super::timeline::{Timeline, same_shape};
use super::{Broadcast, BroadcastError, BroadcastItem, BroadcastReader};

/// Starts a pipeline of its own from a [`Broadcast`]: a recording of a
/// camera that viewers watch at the same time, say.
///
/// One output for each of the broadcast's tracks, in their order, each
/// handing on that track's packets as they were written — shared with
/// every other reader, nothing decoded. It starts at the newest keyframe
/// the broadcast holds, every track from there, so the first picture
/// handed on can be shown.
///
/// What it hands on is on one timeline of its own, beginning at zero: the
/// broadcast's runs one after another, each after the last as far as the
/// time between them, so a camera that dropped for three seconds leaves a
/// three-second gap in the recording rather than timestamps starting over.
/// What a track has from before the keyframe it starts at is not handed on.
///
/// It is live, and never ends while the broadcast goes on: it ends its
/// stream once the broadcast is closed, or once the broadcast is published
/// again with tracks it could not go on with — a different number of them,
/// or a different codec. A reader that falls behind as far as
/// [`BroadcastOptions::retain`](super::BroadcastOptions::retain) is moved on
/// to the newest keyframe, and says so in the log; nothing in between is
/// handed on.
///
/// It can be driven — run in a
/// [`DrivenPipeline`](crate::pipeline::DrivenPipeline), a server's player
/// each: it waits on its reader, which a write wakes, and asks for no time
/// of its own.
pub struct BroadcastSource(SourceStage<Relaying>);

source_stage!(BroadcastSource);

/// What a [`BroadcastSource`] does: hands on what its reader reads, on a
/// timeline of its own.
pub(crate) struct Relaying {
    reader: BroadcastReader,
    bell: Receiver<()>,
    broadcast: Arc<str>,
    tracks: Vec<StreamInfo>,
    timeline: Timeline,
    pp_log: PpLog,
}

impl BroadcastSource {
    /// A source reading `broadcast` from its newest keyframe on, and the
    /// tracks its outputs carry, in their order.
    ///
    /// # Errors
    ///
    /// [`BroadcastError::NotPublished`] where nothing has been published to
    /// the broadcast yet, and [`BroadcastError::Closed`] where it has been
    /// closed: there are no tracks to make outputs of.
    pub fn new(
        name: impl Into<String>,
        broadcast: &Broadcast,
    ) -> std::result::Result<(Self, Vec<StreamInfo>), BroadcastError> {
        let mut reader = broadcast.subscribe();
        let tracks = loop {
            match reader.try_next() {
                Some(BroadcastItem::Started(tracks)) => break tracks,
                Some(BroadcastItem::Closed) => {
                    return Err(BroadcastError::Closed(broadcast.name()));
                }
                Some(_) => {}
                None if broadcast.is_closed() => {
                    return Err(BroadcastError::Closed(broadcast.name()));
                }
                None => return Err(BroadcastError::NotPublished(broadcast.name())),
            }
        };
        let bell = reader.bell();
        let identity = Identity::new(ElementType::BroadcastSource, name);
        let pp_log = identity.pp_log().clone();
        pp_info!(
            pp_log: &pp_log,
            "created: broadcast={} tracks=[{}]",
            broadcast.name(),
            super::describe(&tracks)
        );
        let streams = tracks.to_vec();
        Ok((
            Self(SourceStage::new(
                identity,
                Relaying {
                    reader,
                    bell,
                    broadcast: broadcast.name(),
                    tracks,
                    timeline: Timeline::default(),
                    pp_log,
                },
            )),
            streams,
        ))
    }
}

impl Relaying {
    /// Another run begins, which what it hands on puts after what has
    /// been: whether it can go on with its tracks.
    fn begin_run(&mut self, tracks: &[StreamInfo]) -> bool {
        if !same_shape(&self.tracks, tracks) {
            pp_warn!(
                pp_log: &self.pp_log,
                "event=eos phase=source broadcast={} was published again with tracks [{}], which the outputs made for [{}] cannot carry; ending",
                self.broadcast,
                super::describe(tracks),
                super::describe(&self.tracks)
            );
            return false;
        }
        self.timeline.next_run();
        pp_info!(pp_log: &self.pp_log, "broadcast={} began another run", self.broadcast);
        true
    }

    /// `packet` of `track`, put on this source's timeline: `None` where it
    /// would come before where the run began on it.
    fn restamp(
        &mut self,
        track: usize,
        packet: &PacketBuffer,
        arrived: Instant,
    ) -> Option<MediaBuffer> {
        let unit = time_base(packet, Some(self.tracks[track].time_base));
        let first = self.timeline.run_begins_at().is_none();
        let stamped = self.timeline.restamp(packet, unit, arrived)?;
        if first && let Some(at) = self.timeline.run_begins_at() {
            pp_info!(pp_log: &self.pp_log, "broadcast={} goes on at {at:.3}s", self.broadcast);
        }
        let mut stamped: PacketBuffer = Arc::new(stamped).into();
        if let Some(metadata) = packet.metadata_arc() {
            stamped = stamped.with_metadata(Arc::clone(metadata));
        }
        Some(MediaBuffer::Packet(stamped))
    }
}

impl Source for Relaying {
    fn is_live(&self) -> bool {
        true
    }

    fn outputs(&self) -> Option<Vec<SrcPad>> {
        Some(
            self.tracks
                .iter()
                .enumerate()
                .map(|(index, track)| {
                    let contract = MediaKind::packet_for(track.kind)
                        .map_or(OutputContract::Unknown, |kind| {
                            OutputContract::Fixed(PortContract::packet(kind))
                        });
                    SrcPad::with_contract(format!("src_{index}"), contract)
                })
                .collect(),
        )
    }

    /// Driven, it waits on its reader, which wakes the poll it hands the
    /// waker to.
    fn drivable(&self) -> bool {
        true
    }

    /// The next packet its reader reads, waiting for one to be written —
    /// on its bell beside the pipeline on a thread of its own, and driven,
    /// by handing the reader the poll's waker.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        loop {
            let item = if let Some(waker) = wait.waker() {
                match self.reader.poll_next(&mut Context::from_waker(waker)) {
                    Poll::Ready(Some(item)) => item,
                    // Only after `Closed`, which has ended it already.
                    Poll::Ready(None) => return Ok(Produced::End),
                    Poll::Pending => return Ok(Produced::Nothing),
                }
            } else if let Some(item) = self.reader.try_next() {
                item
            } else {
                match wait.recv(&self.bell) {
                    Received::Got(()) => continue,
                    Received::LetGo | Received::Gone => return Ok(Produced::Nothing),
                }
            };
            match item {
                BroadcastItem::Packet {
                    track,
                    packet,
                    arrived,
                } => {
                    if let Some(buf) = self.restamp(track, &packet, arrived) {
                        return Ok(Produced::On(track, buf));
                    }
                }
                BroadcastItem::Started(tracks) => {
                    if !self.begin_run(&tracks) {
                        return Ok(Produced::End);
                    }
                }
                BroadcastItem::Skipped => {
                    pp_warn!(
                        pp_log: &self.pp_log,
                        "fell behind broadcast={} by more than it keeps; going on from its newest keyframe",
                        self.broadcast
                    );
                }
                BroadcastItem::Ended => {
                    pp_debug!(pp_log: &self.pp_log, "broadcast={} run ended; waiting for another", self.broadcast);
                }
                BroadcastItem::Closed => {
                    pp_info!(pp_log: &self.pp_log, "event=eos phase=source broadcast={} closed", self.broadcast);
                    return Ok(Produced::End);
                }
            }
        }
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}
