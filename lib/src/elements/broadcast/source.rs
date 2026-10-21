//! Reading a broadcast into a pipeline of its own: [`BroadcastSource`].

use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::Receiver;
use ffmpeg_next::{Rational, Rescale};

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
use super::{Broadcast, BroadcastError, BroadcastItem, BroadcastReader};

/// The unit the timeline a [`BroadcastSource`] makes is kept in, between
/// tracks.
const MICROSECONDS: Rational = Rational(1, 1_000_000);

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
pub struct BroadcastSource(SourceStage<Relaying>);

source_stage!(BroadcastSource);

/// What a [`BroadcastSource`] does: hands on what its reader reads, on a
/// timeline of its own.
pub(crate) struct Relaying {
    reader: BroadcastReader,
    bell: Receiver<()>,
    broadcast: Arc<str>,
    tracks: Vec<StreamInfo>,
    /// Where the run being read began on the broadcast's timeline: the
    /// first packet handed on of it, and that packet's unit.
    origin: Option<(i64, Rational)>,
    /// Where the run being read begins on this source's own timeline, in
    /// [`MICROSECONDS`].
    offset: i64,
    /// Where what has been handed on so far ends, in [`MICROSECONDS`].
    end: i64,
    /// When the last packet handed on had arrived, for how far after it the
    /// next run is put.
    last_arrival: Option<Instant>,
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
                    origin: None,
                    offset: 0,
                    end: 0,
                    last_arrival: None,
                    pp_log,
                },
            )),
            streams,
        ))
    }
}

impl Relaying {
    /// Whether a run of `tracks` can go on through the outputs made for
    /// this source's: as many, each of the same medium and codec.
    fn fits(&self, tracks: &[StreamInfo]) -> bool {
        tracks.len() == self.tracks.len()
            && tracks.iter().zip(self.tracks.iter()).all(|(new, old)| {
                new.kind == old.kind && new.parameters.id() == old.parameters.id()
            })
    }

    /// Another run begins, which what it hands on puts after what has
    /// been: whether it can go on with its tracks.
    fn begin_run(&mut self, tracks: &[StreamInfo]) -> bool {
        if !self.fits(tracks) {
            pp_warn!(
                pp_log: &self.pp_log,
                "event=eos phase=source broadcast={} was published again with tracks [{}], which the outputs made for [{}] cannot carry; ending",
                self.broadcast,
                super::describe(tracks),
                super::describe(&self.tracks)
            );
            return false;
        }
        self.origin = None;
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
        if unit.numerator() <= 0 || unit.denominator() <= 0 {
            return None;
        }
        let at = packet.dts().or(packet.pts())?;
        let (origin, origin_unit) = match self.origin {
            Some(origin) => origin,
            None => {
                // The first packet of a run: it goes as far after the last
                // run's last packet as it arrived after it.
                if let Some(last) = self.last_arrival {
                    let gap = arrived.saturating_duration_since(last).as_micros();
                    self.offset = self
                        .end
                        .saturating_add(i64::try_from(gap).unwrap_or(i64::MAX));
                    pp_info!(
                        pp_log: &self.pp_log,
                        "broadcast={} goes on at {:.3}s",
                        self.broadcast,
                        self.offset as f64 / 1e6
                    );
                }
                *self.origin.insert((at, unit))
            }
        };
        let shift = self.offset.rescale(MICROSECONDS, unit) - origin.rescale(origin_unit, unit);
        let dts = packet.dts().map(|dts| dts + shift);
        if dts
            .or(packet.pts().map(|pts| pts + shift))
            .is_some_and(|ts| ts < 0)
        {
            return None;
        }
        let mut stamped = packet.as_ref().clone();
        stamped.set_pts(packet.pts().map(|pts| pts + shift));
        stamped.set_dts(dts);
        if let Some(last) = stamped.pts().or(dts) {
            let ends = (last + stamped.duration().max(0)).rescale(unit, MICROSECONDS);
            self.end = self.end.max(ends);
        }
        self.last_arrival = Some(arrived);
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

    /// The next packet its reader reads, waiting for one to be written.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        loop {
            let Some(item) = self.reader.try_next() else {
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
