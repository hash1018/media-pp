//! A publisher's stream handed out as a source: [`Recording`].

use std::time::{Duration, Instant};

use crossbeam_channel::TrySendError;
use ffmpeg_next as ffmpeg;

use crate::elements::source::{PublishedItem, PublishedOutbox};
use crate::elements::{RtspSource, StreamInfo};
use crate::pp_log::{PpLog, pp_warn};

use crate::elements::driver::publishers::Arrivals;
use crate::elements::rtsp::depacketize::Depacketizer;
use crate::elements::rtsp::sdp::Announced;

/// How long a recording holds what it is sent while it waits for every
/// picture's size, and how much — past either, it is handed out without.
const HOLD_FOR: Duration = Duration::from_secs(10);
const HOLD_AT_MOST: usize = 4096;

/// Why a recording cannot go on.
pub(crate) enum Stopped {
    /// Nothing takes the publisher: its path's `Publishers` has gone, or
    /// the source it was handed out as was dropped.
    Gone,
}

/// A publisher's tracks being handed on, as an [`RtspSource`] of them.
///
/// What the description does not say — a picture's size, and parameter
/// sets a publisher sends only in band — is read from each picture track's
/// first keyframe before the source is handed out, so that what reads it,
/// a player's description or a recording's file, has it; what came before
/// waits in the source.
pub(crate) struct Recording {
    arrivals: Arrivals<RtspSource>,
    path: String,
    tracks: Vec<Announced>,
    depacketizers: Vec<Depacketizer>,
    parameters: Vec<ffmpeg::codec::Parameters>,
    /// Whether each track's parameters are complete.
    complete: Vec<bool>,
    /// What came before they all were.
    held: Vec<(usize, ffmpeg::Packet)>,
    began: Instant,
    /// Where the source handed out is fed, once it is.
    outbox: Option<PublishedOutbox>,
    /// Each picture track that lost a packet the source had no room for,
    /// going on from its next keyframe.
    losing: Vec<bool>,
}

// SAFETY: every field is `Send` but the codec parameters, which are this
// value's own copies, used only through `&mut self`.
unsafe impl Send for Recording {}

impl Recording {
    pub(crate) fn new(arrivals: Arrivals<RtspSource>, path: &str, tracks: Vec<Announced>) -> Self {
        let complete = tracks
            .iter()
            .map(|track| track.medium != ffmpeg::media::Type::Video)
            .collect();
        Self {
            arrivals,
            path: path.to_owned(),
            depacketizers: tracks
                .iter()
                .map(|track| Depacketizer::new(track.payload))
                .collect(),
            parameters: tracks.iter().map(Announced::parameters).collect(),
            losing: vec![false; tracks.len()],
            tracks,
            complete,
            held: Vec::new(),
            began: Instant::now(),
            outbox: None,
        }
    }

    /// Takes one RTP packet of `track`.
    pub(crate) fn take(&mut self, track: usize, rtp: &[u8], pp_log: &PpLog) -> Result<(), Stopped> {
        let Some(depacketizer) = self.depacketizers.get_mut(track) else {
            return Ok(());
        };
        for mut packet in depacketizer.push(rtp) {
            packet.set_time_base(self.tracks[track].time_base());
            if self.outbox.is_some() {
                self.write(track, packet, pp_log)?;
                continue;
            }
            if !self.complete[track] && packet.is_key() {
                self.complete[track] =
                    self.tracks[track].complete(&mut self.parameters[track], &packet);
            }
            self.held.push((track, packet));
            let all = self.complete.iter().all(|complete| *complete);
            if all || self.held.len() >= HOLD_AT_MOST || self.began.elapsed() >= HOLD_FOR {
                self.hand_out(pp_log)?;
            }
        }
        Ok(())
    }

    /// Hands the publisher out as a source of its tracks, and feeds it what
    /// was held.
    fn hand_out(&mut self, pp_log: &PpLog) -> Result<(), Stopped> {
        let streams: Vec<StreamInfo> = self
            .parameters
            .iter()
            .zip(&self.tracks)
            .enumerate()
            .map(|(index, (parameters, track))| {
                StreamInfo::new(parameters.clone(), track.time_base()).with_index(index)
            })
            .collect();
        let (source, outbox) = RtspSource::published(&self.path, streams.clone());
        if !self.arrivals.hand(source, streams) {
            return Err(Stopped::Gone);
        }
        self.outbox = Some(outbox);
        for (track, packet) in std::mem::take(&mut self.held) {
            self.write(track, packet, pp_log)?;
        }
        Ok(())
    }

    fn write(
        &mut self,
        track: usize,
        packet: ffmpeg::Packet,
        pp_log: &PpLog,
    ) -> Result<(), Stopped> {
        let Some(outbox) = &self.outbox else {
            return Ok(());
        };
        if self.losing[track] && !packet.is_key() {
            return Ok(());
        }
        match outbox.items.try_send(PublishedItem::Packet(track, packet)) {
            Ok(()) => {
                outbox.bell.ring();
                self.losing[track] = false;
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                // A picture lost leaves those after it to the next keyframe;
                // a sound lost is only itself.
                if self.tracks[track].medium == ffmpeg::media::Type::Video && !self.losing[track] {
                    pp_warn!(
                        pp_log: pp_log,
                        "{}: track {track}'s source is not reading; going on from the next keyframe",
                        self.path
                    );
                    self.losing[track] = true;
                }
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => Err(Stopped::Gone),
        }
    }

    /// Whether the publisher has been handed out yet.
    pub(crate) fn published(&self) -> bool {
        self.outbox.is_some()
    }
}
