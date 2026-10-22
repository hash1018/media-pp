//! A publisher's stream written into its path's broadcast: [`Recording`].

use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;

use crate::buffer::MediaBuffer;
use crate::element::RawSink;
use crate::elements::{Broadcast, BroadcastError, BroadcastSink, StreamInfo};
use crate::error::Error;

use super::announce::{Announced, Payload};
use super::depacketize::{Depacketizer, key_unit, nal_units};
use crate::elements::driver::parameters::{picture_size, set_extradata};

/// How long a recording holds what it is sent while it waits for every
/// picture's size, and how much — past either, it publishes without.
const HOLD_FOR: Duration = Duration::from_secs(10);
const HOLD_AT_MOST: usize = 4096;

/// Why a recording cannot go on.
pub(crate) enum Stopped {
    /// The broadcast was closed, or published again by someone else.
    Broadcast(BroadcastError),
}

/// A publisher's tracks being written into a broadcast.
///
/// What the description does not say — a picture's size, and parameter
/// sets a publisher sends only in band — is read from each picture track's
/// first keyframe before the broadcast is published, so that what reads it,
/// a player's description or a recording's file, has it.
pub(crate) struct Recording {
    broadcast: Broadcast,
    tracks: Vec<Announced>,
    depacketizers: Vec<Depacketizer>,
    parameters: Vec<ffmpeg::codec::Parameters>,
    /// Whether each track's parameters are complete.
    complete: Vec<bool>,
    /// What came before they all were.
    held: Vec<(usize, ffmpeg::Packet)>,
    began: Instant,
    sinks: Option<Vec<BroadcastSink>>,
}

// SAFETY: every field is `Send` but the codec parameters, which are this
// value's own copies, used only through `&mut self`.
unsafe impl Send for Recording {}

impl Recording {
    pub(crate) fn new(broadcast: Broadcast, tracks: Vec<Announced>) -> Self {
        let complete = tracks
            .iter()
            .map(|track| track.medium != ffmpeg::media::Type::Video)
            .collect();
        Self {
            broadcast,
            depacketizers: tracks
                .iter()
                .map(|track| Depacketizer::new(track.payload))
                .collect(),
            parameters: tracks.iter().map(Announced::parameters).collect(),
            tracks,
            complete,
            held: Vec::new(),
            began: Instant::now(),
            sinks: None,
        }
    }

    /// Takes one RTP packet of `track`.
    pub(crate) fn take(&mut self, track: usize, rtp: &[u8]) -> Result<(), Stopped> {
        let Some(depacketizer) = self.depacketizers.get_mut(track) else {
            return Ok(());
        };
        for mut packet in depacketizer.push(rtp) {
            packet.set_time_base(self.tracks[track].time_base());
            if self.sinks.is_some() {
                self.write(track, packet)?;
                continue;
            }
            if !self.complete[track] && packet.is_key() {
                self.complete[track] =
                    complete(&self.tracks[track], &mut self.parameters[track], &packet);
            }
            self.held.push((track, packet));
            let all = self.complete.iter().all(|complete| *complete);
            if all || self.held.len() >= HOLD_AT_MOST || self.began.elapsed() >= HOLD_FOR {
                self.publish()?;
            }
        }
        Ok(())
    }

    fn publish(&mut self) -> Result<(), Stopped> {
        let formats: Vec<StreamInfo> = self
            .parameters
            .iter()
            .zip(&self.tracks)
            .map(|(parameters, track)| StreamInfo::new(parameters.clone(), track.time_base()))
            .collect();
        let sinks = self
            .broadcast
            .publish(&formats)
            .map_err(Stopped::Broadcast)?;
        self.sinks = Some(sinks);
        for (track, packet) in std::mem::take(&mut self.held) {
            self.write(track, packet)?;
        }
        Ok(())
    }

    fn write(&mut self, track: usize, packet: ffmpeg::Packet) -> Result<(), Stopped> {
        let Some(sink) = self.sinks.as_mut().and_then(|sinks| sinks.get_mut(track)) else {
            return Ok(());
        };
        match sink.consume(MediaBuffer::packet(packet)) {
            Ok(()) => Ok(()),
            Err(Error::BroadcastError(error)) => Err(Stopped::Broadcast(error)),
            Err(_) => Ok(()),
        }
    }

    /// Whether the broadcast has been published to by this recording yet.
    pub(crate) fn published(&self) -> bool {
        self.sinks.is_some()
    }
}

/// Completes a picture track's `parameters` from its first keyframe:
/// its size, read by FFmpeg's parser, and its parameter sets where the
/// description gave none. Whether the size was found.
fn complete(
    track: &Announced,
    parameters: &mut ffmpeg::codec::Parameters,
    key: &ffmpeg::Packet,
) -> bool {
    let Some(unit) = key.data() else {
        return false;
    };
    let h265 = track.payload == Payload::H265;
    if !matches!(track.payload, Payload::H264 | Payload::H265) || !key_unit(unit, h265) {
        return false;
    }
    let mut extradata = track.extradata.clone();
    if extradata.is_empty() {
        // Parameter sets sent only in band: kept from the keyframe.
        for nal in nal_units(unit) {
            let kind = if h265 {
                (nal[0] >> 1) & 0x3f
            } else {
                nal[0] & 0x1f
            };
            let parameter_set = if h265 {
                (32..=34).contains(&kind)
            } else {
                kind == 7 || kind == 8
            };
            if parameter_set {
                extradata.extend_from_slice(&[0, 0, 0, 1]);
                extradata.extend_from_slice(nal);
            }
        }
        if !extradata.is_empty() {
            set_extradata(parameters, &extradata);
        }
    }
    let mut stream = extradata;
    stream.extend_from_slice(unit);
    let Some((width, height)) = picture_size(track.codec, &[], &stream) else {
        return false;
    };
    // SAFETY: parameters this recording owns alone; plain fields.
    unsafe {
        (*parameters.as_mut_ptr()).width = width;
        (*parameters.as_mut_ptr()).height = height;
    }
    true
}
