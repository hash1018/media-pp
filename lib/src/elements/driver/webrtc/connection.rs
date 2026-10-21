//! What every WebRTC connection of this crate does with its `str0m`
//! [`Rtc`] the same way, whoever drives it — a [`WebRtcPeer`] on its own
//! thread, or a server session on one of the server's.
//!
//! [`WebRtcPeer`]: super::WebRtcPeer

use std::time::Instant;

use ffmpeg_next as ffmpeg;
use str0m::{
    Event, Input, Output, Rtc, RtcError,
    format::Codec,
    media::{MediaData, Mid},
    net::Transmit,
};

use super::{
    command::{TrackId, WebRtcError},
    peer::packet_rtp_time,
};

/// What a connection has for its owner next.
// Handed straight on and never kept, so the size of an event costs a move,
// where boxing it would cost an allocation for every packet received.
#[allow(clippy::large_enum_variant)]
pub(super) enum Next {
    /// A datagram to send.
    Transmit(Transmit),
    /// Something that happened.
    Event(Event),
    /// Nothing until this time, or until something is put in.
    Wait(Instant),
}

/// Takes what a connection has for its owner, one thing at a time, until it
/// says to wait.
///
/// str0m's pacer lets one packet out and then asks for the time again with
/// a deadline already past. Such a deadline is answered here and the
/// sending goes on, rather than one packet leaving per turn of the owner's
/// loop — which held a peer to a thousand packets a second and let a
/// server's players pile up without bound. Eight in a row with nothing sent
/// between is taken as a connection that keeps asking without moving on.
pub(super) struct Outputs {
    asked: u32,
}

impl Outputs {
    pub(super) fn new() -> Self {
        Self { asked: 0 }
    }

    pub(super) fn next(&mut self, rtc: &mut Rtc) -> Result<Next, RtcError> {
        loop {
            match rtc.poll_output()? {
                Output::Timeout(deadline) => {
                    let now = Instant::now();
                    self.asked += 1;
                    if deadline > now || self.asked > 8 {
                        return Ok(Next::Wait(deadline));
                    }
                    rtc.handle_input(Input::Timeout(now))?;
                }
                Output::Transmit(transmit) => {
                    self.asked = 0;
                    return Ok(Next::Transmit(transmit));
                }
                Output::Event(event) => {
                    self.asked = 0;
                    return Ok(Next::Event(event));
                }
            }
        }
    }
}

/// Writes `packet` — `codec`, Annex-B where it is H.264 or later, its time
/// from zero — into the media `mid` of `rtc`, under the payload type its
/// negotiation gave `codec`. Never guesses another: str0m packetizes
/// whatever bytes it is handed under the type chosen, so an H.264 packet
/// sent as VP8 is a well-formed stream no receiver can decode.
///
/// Nothing is written, and nothing said, where `mid` has no writer — a
/// track not negotiated yet, or one gone.
pub(super) fn write(
    rtc: &mut Rtc,
    mid: Mid,
    track: TrackId,
    codec: Codec,
    packet: &ffmpeg::Packet,
) -> Result<(), WebRtcError> {
    let Some(writer) = rtc.writer(mid) else {
        return Ok(());
    };
    let mut negotiated = Vec::new();
    let mut pt = None;
    for params in writer.payload_params() {
        let candidate = params.spec().codec;
        if !negotiated.contains(&candidate) {
            negotiated.push(candidate);
        }
        if candidate == codec && pt.is_none() {
            pt = Some(params.pt());
        }
    }
    let Some(pt) = pt else {
        return Err(WebRtcError::OutboundCodecNotNegotiated {
            track_id: track,
            codec,
            negotiated,
        });
    };
    let data = packet.data().unwrap_or(&[]).to_vec();
    let rtp_time = packet_rtp_time(packet)?;
    writer.write(pt, Instant::now(), rtp_time, data)?;
    Ok(())
}

/// A packet the remote side sent, as a packet of this crate: its RTP time
/// over the codec's clock as its timestamps — RTP carries no reordering,
/// so they are the same — and a keyframe flagged as one.
pub(super) fn received(data: &MediaData) -> ffmpeg::Packet {
    let mut packet = ffmpeg::Packet::copy(&data.data);
    packet.set_time_base(ffmpeg::Rational::new(1, data.time.denom() as i32));
    let pts = data.time.numer() as i64;
    packet.set_pts(Some(pts));
    packet.set_dts(Some(pts));
    if data.is_keyframe() {
        packet.set_flags(packet.flags() | ffmpeg::codec::packet::Flags::KEY);
    }
    packet
}
