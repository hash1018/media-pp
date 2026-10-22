//! What a playing session sends each of its tracks through: [`RtspTrackSink`].

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ffmpeg_next::Rational;

use crate::buffer::{MediaBuffer, PacketBuffer};
use crate::element::{Element, ElementType, RawSink, element_pp_log};
use crate::elements::broadcast::time_base;
use crate::error::Result;
use crate::pp_log::PpLog;

/// How much faster than it plays a UDP player is sent what the broadcast
/// held when it joined: twice, so it catches up in as long as that was.
/// Sent at once, the group of pictures it joins with overflows the
/// player's socket, and UDP has no way to say slow down.
const UDP_CATCH_UP: f64 = 2.0;

/// Where one track of a playing session is sent.
#[derive(Debug, Clone, Copy)]
pub(super) enum Delivery {
    /// Interleaved on the connection: RTP on `channel`, RTCP on the next.
    Tcp { channel: u8 },
    /// To the player's ports.
    Udp { rtp: SocketAddr, rtcp: SocketAddr },
}

/// A packet to be sent to a player, and where: cut into RTP by the
/// connection as it sends it, so that what waits to be sent is the packet
/// every other reader shares, not RTP of this player's own.
pub(super) struct Outgoing {
    pub(super) delivery: Delivery,
    /// Which of the session's described tracks it is — its packetizer's.
    pub(super) stream: usize,
    pub(super) packet: PacketBuffer,
    /// Not before when, for a UDP player catching up.
    pub(super) due: Option<Instant>,
}

/// What a playing session's track sinks have for its connection to send,
/// in order — every track's in one, so they leave in the order the
/// broadcast had them.
#[derive(Clone, Default)]
pub(super) struct Outbox(Arc<Mutex<Held>>);

#[derive(Default)]
struct Held {
    queue: VecDeque<Outgoing>,
    /// Where a UDP player's sending is paced from: when the first packet
    /// was sent, and where it is on the session's timeline — one for every
    /// track, so they stay together.
    paced_from: Option<(Instant, f64)>,
}

impl Outbox {
    fn lock(&self) -> MutexGuard<'_, Held> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The next thing to send, if anything is held.
    pub(super) fn take(&self) -> Option<Outgoing> {
        self.lock().queue.pop_front()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.lock().queue.is_empty()
    }
}

/// Sends one track of a playing RTSP session: puts each packet it is
/// handed in the session's outbox, with when a UDP player catching up is
/// due it, for the connection to cut into RTP and send.
pub(super) struct RtspTrackSink {
    pp_log: PpLog,
    name: Arc<str>,
    delivery: Delivery,
    stream: usize,
    /// The unit the track's packets count in, where one does not say.
    unit: Rational,
    outbox: Outbox,
}

impl RtspTrackSink {
    pub(super) fn new(
        name: String,
        delivery: Delivery,
        stream: usize,
        unit: Rational,
        outbox: Outbox,
    ) -> Self {
        let name: Arc<str> = name.into();
        Self {
            pp_log: element_pp_log(ElementType::RtspServer, &name, None),
            name,
            delivery,
            stream,
            unit,
            outbox,
        }
    }
}

impl Element for RtspTrackSink {
    fn name(&self) -> Arc<str> {
        Arc::clone(&self.name)
    }

    fn element_type(&self) -> ElementType {
        ElementType::RtspServer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl RawSink for RtspTrackSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Packet(packet) = buf else {
            return Ok(());
        };
        let mut held = self.outbox.lock();
        let due = match self.delivery {
            Delivery::Udp { .. } => packet.dts().or(packet.pts()).map(|ts| {
                let at = ts as f64 * f64::from(time_base(&packet, Some(self.unit)));
                let (began, from) = *held.paced_from.get_or_insert((Instant::now(), at));
                began + Duration::from_secs_f64(((at - from) / UDP_CATCH_UP).max(0.0))
            }),
            Delivery::Tcp { .. } => None,
        };
        held.queue.push_back(Outgoing {
            delivery: self.delivery,
            stream: self.stream,
            packet,
            due,
        });
        Ok(())
    }
}
