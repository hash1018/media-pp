//! What a session delivers, read on threads of its own and handed to the
//! source as packets: the connection — the session's requests and
//! answers, and its media where that is interleaved in it — on one, and
//! each track's UDP socket, where that is how its media comes, on another.
//!
//! A thread hands what it reads to the source through a channel the source
//! waits on, so the source answers a pause or a stop the moment it comes,
//! whatever the network does. The channel holds a few seconds of packets;
//! a source not reading — paused, or a pipeline slower than the stream —
//! loses what comes past that, and a picture track then goes on from its
//! next keyframe rather than handing on pictures that cannot be decoded.

use std::io::{self, Read, Write};
use std::net::{TcpStream, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{SendTimeoutError, Sender, TrySendError};
use ffmpeg_next as ffmpeg;

use crate::core::bell::Bell;
use crate::elements::rtsp::depacketize::{Depacketizer, nal_units};
use crate::elements::rtsp::dts::Dts;
use crate::elements::rtsp::message::{self, Incoming, Request, Response};
use crate::elements::rtsp::sdp::Announced;
use crate::pp_log::{PpLog, pp_debug, pp_info, pp_warn};

use super::RtspSourceError;
use super::pull::Client;

/// How long a thread of a session waits on its socket before it looks
/// again at whether it is to stop, or a keepalive is due.
pub(super) const POLL: Duration = Duration::from_millis(100);

/// How many packets wait for the source at most: some seconds of a
/// camera's pictures and sound.
pub(super) const BACKLOG: usize = 256;

/// What the source is handed.
pub(super) enum Item {
    Packet(usize, ffmpeg::Packet),
    /// The session ended as it should: the server or publisher closed it.
    End,
    Failed(RtspSourceError),
}

/// One track as it is received: its RTP taken apart into packets, each
/// given its time base and its decoding time.
///
/// A picture track begins at a keyframe — a server may begin a session
/// anywhere, and what comes before the first keyframe cannot be decoded —
/// and so does it again after pictures were lost to a full channel. The
/// pictures an H.265 keyframe it begins at was coded ahead of, and are
/// shown before it (RASL), are left out with it: they refer to pictures
/// that were never received.
pub(super) struct Track {
    pub(super) announced: Announced,
    depacketizer: Depacketizer,
    dts: Option<Dts>,
    begun: Begun,
    /// Whether the last packet was lost to a full channel.
    losing: bool,
}

/// Where a picture track is in beginning.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Begun {
    /// Waiting for a keyframe.
    No,
    /// At a keyframe, its leading pictures still to come.
    Leading,
    Yes,
}

impl Track {
    pub(super) fn new(announced: Announced) -> Self {
        let dts = Dts::new(announced.codec, &announced.extradata, announced.clock_rate);
        let begun = if announced.medium == ffmpeg::media::Type::Video {
            Begun::No
        } else {
            Begun::Yes
        };
        Self {
            depacketizer: Depacketizer::new(announced.payload),
            dts,
            announced,
            begun,
            losing: false,
        }
    }

    /// The packets one RTP packet of this track completes.
    fn take(&mut self, rtp: &[u8]) -> Vec<ffmpeg::Packet> {
        let time_base = self.announced.time_base();
        let mut done = self.depacketizer.push(rtp);
        for packet in &mut done {
            packet.set_time_base(time_base);
        }
        done
    }

    /// Whether `packet` is one to hand on, as far as beginning goes.
    fn begins(&mut self, packet: &ffmpeg::Packet) -> bool {
        match self.begun {
            Begun::Yes => true,
            Begun::No if !packet.is_key() => false,
            Begun::No => {
                self.begun = if self.announced.codec == ffmpeg::codec::Id::HEVC {
                    Begun::Leading
                } else {
                    Begun::Yes
                };
                true
            }
            Begun::Leading => {
                match packet.data().and_then(hevc_picture) {
                    // Skipped leading pictures, with their keyframe the
                    // first there is.
                    Some(8 | 9) => return false,
                    // Decodable leading pictures, and keyframes.
                    Some(6 | 7 | 16..=23) => {}
                    _ => self.begun = Begun::Yes,
                }
                true
            }
        }
    }

    /// Hands `packets` of track `index` to the source, each with its
    /// decoding time; `false` once the source is gone.
    fn deliver(
        &mut self,
        index: usize,
        packets: Vec<ffmpeg::Packet>,
        tx: &Outbox,
        pp_log: &PpLog,
    ) -> bool {
        for mut packet in packets {
            if !self.begins(&packet) {
                continue;
            }
            let pts = packet.pts().unwrap_or(0);
            let key = packet.is_key();
            let dts = match (self.dts.as_mut(), packet.data()) {
                (Some(dts), Some(unit)) => dts.stamp(unit, key, pts, self.announced.clock_rate),
                _ => pts,
            };
            packet.set_dts(Some(dts));
            match tx.items.try_send(Item::Packet(index, packet)) {
                Ok(()) => {
                    tx.bell.ring();
                    self.losing = false;
                }
                Err(TrySendError::Full(_)) => {
                    if !self.losing {
                        pp_warn!(
                            pp_log: pp_log,
                            "track {index}: the source is not reading; losing what arrives"
                        );
                    }
                    self.losing = true;
                    // A picture lost leaves those after it to the next
                    // keyframe; a sound lost is only itself.
                    if self.announced.medium == ffmpeg::media::Type::Video {
                        self.begun = Begun::No;
                    }
                }
                Err(TrySendError::Disconnected(_)) => return false,
            }
        }
        true
    }
}

/// The NAL unit type of an H.265 access unit's first slice.
fn hevc_picture(unit: &[u8]) -> Option<u8> {
    nal_units(unit)
        .map(|nal| (nal.first().copied().unwrap_or(0xff) >> 1) & 0x3f)
        .find(|kind| *kind <= 31)
}

/// Where a session's threads hand the source what they read: the channel,
/// and the bell its thread parks on — rung with each item, and as each
/// thread ends, so that the source sees the channel gone.
#[derive(Clone)]
pub(super) struct Outbox {
    pub(super) items: Sender<Item>,
    pub(super) bell: Bell,
}

impl Drop for Outbox {
    fn drop(&mut self) {
        self.bell.ring();
    }
}

/// How much of a connection is read at once.
const READ_AHEAD: usize = 64 * 1024;

/// A session's RTSP connection: what has been read of it, and what is
/// taken off that a message at a time.
///
/// What is read goes into one buffer for the whole connection, taken from
/// the front as each message is: nothing is cleared or moved for a packet,
/// only what is left of a message read in part, to the front, when the
/// buffer's end is near. A camera's RTP is most of what comes, and its
/// packets are handed on from where they lie (`take_interleaved`).
pub(super) struct Connection {
    pub(super) stream: TcpStream,
    buffer: Vec<u8>,
    /// What of `buffer` has been read and not yet taken.
    start: usize,
    end: usize,
}

impl Connection {
    pub(super) fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            buffer: vec![0; READ_AHEAD],
            start: 0,
            end: 0,
        }
    }

    /// Reads what the socket has, waiting as long as its read timeout:
    /// `Ok(false)` once the peer has closed it.
    pub(super) fn fill(&mut self) -> io::Result<bool> {
        if self.start == self.end {
            (self.start, self.end) = (0, 0);
        } else if self.buffer.len() - self.end < READ_AHEAD / 4 {
            // What is left of a message read in part, to the front — and
            // room for a message larger than the buffer, where it is one.
            self.buffer.copy_within(self.start..self.end, 0);
            (self.start, self.end) = (0, self.end - self.start);
            if self.buffer.len() - self.end < READ_AHEAD / 4 {
                self.buffer.resize(self.buffer.len() * 2, 0);
            }
        }
        let read = self.stream.read(&mut self.buffer[self.end..])?;
        self.end += read;
        Ok(read > 0)
    }

    /// The next whole message read, if there is one.
    pub(super) fn next(&mut self) -> Result<Option<Incoming>, message::Malformed> {
        match message::parse(&self.buffer[self.start..self.end])? {
            Some((incoming, taken)) => {
                self.start += taken;
                Ok(Some(incoming))
            }
            None => Ok(None),
        }
    }

    /// The next whole message where it is an interleaved packet: its
    /// channel, and where its payload lies in [`Self::read`] — taken, not
    /// copied out.
    fn take_interleaved(&mut self) -> Option<(u8, std::ops::Range<usize>)> {
        let read = &self.buffer[self.start..self.end];
        if read.len() < 4 || read[0] != b'$' {
            return None;
        }
        let length = usize::from(u16::from_be_bytes([read[2], read[3]]));
        if read.len() < 4 + length {
            return None;
        }
        let channel = read[1];
        let payload = self.start + 4..self.start + 4 + length;
        self.start += 4 + length;
        Some((channel, payload))
    }

    /// What was read, where [`Self::take_interleaved`] says.
    fn read(&self, at: std::ops::Range<usize>) -> &[u8] {
        &self.buffer[at]
    }

    pub(super) fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.stream.write_all(bytes)
    }

    /// Answers `request` with `response`.
    pub(super) fn answer(&mut self, request: &Request, response: Response) -> io::Result<()> {
        self.send(&response.to_bytes(request.header("CSeq")))
    }
}

/// Whether `error` is a read that waited its time out, not a failure.
pub(super) fn timed_out(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

/// Who is at the other end of the connection.
pub(super) enum Peer {
    /// A server this source asked for the session: it is kept alive with
    /// requests, and torn down when the source stops.
    Server(Box<Client>),
    /// A publisher that pushed the session to this source: its requests
    /// are answered, and its `TEARDOWN` or its closing ends the stream.
    Publisher,
}

/// How a session's media arrives.
pub(super) enum Media {
    /// Interleaved in the connection: the channel each track's RTP comes
    /// on.
    Interleaved(Vec<u8>),
    /// Over UDP: each track's RTP socket, and its RTCP's, kept bound so
    /// that the port pair stays the session's.
    Udp(Vec<(UdpSocket, UdpSocket)>),
}

/// A session as the handshake leaves it, ready to be read.
pub(super) struct Session {
    pub(super) connection: Connection,
    pub(super) peer: Peer,
    pub(super) tracks: Vec<Track>,
    pub(super) media: Media,
    /// Media interleaved in the connection that came with the handshake's
    /// last answers, before the session was read here.
    pub(super) early: Vec<(u8, Vec<u8>)>,
}

/// How long a session may deliver nothing, and what that is when it does.
#[derive(Clone, Copy)]
pub(super) struct Silence {
    pub(super) after: Duration,
    pub(super) error: fn(Duration) -> RtspSourceError,
}

/// The threads reading a session; stopping them — or dropping this — tears
/// the session down where this source asked for it, and waits for them.
pub(super) struct Reader {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Reader {
    /// Starts reading `session` into `tx`.
    pub(super) fn start(
        session: Session,
        tx: Outbox,
        silence: Silence,
        pp_log: &PpLog,
    ) -> io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let heard = Heard::new();
        let mut threads = Vec::new();
        let Session {
            connection,
            peer,
            tracks,
            media,
            early,
        } = session;
        let (tracks, channels) = match media {
            Media::Interleaved(channels) => (tracks, channels),
            Media::Udp(sockets) => {
                for (index, (track, (rtp, rtcp))) in tracks.into_iter().zip(sockets).enumerate() {
                    rtp.set_read_timeout(Some(POLL))?;
                    let (tx, stop, heard, pp_log) =
                        (tx.clone(), Arc::clone(&stop), heard.clone(), pp_log.clone());
                    threads.push(
                        std::thread::Builder::new()
                            .name(format!("rtsp-udp-{index}"))
                            .spawn(move || {
                                let _rtcp = rtcp;
                                udp(index, track, &rtp, &tx, &stop, &heard, &pp_log);
                            })?,
                    );
                }
                (Vec::new(), Vec::new())
            }
        };
        connection.stream.set_read_timeout(Some(POLL))?;
        let control = Control {
            connection,
            peer,
            tracks,
            channels,
            tx,
            stop: Arc::clone(&stop),
            heard,
            silence,
            pp_log: pp_log.clone(),
        };
        threads.push(
            std::thread::Builder::new()
                .name("rtsp-session".to_owned())
                .spawn(move || control.run(early))?,
        );
        Ok(Self { stop, threads })
    }

    /// Stops the threads, tearing the session down where this source asked
    /// for it, and waits for them: at most a poll and a `TEARDOWN`'s write.
    pub(super) fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.stop();
    }
}

/// When media last arrived, shared by a session's threads: milliseconds
/// since the session began to be read.
#[derive(Clone)]
struct Heard {
    began: Instant,
    at: Arc<AtomicU64>,
}

impl Heard {
    fn new() -> Self {
        Self {
            began: Instant::now(),
            at: Arc::new(AtomicU64::new(0)),
        }
    }

    fn now(&self) {
        let at = u64::try_from(self.began.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.at.store(at, Ordering::Relaxed);
    }

    fn since(&self) -> Duration {
        let at = Duration::from_millis(self.at.load(Ordering::Relaxed));
        self.began.elapsed().saturating_sub(at)
    }
}

/// The connection's thread.
struct Control {
    connection: Connection,
    peer: Peer,
    /// Interleaved, the tracks, and the channel each one's RTP comes on;
    /// over UDP, none — their threads have them.
    tracks: Vec<Track>,
    channels: Vec<u8>,
    tx: Outbox,
    stop: Arc<AtomicBool>,
    heard: Heard,
    silence: Silence,
    pp_log: PpLog,
}

impl Control {
    fn run(mut self, early: Vec<(u8, Vec<u8>)>) {
        for (channel, data) in early {
            if !interleaved(
                &mut self.tracks,
                &self.channels,
                (&self.tx, &self.heard, &self.pp_log),
                channel,
                &data,
            ) {
                return;
            }
        }
        let ending = self.read();
        if let Peer::Server(client) = &mut self.peer {
            // Asked to stop, or failed: a server that may still be there
            // is told. One that closed the connection has nothing to tell.
            if !matches!(ending, Some(Item::End)) {
                client.tear_down(&mut self.connection);
            }
        }
        if let Some(item) = ending {
            finish(&self.tx, item, &self.stop);
        }
    }

    /// Reads the session until it ends — what it ended with — or until the
    /// source stops it: `None`.
    fn read(&mut self) -> Option<Item> {
        loop {
            if self.stop.load(Ordering::Acquire) {
                return None;
            }
            if let Peer::Server(client) = &mut self.peer {
                client.keep_alive(&mut self.connection);
            }
            if self.heard.since() > self.silence.after {
                return Some(Item::Failed((self.silence.error)(self.silence.after)));
            }
            match self.connection.fill() {
                Ok(true) => {}
                Ok(false) => {
                    pp_info!(pp_log: &self.pp_log, "the peer closed the connection");
                    return Some(Item::End);
                }
                Err(error) if timed_out(&error) => continue,
                // A publisher's close can arrive as a reset: its own
                // `TEARDOWN` answered after it had gone, say.
                Err(error)
                    if matches!(self.peer, Peer::Publisher)
                        && matches!(
                            error.kind(),
                            io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted
                        ) =>
                {
                    pp_info!(pp_log: &self.pp_log, "the publisher closed its connection: {error}");
                    return Some(Item::End);
                }
                Err(error) => return Some(Item::Failed(RtspSourceError::Io(error))),
            }
            loop {
                if let Some((channel, at)) = self.connection.take_interleaved() {
                    let data = self.connection.read(at);
                    if !interleaved(
                        &mut self.tracks,
                        &self.channels,
                        (&self.tx, &self.heard, &self.pp_log),
                        channel,
                        data,
                    ) {
                        return None;
                    }
                    continue;
                }
                match self.connection.next() {
                    Ok(None) => break,
                    // Read whole by `take_interleaved`, first.
                    Ok(Some(Incoming::Interleaved { .. })) => {}
                    // A keepalive's answer: nothing waits for it.
                    Ok(Some(Incoming::Reply(reply))) => {
                        if !(200..300).contains(&reply.status) {
                            pp_debug!(pp_log: &self.pp_log, "a keepalive was answered {}", reply.status);
                        }
                    }
                    Ok(Some(Incoming::Request(request))) => {
                        if let Some(ending) = self.requested(&request) {
                            return Some(ending);
                        }
                    }
                    Err(message::Malformed(what)) => {
                        return Some(Item::Failed(RtspSourceError::Malformed(what)));
                    }
                }
            }
        }
    }

    /// Answers a request the peer sent; what the session ends with where
    /// it ends it.
    fn requested(&mut self, request: &Request) -> Option<Item> {
        let (response, ending) = match (&self.peer, request.method.as_str()) {
            (Peer::Publisher, "TEARDOWN") => (Response::new(200), Some(Item::End)),
            (Peer::Publisher, "OPTIONS") => (
                Response::new(200).header("Public", super::push::PUBLIC),
                None,
            ),
            (Peer::Publisher, "GET_PARAMETER" | "SET_PARAMETER") => (Response::new(200), None),
            (Peer::Publisher, _) => (Response::new(455), None),
            // A server's own request — an `ANNOUNCE` of a change, a
            // `SET_PARAMETER` — is not one this client acts on.
            (Peer::Server(_), _) => (Response::new(501), None),
        };
        let response = match self.session_id() {
            Some(id) => response.header("Session", id),
            None => response,
        };
        if let Err(error) = self.connection.answer(request, response) {
            return Some(Item::Failed(RtspSourceError::Io(error)));
        }
        ending
    }

    fn session_id(&self) -> Option<String> {
        match &self.peer {
            Peer::Server(_) => None,
            Peer::Publisher => Some(super::push::SESSION.to_owned()),
        }
    }
}

/// Takes one interleaved packet into the track whose RTP comes on
/// `channel`; `false` once the source is gone.
fn interleaved(
    tracks: &mut [Track],
    channels: &[u8],
    (tx, heard, pp_log): (&Outbox, &Heard, &PpLog),
    channel: u8,
    data: &[u8],
) -> bool {
    let Some(index) = channels.iter().position(|&rtp| rtp == channel) else {
        // RTCP, or a channel no track was set up on.
        return true;
    };
    heard.now();
    let track = &mut tracks[index];
    let packets = track.take(data);
    track.deliver(index, packets, tx, pp_log)
}

/// A track's UDP thread: its RTP put back in order and taken apart.
fn udp(
    index: usize,
    mut track: Track,
    socket: &UdpSocket,
    tx: &Outbox,
    stop: &AtomicBool,
    heard: &Heard,
    pp_log: &PpLog,
) {
    let mut buffer = vec![0u8; 64 * 1024];
    let mut order = Reorder::default();
    while !stop.load(Ordering::Acquire) {
        let released = match socket.recv(&mut buffer) {
            Ok(size) => {
                heard.now();
                order.push(&buffer[..size])
            }
            Err(error) if timed_out(&error) => order.flush(),
            // Windows reports an ICMP unreachable for an earlier send as
            // a reset of the next receive; it is no end of this socket.
            Err(error) if error.kind() == io::ErrorKind::ConnectionReset => continue,
            Err(error) => {
                pp_warn!(pp_log: pp_log, "track {index}: its UDP socket failed: {error}");
                return;
            }
        };
        for rtp in released {
            let packets = track.take(&rtp);
            if !track.deliver(index, packets, tx, pp_log) {
                return;
            }
        }
    }
}

/// UDP's RTP put back in the order it was sent, by sequence number: a
/// packet waits for those before it a little while, and is let go on its
/// own past that, the gap left for what takes the packets apart to see.
#[derive(Default)]
struct Reorder {
    next: Option<u16>,
    /// Held, by sequence number, with when each arrived.
    held: Vec<(u16, Vec<u8>, Instant)>,
}

/// How many packets wait for one missing, and how long.
const HOLD: usize = 64;
const HOLD_FOR: Duration = Duration::from_millis(50);

impl Reorder {
    fn push(&mut self, packet: &[u8]) -> Vec<Vec<u8>> {
        if packet.len() < 12 {
            return Vec::new();
        }
        let sequence = u16::from_be_bytes([packet[2], packet[3]]);
        let Some(next) = self.next else {
            self.next = Some(sequence.wrapping_add(1));
            return vec![packet.to_vec()];
        };
        // Behind what was let go already — a duplicate, or too late.
        if (sequence.wrapping_sub(next) as i16) < 0 {
            return Vec::new();
        }
        if !self.held.iter().any(|(held, ..)| *held == sequence) {
            let at = self
                .held
                .iter()
                .position(|(held, ..)| (held.wrapping_sub(sequence) as i16) > 0)
                .unwrap_or(self.held.len());
            self.held
                .insert(at, (sequence, packet.to_vec(), Instant::now()));
        }
        self.release()
    }

    /// What has waited long enough, on a quiet socket.
    fn flush(&mut self) -> Vec<Vec<u8>> {
        self.release()
    }

    fn release(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some((sequence, _, arrived)) = self.held.first() {
            let due = Some(*sequence) == self.next
                || self.held.len() > HOLD
                || arrived.elapsed() >= HOLD_FOR;
            if !due {
                break;
            }
            let (sequence, packet, _) = self.held.remove(0);
            self.next = Some(sequence.wrapping_add(1));
            out.push(packet);
        }
        out
    }
}

/// Hands the source what the session ended with, waiting for room —
/// but not past a stop, nor for a source that is gone.
fn finish(tx: &Outbox, mut item: Item, stop: &AtomicBool) {
    loop {
        match tx.items.send_timeout(item, POLL) {
            Ok(()) => {
                tx.bell.ring();
                return;
            }
            Err(SendTimeoutError::Disconnected(_)) => return,
            Err(SendTimeoutError::Timeout(back)) => {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                item = back;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rtp(sequence: u16) -> Vec<u8> {
        let mut packet = vec![0x80, 96];
        packet.extend_from_slice(&sequence.to_be_bytes());
        packet.extend_from_slice(&[0; 8]);
        packet
    }

    fn sequences(packets: &[Vec<u8>]) -> Vec<u16> {
        packets
            .iter()
            .map(|packet| u16::from_be_bytes([packet[2], packet[3]]))
            .collect()
    }

    #[test]
    fn udp_packets_come_back_in_order() {
        let mut order = Reorder::default();
        assert_eq!(sequences(&order.push(&rtp(65_534))), [65_534]);
        assert!(order.push(&rtp(0)).is_empty(), "waits for 65535");
        assert_eq!(sequences(&order.push(&rtp(65_535))), [65_535, 0]);
        assert!(order.push(&rtp(65_535)).is_empty(), "a duplicate");
        assert_eq!(sequences(&order.push(&rtp(1))), [1]);
    }

    #[test]
    fn a_packet_lost_is_waited_for_only_so_long() {
        let mut order = Reorder::default();
        order.push(&rtp(10));
        assert!(order.push(&rtp(12)).is_empty());
        std::thread::sleep(HOLD_FOR);
        assert_eq!(sequences(&order.flush()), [12], "the gap is left");
        let mut order = Reorder::default();
        order.push(&rtp(0));
        let mut released = Vec::new();
        for sequence in 2..=(HOLD as u16 + 2) {
            released.extend(order.push(&rtp(sequence)));
        }
        assert_eq!(sequences(&released)[0], 2, "let go past what it holds");
    }
}
