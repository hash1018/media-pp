//! One session of a WebRTC server: a player or a publisher, its `str0m`
//! connection, and what goes through it.

use std::collections::HashMap;
use std::net::UdpSocket;
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;
use str0m::change::SdpOffer;
use str0m::channel::ChannelId;
use str0m::format::Codec;
use str0m::media::{MediaData, MediaKind, Mid};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc};

use crate::buffer::MediaBuffer;
use crate::element::RawSink;
use crate::elements::broadcast::{Timeline, same_shape, time_base};
use crate::elements::{
    Broadcast, BroadcastItem, BroadcastReader, BroadcastSink, StreamInfo, TrackFormat,
};
use crate::pp_log::{PpLog, pp_debug, pp_info};

use super::super::peer::packet_rtp_time;
use super::super::stream_info::{StreamInfoProbe, str0m_codec};
use super::annex_b::AnnexB;
use super::{SessionId, SessionRole, Shared, WebRtcServerError, WebRtcServerEvent};

/// How much faster than it plays a player is sent what the broadcast held
/// when it connected: twice, so it catches up in as long as that was,
/// without a burst its socket cannot hold.
const CATCH_UP: f64 = 2.0;

/// How long a session may take to connect after its offer is answered.
const CONNECT_WITHIN: Duration = Duration::from_secs(30);

/// How long a publisher's packets are held while its tracks say what they
/// are, before what has said so is published without the rest.
const HOLD_FOR: Duration = Duration::from_secs(10);
const HOLD_AT_MOST: usize = 4096;

/// Whether `broadcast` has anything a WebRTC player can be sent.
pub(crate) fn playable(broadcast: &Broadcast) -> bool {
    !broadcast.is_closed()
        && broadcast.tracks().is_some_and(|tracks| {
            tracks
                .iter()
                .any(|track| str0m_codec(track.parameters.id()).is_some())
        })
}

/// One session.
pub(crate) struct Session {
    id: SessionId,
    path: String,
    rtc: Rtc,
    broadcast: Broadcast,
    /// When str0m next asks for the time.
    deadline: Instant,
    opened: Instant,
    connected: bool,
    /// The media the client offered, as `str0m` added it.
    media: Vec<(Mid, MediaKind)>,
    /// The client's data channels, by label.
    channels: HashMap<String, ChannelId>,
    role: Role,
    pp_log: PpLog,
}

// SAFETY: everything a session holds is `Send` but the codec parameters
// inside the stream descriptions and the publisher's probes, which are its
// own copies, used only through `&mut self` on the server's one thread.
unsafe impl Send for Session {}

enum Role {
    Play(Box<Playing>),
    Publish(Box<Publishing>),
}

/// A player's reading of its path.
#[derive(Default)]
struct Playing {
    /// Made once it connects, so it starts at the newest keyframe then.
    reader: Option<BroadcastReader>,
    timeline: Timeline,
    /// The run being read's tracks.
    tracks: Vec<StreamInfo>,
    /// Where each of them is sent, if anywhere.
    routes: Vec<Option<Route>>,
    /// The next packet, held until it is due.
    pending: Option<Pending>,
    /// Where pacing is measured from: when the first packet was sent, and
    /// where it is on the timeline.
    paced_from: Option<(Instant, f64)>,
}

/// Where one of a broadcast's tracks is sent.
struct Route {
    mid: Mid,
    codec: Codec,
    annex_b: Option<AnnexB>,
}

struct Pending {
    track: usize,
    packet: ffmpeg::Packet,
    due: Instant,
}

/// A publisher's tracks on their way into its path.
#[derive(Default)]
struct Publishing {
    /// Each medium sent, in the order the publisher offered them, and what
    /// it has said of itself.
    tracks: Vec<(Mid, StreamInfoProbe, Option<TrackFormat>)>,
    held: Vec<(Mid, ffmpeg::Packet)>,
    sinks: Option<HashMap<Mid, BroadcastSink>>,
    first_packet: Option<Instant>,
}

impl Session {
    /// A session of `role` on `path`, answering `offer`: the session and
    /// its SDP answer.
    pub(crate) fn open(
        id: SessionId,
        path: &str,
        role: SessionRole,
        broadcast: Broadcast,
        offer: &str,
        candidates: &[Candidate],
        pp_log: PpLog,
    ) -> Result<(Self, String), WebRtcServerError> {
        let offer = SdpOffer::from_sdp_string(offer)
            .map_err(|error| WebRtcServerError::InvalidOffer(error.to_string()))?;
        let now = Instant::now();
        let mut rtc = Rtc::builder().set_ice_lite(true).build(now);
        for candidate in candidates {
            rtc.add_local_candidate(candidate.clone());
        }
        let answer = rtc
            .sdp_api()
            .accept_offer(offer)
            .map_err(|error| WebRtcServerError::InvalidOffer(error.to_string()))?;
        let role = match role {
            SessionRole::Play => Role::Play(Box::default()),
            SessionRole::Publish => Role::Publish(Box::default()),
        };
        Ok((
            Self {
                id,
                path: path.to_owned(),
                rtc,
                broadcast,
                deadline: now,
                opened: now,
                connected: false,
                media: Vec::new(),
                channels: HashMap::new(),
                role,
                pp_log,
            },
            answer.to_sdp_string(),
        ))
    }

    pub(crate) fn path(&self) -> &str {
        &self.path
    }

    pub(crate) fn accepts(&self, input: &Input) -> bool {
        self.rtc.accepts(input)
    }

    pub(crate) fn input(&mut self, input: Input) {
        let _ = self.rtc.handle_input(input);
    }

    /// Moves the session on to `now`: what is due is written, what the
    /// connection has to send is sent, what happened is told. When it next
    /// wants to be driven, or why it has ended.
    pub(crate) fn drive(
        &mut self,
        now: Instant,
        socket: &UdpSocket,
        shared: &Shared,
    ) -> Result<Instant, String> {
        if now >= self.deadline {
            self.rtc
                .handle_input(Input::Timeout(now))
                .map_err(|error| error.to_string())?;
        }
        let mut next = now + Duration::from_secs(1);
        if let Some(due) = self.pump(now)? {
            next = next.min(due);
        }
        loop {
            match self.rtc.poll_output().map_err(|error| error.to_string())? {
                Output::Timeout(deadline) => {
                    self.deadline = deadline;
                    next = next.min(deadline);
                    break;
                }
                Output::Transmit(transmit) => {
                    let _ = socket.send_to(&transmit.contents, transmit.destination);
                }
                Output::Event(event) => self.event(event, shared)?,
            }
        }
        if !self.rtc.is_alive() {
            return Err("the connection closed".into());
        }
        if !self.connected && now.duration_since(self.opened) > CONNECT_WITHIN {
            return Err("it never connected".into());
        }
        Ok(next)
    }

    /// Closes the connection, telling the client.
    pub(crate) fn close(&mut self, socket: &UdpSocket) {
        if !self.rtc.is_alive() {
            return;
        }
        let _ = self.rtc.close();
        while let Ok(output) = self.rtc.poll_output() {
            match output {
                Output::Transmit(transmit) => {
                    let _ = socket.send_to(&transmit.contents, transmit.destination);
                }
                Output::Timeout(_) => break,
                Output::Event(_) => {}
            }
        }
    }

    /// Sends `data` on the data channel `label`.
    pub(crate) fn send(&mut self, label: &str, binary: bool, data: &[u8]) -> Result<(), ()> {
        let id = *self.channels.get(label).ok_or(())?;
        let mut channel = self.rtc.channel(id).ok_or(())?;
        channel.write(binary, data).map(|_| ()).map_err(|_| ())
    }

    fn event(&mut self, event: Event, shared: &Shared) -> Result<(), String> {
        let tell = |event| {
            let _ = shared.events.0.send(event);
        };
        match event {
            Event::Connected => {
                self.connected = true;
                pp_info!(pp_log: &self.pp_log, "session {} connected", self.id);
                let role = match &mut self.role {
                    Role::Play(playing) => {
                        playing.reader = Some(self.broadcast.subscribe());
                        SessionRole::Play
                    }
                    Role::Publish(_) => SessionRole::Publish,
                };
                tell(WebRtcServerEvent::Connected {
                    session: self.id,
                    path: self.path.clone(),
                    role,
                });
            }
            Event::IceConnectionStateChange(IceConnectionState::Disconnected) => {
                return Err("the client left".into());
            }
            Event::Closed => return Err("the client closed the connection".into()),
            Event::MediaAdded(added) => {
                self.media.push((added.mid, added.kind));
                if let Role::Publish(publishing) = &mut self.role {
                    publishing
                        .tracks
                        .push((added.mid, StreamInfoProbe::new(), None));
                }
            }
            Event::MediaData(data) => {
                if let Role::Publish(publishing) = &mut self.role {
                    publishing.take(data, &self.broadcast, &self.pp_log)?;
                }
            }
            Event::ChannelOpen(id, label) => {
                self.channels.insert(label.clone(), id);
                tell(WebRtcServerEvent::ChannelOpen {
                    session: self.id,
                    channel: label,
                });
            }
            Event::ChannelData(data) => {
                let label = self
                    .channels
                    .iter()
                    .find(|(_, id)| **id == data.id)
                    .map(|(label, _)| label.clone())
                    .unwrap_or_default();
                tell(WebRtcServerEvent::Data {
                    session: self.id,
                    channel: label,
                    binary: data.binary,
                    data: data.data,
                });
            }
            Event::ChannelClose(id) => self.channels.retain(|_, open| *open != id),
            _ => {}
        }
        Ok(())
    }

    /// Writes what is due of a player's broadcast into its connection: when
    /// the next packet is due, if one is held.
    fn pump(&mut self, now: Instant) -> Result<Option<Instant>, String> {
        let Role::Play(playing) = &mut self.role else {
            return Ok(None);
        };
        if playing.reader.is_none() {
            return Ok(None);
        }
        // Bounded, so that one player cannot keep the others waiting.
        for _ in 0..512 {
            if let Some(pending) = playing.pending.take() {
                if pending.due > now {
                    let due = pending.due;
                    playing.pending = Some(pending);
                    return Ok(Some(due));
                }
                write(&mut self.rtc, playing, pending);
                continue;
            }
            let Some(item) = playing.reader.as_mut().and_then(BroadcastReader::try_next) else {
                return Ok(None);
            };
            match item {
                BroadcastItem::Started(tracks) => {
                    if !playing.tracks.is_empty() {
                        if !same_shape(&playing.tracks, &tracks) {
                            return Err("its broadcast's tracks changed".into());
                        }
                        playing.timeline.next_run();
                    }
                    playing.routes = routes(&mut self.rtc, &self.media, &tracks);
                    if playing.routes.iter().all(Option::is_none) {
                        return Err("nothing it offered to take can be sent".into());
                    }
                    pp_debug!(
                        pp_log: &self.pp_log,
                        "session {} sends {} of {} tracks",
                        self.id,
                        playing.routes.iter().flatten().count(),
                        tracks.len()
                    );
                    playing.tracks = tracks;
                }
                BroadcastItem::Packet {
                    track,
                    packet,
                    arrived,
                } => {
                    let Some(Some(_)) = playing.routes.get(track) else {
                        continue;
                    };
                    let unit = time_base(&packet, Some(playing.tracks[track].time_base));
                    let Some(mut stamped) = playing.timeline.restamp(&packet, unit, arrived) else {
                        continue;
                    };
                    stamped.set_time_base(unit);
                    let Some(at) = stamped
                        .dts()
                        .or(stamped.pts())
                        .map(|ts| ts as f64 * f64::from(unit))
                    else {
                        continue;
                    };
                    let (began, from) = *playing.paced_from.get_or_insert((now, at));
                    let due = began + Duration::from_secs_f64(((at - from) / CATCH_UP).max(0.0));
                    playing.pending = Some(Pending {
                        track,
                        packet: stamped,
                        due,
                    });
                }
                BroadcastItem::Skipped => {
                    pp_debug!(pp_log: &self.pp_log, "session {} fell behind and goes on from a keyframe", self.id);
                }
                BroadcastItem::Ended => {}
                BroadcastItem::Closed => return Err("its broadcast closed".into()),
            }
        }
        Ok(Some(now))
    }
}

/// Where each of `tracks` is sent: the first track of each kind in a codec
/// both WebRTC and the client's media take, to that media.
fn routes(rtc: &mut Rtc, media: &[(Mid, MediaKind)], tracks: &[StreamInfo]) -> Vec<Option<Route>> {
    let mut routes: Vec<Option<Route>> = tracks.iter().map(|_| None).collect();
    let mut used = Vec::new();
    for (index, track) in tracks.iter().enumerate() {
        let Some(codec) = str0m_codec(track.parameters.id()) else {
            continue;
        };
        let kind = match track.kind {
            ffmpeg::media::Type::Video => MediaKind::Video,
            ffmpeg::media::Type::Audio => MediaKind::Audio,
            _ => continue,
        };
        let Some(&(mid, _)) = media.iter().find(|(mid, offered)| {
            *offered == kind
                && !used.contains(mid)
                && rtc.writer(*mid).is_some_and(|writer| {
                    writer
                        .payload_params()
                        .any(|params| params.spec().codec == codec)
                })
        }) else {
            continue;
        };
        used.push(mid);
        let annex_b = match codec {
            Codec::H264 | Codec::H265 => Some(AnnexB::for_track(
                extradata(&track.parameters),
                codec == Codec::H265,
            )),
            _ => None,
        };
        routes[index] = Some(Route {
            mid,
            codec,
            annex_b,
        });
    }
    routes
}

pub(super) fn extradata(parameters: &ffmpeg::codec::Parameters) -> &[u8] {
    // SAFETY: reading the parameters' own extradata, which lives as long as
    // they do, by the size they give.
    unsafe {
        let par = parameters.as_ptr();
        if (*par).extradata.is_null() || (*par).extradata_size <= 0 {
            return &[];
        }
        std::slice::from_raw_parts((*par).extradata, (*par).extradata_size as usize)
    }
}

/// Writes `pending` into the player's connection.
fn write(rtc: &mut Rtc, playing: &mut Playing, pending: Pending) {
    let Some(Some(route)) = playing.routes.get(pending.track) else {
        return;
    };
    let Some(data) = pending.packet.data() else {
        return;
    };
    let data = match &route.annex_b {
        Some(annex_b) => match annex_b.convert(data, pending.packet.is_key()) {
            Some(data) => data,
            None => return,
        },
        None => data.to_vec(),
    };
    let Ok(rtp_time) = packet_rtp_time(&pending.packet) else {
        return;
    };
    let Some(writer) = rtc.writer(route.mid) else {
        return;
    };
    let Some(pt) = writer
        .payload_params()
        .find(|params| params.spec().codec == route.codec)
        .map(|params| params.pt())
    else {
        return;
    };
    let _ = writer.write(pt, Instant::now(), rtp_time, data);
}

impl Publishing {
    /// One of the publisher's packets.
    fn take(
        &mut self,
        data: MediaData,
        broadcast: &Broadcast,
        pp_log: &PpLog,
    ) -> Result<(), String> {
        let Some(index) = self.tracks.iter().position(|(mid, ..)| *mid == data.mid) else {
            return Ok(());
        };
        let spec = data.params.spec();
        let mut packet = ffmpeg::Packet::copy(&data.data);
        packet.set_time_base(ffmpeg::Rational::new(1, data.time.denom() as i32));
        let pts = data.time.numer() as i64;
        packet.set_pts(Some(pts));
        packet.set_dts(Some(pts));
        if data.is_keyframe() {
            packet.set_flags(packet.flags() | ffmpeg::codec::packet::Flags::KEY);
        }
        if let Some(sinks) = self.sinks.as_mut() {
            return write_published(sinks, data.mid, packet);
        }
        let (_, probe, format) = &mut self.tracks[index];
        if format.is_none()
            && let Some(info) = probe.observe(spec, &data.data)
        {
            *format = info.track_format().ok();
        }
        self.held.push((data.mid, packet));
        let first = *self.first_packet.get_or_insert_with(Instant::now);
        let all = self.tracks.iter().all(|(.., format)| format.is_some());
        if all || self.held.len() >= HOLD_AT_MOST || first.elapsed() >= HOLD_FOR {
            self.publish(broadcast, pp_log)?;
        }
        Ok(())
    }

    fn publish(&mut self, broadcast: &Broadcast, pp_log: &PpLog) -> Result<(), String> {
        let ready: Vec<(Mid, TrackFormat)> = self
            .tracks
            .iter()
            .filter_map(|(mid, _, format)| Some((*mid, format.clone()?)))
            .collect();
        if ready.is_empty() {
            self.held.clear();
            return Err("nothing it sent could be described".into());
        }
        let formats: Vec<TrackFormat> = ready.iter().map(|(_, format)| format.clone()).collect();
        let sinks = broadcast
            .publish(&formats)
            .map_err(|error| error.to_string())?;
        pp_info!(pp_log: pp_log, "published {} tracks into {}", ready.len(), broadcast.name());
        let mut sinks: HashMap<Mid, BroadcastSink> =
            ready.into_iter().map(|(mid, _)| mid).zip(sinks).collect();
        for (mid, packet) in std::mem::take(&mut self.held) {
            write_published(&mut sinks, mid, packet)?;
        }
        self.sinks = Some(sinks);
        Ok(())
    }
}

fn write_published(
    sinks: &mut HashMap<Mid, BroadcastSink>,
    mid: Mid,
    packet: ffmpeg::Packet,
) -> Result<(), String> {
    let Some(sink) = sinks.get_mut(&mid) else {
        return Ok(());
    };
    match sink.consume(MediaBuffer::packet(packet)) {
        Ok(()) => Ok(()),
        Err(crate::error::Error::BroadcastError(error)) => Err(error.to_string()),
        Err(_) => Ok(()),
    }
}
