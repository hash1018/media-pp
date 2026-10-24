//! What a WHIP or WHEP client is made of — one HTTP exchange, and a thread
//! driving the connection it makes; see [`super::WebRtcMuxer`] and
//! [`super::WebRtcSource`].

use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use str0m::media::Mid;
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Rtc};

use crate::buffer::MediaBuffer;
use crate::pp_log::{PpLog, pp_debug, pp_info, pp_warn};

use super::command::{TrackId, WebRtcError};

/// How a [`WebRtcSource`](super::WebRtcSource) or a
/// [`WebRtcMuxer`](super::WebRtcMuxer) reaches its server — the URL it is
/// made with being the WHEP or WHIP endpoint.
#[derive(Debug, Clone)]
pub struct WebRtcOptions {
    /// Sent as `Authorization: Bearer <token>`, where the server asks for
    /// one — a stream key, as WHIP servers call it.
    pub bearer_token: Option<String>,
    /// How long opening waits for the server's answer, and then for the
    /// connection to come up. Ten seconds by default.
    pub connect_within: Duration,
    /// How long a [`WebRtcSource`](super::WebRtcSource) then waits for its
    /// tracks to say what they carry — for H.264, until its parameter sets
    /// arrive, which a stream sends ahead of each keyframe. It returns once
    /// every track has, or at this with the ones that have. Five seconds by
    /// default. A muxer, which says what it sends, does not wait.
    pub describe_within: Duration,
}

impl Default for WebRtcOptions {
    fn default() -> Self {
        Self {
            bearer_token: None,
            connect_within: Duration::from_secs(10),
            describe_within: Duration::from_secs(5),
        }
    }
}
use super::connection::{self, Next, Outputs};
use super::inbox::{Delivered, Inbox};
use super::outbox::Outbox;
use super::stream_info::{StreamInfoProbe, WebRtcStreamInfo};

/// The longest the connection's thread waits on its socket before it looks
/// at what its track sinks have put in the outbox, and whether to stop.
const POLL: Duration = Duration::from_millis(5);

/// How long ending a session on its server may take, as the client goes.
const DELETE_WITHIN: Duration = Duration::from_secs(2);

/// A WHIP or WHEP URL, and the host and port its server is reached at.
pub(super) struct Url {
    pub(super) full: String,
    host: String,
    port: u16,
}

impl Url {
    pub(super) fn parse(url: &str) -> Result<Self, WebRtcError> {
        let invalid = || WebRtcError::InvalidUrl(url.to_owned());
        let (scheme, rest) = url.split_once("://").ok_or_else(invalid)?;
        let default_port = match scheme.to_ascii_lowercase().as_str() {
            "http" => 80,
            "https" => 443,
            _ => return Err(invalid()),
        };
        let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
        let authority = authority.rsplit('@').next().unwrap_or_default();
        let (host, port) = match authority.strip_prefix('[') {
            // [v6]:port
            Some(v6) => {
                let (host, after) = v6.split_once(']').ok_or_else(invalid)?;
                let port = match after.strip_prefix(':') {
                    Some(port) => port.parse().map_err(|_| invalid())?,
                    None => default_port,
                };
                (host.to_owned(), port)
            }
            None => match authority.rsplit_once(':') {
                Some((host, port)) => (host.to_owned(), port.parse().map_err(|_| invalid())?),
                None => (authority.to_owned(), default_port),
            },
        };
        if host.is_empty() {
            return Err(invalid());
        }
        Ok(Self {
            full: url.to_owned(),
            host,
            port,
        })
    }

    /// `location`, as a `Location` header gives it, made absolute against
    /// this URL.
    fn resolve(&self, location: &str) -> String {
        if location.contains("://") {
            return location.to_owned();
        }
        let (scheme, rest) = self.full.split_once("://").unwrap_or(("http", ""));
        let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
        if location.starts_with('/') {
            return format!("{scheme}://{authority}{location}");
        }
        let path = &rest[authority.len()..];
        let path = path.split(['?', '#']).next().unwrap_or_default();
        let dir = path.rsplit_once('/').map_or("", |(dir, _)| dir);
        format!("{scheme}://{authority}{dir}/{location}")
    }
}

/// Where a WHIP or WHEP session lives on its server, to end it there.
pub(super) struct Resource {
    agent: ureq::Agent,
    url: String,
    token: Option<String>,
}

impl Resource {
    /// Ends the session on its server — best effort, as the client goes.
    fn delete(self) {
        let mut request = self.agent.delete(&self.url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let _ = request.call();
    }
}

fn http_agent(within: Duration) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(within))
            .build(),
    )
}

/// Posts `offer` to `url`: the answer, and where the session lives — `None`
/// where the server gave no `Location`, which leaves nothing to end there.
pub(super) fn exchange(
    url: &Url,
    token: Option<&str>,
    offer: &str,
    within: Duration,
) -> Result<(String, Option<Resource>), WebRtcError> {
    let failed = |reason: String| WebRtcError::Request {
        url: url.full.clone(),
        reason,
    };
    let agent = http_agent(within);
    let mut request = agent
        .post(&url.full)
        .header("Content-Type", "application/sdp")
        .header("Accept", "application/sdp");
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let mut response = request
        .send(offer)
        .map_err(|error| failed(error.to_string()))?;
    let status = response.status().as_u16();
    let location = response
        .headers()
        .get("location")
        .and_then(|value| value.to_str().ok())
        .map(|location| url.resolve(location));
    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|error| failed(error.to_string()))?;
    if !matches!(status, 200 | 201) {
        return Err(WebRtcError::Refused {
            url: url.full.clone(),
            status,
            body: body.trim().to_owned(),
        });
    }
    let resource = location.map(|location| Resource {
        agent: http_agent(DELETE_WITHIN),
        url: location,
        token: token.map(str::to_owned),
    });
    Ok((body, resource))
}

/// A UDP socket on the address this machine reaches `url`'s server from,
/// and that address — the candidate the offer gives.
pub(super) fn socket_toward(url: &Url) -> Result<(UdpSocket, SocketAddr), WebRtcError> {
    let failed = |reason: String| WebRtcError::Request {
        url: url.full.clone(),
        reason,
    };
    let server = (url.host.as_str(), url.port)
        .to_socket_addrs()
        .map_err(|error| failed(error.to_string()))?
        .next()
        .ok_or_else(|| failed("its host has no address".into()))?;
    // Which of this machine's addresses the route to the server leaves
    // from: a UDP socket "connected" there says, sending nothing.
    let any: SocketAddr = if server.is_ipv4() {
        "0.0.0.0:0".parse().expect("an address")
    } else {
        "[::]:0".parse().expect("an address")
    };
    let probe = UdpSocket::bind(any).map_err(WebRtcError::Io)?;
    probe.connect(server).map_err(WebRtcError::Io)?;
    let local_ip = probe.local_addr().map_err(WebRtcError::Io)?.ip();
    let socket = UdpSocket::bind(SocketAddr::new(local_ip, 0)).map_err(WebRtcError::Io)?;
    let local = socket.local_addr().map_err(WebRtcError::Io)?;
    Ok((socket, local))
}

/// A host candidate at `local`.
pub(super) fn candidate(local: SocketAddr) -> Result<Candidate, WebRtcError> {
    Candidate::host(local, "udp").map_err(|error| WebRtcError::Request {
        url: local.to_string(),
        reason: error.to_string(),
    })
}

/// How the connection came up, for the call that opened it.
pub(super) enum Up {
    Connected,
    Failed(String),
}

/// One track a client receives: where its packets go, and what reads what
/// it is from them.
pub(super) struct Incoming {
    pub(super) mid: Mid,
    pub(super) inbox: Inbox,
}

/// What a client's connection is made with.
pub(super) struct Parts {
    pub(super) name: String,
    pub(super) rtc: Rtc,
    pub(super) socket: UdpSocket,
    pub(super) local: SocketAddr,
    /// Where each outgoing track's packets go, by the id its sink writes
    /// with, and the outbox its sinks write into.
    pub(super) outgoing: HashMap<TrackId, Mid>,
    pub(super) outbox: Option<Outbox>,
    pub(super) incoming: Vec<Incoming>,
    pub(super) resource: Option<Resource>,
    pub(super) pp_log: PpLog,
}

/// A client's connection, driven on a thread of its own until it is
/// dropped or the connection ends. Dropping it stops the thread — which
/// closes the connection, ends the session on its server, and lets the
/// tracks it fed end — and joins it.
pub(super) struct Client {
    stop: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// What a client's connection says as it goes.
pub(super) struct Said {
    /// How it came up — said once.
    pub(super) up: Receiver<Up>,
    /// Each incoming track's description, as its packets give it.
    pub(super) described: Receiver<(Mid, WebRtcStreamInfo)>,
}

impl Client {
    pub(super) fn spawn(parts: Parts) -> Result<(Arc<Self>, Said), WebRtcError> {
        let stop = Arc::new(AtomicBool::new(false));
        let (up_tx, up) = bounded(1);
        let (described_tx, described) = unbounded();
        let name = format!("{}-webrtc", parts.name);
        let driving = Driving {
            incoming: parts
                .incoming
                .into_iter()
                .map(|incoming| {
                    (
                        incoming.mid,
                        Track {
                            inbox: incoming.inbox,
                            probe: StreamInfoProbe::new(),
                            described: false,
                        },
                    )
                })
                .collect(),
            rtc: parts.rtc,
            socket: parts.socket,
            local: parts.local,
            outgoing: parts.outgoing,
            outbox: parts.outbox,
            resource: parts.resource,
            up: Some(up_tx),
            described: described_tx,
            ended: None,
            pp_log: parts.pp_log,
        };
        let thread = std::thread::Builder::new()
            .name(name)
            .spawn({
                let stop = Arc::clone(&stop);
                move || driving.run(&stop)
            })
            .map_err(WebRtcError::Io)?;
        Ok((
            Arc::new(Self {
                stop,
                thread: Mutex::new(Some(thread)),
            }),
            Said { up, described },
        ))
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let thread = self
            .thread
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(thread) = thread
            && thread.thread().id() != std::thread::current().id()
        {
            let _ = thread.join();
        }
    }
}

struct Track {
    inbox: Inbox,
    probe: StreamInfoProbe,
    described: bool,
}

/// The connection's thread's own.
struct Driving {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    outgoing: HashMap<TrackId, Mid>,
    outbox: Option<Outbox>,
    incoming: HashMap<Mid, Track>,
    resource: Option<Resource>,
    up: Option<Sender<Up>>,
    described: Sender<(Mid, WebRtcStreamInfo)>,
    /// Why the connection ended, once it has.
    ended: Option<String>,
    pp_log: PpLog,
}

// SAFETY: everything here is `Send` but the codec parameters inside the
// tracks' probes, which are this thread's own copies.
unsafe impl Send for Driving {}

impl Driving {
    fn run(mut self, stop: &AtomicBool) {
        let mut buffer = vec![0u8; 65_536];
        loop {
            let deadline = self.drain();
            if self.write_outbox() {
                // str0m sends what was written once time has moved on.
                if let Err(error) = self.rtc.handle_input(Input::Timeout(Instant::now())) {
                    self.ended = Some(error.to_string());
                }
                continue;
            }
            if self.ended.is_some() || !self.rtc.is_alive() || stop.load(Ordering::Acquire) {
                break;
            }
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .clamp(Duration::from_millis(1), POLL);
            if self.socket.set_read_timeout(Some(wait)).is_err() {
                break;
            }
            let input = match self.socket.recv_from(&mut buffer) {
                Ok((length, source)) => match buffer[..length].try_into() {
                    Ok(contents) => Input::Receive(
                        Instant::now(),
                        Receive {
                            proto: Protocol::Udp,
                            source,
                            destination: self.local,
                            contents,
                        },
                    ),
                    Err(_) => continue,
                },
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    Input::Timeout(Instant::now())
                }
                Err(error) => {
                    self.ended = Some(error.to_string());
                    break;
                }
            };
            if let Err(error) = self.rtc.handle_input(input) {
                self.ended = Some(error.to_string());
            }
        }
        self.end();
    }

    /// Takes what the connection has to send and to say until it waits:
    /// when it next asks for the time.
    fn drain(&mut self) -> Instant {
        let mut outputs = Outputs::new();
        loop {
            match outputs.next(&mut self.rtc) {
                Ok(Next::Wait(deadline)) => return deadline,
                Ok(Next::Transmit(transmit)) => {
                    let _ = self
                        .socket
                        .send_to(&transmit.contents, transmit.destination);
                }
                Ok(Next::Event(event)) => self.event(event),
                Err(error) => {
                    self.ended = Some(error.to_string());
                    return Instant::now();
                }
            }
        }
    }

    /// Writes what the track sinks have put in the outbox: whether anything
    /// was.
    fn write_outbox(&mut self) -> bool {
        let Some(outbox) = &self.outbox else {
            return false;
        };
        let held = outbox.take_all();
        let wrote = !held.is_empty();
        for outgoing in held {
            let Some(&mid) = self.outgoing.get(&outgoing.track) else {
                continue;
            };
            if let Err(error) = connection::write(&mut self.rtc, mid, &outgoing) {
                pp_debug!(pp_log: &self.pp_log, "a packet was not sent: {error}");
            }
        }
        wrote
    }

    fn event(&mut self, event: Event) {
        match event {
            Event::Connected => {
                pp_info!(pp_log: &self.pp_log, "connected");
                if let Some(up) = self.up.take() {
                    let _ = up.send(Up::Connected);
                }
            }
            Event::IceConnectionStateChange(IceConnectionState::Disconnected) => {
                self.ended = Some("the connection was lost".into());
            }
            Event::Closed => self.ended = Some("the server closed the connection".into()),
            Event::MediaData(data) => {
                let Some(track) = self.incoming.get_mut(&data.mid) else {
                    return;
                };
                if !track.described
                    && let Some(info) = track.probe.observe(data.params.spec(), &data.data)
                {
                    track.described = true;
                    let _ = self.described.send((data.mid, info));
                }
                let packet = connection::received(&data);
                match track.inbox.push(MediaBuffer::packet(packet)) {
                    Delivered::Taken => {}
                    Delivered::Full => {
                        pp_debug!(pp_log: &self.pp_log, "a packet was dropped: its track is not read fast enough");
                    }
                    Delivered::Gone => {
                        self.incoming.remove(&data.mid);
                    }
                }
            }
            _ => {}
        }
    }

    /// Closes the connection, ends what it fed, and ends the session on
    /// its server.
    fn end(mut self) {
        let why = self.ended.take();
        if self.rtc.is_alive() {
            let _ = self.rtc.close();
            let _ = self.drain();
        }
        for track in self.incoming.values() {
            track.inbox.close();
        }
        if let Some(outbox) = &self.outbox {
            outbox.close();
        }
        if let Some(up) = self.up.take() {
            let _ = up
                .send(Up::Failed(why.clone().unwrap_or_else(|| {
                    "it was stopped before it connected".into()
                })));
        }
        match &why {
            Some(why) => pp_warn!(pp_log: &self.pp_log, "ended: {why}"),
            None => pp_info!(pp_log: &self.pp_log, "ended"),
        }
        if let Some(resource) = self.resource.take() {
            resource.delete();
        }
    }
}

/// Waits for `said` to say the connection to `url`'s server is up, within
/// `within`.
pub(super) fn connected(said: &Said, url: &Url, within: Duration) -> Result<(), WebRtcError> {
    match said.up.recv_timeout(within) {
        Ok(Up::Connected) => Ok(()),
        Ok(Up::Failed(reason)) => Err(WebRtcError::Request {
            url: url.full.clone(),
            reason,
        }),
        Err(_) => Err(WebRtcError::NotConnected(within)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_url_gives_its_host_and_port() {
        let url = Url::parse("https://user:pw@example.com/live/whip").unwrap();
        assert_eq!((url.host.as_str(), url.port), ("example.com", 443));
        let url = Url::parse("http://127.0.0.1:8080/cam/whep?x=1").unwrap();
        assert_eq!((url.host.as_str(), url.port), ("127.0.0.1", 8080));
        let url = Url::parse("http://[::1]:9/a").unwrap();
        assert_eq!((url.host.as_str(), url.port), ("::1", 9));
        assert!(Url::parse("rtsp://example.com/x").is_err());
        assert!(Url::parse("http:///x").is_err());
    }

    #[test]
    fn a_location_is_made_absolute_against_the_url() {
        let url = Url::parse("http://h:1/live/whip").unwrap();
        assert_eq!(url.resolve("/live/whip/abc"), "http://h:1/live/whip/abc");
        assert_eq!(url.resolve("whip/abc"), "http://h:1/live/whip/abc");
        assert_eq!(url.resolve("https://other/x"), "https://other/x");
    }
}
