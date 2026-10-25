//! A session asked of a server — a camera, another RTSP server — as
//! [`RtspSource::open`](super::RtspSource::open) makes it: `OPTIONS`,
//! `DESCRIBE`, a `SETUP` for each track this can take apart, `PLAY`; then
//! kept alive while it is read, and torn down when the source stops.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::time::{Duration, Instant};

use crate::elements::rtsp::digest::Authenticator;
use crate::elements::rtsp::message::{self, Incoming, Reply};
use crate::elements::rtsp::sdp::{self, Refused};
use crate::elements::{RtspOptions, RtspTransport};
use crate::pp_log::{PpLog, pp_debug};

use super::RtspSourceError;
use super::receive::{Connection, Media, Peer, Session, Track, timed_out};

/// What keeps a session asked of a server, and tears it down: the
/// requests this end still sends once the media flows.
pub(crate) struct Client {
    cseq: u32,
    credentials: Option<(String, String)>,
    auth: Option<Authenticator>,
    session: Option<String>,
    /// The URL a request about the whole session goes to.
    aggregate: String,
    /// `GET_PARAMETER` where the server says it takes it, else `OPTIONS`.
    keepalive: &'static str,
    keep_every: Duration,
    kept: Instant,
    timeout: Duration,
}

impl Client {
    fn new(credentials: Option<(String, String)>, timeout: Duration) -> Self {
        Self {
            cseq: 0,
            credentials,
            auth: None,
            session: None,
            aggregate: String::new(),
            keepalive: "OPTIONS",
            keep_every: Duration::from_secs(30),
            kept: Instant::now(),
            timeout,
        }
    }

    /// Writes `method` on `url` with `headers`, and its credentials and
    /// session where it has them.
    fn send(
        &mut self,
        connection: &mut Connection,
        method: &str,
        url: &str,
        headers: &[(&str, &str)],
    ) -> io::Result<u32> {
        self.cseq += 1;
        let authorization = self
            .auth
            .as_mut()
            .map(|auth| auth.authorization(method, url));
        let mut all: Vec<(&str, &str)> = headers.to_vec();
        if let Some(authorization) = &authorization {
            all.push(("Authorization", authorization));
        }
        if let Some(session) = &self.session {
            all.push(("Session", session));
        }
        connection.send(&message::request(method, url, self.cseq, &all, b""))?;
        Ok(self.cseq)
    }

    /// Asks `method` on `url` and waits for the answer, answering a
    /// challenge once with the credentials the URL carried. Media
    /// interleaved before the answer is kept in `early`.
    fn ask(
        &mut self,
        connection: &mut Connection,
        early: &mut Vec<(u8, Vec<u8>)>,
        method: &'static str,
        url: &str,
        headers: &[(&str, &str)],
    ) -> Result<Reply, RtspSourceError> {
        let mut challenged = false;
        loop {
            let cseq = self.send(connection, method, url, headers)?;
            let reply = self.answer(connection, early, cseq)?;
            if reply.status == 401 && !challenged {
                challenged = true;
                if let Some((user, password)) = &self.credentials
                    && let Some(auth) =
                        Authenticator::new(user, password, reply.all("WWW-Authenticate"))
                {
                    self.auth = Some(auth);
                    continue;
                }
            }
            return match reply.status {
                200..=299 => Ok(reply),
                401 => Err(RtspSourceError::Unauthorized),
                status => Err(RtspSourceError::Refused { method, status }),
            };
        }
    }

    /// The answer to request `cseq`, within the timeout.
    fn answer(
        &mut self,
        connection: &mut Connection,
        early: &mut Vec<(u8, Vec<u8>)>,
        cseq: u32,
    ) -> Result<Reply, RtspSourceError> {
        let deadline = Instant::now() + self.timeout;
        loop {
            while let Some(incoming) = connection.next()? {
                match incoming {
                    Incoming::Reply(reply)
                        if reply
                            .header("CSeq")
                            .and_then(|value| value.trim().parse::<u32>().ok())
                            == Some(cseq) =>
                    {
                        return Ok(reply);
                    }
                    Incoming::Interleaved { channel, data } => early.push((channel, data)),
                    Incoming::Reply(_) | Incoming::Request(_) => {}
                }
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(RtspSourceError::NoAnswer(self.timeout));
            }
            match connection.fill(Some(left.max(Duration::from_millis(1)))) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(RtspSourceError::Io(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the server closed the connection",
                    )));
                }
                Err(error) if timed_out(&error) => {}
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// Sends a keepalive where one is due — its answer is read with the
    /// media, and nothing waits for it.
    pub(super) fn keep_alive(&mut self, connection: &mut Connection) {
        if self.kept.elapsed() < self.keep_every {
            return;
        }
        self.kept = Instant::now();
        let url = self.aggregate.clone();
        let _ = self.send(connection, self.keepalive, &url, &[]);
    }

    /// Tells the server the session is over, not waiting for its answer:
    /// it is gone either way.
    pub(super) fn tear_down(&mut self, connection: &mut Connection) {
        let _ = connection
            .stream
            .set_write_timeout(Some(Duration::from_secs(1)));
        let url = self.aggregate.clone();
        let _ = self.send(connection, "TEARDOWN", &url, &[]);
    }
}

/// Connects to `url` and asks for its session: what is read from it, and
/// what each track the server described was — taken or refused.
pub(super) fn connect(
    url: &str,
    options: RtspOptions,
    pp_log: &PpLog,
) -> Result<(Session, Vec<String>), RtspSourceError> {
    let address = message::address(url).ok_or(RtspSourceError::InvalidUrl)?;
    let stream = reach(&address.host, address.port, options.timeout)?;
    stream.set_nodelay(true)?;
    stream.set_write_timeout(Some(options.timeout))?;
    let local = stream.local_addr()?.ip();
    let mut connection = Connection::new(stream);
    let mut client = Client::new(address.credentials, options.timeout);
    let mut early = Vec::new();

    // What the server takes, for the keepalive; a server that answers
    // `OPTIONS` with a refusal may still play.
    match client.ask(&mut connection, &mut early, "OPTIONS", &address.url, &[]) {
        Ok(reply) => {
            if reply.header("Public").is_some_and(|public| {
                public
                    .split(',')
                    .any(|method| method.trim().eq_ignore_ascii_case("GET_PARAMETER"))
            }) {
                client.keepalive = "GET_PARAMETER";
            }
        }
        Err(RtspSourceError::Refused { .. }) => {}
        Err(error) => return Err(error),
    }

    let described = client.ask(
        &mut connection,
        &mut early,
        "DESCRIBE",
        &address.url,
        &[("Accept", "application/sdp")],
    )?;
    let base = described
        .header("Content-Base")
        .or_else(|| described.header("Content-Location"))
        .unwrap_or(&address.url)
        .to_owned();
    let text = String::from_utf8_lossy(&described.body);
    let description =
        sdp::describe(&text).map_err(|_| RtspSourceError::Malformed("description"))?;
    client.aggregate = message::resolve(&base, description.control.as_deref().unwrap_or("*"));

    let mut tracks = Vec::new();
    let mut refused = Vec::new();
    for track in description.tracks {
        match track {
            Ok(announced) => tracks.push(Track::new(announced)),
            Err(Refused::Unsupported(what)) => refused.push(what),
            Err(Refused::Malformed) => refused.push("a media section it cannot read".to_owned()),
        }
    }
    if tracks.is_empty() {
        return Err(RtspSourceError::Unsupported(refused.join(", ")));
    }

    let mut channels = Vec::new();
    let mut sockets = Vec::new();
    for (index, track) in tracks.iter().enumerate() {
        let url = message::resolve(&base, &track.announced.control);
        let asked = match options.transport {
            RtspTransport::Tcp => {
                let rtp = u8::try_from(2 * index).map_err(|_| {
                    RtspSourceError::Unsupported("more tracks than channels".to_owned())
                })?;
                format!("RTP/AVP/TCP;unicast;interleaved={rtp}-{}", rtp + 1)
            }
            RtspTransport::Udp => {
                let (rtp, rtcp) = port_pair(local)?;
                let port = rtp.local_addr()?.port();
                sockets.push((rtp, rtcp));
                format!("RTP/AVP;unicast;client_port={port}-{}", port + 1)
            }
        };
        let reply = client.ask(
            &mut connection,
            &mut early,
            "SETUP",
            &url,
            &[("Transport", &asked)],
        )?;
        if client.session.is_none()
            && let Some(header) = reply.header("Session")
        {
            let (id, timeout) = message::session(header);
            client.session = Some(id);
            client.keep_every = Duration::from_secs(timeout / 2).max(Duration::from_secs(1));
        }
        let delivery = reply
            .header("Transport")
            .map(message::delivery)
            .unwrap_or_default();
        if options.transport == RtspTransport::Tcp {
            if delivery.interleaved.is_none() && delivery.server_ports.is_some() {
                return Err(RtspSourceError::Unsupported(
                    "the server would deliver over UDP, not TCP".to_owned(),
                ));
            }
            channels.push(delivery.interleaved.map_or(2 * index as u8, |(rtp, _)| rtp));
        }
    }

    let aggregate = client.aggregate.clone();
    client.ask(
        &mut connection,
        &mut early,
        "PLAY",
        &aggregate,
        &[("Range", "npt=0.000-")],
    )?;
    client.kept = Instant::now();
    pp_debug!(
        pp_log: pp_log,
        "playing {} track(s), {} left out",
        tracks.len(),
        refused.len()
    );
    let media = match options.transport {
        RtspTransport::Tcp => Media::Interleaved(channels),
        RtspTransport::Udp => Media::Udp(sockets),
    };
    Ok((
        Session {
            connection,
            peer: Peer::Server(Box::new(client)),
            tracks,
            media,
            early,
        },
        refused,
    ))
}

/// A connection to `host`'s `port`, trying each address it has, within
/// `timeout` each.
fn reach(host: &str, port: u16, timeout: Duration) -> Result<TcpStream, RtspSourceError> {
    let addresses: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|error| RtspSourceError::Connect {
            address: format!("{host}:{port}"),
            error,
        })?
        .collect();
    let mut last = io::Error::new(io::ErrorKind::NotFound, "the host has no address");
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(stream) => return Ok(stream),
            Err(error) => last = error,
        }
    }
    Err(RtspSourceError::Connect {
        address: format!("{host}:{port}"),
        error: last,
    })
}

/// Two UDP sockets on `local`, an even port and the one after it: a
/// track's RTP and RTCP.
pub(super) fn port_pair(local: IpAddr) -> Result<(UdpSocket, UdpSocket), RtspSourceError> {
    let any = match local {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    };
    let mut last = io::Error::other("no pair of ports came free");
    for _ in 0..32 {
        let rtp = UdpSocket::bind((any, 0)).map_err(RtspSourceError::Bind)?;
        let port = rtp.local_addr().map_err(RtspSourceError::Bind)?.port();
        if port % 2 != 0 || port == u16::MAX {
            continue;
        }
        match UdpSocket::bind((any, port + 1)) {
            Ok(rtcp) => return Ok((rtp, rtcp)),
            Err(error) => last = error,
        }
    }
    Err(RtspSourceError::Bind(last))
}
