//! A session pushed to this source, as
//! [`RtspSource::listen`](super::RtspSource::listen) takes it: one
//! publisher's connection accepted on the URL's port, its `ANNOUNCE`,
//! `SETUP`s and `RECORD` answered, and the port closed again — then its
//! media read until it tears the session down or closes.

use std::io;
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use crate::elements::rtsp::message::{self, Incoming, Request, Response, Transport};
use crate::elements::rtsp::sdp::{self, Announced, Refused};
use crate::pp_log::{PpLog, pp_debug};

use super::pull::port_pair;
use super::receive::{Connection, Media, Peer, Session, Track, timed_out};
use super::{RtspListenOptions, RtspSourceError};

/// What a publisher may ask of this end.
pub(super) const PUBLIC: &str =
    "OPTIONS, ANNOUNCE, SETUP, RECORD, TEARDOWN, GET_PARAMETER, SET_PARAMETER";

/// The one session there is: one publisher at a time.
pub(super) const SESSION: &str = "1";

/// Waits for a publisher at `url` and takes its session through to
/// `RECORD`.
pub(super) fn accept(
    url: &str,
    options: RtspListenOptions,
    pp_log: &PpLog,
) -> Result<Session, RtspSourceError> {
    if url.contains('\0') {
        return Err(RtspSourceError::InvalidListenUrl);
    }
    let address = message::address(url).ok_or(RtspSourceError::InvalidListenUrl)?;
    let deadline = Instant::now().checked_add(options.wait);
    let stream = {
        let listener = TcpListener::bind((address.host.as_str(), address.port))
            .map_err(RtspSourceError::Bind)?;
        listener
            .set_nonblocking(true)
            .map_err(RtspSourceError::Bind)?;
        loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if timed_out(&error) => {
                    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        return Err(RtspSourceError::NoPublisher(options.wait));
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(error) => return Err(RtspSourceError::Bind(error)),
            }
        }
        // The listener goes here: a second publisher is refused while the
        // first is received.
    };
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_write_timeout(Some(options.timeout))?;
    handshake(stream, deadline, options.wait, pp_log)
}

/// What a publisher has set up so far.
struct Setting {
    tracks: Vec<Announced>,
    /// Each track's delivery, once set up: interleaved on a channel, or
    /// over a pair of UDP sockets.
    delivery: Vec<Option<Delivery>>,
}

enum Delivery {
    Channel(u8),
    Udp(Box<(std::net::UdpSocket, std::net::UdpSocket)>),
}

/// Answers the publisher's requests until it records.
fn handshake(
    stream: TcpStream,
    deadline: Option<Instant>,
    wait: Duration,
    pp_log: &PpLog,
) -> Result<Session, RtspSourceError> {
    let local = stream.local_addr()?.ip();
    let mut connection = Connection::new(stream);
    let mut setting: Option<Setting> = None;
    loop {
        let request = next_request(&mut connection, deadline, wait)?;
        let mut response = match request.method.as_str() {
            "OPTIONS" => Response::new(200).header("Public", PUBLIC),
            "ANNOUNCE" => {
                let text = String::from_utf8_lossy(&request.body);
                match sdp::parse(&text) {
                    Ok(tracks) => {
                        setting = Some(Setting {
                            delivery: tracks.iter().map(|_| None).collect(),
                            tracks,
                        });
                        Response::new(200)
                    }
                    Err(refused) => {
                        let status = match refused {
                            Refused::Malformed => 400,
                            Refused::Unsupported(_) => 415,
                        };
                        connection.answer(&request, Response::new(status))?;
                        return Err(match refused {
                            Refused::Malformed => RtspSourceError::Malformed("announcement"),
                            Refused::Unsupported(what) => RtspSourceError::Unsupported(what),
                        });
                    }
                }
            }
            "SETUP" => match setting.as_mut() {
                Some(setting) => set_up(setting, &request, local)?,
                None => Response::new(455),
            },
            "RECORD" => match setting.take() {
                Some(setting) => {
                    connection.answer(&request, Response::new(200).header("Session", SESSION))?;
                    pp_debug!(pp_log: pp_log, "the publisher records");
                    return recording(connection, setting);
                }
                None => Response::new(455),
            },
            "TEARDOWN" => {
                connection.answer(&request, Response::new(200))?;
                return Err(RtspSourceError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "the publisher left before it recorded",
                )));
            }
            "GET_PARAMETER" | "SET_PARAMETER" => Response::new(200),
            _ => Response::new(501),
        };
        if setting.is_some() {
            response = response.header("Session", SESSION);
        }
        connection.answer(&request, response)?;
    }
}

/// The publisher's next request, by the deadline.
fn next_request(
    connection: &mut Connection,
    deadline: Option<Instant>,
    wait: Duration,
) -> Result<Request, RtspSourceError> {
    loop {
        while let Some(incoming) = connection.next()? {
            if let Incoming::Request(request) = incoming {
                return Ok(request);
            }
        }
        let left = match deadline {
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Err(RtspSourceError::NoPublisher(wait));
                }
                Some(left.max(Duration::from_millis(1)))
            }
            None => None,
        };
        match connection.fill(left) {
            Ok(true) => {}
            Ok(false) => {
                return Err(RtspSourceError::Io(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "the publisher closed the connection before it recorded",
                )));
            }
            Err(error) if timed_out(&error) => {}
            Err(error) => return Err(error.into()),
        }
    }
}

/// The answer to a `SETUP` of one of the announced tracks.
fn set_up(
    setting: &mut Setting,
    request: &Request,
    local: std::net::IpAddr,
) -> Result<Response, RtspSourceError> {
    let Some(track) = sdp::track_named(&request.url, &setting.tracks) else {
        return Ok(Response::new(404));
    };
    let Some((transport, _)) = request.header("Transport").and_then(message::transport) else {
        return Ok(Response::new(461));
    };
    let answer = match transport {
        Transport::Tcp { channels } => {
            let (rtp, rtcp) = channels.unwrap_or_else(|| {
                let rtp = u8::try_from(2 * track).unwrap_or(0);
                (rtp, rtp.saturating_add(1))
            });
            setting.delivery[track] = Some(Delivery::Channel(rtp));
            format!("RTP/AVP/TCP;unicast;interleaved={rtp}-{rtcp};mode=record")
        }
        Transport::Udp { rtp, rtcp } => {
            let (ours, theirs) = port_pair(local)?;
            let port = ours.local_addr()?.port();
            setting.delivery[track] = Some(Delivery::Udp(Box::new((ours, theirs))));
            format!(
                "RTP/AVP/UDP;unicast;client_port={rtp}-{rtcp};server_port={port}-{};mode=record",
                port + 1
            )
        }
    };
    Ok(Response::new(200).header("Transport", answer))
}

/// The session as the publisher set it up: the tracks it set up, all
/// delivered one way.
fn recording(connection: Connection, setting: Setting) -> Result<Session, RtspSourceError> {
    let mut tracks = Vec::new();
    let mut channels = Vec::new();
    let mut sockets = Vec::new();
    for (announced, delivery) in setting.tracks.into_iter().zip(setting.delivery) {
        match delivery {
            Some(Delivery::Channel(channel)) => {
                channels.push(channel);
                tracks.push(Track::new(announced));
            }
            Some(Delivery::Udp(pair)) => {
                sockets.push(*pair);
                tracks.push(Track::new(announced));
            }
            None => {}
        }
    }
    let media = match (channels.is_empty(), sockets.is_empty()) {
        (false, true) => Media::Interleaved(channels),
        (true, false) => Media::Udp(sockets),
        (true, true) => {
            return Err(RtspSourceError::Unsupported(
                "a publisher that set up no track".to_owned(),
            ));
        }
        (false, false) => {
            return Err(RtspSourceError::Unsupported(
                "a publisher sending some tracks over TCP and others over UDP".to_owned(),
            ));
        }
    };
    Ok(Session {
        connection,
        peer: Peer::Publisher,
        tracks,
        media,
        early: Vec::new(),
    })
}
