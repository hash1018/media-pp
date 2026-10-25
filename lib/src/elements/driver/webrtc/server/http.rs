//! WHEP and WHIP over HTTP: [`WebRtcHttpEndpoint`].

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::elements::driver::access::{Access, AccessRequest, Credentials};
use crate::pp_log::{pp_debug, pp_info, pp_warn};

use super::{IceServer, SessionId, WebRtcServerError, WebRtcServerHandle};

/// The most an offer may take.
const MAX_BODY: usize = 256 * 1024;

/// The longest a request's line or a header may be, and the most headers
/// it may have; one longer is answered `431`.
const MAX_LINE: u64 = 8 * 1024;
const MAX_HEADERS: usize = 100;

/// How long a client may take to send its request.
const READ_WITHIN: Duration = Duration::from_secs(10);

/// How many requests are answered at once; one more is answered `503`.
const MAX_IN_FLIGHT: usize = 512;

/// A [`WebRtcServer`](super::WebRtcServer)'s offers and answers over HTTP:
/// WHEP for players, WHIP for publishers — what a browser, OBS or
/// `ffmpeg -f whip` speak.
///
/// - `POST /<path>/whep` with an SDP offer plays `<path>`; `POST
///   /<path>/whip` publishes into it. Either answers `201 Created` with the
///   SDP answer, and the session's URL in `Location`.
/// - `DELETE` on that URL ends the session.
/// - `GET /<path>` is a page that plays `<path>` in a browser, through
///   WHEP: what to open to see that a path plays. Opened as
///   `/<path>?token=<token>`, it posts its offer with that token as
///   `Authorization: Bearer`.
/// - `OPTIONS` answers what a browser asks before it posts from another
///   origin, and every answer allows any origin.
///
/// Made with [`WebRtcServerHandle::serve_http`]; it listens on a thread of
/// its own until dropped, and answers each request on a thread of its own —
/// one waits for a server thread to answer its offer, and players arriving
/// together do not wait on one another. A request still being answered as it
/// is dropped is answered first, within the few seconds a request is given.
/// Plain HTTP: put it behind a proxy that terminates TLS for a page served
/// over HTTPS — and for any token sent to it.
///
/// With [`WebRtcServerOptions::authorize`](super::WebRtcServerOptions::authorize),
/// each offer is put to it first, with the `Authorization` header it came
/// with, and answered `401 Unauthorized` where it is refused.
pub struct WebRtcHttpEndpoint {
    local_addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WebRtcHttpEndpoint {
    pub(super) fn bind(
        addr: impl ToSocketAddrs,
        server: WebRtcServerHandle,
    ) -> Result<Self, WebRtcServerError> {
        let shown = |source| WebRtcServerError::Http {
            addr: String::new(),
            source,
        };
        let listener = TcpListener::bind(addr).map_err(shown)?;
        let local_addr = listener.local_addr().map_err(shown)?;
        listener.set_nonblocking(true).map_err(shown)?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("webrtc-http".into())
            .spawn(move || serve(&listener, &server, &stopping))
            .map_err(shown)?;
        Ok(Self {
            local_addr,
            stop,
            thread: Some(thread),
        })
    }

    /// The address it listens on.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// The URL a WHEP player posts its offer to, to play `path`.
    pub fn whep_url(&self, path: &str) -> String {
        format!("http://{}/{path}/whep", self.local_addr)
    }

    /// The URL a WHIP publisher posts its offer to, to publish into `path`.
    pub fn whip_url(&self, path: &str) -> String {
        format!("http://{}/{path}/whip", self.local_addr)
    }
}

impl Drop for WebRtcHttpEndpoint {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl std::fmt::Debug for WebRtcHttpEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebRtcHttpEndpoint")
            .field("local_addr", &self.local_addr)
            .finish()
    }
}

fn serve(listener: &TcpListener, server: &WebRtcServerHandle, stop: &AtomicBool) {
    let in_flight = Arc::new(AtomicUsize::new(0));
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, peer)) => {
                // Each request on a thread of its own: an offer waits for a
                // server thread to answer it, and a crowd of players arriving
                // at once must not wait in the listener's backlog behind one
                // another — past its length, they are refused.
                if in_flight.fetch_add(1, Ordering::AcqRel) >= MAX_IN_FLIGHT {
                    in_flight.fetch_sub(1, Ordering::AcqRel);
                    let _ = stream.set_nonblocking(false);
                    let _ = respond(stream, 503, &[("Retry-After", "1".into())], "");
                    continue;
                }
                let server = server.clone();
                let counted = Arc::clone(&in_flight);
                let spawned = thread::Builder::new()
                    .name("webrtc-http-request".into())
                    .spawn(move || {
                        if let Err(error) = answer(stream, peer, &server) {
                            pp_debug!(
                                pp_log: &server.shared.pp_log,
                                "an HTTP request from {peer} failed: {error}"
                            );
                        }
                        counted.fetch_sub(1, Ordering::AcqRel);
                    });
                if spawned.is_err() {
                    in_flight.fetch_sub(1, Ordering::AcqRel);
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) => {
                pp_warn!(pp_log: &server.shared.pp_log, "accepting an HTTP connection failed: {error}");
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

/// One request, answered.
/// What a request says ahead of its body, as far as it is answered.
#[derive(Debug, PartialEq)]
pub(crate) struct Head {
    method: String,
    target: String,
    length: usize,
    authorization: Option<String>,
}

/// Reads a request's line and headers: `None` for one past [`MAX_LINE`] a
/// line or [`MAX_HEADERS`] headers. A line read to its end however long
/// grew without bound for a client that sent one without a newline.
pub(crate) fn read_head(reader: &mut impl BufRead) -> io::Result<Option<Head>> {
    fn line(reader: &mut impl BufRead) -> io::Result<Option<String>> {
        let mut line = String::new();
        let read = Read::take(&mut *reader, MAX_LINE).read_line(&mut line)?;
        Ok((read as u64 != MAX_LINE || line.ends_with('\n')).then_some(line))
    }
    let Some(first) = line(reader)? else {
        return Ok(None);
    };
    let mut parts = first.split_whitespace();
    let mut head = Head {
        method: parts.next().unwrap_or("").to_owned(),
        target: parts.next().unwrap_or("").to_owned(),
        length: 0,
        authorization: None,
    };
    for _ in 0..=MAX_HEADERS {
        let Some(header) = line(reader)? else {
            return Ok(None);
        };
        let header = header.trim_end();
        if header.is_empty() {
            return Ok(Some(head));
        }
        if let Some((name, value)) = header.split_once(':') {
            let name = name.trim();
            if name.eq_ignore_ascii_case("content-length") {
                head.length = value.trim().parse().unwrap_or(0);
            } else if name.eq_ignore_ascii_case("authorization") {
                head.authorization = Some(value.trim().to_owned());
            }
        }
    }
    Ok(None)
}

fn answer(stream: TcpStream, peer: SocketAddr, server: &WebRtcServerHandle) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(READ_WITHIN))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let Some(Head {
        method,
        target,
        length,
        authorization,
    }) = read_head(&mut reader)?
    else {
        return respond(stream, 431, &[], "");
    };
    if length > MAX_BODY {
        return respond(stream, 413, &[], "");
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    let path = target.split('?').next().unwrap_or("").trim_matches('/');
    let access = |kind| {
        if kind == "whep" {
            Access::Play
        } else {
            Access::Publish
        }
    };
    match (method.as_str(), path.rsplit_once('/')) {
        ("OPTIONS", Some((path, kind @ ("whep" | "whip")))) => {
            // A browser's preflight carries no credentials, and is answered
            // as ever; a client asking what ICE servers to gather from before
            // it offers is told them once its credentials pass — a TURN
            // server's credentials are not for anyone who asks.
            let mut headers = vec![("Access-Control-Max-Age", "86400".to_owned())];
            if authorization.is_some() || server.shared.authorize.is_none() {
                let request = Asked {
                    path,
                    access: access(kind),
                    authorization: authorization.as_deref(),
                    peer,
                };
                if allowed(server, &request, false) {
                    headers.extend(ice_server_links(&server.shared.ice_servers));
                }
            }
            respond(stream, 204, &headers, "")
        }
        ("OPTIONS", _) => respond(
            stream,
            204,
            &[("Access-Control-Max-Age", "86400".into())],
            "",
        ),
        ("POST", Some((path, kind @ ("whep" | "whip")))) => {
            let request = Asked {
                path,
                access: access(kind),
                authorization: authorization.as_deref(),
                peer,
            };
            if !allowed(server, &request, true) {
                return respond(stream, 401, &[("WWW-Authenticate", "Bearer".into())], "");
            }
            let offer = String::from_utf8_lossy(&body);
            let opened = if kind == "whep" {
                server.play(path, &offer)
            } else {
                server.publish(path, &offer)
            };
            match opened {
                Ok((session, sdp)) => {
                    let mut headers = vec![
                        ("Content-Type", "application/sdp".to_owned()),
                        ("Location", format!("/session/{session}")),
                    ];
                    headers.extend(ice_server_links(&server.shared.ice_servers));
                    respond(stream, 201, &headers, &sdp)
                }
                Err(error) => {
                    let status = match error {
                        WebRtcServerError::NoSuchPath(_) => 404,
                        WebRtcServerError::NotPublishable(_) => 409,
                        WebRtcServerError::NothingToPlay(_) => 404,
                        WebRtcServerError::InvalidOffer(_) => 400,
                        _ => 503,
                    };
                    respond(stream, status, &[], &error.to_string())
                }
            }
        }
        ("GET", _) if !path.is_empty() && super::valid_path(path) => respond(
            stream,
            200,
            &[("Content-Type", "text/html; charset=utf-8".into())],
            &PAGE.replace("{path}", path).replace(
                "{ice_servers}",
                &ice_servers_json(&server.shared.ice_servers),
            ),
        ),
        ("DELETE", Some(("session", id))) => match id.parse::<SessionId>() {
            Ok(session) if server.close(session) => respond(stream, 200, &[], ""),
            _ => respond(stream, 404, &[], ""),
        },
        _ => respond(stream, 404, &[], ""),
    }
}

/// What a request asks, for the server's `Authorize`.
struct Asked<'a> {
    path: &'a str,
    access: Access,
    authorization: Option<&'a str>,
    peer: SocketAddr,
}

/// Whether the server's `Authorize`, where it has one, lets `request` in;
/// a refusal logged where `log` says so — not a preflight's.
fn allowed(server: &WebRtcServerHandle, request: &Asked<'_>, log: bool) -> bool {
    let Some(authorize) = &server.shared.authorize else {
        return true;
    };
    let asked = AccessRequest {
        path: request.path.to_owned(),
        access: request.access,
        credentials: Credentials::from_header(request.authorization),
        peer: request.peer,
    };
    let allowed = authorize.allows(&asked);
    if !allowed && log {
        pp_info!(
            pp_log: &server.shared.pp_log,
            "refused {:?} of {} to {} presenting {:?}",
            asked.access,
            asked.path,
            asked.peer,
            asked.credentials
        );
    }
    allowed
}

/// `servers` as WHIP and WHEP tell them (RFC 9725): a `Link` header for each
/// URL, with its credentials where it has them.
fn ice_server_links(servers: &[IceServer]) -> Vec<(&'static str, String)> {
    let mut links = Vec::new();
    for server in servers {
        for url in &server.urls {
            let mut link = format!("<{}>; rel=\"ice-server\"", url.replace(['<', '>'], ""));
            if let Some(username) = &server.username {
                link.push_str(&format!("; username=\"{}\"", quoted(username)));
            }
            if let Some(credential) = &server.credential {
                link.push_str(&format!(
                    "; credential=\"{}\"; credential-type=\"password\"",
                    quoted(credential)
                ));
            }
            links.push(("Link", link));
        }
    }
    links
}

/// `text` inside an HTTP quoted string.
fn quoted(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control())
        .flat_map(|c| match c {
            '"' | '\\' => vec!['\\', c],
            c => vec![c],
        })
        .collect()
}

/// `servers` as the page's `RTCPeerConnection` takes them: a JSON array,
/// written so that nothing in it can end the page's script.
fn ice_servers_json(servers: &[IceServer]) -> String {
    let string = |text: &str| {
        let mut out = String::from("\"");
        for c in text.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '<' => out.push_str("\\u003c"),
                '>' => out.push_str("\\u003e"),
                c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    };
    let entries: Vec<String> = servers
        .iter()
        .map(|server| {
            let urls: Vec<String> = server.urls.iter().map(|url| string(url)).collect();
            let mut entry = format!("{{\"urls\":[{}]", urls.join(","));
            if let Some(username) = &server.username {
                entry.push_str(&format!(",\"username\":{}", string(username)));
            }
            if let Some(credential) = &server.credential {
                entry.push_str(&format!(",\"credential\":{}", string(credential)));
            }
            entry.push('}');
            entry
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// The page `GET /<path>` answers with: a video playing `<path>` through
/// WHEP, with what it is doing under it.
const PAGE: &str = r#"<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{path}</title>
<style>
  html, body { margin: 0; height: 100%; background: #000; color: #ccc; font: 14px sans-serif; }
  video { width: 100%; height: calc(100% - 2em); object-fit: contain; background: #000; }
  p { margin: 0; padding: 0.5em; height: 1em; }
</style>
</head>
<body>
<video id="video" autoplay muted playsinline controls></video>
<p id="state">connecting</p>
<script>
const state = document.getElementById("state");
const video = document.getElementById("video");
async function play() {
  const pc = new RTCPeerConnection({ iceServers: {ice_servers} });
  pc.addTransceiver("video", { direction: "recvonly" });
  pc.addTransceiver("audio", { direction: "recvonly" });
  pc.ontrack = (event) => {
    if (!video.srcObject) video.srcObject = new MediaStream();
    video.srcObject.addTrack(event.track);
  };
  pc.onconnectionstatechange = () => { state.textContent = pc.connectionState; };
  await pc.setLocalDescription(await pc.createOffer());
  await new Promise((done) => {
    if (pc.iceGatheringState === "complete") return done();
    pc.onicegatheringstatechange = () => { if (pc.iceGatheringState === "complete") done(); };
    setTimeout(done, 2000);
  });
  const headers = { "Content-Type": "application/sdp" };
  const token = new URLSearchParams(location.search).get("token");
  if (token) headers["Authorization"] = "Bearer " + token;
  const answer = await fetch("/{path}/whep", {
    method: "POST",
    headers,
    body: pc.localDescription.sdp,
  });
  if (answer.status !== 201) {
    state.textContent = "refused: " + answer.status + " " + (await answer.text());
    return;
  }
  await pc.setRemoteDescription({ type: "answer", sdp: await answer.text() });
}
play().catch((error) => { state.textContent = "failed: " + error; });
</script>
</body>
</html>
"#;

fn respond(
    mut stream: TcpStream,
    status: u16,
    headers: &[(&str, String)],
    body: &str,
) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Service Unavailable",
    };
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: POST, DELETE, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type, Authorization\r\n\
         Access-Control-Expose-Headers: Location, Link\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(request: &[u8]) -> Option<Head> {
        read_head(&mut &request[..]).unwrap()
    }

    #[test]
    fn a_request_s_head_is_read_up_to_its_body() {
        let mut request =
            b"POST /cam/whep HTTP/1.1\r\nContent-Length: 4\r\nAuthorization: Bearer t\r\n\r\n"
                .to_vec();
        request.extend_from_slice(b"v=0\n");
        let mut reader = &request[..];
        let read = read_head(&mut reader).unwrap().expect("a whole head");
        assert_eq!(
            read,
            Head {
                method: "POST".into(),
                target: "/cam/whep".into(),
                length: 4,
                authorization: Some("Bearer t".into()),
            }
        );
        assert_eq!(reader, b"v=0\n", "the body is left to read");
    }

    /// A line without end, or headers without end, is refused once past
    /// what may be: read to its end, a line grew without bound.
    #[test]
    fn a_head_past_its_limits_is_refused() {
        let mut long = b"POST /".to_vec();
        long.extend(std::iter::repeat_n(b'a', MAX_LINE as usize * 4));
        assert_eq!(head(&long), None, "a request line without end");
        let mut header = b"POST /cam/whip HTTP/1.1\r\nX: ".to_vec();
        header.extend(std::iter::repeat_n(b'b', MAX_LINE as usize));
        header.extend_from_slice(b"\r\n\r\n");
        assert_eq!(head(&header), None, "a header line too long");
        let many = format!(
            "POST /cam/whip HTTP/1.1\r\n{}\r\n",
            "X: y\r\n".repeat(MAX_HEADERS + 1)
        );
        assert_eq!(head(many.as_bytes()), None, "too many headers");
        let enough = format!(
            "POST /cam/whip HTTP/1.1\r\n{}\r\n",
            "X: y\r\n".repeat(MAX_HEADERS)
        );
        assert!(head(enough.as_bytes()).is_some(), "as many as may be");
    }
}
