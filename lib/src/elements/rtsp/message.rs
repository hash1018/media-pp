//! RTSP 1.0 on the wire: requests read, responses written, and the headers
//! a server has to understand.

use std::fmt::Write as _;

/// The most a request's header block may take, and its body.
const MAX_HEADERS: usize = 64 * 1024;
const MAX_BODY: usize = 1024 * 1024;

/// One request.
#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) url: String,
    headers: Vec<(String, String)>,
    pub(crate) body: Vec<u8>,
}

impl Request {
    /// The value of header `name`, matched without regard to case.
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// The session it names, without the parameters after it.
    #[cfg(feature = "rtsp-server")]
    pub(crate) fn session(&self) -> Option<&str> {
        self.header("Session")
            .map(|value| value.split(';').next().unwrap_or("").trim())
    }
}

/// One response, as a client reads it.
#[derive(Debug)]
pub(crate) struct Reply {
    pub(crate) status: u16,
    headers: Vec<(String, String)>,
    pub(crate) body: Vec<u8>,
}

impl Reply {
    /// The value of header `name`, matched without regard to case.
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// Every value of header `name` — a `401` may offer several schemes,
    /// each in a `WWW-Authenticate` of its own.
    pub(crate) fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.headers
            .iter()
            .filter(move |(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// What a connection reads: a request, a response, or a packet sent
/// interleaved on the connection — RTCP from a player, RTP from a
/// publisher or a server.
#[derive(Debug)]
pub(crate) enum Incoming {
    Request(Request),
    Reply(Reply),
    Interleaved { channel: u8, data: Vec<u8> },
}

/// Why what was read is not RTSP: the connection is closed after it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Malformed(pub(crate) &'static str);

/// The first message `buf` holds, and how many bytes it took — `None`
/// where it holds only part of one so far.
pub(crate) fn parse(buf: &[u8]) -> Result<Option<(Incoming, usize)>, Malformed> {
    if buf.first() == Some(&b'$') {
        if buf.len() < 4 {
            return Ok(None);
        }
        let length = usize::from(u16::from_be_bytes([buf[2], buf[3]]));
        if buf.len() < 4 + length {
            return Ok(None);
        }
        return Ok(Some((
            Incoming::Interleaved {
                channel: buf[1],
                data: buf[4..4 + length].to_vec(),
            },
            4 + length,
        )));
    }
    // Blank lines between messages are allowed.
    let skipped = buf
        .iter()
        .take_while(|byte| matches!(byte, b'\r' | b'\n'))
        .count();
    let buf = &buf[skipped..];
    let Some(end) = find(buf, b"\r\n\r\n") else {
        // The blank lines counted too: skipped and never taken, a peer
        // sending nothing else grew what was held without end.
        if skipped + buf.len() > MAX_HEADERS {
            return Err(Malformed("header block too long"));
        }
        return Ok(None);
    };
    let head = std::str::from_utf8(&buf[..end]).map_err(|_| Malformed("headers are not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut headers = Vec::new();
    for line in lines {
        let Some((key, value)) = line.split_once(':') else {
            return Err(Malformed("header line"));
        };
        headers.push((key.trim().to_owned(), value.trim().to_owned()));
    }
    let length = match headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("Content-Length"))
    {
        Some((_, length)) => length
            .parse::<usize>()
            .map_err(|_| Malformed("Content-Length"))?,
        None => 0,
    };
    if length > MAX_BODY {
        return Err(Malformed("body too long"));
    }
    let start = end + 4;
    if buf.len() < start + length {
        return Ok(None);
    }
    let body = buf[start..start + length].to_vec();
    let taken = skipped + start + length;

    // `RTSP/1.0 200 OK`: a response.
    if let Some(rest) = first.strip_prefix("RTSP/1.") {
        let status = rest
            .split(' ')
            .nth(1)
            .and_then(|status| status.parse().ok())
            .ok_or(Malformed("status line"))?;
        return Ok(Some((
            Incoming::Reply(Reply {
                status,
                headers,
                body,
            }),
            taken,
        )));
    }
    let mut first = first.split(' ');
    let (Some(method), Some(url), Some(version)) = (first.next(), first.next(), first.next())
    else {
        return Err(Malformed("request line"));
    };
    if !version.starts_with("RTSP/1.") || method.is_empty() {
        return Err(Malformed("not an RTSP/1.x request"));
    }
    let request = Request {
        method: method.to_owned(),
        url: url.to_owned(),
        headers,
        body,
    };
    Ok(Some((Incoming::Request(request), taken)))
}

/// A request to send: `method` on `url`, the `cseq`-th of its connection,
/// with `headers` and `body`.
pub(crate) fn request(
    method: &str,
    url: &str,
    cseq: u32,
    headers: &[(&str, &str)],
    body: &[u8],
) -> Vec<u8> {
    let mut head = format!("{method} {url} RTSP/1.0\r\nCSeq: {cseq}\r\nUser-Agent: media-pp\r\n");
    for (name, value) in headers {
        let _ = write!(head, "{name}: {value}\r\n");
    }
    if !body.is_empty() {
        let _ = write!(head, "Content-Length: {}\r\n", body.len());
    }
    head.push_str("\r\n");
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

/// An `rtsp://` URL taken apart for a client: where to connect, the URL to
/// ask for — its credentials taken out, which go in an `Authorization`
/// instead — and the credentials.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Address {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) url: String,
    pub(crate) credentials: Option<(String, String)>,
}

/// `url` as a client connects to it — `rtsp://[user:password@]host[:port]/…`,
/// port 554 by default, `%XX` escapes in the credentials decoded; `None`
/// for what is not such a URL.
pub(crate) fn address(url: &str) -> Option<Address> {
    let rest = url
        .strip_prefix("rtsp://")
        .or_else(|| url.strip_prefix("RTSP://"))?;
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, path) = rest.split_at(authority_end);
    let (credentials, host_port) = match authority.rfind('@') {
        Some(at) => {
            let (user, password) = authority[..at]
                .split_once(':')
                .unwrap_or((&authority[..at], ""));
            (
                Some((decode(user)?, decode(password)?)),
                &authority[at + 1..],
            )
        }
        None => (None, authority),
    };
    let (host, port) = if let Some(bracketed) = host_port.strip_prefix('[') {
        let (host, after) = bracketed.split_once(']')?;
        let port = match after.strip_prefix(':') {
            Some(port) => port.parse().ok()?,
            None if after.is_empty() => 554,
            None => return None,
        };
        (host.to_owned(), port)
    } else {
        match host_port.rsplit_once(':') {
            Some((host, port)) => (host.to_owned(), port.parse().ok()?),
            None => (host_port.to_owned(), 554),
        }
    };
    if host.is_empty() {
        return None;
    }
    Some(Address {
        url: format!("rtsp://{host_port}{path}"),
        host,
        port,
        credentials,
    })
}

/// `text` with its `%XX` escapes decoded.
fn decode(text: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(text.len());
    let mut rest = text.bytes();
    while let Some(byte) = rest.next() {
        if byte == b'%' {
            let high = rest.next().and_then(|digit| (digit as char).to_digit(16))?;
            let low = rest.next().and_then(|digit| (digit as char).to_digit(16))?;
            bytes.push((high * 16 + low) as u8);
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).ok()
}

/// A track's or a session's control URL — what a description's
/// `a=control` says — made whole against the description's base: an
/// absolute URL as it is, `*` the base itself, anything else after it.
pub(crate) fn resolve(base: &str, control: &str) -> String {
    let control = control.trim();
    if control.is_empty() || control == "*" {
        return base.to_owned();
    }
    if control.contains("://") {
        return control.to_owned();
    }
    let base = without_query(base);
    if base.ends_with('/') {
        format!("{base}{control}")
    } else {
        format!("{base}/{control}")
    }
}

/// What a server's `SETUP` response says it delivers a track by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Delivery {
    /// The channels it interleaves the track's RTP and RTCP on.
    pub(crate) interleaved: Option<(u8, u8)>,
    /// The ports it sends the track's RTP and RTCP from.
    pub(crate) server_ports: Option<(u16, u16)>,
}

/// A server's `Transport` response header read.
pub(crate) fn delivery(header: &str) -> Delivery {
    let mut delivery = Delivery::default();
    let spec = header.split(',').next().unwrap_or("");
    for part in spec.split(';').map(str::trim) {
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        match key.to_ascii_lowercase().as_str() {
            "interleaved" => {
                delivery.interleaved = pair(value)
                    .and_then(|(a, b)| Some((u8::try_from(a).ok()?, u8::try_from(b).ok()?)));
            }
            "server_port" => delivery.server_ports = pair(value),
            _ => {}
        }
    }
    delivery
}

/// A `Session` response header read: the session's id, and how long the
/// server keeps it without a request — sixty seconds where it does not
/// say, as the standard has it.
pub(crate) fn session(header: &str) -> (String, u64) {
    let mut parts = header.split(';').map(str::trim);
    let id = parts.next().unwrap_or("").to_owned();
    let timeout = parts
        .filter_map(|part| part.split_once('='))
        .find(|(key, _)| key.trim().eq_ignore_ascii_case("timeout"))
        .and_then(|(_, value)| value.trim().parse().ok())
        .filter(|timeout| *timeout > 0)
        .unwrap_or(60);
    (id, timeout)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// One response, built up and then written with [`Self::to_bytes`].
pub(crate) struct Response {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

impl Response {
    pub(crate) fn new(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    #[cfg(feature = "rtsp-server")]
    pub(crate) fn status(&self) -> u16 {
        self.status
    }

    pub(crate) fn header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    #[cfg(feature = "rtsp-server")]
    pub(crate) fn body(mut self, content_type: &'static str, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self.header("Content-Type", content_type)
    }

    /// The response to the request whose `CSeq` was `cseq`.
    pub(crate) fn to_bytes(&self, cseq: Option<&str>) -> Vec<u8> {
        let mut head = format!("RTSP/1.0 {} {}\r\n", self.status, reason(self.status));
        if let Some(cseq) = cseq {
            let _ = write!(head, "CSeq: {cseq}\r\n");
        }
        head.push_str("Server: media-pp\r\n");
        for (name, value) in &self.headers {
            let _ = write!(head, "{name}: {value}\r\n");
        }
        if !self.body.is_empty() {
            let _ = write!(head, "Content-Length: {}\r\n", self.body.len());
        }
        head.push_str("\r\n");
        let mut bytes = head.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        415 => "Unsupported Media Type",
        453 => "Not Enough Bandwidth",
        454 => "Session Not Found",
        455 => "Method Not Valid in This State",
        459 => "Aggregate Operation Not Allowed",
        461 => "Unsupported Transport",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

/// A request URL taken apart: the path it names, and the track where it
/// names one by the control URL a description gave it.
#[cfg(feature = "rtsp-server")]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Target {
    pub(crate) path: String,
    pub(crate) track: Option<usize>,
}

/// What `url` names: `rtsp://host:port/<path>[/streamid=<n>][?query]`.
/// `None` for one that is not an `rtsp://` URL with a path.
#[cfg(feature = "rtsp-server")]
pub(crate) fn target(url: &str) -> Option<Target> {
    let rest = url
        .strip_prefix("rtsp://")
        .or_else(|| url.strip_prefix("rtsps://"))
        .or_else(|| url.strip_prefix("RTSP://"))?;
    let path = rest.find('/').map_or("", |slash| &rest[slash + 1..]);
    let path = path.split(['?', '#']).next().unwrap_or("");
    let path = path.trim_end_matches('/');
    let (path, track) = match path.rsplit_once('/') {
        Some((head, last)) if last.starts_with("streamid=") => {
            (head, last["streamid=".len()..].parse().ok())
        }
        _ if path.starts_with("streamid=") => ("", path["streamid=".len()..].parse().ok()),
        _ => (path, None),
    };
    Some(Target {
        path: path.trim_matches('/').to_owned(),
        track,
    })
}

/// `url` without its query or fragment: what a description's
/// `Content-Base` is, so that a track's control URL a player makes of it
/// keeps the track where a parser finds it.
pub(crate) fn without_query(url: &str) -> &str {
    url.split(['?', '#']).next().unwrap_or(url)
}

/// The `token` `url`'s query carries, `%XX` escapes and `+` decoded: what
/// a player that sends no `Authorization` presents — `rtsp://h/cam?token=…`
/// — as an RTMP publisher does in its stream name.
#[cfg(feature = "rtsp-server")]
pub(crate) fn query_token(url: &str) -> Option<String> {
    let query = url.split_once('?')?.1.split('#').next()?;
    let value = query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(key, _)| *key == "token")?
        .1;
    let mut bytes = Vec::with_capacity(value.len());
    let mut rest = value.bytes();
    while let Some(byte) = rest.next() {
        match byte {
            b'+' => bytes.push(b' '),
            b'%' => {
                let high = rest.next().and_then(|digit| (digit as char).to_digit(16))?;
                let low = rest.next().and_then(|digit| (digit as char).to_digit(16))?;
                bytes.push((high * 16 + low) as u8);
            }
            _ => bytes.push(byte),
        }
    }
    String::from_utf8(bytes)
        .ok()
        .filter(|token| !token.is_empty())
}

/// How a track asked to be delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transport {
    /// Interleaved on the RTSP connection, on these channels — the server
    /// picks where the request named none.
    Tcp { channels: Option<(u8, u8)> },
    /// To these ports of the player's address.
    Udp { rtp: u16, rtcp: u16 },
}

/// The first transport of a `Transport` header this server can deliver,
/// with whether it asks to record rather than play.
pub(crate) fn transport(header: &str) -> Option<(Transport, bool)> {
    header.split(',').find_map(|spec| {
        let mut parts = spec.split(';').map(str::trim);
        let protocol = parts.next()?.to_ascii_uppercase();
        let tcp = match protocol.as_str() {
            "RTP/AVP/TCP" => true,
            "RTP/AVP" | "RTP/AVP/UDP" => false,
            _ => return None,
        };
        let mut channels = None;
        let mut ports = None;
        let mut record = false;
        for part in parts {
            let (key, value) = part.split_once('=').unwrap_or((part, ""));
            match key.to_ascii_lowercase().as_str() {
                "multicast" => return None,
                "interleaved" => channels = Some(pair(value)?),
                "client_port" => ports = Some(pair(value)?),
                "mode" => record = value.trim_matches('"').eq_ignore_ascii_case("record"),
                _ => {}
            }
        }
        let transport = if tcp {
            Transport::Tcp {
                channels: channels.and_then(|(a, b): (u16, u16)| {
                    Some((u8::try_from(a).ok()?, u8::try_from(b).ok()?))
                }),
            }
        } else {
            let (rtp, rtcp) = ports?;
            Transport::Udp { rtp, rtcp }
        };
        Some((transport, record))
    })
}

/// `a-b`, or `a` alone meaning `a-(a+1)`.
fn pair(value: &str) -> Option<(u16, u16)> {
    match value.split_once('-') {
        Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
        None => {
            let a: u16 = value.trim().parse().ok()?;
            Some((a, a.checked_add(1)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_read_once_it_is_whole() {
        let text = b"DESCRIBE rtsp://127.0.0.1:8554/cam RTSP/1.0\r\nCSeq: 2\r\nAccept: application/sdp\r\n\r\n";
        for cut in 0..text.len() {
            assert_eq!(
                parse(&text[..cut]).unwrap().map(|(_, n)| n),
                None,
                "at {cut}"
            );
        }
        let (Incoming::Request(request), taken) = parse(text).unwrap().unwrap() else {
            panic!("a request");
        };
        assert_eq!(taken, text.len());
        assert_eq!(request.method, "DESCRIBE");
        assert_eq!(request.url, "rtsp://127.0.0.1:8554/cam");
        assert_eq!(request.header("cseq"), Some("2"));
    }

    #[test]
    fn a_body_is_read_by_its_length() {
        let text =
            b"ANNOUNCE rtsp://h/live RTSP/1.0\r\nCSeq: 1\r\nContent-Length: 4\r\n\r\nv=0\nNEXT";
        let (Incoming::Request(request), taken) = parse(text).unwrap().unwrap() else {
            panic!("a request");
        };
        assert_eq!(request.body, b"v=0\n");
        assert_eq!(&text[taken..], b"NEXT");
    }

    #[test]
    fn an_interleaved_packet_is_read_whole() {
        let text = [b'$', 1, 0, 3, 7, 8, 9, b'O'];
        assert!(parse(&text[..5]).unwrap().is_none());
        let (Incoming::Interleaved { channel, data }, taken) = parse(&text).unwrap().unwrap()
        else {
            panic!("interleaved");
        };
        assert_eq!((channel, data, taken), (1, vec![7, 8, 9], 7));
    }

    #[test]
    fn what_is_not_rtsp_is_refused() {
        assert!(parse(b"GET / HTTP/1.1\r\n\r\n").is_err());
        assert!(parse(b"nonsense\r\n\r\n").is_err());
        assert!(parse(&vec![b'a'; MAX_HEADERS + 1]).is_err());
    }

    /// Blank lines and nothing else, past what a header block may be, are
    /// refused: skipped and never taken, they were held without end.
    #[test]
    fn blank_lines_without_end_are_refused() {
        assert!(matches!(parse(b"\r\n\r\n"), Ok(None)), "a few are allowed");
        assert!(parse(&b"\r\n".repeat(MAX_HEADERS / 2 + 1)).is_err());
        let mut late = b"\r\n".repeat(8);
        late.extend_from_slice(b"OPTIONS * RTSP/1.0\r\nCSeq: 1\r\n\r\n");
        assert!(matches!(parse(&late), Ok(Some(_))), "a request after some");
    }

    #[cfg(feature = "rtsp-server")]
    #[test]
    fn a_url_names_its_path_and_track() {
        let t = |path: &str, track| {
            Some(Target {
                path: path.into(),
                track,
            })
        };
        assert_eq!(target("rtsp://127.0.0.1:8554/cam"), t("cam", None));
        assert_eq!(target("rtsp://127.0.0.1:8554/cam/"), t("cam", None));
        assert_eq!(target("rtsp://h/a/b/streamid=1"), t("a/b", Some(1)));
        assert_eq!(target("rtsp://h/cam/streamid=0?x=1"), t("cam", Some(0)));
        assert_eq!(target("rtsp://h:1"), t("", None));
        assert_eq!(target("http://h/cam"), None);
    }

    #[cfg(feature = "rtsp-server")]
    #[test]
    fn a_url_carries_a_token_in_its_query() {
        assert_eq!(
            query_token("rtsp://h/cam?token=a.b-c_d"),
            Some("a.b-c_d".to_owned())
        );
        assert_eq!(
            query_token("rtsp://h/cam?x=1&token=a%2Bb#f"),
            Some("a+b".to_owned())
        );
        assert_eq!(query_token("rtsp://h/cam?tokens=a"), None);
        assert_eq!(query_token("rtsp://h/cam?token="), None);
        assert_eq!(query_token("rtsp://h/cam"), None);
        assert_eq!(without_query("rtsp://h/cam?token=a"), "rtsp://h/cam");
        assert_eq!(without_query("rtsp://h/cam"), "rtsp://h/cam");
    }

    #[test]
    fn a_transport_is_the_first_that_can_be_delivered() {
        assert_eq!(
            transport("RTP/AVP/TCP;unicast;interleaved=2-3"),
            Some((
                Transport::Tcp {
                    channels: Some((2, 3))
                },
                false
            ))
        );
        assert_eq!(
            transport("RTP/AVP;unicast;client_port=5000-5001"),
            Some((
                Transport::Udp {
                    rtp: 5000,
                    rtcp: 5001
                },
                false
            ))
        );
        assert_eq!(
            transport("RTP/AVP;multicast;client_port=5000-5001,RTP/AVP/TCP;unicast;mode=record"),
            Some((Transport::Tcp { channels: None }, true))
        );
        assert_eq!(transport("RTP/SAVP;unicast;client_port=1-2"), None);
        assert_eq!(
            transport("RTP/AVP;unicast"),
            None,
            "UDP needs the player's ports"
        );
    }

    #[test]
    fn a_response_is_read_with_every_header_it_repeats() {
        let text =
            b"RTSP/1.0 401 Unauthorized\r\nCSeq: 3\r\nWWW-Authenticate: Digest realm=\"a\"\r\n\
                     WWW-Authenticate: Basic realm=\"a\"\r\nContent-Length: 2\r\n\r\nhi$";
        let (Incoming::Reply(reply), taken) = parse(text).unwrap().unwrap() else {
            panic!("a reply");
        };
        assert_eq!(reply.status, 401);
        assert_eq!(reply.header("cseq"), Some("3"));
        assert_eq!(reply.all("www-authenticate").count(), 2);
        assert_eq!(reply.body, b"hi");
        assert_eq!(&text[taken..], b"$");
    }

    #[test]
    fn a_request_is_written_with_its_sequence_and_length() {
        let bytes = request(
            "SETUP",
            "rtsp://h/cam/streamid=0",
            4,
            &[("Transport", "RTP/AVP/TCP;unicast;interleaved=0-1")],
            b"",
        );
        let (Incoming::Request(parsed), _) = parse(&bytes).unwrap().unwrap() else {
            panic!("a request");
        };
        assert_eq!(parsed.method, "SETUP");
        assert_eq!(parsed.header("CSeq"), Some("4"));
        assert_eq!(
            parsed.header("Transport"),
            Some("RTP/AVP/TCP;unicast;interleaved=0-1")
        );
    }

    #[test]
    fn an_address_takes_its_credentials_out() {
        assert_eq!(
            address("rtsp://admin:p%40ss@10.0.0.1:8554/live?x=1"),
            Some(Address {
                host: "10.0.0.1".into(),
                port: 8554,
                url: "rtsp://10.0.0.1:8554/live?x=1".into(),
                credentials: Some(("admin".into(), "p@ss".into())),
            })
        );
        let plain = address("rtsp://cam.local/stream1").unwrap();
        assert_eq!((plain.host.as_str(), plain.port), ("cam.local", 554));
        assert_eq!(plain.credentials, None);
        let v6 = address("rtsp://[::1]:9000/a").unwrap();
        assert_eq!((v6.host.as_str(), v6.port), ("::1", 9000));
        assert_eq!(address("http://h/a"), None);
        assert_eq!(address("rtsp://h:notaport/a"), None);
        assert_eq!(address("rtsp:///a"), None);
    }

    #[test]
    fn a_control_url_is_made_whole_against_its_base() {
        let base = "rtsp://h/cam/";
        assert_eq!(resolve(base, "trackID=1"), "rtsp://h/cam/trackID=1");
        assert_eq!(
            resolve("rtsp://h/cam", "streamid=0"),
            "rtsp://h/cam/streamid=0"
        );
        assert_eq!(resolve(base, "*"), base);
        assert_eq!(resolve(base, "rtsp://other/t"), "rtsp://other/t");
        assert_eq!(
            resolve("rtsp://h/cam?token=a", "streamid=0"),
            "rtsp://h/cam/streamid=0"
        );
    }

    #[test]
    fn a_delivery_and_a_session_are_read() {
        assert_eq!(
            delivery("RTP/AVP/TCP;unicast;interleaved=2-3;ssrc=1234"),
            Delivery {
                interleaved: Some((2, 3)),
                server_ports: None
            }
        );
        assert_eq!(
            delivery("RTP/AVP;unicast;client_port=5000-5001;server_port=6970-6971"),
            Delivery {
                interleaved: None,
                server_ports: Some((6970, 6971))
            }
        );
        assert_eq!(session("12345678;timeout=30"), ("12345678".into(), 30));
        assert_eq!(session("abc"), ("abc".into(), 60));
    }

    #[cfg(feature = "rtsp-server")]
    #[test]
    fn a_response_says_its_length_and_sequence() {
        let bytes = Response::new(200)
            .header("Session", "abc;timeout=60")
            .body("application/sdp", "v=0\r\n")
            .to_bytes(Some("3"));
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("RTSP/1.0 200 OK\r\nCSeq: 3\r\n"), "{text}");
        assert!(text.contains("Content-Length: 5\r\n"));
        assert!(text.ends_with("\r\n\r\nv=0\r\n"));
    }
}
