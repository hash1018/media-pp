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
    pub(crate) fn session(&self) -> Option<&str> {
        self.header("Session")
            .map(|value| value.split(';').next().unwrap_or("").trim())
    }
}

/// What a connection reads: a request, or a packet sent interleaved on the
/// connection — RTCP from a player, RTP from a publisher.
#[derive(Debug)]
pub(crate) enum Incoming {
    Request(Request),
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
        if buf.len() > MAX_HEADERS {
            return Err(Malformed("header block too long"));
        }
        return Ok(None);
    };
    let head = std::str::from_utf8(&buf[..end]).map_err(|_| Malformed("headers are not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (Some(method), Some(url), Some(version)) = (first.next(), first.next(), first.next())
    else {
        return Err(Malformed("request line"));
    };
    if !version.starts_with("RTSP/1.") || method.is_empty() {
        return Err(Malformed("not an RTSP/1.x request"));
    }
    let mut headers = Vec::new();
    for line in lines {
        let Some((key, value)) = line.split_once(':') else {
            return Err(Malformed("header line"));
        };
        headers.push((key.trim().to_owned(), value.trim().to_owned()));
    }
    let mut request = Request {
        method: method.to_owned(),
        url: url.to_owned(),
        headers,
        body: Vec::new(),
    };
    let length = match request.header("Content-Length") {
        Some(length) => length
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
    request.body = buf[start..start + length].to_vec();
    Ok(Some((Incoming::Request(request), skipped + start + length)))
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

    pub(crate) fn status(&self) -> u16 {
        self.status
    }

    pub(crate) fn header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

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
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Target {
    pub(crate) path: String,
    pub(crate) track: Option<usize>,
}

/// What `url` names: `rtsp://host:port/<path>[/streamid=<n>][?query]`.
/// `None` for one that is not an `rtsp://` URL with a path.
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
