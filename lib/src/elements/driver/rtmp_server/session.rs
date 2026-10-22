//! One RTMP connection: its handshake, its commands, and what it publishes.

use std::io::{self, BufReader, BufWriter};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::elements::driver::access::{Access, AccessRequest, Credentials};
use crate::pp_log::{pp_debug, pp_info, pp_warn};

use super::Shared;
use super::amf::{self, Amf};
use super::chunk::{self, ChunkReader, ChunkWriter, Message};
use super::flv;
use super::ingest::{Ingest, Medium, Stopped};

/// How long writing an answer may take.
const WRITE_WITHIN: Duration = Duration::from_secs(10);

/// What the server tells a client to acknowledge after, and to let it send.
const WINDOW: u32 = 2_500_000;

/// The chunk streams this server answers on: commands, and a stream's
/// status.
const COMMANDS: u8 = 3;
const STATUS: u8 = 5;

/// Why a connection ends.
enum Ended {
    Io(io::Error),
    Refused(&'static str),
    Broadcast(String),
}

impl From<io::Error> for Ended {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// A publication under way: its path, the claim on it, and the ingest.
struct Publishing {
    path: String,
    ingest: Ingest,
    _claim: Claim,
}

/// Holds a path for one publisher, and counts it, until dropped.
struct Claim {
    shared: Arc<Shared>,
    path: String,
}

impl Claim {
    fn take(shared: &Arc<Shared>, path: &str) -> Option<Self> {
        let mut publishing = shared.publishing.lock().unwrap_or_else(|p| p.into_inner());
        if !publishing.insert(path.to_owned()) {
            return None;
        }
        shared.publishers.fetch_add(1, Ordering::AcqRel);
        Some(Self {
            shared: Arc::clone(shared),
            path: path.to_owned(),
        })
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.shared
            .publishing
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&self.path);
        self.shared.publishers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One connection's state.
struct Connection<'a> {
    shared: &'a Arc<Shared>,
    peer: SocketAddr,
    writer: ChunkWriter<BufWriter<TcpStream>>,
    /// The application `connect` named, and the query it came with.
    app: Option<(String, Option<String>)>,
    publishing: Option<Publishing>,
    /// The window the client asked to be acknowledged after, and where the
    /// last acknowledgement was.
    window: Option<u32>,
    acknowledged: u64,
}

/// Serves one connection until it closes, its publisher goes quiet, or the
/// server stops.
pub(super) fn serve(stream: TcpStream, peer: SocketAddr, shared: &Arc<Shared>) {
    match run(stream, peer, shared) {
        Ok(()) => {}
        Err(Ended::Io(error)) => {
            pp_debug!(pp_log: &shared.pp_log, "connection from {peer} closed: {error}");
        }
        Err(Ended::Refused(why)) => {
            pp_info!(pp_log: &shared.pp_log, "refused {peer}: {why}");
        }
        Err(Ended::Broadcast(why)) => {
            pp_warn!(pp_log: &shared.pp_log, "the publication of {peer} ended: {why}");
        }
    }
}

fn run(stream: TcpStream, peer: SocketAddr, shared: &Arc<Shared>) -> Result<(), Ended> {
    // Accepted from a listener that does not block, it might not either.
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(shared.publisher_timeout))?;
    stream.set_write_timeout(Some(WRITE_WITHIN))?;
    let mut handshaking = stream.try_clone()?;
    chunk::handshake(&mut handshaking)?;
    let mut reader = ChunkReader::new(BufReader::with_capacity(64 * 1024, stream.try_clone()?));
    let mut connection = Connection {
        shared,
        peer,
        writer: ChunkWriter::new(BufWriter::new(stream)),
        app: None,
        publishing: None,
        window: None,
        acknowledged: 0,
    };
    loop {
        let message = reader.next()?;
        connection.acknowledge(reader.bytes_read)?;
        connection.message(message)?;
    }
}

impl Connection<'_> {
    /// Acknowledges what has been read, as often as the client asked.
    fn acknowledge(&mut self, read: u64) -> io::Result<()> {
        let Some(window) = self.window else {
            return Ok(());
        };
        if read - self.acknowledged >= u64::from(window) {
            self.acknowledged = read;
            self.writer
                .write(2, chunk::ACKNOWLEDGEMENT, 0, &(read as u32).to_be_bytes())?;
        }
        Ok(())
    }

    fn message(&mut self, message: Message) -> Result<(), Ended> {
        match message.kind {
            chunk::WINDOW_ACK_SIZE if message.payload.len() >= 4 => {
                let window = u32::from_be_bytes(message.payload[..4].try_into().unwrap());
                self.window = (window > 0).then_some(window);
            }
            chunk::COMMAND_AMF0 => {
                let Some(values) = amf::decode_all(&message.payload) else {
                    return Err(Ended::Refused("a command that does not read as AMF0"));
                };
                self.command(&values, message.stream)?;
            }
            chunk::DATA_AMF0 => {
                if let (Some(publishing), Some(values)) =
                    (&mut self.publishing, amf::decode_all(&message.payload))
                {
                    // `@setDataFrame("onMetaData", {...})`, or `onMetaData`
                    // alone.
                    let metadata = values
                        .iter()
                        .skip_while(|value| value.as_str() != Some("onMetaData"))
                        .nth(1);
                    if let Some(metadata) = metadata {
                        publishing.ingest.metadata(metadata);
                    }
                }
            }
            chunk::VIDEO | chunk::AUDIO => {
                let Some(publishing) = &mut self.publishing else {
                    return Ok(());
                };
                let (medium, tag) = if message.kind == chunk::VIDEO {
                    (Medium::Video, flv::video(&message.payload))
                } else {
                    (Medium::Audio, flv::audio(&message.payload))
                };
                match tag {
                    Ok(flv::Tag::Ignored(what)) => {
                        pp_debug!(pp_log: &self.shared.pp_log, "{} sent {what}; passed over", self.peer);
                    }
                    Ok(tag) => {
                        if let Err(Stopped::Broadcast(error)) =
                            publishing.ingest.take(medium, tag, message.timestamp)
                        {
                            return Err(Ended::Broadcast(error.to_string()));
                        }
                    }
                    Err(what) => {
                        pp_debug!(pp_log: &self.shared.pp_log, "{} sent {what}; passed over", self.peer);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn command(&mut self, values: &[Amf], stream: u32) -> Result<(), Ended> {
        let name = values.first().and_then(Amf::as_str).unwrap_or("");
        let transaction = values.get(1).and_then(Amf::as_number).unwrap_or(0.0);
        pp_debug!(pp_log: &self.shared.pp_log, "{} asked {name}", self.peer);
        match name {
            "connect" => {
                let app = values
                    .get(2)
                    .and_then(|object| object.get("app"))
                    .and_then(Amf::as_str)
                    .map(|app| app.trim_matches('/'))
                    .filter(|app| !app.is_empty())
                    .ok_or(Ended::Refused("connect named no application"))?;
                let (app, query) = match app.split_once('?') {
                    Some((app, query)) => (app.to_owned(), Some(query.to_owned())),
                    None => (app.to_owned(), None),
                };
                self.app = Some((app, query));
                self.writer
                    .write(2, chunk::WINDOW_ACK_SIZE, 0, &WINDOW.to_be_bytes())?;
                let mut bandwidth = WINDOW.to_be_bytes().to_vec();
                bandwidth.push(2);
                self.writer
                    .write(2, chunk::SET_PEER_BANDWIDTH, 0, &bandwidth)?;
                self.writer.announce_chunk_size()?;
                self.answer(
                    transaction,
                    &[
                        amf::object(&[
                            ("fmsVer", Amf::String("FMS/3,0,1,123".into())),
                            ("capabilities", Amf::Number(31.0)),
                        ]),
                        status("status", "NetConnection.Connect.Success", "Connected."),
                    ],
                )?;
            }
            "createStream" => self.answer(transaction, &[Amf::Null, Amf::Number(1.0)])?,
            "releaseStream" | "FCPublish" | "_checkbw" => {
                self.answer(transaction, &[Amf::Null, Amf::Undefined])?;
            }
            "publish" => self.publish(values, stream)?,
            "FCUnpublish" | "deleteStream" | "closeStream" => {
                if let Some(publishing) = self.publishing.take() {
                    pp_info!(
                        pp_log: &self.shared.pp_log,
                        "{} stopped publishing {}",
                        self.peer,
                        publishing.path
                    );
                }
            }
            "play" => {
                self.status(
                    stream,
                    status(
                        "error",
                        "NetStream.Play.Failed",
                        "This server takes publishers only.",
                    ),
                )?;
                return Err(Ended::Refused("it asked to play"));
            }
            _ => {}
        }
        Ok(())
    }

    fn publish(&mut self, values: &[Amf], stream: u32) -> Result<(), Ended> {
        let Some((app, app_query)) = self.app.clone() else {
            return Err(Ended::Refused("publish before connect"));
        };
        let name = values.get(3).and_then(Amf::as_str).unwrap_or("");
        let (name, query) = match name.split_once('?') {
            Some((name, query)) => (name, Some(query)),
            None => (name, app_query.as_deref()),
        };
        let path = format!("{app}/{}", name.trim_matches('/'));
        if self.publishing.is_some() {
            return Err(Ended::Refused("a second publish on one connection"));
        }
        if let Some(authorize) = &self.shared.authorize {
            let asked = AccessRequest {
                path: path.clone(),
                access: Access::Publish,
                credentials: credentials(query),
                peer: self.peer,
            };
            if !authorize.allows(&asked) {
                pp_info!(
                    pp_log: &self.shared.pp_log,
                    "refused Publish of {path} to {} presenting {:?}",
                    self.peer,
                    asked.credentials
                );
                self.status(
                    stream,
                    status("error", "NetStream.Publish.Unauthorized", "Not allowed."),
                )?;
                return Err(Ended::Refused("not authorized"));
            }
        }
        let Some(broadcast) = self.shared.path(&path) else {
            self.status(
                stream,
                status("error", "NetStream.Publish.BadName", "No such path."),
            )?;
            return Err(Ended::Refused("no such path"));
        };
        let Some(claim) = Claim::take(self.shared, &path) else {
            self.status(
                stream,
                status("error", "NetStream.Publish.BadName", "Already publishing."),
            )?;
            return Err(Ended::Refused("another publisher holds its path"));
        };
        pp_info!(
            pp_log: &self.shared.pp_log,
            "{} publishes {path} into broadcast={}",
            self.peer,
            broadcast.name()
        );
        self.publishing = Some(Publishing {
            ingest: Ingest::new(broadcast),
            path,
            _claim: claim,
        });
        self.status(
            stream,
            status("status", "NetStream.Publish.Start", "Publishing."),
        )?;
        Ok(())
    }

    /// `_result` for `transaction`.
    fn answer(&mut self, transaction: f64, values: &[Amf]) -> io::Result<()> {
        let mut all = vec![Amf::String("_result".into()), Amf::Number(transaction)];
        all.extend_from_slice(values);
        self.writer
            .write(COMMANDS, chunk::COMMAND_AMF0, 0, &amf::encode_all(&all))
    }

    /// `onStatus` on message stream `stream`.
    fn status(&mut self, stream: u32, info: Amf) -> io::Result<()> {
        let all = [
            Amf::String("onStatus".into()),
            Amf::Number(0.0),
            Amf::Null,
            info,
        ];
        self.writer
            .write(STATUS, chunk::COMMAND_AMF0, stream, &amf::encode_all(&all))
    }
}

fn status(level: &str, code: &str, description: &str) -> Amf {
    amf::object(&[
        ("level", Amf::String(level.into())),
        ("code", Amf::String(code.into())),
        ("description", Amf::String(description.into())),
    ])
}

/// What a stream name's query presents: `user` and `pass` as Basic, or
/// `token` as Bearer.
fn credentials(query: Option<&str>) -> Credentials {
    let Some(query) = query else {
        return Credentials::None;
    };
    let value = |key: &str| {
        query
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(name, _)| *name == key)
            .map(|(_, value)| decode_percent(value))
    };
    if let (Some(user), Some(password)) = (value("user"), value("pass")) {
        return Credentials::Basic { user, password };
    }
    match value("token") {
        Some(token) => Credentials::Bearer(token),
        None => Credentials::None,
    }
}

/// `%XX` escapes and `+` as a URL query writes them.
fn decode_percent(text: &str) -> String {
    let hex = |byte: u8| (byte as char).to_digit(16);
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'%' if index + 2 < bytes.len() => {
                if let (Some(high), Some(low)) = (hex(bytes[index + 1]), hex(bytes[index + 2])) {
                    out.push((high * 16 + low) as u8);
                    index += 3;
                    continue;
                }
                out.push(b'%');
            }
            b'+' => out.push(b' '),
            byte => out.push(byte),
        }
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_says_the_credentials() {
        assert_eq!(
            credentials(Some("user=ana&pass=s%3Acret+x")),
            Credentials::Basic {
                user: "ana".into(),
                password: "s:cret x".into()
            }
        );
        assert_eq!(
            credentials(Some("token=abc")),
            Credentials::Bearer("abc".into())
        );
        assert_eq!(credentials(Some("other=1")), Credentials::None);
        assert_eq!(credentials(None), Credentials::None);
        assert_eq!(decode_percent("100%"), "100%");
        assert_eq!(decode_percent("%4"), "%4");
        assert_eq!(decode_percent("%e2%82%acé%zz"), "€é%zz");
    }
}
