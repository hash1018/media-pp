//! RTMP below its messages: the handshake, and messages cut into chunks
//! and put back together.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::io::{self, Read, Write};

/// The handshake's blocks after its version byte.
const HANDSHAKE: usize = 1536;

/// The most a message may hold: a 1080p keyframe is a few hundred kB, and
/// a length past this is a stream gone wrong or an attempt on memory.
const MAX_MESSAGE: usize = 16 * 1024 * 1024;

/// The most chunk streams one connection may have messages under way on.
const MAX_CHUNK_STREAMS: usize = 64;

/// The largest chunk this server sends.
pub(crate) const OUT_CHUNK: usize = 4096;

/// Message types.
pub(crate) const SET_CHUNK_SIZE: u8 = 1;
pub(crate) const ABORT: u8 = 2;
pub(crate) const ACKNOWLEDGEMENT: u8 = 3;
pub(crate) const WINDOW_ACK_SIZE: u8 = 5;
pub(crate) const SET_PEER_BANDWIDTH: u8 = 6;
pub(crate) const AUDIO: u8 = 8;
pub(crate) const VIDEO: u8 = 9;
pub(crate) const DATA_AMF0: u8 = 18;
pub(crate) const COMMAND_AMF0: u8 = 20;

/// Answers a client's handshake as a server, the plain way every publisher
/// takes: S1 random, S2 the client's C1 echoed.
pub(crate) fn handshake(stream: &mut (impl Read + Write)) -> io::Result<()> {
    let mut c0c1 = vec![0u8; 1 + HANDSHAKE];
    stream.read_exact(&mut c0c1)?;
    if c0c1[0] != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("RTMP version {} asked for, 3 spoken", c0c1[0]),
        ));
    }
    let mut out = Vec::with_capacity(1 + 2 * HANDSHAKE);
    out.push(3);
    // S1: a time and zeros, then random bytes.
    out.extend_from_slice(&[0; 8]);
    let state = RandomState::new();
    while out.len() < 1 + HANDSHAKE {
        out.extend_from_slice(&state.hash_one(out.len()).to_le_bytes());
    }
    out.truncate(1 + HANDSHAKE);
    out.extend_from_slice(&c0c1[1..]);
    stream.write_all(&out)?;
    stream.flush()?;
    let mut c2 = vec![0u8; HANDSHAKE];
    stream.read_exact(&mut c2)
}

/// One whole message.
#[derive(Debug)]
pub(crate) struct Message {
    pub(crate) kind: u8,
    pub(crate) stream: u32,
    /// Milliseconds, as the sender counts them — wrapping at 2^32.
    pub(crate) timestamp: u32,
    pub(crate) payload: Vec<u8>,
}

/// What a chunk stream carries over from one chunk to the next.
#[derive(Default)]
struct ChunkStream {
    timestamp: u32,
    delta: u32,
    length: usize,
    kind: u8,
    stream: u32,
    /// Whether its last header's timestamp was extended, which its
    /// continuation chunks then repeat.
    extended: bool,
    /// The message under way.
    partial: Vec<u8>,
}

/// Reads messages out of chunks.
pub(crate) struct ChunkReader<R> {
    input: R,
    chunk_size: usize,
    streams: HashMap<u32, ChunkStream>,
    /// Everything read, handshake aside — what an acknowledgement counts.
    pub(crate) bytes_read: u64,
}

impl<R: Read> ChunkReader<R> {
    pub(crate) fn new(input: R) -> Self {
        Self {
            input,
            chunk_size: 128,
            streams: HashMap::new(),
            bytes_read: 0,
        }
    }

    fn read(&mut self, buffer: &mut [u8]) -> io::Result<()> {
        self.input.read_exact(buffer)?;
        self.bytes_read += buffer.len() as u64;
        Ok(())
    }

    fn byte(&mut self) -> io::Result<u8> {
        let mut byte = [0u8; 1];
        self.read(&mut byte)?;
        Ok(byte[0])
    }

    fn u24(&mut self) -> io::Result<u32> {
        let mut bytes = [0u8; 3];
        self.read(&mut bytes)?;
        Ok(u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]))
    }

    fn u32_be(&mut self) -> io::Result<u32> {
        let mut bytes = [0u8; 4];
        self.read(&mut bytes)?;
        Ok(u32::from_be_bytes(bytes))
    }

    /// The next whole message. Chunk size changes and aborts are taken here
    /// and handed on too, for the record.
    pub(crate) fn next(&mut self) -> io::Result<Message> {
        loop {
            if let Some(message) = self.chunk()? {
                match message.kind {
                    SET_CHUNK_SIZE if message.payload.len() >= 4 => {
                        let size = u32::from_be_bytes(message.payload[..4].try_into().unwrap())
                            & 0x7fff_ffff;
                        if size == 0 || size as usize > MAX_MESSAGE {
                            return Err(invalid(format!("a chunk size of {size}")));
                        }
                        self.chunk_size = size as usize;
                    }
                    ABORT if message.payload.len() >= 4 => {
                        let id = u32::from_be_bytes(message.payload[..4].try_into().unwrap());
                        if let Some(stream) = self.streams.get_mut(&id) {
                            stream.partial.clear();
                        }
                    }
                    _ => {}
                }
                return Ok(message);
            }
        }
    }

    /// One chunk; the message it completes, if it does.
    fn chunk(&mut self) -> io::Result<Option<Message>> {
        let first = self.byte()?;
        let format = first >> 6;
        let id = match first & 0x3f {
            0 => u32::from(self.byte()?) + 64,
            1 => {
                let low = self.byte()?;
                let high = self.byte()?;
                u32::from(high) * 256 + u32::from(low) + 64
            }
            id => u32::from(id),
        };
        if !self.streams.contains_key(&id) {
            if self.streams.len() >= MAX_CHUNK_STREAMS {
                return Err(invalid("too many chunk streams".into()));
            }
            if format != 0 {
                return Err(invalid(format!(
                    "chunk stream {id} begins without a full header"
                )));
            }
        }
        let mut header = [0u32; 4];
        // Timestamp (or delta), length, type, stream: as many as the format
        // carries.
        if format <= 2 {
            header[0] = self.u24()?;
        }
        if format <= 1 {
            header[1] = self.u24()?;
            header[2] = u32::from(self.byte()?);
        }
        if format == 0 {
            let mut bytes = [0u8; 4];
            self.read(&mut bytes)?;
            header[3] = u32::from_le_bytes(bytes);
        }
        let continuing = {
            let stream = self.streams.entry(id).or_default();
            format == 3 && !stream.partial.is_empty()
        };
        let extended = if format <= 2 {
            header[0] == 0xff_ffff
        } else {
            self.streams[&id].extended
        };
        let extended_value = if extended { Some(self.u32_be()?) } else { None };
        let stream = self.streams.get_mut(&id).expect("made above");
        if format <= 2 {
            stream.extended = extended;
        }
        if !continuing {
            // A header begins a message: what was read of one before it,
            // left unfinished, is let go of — taken as this one's start,
            // it was more than a shorter one's length.
            stream.partial.clear();
            let field = extended_value.unwrap_or(header[0]);
            match format {
                0 => {
                    stream.timestamp = field;
                    stream.delta = 0;
                }
                1 | 2 => {
                    stream.delta = field;
                    stream.timestamp = stream.timestamp.wrapping_add(field);
                }
                _ => stream.timestamp = stream.timestamp.wrapping_add(stream.delta),
            }
            if format <= 1 {
                stream.length = header[1] as usize;
                stream.kind = header[2] as u8;
            }
            if format == 0 {
                stream.stream = header[3];
            }
            if stream.length > MAX_MESSAGE {
                return Err(invalid(format!("a message of {} bytes", stream.length)));
            }
        }
        let want = (stream.length - stream.partial.len()).min(self.chunk_size);
        let start = stream.partial.len();
        stream.partial.resize(start + want, 0);
        let mut body = std::mem::take(&mut stream.partial);
        let read = self.read(&mut body[start..]);
        let stream = self.streams.get_mut(&id).expect("made above");
        stream.partial = body;
        read?;
        if stream.partial.len() < stream.length {
            return Ok(None);
        }
        Ok(Some(Message {
            kind: stream.kind,
            stream: stream.stream,
            timestamp: stream.timestamp,
            payload: std::mem::take(&mut stream.partial),
        }))
    }
}

fn invalid(what: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

/// Writes messages as chunks of [`OUT_CHUNK`] once [`Self::announce_chunk_size`]
/// has said so, of RTMP's 128 before.
pub(crate) struct ChunkWriter<W> {
    output: W,
    chunk_size: usize,
}

impl<W: Write> ChunkWriter<W> {
    pub(crate) fn new(output: W) -> Self {
        Self {
            output,
            chunk_size: 128,
        }
    }

    /// Tells the client this server's chunks are [`OUT_CHUNK`] from now on.
    pub(crate) fn announce_chunk_size(&mut self) -> io::Result<()> {
        self.write(2, SET_CHUNK_SIZE, 0, &(OUT_CHUNK as u32).to_be_bytes())?;
        self.chunk_size = OUT_CHUNK;
        Ok(())
    }

    /// `payload` as one message of `kind` on chunk stream `id`.
    pub(crate) fn write(
        &mut self,
        id: u8,
        kind: u8,
        stream: u32,
        payload: &[u8],
    ) -> io::Result<()> {
        debug_assert!((2..64).contains(&id));
        let mut out = Vec::with_capacity(payload.len() + 16);
        out.push(id);
        out.extend_from_slice(&[0, 0, 0]);
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes()[1..]);
        out.push(kind);
        out.extend_from_slice(&stream.to_le_bytes());
        for (index, chunk) in payload.chunks(self.chunk_size).enumerate() {
            if index > 0 {
                out.push(0xc0 | id);
            }
            out.extend_from_slice(chunk);
        }
        self.output.write_all(&out)?;
        self.output.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk as a sender writes it, by hand.
    fn chunk(format: u8, id: u8, header: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = vec![(format << 6) | id];
        out.extend_from_slice(header);
        out.extend_from_slice(body);
        out
    }

    fn full_header(timestamp: u32, length: usize, kind: u8, stream: u32) -> Vec<u8> {
        let mut header = timestamp.min(0xff_ffff).to_be_bytes()[1..].to_vec();
        header.extend_from_slice(&(length as u32).to_be_bytes()[1..]);
        header.push(kind);
        header.extend_from_slice(&stream.to_le_bytes());
        if timestamp >= 0xff_ffff {
            header.extend_from_slice(&timestamp.to_be_bytes());
        }
        header
    }

    #[test]
    fn messages_are_put_back_together_across_chunks_and_streams() {
        let video = vec![7u8; 300];
        let mut wire = Vec::new();
        // A video message of 300 bytes in chunks of 128, with an audio
        // message between its chunks on another chunk stream.
        wire.extend(chunk(
            0,
            6,
            &full_header(1000, 300, VIDEO, 1),
            &video[..128],
        ));
        wire.extend(chunk(0, 4, &full_header(1005, 2, AUDIO, 1), &[1, 2]));
        wire.extend(chunk(3, 6, &[], &video[128..256]));
        wire.extend(chunk(3, 6, &[], &video[256..]));
        // The next video message: a delta alone (format 2), then one that
        // reuses it whole (format 3).
        wire.extend(chunk(2, 6, &40u32.to_be_bytes()[1..], &video[..128]));
        wire.extend(chunk(3, 6, &[], &video[128..256]));
        wire.extend(chunk(3, 6, &[], &video[256..]));
        wire.extend(chunk(3, 6, &[], &video[..128]));
        wire.extend(chunk(3, 6, &[], &video[128..256]));
        wire.extend(chunk(3, 6, &[], &video[256..]));
        let mut reader = ChunkReader::new(&wire[..]);
        let audio = reader.next().unwrap();
        assert_eq!(
            (audio.kind, audio.timestamp, audio.payload),
            (AUDIO, 1005, vec![1, 2])
        );
        let first = reader.next().unwrap();
        assert_eq!(
            (first.kind, first.stream, first.timestamp),
            (VIDEO, 1, 1000)
        );
        assert_eq!(first.payload, video);
        assert_eq!(reader.next().unwrap().timestamp, 1040);
        assert_eq!(reader.next().unwrap().timestamp, 1080);
        assert_eq!(reader.bytes_read, wire.len() as u64);
    }

    /// A header in the middle of a message begins another: the one left
    /// unfinished is let go of, and the new one — shorter than what was read
    /// of the old — is read whole. Taken for the old one's next chunk, its
    /// length less what was read of the old overflowed (found fuzzing).
    #[test]
    fn a_message_begun_over_an_unfinished_one_is_read_whole() {
        let mut wire = Vec::new();
        wire.extend(chunk(0, 6, &full_header(1000, 300, VIDEO, 1), &[7; 128]));
        let mut short = 10u32.to_be_bytes()[1..].to_vec();
        short.splice(0..0, 40u32.to_be_bytes()[1..].iter().copied());
        short.push(AUDIO);
        wire.extend(chunk(1, 6, &short, &[1; 10]));
        let mut reader = ChunkReader::new(&wire[..]);
        let message = reader.next().unwrap();
        assert_eq!(
            (message.kind, message.timestamp, message.payload),
            (AUDIO, 1040, vec![1; 10])
        );
        assert!(reader.next().is_err(), "nothing more: the stream ends");
    }

    #[test]
    fn an_extended_timestamp_is_read_and_repeated_on_continuations() {
        let body = vec![1u8; 200];
        let late = 0x0123_4567;
        let mut wire = Vec::new();
        wire.extend(chunk(0, 6, &full_header(late, 200, VIDEO, 1), &body[..128]));
        let mut continuation = late.to_be_bytes().to_vec();
        continuation.extend_from_slice(&body[128..]);
        wire.extend(chunk(3, 6, &[], &continuation));
        let mut reader = ChunkReader::new(&wire[..]);
        let message = reader.next().unwrap();
        assert_eq!(message.timestamp, late);
        assert_eq!(message.payload, body);
    }

    #[test]
    fn a_larger_chunk_size_is_taken() {
        let body = vec![3u8; 1000];
        let mut wire = Vec::new();
        wire.extend(chunk(
            0,
            2,
            &full_header(0, 4, SET_CHUNK_SIZE, 0),
            &4096u32.to_be_bytes(),
        ));
        wire.extend(chunk(0, 6, &full_header(0, 1000, VIDEO, 1), &body));
        let mut reader = ChunkReader::new(&wire[..]);
        assert_eq!(reader.next().unwrap().kind, SET_CHUNK_SIZE);
        assert_eq!(reader.next().unwrap().payload, body);
    }

    #[test]
    fn what_cannot_be_right_is_refused() {
        let huge = chunk(0, 6, &full_header(0, MAX_MESSAGE + 1, VIDEO, 1), &[]);
        assert!(ChunkReader::new(&huge[..]).next().is_err());
        let headless = chunk(3, 6, &[], &[0; 10]);
        assert!(ChunkReader::new(&headless[..]).next().is_err());
    }

    #[test]
    fn what_is_written_reads_back() {
        let mut wire = Vec::new();
        let mut writer = ChunkWriter::new(&mut wire);
        writer.announce_chunk_size().unwrap();
        let payload = vec![9u8; 10_000];
        writer.write(3, COMMAND_AMF0, 0, &payload).unwrap();
        let mut reader = ChunkReader::new(&wire[..]);
        assert_eq!(reader.next().unwrap().kind, SET_CHUNK_SIZE);
        let message = reader.next().unwrap();
        assert_eq!(
            (message.kind, message.payload.len()),
            (COMMAND_AMF0, 10_000)
        );
    }

    #[test]
    fn a_handshake_echoes_c1() {
        struct Duplex {
            input: io::Cursor<Vec<u8>>,
            output: Vec<u8>,
        }
        impl Read for Duplex {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                self.input.read(buffer)
            }
        }
        impl Write for Duplex {
            fn write(&mut self, data: &[u8]) -> io::Result<usize> {
                self.output.write(data)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut input = vec![3u8];
        input.extend((0..HANDSHAKE).map(|i| i as u8));
        input.extend(vec![0u8; HANDSHAKE]);
        let mut duplex = Duplex {
            input: io::Cursor::new(input.clone()),
            output: Vec::new(),
        };
        handshake(&mut duplex).unwrap();
        assert_eq!(duplex.output.len(), 1 + 2 * HANDSHAKE);
        assert_eq!(duplex.output[0], 3);
        assert_eq!(&duplex.output[1 + HANDSHAKE..], &input[1..1 + HANDSHAKE]);

        let mut wrong = Duplex {
            input: io::Cursor::new(vec![6u8; 1 + 2 * HANDSHAKE]),
            output: Vec::new(),
        };
        assert!(handshake(&mut wrong).is_err());
    }
}
