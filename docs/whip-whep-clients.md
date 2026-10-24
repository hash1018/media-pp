# WHIP and WHEP clients: WebRtcMuxer and WebRtcSource

Status: **accepted**, 2026-10-22. File references are relative to
`lib/src/`.

`WebRtcServer` answers WHIP and WHEP: it takes a publisher's stream into a
`Broadcast` and plays one to any number of players. What media-pp has no
side of yet is the client's: a pipeline that publishes to a WHIP server, or
plays from a WHEP one — the server ours, MediaMTX, Cloudflare, or any other.
This adds the two, and a set of tests that carry a stream across every pair
of protocols the crate serves.

## 1. What each is

| | RTSP | WebRTC |
|---|---|---|
| a server | `RtspServer` | `WebRtcServer` |
| a client pushing | `RtspMuxer` | **`WebRtcMuxer`** |
| a client pulling | `RtspSource` | **`WebRtcSource`** |

Both are one WebRTC connection with full ICE (the server's is ICE-lite),
made by one HTTP exchange — the offer `POST`ed, the answer and the
session's `Location` back — and ended by a `DELETE` of that location.
Neither has a signalling server: the URL is all a caller gives.

## 2. WebRtcMuxer

```rust
let mut muxer = WebRtcMuxer::create("cam", "https://server/live/whip", WebRtcOptions {
    bearer_token: Some(token),
    ..WebRtcOptions::default()
})?;
let video = muxer.add_stream("video", StreamInfo::from(&encoder))?;
let audio = muxer.add_stream("audio", StreamInfo::from(&opus))?;
let mut sinks = muxer.open()?;
// .to(sinks.take(video)?) at the end of the video chain, and the audio's
```

Shaped as every muxer is — `RtspMuxer`, `RtmpMuxer`, `FileMuxer`: made,
given its tracks with `add_stream`, opened, and handing out a sink for each
track out of a `MuxerSinks`, by the `MuxerTrack` its `add_stream` returned.
Underneath, each is a `WebRtcTrackSink` on the connection pieces
`WebRtcPeer` and `WebRtcServer` share (`elements/driver/webrtc/connection.rs`,
`outbox.rs`). A track's codec is read from its stream, and one WebRTC
cannot carry is refused at `add_stream`.

`open` does the exchange and waits for the connection — ICE and DTLS —
within `WebRtcOptions::connect_within`, so what goes wrong goes wrong there:
a status the server answers with (401, 404, 409 for a path already
published), a server that cannot be reached, a connection that does not
come up. What it hands out keeps the connection: the last of its sinks
dropped sends the `DELETE`, closes the connection and joins its thread.

## 3. WebRtcSource

```rust
let (source, streams) = WebRtcSource::open("relay", "https://server/cam1/whep", WebRtcOptions::default())?;
```

Shaped as `RtspSource` is: opened on a URL, it answers with the streams it
carries, each an output, so the pipeline after it is built from what is
known. It offers to receive a picture and a sound; `open` returns once each
track the answer took has said what it is — for H.264, its parameter sets —
or, after `WebRtcOptions::describe_within`, with those that have. It is a
`Source` reading an inbox per track that its connection's thread fills, so
it runs on a thread of its own or driven alike. The connection going away
ends its stream; the source dropped sends the `DELETE`.

## 4. What both leave out, for now

- Trickle ICE (`PATCH`): every candidate is in the offer.
- ICE servers from the `Link` header (STUN, TURN): a client behind a NAT
  reaches a server with a public address without them.
- Data channels, and asking a publisher's encoder for a keyframe.

## 5. HTTP

`ureq` with `rustls` (and `ring`) under the `webrtc` feature: blocking, as
the open calls are, needing no runtime, and no system TLS library — `https`
being what a WHIP service is reached at. `WebRtcServer::serve_http` stays
plain HTTP; a proxy in front of it is what terminates TLS.

## 6. Tests across protocols

Each a pipeline into the server and one out of it, on the loopback, with a
picture of OpenH264 (no B-frames, which WebRTC needs) and, where the FFmpeg
build has libopus, Opus sound — judged by what comes out: pictures that
decode, from a keyframe, and sound where both ends carry it.

| In | Out | Checks |
|---|---|---|
| `WebRtcMuxer` → `WebRtcServer` | `WebRtcSource` | the round trip: pictures decode |
| `WebRtcMuxer` → `WebRtcServer` | `RtspSource` from `RtspServer` (same broadcast) | WebRTC in, RTSP out |
| `RtspMuxer` → `RtspServer` | `WebRtcSource` from `WebRtcServer` | RTSP in, WebRTC out; AAC is left out, Opus goes |
| `WebRtcMuxer` → broadcast | `BroadcastSource` → `FileMuxer` (MP4) | WebRTC in, recorded: the file plays |
| `RtspMuxer` → broadcast | `BroadcastSource` → `FileMuxer` (MP4) | RTSP in, recorded |
| `WebRtcSource` from `WebRtcServer` | `FileMuxer` | a remote WebRTC stream recorded |
| a publisher leaving and coming back | a `WebRtcSource` playing throughout | the player goes on across the runs |
| errors | — | 404 path, 409 second publisher, a server that is not there |

They live in `lib/tests/streaming.rs`, under `webrtc` and `rtsp-server`
both, which CI's WebRTC and RTSP steps then build.

## 7. Then, measured

Stage 5 of `docs/driven-pipelines.md`: `docs/benchmarks/streaming` measured
again, after the servers moved onto driven pipelines and readers stopped
copying what they read — each figure the range of several runs, since one
run varies by a tenth.
