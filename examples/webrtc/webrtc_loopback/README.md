# webrtc_loopback

Two `WebRtcPeer`s connected over real loopback UDP, with no browser and no
signaling server. One `Direction::SendRecv` track, opened by
`WebRtcHandle::add_track` on peer-a, carries packets both ways: each side
pushes into the `WebRtcTrackSink` its own `next_track()` returned, and
peer-b, which did not originate the track, declares its codec with
`WebRtcTrackSink::set_codec` before replying — no second `add_track` or
renegotiation for the reverse direction. Each side's inbound track is wired as
its own `WebRtcTrackSource -> CountingSink` pipeline.

The initial connection (ICE and a bootstrap data channel, to get DTLS up
before any media) is made directly through str0m; a real application would
carry the same offer and answer over its own signaling (HTTP, WebSocket, ...).

```sh
cargo run -p webrtc_loopback
```
