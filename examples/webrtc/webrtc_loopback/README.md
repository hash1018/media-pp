# webrtc_loopback

Two `WebRtcPeer`s connected over real loopback UDP, with no browser and no
signaling server. Each peer sends one VP8 track through a `WebRtcMuxer` its
handle makes (`WebRtcHandle::muxer`), opening it in a renegotiation of its own,
and takes the other's track as a `WebRtcSource` (`WebRtcHandle::next_source`).
Packets are pushed before either source exists: they wait in the track's
inbox, and each side's `WebRtcSource -> CountingSink` pipeline counts all five.

The initial connection (ICE and a bootstrap data channel, to get DTLS up
before any media) is made directly through str0m; a real application would
carry the same offer and answer over its own signaling (HTTP, WebSocket, ...).

```sh
cargo run -p webrtc_loopback
```
