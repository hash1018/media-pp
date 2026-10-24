# webrtc_av_loopback

Shows that one `WebRtcPeer` connection carries more than one track: two peers
over loopback UDP (set up as in `webrtc_loopback`), where peer-a sends a video
and an audio track through one `WebRtcMuxer` — one muxer, one renegotiation,
one socket.

Send side, one `PipelineBuilder` pipeline with two sources:
`TestVideoSource -> Queue -> SwEncoder -> WebRtcMuxer` and
`TestAudioSource -> Queue -> SwAudioEncoder -> WebRtcMuxer`, each encoder's
stream added to the muxer and its output pushed onto that track's sink.
Receive side, the two tracks come out of `WebRtcHandle::next_source` as one
`WebRtcSource` with an output each, in the order the muxer was given them; one
pipeline sends each output to its own `CountingSink`, counted separately to
show the tracks do not cross.

```sh
cargo run -p webrtc_av_loopback
```
