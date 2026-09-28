# webrtc_av_loopback

Shows that one `WebRtcPeer` connection carries more than one track: two peers
over loopback UDP (set up as in `webrtc_loopback`), where peer-a adds a video
and an audio track to the same connection — two `add_track` calls, two
renegotiations, one socket.

Send side, one `PipelineBuilder` pipeline with two sources:
`TestVideoSource -> Queue -> SwEncoder -> WebRtcTrackSink` and
`TestAudioSource -> Queue -> SwAudioEncoder -> WebRtcTrackSink`, each encoder's
output pushed onto the sink `next_track()` returned for that track's id.
Receive side, one `PipelineBuilder` pipeline with two
`WebRtcTrackSource -> CountingSink` sources, counted separately to show the
tracks do not cross.

```sh
cargo run -p webrtc_av_loopback
```
