# webrtc_av_loopback

Answers the question of whether *one* `WebRtcPeer` connection can carry more
than one track at once: two `WebRtcPeer`s over loopback UDP (same setup as
`webrtc_loopback`), but peer-a adds *two* tracks — one video, one audio —
onto the *same* connection (two `WebRtcHandle::add_track` calls, two
sequential renegotiations, no second `Rtc`/socket/peer).

Send side, one `PipelineBuilder`-built `Pipeline` with two sources (the same
shape `screen_record_av` uses for two *capture* sources): `TestVideoSource
-> Queue -> SwEncoder -> WebRtcTrackSink` and `TestAudioSource -> Queue ->
SwAudioEncoder -> WebRtcTrackSink`, each track's real encoded output pushed
straight onto the `WebRtcTrackSink` its own `add_track` call returned. Receive
side, one `WebRtcTrackSource -> CountingSink` `Pipeline` per track, counting
packets on each independently to prove they don't cross-contaminate.

```sh
cargo run -p webrtc_av_loopback
```
