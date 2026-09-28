# rtmp_publish

`TestVideoSource + TestAudioSource -> SwEncoder / SwAudioEncoder ->
RtmpMuxer`: publishes a live H.264 + AAC broadcast to an RTMP server, which
is what Twitch and YouTube receive.

Two sources into one muxer is the point: a broadcast is video and audio in
one FLV container, `RtmpMuxer`'s two-track path. Both sources are synthetic,
so nothing has to be captured or granted a permission to run it.

This publishes and does not listen. [MediaMTX] accepts RTMP on port 1935
with its shipped configuration:

```text
./mediamtx                                          # in another terminal
cargo run -p rtmp_publish -- [rtmp://host/app/key] [seconds]
ffplay -fflags nobuffer rtmp://127.0.0.1:1935/live/stream   # in a third
```

The address defaults to `rtmp://127.0.0.1:1935/live/stream` and the length
to 10 seconds.

[MediaMTX]: https://github.com/bluenviron/mediamtx
