# rtsp_serve_seek

Same Demux -> Queue -> Pacer -> RtspSink chain as `rtsp_serve`, plus a
terminal prompt that reads timestamps and calls `Pipeline::seek` with them
while the stream is live — lets a viewer jump around a live-served RTSP
stream instead of only watching it play straight through.

```sh
cargo run -p rtsp_serve_seek -- path/to/video.mp4 [rtsp://host:port/path]
ffplay rtsp://127.0.0.1:8554/stream    # in another terminal; default URL shown
pause
resume
seek 30
seek 1:15
q
```
