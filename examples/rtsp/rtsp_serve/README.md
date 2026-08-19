# rtsp_serve

Demux -> Queue -> Pacer -> RtspSink: remuxes a file's video packets (no
re-encoding) and publishes them at real playback speed to an already-running
RTSP server.

```sh
cargo run -p rtsp_serve -- path/to/video.mp4 [rtsp://host:port/path]
ffplay rtsp://127.0.0.1:8554/stream    # in another terminal; default URL shown
```
