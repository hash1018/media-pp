# rtsp_serve

Demux -> Queue -> Pacer -> RtspMuxer: remuxes a file's packets (no
re-encoding) and publishes them at real playback speed to an already-running
RTSP server.

Video and audio go out as two tracks of one RTSP session when the file has
both; a file with no audio track publishes video alone.

```sh
cargo run -p rtsp_serve -- path/to/video.mp4 [rtsp://host:port/path]
ffplay rtsp://127.0.0.1:8554/stream    # in another terminal; default URL shown
```
