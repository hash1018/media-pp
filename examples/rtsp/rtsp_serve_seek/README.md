# rtsp_serve_seek

The same `FileDemuxer -> Queue -> Pacer -> RtspMuxer` chain as `rtsp_serve`
(video and audio both, when the file has them), plus a terminal prompt that
pauses, resumes and seeks the pipeline while the stream is live, so a viewer
can jump around a served stream instead of only watching it play through.

```sh
cargo run -p rtsp_serve_seek -- path/to/video.mp4 [rtsp://host:port/path]
ffplay rtsp://127.0.0.1:8554/stream    # in another terminal; default URL shown
pause
resume
seek 30
seek 1:15
q
```
