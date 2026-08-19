# app_source

A background thread demuxes a file with plain `ffmpeg_next` — standing in for
whatever a real caller would push from (a network receive loop, a camera
SDK's callback, ...) — and pushes its video packets into an `AppSource` one
by one via `AppSourceHandle::push`. The pipeline itself (`AppSource ->
SwDecoder -> FrameCounter`) never touches the file directly; proves
`AppSource` (GStreamer's `appsrc` equivalent) actually carries
externally-produced buffers all the way through decode.

```sh
cargo run -p app_source -- path/to/video.mp4
```
