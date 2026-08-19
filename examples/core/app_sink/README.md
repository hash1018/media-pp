# app_sink

Demux -> SwDecoder -> AppSink: same shape as `decode`, but the terminal sink
is a plain closure instead of a bespoke `FrameCounter` — proves `AppSink` lets
a caller consume frames without writing a dedicated `Element`/`Sink` impl at
all (the GStreamer `appsink` equivalent).

```sh
cargo run -p app_sink -- path/to/video.mp4
```
