# pace

Demux -> SwDecoder -> Pacer -> FrameCounter: proves `Pacer` paces decoded
frames out at real playback speed (via PTS + `Clock`) instead of as fast as
decode can produce them. Compare against `decode`, which runs the same chain
without a `Pacer` and finishes as fast as possible.

```sh
cargo run -p pace -- path/to/video.mp4
```
