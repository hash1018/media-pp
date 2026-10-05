# decode

Demux -> SwDecoder -> FrameCounter: proves `SwDecoder` (a `RawFilter`, both
`SrcPads` and `RawSink`) actually decodes packets into frames, not just that it
compiles.

```sh
cargo run -p decode -- path/to/video.mp4
```
