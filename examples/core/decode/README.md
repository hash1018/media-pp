# decode

Demux -> SwDecoder -> FrameCounter: proves `SwDecoder` (a `Filter`, both
`Source` and `Sink`) actually decodes packets into frames, not just that it
compiles.

```sh
cargo run -p decode -- path/to/video.mp4
```
