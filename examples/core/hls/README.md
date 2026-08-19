# hls

TestVideoSource -> SwEncoder -> HlsMuxer: produces a live fMP4 media
playlist, initialization file, and keyframe-aligned media segments.

```sh
cargo run -p hls -- [output-directory] [seconds]
```
