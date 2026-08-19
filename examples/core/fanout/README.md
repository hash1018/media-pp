# fanout

`FileDemuxer -> Queue -> PacketCounter`, twice — once from the video src pad,
once from the audio one. Demonstrates fan-out: open a file, inspect its
streams, then link video and audio to separate branches (each behind its own
`Queue` thread boundary) — just two of the demuxer's src pads, no separate
"Tee" element involved.

```sh
cargo run -p fanout -- path/to/video_and_audio.mp4
```
