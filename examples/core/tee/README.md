# tee

Demux -> Tee, fanning the same packets out to two independent branches:

- SwDecoder -> FrameCounter: decodes and counts frames
- PacketCounter: counts the raw (still-encoded) packets

Proves `Tee` delivers every packet to both branches — same source data, two
unrelated consumers.

```sh
cargo run -p tee -- path/to/video.mp4
```
