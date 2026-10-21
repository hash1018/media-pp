# rtsp_listen

`RtspSource::listen -> PacketCounter`, one counter per stream: waits for a
publisher to push an RTSP stream into it — an encoder, a camera that pushes,
`ffmpeg -f rtsp`, or the `rtsp_serve` example — and counts the packets of
every stream the publisher announced, once a second, until it finishes.

It receives one publisher and then ends: when the publisher closes its
connection the counts stop, `finished` is printed and it exits. A publisher
that goes silent, or none arriving within a minute, is an error and a
non-zero exit. Nothing is decoded, so any codec the publisher sends is
counted.

```sh
cargo run -p rtsp_listen -- rtsp://0.0.0.0:8554/live
ffmpeg -re -i video.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live
cargo run -p rtsp_serve -- video.mp4 rtsp://127.0.0.1:8554/live   # or this
```

The publisher picks the transport. For `-rtsp_transport udp`, listen at the
address the publisher reaches this machine at — `rtsp://127.0.0.1:8554/live`
for one on the same machine — not `0.0.0.0`, which FFmpeg's listener would
hand the publisher as where to send its packets.
