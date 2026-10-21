# rtsp_load

A load test for an RTSP server: as many players as asked, each a thread that
plays the stream and counts what it is sent — packets, bytes, the ones lost,
and how steadily its pictures arrive — and a summary at the end.
[`docs/benchmarks/streaming`](../../../docs/benchmarks/streaming) is made
with it, against the [`rtsp_server`](../rtsp_server) example.

The players are written for this, with nothing decoded, so that what is
measured is the server: each is a socket and a loop reading RTP out of what
it is sent. Each asks for every track the server describes, interleaved on
its RTSP connection or — with `--udp` — on two UDP ports of its own per track.

```sh
cargo run --release -p rtsp_load -- rtsp://127.0.0.1:8554/file 100 [--seconds 30] [--udp]
```

It prints how much every player received together every five seconds, then:

- `received` — every player's RTP together over the run, and the least and
  most one player received;
- `packets lost` — by the gaps in each track's sequence numbers;
- `lag spread` — for each picture's last packet, its arrival less the newest
  timestamp sent before it, the largest less the smallest after a player's
  first three seconds: how much later than its steadiest a player was handed
  a picture. The median, 99th percentile and largest over the players.

[`bench.ps1`](bench.ps1) runs it against `rtsp_server` for several numbers
of players on Windows, sampling the server's processor time and memory; see
[the method](../../../docs/benchmarks/streaming/method.md).
