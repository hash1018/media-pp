# webrtc_load

A load test for a WHEP server: as many WebRTC players as asked, each a thread
with a `str0m` connection of its own that plays the stream and counts what it
is sent — pictures, bytes, the gaps where packets were lost, and how steadily
its pictures arrive — and a summary at the end.
[`docs/benchmarks/streaming`](../../../docs/benchmarks/streaming) is made with
it, against the [`webrtc_server`](../webrtc_server) example.

Each player posts an offer for a picture and a sound to the WHEP URL, as a
browser's player does, and decodes nothing: what it costs is the connection's
own — ICE, DTLS, SRTP — and taking RTP apart, which a real player pays too. It
ends its session with a `DELETE` when the run ends.

```sh
cargo run --release -p webrtc_load -- http://127.0.0.1:8080/file/whep 100 [--seconds 30]
```

It prints how much every player received together every five seconds, then:

- `received` — every player's media together over the run, and the least and
  most one player received;
- `pictures` — how many arrived, and how many after a gap: some of what came
  before them was lost;
- `lag spread` — for each picture, its arrival less the newest timestamp handed
  over yet, the largest less the smallest after a player's first three seconds.
  The median, 99th percentile and largest over the players.

[`bench.ps1`](bench.ps1) runs it against `webrtc_server` for several numbers of
players on Windows, sampling the server's processor time and memory; see
[the method](../../../docs/benchmarks/streaming/method.md).
