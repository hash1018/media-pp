# Streaming benchmarks

How many players one `RtspServer` serves from one `Broadcast`, what each
costs the server, and whether any of them falls behind. The questions are
the ones a deployment has to answer:

- how many players a machine serves before one of them loses anything;
- what each player costs: processor and memory;
- TCP — interleaved on the RTSP connection — against UDP;
- how many threads the server needs.

| Page | What is in it |
|---|---|
| [method.md](method.md) | The harness, the stream, what is measured, and how to run it |
| [windows-i5-12400f.md](windows-i5-12400f.md) | The machine, and every run's results on it, each read |
| [findings.md](findings.md) | What measuring changed: UDP players losing half of what they were sent, and why |

## In short

On a Core i5-12400F under Windows, a 1080p stream at 12.3 Mbit/s (H.264,
a keyframe every 2 s, with AAC), players on the same machine:

| Players | Over | Server threads | Delivered | Server processor | Server memory | Lost |
|---:|---|---:|---:|---:|---:|---:|
| 100 | TCP | 2 | 1.3 Gbit/s | 0.32 cores | 48 MB | 0 |
| 400 | TCP | 2 | 5.0 Gbit/s | 0.99 cores | 60 MB | 0 |
| 800 | TCP | 2 | 9.7 Gbit/s | 1.75 cores | 71 MB | 0 |
| 100 | UDP | 2 | 1.3 Gbit/s | 0.88 cores | 47 MB | 0 |
| 400 | UDP | 6 | 5.0 Gbit/s | 4.43 cores | 66 MB | 0 |

A player over TCP costs the server about 0.18 of a core per Gbit/s, and
memory hardly at all: every player reads the one broadcast, and only its
RTP packets are its own. Over UDP a player costs four to five times as much
— a system call for every RTP packet rather than one for every picture —
so the two threads a server has by default carry about a hundred UDP
players of this stream before they fall behind, and six about four
hundred. No run lost a packet. A network card, not the server, is what
runs out first: a gigabit link carries some 75 players of this stream.

## Running it

```sh
cargo build --release -p rtsp_server -p rtsp_load
target/release/rtsp_server stream.mp4 127.0.0.1:8554 --seconds 100000 &
target/release/rtsp_load rtsp://127.0.0.1:8554/file 400 --seconds 30
```

[method.md](method.md#running-it) has the stream, and the script that
measures the server's processor and memory while it runs.
