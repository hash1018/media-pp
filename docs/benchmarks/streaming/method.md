# How the streaming benchmarks are made

## The harness

Two programs of this repository, each a process of its own:

- [`rtsp_server`](../../../examples/rtsp/rtsp_server) — the server under
  test. It plays a file into a `Broadcast` at the file's own speed, looped,
  as a camera would write one, and serves it at `/file`. `--threads` sets
  `RtspServerOptions::threads`.
- [`rtsp_load`](../../../examples/rtsp/rtsp_load) — the players. Each is a
  thread with an RTSP connection of its own: `DESCRIBE`, a `SETUP` for every
  track the server describes — interleaved on the connection, or with
  `--udp` on two UDP ports of its own per track — and `PLAY`. It then reads
  RTP out of what it is sent and decodes nothing, so that the server is what
  is measured, not a decoder. Players start 5 ms apart.

The players run on the same machine as the server, over the loopback: the
numbers are the server's, with no network between, and the players' own
processor time — given beside each run — is taken from the same cores.

## The stream

`sample.mp4`, 1080p at 24 pictures a second, re-encoded as a live encoder
would send it — a keyframe every two seconds, constant bit rate — with a
tone as its sound:

```sh
ffmpeg -i sample.mp4 -f lavfi -i "sine=frequency=440:sample_rate=48000" \
  -map 0:v -map 1:a -c:v h264_nvenc -preset p4 -rc cbr -g 48 -bf 2 \
  -b:v 12M -maxrate 12M -bufsize 12M -c:a aac -b:a 128k -shortest stream.mp4
```

H.264 at 12.3 Mbit/s with two B-frames, AAC at 128 kbit/s; any encoder with
those settings will do. The keyframe interval matters: a player starts at
the newest keyframe the broadcast holds, and is sent what came after it at
once over TCP — at twice its speed over UDP — so a stream with keyframes ten
seconds apart has every player start with up to ten seconds to catch up,
which is the stream's, not the server's.

## What is measured

For each number of players, a run of 30 seconds:

- **Delivered** — the bytes of RTP every player received, over the run,
  together. It includes what each player catches up on as it joins.
- **Server processor** — the server process's processor time during the
  run, kernel and user, over the run's length: how many cores it kept
  busy.
- **Server memory** — the largest working set the server process had
  during the run, sampled every half second.
- **Lost** — RTP packets a player never received, by the gaps in each
  track's sequence numbers. A player the server moves on to a newer
  keyframe — one that fell behind further than the broadcast keeps — shows
  here.
- **Lag spread** — for each picture's last packet, its arrival less the
  newest timestamp sent before it; the spread of that over the run after a
  player's first three seconds, the largest less the smallest. It is how
  much later than its steadiest a player was handed a picture: the
  stream's own jitter on a server that keeps up, growing without end on
  one that does not. The median, 99th percentile and largest over the
  players are given.

## Running it

Build both, in release:

```sh
cargo build --release -p rtsp_server -p rtsp_load
```

On Windows, [`bench.ps1`](../../../examples/rtsp/rtsp_load/bench.ps1)
starts the server, runs the players for each number asked, and samples the
server's processor time and memory as they play:

```powershell
examples\rtsp\rtsp_load\bench.ps1 -Video stream.mp4 -Players 1,50,100,200,400 -Seconds 30 -Threads 2
examples\rtsp\rtsp_load\bench.ps1 -Video stream.mp4 -Players 100,200 -Seconds 30 -Threads 2 -Udp
```

Elsewhere, run the two by hand, and read the server's processor time
before and after a run — `ps -o cputime= -p <pid>` — and its memory with
`ps -o rss=`:

```sh
target/release/rtsp_server stream.mp4 127.0.0.1:8554 --seconds 100000 --threads 2 &
target/release/rtsp_load rtsp://127.0.0.1:8554/file 400 --seconds 30
```

The server listens on the loopback alone, so no firewall asks about it.
