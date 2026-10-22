# rtsp_server

An RTSP server: plays a file to any number of RTSP players as a camera would,
and takes a publisher's push on a second path, which players play too.

```text
FileDemuxer -> Pacer -> BroadcastSink    (the file, looped, at its own speed)
                             |
                     Broadcast "file" ----\
                                           RtspServer -> players
publisher -> RtspServer -> Broadcast "live" /
```

Each player is sent the broadcast's packets as they are — nothing is decoded
or encoded for it — from the newest keyframe on. `live` takes one publisher at
a time; one that leaves ends its run, and the next one's players go on.

```sh
cargo run -p rtsp_server -- input.mp4 [address] [--seconds N] [--threads N] [--record DIR]
ffplay rtsp://127.0.0.1:8554/file
ffmpeg -re -i video.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/live
ffplay rtsp://127.0.0.1:8554/live
```

The address defaults to `127.0.0.1:8554`, this machine alone; give
`0.0.0.0:8554` to serve the network. It runs until Enter is pressed, or for
`--seconds`, printing how many players there are every five seconds;
`--threads` is how many threads the server runs its connections on, two by
default.

With `--record DIR` it also records what `file` plays, as it plays it, into
an HLS event playlist under `DIR` — `index.m3u8`, fMP4 segments of about two
seconds cut at keyframes, and `init.mp4` — which plays while it grows, back to
its start, and is finished as the server stops:

```text
Broadcast "file" -> BroadcastSource -> HlsMuxer (event playlist)
```
