# webrtc_server

A WebRTC server a browser plays: a file played to any number of WebRTC
players as a camera would, and a second path a WHIP publisher pushes into,
which players play too.

```text
FileDemuxer -> Pacer -> BroadcastSink    (the file, looped, at its own speed)
                             |
                     Broadcast "file" ----\
                                           WebRtcServer -> players (WHEP)
publisher (WHIP) -> Broadcast "live" -----/
```

Open `http://127.0.0.1:8080/file` in a browser: the page offers to play the
path through WHEP, and the server answers. Each player is sent the broadcast's
packets as they are, from the newest keyframe. WebRTC carries H.264, VP8, VP9
and AV1 pictures and Opus sound, so a file's AAC is left out; a browser takes
H.264 without B-frames.

```sh
cargo run -p webrtc_server -- input.mp4 [--udp 127.0.0.1:8189] [--http 127.0.0.1:8080] [--threads N] [--seconds N] [--record DIR] [--rtmp ADDRESS]
```

Both addresses default to this machine alone. To serve the network, give the
address the network reaches this machine at to `--udp` — a browser is told to
send there — and `0.0.0.0:8080` to `--http`. A WHIP publisher — OBS, or a
page of your own — posts to `http://127.0.0.1:8080/live/whip`, and
`http://127.0.0.1:8080/live` plays what it sends. It runs until Enter is
pressed, or for `--seconds`, printing what each session does. `--threads` is
how many threads — each with a UDP port of its own, from the one `--udp` names
on — it runs its sessions on, two by default.

With `--rtmp ADDRESS` it also takes an RTMP publisher into `live` — OBS with
the "Custom" service, its server `rtmp://127.0.0.1:1935/live` and its stream
key `stream` — and `http://127.0.0.1:8080/live` plays it in a browser. Set
the encoder to send no B-frames (x264 or NVENC, "B-frames" 0), which a
browser's H.264 needs; its AAC, which WebRTC does not carry, is left out:

```text
OBS -> RtmpServer "live/stream" -> Broadcast "live" -> WebRtcServer -> browsers
```

With `--record DIR` it also records what `file` plays, as it plays it, into
an HLS event playlist under `DIR` — `index.m3u8`, fMP4 segments of about two
seconds cut at keyframes, and `init.mp4` — which plays while it grows, back to
its start, and is finished as the server stops:

```text
Broadcast "file" -> BroadcastSource -> HlsMuxer (event playlist)
```
