# Streaming on Windows, Core i5-12400F

## The machine

| | |
|---|---|
| Processor | Intel Core i5-12400F, 6 cores, 12 threads |
| Memory | 32 GB |
| System | Windows 11 Pro 26200 |
| Build | release, media-pp at the commit adding this page |
| Stream | [method.md](method.md#the-stream)'s: 1080p24 H.264 at 12.3 Mbit/s, a keyframe every 2 s, two B-frames, AAC at 128 kbit/s |
| Players | `rtsp_load` and `webrtc_load` on the same machine, over the loopback; 30 s a run |

Every figure up to the ceilings is the range over three rounds of the same
runs, one after another (see [method.md](method.md#rounds-and-ceilings)):
`0.26–0.39` is the least and the most of the three.

## TCP

Players asking for RTP interleaved on their RTSP connections, the server on
its default two threads:

| Players | Delivered | Server processor | Server memory | Lost | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 Mbit/s | 0.03–0.05 cores | 39–40 MB | 0 | 18–28 / 18–28 / 18–28 ms | 0–0.01 cores |
| 50 | 644–652 Mbit/s | 0.15–0.29 cores | 45–46 MB | 0 | 24–32 / 30–36 / 30–36 ms | 0.05–0.08 cores |
| 100 | 1.25–1.29 Gbit/s | 0.26–0.39 cores | 49 MB | 0 | 28–29 / 35–40 / 35–40 ms | 0.08–0.11 cores |
| 200 | 2.56–2.59 Gbit/s | 0.42–0.57 cores | 53–54 MB | 0 | 39–44 / 64–81 / 65–84 ms | 0.13–0.15 cores |
| 300 | 3.87–3.88 Gbit/s | 0.44–0.76 cores | 57–58 MB | 0 | 50–63 / 71–109 / 75–110 ms | 0.12–0.24 cores |
| 400 | 4.98–5.00 Gbit/s | 0.67–0.97 cores | 63 MB | 0 | 64–83 / 102–130 / 105–137 ms | 0.22–0.29 cores |
| 600 | 7.36–7.42 Gbit/s | 1.27–1.50 cores | 71–82 MB | 0 | 136–163 / 216–291 / 221–297 ms | 0.38–0.48 cores |
| 800 | 9.78–9.81 Gbit/s | 1.68–1.76 cores | 81–85 MB | 0 | 183–287 / 324–469 / 332–511 ms | 0.51–0.55 cores |

Each player costs the server about the same, 0.18 of a core per Gbit/s
— 0.0022 cores for a 12 Mbit/s player — from one to eight hundred: a
picture's RTP packets go to a TCP player in one write, so the server makes
24 system calls a second for each. Memory grows by about 55 kB a player;
the packets every player is sent are the broadcast's, shared, and each
player's own are only the RTP cut from them. The rounds differ by a
third in processor time at a few hundred players and by far less at
eight: the machine's other work, not the server's.

The lag spread grows with the players — the server's threads take them in
turn, and the eight-hundredth waits for the others — but it stops growing:
at 800 the largest is half a second, and no player lost a packet.

### Where it stops keeping up

Past eight hundred, once each (one round):

| Players | Server threads | Delivered | Server processor | Server memory | Lowest a player got | Lag spread, median / p99 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 2 | 11.8 Gbit/s | 1.80 cores | 81 MB | 10.5 Mbit/s | 250 / 442 / 509 ms |
| 1200 | 2 | 12.9 Gbit/s | 1.89 cores | 95 MB | 7.9 Mbit/s | 2972 / 4546 / 5817 ms |
| 1600 | 2 | 12.6 Gbit/s | 1.90 cores | 114 MB | 2.4 Mbit/s | 5098 / 6749 / 8477 ms |
| 2000 | 2 | 12.6 Gbit/s | 1.90 cores | 135 MB | 0.6 Mbit/s | 4456 / 8540 / 9513 ms |
| 1000 | 6 | 11.8 Gbit/s | 3.19 cores | 92 MB | 10.4 Mbit/s | 246 / 525 / 534 ms |
| 1200 | 6 | 14.1 Gbit/s | 3.84 cores | 106 MB | 10.4 Mbit/s | 237 / 423 / 435 ms |
| 1600 | 6 | 17.8 Gbit/s | 4.57 cores | 165 MB | 9.2 Mbit/s | 584 / 1226 / 1525 ms |

Two threads carry a thousand players and then are full — 1.9 cores is two
threads busy all the time — and what they deliver stays at some 12.6
Gbit/s however many more ask: each player gets less, and falls behind.
Six threads carry twelve hundred as two carried a thousand. At sixteen
hundred they deliver nearly 18 Gbit/s and have begun to fall behind,
with the server's four and a half cores and the players' one and a half
taking the whole of the machine's six; a dedicated machine's ceiling is
higher than this, by how much was not measured. No player lost a packet
in any of these runs: a TCP player that falls behind is slowed, not cut.

## UDP

Players asking for RTP on UDP ports of their own:

| Players | Server threads | Delivered | Server processor | Server memory | Lost | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2 | 650 Mbit/s | 0.51–0.53 cores | 42–43 MB | 0 | 62–79 / 85–138 / 85–138 ms | 0.46–0.52 cores |
| 100 | 2 | 1.29 Gbit/s | 0.89–0.92 cores | 46 MB | 0 | 172–220 / 310–321 / 316–330 ms | 0.63–0.77 cores |
| 200 | 2 | 2.52–2.54 Gbit/s | 1.62–1.66 cores | 54–60 MB | 0 | 711–1424 / 1427–2269 / 1625–2376 ms | 0.94–1.00 cores |
| 200 | 6 | 2.59 Gbit/s | 2.84–2.97 cores | 52–54 MB | 0 | 131–151 / 250–258 / 258–262 ms | 1.17–1.33 cores |
| 400 | 6 | 5.03–5.07 Gbit/s | 4.30–4.47 cores | 64–66 MB | 0 | 435–465 / 773–939 / 804–1000 ms | 1.48–1.69 cores |

A UDP player costs the server four to five times what a TCP one does: every
RTP packet — some 1,100 a second of this stream — is a system call of its
own, where TCP takes a picture's in one. Two threads keep up with a hundred
players; at two hundred they are busy all the time, and the lag spread of a
second and more is players falling behind. Six threads carry two hundred as
two carried a hundred, and four hundred with under a second's spread.

### Where it stops keeping up

| Players | Server threads | Delivered | Server processor | Server memory | Lost | Lag spread, median / p99 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 6 | 6.18 Gbit/s | 5.29 cores | 69 MB | 0.6 %, one player | 1288 / 5360 / 25113 ms |
| 600 | 6 | 6.22 Gbit/s | 5.35 cores | 95 MB | 0 | 4969 / 7730 / 7919 ms |
| 800 | 6 | 6.08 Gbit/s | 5.27 cores | 132 MB | 0.9 %, two players | 5542 / 8926 / 23081 ms |

Six threads deliver some 6.2 Gbit/s over UDP and no more — five and a
third cores, beside the players' two — so five hundred players of this
stream is past what they carry. There a player that fell more than the
broadcast keeps behind was moved on to a newer keyframe, which is the
loss: whole groups of pictures, never a packet here and there.

Nothing was lost below the ceilings: a player is moved on only once it is
further behind than the broadcast keeps — five seconds by default.
[findings.md](findings.md) says why UDP players lost half of what they
were sent before.

## WebRTC

WHEP players of the same stream without B-frames and with Opus sound
(`webrtc-1080p`, 12.2 Mbit/s), on as many server threads as given:

| Players | Server threads | Delivered | Server processor | Server memory | Pictures after a gap | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1 | 597–600 Mbit/s | 0.59–0.75 cores | 140–143 MB | 1–5 | 271–334 / 1226–1282 / 1226–1282 ms | 0.45–0.52 cores |
| 100 | 1 | 821–1032 Mbit/s | 0.97–0.99 cores | 262–266 MB | 0–4 | 3704–4664 / 4645–5357 / 4646–5358 ms | 0.70–0.74 cores |
| 100 | 4 | 1.19 Gbit/s | 2.07–2.48 cores | 257–262 MB | 9–12 | 208–236 / 1229–1277 / 1248–1291 ms | 1.38–1.62 cores |
| 200 | 4 | 2.11–2.25 Gbit/s | 3.35–3.66 cores | 476–479 MB | 9–16 | 892–2681 / 3153–4962 / 3179–5195 ms | 2.18–2.36 cores |
| 200 | 6 | 2.37–2.38 Gbit/s | 4.23–4.36 cores | 473–479 MB | 14–22 | 352–381 / 1245–1494 / 1287–1991 ms | 2.63–2.73 cores |
| 300 | 6 | 2.96–2.99 Gbit/s | 5.18–5.45 cores | 685–694 MB | 31–44 | 1273–3646 / 5510–5513 / 5579–5818 ms | 3.26–3.37 cores |

and once, between those:

| Players | Server threads | Delivered | Server processor | Server memory | Pictures after a gap | Lag spread, median / p99 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 6 | 1.79 Gbit/s | 3.26 cores | 371 MB | 16 | 232 / 1741 / 1836 ms |
| 250 | 6 | 2.64 Gbit/s | 4.78 cores | 581 MB | 33 | 1891 / 5235 / 5441 ms |

A WebRTC player costs the server 0.015 to 0.02 of a core — six to eight
times an RTSP player over TCP: every packet is encrypted for that player
alone, and `str0m`, which sends it, takes a packet at a time. One thread
keeps up with some 70 players of this stream and then falls behind — a
hundred are all sent everything, four to five seconds late. Threads
scale it: four carry a hundred as one carries fifty, and six two hundred.
At two hundred and fifty, six threads fall behind, with the server's
five cores and the players' three past the machine's six physical ones;
on a machine of its own a server would carry more. Each player takes
some 2.4 MB of the server's memory, most of it `str0m`'s buffers of what
it has sent, kept for resending.

Pictures after a gap are packets lost on the loopback, nearly all by
players at the start of a run; `str0m` waits up to thirty pictures — about
1.3 s — for a lost packet before it goes on, which is the p99 lag spread of
the runs that kept up.

## Recording while serving

The same runs with the server also recording what it serves (`--record`):
a `BroadcastSource` of the same broadcast into an `HlsMuxer`, an event
playlist of fMP4 segments, written to the same disk the whole time:

| Players | Over | Server threads | Server processor | Server memory | Lost or after a gap | Lag spread, median / p99 / max | Without recording |
|---:|---|---:|---:|---:|---:|---:|---|
| 400 | TCP | 2 | 0.89 cores | 64 MB | 0 | 73 / 112 / 114 ms | 0.67–0.97 cores, 63 MB |
| 100 | WebRTC | 4 | 2.38 cores | 269 MB | 7 pictures | 241 / 1244 / 1262 ms | 2.07–2.48 cores, 257–262 MB |

Recording costs nothing the rounds' own spread does not cover: it is one
reader more of the broadcast, writing what it reads to a file without
decoding it. Each recording was 48 seconds — the server's whole run — of
1,152 pictures, every one of the stream's 24 a second; FFmpeg read each
back from its playlist without an error, and both ended with
`#EXT-X-ENDLIST` once the server stopped.

## Read together

- For many players, TCP: it costs a quarter of what UDP does, and every
  player — VLC, FFmpeg, a camera viewer — can ask for it. Two threads
  carry a thousand players of a 12 Mbit/s stream, six at least twelve
  hundred.
- `RtspServerOptions::threads` is what UDP players need more of: about one
  thread for every 70 of this stream's players, and six carry four
  hundred and stop at some 6 Gbit/s.
- Memory is not what runs out: 85 MB served eight hundred players over
  TCP.
- What runs out first is the network: a gigabit link carries some 75 of
  these players, at well under a tenth of a core.
- A WebRTC player costs most: give `WebRtcServerOptions::threads` one
  thread for every 70 players of a stream like this one, up to the cores
  the machine has to spare.
- Recording what is served costs a reader, not a player's worth.
