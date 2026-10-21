# Streaming on Windows, Core i5-12400F

## The machine

| | |
|---|---|
| Processor | Intel Core i5-12400F, 6 cores, 12 threads |
| Memory | 32 GB |
| System | Windows 11 Pro 26200 |
| Build | release, media-pp at the commit adding this page |
| Stream | [method.md](method.md#the-stream)'s: 1080p24 H.264 at 12.3 Mbit/s, a keyframe every 2 s, two B-frames, AAC at 128 kbit/s |
| Players | `rtsp_load` on the same machine, over the loopback; 30 s a run |

## TCP

Players asking for RTP interleaved on their RTSP connections, the server on
its default two threads:

| Players | Delivered | Server processor | Server memory | Lost | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 Mbit/s | 0.02 cores | 40 MB | 0 | 18 / 18 / 18 ms | 0.01 cores |
| 50 | 651 Mbit/s | 0.17 cores | 48 MB | 0 | 28 / 38 / 38 ms | 0.06 cores |
| 100 | 1.26 Gbit/s | 0.32 cores | 48 MB | 0 | 31 / 49 / 50 ms | 0.10 cores |
| 200 | 2.54 Gbit/s | 0.55 cores | 53 MB | 0 | 51 / 78 / 79 ms | 0.18 cores |
| 300 | 3.85 Gbit/s | 0.81 cores | 56 MB | 0 | 73 / 122 / 124 ms | 0.27 cores |
| 400 | 5.04 Gbit/s | 0.99 cores | 60 MB | 0 | 141 / 199 / 201 ms | 0.33 cores |
| 600 | 7.39 Gbit/s | 1.46 cores | 62 MB | 0 | 201 / 339 / 346 ms | 0.47 cores |
| 800 | 9.74 Gbit/s | 1.75 cores | 71 MB | 0 | 292 / 449 / 459 ms | 0.56 cores |

Each player costs the server about the same, 0.18 of a core per Gbit/s
— 0.0022 cores for a 12 Mbit/s player — from one to eight hundred: a
picture's RTP packets go to a TCP player in one write, so the server makes
24 system calls a second for each. Memory grows by about 40 kB a player;
the packets every player is sent are the broadcast's, shared, and each
player's own are only the RTP cut from them.

The lag spread grows with the players — the server's two threads take them
in turn, and the eight-hundredth waits for the others — but it stops
growing: at 800 the largest is under half a second, and no player lost a
packet. Eight hundred was as far as this was taken; the server had a
quarter of the processor's cores busy. A 10-gigabit network card carries
about as much as the loopback did here.

## UDP

Players asking for RTP on UDP ports of their own:

| Players | Server threads | Delivered | Server processor | Server memory | Lost | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2 | 650 Mbit/s | 0.47 cores | 45 MB | 0 | 79 / 111 / 111 ms | 0.32 cores |
| 100 | 2 | 1.28 Gbit/s | 0.88 cores | 47 MB | 0 | 243 / 329 / 334 ms | 0.61 cores |
| 200 | 2 | 2.54 Gbit/s | 1.67 cores | 57 MB | 0 | 1263 / 2335 / 2384 ms | 0.94 cores |
| 200 | 6 | 2.59 Gbit/s | 2.60 cores | 57 MB | 0 | 152 / 253 / 255 ms | 0.91 cores |
| 400 | 6 | 5.05 Gbit/s | 4.43 cores | 66 MB | 0 | 473 / 841 / 877 ms | 1.54 cores |

A UDP player costs the server four to five times what a TCP one does: every
RTP packet — some 1,100 a second of this stream — is a system call of its
own, where TCP takes a picture's in one. Two threads keep up with a hundred
players; at two hundred they are busy all the time, and the lag spread of a
second and more is players falling behind. Six threads carry two hundred as
two carried a hundred, and four hundred with half a second's spread; there
the server and the players together had most of the machine's cores busy,
so the players on the same machine are part of the figure.

Nothing was lost even falling behind: a player is moved on to a newer
keyframe only once it is further behind than the broadcast keeps — five
seconds by default — and none got that far. [findings.md](findings.md)
says why UDP players lost half of what they were sent before.

## WebRTC

WHEP players of the same stream without B-frames and with Opus sound
(`webrtc-1080p`, 12.2 Mbit/s), on as many server threads as given:

| Players | Server threads | Delivered | Server processor | Server memory | Pictures after a gap | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1 | 601 Mbit/s | 0.76 cores | 143 MB | 9, 9 players | 329 / 1331 / 1331 ms | 0.55 cores |
| 100 | 1 | 829 Mbit/s | 0.98 cores | 262 MB | 3 | 5904 / 6818 / 6849 ms | 0.53 cores |
| 100 | 4 | 1.19 Gbit/s | 2.58 cores | 251 MB | 21, 18 players | 228 / 1266 / 1267 ms | 1.70 cores |
| 200 | 4 | 2.23 Gbit/s | 3.64 cores | 481 MB | 23 | 2123 / 2241 / 2256 ms | 2.02 cores |
| 200 | 6 | 2.38 Gbit/s | 4.23 cores | 477 MB | 30, 28 players | 406 / 2271 / 2496 ms | 2.75 cores |
| 300 | 6 | 2.80 Gbit/s | 5.22 cores | 702 MB | 11 | 5323 / 6928 / 7052 ms | 3.25 cores |

A WebRTC player costs the server 0.015 to 0.02 of a core — six to eight
times an RTSP player over TCP: every packet is encrypted for that player
alone, and `str0m`, which sends it, takes a packet at a time. One thread
keeps up with some 70 players of this stream and then falls behind — a
hundred are all sent everything, five seconds late. Threads scale it: four
carry a hundred as one carries fifty, and six two hundred; at three hundred
the server's five cores and the players' three had the machine's six
physical cores and more, so where a dedicated machine's limit lies was not
measured. Each player takes some 2.4 MB of the server's memory, most of it
`str0m`'s buffers of what it has sent, kept for resending.

Pictures after a gap are packets lost on the loopback, nearly all by
players at the start of a run; `str0m` waits up to thirty pictures — about
1.3 s — for a lost packet before it goes on, which is the p99 lag spread of
the runs that kept up.

## Read together

- For many players, TCP: it costs a quarter of what UDP does, and every
  player — VLC, FFmpeg, a camera viewer — can ask for it.
- `RtspServerOptions::threads` is what UDP players need more of: about one
  thread for every 70 of this stream's players.
- Memory is not what runs out: 71 MB served eight hundred players.
- What runs out first is the network: a gigabit link carries some 75 of
  these players, at well under a tenth of a core.
- A WebRTC player costs most: give `WebRtcServerOptions::threads` one
  thread for every 70 players of a stream like this one, up to the cores
  the machine has to spare.
