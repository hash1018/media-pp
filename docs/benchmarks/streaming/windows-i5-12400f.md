# Streaming on Windows, Core i5-12400F

## The machine

| | |
|---|---|
| Processor | Intel Core i5-12400F, 6 cores, 12 threads |
| Memory | 32 GB |
| System | Windows 11 Pro 26200 |
| Build | release, media-pp at the commit updating this page: a UDP or WebRTC player starts no more than a second behind the newest picture |
| Stream | [method.md](method.md#the-stream)'s: 1080p24 H.264 at 12.3 Mbit/s, a keyframe every 2 s, two B-frames, AAC at 128 kbit/s |
| Players | `rtsp_load` and `webrtc_load` on the same machine, over the loopback; 30 s a run |

Every figure up to the ceilings is the range over three rounds of the same
runs, one after another (see [method.md](method.md#rounds-and-ceilings)):
`0.49–0.58` is the least and the most of the three. The figures before the
second-long start were measured at d95111c; this page's history has them.

## TCP

Players asking for RTP interleaved on their RTSP connections, the server on
its default two threads:

| Players | Delivered | Server processor | Server memory | Lost | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13 Mbit/s | 0.08–0.10 cores | 39–40 MB | 0 | 20–21 / 20–21 / 20–21 ms | 0.01 cores |
| 50 | 637–644 Mbit/s | 0.40–0.41 cores | 45 MB | 0 | 29–40 / 33–43 / 33–43 ms | 0.12–0.13 cores |
| 100 | 1.29–1.30 Gbit/s | 0.49–0.58 cores | 48–49 MB | 0 | 29–32 / 33–43 / 36–44 ms | 0.15–0.19 cores |
| 200 | 2.56–2.58 Gbit/s | 0.62–0.76 cores | 53–54 MB | 0 | 39–42 / 45–60 / 46–62 ms | 0.19–0.24 cores |
| 300 | 3.82–3.87 Gbit/s | 0.77–0.85 cores | 57–58 MB | 0 | 49–64 / 71–93 / 72–94 ms | 0.22–0.26 cores |
| 400 | 4.99–5.01 Gbit/s | 0.89–0.95 cores | 62–63 MB | 0 | 63–72 / 86–140 / 91–143 ms | 0.29–0.30 cores |
| 600 | 7.39–7.40 Gbit/s | 1.24–1.32 cores | 71–72 MB | 0 | 104–107 / 168–205 / 193–207 ms | 0.38–0.43 cores |
| 800 | 9.78–9.80 Gbit/s | 1.58–1.63 cores | 81–82 MB | 0 | 171–180 / 255–313 / 261–365 ms | 0.52–0.53 cores |

From a hundred players up each costs the server about the same, 0.16 to
0.19 of a core per Gbit/s — some 0.002 cores for a 12 Mbit/s player: a
picture's RTP packets go to a TCP player in one write, so the server makes
24 system calls a second for each. Below a hundred these rounds measured
more than that slope gives — 0.40 cores at fifty, 0.08 at one, against
0.15–0.29 and 0.03–0.05 the time before with nothing in the server's TCP
path changed since — which is the machine's other work, not the players.
Memory grows by about 55 kB a player; the packets every player is sent are
the broadcast's, shared, and each player's own are only the RTP cut from
them.

The lag spread grows with the players — the server's threads take them in
turn, and the eight-hundredth waits for the others — but it stops growing:
at 800 the largest is a third of a second, and no player lost a packet.
TCP players start at the newest keyframe whatever its age, as before: the
connection paces what they catch up on.

### Where it stops keeping up

Past eight hundred, once each (one round):

| Players | Server threads | Delivered | Server processor | Server memory | Lowest a player got | Lag spread, median / p99 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 2 | 11.8 Gbit/s | 1.86 cores | 84 MB | 10.4 Mbit/s | 238 / 388 / 436 ms |
| 1200 | 2 | 13.2 Gbit/s | 1.87 cores | 96 MB | 8.3 Mbit/s | 2804 / 4464 / 5716 ms |
| 1600 | 2 | 12.7 Gbit/s | 1.91 cores | 116 MB | 0.8 Mbit/s | 5059 / 7265 / 9496 ms |
| 2000 | 2 | 13.3 Gbit/s | 1.89 cores | 141 MB | 0.9 Mbit/s | 4574 / 7767 / 9164 ms |
| 1000 | 6 | 11.9 Gbit/s | 3.75 cores | 91 MB | 10.4 Mbit/s | 122 / 213 / 215 ms |
| 1200 | 6 | 14.1 Gbit/s | 4.14 cores | 105 MB | 10.4 Mbit/s | 148 / 300 / 313 ms |
| 1600 | 6 | 17.8 Gbit/s | 4.55 cores | 125 MB | 9.1 Mbit/s | 240 / 421 / 475 ms |

Two threads carry a thousand players and then are full — 1.9 cores is two
threads busy all the time — and what they deliver stays at some 13 Gbit/s
however many more ask: each player gets less, and falls behind. Six
threads carry twelve hundred as two carried a thousand. At sixteen hundred
they deliver nearly 18 Gbit/s — 11.1 Mbit/s a player of the stream's
12.3 — and have begun to fall behind, with the server's four and a half
cores and the players' one and a half taking the whole of the machine's
six; a dedicated machine's ceiling is higher than this, by how much was not
measured. No player lost a packet in any of these runs: a TCP player that
falls behind is slowed, not cut.

## UDP

Players asking for RTP on UDP ports of their own:

| Players | Server threads | Delivered | Server processor | Server memory | Lost | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2 | 624–631 Mbit/s | 0.45–0.55 cores | 42 MB | 0 | 61–83 / 79–145 / 79–145 ms | 0.49–0.56 cores |
| 100 | 2 | 1.29–1.30 Gbit/s | 0.87–0.94 cores | 46–47 MB | 0 | 174–227 / 290–360 / 294–360 ms | 0.68–0.77 cores |
| 200 | 2 | 2.48–2.49 Gbit/s | 1.53–1.61 cores | 51–52 MB | 0 | 433–503 / 690–729 / 705–747 ms | 0.81–0.99 cores |
| 200 | 6 | 2.44–2.45 Gbit/s | 2.74–2.94 cores | 51–52 MB | 0 | 120–139 / 239–250 / 241–255 ms | 1.28–1.53 cores |
| 400 | 6 | 4.94–4.95 Gbit/s | 4.28–4.38 cores | 63–64 MB | 0 | 395–449 / 769–878 / 783–941 ms | 1.62–1.81 cores |

A UDP player costs the server four to five times what a TCP one does: every
RTP packet — some 1,100 a second of this stream — is a system call of its
own, where TCP takes a picture's in one. Two threads keep up with a hundred
players, and at two hundred are busy all the time. Six threads carry two
hundred as two carried a hundred, and four hundred with under a second's
spread.

What the second-long start changed is the spread at two hundred players on
two threads: under three quarters of a second, where it was 1.4 to 2.4
seconds. A player used to start at the newest keyframe the broadcast held,
up to a whole group of pictures — two seconds — behind, and catch up at
twice the stream's speed while the threads were already full; it now waits
for the next keyframe where the newest is more than a second old, so there
is less to catch up on. Where the threads keep up, the figures are the
ones before.

### Where it stops keeping up

| Players | Server threads | Delivered | Server processor | Server memory | Lowest a player got | Lag spread, median / p99 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 6 | 5.98 Gbit/s | 4.94 cores | 72 MB | 11.2 Mbit/s | 564 / 1018 / 1072 ms |
| 600 | 6 | 6.43 Gbit/s | 5.35 cores | 104 MB | 6.6 Mbit/s | 4525 / 6264 / 6447 ms |
| 800 | 6 | 6.08 Gbit/s | 5.14 cores | 107 MB | 3.4 Mbit/s | 5368 / 7121 / 7692 ms |

Six threads deliver some 6 to 6.4 Gbit/s over UDP and no more — five and a
third cores, beside the players' two — so five hundred players of this
stream is about what they carry, and six hundred past it. No player lost a
packet in these runs either. The losses this page once gave at five and
eight hundred — a player or two each — were the load tool's: it let go of
each track's RTCP port, and the server's sender reports to one player
arrived at another's RTP port and read as a jump in its sequence numbers
(see [findings.md](findings.md#sender-reports-were-counted-as-lost-packets)).

## WebRTC

WHEP players of the same stream without B-frames and with Opus sound
(`webrtc-1080p`, 12.2 Mbit/s), on as many server threads as given:

| Players | Server threads | Delivered | Server processor | Server memory | Pictures after a gap | Lag spread, median / p99 / max | Players' processor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1 | 592–595 Mbit/s | 0.59–0.62 cores | 145–146 MB | 1–3 | 242–273 / 1222–1235 / 1222–1235 ms | 0.46–0.48 cores |
| 100 | 1 | 707–737 Mbit/s | 0.73–0.75 cores | 259–261 MB | 5–16 | 1911–2047 / 2872–3539 / 3044–3543 ms | 0.59–0.61 cores |
| 100 | 4 | 1.19 Gbit/s | 2.20–2.28 cores | 262–263 MB | 7–21 | 205–246 / 1234–1258 / 1276–1303 ms | 1.50–1.57 cores |
| 200 | 4 | 1.97–2.20 Gbit/s | 3.03–3.32 cores | 476 MB | 10–23 | 411–591 / 1778–4711 / 1782–5075 ms | 2.03–2.19 cores |
| 200 | 6 | 2.37–2.38 Gbit/s | 3.99–4.17 cores | 470–478 MB | 14–30 | 354–406 / 1279–1295 / 1369–1540 ms | 2.63–2.73 cores |
| 300 | 6 | 2.63–2.87 Gbit/s | 4.70–4.96 cores | 680–689 MB | 35–62 | 1228–2857 / 5352–5663 / 5596–7514 ms | 3.11–3.16 cores |

and once, between those:

| Players | Server threads | Delivered | Server processor | Server memory | Pictures after a gap | Lag spread, median / p99 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 6 | 1.78 Gbit/s | 3.54 cores | 370 MB | 20 | 268 / 1309 / 1313 ms |
| 250 | 6 | 2.73 Gbit/s | 4.64 cores | 576 MB | 39 | 658 / 3603 / 4029 ms |

A WebRTC player costs the server 0.015 to 0.02 of a core — six to eight
times an RTSP player over TCP: every packet is encrypted for that player
alone, and `str0m`, which sends it, takes a packet at a time. One thread
keeps up with some 70 players of this stream and then falls behind: at a
hundred it sends 0.7 Gbit/s of the 1.2 they need, and the slowest player
gets 4.7 Mbit/s. Threads scale it: four carry a hundred as one carries
fifty, and six two hundred. At two hundred and fifty, six threads fall
behind, with the server's five cores and the players' three past the
machine's six physical ones; on a machine of its own a server would carry
more. Each player takes some 2.4 MB of the server's memory, most of it
`str0m`'s buffers of what it has sent, kept for resending.

Where the threads keep up, the second-long start changed nothing
measurable. Where they do not, the spread is smaller than before — a
median of two seconds at a hundred players on one thread, where it was
four — and less is delivered: a player starts nearer the newest picture
and has less to be sent at once, and a thread that is behind sends what
it can of the rest.

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
| 400 | TCP | 2 | 1.00 cores | 63 MB | 0 | 68 / 96 / 100 ms | 0.89–0.95 cores, 62–63 MB |
| 100 | WebRTC | 4 | 2.37 cores | 271 MB | 12 pictures | 215 / 1251 / 1286 ms | 2.20–2.28 cores, 262–263 MB |

Recording costs a tenth of a core at most: it is one reader more of the
broadcast, writing what it reads to a file without decoding it. Each
recording was 48 seconds — the server's whole run — of 1,152 pictures,
every one of the stream's 24 a second; FFmpeg read each back from its
playlist without an error, and both ended with `#EXT-X-ENDLIST` once the
server stopped.

## Read together

- For many players, TCP: it costs a quarter of what UDP does, and every
  player — VLC, FFmpeg, a camera viewer — can ask for it. Two threads
  carry a thousand players of a 12 Mbit/s stream, six at least twelve
  hundred.
- `RtspServerOptions::threads` is what UDP players need more of: about one
  thread for every 70 to 100 of this stream's players, and six carry five
  hundred and stop at some 6 Gbit/s.
- Memory is not what runs out: 82 MB served eight hundred players over
  TCP.
- What runs out first is the network: a gigabit link carries some 75 of
  these players, at well under a tenth of a core.
- A WebRTC player costs most: give `WebRtcServerOptions::threads` one
  thread for every 70 players of a stream like this one, up to the cores
  the machine has to spare.
- Recording what is served costs a reader, not a player's worth.
