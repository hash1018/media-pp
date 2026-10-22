# What measuring the servers changed

## A WebRTC server sent a packet a turn of its loop

The first WebRTC run kept up with fifty players on one thread, but its
memory grew without bound past that — 3 GB at a hundred — and a picture
larger than a few dozen packets could arrive whole seconds late. Three things
in how the server drove `str0m`:

- **What was written waited for the next turn.** `str0m` sends what is
  written only once it has been told the time has moved on; the server told
  it at the start of each turn, before writing, so everything written waited
  a turn — and, with the thread busy, piled up in `str0m`'s queue, which has
  no bound. The server now tells it after writing.
- **One packet a turn.** `str0m`'s pacer lets one packet out and then asks
  for the time again, with a deadline already past. The server took any
  deadline as the end of the turn, so a turn sent one packet. It now answers
  a deadline that has come and goes on sending. `WebRtcPeer` had the same
  loop, which held a peer to about a thousand packets a second, and is fixed
  the same way.
- **Pictures written over.** `str0m` keeps what it is to send in a ring of a
  thousand packets a video track, and the server wrote sixteen pictures
  before sending any: a few large ones together were written over before
  they left. It now sends each picture before writing the next.

With the three, memory stays at some 2.4 MB a player, and a test pins the
throughput: without the second, two seconds of 100 kB pictures take ten
seconds to catch up on. The server's threads were added then
(`WebRtcServerOptions::threads`), each a socket and a loop of its own, since
one thread is one core.

## Players arriving at once were refused

At three hundred players arriving five milliseconds apart, some were
refused a connection: the WHEP endpoint answered a request at a time,
waiting for a server thread's answer to each, and the rest overflowed the
listener's backlog. It now answers each on a thread of its own, up to 512 at
once.



## UDP players lost half of what they were sent

The first run over UDP lost 47% of the RTP packets at a hundred players,
and 52% at four hundred, while over TCP nothing was lost. Two things lost
them.

**The group of pictures a player joins with, sent at once.** A player
starts at the newest keyframe the broadcast holds, and what came after it
— up to a whole group of pictures, ten seconds of it for the stream that
was first measured — was sent the moment the player asked. Over TCP the
connection paces that: the server's writes wait while the player reads.
UDP has nothing that does, so seconds of video arrived in a burst, far
more than a player's socket holds, and most of it was dropped there.

A UDP player is now sent what the broadcast held when it joined at twice
its speed (`UDP_CATCH_UP`), so it catches up in as long as it was behind,
and is then sent each packet as it is written. It starts at once, at the
keyframe, where that keyframe is no more than a second behind; further
behind, it waits for the next one.

**Sockets the size of a fraction of a keyframe.** Every UDP player's
packets leave through the server's one RTP socket, and on Windows a socket
buffers 64 kB unless asked for more — a single keyframe of a 1080p stream
is several times that. The server's two UDP sockets now ask for 8 MB
(through `socket2`, which tokio already brings), and `rtsp_load`'s players
ask the same of theirs, as FFmpeg's own RTSP client does.

With both, no run loses anything. What is left is the cost of a system
call per packet, which [the results](windows-i5-12400f.md#udp) measure.

## The keyframe interval was measured as the server's lag

The first stream had a keyframe every 10.4 seconds. A player that joins
starts at the newest one, so it began up to ten seconds behind and spent
the run catching up, and the lag spread — taken after a player's first
three seconds — measured that rather than the server. The stream is now
re-encoded with a keyframe every two seconds, as a live encoder sends one
(see [method.md](method.md#the-stream)).

## B-frames were measured as jitter

A picture's lag was its arrival less its own timestamp. Pictures are sent
in decoding order, and with B-frames a picture is sent before ones shown
ahead of it, so the lag went up and down by the reordering — 125 ms at two
B-frames — on a single player with nothing else running. The lag is now
taken against the newest timestamp sent before a picture, which never goes
back.

## Sending what a UDP player is sent

A UDP player's packets were each sent with `send_to().await`; they are now
tried at once with `try_send_to` and awaited only when the socket's buffer
is full. That took a fifth off the server's processor time at a hundred
players of the first stream and two threads (0.99 to 0.81 cores); the
system call itself is what remains.

## Sender reports were counted as lost packets

Measured again after UDP players began to start no more than a second
behind, two of the six runs of two hundred UDP players each had two
players that lost some twelve thousand packets — ten seconds of the
stream — while every player received the stream's whole rate. Nothing was
missing: `rtsp_load` let go of each track's RTCP port as soon as it had
asked for it, the system gave that port to another player as its RTP port,
and the server's sender reports to the first player arrived there. Read as
RTP, a report's length field is a sequence number far from the expected
one, and the next real packet after it counted as everything in between
lost. The players now hold both ports for as long as they play, and no run
loses a packet; the losses the first measurement gave at five and eight
hundred players, past the ceiling, were most likely the same. The RTSP
server's own tests had the same fault, and hold both ports since b6e0e00.
