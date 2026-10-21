# What measuring the server changed

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
and is then sent each packet as it is written. It still starts at once,
at the keyframe.

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
