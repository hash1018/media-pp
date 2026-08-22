# webrtc_loopback

Two `WebRtcPeer`s, connected over real loopback UDP — no browser, no
signaling server. One `Direction::SendRecv` track, opened by
`WebRtcHandle::add_track` on peer-a, carries data *both* ways: peer-a pushes
into the `WebRtcTrackSink` its own `next_track()` returned for the track it
just added (str0m never fires `Event::MediaAdded` for a track a side added
itself), and peer-b pushes back on the exact same `Mid` via the
`WebRtcTrackSink` its own `next_track()` returned for the incoming
`Event::MediaAdded` — no second `add_track`/renegotiation needed for the
reverse direction. Because peer-b did not originate the track, it calls
`WebRtcTrackSink::set_codec(Codec::Vp8)` before replying. Both endpoints expose
the codec families retained by SDP negotiation at attachment time: the source
list says what may arrive, while the sink list says what peer-b may send.
`set_codec` validates peer-b's encoder choice against the latter; the source's
`codec()` separately reports the codec actually observed in incoming RTP.
Each side's inbound track is wired as its own
`WebRtcTrackSource -> CountingSink` `Pipeline` — a `WebRtcPeer` connection
isn't one pipeline node, it mints one source per track.

The initial connection (ICE candidates + a bootstrap data channel, just to
get DTLS established with zero media) is done directly against str0m,
exactly like a real caller would before ever touching `WebRtcPeer`. In a
real app, the offer/answer exchanged here would travel over your own
signaling transport (HTTP, WebSocket, ...) instead of a direct function
call.

```sh
cargo run -p webrtc_loopback
```
