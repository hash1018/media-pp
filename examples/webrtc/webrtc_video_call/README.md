# webrtc_video_call

A two-way video call between two `WebRtcPeer`s in one process, each presenting
what the *other* peer sent into its own window. Windows only.

One `Direction::SendRecv` track carries both directions on a single connection
(`webrtc_loopback` is the minimal version of that), so `WebRtcHandle::next_track`
hands each side a `TrackEndpoints::SendRecv` — a `WebRtcTrackSink` to encode
into and a `WebRtcTrackSource` to decode from. There is no second `add_track`
for the return direction.

The two callers deliberately differ in where their video comes from, so the
call has a generated stream going one way and a real file the other:

- peer-a sends `TestVideoSource -> Queue -> SwEncoder -> WebRtcTrackSink`.
- peer-b sends `FileDemuxer -> SwDecoder -> Queue -> Pacer -> SwScaler ->
  Queue -> SwEncoder -> WebRtcTrackSink`. `SwScaler` brings the file to the
  fixed 640x480 both renderers are wired up at, and `Pacer` holds it to
  playback speed — without it the whole file would be encoded and sent in
  seconds rather than played as a call.
- Both receive `WebRtcTrackSource -> Queue -> SwDecoder -> D3d12Renderer`. No
  `Pacer` here: these packets arrive at the rate the other side encoded them,
  so the timeline is already real.

Both windows come from one `render_common::run_windows` call, sharing one
worker thread and one `D3d12GpuContext`. Closing either window ends the whole
call; the file side also ends on its own when the file runs out.

```sh
cargo run -p webrtc_video_call -- path/to/video.mp4
```
