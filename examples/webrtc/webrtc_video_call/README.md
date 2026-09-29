# webrtc_video_call

A two-way video call between two `WebRtcPeer`s in one process, each showing
what the other sent in a window of its own, on Windows, Linux and macOS.

One `Direction::SendRecv` track carries both directions on one connection
(`webrtc_loopback` is the minimal version), so `next_track` hands each side a
`TrackEndpoints::SendRecv`: a sink to encode into and a source to decode from.
Peer-b, which did not originate the track, declares its outbound codec with
`set_source_parameters(&encoder.parameters())`, checked against what was
negotiated.

The two callers differ in where their video comes from:

- peer-a sends `TestVideoSource -> Queue -> SwEncoder -> WebRtcTrackSink`;
- peer-b sends `FileDemuxer -> SwDecoder -> Queue -> Pacer -> SwScaler ->
  Queue -> SwEncoder -> WebRtcTrackSink`, scaled to the 640x480 both windows
  are wired for, and paced so the file plays as a call rather than being sent
  in seconds;
- each side receives `WebRtcTrackSource -> Queue -> SwDecoder -> renderer`
  (`D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux,
  `MetalWindowRenderer` on macOS, where the call runs inside
  `run_with_windows` beside the main thread's event loop), with no
  `Pacer`, since packets arrive at the rate the other side encoded them. Each
  receiver is built once `WebRtcTrackSource::wait_stream_info` has seen the
  codec in actual RTP — for H.264, once SPS and PPS have arrived.

Closing either window, or Escape in it, ends the whole call, and so does the
file running out.

```sh
cargo run -p webrtc_video_call -- path/to/video.mp4
```
