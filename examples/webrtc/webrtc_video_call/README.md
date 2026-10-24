# webrtc_video_call

A two-way video call between two `WebRtcPeer`s in one process, each showing
what the other sent in a window of its own, on Windows, Linux and macOS.

Both directions share one connection, a track each way (`webrtc_loopback` is
the minimal version): each side sends through a `WebRtcMuxer` its handle makes,
whose stream is its encoder's — the payload type and the SPS/PPS together —
and opens it in a renegotiation of its own; the other's track comes out of its
handle's `next_source` as a `WebRtcSource`.

The two callers differ in where their video comes from:

- peer-a sends `TestVideoSource -> Queue -> SwEncoder -> WebRtcMuxer`;
- peer-b sends `FileDemuxer -> SwDecoder -> Queue -> Pacer -> SwScaler ->
  Queue -> SwEncoder -> WebRtcMuxer`, scaled to the 640x480 both windows
  are wired for, and paced so the file plays as a call rather than being sent
  in seconds;
- each side receives `WebRtcSource -> Queue -> SwDecoder -> renderer`
  (`D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux,
  `MetalWindowRenderer` on macOS, where the call runs inside
  `run_with_windows` beside the main thread's event loop), with no
  `Pacer`, since packets arrive at the rate the other side encoded them. Each
  receiver is built once its source has seen the codec in actual RTP — for
  H.264, once SPS and PPS have arrived.

Closing either window, or Escape in it, ends the whole call, and so does the
file running out.

```sh
cargo run -p webrtc_video_call -- path/to/video.mp4
```
