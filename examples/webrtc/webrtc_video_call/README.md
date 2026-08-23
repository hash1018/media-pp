# webrtc_video_call

A two-way video call between two `WebRtcPeer`s in one process, each presenting
what the *other* peer sent into its own window. It runs on Windows and Linux.

One `Direction::SendRecv` track carries both directions on a single connection
(`webrtc_loopback` is the minimal version of that), so `WebRtcHandle::next_track`
hands each side a `TrackEndpoints::SendRecv` — a `WebRtcTrackSink` to encode
into and a `WebRtcTrackSource` to decode from. There is no second `add_track`
for the return direction. Peer-b did not originate that track, so it declares
`Codec::H264` on its outbound sink before sending. Both endpoints expose the
codec families retained by SDP negotiation immediately: the source list says
what may arrive, while the sink list says what peer-b may send. The source's
`codec()` remains `None` until RTP arrives and then reports what peer-a actually
sent. Peer-b selects H.264 from the sink list because its own encoder produces
H.264; the inbound selection does not force the reverse direction to match.

The two callers deliberately differ in where their video comes from, so the
call has a generated stream going one way and a real file the other:

- peer-a sends `TestVideoSource -> Queue -> SwEncoder -> WebRtcTrackSink`.
- peer-b sends `FileDemuxer -> SwDecoder -> Queue -> Pacer -> SwScaler ->
  Queue -> SwEncoder -> WebRtcTrackSink`. `SwScaler` brings the file to the
  fixed 640x480 both renderers are wired up at, and `Pacer` holds it to
  playback speed — without it the whole file would be encoded and sent in
  seconds rather than played as a call.
- Windows receives `WebRtcTrackSource -> Queue -> SwDecoder -> D3d12Renderer`.
  Linux receives `WebRtcTrackSource -> Queue -> SwDecoder -> SwScaler(NV12) ->
  CudaUpload -> CudaRenderer(Vulkan)`: the Linux window bridge consumes CUDA
  NV12 frames, so the decoded system-memory frame needs that conversion and
  upload. Neither receive path needs a `Pacer`: these packets arrive at the
  rate the other side encoded them, so the timeline is already real. The send
  pipelines start first, then each
  receiver calls `WebRtcTrackSource::wait_stream_info` with a two-second
  timeout. That reports the codec from the first actual RTP payload, not from
  the other peer's encoder object or merely from the SDP capability list. The
  returned `WebRtcStreamInfo` creates the minimal FFmpeg decoder parameters,
  and the first packet remains in the source queue while that H.264 decoder
  pipeline is built. No cross-peer encoder-parameter wiring or example-local
  FFmpeg FFI is needed.

Both windows come from one `render_common::run_windows` call and share one
worker thread. They also share one `D3d12GpuContext` on Windows, or one CUDA
device and `VulkanGpuContext` on Linux. Closing either window ends the whole
call; the file side also ends on its own when the file runs out.

```sh
cargo run -p webrtc_video_call -- path/to/video.mp4
```
