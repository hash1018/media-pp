# sw_decode_render

`FileDemuxer -> SwDecoder -> Queue -> Pacer -> SwScaler -> GPU upload ->
Renderer`: decodes a video file in system memory and presents it in a native
window at real playback speed. Windows uploads to D3D12 and draws into
`D3d12WindowRenderer`'s own window. Linux draws the decoded frames with
`VulkanWindowRenderer`, which uploads them itself, in a window of its own —
through a `SwScaler` only for a stream it cannot draw as it comes. Both
platforms require a video path.

```sh
cargo run -p sw_decode_render -- path/to/video.mp4
```
