# sw_decode_render

`FileDemuxer -> SwDecoder -> Queue -> Pacer -> Renderer`: decodes a video file
in system memory and presents it in a native window at real playback speed.
The renderer — `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on
Linux — uploads the decoded frames itself, in a window of its own, through a
`SwScaler` only for a stream it cannot draw as it comes. Both platforms require
a video path.

```sh
cargo run -p sw_decode_render -- path/to/video.mp4
```
