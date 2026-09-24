# sw_decode_render

`FileDemuxer -> SwDecoder -> Queue -> Pacer -> VideoWindow`: decodes a video
file in system memory and presents it in a native window at real playback
speed. `VideoWindow` is whichever window renderer the platform has —
`D3d11WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux — with a
GPU of its own; it uploads the decoded frames itself, and a `SwScaler` goes
in front only for a stream it cannot draw as it comes. One program for both
platforms, with no `#[cfg]` of its own. Escape or closing the window stops.

```sh
cargo run -p sw_decode_render -- path/to/video.mp4
```
