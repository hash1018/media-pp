# sw_decode_render

`FileDemuxer -> SwDecoder -> Queue -> Pacer -> Renderer`: decodes a video file
and presents it in a native window at real playback speed, via
`render_common`'s own `D3d12WindowRenderer` (wrapped as a `D3d12Renderer`).
Windows only.

```sh
cargo run -p sw_decode_render -- path/to/video.mp4
```
