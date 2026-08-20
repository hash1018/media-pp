# d3d11_scale_render

`FileDemuxer -> D3d11Decoder -> D3d11Scaler (960x540) -> Queue -> Pacer ->
D3d11Renderer`: decodes and resizes video entirely on one shared D3D11 device,
then presents the fixed-size NV12 output in a native window at real playback
speed. Decoded array-texture slices go directly through the D3D11 video
processor, and neither scaling nor rendering maps the pixels to system memory.

The scaler sits before the queue deliberately. Once one synchronous scale
finishes, its decoded input surface can return to FFmpeg's fixed D3D11VA pool;
the queue retains the scaler's independent output textures instead.

```sh
cargo run -p d3d11_scale_render -- path/to/video.mp4
```
