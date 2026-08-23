# sw_decode_render

`FileDemuxer -> SwDecoder -> Queue -> Pacer -> SwScaler -> GPU upload ->
Renderer`: decodes a video file in system memory and presents it in a native
window at real playback speed. Windows uploads to D3D12; Linux uploads to CUDA
and presents through Vulkan. Both platforms require a video path.

```sh
cargo run -p sw_decode_render -- path/to/video.mp4
```
