# hw_decode_render

`Demux -> D3d12Decoder -> Queue -> Pacer -> Renderer`: decodes on the GPU
via D3D12VA hardware acceleration and presents the frames in a native window
at real playback speed, without ever copying the decoded pixels back to
system memory — `D3d12Renderer` draws straight from the decoder's own D3D12
texture. Compare against `sw_decode_render`, which uses `SwDecoder` (CPU
decode) and a CPU-upload submit path instead.

```sh
cargo run -p hw_decode_render -- path/to/video.mp4
```
