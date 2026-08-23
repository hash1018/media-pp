# hw_decode_render

`Demux -> hardware decoder -> Queue -> Pacer -> Renderer`: decodes on the GPU
and presents the frames in a native window at real playback speed without
copying decoded pixels back to system memory. Windows uses D3D12VA/D3D12;
Linux uses NVDEC/CUDA with Vulkan presentation. Compare against
`sw_decode_render`, which uses `SwDecoder` and a GPU-upload path instead.

```sh
cargo run -p hw_decode_render -- path/to/video.mp4
```
