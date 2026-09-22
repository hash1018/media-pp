# hw_decode_render

`Demux -> VideoDecodeBin -> Queue -> Pacer -> Renderer`: decodes on the GPU
where it can, and in software onto the same device where it cannot, and
presents the frames in a native window at real playback speed.

- Windows decodes onto D3D12: D3D12VA, or `SwDecoder` and an upload.
- Linux decodes onto CUDA — NVDEC, or `SwDecoder` and an upload — with
  Vulkan presentation. Before linking, `contract::check_elements` asks
  whether the bin's output fits the renderer's input; where it does not —
  the bin hands on BGRA for alpha, an odd side, BT.2020 or HDR colour, and
  the renderer presents NV12 — a `CudaConverter` goes between them:
  `Demux -> VideoDecodeBin -> CudaConverter -> Queue -> Pacer -> Renderer`.

Which way the bin decodes, and why where it is software, is printed when it
is opened (`decoding: ...`) and again when playback ends (`decoded: ...`),
which differs if the GPU refused the stream part way and the bin went on in
software. Compare against `sw_decode_render`, which always decodes on the CPU.

```sh
cargo run -p hw_decode_render -- path/to/video.mp4
```
