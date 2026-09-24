# hw_decode_render

`Demux -> VideoDecodeBin -> Queue -> Pacer -> Renderer`: decodes on the GPU
where it can, and in software onto the same device where it cannot, and
presents the frames in a native window at real playback speed.

- Windows decodes onto D3D12 — D3D12VA, or `SwDecoder` and an upload — and
  draws into `D3d12WindowRenderer`'s own window.
- Linux decodes onto CUDA — NVDEC, or `SwDecoder` and an upload — and draws
  into `VulkanWindowRenderer`'s own window. Before linking,
  `contract::check_elements` asks whether the bin's output fits the
  renderer's input; where it does not — the renderer draws NV12 and BGRA,
  and the bin may hand on P010, or a layout it cannot name where the stream
  does not say — a `CudaConverter` to NV12 goes between them:
  `Demux -> VideoDecodeBin -> CudaConverter -> Queue -> Pacer -> Renderer`.

Which way the bin decodes, and why where it is software, is printed when it
is opened (`decoding: ...`) and again when playback ends (`decoded: ...`),
which differs if the GPU refused the stream part way and the bin went on in
software. Compare against `sw_decode_render`, which always decodes on the CPU.

```sh
cargo run -p hw_decode_render -- path/to/video.mp4
```
