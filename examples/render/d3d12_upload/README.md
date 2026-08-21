# d3d12_upload

`TestVideoSource -> SwScaler -> D3d12Upload -> Renderer`: a synthetic
`Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU, then uploaded
to a GPU `Pixel::D3D12` texture on the renderer's own `ID3D12Device` before
being presented — proves `D3d12Upload`'s frames are structurally identical to
`D3d12Decoder`'s own (same `AVD3D12VAFrame` payload), so `D3d12Renderer`
takes its zero-copy path unmodified even though nothing here ever decoded
anything. Compare against `test_video`, which feeds `D3d12Renderer`'s
CPU-upload path directly instead.

```sh
cargo run -p d3d12_upload
```
