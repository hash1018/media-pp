# d3d11_upload

`TestVideoSource -> Scaler -> D3d11Upload -> Renderer`: a synthetic
`Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU, then uploaded
to a GPU `Pixel::D3D11` texture on the renderer's own `ID3D11Device` before
being presented — proves `D3d11Upload`'s frames (built via plain
`windows-rs` calls + `av_buffer_create`, not FFmpeg's own hwframe pool) are
readable by `D3d11Renderer`'s zero-copy path. Compare against `d3d12_upload`,
the D3D12 sibling of this same smoke test.

```sh
cargo run -p d3d11_upload
```
