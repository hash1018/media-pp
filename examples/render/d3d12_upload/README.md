# d3d12_upload

`TestVideoSource -> Queue -> SwScaler -> Queue -> D3d12Upload -> Queue ->
D3d12WindowRenderer`: a synthetic `Pixel::YUV420P` stream converted to
`Pixel::NV12` on the CPU, then uploaded to a GPU `Pixel::D3D12` texture on the
renderer's own `ID3D12Device` before being presented — proves `D3d12Upload`'s
frames are structurally identical to `D3d12Decoder`'s own (same
`AVD3D12VAFrame` payload), so the renderer takes its zero-copy path unmodified
even though nothing here ever decoded anything. Every stage sits behind its
own `Queue` so each one is exercised on a separate thread; `test_video` runs
the same conversion and upload as a single-thread tail instead, to show
`TestVideoSource` pacing itself without a `Pacer`.

The window is the renderer's own: `D3d12WindowRenderer::open` opens it on a
thread of its own, the way a GStreamer video sink does, and reports what
happens to it — Space pauses and resumes, Escape or closing the window stops.
The one device every element shares is a `D3d12Gpu`.

```sh
cargo run -p d3d12_upload
```
