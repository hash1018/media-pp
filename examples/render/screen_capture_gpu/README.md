# screen_capture_gpu

`DxgiCaptureSource` (GPU mode) -> `Renderer`: captures the desktop straight
to a GPU-resident `Pixel::D3D11` BGRA texture on the renderer's own
`ID3D11Device` (no `Map`, no CPU pixel copy at all) and presents it directly,
no `Scaler` (desktop content is already BGRA/RGB, no YUV conversion needed,
and `D3d11Renderer` letterboxes any capture size into the window on its
own). Compare against `screen_capture`, which captures to a plain CPU
`Pixel::BGRA` frame instead and converts it to YUV420P for the D3D12
CPU-upload path.

No cursor: `CaptureMode::Gpu` doesn't support cursor compositing yet —
`screen_capture`'s CPU path does.

```sh
cargo run -p screen_capture_gpu
```
