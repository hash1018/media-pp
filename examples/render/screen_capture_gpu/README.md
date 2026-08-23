# screen_capture_gpu

Captures the desktop straight into GPU memory and presents it, with no pixel
ever passing through system memory.

- Windows: `DxgiCaptureSource` (GPU mode) `-> Queue -> D3d11Renderer`
- Linux: `PipeWireScreenCaptureSource` (GPU mode) `-> Queue -> CudaConverter
  -> CudaRenderer`

On Windows the capture is a `Pixel::D3D11` BGRA texture on the renderer's own
`ID3D11Device` — no `Map`, no CPU pixel copy at all — and `D3d11Renderer`
presents it directly. No `SwScaler`: desktop content is already BGRA/RGB, and
the renderer letterboxes any capture size into the window on its own. Compare
against `screen_capture`, which captures to a plain CPU `Pixel::BGRA` frame
and converts it to NV12 for a `D3d12Upload`.

The Linux graph is one element longer, and the platform forces exactly that
one. PipeWire hands over a DMA-BUF that `open_gpu` imports as a BGRA CUDA
surface, and `CudaRenderer` presents NV12, so `CudaConverter` sits between
them. That element exists for this shape: without it a GPU capture can only be
encoded — NVENC ingests BGRA directly, which is what `screen_record_nvenc`
does — never shown or composited.

No cursor on Windows: `CaptureMode::Gpu` doesn't support cursor compositing
yet, while `screen_capture`'s CPU path does. On Linux the compositor draws the
cursor itself, so this asks for it.

```sh
cargo run -p screen_capture_gpu
```

On Linux the compositor's own dialog decides what is captured, so the first
run prompts and prints a restore token that later runs can pass to skip it —
the same arguments `screen_record` documents:

```sh
cargo run -p screen_capture_gpu -- [monitor|window] [restore-token]
```
