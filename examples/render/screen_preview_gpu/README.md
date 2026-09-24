# screen_preview_gpu

Captures a desktop or application window straight into GPU memory and presents
it, with no pixel ever passing through system memory.

- Windows desktop: `DxgiCaptureSource` (GPU mode) `-> Queue ->
  D3d11WindowRenderer`
- Windows window: `WgcCaptureSource -> Queue -> D3d11WindowRenderer`
- Linux: `PipeWireScreenCaptureSource` (GPU mode) `-> Queue ->
  VulkanWindowRenderer`

On Windows one BGRA-capable `D3d11Gpu` device is created, opens
`D3d11WindowRenderer` and its window, and is then passed into
`DxgiCaptureSource::open_with_device` or `WgcCaptureSource::open_with_device`.
Captured `Pixel::D3D11` BGRA textures therefore reach `D3d11WindowRenderer`
without `Map`, a CPU pixel copy, or a device transfer. DXGI retains its
internal latest-image and independent per-emission GPU copies. WGC copies each
new image once to detach it from the reusable WGC surface; cadence repeats
share that immutable texture and allocate only a new metadata wrapper with a
new PTS.

No `SwScaler`: captured content is already BGRA/RGB, and the renderer
letterboxes any capture size into the preview window. Compare against
`screen_preview_cpu`, which captures to a plain CPU `Pixel::BGRA` frame — and
on Windows converts it to NV12 before the GPU upload.

The Linux graph has the same shape. One CUDA context serves the whole stack:
PipeWire hands over a DMA-BUF that `open_gpu` imports as a BGRA CUDA surface
on it, and `VulkanWindowRenderer`, on a `VulkanGpu` made for that CUDA
device, copies the BGRA into Vulkan memory and draws it as it comes — no
conversion in between.

The Windows DXGI GPU path has no cursor because `CaptureMode::Gpu` doesn't
support cursor compositing yet; the WGC window path requests WGC's cursor
capture. On Linux the compositor draws the cursor itself, so this asks for it.

```powershell
cargo run -p screen_preview_gpu -- dxgi

# Capture another application's top-level window. With no HWND, this lists
# capturable windows and prompts for one; decimal and 0x-prefixed hexadecimal
# HWND values are also accepted directly.
cargo run -p screen_preview_gpu -- wgc

$hwnd = (Get-Process notepad | Select-Object -First 1).MainWindowHandle
cargo run -p screen_preview_gpu -- wgc $hwnd
```

On Linux the compositor's own dialog decides what is captured, so the first
run prompts and prints a restore token that later runs can pass to skip it —
the same arguments `screen_record_software` documents:

```sh
cargo run -p screen_preview_gpu -- [monitor|window] [restore-token]
```
