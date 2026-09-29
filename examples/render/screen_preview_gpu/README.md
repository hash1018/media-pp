# screen_preview_gpu

Captures a desktop or application window straight into GPU memory and
presents it, with no pixel passing through system memory.

- Windows desktop: `DxgiCaptureSource` (GPU mode) `-> Queue ->
  D3d11WindowRenderer`
- Windows window: `WgcCaptureSource -> Queue -> D3d11WindowRenderer`
- Linux: `PipeWireScreenCaptureSource` (GPU mode) `-> Queue ->
  VulkanWindowRenderer`
- macOS: `ScreenCaptureKitSource` (VideoToolbox frames) `-> Queue ->
  MetalWindowRenderer`

On Windows one `D3d11Gpu` opens the renderer and its window and is passed to
`DxgiCaptureSource::open_with_device` or `WgcCaptureSource::open_with_device`,
so the captured BGRA textures reach the renderer on the same device. On
Linux one CUDA context serves the whole stack: PipeWire hands over a DMA-BUF
that `open_gpu` imports as a BGRA CUDA surface, and `VulkanWindowRenderer`,
on a `VulkanGpu` made for that CUDA device, draws it as it comes. On macOS
`open_videotoolbox` hands on ScreenCaptureKit's own `IOSurface` pixel
buffers, which `MetalWindowRenderer` draws as they are. No `SwScaler` on
any: the renderer letterboxes any capture size. Compare
`screen_preview_cpu`, which captures to system-memory BGRA that the renderer
uploads itself.

The DXGI GPU path has no cursor (`CaptureMode::Gpu` has no cursor
compositing); the WGC path asks for WGC's cursor, and on Linux the
compositor draws it.

```powershell
cargo run -p screen_preview_gpu              # DXGI, the default
cargo run -p screen_preview_gpu -- wgc       # lists windows and prompts for one
$hwnd = (Get-Process notepad | Select-Object -First 1).MainWindowHandle
cargo run -p screen_preview_gpu -- wgc $hwnd # decimal or 0x-prefixed HWND
```

On Linux the portal's dialog chooses what is captured; the first run prints
a restore token that later runs can pass to skip it:

```sh
cargo run -p screen_preview_gpu -- [monitor|window] [restore-token]
```

On macOS the main display is captured, or with `window` the frontmost
window with a title; macOS asks once for the permission to record the
screen, given in System Settings to the terminal this runs in:

```sh
cargo run -p screen_preview_gpu -- [monitor|window]
```
