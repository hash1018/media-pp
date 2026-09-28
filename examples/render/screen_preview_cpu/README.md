# screen_preview_cpu

Previews desktop capture through the CPU-frame path, without an
encode/decode round trip:

- Windows: `DxgiCaptureSource -> Queue -> D3d12WindowRenderer`, DXGI Desktop
  Duplication in CPU mode.
- Linux: `PipeWireScreenCaptureSource -> Queue -> VulkanWindowRenderer`, the
  xdg-desktop-portal PipeWire CPU path.

On both, the renderer draws the capture's system-memory BGRA as it comes,
uploading and scaling it itself, in a window of its own. The capture includes
the cursor.

No `Pacer`: the capture emits at a constant rate on a drift-free schedule of
its own.

```sh
cargo run -p screen_preview_cpu
```

On Linux the portal chooses the capture target. Select a window instead of the
default monitor with `window`, and pass the printed restore token on later runs:

```sh
cargo run -p screen_preview_cpu -- [monitor|window] [restore-token]
```
