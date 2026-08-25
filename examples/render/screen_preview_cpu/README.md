# screen_preview_cpu

`CaptureSource -> Queue -> SwScaler(NV12) -> Queue -> GPU upload -> Renderer`:
captures the desktop into system memory, converts/resizes it to window-sized
NV12, uploads it to the GPU, and presents it without an encode/decode round
trip. Windows uses DXGI capture and D3D12 upload/rendering. Linux uses the
xdg-desktop-portal PipeWire CPU path, CUDA upload, and Vulkan presentation.

No `Pacer` here, confirmed unneeded: `DxgiCaptureSource` previously emitted
variable-rate (real wall-clock pts, push-on-change), and removing `Pacer`
against that measurably caused judder. It's since been rewritten to emit at
a constant rate on a drift-free absolute schedule instead — the same pattern
`TestVideoSource` uses — and with that fixed, `SwScaler` sitting between source
and renderer here doesn't add enough jitter on its own to bring the judder
back. The constant-rate/drift-free change was the actual fix, not the
presence of a `Pacer` stage.

```sh
cargo run -p screen_preview_cpu
```

On Linux the portal chooses the capture target. Select a window instead of the
default monitor with `window`, and pass the printed restore token on later runs:

```sh
cargo run -p screen_preview_cpu -- [monitor|window] [restore-token]
```
