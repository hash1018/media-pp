# screen_capture

`DxgiCaptureSource -> Scaler -> Renderer`: captures the desktop live via DXGI
Desktop Duplication (cursor included) at a constant frame rate
(`DxgiCaptureOptions::fps`) and converts/resizes it to the window's own size
as `Pixel::YUV420P` before rendering — no `SwEncoder`/`SwDecoder` round trip.

No `Pacer` here, confirmed unneeded: `DxgiCaptureSource` previously emitted
variable-rate (real wall-clock pts, push-on-change), and removing `Pacer`
against that measurably caused judder. It's since been rewritten to emit at
a constant rate on a drift-free absolute schedule instead — the same pattern
`TestVideoSource` uses — and with that fixed, `Scaler` sitting between source
and renderer here doesn't add enough jitter on its own to bring the judder
back. The constant-rate/drift-free change was the actual fix, not the
presence of a `Pacer` stage.

```sh
cargo run -p screen_capture
```
