# detect

Demux -> SwDecoder -> Scaler (640x640 RGB24) -> OrtDetector, drawing every
detection straight onto the same 640x640 frame `OrtDetector` saw and
presenting it in a plain window. Deliberately not DX12: `D3d12Renderer` has
no hook for drawing an overlay, so this blits pixels straight into a `winit`
window via `softbuffer` instead — no GPU, no `renderer-engine`.

No `Pacer` in this pipeline, so frames show up as fast as decode + inference
allow, not at real playback speed.

```sh
cargo run -p detect -- path/to/model.onnx path/to/video.mp4
```
