# screen_record_overlay

Records the desktop with a live overlay drawn on top, with every pixel staying
on the GPU from the moment it is captured to the moment it is encoded.

Two pipelines: `PipeWireScreenCaptureSource` (GPU mode) `-> Queue ->
CudaConverter ->` a `CudaVideoCompositor` input, and the compositor (with a
`CudaTextLayerHandle`) `-> Queue -> CudaEncoder -> FileMuxer`. At the end the
capture is stopped and the recording finished.

The contrast with `screen_record_nvenc`, which records the capture untouched,
is the point: here something draws on it. The canvas is NV12, and converting
the BGRA capture up front lets it be copied in rather than blended as BGRA.
Nothing comes back to system memory: the capture is imported as a CUDA
surface, converted and composited by kernels, and encoded by NVENC.

The clock in the corner is redrawn once a second, so the recording proves the
overlay is live rather than a watermark baked in once. Any number of further
layers attach the same way — `add_source` for a video layer, `add_text_layer`
for another caption.

Linux only: it is the GPU screen capture that is Linux-specific here, not the
CUDA half. The Windows shape of the same graph is `DxgiCaptureSource` (GPU
mode) `-> D3d11VideoCompositor -> D3d11VideoEncoder`, with no conversion in
it, since D3D11 composites BGRA directly.

Needs an NVIDIA GPU and an ffmpeg build with NVENC.

```sh
cargo run -p screen_record_overlay -- <output.mp4> [seconds]
```

The compositor's own dialog decides what is captured, so the first run prompts
and prints a restore token that later runs can pass to skip it — the same
arguments `screen_record_software` documents:

```sh
cargo run -p screen_record_overlay -- <output.mp4> [seconds] [monitor|window] [restore-token]
```
