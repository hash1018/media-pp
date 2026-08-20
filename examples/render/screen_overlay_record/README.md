# screen_overlay_record

Records the desktop with a live overlay drawn on top, with every pixel staying
on the GPU from the moment it is captured to the moment it is encoded.

`PipeWireScreenCaptureSource` (GPU mode) `-> Queue -> CudaConverter ->
CudaVideoCompositor` (+ `CudaTextLayerHandle`) `-> Queue -> CudaEncoder ->
Mp4Muxer`

The contrast with `screen_record_nvenc` is the point of the graph. That
example records the capture untouched, which needs no conversion at all: the
capture is BGRA and NVENC ingests BGRA directly. The moment anything wants to
*draw* on the capture, that stops being enough — the compositor works in NV12,
like everything else on the CUDA path that is not the encoder — so
`CudaConverter` sits between them. Nothing here comes back to system memory:
the capture is imported as a CUDA surface, converted by a kernel, composited
by a kernel, and encoded by NVENC.

The clock in the corner is redrawn once a second, so the recording proves the
overlay is live rather than a watermark baked in once. Any number of further
layers attach the same way — `add_source` for a video layer, `add_text_layer`
for another caption.

Linux only: it is the GPU screen capture that is Linux-specific here, not the
CUDA half. The Windows shape of the same graph is `DxgiCaptureSource` (GPU
mode) `-> D3d11VideoCompositor -> D3d11NvencEncoder`, with no conversion in
it, since D3D11 composites BGRA directly.

Needs an NVIDIA GPU and an ffmpeg build with NVENC.

```sh
cargo run -p screen_overlay_record -- <output.mp4> [seconds]
```

The compositor's own dialog decides what is captured, so the first run prompts
and prints a restore token that later runs can pass to skip it — the same
arguments `screen_record` documents:

```sh
cargo run -p screen_overlay_record -- <output.mp4> [seconds] [monitor|window] [restore-token]
```
