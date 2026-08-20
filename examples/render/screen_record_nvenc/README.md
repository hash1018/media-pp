# screen_record_nvenc

`capture -> NVENC -> Mp4Muxer`: records the desktop into a playable `.mp4`
with no CPU color conversion anywhere in the graph.

- Windows: `DxgiCaptureSource` (GPU mode) `-> D3d11NvencEncoder -> Mp4Muxer`
- Linux: `PipeWireScreenCaptureSource` (GPU mode) `-> CudaEncoder -> Mp4Muxer`

The contrast with `screen_record` is the whole point. That example runs
`capture -> SwScaler -> SwEncoder`: every frame is converted BGRA->YUV420P by
libswscale and encoded on the CPU. Here NVENC consumes the captured BGRA
directly — `D3d11NvencInputFormat::Bgra` / `CudaFrameFormat::Bgra`, since NVENC
does its own color conversion inside the encode block — so there is no
`SwScaler` in this graph at all, and only the compressed packets come back.
Recording six seconds of a 1920x1080 desktop at 30fps costs about a third of
the CPU time the software path does.

The pixels start GPU-resident on both platforms. DXGI hands over a texture
under `CaptureMode::Gpu`; `PipeWireScreenCaptureSource::open_gpu` negotiates
DMA-BUF and imports each captured buffer into a CUDA BGRA surface, so the
Linux branch needs no `CudaUpload` and the two graphs have the same elements.
Nothing in either branch copies a frame through system memory.

Needs an NVIDIA GPU and an ffmpeg build with NVENC. `Pipeline::finish` sends
ordered EOS through the encoder and muxer so delayed frames are drained
before the MP4 trailer is finalized.

```sh
cargo run -p screen_record_nvenc -- <output.mp4> [seconds]
```

On Linux the compositor's own dialog decides what is captured, so the first
run prompts and prints a restore token that later runs can pass to skip it —
the same arguments `screen_record` documents:

```sh
cargo run -p screen_record_nvenc -- <output.mp4> [seconds] [monitor|window] [restore-token]
```
