# gpu_video_compositor

Two `TestVideoSource` pipelines -> upload -> GPU compositor -> `Tee` ->
{renderer for live display, `download -> SwScaler -> SwEncoder -> Mp4Muxer`
for simultaneous recording}. The foreground layer moves at runtime through
its layer handle, same as the CPU `video_compositor` example, but every frame
this composites never touches the CPU until the recording branch's own
download.

- Windows: `D3d11Upload` -> `D3d11VideoCompositor` -> `D3d11Renderer` /
  `D3d11Download`
- Linux: `CudaUpload` -> `CudaVideoCompositor` -> `CudaRenderer` (Vulkan
  swapchain) / `CudaDownload`

Both branches run the identical graph, terminal sinks, layer settings, and
CLI — the foreground is drawn at 0.85 opacity with `VideoFit::Cover` on
either backend. On the CUDA side those two are exactly why it composites with
copies and a blend kernel rather than libavfilter: no CUDA filter there can
crop, and none can blend.

```sh
cargo run -p gpu_video_compositor -- [output.mp4] [seconds]
```

`output.mp4` defaults to `gpu_video_compositor.mp4` and `seconds` defaults to
`5`. Needs an NVIDIA GPU on Linux.
