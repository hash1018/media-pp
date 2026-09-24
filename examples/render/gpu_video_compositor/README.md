# gpu_video_compositor

Two `TestVideoSource` pipelines -> `SwScaler(NV12)` -> upload -> GPU compositor
-> `Tee` -> {renderer for live display,
`download -> SwScaler(YUV420P) -> SwEncoder -> FileMuxer` for simultaneous
recording}. The foreground layer moves at runtime through
its layer handle, same as the CPU `video_compositor` example, but every frame
this composites never touches the CPU until the recording branch's own
download.

- Windows: `D3d11Upload` -> `D3d11VideoCompositor` -> `D3d11WindowRenderer` /
  `D3d11Download`
- Linux: `CudaUpload` -> `CudaVideoCompositor` -> `VulkanWindowRenderer`,
  drawing the CUDA frames on a `VulkanGpu` made for that CUDA device /
  `CudaDownload`

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
