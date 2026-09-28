# gpu_video_compositor

Two `TestVideoSource` pipelines -> `SwScaler(NV12)` -> upload -> GPU
compositor -> `Tee` -> {`Queue` -> renderer for live display, `Queue` ->
download -> `SwScaler(YUV420P)` -> `Queue` -> `SwEncoder` -> `FileMuxer` for
simultaneous recording}. The foreground layer moves at runtime through its
layer handle, as in the CPU `video_compositor`, but nothing composited touches
the CPU until the recording branch's download.

- Windows: `D3d11Upload` -> `D3d11VideoCompositor` -> `D3d11WindowRenderer` /
  `D3d11Download`
- Linux: `CudaUpload` -> `CudaVideoCompositor` -> `VulkanWindowRenderer`,
  drawing the CUDA frames on a `VulkanGpu` made for that CUDA device /
  `CudaDownload`

Both platforms run the same graph, layer settings and CLI — the foreground at
0.85 opacity with `VideoFit::Cover`. On CUDA those two are why the compositor
scales with libavfilter's `scale_cuda` but crops by copy and blends with its
own kernel: no CUDA filter there can crop or blend.

```sh
cargo run -p gpu_video_compositor -- [output.mp4] [seconds]
```

It records to `gpu_video_compositor.mp4` for 5 seconds by default; closing
the window or Escape ends it early. Needs an NVIDIA GPU on Linux.
