# gpu_video_compositor

Two `TestVideoSource` pipelines -> `D3d11Upload` -> `D3d11VideoCompositor`
(GPU shader compositing, no CPU round trip for the inputs or the composited
output) -> `Tee` -> {`D3d11Renderer` for live display, `D3d11Download ->
Scaler -> SwEncoder -> Mp4Muxer` for simultaneous recording}. The foreground
layer moves at runtime through its `D3d11VideoLayerHandle`, same as the CPU
`video_compositor` example, but every frame this composites never touches
the CPU until the recording branch's own `D3d11Download`.

```sh
cargo run -p gpu_video_compositor -- [output.mp4] [seconds]
```

`output.mp4` defaults to `gpu_video_compositor.mp4` and `seconds` defaults to
`5`.
