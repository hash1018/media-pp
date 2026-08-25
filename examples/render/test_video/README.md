# test_video

`TestVideoSource -> Queue -> SwScaler -> GPU upload -> Renderer`: a synthetic
moving-gradient stream, no
file/camera/decoder involved at all, presented in a native window via
the platform renderer. Windows converts to NV12 and uploads to D3D12; Linux
converts to NV12, uploads to CUDA, and presents through Vulkan. This proves the
synthetic source and complete presentation path work without a real video.

No `Pacer` here, deliberately, as an experiment: `TestVideoSource` self-paces
with a drift-free absolute schedule and nothing sits between it and the
renderer here except the required format conversion/upload. Testing confirmed
that schedule is enough on its own for a vsync-locked renderer to stay smooth
without a separate pacing stage; `screen_preview_cpu` reached the same result
after its source moved from variable-rate emission to the same absolute
scheduling scheme. Windows and Linux are supported.

```sh
cargo run -p test_video
```
