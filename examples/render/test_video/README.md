# test_video

`TestVideoSource -> Queue -> Renderer`: a synthetic moving-gradient stream, no
file/camera/decoder involved at all, presented in a native window. The
renderer — `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux —
draws the source's YUV420P as it comes and uploads it itself, in a window of
its own. This proves the synthetic source and complete presentation path work
without a real video.

No `Pacer` here, deliberately, as an experiment: `TestVideoSource` self-paces
with a drift-free absolute schedule and nothing sits between it and the
renderer here except a queue. Testing confirmed
that schedule is enough on its own for a vsync-locked renderer to stay smooth
without a separate pacing stage; `screen_preview_cpu` reached the same result
after its source moved from variable-rate emission to the same absolute
scheduling scheme. Windows and Linux are supported.

```sh
cargo run -p test_video
```
