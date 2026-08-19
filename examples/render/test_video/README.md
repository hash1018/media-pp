# test_video

`TestVideoSource -> Renderer`: a synthetic moving-gradient stream, no
file/camera/decoder involved at all, presented in a native window via
`render_common`'s own `D3d12WindowRenderer` (wrapped as a `D3d12Renderer`) —
proves `TestVideoSource`'s frames and `D3d12Renderer`'s CPU-upload path work
end to end without needing a real video source.

No `Pacer` here, deliberately, as an experiment: `TestVideoSource` self-paces
with a drift-free absolute schedule and nothing sits between it and the
renderer here (no `SwScaler`, unlike `screen_capture`). Testing confirmed that
schedule is enough on its own for a vsync-locked renderer to stay smooth
without a separate pacing stage; `screen_capture` reached the same result
after its source moved from variable-rate emission to the same absolute
scheduling scheme. Windows only.

```sh
cargo run -p test_video
```
