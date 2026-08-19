# video_compositor

Two `TestVideoSource` pipelines -> `SwVideoCompositor` -> `SwScaler` ->
`SwEncoder` -> `Mp4Muxer`. The foreground layer moves at runtime through its
`SwVideoLayerHandle` while both source connections stay unchanged.

A different size and frame rate for the two inputs demonstrates that
compositor inputs are independent live pipelines; each sink retains only its
latest frame and the compositor emits on its own 30fps clock.

```sh
cargo run -p video_compositor -- [output.mp4] [seconds]
```
