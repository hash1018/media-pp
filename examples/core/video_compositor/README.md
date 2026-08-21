# video_compositor

A `TestVideoSource` background, and a green-screen foreground fed from an
`AppSource` through `SwChromaKey` — both into one `SwVideoCompositor` ->
`SwScaler` -> `SwEncoder` -> `Mp4Muxer`. The foreground layer's on-canvas
position moves at runtime through its `SwVideoLayerHandle`; the "figure"
inside its own frame stays put, so the only thing chroma-keying visibly
changes is that the green around it disappears to reveal the moving
background layer underneath.

A different size and rate for the two inputs demonstrates that compositor
inputs are independent live pipelines; each sink retains only its latest
frame and the compositor emits on its own 30fps clock.

```sh
cargo run -p video_compositor -- [output.mp4] [seconds]
```
