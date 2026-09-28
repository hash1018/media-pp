# video_compositor

A `TestVideoSource` background, and a green-screen foreground fed from an
`AppSource` through `SwChromaKey`, both into one `SwVideoCompositor ->
SwScaler -> Queue -> SwEncoder -> FileMuxer`. The foreground layer moves
across the canvas at runtime through its `SwVideoLayerHandle`; the figure
inside it stays put, so what keying visibly changes is that the green around
it disappears to reveal the moving background.

The two inputs differ in size and rate: compositor inputs are independent
live pipelines, each keeping only its latest frame, and the compositor emits
on its own 30 fps clock.

```sh
cargo run -p video_compositor -- [output.mp4] [seconds]
```

It records to `video_compositor.mp4` for 5 seconds by default, then finishes
the output pipeline so the file is finalized.
