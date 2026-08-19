# text_overlay

A moving-gradient `TestVideoSource` background composited with a
`D3d11TextLayerHandle` clock in front of it, recorded to an mp4 — proves
dynamic text (not just a static watermark) actually updates on screen: the
overlaid text changes once a second while the recording runs, so the output
file's frames differ over time if `D3d11TextLayerHandle::set_text` is really
re-rasterizing and re-uploading each call.

The background runs as its own `Pipeline` (`TestVideoSource -> SwScaler ->
D3d11Upload`) feeding a `D3d11VideoCompositor` source input; the compositor's
output runs as a second `Pipeline` (`D3d11Download -> SwScaler -> SwEncoder ->
Mp4Muxer`). The text layer itself never receives `Pipeline` frames — it's a
handle driven directly by `set_text`/`set_position`, built against the
compositor's own device via `D3d11VideoCompositorHandle::add_text_layer`.
Windows only.

```sh
cargo run -p text_overlay -- [output.mp4] [seconds]
```

While recording, use the arrow keys to move the text, or `q` to stop early.
