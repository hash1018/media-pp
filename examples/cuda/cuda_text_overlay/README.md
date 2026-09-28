# cuda_text_overlay

A moving-gradient `TestVideoSource` background composited with a
`CudaTextLayerHandle` clock in front of it, recorded to an mp4. The text
changes once a second, so the file's frames differ over time only if
`set_text` really re-rasterizes and re-uploads each call.

The background runs as its own `Pipeline` (`TestVideoSource -> SwScaler ->
CudaUpload`) feeding a compositor input; the compositor's output runs as a
second one (`Queue -> CudaDownload -> SwScaler -> Queue -> SwEncoder ->
FileMuxer`). The text layer receives no frames — it is a handle driven by
`set_text` and `set_position`, made by the compositor's `add_text_layer`.

The graph is the same on Windows and Linux; `d3d11_text_overlay` is the
D3D11 counterpart. Only the raw-key terminal and the system font path differ
per OS; the example prints the font it found, and a machine with none of the
candidates gets a clear error. Needs an NVIDIA GPU.

```sh
cargo run -p cuda_text_overlay -- [output.mp4] [seconds]
```

It records to `cuda_text_overlay.mp4` for 5 seconds by default; while it
runs, the arrow keys move the text and `q` stops early.
