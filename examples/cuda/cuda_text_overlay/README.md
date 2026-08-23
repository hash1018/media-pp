# cuda_text_overlay

A moving-gradient `TestVideoSource` background composited with a
`CudaTextLayerHandle` clock in front of it, recorded to an mp4 — proves dynamic
text (not just a static watermark) actually updates: the overlaid text changes
once a second while the recording runs, so the output file's frames differ over
time if `CudaTextLayerHandle::set_text` is really re-rasterizing and
re-uploading each call.

The background runs as its own `Pipeline` (`TestVideoSource -> SwScaler ->
CudaUpload`) feeding a compositor source input; the compositor's output runs as
a second `Pipeline` (`CudaDownload -> SwScaler -> SwEncoder -> Mp4Muxer`). The
text layer itself never receives `Pipeline` frames — it's a handle driven
directly by `set_text`/`set_position`, built through the compositor's own
`add_text_layer`.

The graph is platform-independent. CUDA is a vendor backend rather than a
platform one, so the library dependency carries no per-target table and the
pipeline has no `cfg` switch — it builds and runs the same way on Windows and
Linux. `d3d11_text_overlay` is the D3D11 counterpart for the same graph.

Two things do differ per OS, and both are about the host rather than the GPU:
the raw-key terminal (`ReadConsoleInputW` on Windows, a termios raw-mode
terminal on Unix) and the system font path. The example prints which font it
found; a machine with none of the candidates gets a clear error rather than an
empty overlay.

Needs an NVIDIA GPU.

```sh
cargo run -p cuda_text_overlay -- [output.mp4] [seconds]
```

While recording, use the arrow keys to move the text, or `q` to stop early.
