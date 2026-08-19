# text_overlay

A moving-gradient `TestVideoSource` background composited with a
`D3d11TextLayerHandle` clock in front of it, recorded to an mp4 — proves
dynamic text (not just a static watermark) actually updates on screen: the
overlaid text changes once a second while the recording runs, so the output
file's frames differ over time if `D3d11TextLayerHandle::set_text` is really
re-rasterizing and re-uploading each call.

The background runs as its own `Pipeline` (`TestVideoSource -> SwScaler ->
upload`) feeding a compositor source input; the compositor's output runs as a
second `Pipeline` (`download -> SwScaler -> SwEncoder -> Mp4Muxer`). The text
layer itself never receives `Pipeline` frames — it's a handle driven directly
by `set_text`/`set_position`, built through the compositor's own
`add_text_layer`.

- Windows: `D3d11Upload` -> `D3d11VideoCompositor` (+ `D3d11TextLayerHandle`)
  -> `D3d11Download`
- Linux: `CudaUpload` -> `CudaVideoCompositor` (+ `CudaTextLayerHandle`) ->
  `CudaDownload`

Both run the identical graph and CLI. Glyph rasterization is shared code;
what differs is how the coverage is drawn — a D3D11 blend state on one side,
a CUDA blend kernel on the other. The keyboard controls differ for the same
kind of reason: a Win32 console on Windows, a termios raw-mode terminal on
Linux. Linux needs an NVIDIA GPU and a system font (DejaVu Sans or Liberation
Sans; the example prints which one it found).

```sh
cargo run -p text_overlay -- [output.mp4] [seconds]
```

While recording, use the arrow keys to move the text, or `q` to stop early.
