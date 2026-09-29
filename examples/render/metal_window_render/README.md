# metal_window_render

macOS only. `TestVideoSource -> SwScaler -> [VideoToolboxUpload] ->
MetalWindowRenderer`: a moving test pattern drawn into a `winit` window by
the library's Metal renderer, from system memory or, with `--videotoolbox`,
from VideoToolbox pixel buffers.

- `--format nv12|yuv420p|bgra` picks the layout the frames reach the
  renderer in (`nv12` by default; VideoToolbox frames are NV12 or BGRA
  only).
- `--file PATH` plays a file instead, decoded on the CPU and paced
  (`FileDemuxer -> SwDecoder -> Pacer -> SwScaler -> ...`) — the way to see
  colour, since the test pattern is grey.
- `--seconds N` closes the window after N seconds, resizing it once halfway,
  for a run nobody watches.

The window, and the event loop on the main thread that AppKit asks for, are
this program's: it hands the renderer an `Arc` of the window through
`MetalWindowRenderer::for_window`, and the renderer gives the window's view a
Metal layer of its own and follows its size before every frame. So nothing
here runs inside `run_with_windows`, which is for a program with no event
loop of its own.

```sh
cargo run -p metal_window_render -- [--videotoolbox] [--format F] [--seconds N] [--file PATH]
```
