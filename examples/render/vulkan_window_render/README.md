# vulkan_window_render

Linux only. `TestVideoSource -> SwScaler -> [CudaUpload] ->
VulkanWindowRenderer`: a moving test pattern drawn into a `winit` window by
the library's Vulkan renderer, from system memory or, with `--cuda`, from
CUDA.

- `--format nv12|yuv420p|bgra` picks the layout the frames reach the
  renderer in (`nv12` by default; CUDA frames are NV12 or BGRA only).
- `--file PATH` plays a file instead, decoded on the CPU and paced — the way
  to see colour, since the test pattern is grey.
- `--seconds N` closes the window after N seconds, resizing it once halfway,
  for a run nobody watches.

The window is this program's: it runs the event loop and hands the renderer
an `Arc` of the window. A resize is followed by the renderer itself on X11,
and through the `WindowSize` this program sets from its `Resized` events on
Wayland.

```sh
cargo run -p vulkan_window_render -- [--cuda] [--format F] [--seconds N] [--file PATH]
```
