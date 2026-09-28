# cuda_decode_render

Linux only. `FileDemuxer -> CudaDecoder -> Queue -> Pacer ->
VulkanWindowRenderer`: decodes on the GPU with NVDEC and shows the frames at
real playback speed, without the decoded pixels leaving the GPU — the
renderer copies each one device to device into memory Vulkan draws from. The
Linux sibling of `d3d11_decode_render`.

The window is the renderer's own: `VulkanWindowRenderer::open` opens it on a
thread of its own and reports what happens to it — Space pauses and resumes,
F or a double click fills the screen and puts it back, Escape or closing the
window stops — and its `WindowControl` shows where playback is in the title.
It is an X11 window, XWayland on a Wayland desktop.

```sh
cargo run -p cuda_decode_render -- path/to/video.mp4
```
