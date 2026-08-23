# render_common

Shared library crate, not a runnable example. It owns the D3D11/D3D12
rendering that this project's render examples share — no external
`renderer-engine` dependency.

`D3d12GpuContext`/`D3d11GpuContext` are the process-wide device/queue/
shader-pipeline owners (create one per stack, share it across every window);
`d3d12_window_renderer`/`d3d11_window_renderer` open one window's
`D3d12WindowRenderer`/`D3d11WindowRenderer`, already wrapped as a
`media_pp::elements::D3d12Renderer`/`D3d11Renderer`. The two stacks are
independent — separate device, separate shader set, nothing shared between
them.

On Linux the same job is done by `VulkanGpuContext` and `cuda_window_renderer`,
which present `media_pp::elements::CudaDecoder` output through a Vulkan
swapchain. That stack is named for what it consumes, matching the library
element it plugs into: a CUDA frame has to be copied into Vulkan-owned memory,
so the graphics API is an implementation detail here exactly as the swapchain
is on the D3D side.

Both stacks present from the pipeline's own thread into a window the main
thread owns, so they share `run_window`: the winit shell that opens that
window, runs the work beside it, and — the part that is easy to get wrong and
fatal to get wrong — stops and joins before the window is dropped. Exiting the
event loop drops the window, which on Wayland frees the surface a
`vkQueuePresentKHR` may still be using; and `Pipeline::stop` cannot be called
from the event loop thread, because it waits for a renderer that in turn waits
for that loop to keep dispatching. `Shutdown` is how a worker publishes what a
close should stop, and how it learns a close arrived before it had anything to
publish.

Depended on by the other `examples/render/*` crates, including `av_playback`,
`screen_record_nvenc`, `seek_render`, `sw_decode_render`, `test_video`,
`text_overlay`, `transcode_render`, `nvenc_record`, `screen_preview_cpu`,
`screen_preview_gpu`, `hw_decode_render`, `gpu_video_compositor`,
`d3d11_upload`, `d3d12_upload`, and `d3d11_decode_render`.
