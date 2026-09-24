# Render examples

Each directory is an independent crate. The names put the user-facing purpose
first; backend differences stay in the same crate when its CLI and output are
the same across platforms.

## Screen preview and recording

| Example | Purpose | Platform | Capture and memory path | Output | Ends by | Arguments |
|---|---|---|---|---|---|---|
| [`screen_preview_cpu`](screen_preview_cpu/) | Preview a CPU-captured desktop | Windows / Linux | DXGI / PipeWire system memory -> D3D12 upload / `VulkanWindowRenderer` | Window | Escape or close window | Linux: `[monitor\|window] [restore-token]` |
| [`screen_preview_gpu`](screen_preview_gpu/) | Preview without a system-memory pixel copy | Windows / Linux | DXGI/WGC D3D11 / PipeWire DMA-BUF -> CUDA | D3D11 renderer / `VulkanWindowRenderer` | Escape or close window | Windows: `[dxgi\|wgc [<HWND>]]` (`wgc` with no `HWND` prompts); Linux: `[monitor\|window] [restore-token]` |
| [`screen_record_software`](screen_record_software/) | Record with software conversion and encoding | Windows / Linux | DXGI / PipeWire -> system-memory BGRA | OpenH264 MP4 | Fixed duration (`Stop`) | `[output.mp4] [seconds]` plus Linux source/token |
| [`screen_record_nvenc`](screen_record_nvenc/) | Record GPU-resident frames with NVENC | Windows / Linux | DXGI D3D11 / PipeWire DMA-BUF -> CUDA | NVENC MP4 | Fixed duration (`Finish`) | `<output.mp4> [seconds]` plus Linux source/token |
| [`screen_record_overlay`](screen_record_overlay/) | Draw a live CUDA overlay and record it | Linux | PipeWire DMA-BUF -> CUDA compositor | NVENC MP4 | Fixed duration | `<output.mp4> [seconds] [monitor\|window] [restore-token]` |
| [`screen_record_av`](screen_record_av/) | Record the desktop and system audio | Windows / Linux | DXGI + WASAPI / PipeWire video + audio | OpenH264 + AAC MP4 | `q` + Enter | `[output.mp4]` plus Linux source/token |

Use `screen_preview_gpu` for the general live-preview path. The
`screen_preview_cpu` example specifically demonstrates system-memory capture,
software conversion, and a platform GPU upload — on Linux, the renderer
drawing the capture's BGRA as it comes and uploading it itself. Use `screen_record_software` for the
portable CPU encode path and `screen_record_nvenc` when the captured frame must
stay GPU-resident through encoding.

The first Linux screen-capture run opens the xdg-desktop-portal picker. Its
restore token can be passed as the last argument on later runs.

## Other render packages

| Example | Purpose | Platform | Main path | Ends by | Required arguments |
|---|---|---|---|---|---|
| [`av_playback`](av_playback/) | Play synchronized audio/video | Windows / Linux | Software audio + platform GPU video | EOS, Escape or close window | `<video>` |
| [`hw_decode_render`](hw_decode_render/) | Decode on the GPU where it can, in software where not, and render | Windows / Linux | `VideoDecodeBin` onto D3D12 / CUDA | EOS, Escape or close window | `<video>` |
| [`seek_render`](seek_render/) | Interactive seek/pause/resume | Windows / Linux | CPU decode -> platform GPU upload | `q`, EOS, Escape or close window | `<video>` |
| [`sw_decode_render`](sw_decode_render/) | Software-decode and render | Windows / Linux | CPU decode -> platform GPU upload | EOS, Escape or close window | `<video>` |
| [`test_video`](test_video/) | Render a synthetic source | Windows / Linux | CPU frame -> platform GPU upload | Escape or close window | None |
| [`transcode_render`](transcode_render/) | Encode/decode round trip | Windows / Linux | OpenH264 round trip -> platform GPU | Escape or close window | None |
| [`gpu_video_compositor`](gpu_video_compositor/) | Composite GPU frames | Windows / Linux | D3D11 / CUDA compositor | Escape or close window | None |
| [`cuda_decode_render`](cuda_decode_render/) | NVDEC decode/render in the renderer's own window | Linux | CUDA, `VulkanWindowRenderer::open` | EOS, Escape or close window; Space pauses | `<video>` |
| [`vulkan_window_render`](vulkan_window_render/) | Render into a `winit` window with the library's Vulkan renderer | Linux | CPU NV12, YUV420P or BGRA (`--format`), or CUDA with `--cuda`, `VulkanWindowRenderer` | Close window, or after `--seconds N` | None; `--file <video>` plays a file |
| [`d3d11_decode_render`](d3d11_decode_render/) | D3D11VA decode/render in the renderer's own window | Windows | D3D11 zero-copy, `D3d11WindowRenderer` | EOS, Escape or close window; Space pauses | `<video>` |
| [`d3d11_scale_render`](d3d11_scale_render/) | D3D11 scale/render | Windows | D3D11 GPU path | EOS, Escape or close window | `<video>` |
| [`d3d11_upload`](d3d11_upload/) | Demonstrate D3D11 upload | Windows | CPU -> D3D11 | Escape or close window | None |
| [`d3d12_upload`](d3d12_upload/) | Demonstrate D3D12 upload into the renderer's own window | Windows | CPU -> D3D12, `D3d12WindowRenderer` | Escape or close window; Space pauses | None |
| [`d3d11_chroma_key`](d3d11_chroma_key/) | Apply a D3D11 chroma key | Windows | D3D11 GPU path | Fixed duration | `[output.mp4] [seconds]` |
| [`nvenc_record`](nvenc_record/) | Demonstrate D3D11 NVENC | Windows | D3D11 -> NVENC | Fixed duration | `[output.mp4] [seconds]` |
| [`d3d11_text_overlay`](d3d11_text_overlay/) | Demonstrate D3D11 text overlay | Windows | D3D11 compositor | Fixed duration / `q` | `[output.mp4] [seconds]` |

`render_common` is a support crate shared by windowed examples, not an
executable example. Every example draws through one of the library's window
renderers in a window of its own — `D3d11WindowRenderer` or
`D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux — and what
is left in it is turning a close of that window into a stop, and on Linux
fitting a software decode to `VulkanWindowRenderer`'s input.
