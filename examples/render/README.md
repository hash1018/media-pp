# Render examples

Each directory is an independent crate. The names put the user-facing purpose
first; backend differences stay in the same crate when its CLI and output are
the same across platforms.

## Screen preview and recording

| Example | Purpose | Platform | Capture and memory path | Output | Ends by | Arguments |
|---|---|---|---|---|---|---|
| [`screen_preview_cpu`](screen_preview_cpu/) | Preview a CPU-captured desktop | Windows | DXGI system-memory BGRA -> D3D12 upload | Window | Close window | None |
| [`screen_preview_gpu`](screen_preview_gpu/) | Preview without a system-memory pixel copy | Windows / Linux | DXGI D3D11 / PipeWire DMA-BUF -> CUDA | D3D11 / CUDA renderer | Close window | Linux: `[monitor\|window] [restore-token]` |
| [`screen_record_software`](screen_record_software/) | Record with software conversion and encoding | Windows / Linux | DXGI / PipeWire -> system-memory BGRA | OpenH264 MP4 | Fixed duration (`Stop`) | `[output.mp4] [seconds]` plus Linux source/token |
| [`screen_record_nvenc`](screen_record_nvenc/) | Record GPU-resident frames with NVENC | Windows / Linux | DXGI D3D11 / PipeWire DMA-BUF -> CUDA | NVENC MP4 | Fixed duration (`Finish`) | `<output.mp4> [seconds]` plus Linux source/token |
| [`screen_record_overlay`](screen_record_overlay/) | Draw a live CUDA overlay and record it | Linux | PipeWire DMA-BUF -> CUDA compositor | NVENC MP4 | Fixed duration | `<output.mp4> [seconds] [monitor\|window] [restore-token]` |
| [`screen_record_av`](screen_record_av/) | Record the desktop and system audio | Windows / Linux | DXGI + WASAPI / PipeWire video + audio | OpenH264 + AAC MP4 | `q` + Enter | `[output.mp4]` plus Linux source/token |

Use `screen_preview_gpu` for the general live-preview path. The Windows-only
`screen_preview_cpu` exists specifically to demonstrate system-memory capture,
software conversion, and D3D12 upload. Use `screen_record_software` for the
portable CPU encode path and `screen_record_nvenc` when the captured frame must
stay GPU-resident through encoding.

The first Linux screen-capture run opens the xdg-desktop-portal picker. Its
restore token can be passed as the last argument on later runs.

## Other render packages

The remaining packages demonstrate a narrower element or playback graph; each
directory's README gives its exact pipeline and CLI:

- [`av_playback`](av_playback/)
- [`d3d11_decode_render`](d3d11_decode_render/)
- [`d3d11_scale_render`](d3d11_scale_render/)
- [`d3d11_upload`](d3d11_upload/)
- [`d3d12_upload`](d3d12_upload/)
- [`gpu_chroma_key`](gpu_chroma_key/)
- [`gpu_video_compositor`](gpu_video_compositor/)
- [`hw_decode_render`](hw_decode_render/)
- [`nvenc_record`](nvenc_record/)
- [`seek_render`](seek_render/)
- [`sw_decode_render`](sw_decode_render/)
- [`test_video`](test_video/)
- [`text_overlay`](text_overlay/)
- [`transcode_render`](transcode_render/)

`render_common` is a support crate shared by windowed examples, not an
executable example.
