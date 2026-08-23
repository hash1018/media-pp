# Render examples

Each directory is an independent crate. The names put the user-facing purpose
first; backend differences stay in the same crate when its CLI and output are
the same across platforms.

## Screen preview and recording

| Example | Purpose | Platform | Capture and memory path | Output | Ends by | Arguments |
|---|---|---|---|---|---|---|
| [`screen_preview_cpu`](screen_preview_cpu/) | Preview a CPU-captured desktop | Windows / Linux | DXGI / PipeWire system memory -> D3D12 / CUDA upload | Window | Close window | Linux: `[monitor\|window] [restore-token]` |
| [`screen_preview_gpu`](screen_preview_gpu/) | Preview without a system-memory pixel copy | Windows / Linux | DXGI D3D11 / PipeWire DMA-BUF -> CUDA | D3D11 / CUDA renderer | Close window | Linux: `[monitor\|window] [restore-token]` |
| [`screen_record_software`](screen_record_software/) | Record with software conversion and encoding | Windows / Linux | DXGI / PipeWire -> system-memory BGRA | OpenH264 MP4 | Fixed duration (`Stop`) | `[output.mp4] [seconds]` plus Linux source/token |
| [`screen_record_nvenc`](screen_record_nvenc/) | Record GPU-resident frames with NVENC | Windows / Linux | DXGI D3D11 / PipeWire DMA-BUF -> CUDA | NVENC MP4 | Fixed duration (`Finish`) | `<output.mp4> [seconds]` plus Linux source/token |
| [`screen_record_overlay`](screen_record_overlay/) | Draw a live CUDA overlay and record it | Linux | PipeWire DMA-BUF -> CUDA compositor | NVENC MP4 | Fixed duration | `<output.mp4> [seconds] [monitor\|window] [restore-token]` |
| [`screen_record_av`](screen_record_av/) | Record the desktop and system audio | Windows / Linux | DXGI + WASAPI / PipeWire video + audio | OpenH264 + AAC MP4 | `q` + Enter | `[output.mp4]` plus Linux source/token |

Use `screen_preview_gpu` for the general live-preview path. The
`screen_preview_cpu` example specifically demonstrates system-memory capture,
software conversion, and a platform GPU upload. Use `screen_record_software` for the
portable CPU encode path and `screen_record_nvenc` when the captured frame must
stay GPU-resident through encoding.

The first Linux screen-capture run opens the xdg-desktop-portal picker. Its
restore token can be passed as the last argument on later runs.

## Other render packages

| Example | Purpose | Platform | Main path | Ends by | Required arguments |
|---|---|---|---|---|---|
| [`av_playback`](av_playback/) | Play synchronized audio/video | Windows / Linux | Software audio + platform GPU video | EOS / close window | `<video>` |
| [`hw_decode_render`](hw_decode_render/) | Hardware-decode and render | Windows / Linux | D3D12VA / NVDEC zero-copy | EOS / close window | `<video>` |
| [`seek_render`](seek_render/) | Interactive seek/pause/resume | Windows / Linux | CPU decode -> platform GPU upload | `q`, EOS, or close | `<video>` |
| [`sw_decode_render`](sw_decode_render/) | Software-decode and render | Windows / Linux | CPU decode -> platform GPU upload | EOS / close window | `<video>` |
| [`test_video`](test_video/) | Render a synthetic source | Windows / Linux | CPU frame -> platform GPU upload | Close window | None |
| [`transcode_render`](transcode_render/) | Encode/decode round trip | Windows / Linux | OpenH264 round trip -> platform GPU | Close window | None |
| [`gpu_video_compositor`](gpu_video_compositor/) | Composite GPU frames | Windows / Linux | D3D11 / CUDA compositor | Close window | None |
| [`d3d11_decode_render`](d3d11_decode_render/) | D3D11VA decode/render | Windows | D3D11 zero-copy | EOS / close window | `<video>` |
| [`d3d11_scale_render`](d3d11_scale_render/) | D3D11 scale/render | Windows | D3D11 GPU path | EOS / close window | `<video>` |
| [`d3d11_upload`](d3d11_upload/) | Demonstrate D3D11 upload | Windows | CPU -> D3D11 | Close window | None |
| [`d3d12_upload`](d3d12_upload/) | Demonstrate D3D12 upload | Windows | CPU -> D3D12 | Close window | None |
| [`d3d11_chroma_key`](d3d11_chroma_key/) | Apply a D3D11 chroma key | Windows | D3D11 GPU path | Fixed duration | `[output.mp4] [seconds]` |
| [`nvenc_record`](nvenc_record/) | Demonstrate D3D11 NVENC | Windows | D3D11 -> NVENC | Fixed duration | `[output.mp4] [seconds]` |
| [`d3d11_text_overlay`](d3d11_text_overlay/) | Demonstrate D3D11 text overlay | Windows | D3D11 compositor | Fixed duration / `q` | `[output.mp4] [seconds]` |

`render_common` is a support crate shared by windowed examples, not an
executable example.
