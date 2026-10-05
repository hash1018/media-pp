# Features

The library has no default features. Backend-specific types carry their
backend's prefix and exist only where their feature is enabled; an
unprefixed type works the same on every platform.

| Feature | Adds | Platform |
|---|---|---|
| `cuda` | NVDEC, NVENC, scaling, compositing, upload/download and rendering on CUDA frames | Linux, Windows |
| `d3d11` | D3D11 decode, encode, scaling, compositing, upload/download and rendering | Windows |
| `d3d12` | D3D12VA decode, scaling, upload/download and rendering | Windows |
| `vulkan` | Vulkan Video decode and encode, compositing, upload/download; `VulkanWindowRenderer` on Linux | Linux, Windows |
| `videotoolbox` | VideoToolbox decode and encode, upload/download | macOS |
| `metal` | Scaling, converting, keying, colour effects and compositing on VideoToolbox frames with Metal, `MetalSharedTextureSource`, and `MetalWindowRenderer`, `MetalRenderer`, `VideoWindow` and, with `coreaudio-renderer`, `Player`; enables `videotoolbox` | macOS |
| `dxgi-capture` | Desktop capture; enables `d3d11` | Windows |
| `wgc-capture` | Window capture through Windows Graphics Capture; enables `d3d11` | Windows |
| `mf-capture` | Camera capture through Media Foundation | Windows |
| `mf-virtual-camera` | The pipeline's pictures as a camera other applications open; Windows 11, with the [`vcam`](../vcam/README.md) DLL installed | Windows |
| `wasapi-capture` | System, per-application and microphone audio capture | Windows |
| `wasapi-renderer` | Audio playback | Windows |
| `pipewire-screen-capture` | Desktop capture through xdg-desktop-portal | Linux |
| `pipewire-audio-capture` | System, per-application and microphone audio capture | Linux |
| `pipewire-audio-renderer` | Audio playback | Linux |
| `v4l2-capture` | Camera capture through Video4Linux2 | Linux |
| `v4l2-virtual-camera` | The pipeline's pictures as a camera other applications open, written into a v4l2loopback device; the module must be loaded | Linux |
| `screencapturekit-capture` | Display and window capture through ScreenCaptureKit (macOS 12.3+) | macOS |
| `avfoundation-capture` | Camera capture through AVFoundation | macOS |
| `coreaudio-capture` | System, per-application and microphone audio capture; the first two need macOS 14.2 | macOS |
| `coreaudio-renderer` | Audio playback | macOS |
| `ort` | ONNX Runtime object detection (YOLOv8/v11 layout) | All |
| `rnnoise` | Speech noise suppression (pure Rust, no model file) | All |
| `whisper`, `whisper-vulkan` | Speech to timed text through whisper.cpp, on the CPU or any Vulkan GPU | All |
| `webrtc` | `str0m`-based WebRTC peer and track elements | All |

`Player` needs a window renderer and an audio output: `d3d11` and
`wasapi-renderer` on Windows, `vulkan` and `pipewire-audio-renderer` on
Linux, `metal` and `coreaudio-renderer` on macOS.

## What a feature needs

Beyond FFmpeg 8.0 and Rust 1.88, which everything needs — see
[building](building/):

- **Hardware decoding**: an FFmpeg build, driver and GPU that support it.
- **`pipewire-*`**: PipeWire 0.3.50 or newer development files.
- **`cuda`**: only the NVIDIA driver; the kernels ship as PTX, and the
  driver library is opened at run time, so a build with `cuda` still starts
  on a machine without one.
- **`vulkan`**: an FFmpeg built with Vulkan (`--enable-vulkan`, or vcpkg's
  `ffmpeg[vulkan]`), found through `FFMPEG_DIR` or pkg-config, and the
  Vulkan headers (`VULKAN_SDK` or `libvulkan-dev`). The loader is opened at
  run time.
- **`whisper-vulkan`**: the Vulkan SDK's shader compiler at build time; on
  Windows also long paths and a short target directory — see
  [`transcribe`](../examples/core/transcribe/README.md).
- **`ort`**: a model in the YOLOv8/v11 output layout.

What an element needs at run time — one shared `D3d11Gpu`, one `CudaDevice`
per process, a portal for screen capture on Linux, a server to publish RTSP
to, the permissions a Mac asks for — is on that element's documentation
page.
