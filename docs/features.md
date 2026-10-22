# Features

The library has no default features: the software elements are always
there, and each GPU backend, capture API and output is a feature you turn
on. Backend-specific types carry their backend's prefix and exist only where
their feature is enabled; an unprefixed type works the same on every
platform.

## Which to turn on

| To | Windows | Linux | macOS |
|---|---|---|---|
| Play a file with `Player` | `d3d11`, `wasapi-renderer` | `vulkan`, `pipewire-audio-renderer` | `metal`, `coreaudio-renderer` |
| Decode and encode on the GPU | `d3d11` or `d3d12`; `cuda` on NVIDIA | `vulkan`; `cuda` on NVIDIA | `videotoolbox` |
| Record the screen and its sound | `dxgi-capture`, `wasapi-capture` | `pipewire-screen-capture`, `pipewire-audio-capture` | `screencapturekit-capture`, `coreaudio-capture` |
| Capture a camera | `mf-capture` | `v4l2-capture` | `avfoundation-capture` |
| Detect objects on the GPU | `ort-directml` (any GPU), `ort-tensorrt` (NVIDIA) | `ort-tensorrt` (NVIDIA) | `ort-coreml` |
| Detect objects on the CPU | `ort` | `ort` | `ort` |

[Choosing a backend](guide/choosing-a-backend.md) says more about the GPU
choice.

## Every feature

| Feature | Adds | Platform |
|---|---|---|
| `cuda` | NVDEC, NVENC, scaling, compositing, upload/download and rendering on CUDA frames | Linux, Windows |
| `d3d11` | D3D11 decode, encode, scaling, compositing, upload/download and rendering | Windows |
| `d3d12` | D3D12VA decode, encoding (`D3d12VideoEncoder`, HEVC through FFmpeg 8.0's `hevc_d3d12va`), scaling, upload/download, HDR to SDR (`D3d12ToneMap`), rendering, finding cuts (`D3d12CutDetector`), drawing or hiding detections (`D3d12DetectionOverlay`), compositing (`D3d12VideoCompositor`) and `ObjectTracker`'s following by look on the GPU for D3D12 pictures | Windows |
| `vulkan` | Vulkan Video decode and encode, compositing, upload/download; `VulkanWindowRenderer` on Linux | Linux, Windows |
| `videotoolbox` | VideoToolbox decode and encode, upload/download | macOS |
| `metal` | Scaling, converting, keying, colour effects and compositing on VideoToolbox frames with Metal, `MetalSharedTextureSource`, `MetalDetectionOverlay`, `ObjectTracker`'s following by look on the GPU for VideoToolbox pictures, and `MetalWindowRenderer`, `MetalRenderer`, `VideoWindow` and, with `coreaudio-renderer`, `Player`; enables `videotoolbox` | macOS |
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
| `ort` | ONNX Runtime object detection, classification and embedding on the CPU, `SwOrtDetector`, `SwOrtClassifier` and `SwOrtEmbedder`; `Detections` and the detection overlays need no feature | All |
| `ort-cuda` | Object detection, classification and embedding on CUDA pictures, `CudaOrtDetector`, `CudaOrtClassifier` and `CudaOrtEmbedder`, on ONNX Runtime's CUDA provider; links CUDA and cuDNN; enables `ort` and `cuda` | Linux, Windows |
| `ort-tensorrt` | `CudaOrtDetector`, `CudaOrtClassifier` and `CudaOrtEmbedder` through TensorRT; links TensorRT as well; enables `ort-cuda` | Linux, Windows |
| `cuda-visual-tracking` | `ObjectTracker`'s following by look on the GPU for CUDA pictures; links cuFFT; enables `cuda` | Linux, Windows |
| `ort-coreml` | Object detection, classification and embedding on VideoToolbox pictures, `MetalOrtDetector`, `MetalOrtClassifier` and `MetalOrtEmbedder`, through ONNX Runtime's Core ML provider; enables `ort` and `metal` | macOS (Apple silicon) |
| `ort-directml` | Object detection, classification and embedding on D3D12 pictures, `D3d12OrtDetector`, `D3d12OrtClassifier` and `D3d12OrtEmbedder`, through ONNX Runtime's DirectML provider on any vendor's GPU; enables `ort` and `d3d12` | Windows |
| `rnnoise` | Speech noise suppression (pure Rust, no model file) | All |
| `whisper` | Speech to timed text through whisper.cpp on the CPU, `SwWhisperTranscriber`; `Transcript` and `SubtitleEncoder` need no feature | All |
| `whisper-vulkan` | `VulkanWhisperTranscriber`, on any GPU through Vulkan; enables `whisper` | Linux, Windows |
| `whisper-cuda` | `CudaWhisperTranscriber`, on an NVIDIA GPU with CUDA; enables `whisper` | Linux, Windows |
| `whisper-metal` | `MetalWhisperTranscriber`, on a Mac's GPU; enables `whisper` | macOS |
| `webrtc` | `str0m`-based WebRTC peer and track elements; `WebRtcServer`: `Broadcast`s to WebRTC players and from WHIP publishers, WHEP and WHIP over HTTP; and the WHIP and WHEP clients `WhipPublisher` and `WhepSource`, over `http` or `https` (`ureq` with `rustls`, no system TLS library) | All |
| `rtsp-server` | `RtspServer`: `Broadcast`s played to RTSP players, and RTSP publishers taken into them; tokio, inside the server | All |

## What a feature needs

Beyond FFmpeg 8.0 and Rust 1.88, which everything needs — see building on
[Windows](building/windows.md), [Linux](building/linux.md) or
[macOS](building/macos.md):

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
  [`transcribe`](../examples/speech/transcribe/README.md).
- **`whisper-cuda`**: the CUDA toolkit to build — `nvcc`, its headers, the
  runtime and cuBLAS, found through `CUDA_PATH` — since whisper.cpp's CUDA
  kernels are compiled with the crate; CUDA's redistributable archives are
  enough, nothing need be installed. On Windows build with
  `CMAKE_GENERATOR=Ninja` from a Visual Studio developer prompt, and set
  `CMAKE_CUDA_ARCHITECTURES` to the card's (`86` for an RTX 30) unless every
  architecture is wanted. The program does not start without CUDA's runtime
  and cuBLAS DLLs on `PATH`.
- **`ort`**: a model exported to ONNX — YOLOv8 and YOLO11, or YOLOv10
  and YOLO26, RetinaFace, or a DETR-family detector. Building downloads
  ONNX Runtime's prebuilt library, a few hundred megabytes and silent while
  it runs, once for each set of `ort-*` features, and keeps it in the
  user's cache (`ort.pyke.io`). For a machine with no network, point
  `ORT_LIB_LOCATION` at an ONNX Runtime of your own; see
  [`ort`'s linking guide](https://ort.pyke.io/setup/linking). media-pp
  requires `ort` 2.0.0-rc.13 exactly, since its release candidates change
  their API between one another.
- **`ort-coreml`**: an Apple silicon Mac. ONNX Runtime's build with Core
  ML is fetched as the CPU one is, and there is none for Intel Macs; Core ML
  itself is part of macOS.
- **`ort-directml`**: a D3D12 GPU of any vendor. ONNX Runtime's build with
  DirectML is fetched as the CPU one is — with `ort-tensorrt`, the one build
  that has both — and its `DirectML.dll` is put beside the program in the
  target directory by the build; ship it beside yours. Export the model
  at **opset 20 or lower** — Ultralytics' export picks the newest opset its
  PyTorch knows, 22 or later, unless told `opset=20` — and with half-precision weights: DirectML hands the
  operators of a later opset to the CPU, many times slower, and the
  element warns of it when made. See
  [d3d12-inference.md](d3d12-inference.md).
- **`ort-cuda`, `ort-tensorrt`**: CUDA 13's runtime, cuBLAS and cuRAND,
  and cuDNN 9 — and with `ort-tensorrt`, TensorRT 10 — are linked into the
  program, which does not start without them, as it does not without
  FFmpeg. On Linux building needs them where the linker finds them
  (`MEDIA_PP_NVIDIA_LIB_DIRS`, `LD_LIBRARY_PATH`, the CUDA installation, or
  the system's directories). On Windows building needs none of them — the
  program names each DLL itself — and running needs the directories that
  hold them on `PATH`. None needs an NVIDIA account: CUDA's and cuDNN's
  redistributable archives are at developer.download.nvidia.com, and
  TensorRT's libraries are NVIDIA's pip wheels (`tensorrt-cu13-libs`), which
  are zip files, at pypi.nvidia.com. At run time it also
  needs a driver for CUDA 13.0 and at least the CUDA 13.2 runtime, cuDNN
  9.23 and TensorRT 10.15; `CudaOrtDetector::runtime` says what falls
  short.
- **`cuda-visual-tracking`**: cuFFT 12 is linked as those are, found the
  same way, and needed the same way at run time.

  An application that ships them beside itself on Linux, as it would
  FFmpeg, needs one thing more where TensorRT comes from NVIDIA's tar
  archive: that `libnvinfer` has no run path of its own, and it opens its
  builder resources by name when it builds an engine, which only the
  loader's own path and an *RPATH* on the executable reach — not the
  RUNPATH linkers write by default. Link with
  `-Wl,--disable-new-dtags,-rpath,$ORIGIN/lib` rather than
  `-Wl,-rpath,$ORIGIN/lib`. Without it the program starts and runs from an
  engine already cached, and fails on a machine building its first one,
  with `TensorRT EP failed to create engine`. NVIDIA's pip wheels set
  `$ORIGIN` on `libnvinfer` and need neither. Preloading the resources
  does not help: their sonames are not the names they are opened by.

What an element needs at run time — one shared `D3d11Gpu`, one `CudaDevice`
per process, a portal for screen capture on Linux, a server to publish RTSP
to, the permissions a Mac asks for — is on that element's documentation
page.
