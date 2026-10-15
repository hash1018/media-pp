# Choosing a backend

media-pp runs the same jobs on several GPU APIs. Each is a Cargo feature, and
its elements carry its prefix. This page says which to pick; the
[elements list](../elements.md) has every element by backend, and
[features](../features.md) what each feature needs to build.

## The short answer

| Machine | Pick | Feature |
|---|---|---|
| Windows, any GPU | **D3D11** for capture, playback, compositing and encoding — the most complete Windows backend | `d3d11` |
| Windows, any GPU, with detection | **D3D12**: decode, detect through DirectML, draw, composite and encode on any vendor's GPU | `d3d12`, `ort-directml` |
| Windows or Linux, NVIDIA | **CUDA**: fastest decode, encode and detection (TensorRT) | `cuda`, `ort-tensorrt` |
| Linux, AMD or Intel | **Vulkan** | `vulkan` |
| macOS, Apple silicon | **VideoToolbox and Metal** | `metal` |
| Anything, or not sure | **Software** elements, or let `VideoDecodeBin` / `VideoEncodeBin` / `Player` choose | none |

You can start in software and move to a GPU later: the pipeline's shape stays
the same, only the elements change.

## Let media-pp choose

Three elements pick a backend themselves, trying hardware first:

- **`Player`** plays a file with its sound in a window.
- **`VideoDecodeBin`** decodes onto the device you give it (a
  `DecodeTarget`) where the stream's codec opens there, and in software
  where it does not; `path()` says which it took.
- **`VideoEncodeBin`** encodes H.264 with NVENC, Media Foundation,
  Vulkan Video, VideoToolbox or software, whichever opens first, from
  pictures wherever the decode bin put them (`EncodeInput::for_decoded`).

Everything else — scaling, compositing, detection, overlays — you choose by
using that backend's elements.

## Keep a pipeline on one device

A GPU pipeline is fastest when pictures never leave the GPU. For that:

1. **Make one device and share it.** Every element of a backend takes the
   device it works on: `D3d11Gpu`, `D3d12Gpu`, `CudaDevice`, `VulkanDevice`,
   `VideoToolboxDevice`. Make one and pass the same one to all of them. A
   picture from another device is refused with an error naming the element.
2. **Use that backend's elements end to end**, for example on D3D12:

   ```text
   FileDemuxer -> D3d12Decoder -> D3d12OrtDetector -> D3d12DetectionOverlay -> D3d12VideoEncoder -> FileMuxer
   ```

3. **Cross between memories explicitly** where you must: `D3d12Upload` puts
   system-memory pictures onto the GPU, `D3d12Download` brings them back —
   and likewise for every backend. A link that needs one and lacks it is
   refused before the pipeline runs, saying which to add.

## Each backend in brief

**D3D11** (Windows). Screen and window capture hand over D3D11 textures, so a
screen recorder stays on the GPU from capture to encoder. Decoding through
D3D11VA, encoding with NVENC or Media Foundation, compositing, keying,
colour effects, HDR to SDR, and window rendering. No detection.

**D3D12** (Windows). Decoding through D3D12VA, and the vision path on any
vendor's GPU: detection, classification and embedding through ONNX
Runtime's DirectML, tracking, drawing and hiding detections, finding scene
cuts. Also compositing, HDR to SDR, rendering and HEVC encoding through
D3D12 Video. Models should be fp16 and exported at ONNX opset 20 or lower —
see [D3D12 inference](../d3d12-inference.md).

**CUDA** (Windows, Linux; NVIDIA). NVDEC and NVENC, scaling and converting,
compositing, and the fastest detection: ONNX Runtime with CUDA or TensorRT,
batched across streams with `StreamMux`. Needs only the NVIDIA driver to run
the basic elements; the `ort-cuda` and `ort-tensorrt` features need NVIDIA's
CUDA, cuDNN and TensorRT libraries — see [features](../features.md).

**Vulkan** (Linux, Windows). Vulkan Video decoding and encoding, compositing,
scaling, keying and effects on any vendor's GPU — the Linux choice for AMD
and Intel. No detection.

**VideoToolbox and Metal** (macOS). Decoding and encoding through
VideoToolbox, everything else with Metal: scaling, compositing, keying,
effects, rendering, and detection through Core ML. Screen, window and
camera capture hand over VideoToolbox pictures.

**Software** (everywhere). Every job has a software element — decoding,
scaling, compositing, detection on the CPU, drawing. Slower, but it runs
anywhere and is the easiest to debug.

## Writing for several platforms

Elements without a prefix behave the same everywhere, and these traits let
one piece of code drive whichever backend a platform has:

- `VideoCompositorControl` builds a composition on any compositor.
- `VideoWindow` shows system-memory pictures in a window on whatever
  renderer the platform has.

Put the rest behind `#[cfg(target_os = …)]` at construction, as the
[examples](../examples.md) do: only the elements change, the pipeline does
not.
