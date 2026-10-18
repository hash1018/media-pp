# Examples

Each directory under `examples/` is a crate of its own, so platform
dependencies stay out of the library. Run one from the repository's root:

```sh
cargo run -p <name> -- <arguments>
```

One that needs a file prints its usage when run without one; no media is
checked in. A platform-only example compiles elsewhere as a stub that says
so. Each has a README with its pipeline drawn out and what it needs.

Some to start with:

```sh
cargo run -p doctor                           # what this machine gives media-pp
cargo run -p player -- path/to/video.mp4      # a file with its sound, in a window
cargo run -p probe -- path/to/video.mp4       # what a file holds, through a queue
cargo run -p transcode -- in.mp4 out.mp4      # decoded and encoded on the GPU where it can
cargo run -p resize -- in.mp4 out.mp4 --width 640 # software resize, with the sound copied
cargo run -p screen_record_software           # the desktop to an MP4
```

- [The basics](#the-basics)
- [Playing and showing](#playing-and-showing)
- [Converting and recording files](#converting-and-recording-files)
- [Screen capture](#screen-capture)
- [Sound](#sound)
- [Streaming](#streaming)
- [Compositing and overlays](#compositing-and-overlays)
- [Detection and analysis](#detection-and-analysis)
- [WebRTC](#webrtc)

"All" in the platform column means Windows, Linux and macOS.

## The basics

| Example | Shows | Platform |
|---|---|---|
| [`doctor`](../examples/core/doctor) | What this machine gives media-pp: FFmpeg's version, its encoders, and which GPU backends open a device | All |
| [`probe`](../examples/core/probe) | Looking at a file's streams, then wiring only the video through a `Queue` | All |
| [`decode`](../examples/core/decode) | `FileDemuxer -> SwDecoder -> FrameCounter` | All |
| [`pace`](../examples/core/pace) | The same, played at real speed through a `Pacer` | All |
| [`fanout`](../examples/core/fanout) | Video and audio each to a branch of its own | All |
| [`tee`](../examples/core/tee) | One stream to two branches through a `Tee` | All |
| [`dynamic_tee`](../examples/core/dynamic_tee) | A branch added to a running `Tee` and taken away again | All |
| [`app_source`](../examples/core/app_source) | Buffers your code makes, pushed into a pipeline | All |
| [`app_sink`](../examples/core/app_sink) | Every decoded picture handed to a closure | All |
| [`custom_element`](../examples/core/custom_element) | A `Source`, a `Filter` and a `Sink` of your own | All |

## Playing and showing

| Example | Shows | Platform |
|---|---|---|
| [`player`](../examples/render/player) | A file with its sound through `Player`, with keyboard control | All |
| [`hw_decode_render`](../examples/render/hw_decode_render) | `VideoDecodeBin -> Pacer ->` a window: the GPU where it decodes, software where not | All |
| [`sw_decode_render`](../examples/render/sw_decode_render) | Decoding in software into `VideoWindow` | All |
| [`seek_render`](../examples/render/seek_render) | Seeking, pausing and resuming from a terminal prompt | All |
| [`av_playback`](../examples/render/av_playback) | Picture and sound kept together by `VideoSynchronizer`, sound added and removed while it plays | All |
| [`test_video`](../examples/render/test_video) | A synthetic picture in a window | All |
| [`transcode_render`](../examples/render/transcode_render) | Encoded and decoded straight back, shown | All |
| [`d3d11_decode_render`](../examples/render/d3d11_decode_render) | D3D11VA decoding drawn without leaving the GPU | Windows |
| [`d3d11_scale_render`](../examples/render/d3d11_scale_render) | Decoding and scaling on one D3D11 device | Windows |
| [`d3d11_upload`](../examples/render/d3d11_upload), [`d3d12_upload`](../examples/render/d3d12_upload) | System-memory pictures uploaded to D3D11 or D3D12 | Windows |
| [`cuda_decode_render`](../examples/render/cuda_decode_render) | NVDEC decoding drawn by Vulkan without leaving the GPU | Linux |
| [`vulkan_window_render`](../examples/render/vulkan_window_render) | `VulkanWindowRenderer` in a `winit` window of your own | Linux |
| [`metal_window_render`](../examples/render/metal_window_render) | `MetalWindowRenderer` in a `winit` window of your own | macOS |

[`examples/render`](../examples/render/README.md) has a table of these with
their keys, arguments and how each ends.

## Converting and recording files

| Example | Shows | Platform |
|---|---|---|
| [`transcode`](../examples/core/transcode) | `VideoDecodeBin -> VideoEncodeBin`, the sound copied across | All |
| [`remux`](../examples/core/remux) | A new container with no re-encoding | All |
| [`resize`](../examples/core/resize) | Software resize saved as H.264, preserving the best audio track by copying it | All |
| [`scale`](../examples/vision/scale) | Converting and resizing with `SwScaler` | All |
| [`audio_record`](../examples/core/audio_record) | A tone encoded to AAC in an MP4 | All |
| [`nvenc_record`](../examples/render/nvenc_record) | D3D11 pictures encoded with NVENC | Windows |
| [`cuda_record`](../examples/cuda/cuda_record) | CUDA pictures encoded with NVENC | Linux, Windows |

`remux`, `transcode` and `resize` stage their outputs and publish only on
success. They refuse input/output aliases and require `--overwrite` to replace
an existing destination. Their shared, unpublished [`file_common`](../examples/core/file_common)
support crate has no executable. `scale` only verifies/counts resized frames;
use `resize` to save a video.

## Screen capture

| Example | Shows | Platform |
|---|---|---|
| [`screen_preview_gpu`](../examples/render/screen_preview_gpu) | The screen or a window shown without leaving the GPU | All |
| [`screen_preview_cpu`](../examples/render/screen_preview_cpu) | The same through system memory | All |
| [`screen_record_software`](../examples/render/screen_record_software) | The screen recorded with software encoding | All |
| [`screen_record_nvenc`](../examples/render/screen_record_nvenc) | The screen recorded on the GPU: NVENC, or VideoToolbox on macOS | All |
| [`screen_record_av`](../examples/render/screen_record_av) | The screen and its sound in one file, two sources in one pipeline | All |
| [`screen_record_overlay`](../examples/render/screen_record_overlay) | Text drawn over the screen on the GPU, recorded | Linux, macOS |
| [`virtual_camera`](../examples/core/virtual_camera) | A pipeline's pictures as a camera other applications open | Windows, Linux |

## Sound

| Example | Shows | Platform |
|---|---|---|
| [`audio_capture`](../examples/core/audio_capture) | Listing sound devices and capturing from one | All |
| [`audio_playback`](../examples/core/audio_playback) | A tone played, its volume changed while it plays | All |

## Streaming

| Example | Shows | Platform |
|---|---|---|
| [`rtmp_publish`](../examples/core/rtmp_publish) | Video and sound published to an RTMP server — Twitch, YouTube | All |
| [`hls`](../examples/core/hls) | A live HLS playlist and its segments | All |
| [`rtsp_serve`](../examples/rtsp/rtsp_serve) | A file published to an RTSP server such as MediaMTX | All |
| [`rtsp_serve_seek`](../examples/rtsp/rtsp_serve_seek) | The same, sought and paused while viewers watch | All |
| [`rtsp_source`](../examples/rtsp/rtsp_source) | Receiving from an RTSP camera or server | All |

## Compositing and overlays

| Example | Shows | Platform |
|---|---|---|
| [`video_compositor`](../examples/core/video_compositor) | Two sources composited on the CPU, one keyed out of a green screen and moved while it runs | All |
| [`gpu_video_compositor`](../examples/render/gpu_video_compositor) | Compositing on the GPU, shown and recorded at once | All |
| [`d3d11_chroma_key`](../examples/render/d3d11_chroma_key) | A green screen keyed out on the GPU | Windows |
| [`d3d11_text_overlay`](../examples/render/d3d11_text_overlay) | A clock drawn over video with D3D11 | Windows |
| [`cuda_text_overlay`](../examples/cuda/cuda_text_overlay) | The same with CUDA | Linux, Windows |

## Detection and analysis

The models are not included; each README says which to download.

| Example | Shows | Platform |
|---|---|---|
| [`detect`](../examples/vision/detect) | Objects found on the CPU and their boxes drawn | All |
| [`cuda_detect`](../examples/vision/cuda_detect) | Detection without leaving the GPU, through TensorRT | Linux, Windows |
| [`cuda_track`](../examples/vision/cuda_track) | Objects followed between pictures, the detector looking at only some | Linux, Windows |
| [`d3d12_detect`](../examples/vision/d3d12_detect) | Detection and tracking on any vendor's GPU through DirectML, boxed or hidden in an HEVC recording | Windows |
| [`cuda_multi_detect`](../examples/vision/cuda_multi_detect) | Several files batched through one detector with `StreamMux` | Linux, Windows |
| [`metal_detect`](../examples/vision/metal_detect) | Detection through Core ML on VideoToolbox pictures | macOS |
| [`metal_multi_detect`](../examples/vision/metal_multi_detect) | Several files batched through one detector | macOS |
| [`cut_detect`](../examples/vision/cut_detect) | Where each shot of an edited video begins — on the CPU, CUDA or Metal | All |
| [`vision_bench`](../examples/vision/vision_bench) | The harness the [vision benchmarks](benchmarks/vision/) are measured with | All |

## Speech

| Example | Shows | Platform |
|---|---|---|
| [`transcribe`](../examples/speech/transcribe) | Speech turned into a subtitle track with Whisper, on the CPU, CUDA, Vulkan or Metal | All |

## WebRTC

| Example | Shows | Platform |
|---|---|---|
| [`webrtc_loopback`](../examples/webrtc/webrtc_loopback) | Two peers over loopback UDP, no browser or signaling server | All |
| [`webrtc_av_loopback`](../examples/webrtc/webrtc_av_loopback) | Video and sound on one connection | All |
| [`webrtc_record`](../examples/webrtc/webrtc_record) | A file sent over WebRTC and the received tracks recorded | All |
| [`webrtc_video_call`](../examples/webrtc/webrtc_video_call) | A two-way video call, each side in a window | All |
