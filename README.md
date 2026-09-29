# media-pp

`media-pp` is a small, GStreamer-flavored media pipeline library for Rust,
built on [`ffmpeg-next`]. Stages are synchronous calls by default, and thread
boundaries are explicit bounded queues.

- **Errors come back as `Result`.** A stage's failure returns straight up the
  call stack; only past a `Queue` does it become a bus event.
- **Running pipelines change shape.** Branches are added and finished while
  buffers flow, and filters are swapped without reopening the source.
- **GPU-resident end to end** on D3D11, D3D12, CUDA or Vulkan: decode, scale,
  composite, key and encode without copying pictures back to the CPU.
  `VideoDecodeBin` and `VideoEncodeBin` pick the hardware where it opens and
  fall back to software where it does not.
- **Capture** of screens, windows, cameras and audio on Windows and Linux;
  **output** to files, HLS, RTMP and RTSP servers, and WebRTC.
- **Playback control**: seek, pause, step, speed and reverse, with a preroll
  that shows the picture sought to before playing on.

The library crate lives in `lib/`; each directory below `examples/` is its
own crate.

## Quick start

```toml
[dependencies]
media-pp = "0.3"
```

FFmpeg 8.0 or newer development libraries must be installed (see
[Requirements](#requirements)). `ffmpeg-next` is re-exported as
`media_pp::ffmpeg`; use that rather than depending on it separately.

A file's pipeline says when everything read has arrived —
`BusEvent::Finished` — and waits there, where a seek could still take it
back, until it is stopped:

```rust,no_run
use media_pp::{
    bus::BusEvent,
    elements::{FileDemuxer, PacketCounter},
    ffmpeg,
    pipeline::Pipeline,
};

fn main() -> media_pp::Result<()> {
    let (source, _) = FileDemuxer::open("file", "video.mp4")?;
    let video = source.best(ffmpeg::media::Type::Video)?;
    let (counter, packets) = PacketCounter::new("counter");
    let (pipeline, ()) = Pipeline::new("count", source, |source, ctx| {
        ctx.attach(source, video.index, ctx.branch().to(counter)?)?;
        Ok(())
    })?;
    pipeline.run()?;
    for event in pipeline.bus().iter() {
        // An error does not end a pipeline by itself either.
        if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }
    println!("packets: {}", packets.get());
    Ok(())
}
```

To play a file with its sound, `Player` builds the whole pipeline — decode on
the GPU where it can, a window, the default audio output — and reports what
happens to it (features `d3d11` and `wasapi-renderer` on Windows, `vulkan`
and `pipewire-audio-renderer` on Linux):

```rust,no_run
use media_pp::player::{Player, PlayerEvent, PlayerOptions};

fn main() -> media_pp::Result<()> {
    let player = Player::open("video.mp4", PlayerOptions::default())?;
    player.play()?;
    while let Some(event) = player.next_event() {
        match event {
            // Space, arrows, `.` `,` `-` `+`, F; `false` for Escape or a close.
            PlayerEvent::Window(event) if !player.respond_to(&event) => break,
            PlayerEvent::Ended => break,
            PlayerEvent::Error { name, error } => eprintln!("{name}: {error}"),
            _ => {}
        }
    }
    Ok(())
}
```

How a pipeline runs — buffers, threads, the end of a stream, seeking,
changing a running graph, what a link refuses before it runs — is the crate
documentation's first page; each type's page says what it accepts, owns and
how it fails. Both are on [docs.rs] for the backend-independent API and in
the [Windows API documentation] for everything. What changed between
versions, and what to write instead, is in [`CHANGELOG.md`].

## What is in it

| | Software | D3D11 | D3D12 | CUDA | Vulkan |
|---|---|---|---|---|---|
| Decode | `SwDecoder` | `D3d11Decoder` | `D3d12Decoder` | `CudaDecoder` | `VulkanDecoder` |
| Encode | `SwEncoder` | `D3d11VideoEncoder` | | `CudaEncoder` | `VulkanEncoder` |
| Scale, convert | `SwScaler` | `D3d11Scaler` | `D3d12Scaler` | `CudaScaler`, `CudaConverter` | `VulkanScaler`, `VulkanConverter` |
| HDR to SDR | | `D3d11ToneMap` | | `CudaConverter` | |
| Composite | `SwVideoCompositor` | `D3d11VideoCompositor` | | `CudaVideoCompositor` | `VulkanVideoCompositor` |
| Key, colour | `SwChromaKey`, `SwVideoEffect` | `D3d11ChromaKey`, `D3d11VideoEffect` | | `CudaChromaKey`, `CudaVideoEffect` | `VulkanChromaKey`, `VulkanVideoEffect` |
| Upload, download | | `D3d11Upload`, `D3d11Download` | `D3d12Upload`, `D3d12Download` | `CudaUpload`, `CudaDownload` | `VulkanUpload`, `VulkanDownload` |
| Render | `VideoWindow` | `D3d11WindowRenderer`, `D3d11Renderer` | `D3d12WindowRenderer`, `D3d12Renderer` | `CudaRenderer` | `VulkanWindowRenderer` (Linux) |

Every compositor has text layers, runs live by default, and renders an export
frame-exact with `RenderMode::Offline`; `VideoCompositorControl` lets one piece
of code build a composition on any of them. `VideoWindow` is a window of its
own on whatever renderer the platform has, taking frames in system memory, so
a software decode goes into it with no `#[cfg]`.

- **Inputs**: `FileDemuxer`, `RtspSource`, `WebRtcTrackSource`, `AppSource`,
  `PipelineBridge`, `TestVideoSource`, `TestAudioSource`.
- **Capture**: Windows — `DxgiCaptureSource` (screen), `WgcCaptureSource`
  (window), `MfCaptureSource` (camera), `WasapiCaptureSource` (audio),
  `D3d11SharedTextureSource`. Linux — `PipeWireScreenCaptureSource`,
  `PipeWireAudioCaptureSource`, `V4l2CaptureSource`. macOS —
  `CoreAudioCaptureSource` (audio).
- **Audio**: `AudioMixer`, `AudioResampler`, `AudioVolume`, `AudioGate`,
  `AudioCompressor`, `AudioLimiter`, `NoiseSuppressor`, `AudioTempo`,
  `AudioWaveform`, `SwAudioEncoder`; playback through `WasapiRenderer`,
  `PipeWireAudioRenderer` and `CoreAudioRenderer`.
- **Outputs**: `FileMuxer`, `SegmentedFileMuxer`, `ReplayBuffer`, `HlsMuxer`,
  `RtmpMuxer`, `RtspMuxer`, `WebRtcTrackSink`, `AppSink`.
- **Flow and timing**: `Queue`, `Tee`, `Rack`, `Pacer`, `VideoSynchronizer`,
  `ChangeGate`, `FrameRateLimiter`, `PauseGate`, `TimestampOrigin`.
- **Analysis**: `OrtDetector`, `WhisperTranscriber`, `FrameCounter`,
  `PacketCounter`.

## Feature flags

No default features. Backend-specific types carry their backend's prefix and
exist only where their feature is enabled.

| Feature | Adds | Platform |
|---|---|---|
| `cuda` | NVDEC, NVENC, scaling, compositing, upload/download and rendering on CUDA frames | Linux, Windows |
| `d3d11` | D3D11 decode, encode, scaling, compositing, upload/download and rendering | Windows |
| `d3d12` | D3D12VA decode, scaling, upload/download and rendering | Windows |
| `vulkan` | Vulkan Video decode and encode, compositing, upload/download; `VulkanWindowRenderer` on Linux | Linux, Windows |
| `videotoolbox` | VideoToolbox decode, upload/download | macOS |
| `dxgi-capture` | Desktop capture; enables `d3d11` | Windows |
| `wgc-capture` | Window capture through Windows Graphics Capture; enables `d3d11` | Windows |
| `mf-capture` | Camera capture through Media Foundation | Windows |
| `wasapi-capture` | System, per-application and microphone audio capture | Windows |
| `wasapi-renderer` | Audio playback | Windows |
| `pipewire-screen-capture` | Desktop capture through xdg-desktop-portal | Linux |
| `pipewire-audio-capture` | System, per-application and microphone audio capture | Linux |
| `pipewire-audio-renderer` | Audio playback | Linux |
| `v4l2-capture` | Camera capture through Video4Linux2 | Linux |
| `coreaudio-capture` | System, per-application and microphone audio capture; the first two need macOS 14.2 | macOS |
| `coreaudio-renderer` | Audio playback | macOS |
| `ort` | ONNX Runtime object detection (YOLOv8/v11 layout) | All |
| `rnnoise` | Speech noise suppression (pure Rust, no model file) | All |
| `whisper`, `whisper-vulkan` | Speech to timed text through whisper.cpp, on the CPU or any Vulkan GPU | All |
| `webrtc` | `str0m`-based WebRTC peer and track elements | All |

## Examples

Each directory under `examples/` is a crate; run one with
`cargo run -p <name> -- <args>`. One that needs a file prints its usage when
run without one; no media is checked in. [`examples/render`](examples/render/README.md) has an
index of the windowed ones.

- `core`: decoding, queues, fan-out, dynamic tees, app sources and sinks,
  audio, remuxing, GPU transcoding, HLS, RTMP, transcription, compositing.
- `cuda`: headless CUDA recording and GPU text compositing.
- `render`: `Player`, playback on every renderer, seeking, capture, GPU
  scaling, keying, hardware encoding and recording.
- `rtsp`: publishing to an RTSP server such as MediaMTX, and receiving from
  one — this crate does not serve RTSP itself.
- `vision`: scaling and ONNX object detection.
- `webrtc`: loopback, a two-way call, and recording received tracks.

## Requirements

- FFmpeg 8.0 or newer development headers and libraries; hardware decoding
  also needs an FFmpeg build, driver and GPU that support it.
- Rust 1.88 or newer.
- `pipewire-*`: PipeWire 0.3.50 or newer development files.
- `cuda`: only the NVIDIA driver; the kernels ship as PTX.
- `vulkan`: an FFmpeg built with Vulkan (`--enable-vulkan`, or vcpkg's
  `ffmpeg[vulkan]`), found through `FFMPEG_DIR` or pkg-config, and the Vulkan
  headers (`VULKAN_SDK` or `libvulkan-dev`). The loader is opened at run time.
- `whisper-vulkan`: the Vulkan SDK's shader compiler at build time; on
  Windows also long paths and a short target directory — see
  [`transcribe`](examples/core/transcribe/README.md).

macOS has audio output (`coreaudio-renderer`) and capture
(`coreaudio-capture`), and no screen capture or GPU backend yet — see
[`docs/macos.md`](docs/macos.md). Building and testing are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.

`media-pp` does not bundle FFmpeg. Users are responsible for complying with
the license of their FFmpeg build and optional codecs.

[`CHANGELOG.md`]: https://github.com/hash1018/media-pp/blob/main/CHANGELOG.md
[`ffmpeg-next`]: https://github.com/zmwangx/rust-ffmpeg
[docs.rs]: https://docs.rs/media-pp
[Windows API documentation]: https://hash1018.github.io/media-pp/media_pp/
