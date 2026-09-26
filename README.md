# media-pp

`media-pp` is a small, GStreamer-flavored media pipeline library for Rust,
built on [`ffmpeg-next`]. Stages are synchronous calls by default, and thread
boundaries are explicit bounded queues.

- **Errors come back as `Result`.** A stage's failure returns straight up the
  call stack; only past a `Queue` does it become a bus event.
- **Running pipelines change shape.** Branches are added and finished while
  buffers flow — a recording finalized while its preview keeps running — and
  filters are swapped without reopening the source.
- **GPU-resident end to end** on D3D11, D3D12 or CUDA: decode, scale,
  composite, key and encode without copying pictures back to the CPU.
- **Streams the GPU does not take still reach it.** `VideoDecodeBin` decodes
  on the hardware where it can, and in software onto the same device where it
  cannot — alpha, 4:4:4, codecs without a hardware decoder — switching over
  by itself if the GPU refuses a stream at a frame. `VideoEncodeBin` does the
  same for H.264 the other way: NVENC or Media Foundation where one opens,
  software otherwise, with each path saying in the stream what its colour is.
- **Capture** of screens, windows, cameras and system or per-application audio
  on Windows and Linux; **output** to files, HLS, RTMP and RTSP servers, and
  WebRTC.
- **Observable**: per-element statistics while it runs, and a private
  structured log with the topology of every pipeline it starts.

The library crate lives in `lib/`; each directory below `examples/` is its
own crate, so platform-specific dependencies stay out of the library.

## Quick start

```toml
[dependencies]
media-pp = "0.3"
```

FFmpeg 8.0 or newer development libraries must be installed (see
[Requirements](#requirements)). `ffmpeg-next` is re-exported as
`media_pp::ffmpeg`; use that rather than depending on it separately. What
changed between versions, and what to write instead, is in [`CHANGELOG.md`].

This pipeline generates video for one second and counts the frames:

```rust
use std::time::Duration;
use media_pp::{
    elements::{FrameCounter, TestVideoOptions, TestVideoSource},
    pipeline::Pipeline,
};

fn main() -> media_pp::Result<()> {
    let source = TestVideoSource::new("source", TestVideoOptions::default());
    let (counter, frames) = FrameCounter::new("counter");
    let (pipeline, ()) = Pipeline::new("demo", source, |source, ctx| {
        let branch = ctx.branch().to(counter)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;
    pipeline.run()?;
    std::thread::sleep(Duration::from_secs(1));
    pipeline.stop();
    println!("frames: {}", frames.get());
    Ok(())
}
```

A file has an end, and a pipeline reading one says when everything read
has arrived — `BusEvent::Finished` — and waits there, where a seek could
still take it back, until it is stopped. So a file's pipeline is watched to
`Finished` and stopped then:

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

To play a file with its sound, `Player` builds the whole pipeline — the
decode, on the window's GPU where it can be, the synchronization, a window
and the default audio output — and
reports what happens to it (features `d3d11` and `wasapi-renderer` on
Windows, `vulkan` and `pipewire-audio-renderer` on Linux):

```rust,no_run
use media_pp::player::{Player, PlayerEvent, PlayerOptions};

fn main() -> media_pp::Result<()> {
    let player = Player::open("video.mp4", PlayerOptions::default())?;
    player.play()?;
    while let Some(event) = player.next_event() {
        match event {
            // Space, arrows, `.` `,` `-` `+`, F, a double click; `false` for
            // Escape or a close.
            PlayerEvent::Window(event) if !player.respond_to(&event) => break,
            PlayerEvent::Ended => break,
            // An element failed; playback goes on where it can.
            PlayerEvent::Error { name, error } => eprintln!("{name}: {error}"),
            _ => {}
        }
    }
    Ok(())
}
```

How a pipeline runs — buffers, threads, EOS, seeking, changing a running
graph — is the crate documentation's first page, and so is connecting
elements: what a branch refuses before it runs and why, how to ask whether
two elements fit before linking them (`contract::check_elements`), and which
element goes between two that do not. Each type's own page states what it
accepts, what it owns and how it fails. Both are on
[docs.rs] for the backend-independent API, and in the
[Windows API documentation] for everything.

## What is in it

Video, by backend:

| | Software | D3D11 | D3D12 | CUDA | Vulkan |
|---|---|---|---|---|---|
| Decode | `SwDecoder` | `D3d11Decoder` | `D3d12Decoder` | `CudaDecoder` | `VulkanDecoder` |
| Encode | `SwEncoder` | `D3d11VideoEncoder` | | `CudaEncoder` | `VulkanEncoder` |
| Scale, convert | `SwScaler` | `D3d11Scaler` | `D3d12Scaler` | `CudaScaler`, `CudaConverter` | |
| HDR to SDR | | `D3d11ToneMap` | | `CudaConverter` | |
| Composite | `SwVideoCompositor` | `D3d11VideoCompositor` | | `CudaVideoCompositor` | `VulkanVideoCompositor` |
| Text overlay | `SwVideoCompositorHandle::add_text_layer` | `D3d11VideoCompositorHandle::add_text_layer` | | `CudaVideoCompositorHandle::add_text_layer` | `VulkanVideoCompositorHandle::add_text_layer` |
| Key, colour | `SwChromaKey`, `SwVideoEffect` | `D3d11ChromaKey`, `D3d11VideoEffect` | | `CudaChromaKey`, `CudaVideoEffect` | |
| Upload, download | | `D3d11Upload`, `D3d11Download` | `D3d12Upload`, `D3d12Download` | `CudaUpload`, `CudaDownload` | `VulkanUpload`, `VulkanDownload` |
| Render | `VideoWindow`; every window renderer | `D3d11WindowRenderer`, `D3d11Renderer` | `D3d12WindowRenderer`, `D3d12Renderer` | `VulkanWindowRenderer`, `CudaRenderer` | |

`VideoDecodeBin` chooses among the decode row and the uploads for a stream —
Vulkan's not yet — and `VideoEncodeBin` among the encode row and the
downloads, for H.264 from system memory, D3D11 or CUDA.

Every compositor runs live by default, emitting at its own rate and drawing
whatever each input last handed over — a preview, a recording, a broadcast.
Built with `RenderMode::Offline` it renders instead: each output frame is
made as soon as every input has said what it shows at that time, placed by
the inputs' own timestamps, and the next follows at once — an export as
fast as decoding allows, frame-exact whatever rates the inputs run at.
`AudioMixer` takes the same `RenderMode`, and mixes an export's sound the
same way, placed by its inputs' timestamps.
What builds a composition can be written once for every backend, against
`VideoCompositorControl`, which each compositor's handle implements.

`VideoWindow` is the one to reach for first: a window of its own on whichever
renderer the platform has, on a GPU of its own, taking frames in system
memory — a software decode goes straight in, with no `#[cfg]` in the program.
Behind it, `D3d11WindowRenderer` and `D3d12WindowRenderer` show frames in a
window: one they open for themselves, as a GStreamer video sink does,
reporting keys, clicks and closing as `WindowEvents` and changed through a
`WindowControl`, or one the application gives them.
`VulkanWindowRenderer` does the same on Linux — its own window is X11,
XWayland on a Wayland desktop; one it is given may be either — from frames in
system memory, and from CUDA frames too on a `VulkanGpu` made for CUDA. All
three take system-memory NV12, YUV420P and BGRA and upload it themselves, so
a software decode goes straight into one; each converts a frame with its own
colour description. The other renderers hand each frame to a presenter the
program supplies — an implementation of `D3d11FrameRenderer`,
`D3d12FrameRenderer` or `CudaFrameRenderer` for its own window or UI — with
the colour description an NV12 frame came with, for it to convert by; they
are for drawing into something other than a window of the renderer's own.
The D3D11 elements of a pipeline share one `D3d11Gpu` — one device and its
immediate context behind one lock — and each is constructed from it,
`D3d11Upload::new("upload", &gpu)`; the D3D12 elements share one `D3d12Gpu`
the same way.

Everything else:

- **Inputs**: `FileDemuxer`, `RtspSource`, `WebRtcTrackSource`, `AppSource`,
  `TestVideoSource`, `TestAudioSource`.
- **Capture**: Windows — `DxgiCaptureSource` (screen), `WgcCaptureSource`
  (window), `MfCaptureSource` (camera), `WasapiCaptureSource` (system,
  application or microphone audio), `D3d11SharedTextureSource` (another
  device's textures). Linux — `PipeWireScreenCaptureSource`,
  `PipeWireAudioCaptureSource`, `V4l2CaptureSource`.
- **Audio**: `AudioMixer`, `AudioResampler`, `AudioVolume`, `AudioGate`,
  `AudioCompressor`, `AudioLimiter`, `NoiseSuppressor`, `SwAudioEncoder`,
  `AudioTempo`, which stretches sound to the playback rate for what takes
  it at the wall clock's pace, and `AudioWaveform`, which draws sound as
  pictures;
  playback through `WasapiRenderer` and `PipeWireAudioRenderer`.
- **Outputs**: `FileMuxer`, `SegmentedFileMuxer`, `ReplayBuffer`, `HlsMuxer`,
  `RtmpMuxer`, `RtspMuxer`, `WebRtcTrackSink`, `AppSink`.
- **Flow and timing**: `Queue`, `Tee`, `Rack`, `PipelineBridge`, `Pacer`,
  `VideoSynchronizer`, `ChangeGate`, `FrameRateLimiter`, `PauseGate`,
  `TimestampOrigin`.
- **Analysis**: `OrtDetector`, `WhisperTranscriber`, `FrameCounter`,
  `PacketCounter`.

## Feature flags

The library has no default features. Backend-specific types carry their
backend's prefix and exist only where their feature is enabled.

| Feature | Adds | Platform |
|---|---|---|
| `cuda` | NVDEC decode, NVENC encode, scaling, compositing, upload/download, and rendering, all on CUDA-resident frames | Linux, Windows |
| `d3d11` | D3D11 decode, scaling, upload/download, rendering, GPU compositing, and hardware encoding | Windows |
| `d3d12` | D3D12VA decode, scaling, upload/download, and rendering | Windows |
| `vulkan` | `VulkanDevice`, Vulkan Video decode and encode, compositing, and upload/download, on any GPU with a Vulkan driver; on Linux also `VulkanWindowRenderer`: video in an X11 or Wayland window — NV12, YUV420P or BGRA from system memory, or NV12 or BGRA from CUDA with `cuda` too | Linux, Windows |
| `dxgi-capture` | Desktop capture; also enables `d3d11` | Windows |
| `wgc-capture` | Individual-window capture through Windows Graphics Capture; also enables `d3d11` | Windows |
| `mf-capture` | Camera capture through Media Foundation | Windows |
| `pipewire-audio-capture` | System-audio, per-application and microphone capture through PipeWire | Linux |
| `pipewire-audio-renderer` | Audio playback through PipeWire | Linux |
| `pipewire-screen-capture` | Desktop capture through xdg-desktop-portal and PipeWire | Linux |
| `v4l2-capture` | Camera capture through Video4Linux2 | Linux |
| `wasapi-capture` | System-audio, per-application and microphone capture | Windows |
| `wasapi-renderer` | Shared-mode audio playback | Windows |
| `ort` | ONNX Runtime object detection | All supported targets |
| `rnnoise` | Noise suppression for speech, through RNNoise (pure Rust, no model file) | All supported targets |
| `whisper` | Speech to timed text, through whisper.cpp | All supported targets |
| `whisper-vulkan` | The same, on any GPU Vulkan reaches | All supported targets |
| `webrtc` | `str0m`-based WebRTC peer and track elements | All supported targets |

## Examples

- `examples/core`: decoding, queues, fan-out, dynamic tees, app sources and
  sinks, audio, remuxing, transcoding on the GPU through the decode and
  encode bins, HLS, RTMP publishing, speech-to-subtitle transcription, and
  CPU compositing.
- `examples/cuda`: headless CUDA recording and GPU text compositing, on
  Windows and Linux.
- `examples/render`: a whole player through `Player`, D3D11/D3D12 and Vulkan
  playback, desktop and window capture, synchronization, GPU scaling and compositing, chroma keying, hardware
  encoding and recording. Start with its [index](examples/render/README.md).
- `examples/rtsp`: publishing to an RTSP server such as MediaMTX, seeking
  what is published, and receiving from a server or camera — this crate does
  not serve RTSP itself.
- `examples/vision`: scaling and ONNX object detection.
- `examples/webrtc`: loopback pipelines, a two-way video call, and recording
  received tracks to MP4.

```sh
cargo run -p player -- path/to/video.mp4
cargo run -p probe -- path/to/video.mp4
cargo run -p fanout -- path/to/video.mp4
cargo run -p d3d11_scale_render -- path/to/video.mp4
```

File-based examples take a media path; none is checked in. Each prints its
usage when run without arguments, and Windows-only ones compile as stubs
elsewhere.

## Requirements

- FFmpeg 8.0 or newer development headers and libraries; the build script
  fails with a clear message on anything older. Hardware decoding also needs
  an FFmpeg build, driver and GPU that support it (`ffmpeg -hwaccels`).
- Rust 1.88 or newer.
- `pipewire-*` features: PipeWire 0.3.50 or newer development files.
- `cuda`: only the NVIDIA driver. The kernels ship as PTX the driver
  compiles, so no CUDA toolkit is needed.
- `vulkan`: an FFmpeg built with Vulkan (`--enable-vulkan`, or vcpkg's
  `ffmpeg[vulkan]`), whose `libavutil/hwcontext_vulkan.h` the build script
  generates bindings from, found through `FFMPEG_DIR` or pkg-config as FFmpeg
  itself is; and the Vulkan headers that header includes — beside FFmpeg's
  own where vcpkg put them, the Vulkan SDK's (`VULKAN_SDK`), or a
  distribution's `libvulkan-dev`. The Vulkan loader itself is opened at run
  time.

What an element needs at run time — one shared `D3d11Gpu`, one
`CudaDevice` per process, a portal for screen capture on Linux, a server to
publish RTSP to — is on that element's documentation page.

Building, testing and the stress scenarios are in
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
