# media-pp

`media-pp` is a small, GStreamer-flavored media pipeline library for Rust,
built on [`ffmpeg-next`]. It provides synchronous pipeline stages by default
and explicit thread boundaries through bounded queues.

The library crate lives in `lib/`. Each directory below `examples/` is an
independent example crate, so platform-specific dependencies do not leak into
the core library.

## Quick start

FFmpeg development libraries must be installed and discoverable by
`ffmpeg-sys-next`.

Add the crate to your project:

```toml
[dependencies]
media-pp = "0.1"
```

This minimal pipeline generates video for one second and counts the frames:

```rust
use std::{sync::atomic::Ordering, time::Duration};
use media_pp::{
    elements::{FrameCounter, TestVideoOptions, TestVideoSource},
    pipeline::Pipeline,
};

fn main() -> media_pp::Result<()> {
    media_pp::init()?;
    let source = TestVideoSource::new("source", TestVideoOptions::default());
    let (counter, frames) = FrameCounter::new("counter");
    let pipeline = Pipeline::new("demo", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(counter))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;
    pipeline.run();
    std::thread::sleep(Duration::from_secs(1));
    pipeline.stop();
    println!("frames: {}", frames.load(Ordering::Relaxed));
    Ok(())
}
```

To work with this repository directly:

```sh
cargo test -p media-pp
cargo run -p decode -- path/to/video.mp4
```

File-based examples require a media path. No media files are checked into the
repository and examples do not use a default path.

## How pipelines work

A pipeline connects a source to filters and a terminal sink:

```text
FileDemuxer → SwDecoder → Queue → Pacer → FrameCounter
```

The core types are deliberately small:

- `MediaBuffer` carries packets, video, audio, and EOS.
- `Sink::consume` is a synchronous call and may return an error.
- `SrcPad` connects one source output to one downstream sink.
- `Queue` introduces a bounded worker-thread boundary. Downstream errors are
  reported through the pipeline `Bus`, and the worker continues.
- `Pipeline` owns source threads, control flow, the shared clock, bus, and
  topology graph.
- `Tee` provides fan-out; `AudioMixer` and the video compositors provide
  fan-in.

Buffers use shared ownership, so fan-out clones references rather than media
payloads. PTS, duration, packet time bases, video color information, and EOS
are preserved through stages that do not intentionally create a new timeline.
Use `Pipeline::finish` to stop a live source with ordered EOS and drain queued
buffers, codecs, and muxers; `Pipeline::stop` abandons buffered work immediately.

## Element inventory

| Kind | Elements |
|---|---|
| Sources | `FileDemuxer`, `AppSource`, `RtspSource`, `TestVideoSource`, `TestAudioSource`, `DxgiCaptureSource`, `PipeWireScreenCaptureSource`, `PipeWireAudioCaptureSource`, `WasapiCaptureSource`, `AudioMixer`, `SwVideoCompositor`, `D3d11VideoCompositor`, `WebRtcTrackSource` |
| Filters | `SwDecoder`, `CudaDecoder`, `D3d11Decoder`, `D3d12vaDecoder`, `SwEncoder`, `CudaEncoder`, `D3d11NvencEncoder`, `SwAudioEncoder`, `AudioResampler`, `AudioVolume`, `SwScaler`, `Pacer`, `VideoSynchronizer`, `CudaScaler`, `CudaUpload`, `CudaDownload`, `D3d11Upload`, `D3d11Download`, `D3d12Upload`, `Tee` |
| Sinks | `FrameCounter`, `PacketCounter`, `AppSink`, `Mp4Muxer`, `SegmentedMp4Muxer`, `HlsMuxer`, `RtspSink`, `CudaRenderer`, `D3d11Renderer`, `D3d12Renderer`, `PipeWireAudioRenderer`, `WasapiRenderer`, `OrtDetector`, `WebRtcTrackSink` |

Backend-specific elements require their corresponding Cargo feature and are
available only on that backend's platform. See each type's Rust documentation
for buffer requirements, ownership, error behavior, and runtime-control
semantics — for example, why `DxgiCaptureSource` and
`PipeWireScreenCaptureSource` are separate types rather than one struct with a
platform switch is explained on `PipeWireScreenCaptureSource` itself.

## Examples

The examples are grouped by purpose:

- `examples/core`: decoding, queues, fan-out, dynamic tees, app sources/sinks,
  audio, muxing, HLS, and CPU compositing.
- `examples/render`: D3D11/D3D12 playback, upload, capture, synchronization,
  GPU compositing, NVENC hardware encoding, and recording.
- `examples/rtsp`: publishing, seeking, and receiving RTSP streams.
- `examples/vision`: scaling and ONNX object detection.
- `examples/webrtc`: data and encoded A/V loopback pipelines.

Useful starting points:

```sh
cargo run -p probe -- path/to/video.mp4
cargo run -p fanout -- path/to/video.mp4
cargo run -p app_sink -- path/to/video.mp4
cargo run -p scale -- path/to/video.mp4
```

Backend-specific examples enable their required library features in their own
`Cargo.toml` files, per target where an example covers more than one
platform. Each such example's module docs explain how the backends differ;
run an example without arguments to see its usage line.

## Feature flags

The library has no default features.

| Feature | Adds | Platform |
|---|---|---|
| `cuda` | NVDEC decode, NVENC encode, scaling, upload/download, and rendering, all on CUDA-resident frames | Linux, Windows |
| `d3d11` | D3D11 decode, upload/download, rendering, GPU compositing, and NVENC encoding | Windows |
| `d3d12` | D3D12VA decode, upload, and rendering interfaces | Windows |
| `dxgi-capture` | Desktop capture; also enables `d3d11` | Windows |
| `pipewire-audio-capture` | System-audio and microphone capture through PipeWire | Linux |
| `pipewire-audio-renderer` | Audio playback through PipeWire | Linux |
| `pipewire-screen-capture` | Desktop capture through xdg-desktop-portal and PipeWire | Linux |
| `wasapi-capture` | System-audio and microphone capture | Windows |
| `wasapi-renderer` | Shared-mode audio playback | Windows |
| `ort` | ONNX Runtime object detection | All supported targets |
| `webrtc` | `str0m`-based WebRTC peer and track elements | All supported targets |

For example, build all Windows API documentation locally. Nightly rustdoc is
what labels each item with the feature that enables it:

```powershell
$env:RUSTDOCFLAGS = "--cfg docsrs"
cargo +nightly doc -p media-pp --open --features d3d11,d3d12,dxgi-capture,wasapi-capture,wasapi-renderer,webrtc
```

[docs.rs] builds this crate for Linux, so it documents only the
backend-independent API and omits Windows-only types. The complete API,
including D3D11, D3D12, DXGI, and WASAPI, is available in the
[Windows API documentation] published on GitHub Pages.

## Logging

Library diagnostics use a private, opt-in logger and never install a global
`log` logger or `tracing` subscriber:

```rust
let _log_guard = media_pp::log::init(
    "media-pp",
    "./logs",
    media_pp::log::Level::Info,
    7,
)?;
```

Keep the returned guard alive until logging is no longer needed. Pipeline
starts and dynamic `Tee` changes include a stable-ID topology diagram; detailed
EOS and control propagation is available at `Trace` level. Ordinary media
buffers are not logged one record per buffer.

## Requirements and platform notes

- Install FFmpeg development headers and libraries in a location discoverable
  by `ffmpeg-sys-next`.
- Rust 1.88 or newer is required.
- D3D11VA/D3D12VA require compatible FFmpeg builds, Windows drivers, and GPU
  hardware. Check available accelerators with `ffmpeg -hwaccels`.
- D3D11 elements in one pipeline must share the same `ID3D11Device` and
  immediate context.
- `D3d11Decoder` uses a fixed-size FFmpeg surface pool; `extra_hw_frames` must
  cover the deepest downstream buffering.
- `PipeWireScreenCaptureSource` needs `libpipewire-0.3` development files, a
  running PipeWire session, and an `xdg-desktop-portal` backend implementing
  `org.freedesktop.portal.ScreenCast`. See its own Rust documentation for the
  interactive portal dialog, restore tokens, window-vs-monitor stall behavior,
  and closed-window detection this implies.
- `PipeWireAudioCaptureSource`/`PipeWireAudioRenderer` need the same PipeWire
  development files and a running session, but no portal.
- `D3d11NvencEncoder` needs an NVIDIA GPU and an FFmpeg build with NVENC. It
  fails to open with a typed error, not a panic, on any other GPU. The other
  `d3d11` elements are vendor-neutral.
- RTSP publishing requires an external server that accepts publishing, such as
  MediaMTX.
- Tests needing real media read `MEDIA_PP_TEST_VIDEO`. They skip when it is
  unset or unreadable, so set it when testing demuxing, seeking, or decoding.
- Windows-backed examples compile as unsupported stubs on other targets.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.

`media-pp` does not bundle FFmpeg. Users are responsible for complying with
the license of their FFmpeg build and optional codecs.

[`ffmpeg-next`]: https://github.com/zmwangx/rust-ffmpeg
[docs.rs]: https://docs.rs/media-pp
[Windows API documentation]: https://hash1018.github.io/media-pp/media_pp/
