# media-pp

`media-pp` is a small, GStreamer-flavored media pipeline library for Rust,
built on [`ffmpeg-next`]. Stages are synchronous calls by default, and thread
boundaries are explicit bounded queues.

The library crate lives in `lib/`. Each directory below `examples/` is an
independent example crate, so platform-specific dependencies do not leak into
the core library. This README is the overview; each type's own Rust
documentation carries its buffer requirements, ownership, error behavior and
runtime-control semantics — on [docs.rs] for the backend-independent API, and
in the [Windows API documentation] for everything.

## Quick start

FFmpeg 8.0 or newer development libraries must be installed and discoverable
by `ffmpeg-sys-next` (see [Requirements](#requirements)).

```toml
[dependencies]
media-pp = "0.2"
```

0.2.0 renames two elements and takes a pair of binding methods away; see
[`CHANGELOG.md`] for what to write instead.

`ffmpeg-next` is part of this crate's API — `MediaBuffer` carries its frames
and packets — so it is re-exported as `media_pp::ffmpeg`. Use that rather
than a separate `ffmpeg-next` dependency: if the two resolve to different
versions, every shared type stops matching and the compiler does not say why.

This pipeline generates video for one second and counts the frames:

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
    pipeline.run()?;
    std::thread::sleep(Duration::from_secs(1));
    pipeline.stop();
    println!("frames: {}", frames.load(Ordering::Relaxed));
    Ok(())
}
```

## How pipelines work

A pipeline connects a source to filters and a terminal sink:

```text
FileDemuxer → SwDecoder → Queue → Pacer → FrameCounter
```

- `MediaBuffer` carries packets, video, audio, and EOS. Payloads are shared,
  so fan-out clones references, not media. PTS, duration, time bases and
  color information survive every stage that does not deliberately start a
  new timeline.
- `Sink::consume` is a synchronous call and may return an error;
  `Sink::ready_consume` carries downstream readiness back up, so
  backpressure does not consume or drop the next buffer.
- `SrcPad` connects one output to one downstream sink.
- `Queue` is the thread boundary. Past it an error can no longer be returned
  to the caller, so it is reported on the pipeline's `Bus` and the worker
  continues.
- `Pipeline` owns the source threads, control flow, clock, bus and topology
  graph. An element is given the clock, playback clock and bus when it is
  wired (`Element::attach_context`) rather than taking them in its
  constructor, so it cannot be handed another pipeline's.
- `Pipeline::finish` ends with an ordered EOS that drains codecs and muxers;
  `Pipeline::stop` abandons buffered work.
- `Pipeline::stats` reads what every element is doing — buffers and packet
  bytes, time inside `consume`, how long it has been idle, errors, a
  `Queue`'s fill and drops, a compositor's frames drawn and missed — as
  running totals, so two readings give a rate.

### Changing a running pipeline

- `Tee` fans out; `AudioMixer` and the video compositors fan in.
  `TeeHandle::attach` adds a branch, `finish_branch` ends one cleanly (so a
  recording finalizes while its preview keeps running), and `detach`
  abandons one.
- `Rack` is a stretch of chain whose contents are replaced between two
  buffers, so a filter can be added or removed without reopening the source.
- `PipelineBridge` carries buffers from one pipeline into another, so a
  source that dies takes only its own pipeline with it.
- Compositor and capture frame rates (`set_frame_rate`, `FrameRateHandle`)
  and the `AudioMixer`'s format (`set_mix_format`) change while running.
  Both re-mean the timestamps that follow, so they are for a preview, not
  the middle of a recording.
- `FileDemuxerHandle` makes a file loop, carrying the timeline across each
  lap so pacing and muxing continue.

### Seeking

A source says whether it is live and whether it is seekable. `Pipeline::seek`
first asks every branch whether it can follow (a recording muxer cannot), and
changes nothing if one refuses; then it runs `Pause → Flush → Seek →
Preroll` and restores the state the caller had. `SeekMode::Accurate` decodes
forward to the exact target; `SeekMode::Keyframe` shows the keyframe the
demuxer landed on.

### Link contracts

Building a branch refuses a connection that could never carry data — encoded
packets into an encoder that takes frames, a container's audio stream into a
video decoder, a D3D11 texture into a CPU or CUDA filter — before anything
runs:

```text
decoder produces VideoFrame (System), which rec cannot accept
(it takes VideoPacket|AudioPacket)
```

It compares only what elements know when they are constructed: the media
kind, and for decoded frames the memory domain (`System`, `Cuda`, `D3d11`,
`D3d12`). It is not caps negotiation — nothing is converted or renegotiated —
and pixel format, size and device identity are still checked against the
real buffer. An element that declares nothing always links. See the
`contract` module.

## Element inventory

| Kind | Elements |
|---|---|
| Sources | `FileDemuxer`, `AppSource`, `RtspSource`, `TestVideoSource`, `TestAudioSource`, `DxgiCaptureSource`, `WgcCaptureSource`, `MfCaptureSource`, `V4l2CaptureSource`, `PipeWireScreenCaptureSource`, `PipeWireAudioCaptureSource`, `WasapiCaptureSource`, `AudioMixer`, `SwVideoCompositor`, `CudaVideoCompositor`, `D3d11VideoCompositor`, `WebRtcTrackSource` |
| Filters | `SwDecoder`, `CudaDecoder`, `D3d11Decoder`, `D3d12Decoder`, `SwEncoder`, `CudaEncoder`, `D3d11VideoEncoder`, `SwAudioEncoder`, `AudioResampler`, `AudioVolume`, `AudioGate`, `AudioCompressor`, `AudioLimiter`, `NoiseSuppressor`, `SwScaler`, `SwChromaKey`, `CudaChromaKey`, `D3d11ChromaKey`, `SwVideoEffect`, `CudaVideoEffect`, `D3d11VideoEffect`, `Pacer`, `VideoSynchronizer`, `CudaScaler`, `D3d11Scaler`, `D3d12Scaler`, `CudaUpload`, `CudaDownload`, `CudaConverter`, `D3d11Upload`, `D3d11Download`, `D3d12Upload`, `D3d12Download`, `Tee`, `ChangeGate`, `TimestampOrigin`, `Rack` |
| Sinks | `FrameCounter`, `PacketCounter`, `AppSink`, `FileMuxer`, `SegmentedFileMuxer`, `HlsMuxer`, `RtmpMuxer`, `RtspMuxer`, `CudaRenderer`, `D3d11Renderer`, `D3d12Renderer`, `PipeWireAudioRenderer`, `WasapiRenderer`, `OrtDetector`, `WhisperTranscriber`, `WebRtcTrackSink` |

Backend-specific elements need their Cargo feature and exist only on that
backend's platform. On Windows, `DxgiCaptureSource` captures a monitor and
`WgcCaptureSource` one window by its `HWND`; both either create the D3D11
device the rest of the pipeline uses or take one you already have.

## Examples

- `examples/core`: decoding, queues, fan-out, dynamic tees, app sources and
  sinks, audio, remuxing, HLS, RTMP publishing, speech-to-subtitle
  transcription, and CPU compositing.
- `examples/cuda`: headless CUDA recording and GPU text compositing, on
  Windows and Linux.
- `examples/render`: D3D11/D3D12 playback, desktop and window capture,
  synchronization, GPU scaling and compositing, chroma keying, hardware
  encoding and recording. Start with its [index](examples/render/README.md).
- `examples/rtsp`: publishing, seeking, and receiving RTSP streams.
- `examples/vision`: scaling and ONNX object detection.
- `examples/webrtc`: loopback pipelines, a two-way video call, and recording
  received tracks to MP4.

```sh
cargo run -p probe -- path/to/video.mp4
cargo run -p fanout -- path/to/video.mp4
cargo run -p d3d11_scale_render -- path/to/video.mp4
```

File-based examples take a media path; none is checked in. Each example
enables the library features it needs, and prints its usage when run without
arguments.

## Feature flags

The library has no default features.

| Feature | Adds | Platform |
|---|---|---|
| `cuda` | NVDEC decode, NVENC encode, scaling, compositing, upload/download, and rendering, all on CUDA-resident frames | Linux, Windows |
| `d3d11` | D3D11 decode, scaling, upload/download, rendering, GPU compositing, and hardware encoding | Windows |
| `d3d12` | D3D12VA decode, scaling, upload/download, and rendering | Windows |
| `dxgi-capture` | Desktop capture; also enables `d3d11` | Windows |
| `wgc-capture` | Individual-window capture through Windows Graphics Capture; also enables `d3d11` | Windows |
| `mf-capture` | Camera capture through Media Foundation | Windows |
| `pipewire-audio-capture` | System-audio and microphone capture through PipeWire | Linux |
| `pipewire-audio-renderer` | Audio playback through PipeWire | Linux |
| `pipewire-screen-capture` | Desktop capture through xdg-desktop-portal and PipeWire | Linux |
| `v4l2-capture` | Camera capture through Video4Linux2 | Linux |
| `wasapi-capture` | System-audio and microphone capture | Windows |
| `wasapi-renderer` | Shared-mode audio playback | Windows |
| `ort` | ONNX Runtime object detection | All supported targets |
| `rnnoise` | Noise suppression for speech, through RNNoise (pure Rust, no model file) | All supported targets |
| `whisper` | Speech to timed text, through whisper.cpp | All supported targets |
| `whisper-vulkan` | The same, on any GPU Vulkan reaches | All supported targets |
| `webrtc` | `str0m`-based WebRTC peer and track elements | All supported targets |

[docs.rs] builds for Linux and so omits the Windows-only API. To build the
complete documentation locally, labelled by feature:

```powershell
$env:RUSTDOCFLAGS = "--cfg docsrs"
cargo +nightly doc -p media-pp --open --features d3d11,d3d12,dxgi-capture,wgc-capture,mf-capture,wasapi-capture,wasapi-renderer,webrtc
```

## Logging

Diagnostics go to a private, opt-in logger; nothing installs a global `log`
logger or `tracing` subscriber:

```rust
let _log_guard = media_pp::log::init("media-pp", "./logs", media_pp::log::Level::Info, 7)?;
```

Keep the guard alive for as long as logs should be written. Pipeline starts
and `Tee` changes log a topology diagram with stable element ids; EOS and
control propagation are logged at `Trace`. Media buffers are never logged one
record per buffer.

## Requirements

Building:

- FFmpeg 8.0 or newer development headers and libraries. The build script
  fails with a clear message on anything older.
- Rust 1.88 or newer.
- `pipewire-*` features: PipeWire 0.3.50 or newer development files.
- `cuda`: only the NVIDIA driver. The kernels ship as PTX the driver
  compiles, so no CUDA toolkit is needed.

Running:

- D3D11VA/D3D12VA need an FFmpeg build, driver and GPU that support them
  (`ffmpeg -hwaccels` lists what yours has).
- Every D3D11 element in a pipeline must share one `ID3D11Device`.
- `D3d11VideoEncoder`'s NVENC codecs need an NVIDIA GPU and an FFmpeg built
  with NVENC; its Media Foundation codecs work on Intel, AMD and NVIDIA. A
  codec the machine lacks fails to open with a typed error.
- Create one `CudaDevice` per process, before starting pipelines. It opens
  the device's primary context, and creating or dropping one while another
  thread decodes or encodes can crash the NVIDIA driver.
- `PipeWireScreenCaptureSource` needs a running PipeWire session and an
  `xdg-desktop-portal` backend with ScreenCast. Its documentation covers the
  portal dialog and restore tokens.
- RTSP publishing needs a server that accepts publishers, such as MediaMTX.
- Windows-only examples compile as stubs on other targets.

## Testing

```sh
cargo test -p media-pp
```

Tests need no media: they synthesize their fixture from the crate's own
sources and encoders, so every machine tests the same file.

Stress and leak scenarios in `lib/tests/soak.rs` run for tens of seconds and
are `#[ignore]`d. They read a real recording from `MEDIA_PP_TEST_VIDEO`:

```sh
cargo test -p media-pp --features d3d11,d3d12,cuda --test soak -- --ignored --nocapture
```

On Linux, `pipewire-screen-capture` takes the place of `d3d11`, and the
capture scenarios also need `MEDIA_PP_SOAK_RESTORE_TOKEN`, since the portal
would otherwise show its picker; any run of `screen_record_software` prints
a token to reuse.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.

`media-pp` does not bundle FFmpeg. Users are responsible for complying with
the license of their FFmpeg build and optional codecs.

[`CHANGELOG.md`]: https://github.com/hash1018/media-pp/blob/main/CHANGELOG.md
[`ffmpeg-next`]: https://github.com/zmwangx/rust-ffmpeg
[docs.rs]: https://docs.rs/media-pp
[Windows API documentation]: https://hash1018.github.io/media-pp/media_pp/
