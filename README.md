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

That is the only dependency you need. `ffmpeg-next` is part of this crate's
API — `MediaBuffer` carries its frames and packets, and an encoder's
`parameters()`/`time_base` are its types — so it is re-exported as
`media_pp::ffmpeg`:

```rust
use media_pp::ffmpeg;

let time_base = ffmpeg::Rational::new(1, 30);
```

Depending on `ffmpeg-next` separately works only while that dependency
resolves to the same version this crate uses. When it does not, the compiler
sees two unrelated crates and every type above stops matching, without naming
the version as the cause.

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
    pipeline.run()?;
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

Stress and leak scenarios live in `lib/tests/soak.rs`. Each runs for tens of
seconds, so they are `#[ignore]`d and stay out of the command above:

```sh
cargo test -p media-pp --features d3d11,d3d12,cuda --test soak -- --ignored --nocapture
```

On Linux, `pipewire-screen-capture` takes the place of `d3d11`. Its two capture
scenarios also need `MEDIA_PP_SOAK_RESTORE_TOKEN`, since xdg-desktop-portal
would otherwise show its picker and block; any run of `screen_record_software`
prints a token to reuse.

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

### Link contracts

Building or attaching a branch rejects a connection that could never carry
data — feeding encoded packets to an encoder that takes decoded frames, or a
D3D11 texture to a CPU filter. The check runs before the pipeline starts and
returns `GraphError::IncompatibleLink`:

```text
decoder produces VideoFrame (System), which rec cannot accept
(it takes VideoPacket|AudioPacket)
```

It compares only what an element already knows when it is constructed. A
`PortContract` is either `Packets` — which `MediaKind`s of encoded media
(`VideoPacket`, `AudioPacket`) — or `Frames`, which decoded kinds
(`VideoFrame`, `AudioFrame`) plus the `MemoryDomain`s they may live in
(`System`, `Cuda`, `D3d11`, `D3d12`). Encoded media is always host memory, so
a packet contract has nowhere to put a domain and nowhere to forget one.
Pixel format, resolution, stride, color space, and the identity of a specific
GPU device are not part of it and stay validated against the real buffer when
it arrives.

Both halves of the kind separate buffers the `MediaBuffer` variant cannot.
The medium splits encoded data, because a container's audio and video pads
emit the same `Packet` — so wiring the audio stream into a video decoder is
caught rather than failing inside libavcodec on the first packet. The memory
domain splits decoded data, because a frame in system memory and one holding
a D3D11 texture are both `Video` — so a `SwDecoder` wired straight into a
`D3d11Scaler` with no `D3d11Upload` between them is caught too. The domain
names the backend rather than just marking a frame as "on a GPU", so a D3D11
texture handed to a CUDA filter is caught the same way.

This is not caps negotiation. Nothing selects a codec, inserts a converter,
renegotiates mid-stream, or reallocates a pool. Declaring a contract is
opt-in, and these elements declare one:

- Packet path: `FileDemuxer`, `SwDecoder`, `SwEncoder`, `SwAudioEncoder`,
  `Mp4Muxer`, `SegmentedMp4Muxer`, `HlsMuxer`, `RtspSink`, `PacketCounter`.
- Video: `SwScaler`, `SwChromaKey`, `SwVideoCompositor`, `OrtDetector`, and
  every backend's upload, download, scaler, converter, chroma key, decoder,
  encoder, renderer, and compositor (`D3d11*`, `D3d12*`, `Cuda*`).
- Audio: `AudioResampler`, `AudioVolume`, `AudioMixer`, `WasapiRenderer`,
  `PipeWireAudioRenderer`.
- Sources: `FileDemuxer`, `RtspSource`, `TestVideoSource`, `TestAudioSource`,
  the capture sources, and inbound WebRTC tracks.
- Either decoded medium: `FrameCounter`.
- Passthrough: `Queue`, `Tee`, `Pacer`, `VideoSynchronizer`. `AppSink`
  accepts anything.

`AppSource` stays undeclared, since only the application knows what it will
push. Anything else undeclared defaults to "unknown", which always links and
leaves the runtime check in charge. A passthrough element carries its
upstream contract forward, so a mismatch is still caught across a thread
boundary and still names the element that actually produces the data.

An element that genuinely handles any backend says so — `VideoSynchronizer`
paces a system frame and a device texture alike, because it never reads the
pixels, so it declares `MemoryDomainSet::ALL`. That is a claim, not a blank:
claiming a narrower domain than an element needs would refuse a pipeline that
works, which is worse than the runtime error the contract was meant to
pre-empt.

Use `Pipeline::finish` to stop a live source with ordered EOS and drain queued
buffers, codecs, and muxers; `Pipeline::stop` abandons buffered work immediately.

## Element inventory

| Kind | Elements |
|---|---|
| Sources | `FileDemuxer`, `AppSource`, `RtspSource`, `TestVideoSource`, `TestAudioSource`, `DxgiCaptureSource`, `PipeWireScreenCaptureSource`, `PipeWireAudioCaptureSource`, `WasapiCaptureSource`, `AudioMixer`, `SwVideoCompositor`, `CudaVideoCompositor`, `D3d11VideoCompositor`, `WebRtcTrackSource` |
| Filters | `SwDecoder`, `CudaDecoder`, `D3d11Decoder`, `D3d12Decoder`, `SwEncoder`, `CudaEncoder`, `D3d11NvencEncoder`, `SwAudioEncoder`, `AudioResampler`, `AudioVolume`, `SwScaler`, `SwChromaKey`, `D3d11ChromaKey`, `Pacer`, `VideoSynchronizer`, `CudaScaler`, `D3d11Scaler`, `D3d12Scaler`, `CudaUpload`, `CudaDownload`, `CudaConverter`, `D3d11Upload`, `D3d11Download`, `D3d12Upload`, `D3d12Download`, `Tee` |
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
- `examples/cuda`: headless CUDA recording and GPU text compositing. CUDA is a
  vendor backend rather than a platform one, so these build and run on both
  Windows and Linux; the `examples/render` crates of the same shape are their
  D3D11 counterparts.
- `examples/render`: D3D11/D3D12 playback, upload, capture, synchronization,
  GPU scaling/compositing, chroma keying, NVENC hardware encoding, and
  recording. The CUDA halves of the display and screen-capture examples stay
  here because their renderer (Vulkan external memory over an fd) and capture
  source (PipeWire) are genuinely Linux-only. Start with the
  [render example index](examples/render/README.md) when choosing among the
  screen preview and recording variants.
- `examples/rtsp`: publishing, seeking, and receiving RTSP streams.
- `examples/vision`: scaling and ONNX object detection.
- `examples/webrtc`: data and encoded A/V loopback pipelines, a two-way video
  call that presents both incoming tracks on Windows and Linux, and an
  all-platform H.264/Opus receive-record example that muxes both WebRTC tracks
  into MP4.

Useful starting points:

```sh
cargo run -p probe -- path/to/video.mp4
cargo run -p fanout -- path/to/video.mp4
cargo run -p app_sink -- path/to/video.mp4
cargo run -p scale -- path/to/video.mp4
cargo run -p d3d11_scale_render -- path/to/video.mp4
```

Backend-specific examples enable their required library features in their own
`Cargo.toml` files, per target where an example covers more than one
platform. Each such example's module docs explain how the backends differ;
run an example without arguments to see its usage line.

## Feature flags

The library has no default features.

| Feature | Adds | Platform |
|---|---|---|
| `cuda` | NVDEC decode, NVENC encode, scaling, compositing, upload/download, and rendering, all on CUDA-resident frames | Linux, Windows |
| `d3d11` | D3D11 decode, scaling, upload/download, rendering, GPU compositing, and NVENC encoding | Windows |
| `d3d12` | D3D12VA decode, scaling, upload/download, and rendering interfaces | Windows |
| `dxgi-capture` | Desktop capture; also enables `d3d11` | Windows |
| `pipewire-audio-capture` | System-audio and microphone capture through PipeWire | Linux |
| `pipewire-audio-renderer` | Audio playback through PipeWire | Linux |
| `pipewire-screen-capture` | Desktop capture through xdg-desktop-portal and PipeWire | Linux |
| `wasapi-capture` | System-audio and microphone capture | Windows |
| `wasapi-renderer` | Shared-mode audio playback | Windows |
| `ort` | ONNX Runtime object detection | All supported targets |
| `webrtc` | `str0m`-based WebRTC peer and track elements | All supported targets |

Each attached WebRTC source and sink exposes the codec families retained by
SDP negotiation. `WebRtcTrackSource::codec()` separately reports the codec
actually observed after RTP starts arriving, while a remotely-created
`WebRtcTrackSink` validates the application's outbound encoder choice through
`set_codec` before accepting packets. A receiver that must configure its graph
from the sender's actual payload can call `WebRtcTrackSource::wait_stream_info`
with an explicit timeout; received packets stay buffered while the downstream
graph is built. H.264 waits until actual SPS/PPS have arrived. The returned
`WebRtcStreamInfo` derives the RTP time base and purpose-independent FFmpeg
codec parameters. H.264 parameters include received SPS/PPS and dimensions,
and Opus parameters include its negotiated channel layout and `OpusHead`;
decoder and muxer compatibility is decided by the consuming element.

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

- Install FFmpeg 8.0 or newer development headers and libraries in a location
  discoverable by `ffmpeg-sys-next`. The build script reads the version
  `ffmpeg-sys-next` detected and fails with an explicit message on anything
  older, rather than letting the mismatch surface as a link or runtime error.
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
- `PipeWireScreenCaptureSource::open_gpu` (needs `cuda` as well) captures into
  CUDA surfaces instead of CPU frames, so `screen_record_nvenc` records with
  no upload element. It negotiates DMA-BUF only and fails rather than falling
  back, and it `dlopen`s the driver's `libEGL.so.1`/`libGLESv2.so.2` at run
  time — no development packages are needed to build it.
- `PipeWireAudioCaptureSource`/`PipeWireAudioRenderer` need PipeWire 0.3.50 or
  newer development files and a running session, but no portal.
- CUDA surfaces carry either NV12 or BGRA (`CudaFrameFormat`). Recording needs
  no conversion between them: NVENC ingests BGRA as directly as NV12,
  converting in hardware, so a capture recorded through `CudaEncoder` stays
  BGRA end to end. `CudaVideoCompositor` and `CudaRenderer` work in NV12
  instead, and `CudaConverter` is what a BGRA capture goes through to reach
  them — with a kernel of this crate's own, since `scale_cuda` resizes but has
  no RGB-to-YUV kernel and `CudaScaler` therefore does not convert.
- A `CudaDevice` opens the device's primary CUDA context, so create one per
  process before starting pipelines rather than per pipeline: creating or
  dropping one while another thread is decoding or encoding can crash inside
  the NVIDIA driver.
- `CudaVideoCompositor` composites NV12 CUDA surfaces with `scale_cuda`, 2D
  device-to-device copies, and one small blend kernel, so every `VideoFit`
  and `opacity` works as it does on the other backends — `Cover` needs
  cropping that no CUDA filter offers, and translucency needs arithmetic no
  copy can do. The kernel ships as PTX text that the driver JIT-compiles at
  startup, so no CUDA toolkit is involved. Layer placement and size are
  aligned to even pixels, since NV12 chroma is subsampled. It also draws text
  layers (`CudaTextLayerHandle`), sharing the glyph rasterizer with the D3D11
  compositor and blending the coverage with the same kernel.
- The `cuda` feature links the NVIDIA driver library directly (`libcuda.so`
  on Linux, `nvcuda.dll` on Windows) for those copies and for the blend
  kernel. No CUDA toolkit is needed — the driver ships both the library and
  the PTX compiler.
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
