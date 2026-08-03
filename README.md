# media-pp

A small, GStreamer-flavored media pipeline library in Rust, built on
[`ffmpeg-next`](https://github.com/zmwangx/rust-ffmpeg). `lib/` is the
library (crate name `media-pp`); `examples/` holds one independent crate
per demo pipeline, grouped into subdirectories by theme (`core/`,
`render/`, `rtsp/`, `vision/`, `webrtc/`).

## Architecture

Everything is built from a handful of primitives in `lib/src/`:

- **`Element`** (`element.rs`) — a named node in the graph. Just identity;
  says nothing about input/output.
- **`Sink`** (`element.rs`) — anything that can receive a buffer
  (`consume(&mut self, buf: MediaBuffer)`). The only "connection"
  primitive in the pipeline. By default, consuming a buffer is a plain
  function call on the caller's thread — zero overhead.
- **`Source`** (`element.rs`) — anything with one or more output ports
  (`src_pads()`). An element with more than one src pad *is* a tee — fan-out
  needs no separate primitive (see `FileDemuxer`, which exposes one pad per
  container stream).
- **`SourceElement`** (`element.rs`) — a pure source that drives its own
  thread via `run()` (e.g. `FileDemuxer`, wrapping blocking file I/O).
- **`Filter`** (`element.rs`) — anything that's both a `Source` and a
  `Sink` (decoder, pacer, ...). Auto-implemented for any `T: Source + Sink`;
  no separate "processing element" trait needed.
- **`SrcPad`** (`pad.rs`) — an output port; links to exactly one
  downstream `Sink`.
- **`MediaBuffer`** (`buffer.rs`) — the unit of data flowing between
  elements (`Packet` / `Video` / `Audio` / `Eos`). Payloads are
  `Arc`-wrapped, so cloning a buffer (e.g. to fan it out) is a refcount
  bump, never a copy of the encoded/decoded data.
- **`Queue`** (`queue.rs`) — the *only* thread boundary in the system.
  Wrapping a `Sink` in a `Queue` hands buffers off through a bounded
  channel to a dedicated worker thread. Elements never spawn their own
  threads; boundaries are introduced explicitly wherever one is wanted.
- **`Bus` / `BusEvent`** (`bus.rs`) — a cross-thread event channel. Once a
  buffer crosses a `Queue` boundary, errors can't propagate with `?`
  anymore, so they're posted here instead (`Error`, `Eos`, `Dropped`).
  `BusReceiver::log_events()` drains and prints them in a default format.
- **`Pipeline` / `ChainBuilder`** (`pipeline.rs`) — `ChainBuilder` builds
  one linear chain (`.pipe(filter)` for same-thread stages, `.queue(name,
  capacity)` for a thread boundary, `.build(sink)` to terminate);
  `Pipeline` drives a `SourceElement` on the calling thread until EOS and
  every `Queue` worker thread has drained and joined.
- **`Clock`** (`clock.rs`) — a shared wall-clock anchor (`Arc<Clock>`) so
  multiple `Pacer`s (e.g. one per stream) agree on the same t=0.

## Elements (`lib/src/elements/`)

One-line index only — each element's own doc comment (`cargo doc --open`)
has the full rationale (why it's built the way it is, what to watch out
for); this table isn't meant to duplicate that.

### Sources

| Element | What it does |
|---|---|
| `FileDemuxer` | Demuxes a file; one src pad per container stream |
| `AppSource` | Application code pushes buffers in via a handle, from any thread — GStreamer's `appsrc` equivalent |
| `RtspSource` | Demuxes a live RTSP stream (the client/receive counterpart to `RtspServer`) — no internal retry/reconnect on a dropped connection, fails fast instead; the caller rebuilds a fresh one to reconnect |
| `TestVideoSource` | Generates a synthetic moving-gradient `Pixel::YUV420P` stream — GStreamer's `videotestsrc` equivalent, no file/camera/decoder needed |
| `WebRtcPeer` (`webrtc`) | Drives one str0m `Rtc` session on its own thread. Not a `Pipeline` source itself — `WebRtcHandle::add_track`/`next_track()` mint a `WebRtcTrackSink`+`WebRtcTrackSource` pair per track (see below), symmetric for tracks either side added, so one `Direction::SendRecv` track carries both directions |
| `WebRtcTrackSource` (`webrtc`) | The receive side of one WebRTC track — a plain `SourceElement`, same shape as `AppSource`; obtained via `WebRtcHandle::next_track()`, not constructed directly |

### Filters

| Element | What it does |
|---|---|
| `SwDecoder` | Decodes `Packet`s into `Video`/`Audio` frames (software) |
| `D3d12vaDecoder` (`dx12-renderer`) | Decodes into GPU-resident `Video` frames via D3D12VA hardware acceleration |
| `SwEncoder` | Encodes `Video` frames into `Packet`s (software only) — `VideoCodec` picks H.264/H.265/VP8/VP9/AV1 across GPL (`libx264`/`libx265`) and non-GPL (`libopenh264`/`libkvazaar`/`libvpx`/`libaom-av1`/`libsvtav1`) encoders; fails with a clear error, not a panic, if the linked ffmpeg build doesn't have the one you asked for |
| `Pacer` | Releases buffers at real playback speed (PTS + a shared `Clock`) |
| `Scaler` | Converts pixel format and resizes `Video` frames in one pass (`libswscale`) |
| `Tee`¹ | Fans one input out to a dynamic set of sinks, addable/removable while the pipeline runs |

¹ Doesn't actually implement `Source` — its pads live behind a lock instead of a plain `&mut [SrcPad]`, so a handle on another thread can add/remove one mid-`consume`. See its own doc comment.

### Sinks

| Element | What it does |
|---|---|
| `FrameCounter` / `PacketCounter` | Count decoded frames / raw packets, expose the count via `Arc<AtomicUsize>` |
| `Dx12Renderer` (`dx12-renderer`) | Submits frames to a `FrameRenderer` impl — zero-copy for `D3d12vaDecoder`'s frames. `media-pp` only defines the trait (plus `RawPlane`/`SubmitError`); the actual DX12 window rendering lives in each example's own `renderer-engine` dependency |
| `RtspServer` (`rtsp-server`) | Spawns a vendored MediaMTX and remuxes packets into it as a live RTSP stream |
| `AppSink` | Hands buffers (and, optionally, control messages) to plain closures — GStreamer's `appsink` equivalent |
| `OrtDetector` (`ort`) | Runs a YOLOv8/v11-style ONNX model on each frame via `ort`, hands decoded/NMS-filtered detections to a closure |
| `WebRtcTrackSink` (`webrtc`) | The send side of one WebRTC track — `consume()` hands off to its `WebRtcPeer`'s own thread; handed out by `WebRtcHandle::next_track()`, not `WebRtcHandle::add_track` (which only returns a `TrackId`) |

## Examples (`examples/`)

Each is its own crate so per-example dependencies (e.g. `winit` for
`sw_decode_render`) don't leak into the others. All default to
`test-video/h265.mp4` when run with no path argument.

### Core concepts

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `decode` | Demux → SwDecoder → FrameCounter | `SwDecoder` actually decodes, direct (same-thread) chaining |
| `probe` | Demux → Queue → PacketCounter | An explicit `Queue` thread boundary |
| `fanout` | Demux → {Queue → PacketCounter} × 2 | Multi-pad fan-out at the source (video + audio to separate branches) |
| `pace` | Demux → SwDecoder → Queue → Pacer → FrameCounter | `Pacer` releasing frames at real playback speed — compare its `wall time` output against `decode`'s near-instant run |
| `tee` | Demux → Tee → {SwDecoder → FrameCounter, PacketCounter} | `Tee` fanning the same packets out to two independent consumers |
| `app_sink` | Demux → SwDecoder → AppSink | Same chain as `decode`, but the terminal sink is a plain closure instead of a bespoke `FrameCounter` |
| `app_source` | AppSource → SwDecoder → FrameCounter | A background thread feeds packets in via `AppSourceHandle`, standing in for whatever a real external producer would push from |

### Playback (Windows + DX12 only)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `sw_decode_render` | Demux → SwDecoder → Queue → Pacer → Dx12Renderer | End-to-end playback in a native window, CPU decode + CPU-upload render |
| `hw_decode_render` | Demux → D3d12vaDecoder → Queue → Pacer → Dx12Renderer | Same, but GPU decode feeding the renderer zero-copy — no decoded pixel ever touches system memory |
| `test_video` | TestVideoSource → Queue → Pacer → Dx12Renderer | A synthetic moving-gradient stream rendered directly (no file/camera/decoder) — proves `TestVideoSource`'s frames and `Dx12Renderer`'s CPU-upload path work end to end |
| `transcode_render` | TestVideoSource → Queue → SwEncoder → Queue → SwDecoder → Queue → Pacer → Dx12Renderer | Encodes the synthetic stream (`libopenh264`) and decodes it straight back, no container/mux involved — proves `SwEncoder`'s `Packet`s are actually valid, decodable bitstream, not just "opened successfully" |
| `seek_render` | Demux → SwDecoder → Queue → Pacer → Dx12Renderer | Same chain as `sw_decode_render`, plus a terminal prompt that calls `Pipeline::seek` while the window is open |

All five of the above build their `Dx12Renderer` through `render_common` (`examples/render/render_common`), a small shared crate holding the one `renderer_engine::window_renderer::WindowRenderer` → `media_pp::elements::FrameRenderer` adapter, instead of each example hand-copying it. `media-pp` itself still doesn't depend on `renderer-engine` — only `render_common` and whichever example needs a `RendererEngine` (e.g. to pass `engine.device()` into `D3d12vaDecoder`) do.

### RTSP streaming (`rtsp-server` feature)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `rtsp_serve` | Demux → Queue → Pacer → RtspServer | Serves a file's video as a live RTSP stream — connect with `ffplay rtsp://127.0.0.1:8554/stream` while it runs |
| `rtsp_serve_seek` | Demux → Queue → Pacer → RtspServer | Same, plus a terminal prompt that calls `Pipeline::seek` — jump around the live stream while it plays |

### RTSP client (no extra feature — just `ffmpeg-next`)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `rtsp_source` | RtspSource → Queue → PacketCounter | Connects to a real RTSP server/camera (TCP transport by default), counts video packets for a fixed window, then stops — `RtspSource` is the client/receive counterpart to `RtspServer` |

### Inference-pipeline building blocks

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `scale` | Demux → SwDecoder → Queue → Scaler → (verify) | `Scaler` converting decoded frames to a fixed RGB24 640x640 — prints the first scaled frame's actual format/size to prove the conversion really happened |
| `detect` | Demux → SwDecoder → Queue → Scaler → OrtDetector | `OrtDetector` running a YOLOv8/v11 ONNX model on the scaled frames and printing every detection |

### WebRTC (`webrtc` feature)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `webrtc_loopback` | Two `WebRtcPeer`s over loopback UDP | One `Direction::SendRecv` track (opened by `WebRtcHandle::add_track` on one side, accepted via `WebRtcHandle::accept_remote_offer` on the other) carrying data both ways over the *same* `Mid` — no second negotiation for the reverse direction. No browser/signaling server: real ICE/DTLS-SRTP over loopback UDP |

```sh
cargo run -p decode -- path/to/video.mp4   # or omit the path to use test-video/h265.mp4
cargo run -p sw_decode_render              # dx12-renderer is already enabled in its own Cargo.toml
```

## Feature flags

- `dx12-renderer` (on `media-pp`) — pulls in `windows` and enables
  `Dx12Renderer`, `FrameRenderer`, `RawPlane`, `SubmitError`, and
  `D3d12vaDecoder`. `media-pp` itself has no dependency on
  `renderer-engine` at all — `Dx12Renderer` takes a `Box<dyn
  FrameRenderer>`, and it's each example's own job to adapt a concrete
  renderer (e.g. `renderer-engine`'s `WindowRenderer`, via a small local
  wrapper) to that trait. Off by default so consumers that don't render
  to a window never build DX12/Windows-only code. `sw_decode_render`,
  `hw_decode_render`, `test_video`, `transcode_render`, and `seek_render`
  turn it on in their own `Cargo.toml`.
- `rtsp-server` (on `media-pp`) — enables `RtspServer` and copies the
  vendored `mediamtx.exe` (`third_party/mediamtx/`, MIT-licensed) next to
  whatever binary depends on `media-pp` (see `lib/build.rs`). Windows-only
  for now, since only a Windows binary is vendored. `rtsp_serve` turns it
  on in its own `Cargo.toml`.
- `ort` (on `media-pp`) — pulls in the `ort` crate (ONNX Runtime bindings;
  downloads a prebuilt onnxruntime binary at build time) and `ndarray`, and
  enables `OrtDetector`. `detect` turns it on in its own `Cargo.toml`.
- `webrtc` (on `media-pp`) — pulls in `str0m` (sans-I/O WebRTC, `wincrypto`
  backend — native Windows crypto, no OpenSSL vendoring) and enables
  `WebRtcPeer`/`WebRtcHandle`/`WebRtcTrackSink`/`WebRtcTrackSource`. The initial SDP
  offer/answer and ICE candidate setup happen via str0m directly, in the
  caller's own code, *before* constructing a `WebRtcPeer` — same posture
  as `RtspServer` not managing RTSP client connections itself; there's no
  signaling server built in. `webrtc_loopback` turns it on in its own
  `Cargo.toml`.

## Requirements

- ffmpeg installed and discoverable by `ffmpeg-sys-next` (see that crate's
  build requirements). `D3d12vaDecoder` additionally needs an ffmpeg build
  with `d3d12va` hwaccel support (check `ffmpeg -hwaccels`) and a GPU/driver
  that supports it.
- `sw_decode_render`/`hw_decode_render` (and the `dx12-renderer` feature)
  only build/run on Windows.
- `D3d12vaDecoder` hand-mirrors a few structs from FFmpeg's
  `libavutil/hwcontext_d3d12va.h` that `ffmpeg-sys-next` doesn't bind
  (see the doc comment at the top of `d3d12va_decoder.rs`) — sourced from
  FFmpeg n8.0's header. A future FFmpeg version changing that header's
  layout would silently break this with no compile-time warning.
