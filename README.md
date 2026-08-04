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
| `DxgiScreenSource` (`dxgi-capture`) | Captures the desktop live via DXGI Desktop Duplication — GStreamer's `d3d11screencapturesrc` equivalent. Pushes `Pixel::BGRA` untouched (chain a `Scaler` for YUV420P); emits at a constant `fps` (default 30, same convention as `TestVideoSource`) rather than one push per real desktop change — repeats the latest captured image if nothing changed since the last tick, since a variable-rate/push-on-change version of this turned out to cause visible judder against a vsync-locked renderer. `CaptureMode::Cpu` (default, optional cursor compositing) or `CaptureMode::Gpu { device }` — zero-copy straight to a `Pixel::D3D11` texture on a caller-supplied `ID3D11Device` shared with the rest of a D3D11 pipeline, no `Map`/CPU pixel copy at all (no cursor support yet in this mode) |
| `WebRtcPeer` (`webrtc`) | Drives one str0m `Rtc` session on its own thread. Not a `Pipeline` source itself — `WebRtcHandle::add_track`/`next_track()` mint a `WebRtcTrackSink`+`WebRtcTrackSource` pair per track (see below), symmetric for tracks either side added, so one `Direction::SendRecv` track carries both directions |
| `WebRtcTrackSource` (`webrtc`) | The receive side of one WebRTC track — a plain `SourceElement`, same shape as `AppSource`; obtained via `WebRtcHandle::next_track()`, not constructed directly |

### Filters

| Element | What it does |
|---|---|
| `SwDecoder` | Decodes `Packet`s into `Video`/`Audio` frames (software) |
| `D3d12vaDecoder` (`d3d12-renderer`) | Decodes into GPU-resident `Video` frames via D3D12VA hardware acceleration |
| `D3d11Decoder` (`d3d11-renderer`) | Decodes into GPU-resident `Video` frames via D3D11VA hardware acceleration — the D3D11 sibling of `D3d12vaDecoder`. `extra_hw_frames` matters here in a way it doesn't for D3D12: D3D11VA's decode surface pool is fixed-size, sized once at open time, so it must cover the deepest downstream queue/buffer or decode itself starts failing once the pool runs out |
| `D3d11Upload` (`d3d11-renderer`) | Uploads CPU-resident `Pixel::NV12` frames to a GPU-resident `Pixel::D3D11` texture — the D3D11 sibling of `D3d12Upload`. Doesn't go through FFmpeg's own hwframe-pool machinery at all (an earlier version that did corrupted memory); builds the `ID3D11Texture2D` directly via plain `windows-rs` calls instead |
| `SwEncoder` | Encodes `Video` frames into `Packet`s (software only) — `VideoCodec` picks H.264/H.265/VP8/VP9/AV1 across GPL (`libx264`/`libx265`) and non-GPL (`libopenh264`/`libkvazaar`/`libvpx`/`libaom-av1`/`libsvtav1`) encoders; fails with a clear error, not a panic, if the linked ffmpeg build doesn't have the one you asked for |
| `Pacer` | Releases buffers at real playback speed (PTS + a shared `Clock`) |
| `Scaler` | Converts pixel format and resizes `Video` frames in one pass (`libswscale`) |
| `Tee`¹ | Fans one input out to a dynamic set of sinks, addable/removable while the pipeline runs |

¹ Doesn't actually implement `Source` — its pads live behind a lock instead of a plain `&mut [SrcPad]`, so a handle on another thread can add/remove one mid-`consume`. See its own doc comment.

### Sinks

| Element | What it does |
|---|---|
| `FrameCounter` / `PacketCounter` | Count decoded frames / raw packets, expose the count via `Arc<AtomicUsize>` |
| `D3d12Renderer` (`d3d12-renderer`) | Submits frames to a `D3d12FrameRenderer` impl — zero-copy for `D3d12vaDecoder`'s frames. `media-pp` only defines the trait (plus `RawPlane`/`SubmitError`); the actual DX12 window rendering lives in `examples/render/render_common`'s own `D3d12WindowRenderer` |
| `D3d11Renderer` (`d3d11-renderer`) | Submits frames to a `D3d11FrameRenderer` impl — zero-copy for `D3d11Upload`/`D3d11Decoder`/`DxgiScreenSource`'s GPU mode. No fence, no `keep_alive` (unlike `D3d12FrameRenderer`): every producer in this crate's D3D11 stack shares one `ID3D11Device`+context, and D3D11's own driver-deferred resource destruction means the runtime — not this crate — keeps a texture alive for as long as the GPU still needs it. `examples/render/render_common`'s own `D3d11WindowRenderer` is the concrete implementation |
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

### Playback (Windows only)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `sw_decode_render` | Demux → SwDecoder → Queue → Pacer → D3d12Renderer | End-to-end playback in a native window, CPU decode + CPU-upload render |
| `hw_decode_render` | Demux → D3d12vaDecoder → Queue → Pacer → D3d12Renderer | Same, but GPU decode feeding the renderer zero-copy — no decoded pixel ever touches system memory |
| `d3d11_decode_render` | Demux → D3d11Decoder → Queue → Pacer → D3d11Renderer | The D3D11 sibling of `hw_decode_render` — GPU decode via D3D11VA, zero-copy render. What actually proved `D3d11Decoder` safe on real hardware: `D3d11Decoder` never touches FFmpeg's `hw_frames_ctx` struct layout itself (only `bind_flags`, via the documented `avcodec_get_hw_frames_parameters` API, from inside `get_format`) — unlike an earlier, abandoned attempt at manual `AVD3D11VAFramesContext` construction, which corrupted memory |
| `test_video` | TestVideoSource → Queue → D3d12Renderer | A synthetic moving-gradient stream rendered directly (no file/camera/decoder, no `Pacer`) — proves `TestVideoSource`'s frames and `D3d12Renderer`'s CPU-upload path work end to end. Confirmed smooth without a `Pacer`: `TestVideoSource` self-paces on a drift-free absolute schedule, which turned out to be what actually mattered (see `screen_capture`, which confirmed the same thing even with a `Scaler` in between); `transcode_render` (below) keeps one, since its `SwEncoder`/`SwDecoder` stages have their own real per-frame variance, untested without |
| `transcode_render` | TestVideoSource → Queue → SwEncoder → Queue → SwDecoder → Queue → Pacer → D3d12Renderer | Encodes the synthetic stream (`libopenh264`) and decodes it straight back, no container/mux involved — proves `SwEncoder`'s `Packet`s are actually valid, decodable bitstream, not just "opened successfully" |
| `seek_render` | Demux → SwDecoder → Queue → Pacer → D3d12Renderer | Same chain as `sw_decode_render`, plus a terminal prompt that calls `Pipeline::seek` while the window is open |
| `screen_capture` | DxgiScreenSource (CPU mode) → Queue → Scaler → Queue → D3d12Renderer | Live desktop capture (DXGI Desktop Duplication, cursor included) at a constant frame rate, converted/resized to the window's own size and rendered directly, no `Pacer`. Confirmed smooth without one: an earlier, variable-rate version of `DxgiScreenSource` measurably needed a `Pacer` here to avoid judder, but once it moved to constant-rate, drift-free-scheduled emission (same pattern as `TestVideoSource`), `Scaler` alone wasn't enough to bring the judder back |
| `screen_capture_gpu` | DxgiScreenSource (GPU mode) → Queue → D3d11Renderer | The zero-copy sibling of `screen_capture`: captures straight to a GPU-resident `Pixel::D3D11` BGRA texture on the renderer's own `ID3D11Device` — no `Map`, no CPU pixel copy, no `Scaler` (desktop content is already BGRA/RGB). No cursor (`CaptureMode::Gpu` doesn't support it yet) |
| `d3d12_upload` | TestVideoSource → Queue → Scaler → Queue → D3d12Upload → Queue → D3d12Renderer | A CPU `Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU, then uploaded to a GPU `Pixel::D3D12` texture on the renderer's own device via `D3d12Upload` before being presented zero-copy — proves `D3d12Upload`'s frames are structurally identical to `D3d12vaDecoder`'s own, so `D3d12Renderer` takes its zero-copy path unmodified even though nothing here ever decoded anything |
| `d3d11_upload` | TestVideoSource → Queue → Scaler → Queue → D3d11Upload → Queue → D3d11Renderer | The D3D11 sibling of `d3d12_upload`, same proof for `D3d11Upload`/`D3d11Renderer` |

The D3D12 examples above build their `D3d12Renderer`, and the D3D11 ones their `D3d11Renderer`, through `render_common` (`examples/render/render_common`) — a small shared crate holding its own minimal window renderers (`D3d12GpuContext`/`D3d12WindowRenderer` for D3D12, `D3d11GpuContext`/`D3d11WindowRenderer` for D3D11) instead of each example hand-copying them. `media-pp` itself has no dependency on any rendering crate at all — only `render_common` depends on `windows`' D3D11/D3D12/DXGI bindings to actually draw. The two stacks are independent (separate device, separate shader set) — nothing shares a device across them.

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
cargo run -p sw_decode_render              # d3d12-renderer is already enabled in its own Cargo.toml
```

## Feature flags

- `d3d12-renderer` (on `media-pp`) — pulls in `windows` and enables
  `D3d12Renderer`, `D3d12FrameRenderer`, `RawPlane`, `SubmitError`,
  `D3d12vaDecoder`, and `D3d12Upload`. `media-pp` itself has no dependency
  on any rendering crate at all — `D3d12Renderer` takes a `Box<dyn
  D3d12FrameRenderer>`, and it's each example's own job to provide a
  concrete renderer (`examples/render/render_common`'s own
  `D3d12WindowRenderer`) implementing that trait. Off by default so
  consumers that don't render to a window never build DX12/Windows-only
  code. Every `examples/render/*` crate turns it on in its own
  `Cargo.toml`.
- `d3d11-renderer` (on `media-pp`) — pulls in `windows` and enables
  `D3d11Renderer`, `D3d11FrameRenderer`, `D3d11Decoder`, `D3d11Upload`, and
  `SubmitError` (shared with `d3d12-renderer`). Independent of
  `d3d12-renderer` — separate device, separate shader set, nothing shared
  between the two stacks. Off by default, same reasoning as
  `d3d12-renderer`. Every `d3d11_*`/`screen_capture_gpu` example crate
  turns it on in its own `Cargo.toml`.
- `dxgi-capture` (on `media-pp`) — pulls in `windows` (DXGI sub-features)
  and enables `DxgiScreenSource`/`CaptureMode`. Requires `d3d11-renderer`
  (`DxgiScreenOptions`' `CaptureMode::Gpu` produces a `Pixel::D3D11` frame
  the same way `D3d11Upload` does, via the same shared helper) — enabling
  `dxgi-capture` pulls `d3d11-renderer` in automatically. Windows-only.
  `screen_capture`/`screen_capture_gpu` turn it on in their own
  `Cargo.toml` (alongside `d3d12-renderer`/`d3d11-renderer` respectively,
  to actually render what they capture).
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
  build requirements). `D3d12vaDecoder`/`D3d11Decoder` additionally need an
  ffmpeg build with `d3d12va`/`d3d11va` hwaccel support respectively (check
  `ffmpeg -hwaccels`) and a GPU/driver that supports it.
- All `examples/render/*` crates (and the
  `d3d12-renderer`/`d3d11-renderer`/`dxgi-capture` features themselves)
  only build/run on Windows.
- `D3d12vaDecoder` hand-mirrors a few structs from FFmpeg's
  `libavutil/hwcontext_d3d12va.h` that `ffmpeg-sys-next` doesn't bind
  (see the doc comment at the top of
  `elements/filter/decoder/d3d12va_decoder.rs`) — sourced from FFmpeg
  n8.0's header. A future FFmpeg version changing that header's layout
  would silently break this with no compile-time warning.
  `D3d11Decoder`/`D3d11Upload`/`DxgiScreenSource`'s GPU mode do the same
  for a couple of small D3D11VA-specific structs
  (`elements/filter/decoder/d3d11va_decoder.rs`),
  but deliberately touch only a handful of already-initialized fields
  (never construct FFmpeg's `AVHWFramesContext` from scratch) — an earlier
  version that did corrupted memory badly enough to trip `/GS`
  (`STATUS_STACK_BUFFER_OVERRUN`), for a reason never fully root-caused;
  see that file's own doc comments for the history.
- `D3d11Decoder`'s decode surface pool is fixed-size (unlike D3D12VA's) —
  its `extra_hw_frames` parameter must cover whatever the deepest
  downstream queue/buffer can hold, or decode itself starts failing once
  the pool runs out (see its own doc comment).
- Every D3D11 element in one pipeline (`D3d11Decoder`, `D3d11Upload`,
  `D3d11Renderer`, `DxgiScreenSource`'s GPU mode) must share exactly one
  `ID3D11Device` — that's what lets this stack skip explicit GPU-side
  fences entirely, unlike the D3D12 side (see `D3d11Renderer`'s own doc
  comment for why).
