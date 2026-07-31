# media-pp

A small, GStreamer-flavored media pipeline library in Rust, built on
[`ffmpeg-next`](https://github.com/zmwangx/rust-ffmpeg). `lib/` is the
library (crate name `media-pp`); `examples/` holds one independent crate
per demo pipeline.

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

| Element | Kind | What it does |
|---|---|---|
| `FileDemuxer` | Source | Demuxes a file; one src pad per container stream |
| `SwDecoder` | Filter | Decodes `Packet`s into `Video`/`Audio` frames (software, plain libavcodec) |
| `Pacer` | Filter | Sleeps in `consume` to release frames at real playback time (PTS + shared `Clock`) — must run on its own thread (behind a `Queue`) |
| `Tee` | Filter-shaped, but no `Source` | Fans one input out to a *dynamic* set of sinks via a cloneable `TeeHandle` (`add_sink`/`remove_sink`/`sink_count`), addable/removable from any thread while the pipeline runs |
| `FrameCounter` / `PacketCounter` | Sink | Count decoded frames / raw packets, expose the count via `Arc<AtomicUsize>` |
| `D3d12vaDecoder` | Filter (feature `dx12-renderer`) | Decodes `Packet`s into `Video` frames via D3D12VA hardware acceleration — GPU-resident, no software decode |
| `Dx12Renderer` | Sink (feature `dx12-renderer`) | Submits frames to a native window via [renderer-engine](https://github.com/hash1018/RendererEngine)'s DX12 `WindowRenderer`. Dispatches on `frame.format()`: `YUV420P` copies pixels up (CPU decode path); `D3D12` (from `D3d12vaDecoder`) draws zero-copy straight from the decoder's own texture |
| `RtspServer` | Sink (feature `rtsp-server`) | Spawns a vendored [MediaMTX](https://github.com/bluenviron/mediamtx) as a child process and remuxes incoming `Packet`s into it (RTSP `ANNOUNCE`/`RECORD`) — a self-contained RTSP server from the outside, no separate process to start first. Packets only, no encoding: link it straight after `FileDemuxer`, not a decoder |
| `Scaler` | Filter | Converts pixel format and resizes `Video` frames in one pass (`libswscale`) — e.g. a decoder's YUV output to the fixed RGB size an inference model expects. Source format/size is learned from the first frame it sees, not passed up front |
| `AppSink` | Sink | Hands every buffer to a plain closure — GStreamer's `appsink` equivalent, for consuming a pipeline's output without writing a dedicated `Element`/`Sink` impl |

## Examples (`examples/`)

Each is its own crate so per-example dependencies (e.g. `winit` for
`sw_decode_render`) don't leak into the others. All default to
`test-video/h265.mp4` when run with no path argument.

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `decode` | Demux → SwDecoder → FrameCounter | `SwDecoder` actually decodes, direct (same-thread) chaining |
| `probe` | Demux → Queue → PacketCounter | An explicit `Queue` thread boundary |
| `fanout` | Demux → {Queue → PacketCounter} × 2 | Multi-pad fan-out at the source (video + audio to separate branches) |
| `pace` | Demux → SwDecoder → Queue → Pacer → FrameCounter | `Pacer` releasing frames at real playback speed — compare its `wall time` output against `decode`'s near-instant run |
| `tee` | Demux → Tee → {SwDecoder → FrameCounter, PacketCounter} | `Tee` fanning the same packets out to two independent consumers |
| `sw_decode_render` | Demux → SwDecoder → Queue → Pacer → Dx12Renderer | End-to-end playback in a native window, CPU decode + CPU-upload render (Windows + DX12 only) |
| `hw_decode_render` | Demux → D3d12vaDecoder → Queue → Pacer → Dx12Renderer | Same as `sw_decode_render`, but GPU decode (D3D12VA) feeding the renderer zero-copy — no decoded pixel ever touches system memory (Windows + DX12 only) |
| `rtsp_serve` | Demux → Queue → Pacer → RtspServer | Serves a file's video as a live RTSP stream (`rtsp-server` feature) — connect with `ffplay rtsp://127.0.0.1:8554/stream` while it runs |
| `rtsp_serve_seek` | Demux → Queue → Pacer → RtspServer | Same as `rtsp_serve`, plus a terminal prompt that calls `Pipeline::seek` — jump around a live-served RTSP stream while it plays |
| `scale` | Demux → SwDecoder → Queue → Scaler → (verify) | `Scaler` converting decoded frames to a fixed RGB24 640x640 — prints the first scaled frame's actual format/size to prove the conversion really happened |
| `app_sink` | Demux → SwDecoder → AppSink | Same chain as `decode`, but the terminal sink is a plain closure instead of a bespoke `FrameCounter` — proves `AppSink` needs no dedicated type at all |

```sh
cargo run -p decode -- path/to/video.mp4   # or omit the path to use test-video/h265.mp4
cargo run -p sw_decode_render              # dx12-renderer is already enabled in its own Cargo.toml
```

## Feature flags

- `dx12-renderer` (on `media-pp`) — pulls in the optional `renderer-engine`
  git dependency (plus `windows`) and enables `Dx12Renderer` and
  `D3d12vaDecoder`. Off by default so consumers that don't render to a
  window never build DX12/Windows-only code. `sw_decode_render` and
  `hw_decode_render` turn it on in their own `Cargo.toml`.
- `rtsp-server` (on `media-pp`) — enables `RtspServer` and copies the
  vendored `mediamtx.exe` (`third_party/mediamtx/`, MIT-licensed) next to
  whatever binary depends on `media-pp` (see `lib/build.rs`). Windows-only
  for now, since only a Windows binary is vendored. `rtsp_serve` turns it
  on in its own `Cargo.toml`.

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
