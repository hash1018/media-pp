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
| `Decoder` | Filter | Decodes `Packet`s into `Video`/`Audio` frames |
| `Pacer` | Filter | Sleeps in `consume` to release frames at real playback time (PTS + shared `Clock`) — must run on its own thread (behind a `Queue`) |
| `Tee` | Filter-shaped, but no `Source` | Fans one input out to a *dynamic* set of sinks via a cloneable `TeeHandle` (`add_sink`/`remove_sink`/`sink_count`), addable/removable from any thread while the pipeline runs |
| `FrameCounter` / `PacketCounter` | Sink | Count decoded frames / raw packets, expose the count via `Arc<AtomicUsize>` |
| `Dx12Renderer` | Sink (feature `dx12-renderer`) | Submits YUV420P frames to a native window via [renderer-engine](https://github.com/hash1018/RendererEngine)'s DX12 `WindowRenderer` |

## Examples (`examples/`)

Each is its own crate so per-example dependencies (e.g. `winit` for
`render`) don't leak into the others. All default to `test-video/h265.mp4`
when run with no path argument.

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `decode` | Demux → Decoder → FrameCounter | `Decoder` actually decodes, direct (same-thread) chaining |
| `probe` | Demux → Queue → PacketCounter | An explicit `Queue` thread boundary |
| `fanout` | Demux → {Queue → PacketCounter} × 2 | Multi-pad fan-out at the source (video + audio to separate branches) |
| `pace` | Demux → Decoder → Queue → Pacer → FrameCounter | `Pacer` releasing frames at real playback speed — compare its `wall time` output against `decode`'s near-instant run |
| `tee` | Demux → Tee → {Decoder → FrameCounter, PacketCounter} | `Tee` fanning the same packets out to two independent consumers |
| `render` | Demux → Decoder → Queue → Pacer → Dx12Renderer | End-to-end playback in a native window (Windows + DX12 only) |

```sh
cargo run -p decode -- path/to/video.mp4   # or omit the path to use test-video/h265.mp4
cargo run -p render                        # dx12-renderer is already enabled in render's own Cargo.toml
```

## Feature flags

- `dx12-renderer` (on `media-pp`) — pulls in the optional `renderer-engine`
  git dependency and enables `Dx12Renderer`. Off by default so consumers
  that don't render to a window never build DX12/Windows-only code. Only
  the `render` example crate turns it on.

## Requirements

- ffmpeg installed and discoverable by `ffmpeg-sys-next` (see that crate's
  build requirements).
- `render` (and the `dx12-renderer` feature) only build/run on Windows.
