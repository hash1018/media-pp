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
- **`SrcPad`** (`pad.rs`) — an output port with exactly one runtime peer.
  Applications connect it through `Context::attach`, so the runtime peer
  and topology graph cannot diverge.
- **`MediaBuffer`** (`buffer.rs`) — the unit of data flowing between
  elements (`Packet` / `Video` / `Audio` / `Eos`). Payloads are
  `Arc`-wrapped, so cloning a buffer (e.g. to fan it out) is a refcount
  bump, never a copy of the encoded/decoded data.
- **`Queue`** (`queue.rs`) — the explicit thread boundary between stages
  in a dataflow chain.
  Wrapping a `Sink` in a `Queue` hands buffers off through a bounded
  channel to a dedicated worker thread. Directly-linked stages stay on
  their upstream caller's thread; the top-level `Pipeline` and
  `DriverRunner` separately own the background threads that drive a source
  or driver.
- **`Bus` / `BusEvent`** (`bus.rs`) — a cross-thread event channel. Once a
  buffer crosses a `Queue` boundary, errors can't propagate with `?`
  anymore, so they're posted here instead (`Error`, `Eos`, `Dropped`).
  `BusReceiver::iter_with_ids()` pairs each event with its stable graph
  `ElementId`; `log_events()` drains and prints them in a default format.
- **`PpLog`** (`pp_log.rs`) — the contextual log identity stored privately by
  every element. `pp_info!`, `pp_debug!`, `pp_warn!`, `pp_error!`, and `pp_trace!` attach
  that identity as the `log` target; `element_pp_log()` creates the canonical
  `ElementType(name):Pipeline(id)` identity used throughout the graph.
- **`Pipeline` / `ChainBuilder` / `PipelineBuilder`** (`pipeline.rs`) —
  `ctx.branch()` builds one linear, detached chain (`.pipe(filter)` for
  same-thread stages, `.queue(name, capacity)` for a thread boundary,
  `.to(sink)` to terminate). `ctx.attach(source, pad, branch)` commits the
  runtime connection and graph in one operation. `Pipeline::run()` drives
  a `SourceElement` on a background thread and returns immediately;
  draining its bus waits for that source and every reachable `Queue`
  worker to finish, provided the application has not retained an extra
  `Context`/`Bus` sender. `Pipeline::new` is the
  single-source case; `PipelineBuilder::new(id).add_source(source, wire)…`
  combines more than one independent `SourceElement` (e.g. a video capture
  and an audio capture both feeding one `Mp4Muxer`) into one `Pipeline` —
  each source gets its own thread, but they share one `Bus`/`Clock`/
  graph, and `run`/`pause`/`resume`/`stop`/`seek` reach every source
  from a single call.
- **`PipelineGraph`** (`graph.rs`) — the live node/edge graph. Elements,
  edges, and dynamic branches use stable `ElementId`/`EdgeId`/`BranchId`
  values; names are display labels only. `Pipeline::graph()` returns a
  revisioned, consistent snapshot, and `Pipeline::topology()` renders it.
  A detached branch never appears until attachment succeeds.
- **`Clock`** (`clock.rs`) — a shared wall-clock anchor (`Arc<Clock>`) so
  multiple `Pacer`s (e.g. one per stream) agree on the same t=0.
- **`PlaybackClock`** (`playback_clock.rs`) — a shared media-position clock.
  Video-only playback derives it from `Clock`; a bound audio renderer can
  take over with its actual played-sample position without moving the
  timeline backwards.

## Elements (`lib/src/elements/`)

One-line index only — each element's own doc comment (`cargo doc --open`)
has the full rationale (why it's built the way it is, what to watch out
for); this table isn't meant to duplicate that.

### Sources

| Element | What it does |
|---|---|
| `FileDemuxer` | Demuxes a file; one src pad per container stream |
| `AppSource` | Application code pushes buffers in via a handle, from any thread — GStreamer's `appsrc` equivalent |
| `RtspSource` | Demuxes a live RTSP stream (the receive counterpart to `RtspSink`) — no internal retry/reconnect on a dropped connection, fails fast instead; the caller rebuilds a fresh one to reconnect |
| `TestVideoSource` | Generates a synthetic moving-gradient `Pixel::YUV420P` stream — GStreamer's `videotestsrc` equivalent, no file/camera/decoder needed |
| `TestAudioSource` | Generates a synthetic sine-tone `Sample::F32(Packed)` audio stream — the audio sibling of `TestVideoSource`, no file/microphone/decoder needed |
| `DxgiCaptureSource` (`dxgi-capture`) | Captures the desktop live via DXGI Desktop Duplication — GStreamer's `d3d11screencapturesrc` equivalent. Pushes `Pixel::BGRA` untouched (chain a `Scaler` for YUV420P); emits at a constant `fps` (default 30, same convention as `TestVideoSource`) rather than one push per real desktop change — repeats the latest captured image if nothing changed since the last tick, since a variable-rate/push-on-change version of this turned out to cause visible judder against a vsync-locked renderer. `CaptureMode::Cpu` (default, optional cursor compositing) or `CaptureMode::Gpu` — the GPU mode resolves the capture adapter, creates its own `ID3D11Device`, and returns that device from `open()` so the renderer and other D3D11 stages can share it; capture then emits zero-copy `Pixel::D3D11` textures with no `Map`/CPU pixel copy (no cursor support yet in this mode) |
| `WasapiCaptureSource` (`wasapi-capture`) | Captures audio live via WASAPI — either a playback endpoint's own outgoing mix (loopback, i.e. system audio — the audio counterpart to record alongside `DxgiCaptureSource`) or a microphone, picked from `WasapiCaptureSource::list_devices()` |
| `AudioMixer`¹ | Live-mixes any number of inputs, attachable/detachable while running via `MixerHandle::add_source`/`remove_source` (`add_source` returns a terminal `Sink` that a different pipeline can pass to `ctx.branch().to(...)`) — the fan-in counterpart to `Tee`'s fan-out |
| `VideoCompositor` | Composites the latest frames from independently-driven input pipelines into a fixed-rate BGRA output, entirely on the CPU via `libswscale` + a hand-written alpha blend. Each latest-frame slot uses an atomic `ArcSwapOption` instead of a Mutex; `add_source` returns both a terminal Sink and a `VideoLayerHandle` for runtime position, size, opacity, visibility, fit, and z-order changes. `VideoRect`/`VideoLayer`/`VideoFit` and the `layer_geometry` math live in a shared `video_layer` module; colors use the crate-wide `Color` type so `D3d11VideoCompositor` uses the exact same layer-control API |
| `D3d11VideoCompositor` (`d3d11-renderer`) | The GPU sibling of `VideoCompositor` — same `add_source`/`VideoLayerHandle` API, but every input must already be a `Pixel::D3D11` texture (BGRA or NV12) and compositing happens via a D3D11 pixel shader into an offscreen render target, never touching the CPU. Draws each layer as a screen-covering triangle (no vertex buffer) clipped by viewport+scissor to its `VideoRect`; output textures are returned to a growable pool only after the last downstream frame reference is dropped, so queued/Tee'd frames are never overwritten. NV12 conversion follows each frame's color-space/range metadata (with an SD=BT.601, HD=BT.709 limited-range fallback when unspecified) |
| `WebRtcPeer` (`webrtc`) | Drives one str0m `Rtc` session on its own thread. Not a `Pipeline` source itself — `WebRtcHandle::add_track`/`next_track()` mint a `WebRtcTrackSink`+`WebRtcTrackSource` pair per track (see below), symmetric for tracks either side added, so one `Direction::SendRecv` track carries both directions |
| `WebRtcTrackSource` (`webrtc`) | The receive side of one WebRTC track — a plain `SourceElement`, same shape as `AppSource`; obtained via `WebRtcHandle::next_track()`, not constructed directly |

¹ Each input is driven from wherever it was attached — typically a *different* `Pipeline`/thread than the one the `AudioMixer` itself is the source of, which is the whole point (e.g. a capture pipeline feeding a mixer that another pipeline reads from). For combining a fixed, known-up-front set of live sources into one output instead (no dynamic attach/detach needed), see `PipelineBuilder` — a simpler fit for e.g. one video capture + one audio capture feeding a single `Mp4Muxer`.

### Filters

| Element | What it does |
|---|---|
| `SwDecoder` | Decodes `Packet`s into `Video`/`Audio` frames (software) |
| `D3d12vaDecoder` (`d3d12-renderer`) | Decodes into GPU-resident `Video` frames via D3D12VA hardware acceleration |
| `D3d11Decoder` (`d3d11-renderer`) | Decodes into GPU-resident `Video` frames via D3D11VA hardware acceleration — the D3D11 sibling of `D3d12vaDecoder`. `extra_hw_frames` matters here in a way it doesn't for D3D12: D3D11VA's decode surface pool is fixed-size, sized once at open time, so it must cover the deepest downstream queue/buffer or decode itself starts failing once the pool runs out |
| `D3d11Upload` (`d3d11-renderer`) | Uploads CPU-resident `Pixel::NV12` frames to a GPU-resident `Pixel::D3D11` texture — the D3D11 sibling of `D3d12Upload`. Doesn't go through FFmpeg's own hwframe-pool machinery at all (an earlier version that did corrupted memory); builds the `ID3D11Texture2D` directly via plain `windows-rs` calls instead |
| `D3d11Download` (`d3d11-renderer`) | The mirror of `D3d11Upload` — downloads a GPU-resident `Pixel::D3D11` BGRA texture (e.g. from `D3d11VideoCompositor`) back to a CPU-resident `Pixel::BGRA` frame via `CopySubresourceRegion`/`Map` into a cached staging texture, including a selected slice of a texture array. Needed because `SwEncoder` is software-only and has no zero-copy GPU input path; chain a `Scaler` after this for whatever pixel format the encoder actually needs |
| `SwEncoder` | Encodes `Video` frames into `Packet`s (software only) — `VideoCodec` picks H.264/H.265/VP8/VP9/AV1 across GPL (`libx264`/`libx265`) and non-GPL (`libopenh264`/`libkvazaar`/`libvpx`/`libaom-av1`/`libsvtav1`) encoders; fails with a clear error, not a panic, if the linked ffmpeg build doesn't have the one you asked for |
| `SwAudioEncoder` | Encodes `Audio` frames into `Packet`s (software `aac`) — resamples to whatever format/channel layout the codec actually needs, built lazily from the first frame it sees |
| `AudioResampler` | Converts decoded `Audio` sample format/rate/channels through `libswresample`; its explicit input time base preserves the media PTS across conversion, and it flushes delayed samples at EOS |
| `AudioVolume` | Applies runtime-adjustable gain/mute through `AudioVolumeHandle`; uses a configurable 10 ms default ramp to prevent clicks and preserves the input audio format/timestamps |
| `Pacer` | Releases buffers at real playback speed (PTS + a shared `Clock`) — `new` rejects an invalid `time_base` with a typed `PacerError` rather than panicking, since it comes from a demuxed/externally supplied stream |
| `VideoSynchronizer` | Replaces `Pacer` for A/V playback: uses the pipeline wall clock for video-only playback, then automatically waits/drops video against a registered audio playback master |
| `Scaler` | Converts pixel format and resizes `Video` frames in one pass (`libswscale`) |
| `Tee`² | Fans one input out to multiple branches; `TeeBuilder` defines the initial fan-out and `TeeHandle::attach`/`detach` changes runtime branches by stable `BranchId` |

² Doesn't actually implement `Source` — its pads live in individually locked branch slots instead of a plain `&mut [SrcPad]`. `consume` only holds the branch-list lock long enough to clone an `Arc` snapshot, so a slow downstream does not block unrelated `TeeHandle::attach`/`detach` operations. Detach prevents a push that has not started yet; one already executing downstream call may finish. See its own doc comment.

### Sinks

| Element | What it does |
|---|---|
| `FrameCounter` / `PacketCounter` | Count decoded frames / raw packets, expose the count via `Arc<AtomicUsize>` |
| `Mp4Muxer`³ | Muxes one or more `Packet` streams — encoder output (`SwEncoder`/`SwAudioEncoder`) or a `FileDemuxer`'s own streams for a pure remux — into an MP4 file, one or more tracks |
| `SegmentedMp4Muxer`⁴ | Same shape as `Mp4Muxer`, but cuts to a new file every so often (`SegmentPolicy::Duration`) instead of writing one file for the whole recording — e.g. `rec_000.mp4`, `rec_001.mp4`, ... — so a crash mid-recording only loses the currently-open segment |
| `HlsMuxer`⁵ | Muxes one or more encoded packet streams into an HLS media playlist with MPEG-TS or fMP4 segments; supports sliding live windows, EVENT/VOD playlists, atomic manifest replacement, and optional deletion of expired live segments |
| `D3d12Renderer` (`d3d12-renderer`) | Submits frames to a `D3d12FrameRenderer` impl — zero-copy for `D3d12vaDecoder`'s frames. `media-pp` only defines the trait (plus `RawPlane`/`SubmitError`); the actual DX12 window rendering lives in `examples/render/render_common`'s own `D3d12WindowRenderer` |
| `D3d11Renderer` (`d3d11-renderer`) | Submits frames to a `D3d11FrameRenderer` impl — zero-copy for `D3d11Upload`/`D3d11Decoder`/`DxgiCaptureSource`'s GPU mode. No fence, no `keep_alive` (unlike `D3d12FrameRenderer`): every producer in this crate's D3D11 stack shares one `ID3D11Device`+context, and D3D11's own driver-deferred resource destruction means the runtime — not this crate — keeps a texture alive for as long as the GPU still needs it. `examples/render/render_common`'s own `D3d11WindowRenderer` is the concrete implementation |
| `WasapiRenderer` (`wasapi-renderer`) | Plays decoded audio through a WASAPI shared-mode render endpoint. `open()` returns the endpoint's `AudioFormat`; place `AudioResampler` before it and a `Queue` at the blocking device boundary |
| `RtspSink` | Publishes one compressed packet stream to an already-running RTSP server; it remuxes rather than re-encoding and works with any server that accepts RTSP publishing |
| `AppSink` | Hands buffers (and, optionally, control messages) to plain closures — GStreamer's `appsink` equivalent |
| `OrtDetector` (`ort`) | Runs a YOLOv8/v11-style ONNX model on each frame via `ort`, hands decoded/NMS-filtered detections to a closure |
| `WebRtcTrackSink` (`webrtc`) | The send side of one WebRTC track — `consume()` hands off to its `WebRtcPeer`'s own thread; handed out by `WebRtcHandle::next_track()`, not `WebRtcHandle::add_track` (which only returns a `TrackId`) |

³ Not a plain `Sink` itself — `Mp4Muxer::create`/`add_stream`/`open` is a two-phase builder, since a container's header has to describe every track's codec parameters before it can be written at all. `create` opens the file, `add_stream` registers one track at a time (name + `codec::Parameters` + `time_base`), and `open` writes the header and returns one real `Sink` per track, in registration order — all sharing one lock around the file, so tracks fed from independently-threaded branches (e.g. one video encode chain, one audio encode chain) can write concurrently without racing. The trailer is written once *every* track reports done (`Eos` or `Stop`), not on whichever finishes first. See its own doc comment, and `PipelineBuilder` for wiring two independent live sources (e.g. video + audio capture) into the tracks it expects.

⁴ Same two-phase builder shape as `Mp4Muxer` (`create`/`add_stream`/`open`), plus a naming closure (`FnMut(u64) -> PathBuf`, called with the segment index) instead of one fixed path. A rotation only actually cuts once the configured duration has elapsed *and* the video track's next packet is a keyframe — never mid-GOP — so every segment file is independently decodable from its own frame 0, closing the outgoing segment (writing its trailer) via the exact same all-tracks-report-done mechanism `Mp4Muxer` already uses for a normal `Eos`/`Stop`. Building this is what surfaced a real gap in `SwEncoder`: it now always sets a ~2-second keyframe interval itself, since at least one codec (`libopenh264`) was found to otherwise go an entire recording without a second keyframe against smoothly-changing content — relying on scene-change detection alone, which would have meant a `SegmentedMp4Muxer` using it might never rotate at all.

⁵ `HlsOptions::new` defaults to live fMP4 with two-second target segments and a six-entry sliding window. `HlsMode` selects live/EVENT/VOD behavior and `HlsSegmentFormat` selects fMP4 or MPEG-TS. Like the other muxers, `HlsMuxer::open` returns one sink per registered track and writes `#EXT-X-ENDLIST` only after every track reports `Eos` or `Stop`. Segment timing is media-timestamp based inside FFmpeg's HLS muxer; video segments cut on keyframes, so the encoder GOP should be close to the requested segment duration.

## Examples (`examples/`)

Each is its own crate so per-example dependencies (e.g. `winit` for
`sw_decode_render`) don't leak into the others. File-based playback/core
examples that accept an optional media path generally default to
`test-video/h265.mp4`; live capture, WebRTC, and synthetic-source examples
have their own arguments or need none.

### Core concepts

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `decode` | Demux → SwDecoder → FrameCounter | `SwDecoder` actually decodes, direct (same-thread) chaining |
| `probe` | Demux → Queue → PacketCounter | An explicit `Queue` thread boundary |
| `fanout` | Demux → {Queue → PacketCounter} × 2 | Multi-pad fan-out at the source (video + audio to separate branches) |
| `pace` | Demux → SwDecoder → Queue → Pacer → FrameCounter | `Pacer` releasing frames at real playback speed — compare its `wall time` output against `decode`'s near-instant run |
| `tee` | Demux → Tee → {SwDecoder → FrameCounter, PacketCounter} | `TeeBuilder` committing a fixed initial fan-out as one subgraph |
| `dynamic_tee` | TestVideoSource → Tee → {FrameCounter, runtime FrameCounter} | `TeeHandle` attaching and detaching a branch while frames flow |
| `app_sink` | Demux → SwDecoder → AppSink | Same chain as `decode`, but the terminal sink is a plain closure instead of a bespoke `FrameCounter` |
| `app_source` | AppSource → SwDecoder → FrameCounter | A background thread feeds packets in via `AppSourceHandle`, standing in for whatever a real external producer would push from |
| `audio_record` | TestAudioSource → SwAudioEncoder → Mp4Muxer | Encodes a synthetic sine tone straight into a playable `.mp4` — `Mp4Muxer`'s single-track path, the audio counterpart to `transcode_render`'s `SwEncoder` proof |
| `audio_playback` (`wasapi-renderer`) | TestAudioSource → AudioResampler → AudioVolume → Queue → WasapiRenderer | Lists render endpoints and demonstrates runtime gain/mute changes while playing a three-second tone in the selected device's native mix format |
| `video_compositor` | TestVideoSource × 2 → VideoCompositor → Scaler → SwEncoder → Mp4Muxer | Composites two independently-paced inputs and moves one layer at runtime without changing source connections |
| `hls` | TestVideoSource → SwEncoder → HlsMuxer | Writes a live fMP4 `index.m3u8`, `init.mp4`, and keyframe-aligned `.m4s` segments with a sliding playlist window |
| `remux` | FileDemuxer → Mp4Muxer (one track per kept stream) | Remuxes a file's video + audio streams into a new `.mp4` with no decode/re-encode — `Mp4Muxer`'s multi-track builder driven by a single source's multiple `src_pads`, packets passed through untouched |

### Recording (Windows only)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `audio_capture` (`wasapi-capture`) | WasapiCaptureSource → FrameCounter | Lists WASAPI endpoints, captures ~3s from one (system-audio loopback by default, or a microphone), reports how many buffers came through |
| `screen_record` (`dxgi-capture`) | DxgiCaptureSource → Scaler → SwEncoder → Mp4Muxer | Headless desktop recording straight to `.mp4` — no window, no renderer (compare `screen_capture`, which renders instead of encoding) |
| `screen_audio_record` (`dxgi-capture` + `wasapi-capture`) | DxgiCaptureSource + WasapiCaptureSource → Mp4Muxer | Desktop + system-audio recording combined into one file — two independent live sources driven by one `PipelineBuilder`-built `Pipeline`, both tracks finalized together; stops on `q` + Enter in the terminal |

### Playback (Windows only)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `sw_decode_render` | Demux → SwDecoder → Queue → Pacer → D3d12Renderer | End-to-end playback in a native window, CPU decode + CPU-upload render |
| `av_playback` | Demux → {SwDecoder → Queue → VideoSynchronizer → D3d12Renderer, dynamic Tee → SwDecoder → AudioResampler → Queue → WasapiRenderer} | Starts video-only, then accepts terminal commands to attach/detach WASAPI audio and seek; video switches between wall-clock pacing and the played-audio master without rebuilding the pipeline |
| `hw_decode_render` | Demux → D3d12vaDecoder → Queue → Pacer → D3d12Renderer | Same, but GPU decode feeding the renderer zero-copy — no decoded pixel ever touches system memory |
| `d3d11_decode_render` | Demux → D3d11Decoder → Queue → Pacer → D3d11Renderer | The D3D11 sibling of `hw_decode_render` — GPU decode via D3D11VA, zero-copy render. What actually proved `D3d11Decoder` safe on real hardware: `D3d11Decoder` never touches FFmpeg's `hw_frames_ctx` struct layout itself (only `bind_flags`, via the documented `avcodec_get_hw_frames_parameters` API, from inside `get_format`) — unlike an earlier, abandoned attempt at manual `AVD3D11VAFramesContext` construction, which corrupted memory |
| `test_video` | TestVideoSource → Queue → D3d12Renderer | A synthetic moving-gradient stream rendered directly (no file/camera/decoder, no `Pacer`) — proves `TestVideoSource`'s frames and `D3d12Renderer`'s CPU-upload path work end to end. Confirmed smooth without a `Pacer`: `TestVideoSource` self-paces on a drift-free absolute schedule, which turned out to be what actually mattered (see `screen_capture`, which confirmed the same thing even with a `Scaler` in between); `transcode_render` (below) keeps one, since its `SwEncoder`/`SwDecoder` stages have their own real per-frame variance, untested without |
| `transcode_render` | TestVideoSource → Queue → SwEncoder → Queue → SwDecoder → Queue → Pacer → D3d12Renderer | Encodes the synthetic stream (`libopenh264`) and decodes it straight back, no container/mux involved — proves `SwEncoder`'s `Packet`s are actually valid, decodable bitstream, not just "opened successfully" |
| `seek_render` | Demux → SwDecoder → Queue → Pacer → D3d12Renderer | Same chain as `sw_decode_render`, plus a terminal prompt that calls `Pipeline::seek` while the window is open |
| `screen_capture` | DxgiCaptureSource (CPU mode) → Queue → Scaler → Queue → D3d12Renderer | Live desktop capture (DXGI Desktop Duplication, cursor included) at a constant frame rate, converted/resized to the window's own size and rendered directly, no `Pacer`. Confirmed smooth without one: an earlier, variable-rate version of `DxgiCaptureSource` measurably needed a `Pacer` here to avoid judder, but once it moved to constant-rate, drift-free-scheduled emission (same pattern as `TestVideoSource`), `Scaler` alone wasn't enough to bring the judder back |
| `screen_capture_gpu` | DxgiCaptureSource (GPU mode) → Queue → D3d11Renderer | The zero-copy sibling of `screen_capture`: captures straight to a GPU-resident `Pixel::D3D11` BGRA texture on the renderer's own `ID3D11Device` — no `Map`, no CPU pixel copy, no `Scaler` (desktop content is already BGRA/RGB). No cursor (`CaptureMode::Gpu` doesn't support it yet) |
| `d3d12_upload` | TestVideoSource → Queue → Scaler → Queue → D3d12Upload → Queue → D3d12Renderer | A CPU `Pixel::YUV420P` stream converted to `Pixel::NV12` on the CPU, then uploaded to a GPU `Pixel::D3D12` texture on the renderer's own device via `D3d12Upload` before being presented zero-copy — proves `D3d12Upload`'s frames are structurally identical to `D3d12vaDecoder`'s own, so `D3d12Renderer` takes its zero-copy path unmodified even though nothing here ever decoded anything |
| `d3d11_upload` | TestVideoSource → Queue → Scaler → Queue → D3d11Upload → Queue → D3d11Renderer | The D3D11 sibling of `d3d12_upload`, same proof for `D3d11Upload`/`D3d11Renderer` |
| `gpu_video_compositor` | TestVideoSource × 2 → Scaler(NV12) → D3d11Upload → D3d11VideoCompositor → Tee → {D3d11Renderer, D3d11Download → Scaler → SwEncoder → Mp4Muxer} | The GPU sibling of `video_compositor`: composites two GPU-resident inputs with a moving PiP layer entirely via shader, then fans the composited output to a live window *and* a recording, proving one `D3d11VideoCompositor` frame serves both a display consumer and a CPU-readback consumer without being recomposed |

The D3D12 examples above build their `D3d12Renderer`, and the D3D11 ones their `D3d11Renderer`, through `render_common` (`examples/render/render_common`) — a small shared crate holding its own minimal window renderers (`D3d12GpuContext`/`D3d12WindowRenderer` for D3D12, `D3d11GpuContext`/`D3d11WindowRenderer` for D3D11) instead of each example hand-copying them. `media-pp` has no dependency on any *window*-rendering crate — only `render_common` depends on `windows`' DXGI swap-chain bindings to actually present to an `HWND`. `D3d11VideoCompositor` is the one exception to "no shader code in `media-pp`": compositing is window-independent (pure texture-to-texture), so its D3D11 pipeline/shader setup lives directly in `lib` rather than being pushed out to example code the way window presentation is. The D3D11/D3D12 stacks remain independent (separate device, separate shader set) — nothing shares a device across them.

### RTSP publishing

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `rtsp_serve` | Demux → Queue → Pacer → RtspSink | Publishes a file's video to an already-running RTSP server; pass the file and publishing URL as arguments |
| `rtsp_serve_seek` | Demux → Queue → Pacer → RtspSink | Same, plus terminal commands for pause, resume, seek, and stop while publishing |

### RTSP client (no extra feature — just `ffmpeg-next`)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `rtsp_source` | RtspSource → Queue → PacketCounter | Connects to a real RTSP server/camera (TCP transport by default), counts video packets for a fixed window, then stops — `RtspSource` is the receive counterpart to `RtspSink` |

### Inference-pipeline building blocks

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `scale` | Demux → SwDecoder → Queue → Scaler → (verify) | `Scaler` converting decoded frames to a fixed RGB24 640x640 — prints the first scaled frame's actual format/size to prove the conversion really happened |
| `detect` | Demux → SwDecoder → Queue → Scaler → OrtDetector | `OrtDetector` running a YOLOv8/v11 ONNX model on the scaled frames and printing every detection |

### WebRTC (`webrtc` feature)

| Crate | Pipeline | Demonstrates |
|---|---|---|
| `webrtc_loopback` | Two `WebRtcPeer`s over loopback UDP | One `Direction::SendRecv` track (opened by `WebRtcHandle::add_track` on one side, accepted via `WebRtcHandle::accept_remote_offer` on the other) carrying data both ways over the *same* `Mid` — no second negotiation for the reverse direction. No browser/signaling server: real ICE/DTLS-SRTP over loopback UDP |
| `webrtc_av_loopback` | TestVideoSource → SwEncoder → WebRtcTrackSink, TestAudioSource → SwAudioEncoder → WebRtcTrackSink (two `PipelineBuilder` sources) | Two tracks — one video, one audio — negotiated onto the *same* `WebRtcPeer` connection (two sequential `add_track` renegotiations, one `Rtc`/socket/peer pair), each carrying real encoded media; peer-b counts packets per track to prove they arrive independently, no cross-contamination |

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
  code. The D3D12-based render examples turn it on in their own
  `Cargo.toml`; D3D11-only examples enable `d3d11-renderer` instead.
- `d3d11-renderer` (on `media-pp`) — pulls in `windows` (including
  `Win32_Graphics_Direct3D_Fxc`, needed for `D3d11VideoCompositor`'s own
  runtime `D3DCompile` calls — the one place this crate compiles HLSL
  itself, everywhere else that's `render_common`'s job) and enables
  `D3d11Renderer`, `D3d11FrameRenderer`, `D3d11Decoder`, `D3d11Upload`,
  `D3d11Download`, `D3d11VideoCompositor` (+ handle types), and
  `SubmitError` (shared with `d3d12-renderer`). Independent of
  `d3d12-renderer` — separate device, separate shader set, nothing shared
  between the two stacks. Off by default, same reasoning as
  `d3d12-renderer`. Every `d3d11_*`/`screen_capture_gpu`/
  `gpu_video_compositor` example crate turns it on in its own
  `Cargo.toml`.
- `dxgi-capture` (on `media-pp`) — pulls in `windows` (DXGI sub-features)
  and enables `DxgiCaptureSource`/`CaptureMode`. Requires `d3d11-renderer`
  (`DxgiCaptureOptions`' `CaptureMode::Gpu` produces a `Pixel::D3D11` frame
  the same way `D3d11Upload` does, via the same shared helper) — enabling
  `dxgi-capture` pulls `d3d11-renderer` in automatically. Windows-only.
  `screen_capture`/`screen_capture_gpu` turn it on in their own
  `Cargo.toml` (alongside `d3d12-renderer`/`d3d11-renderer` respectively,
  to actually render what they capture).
- `wasapi-capture` (on `media-pp`) — pulls in `windows` (WASAPI/Core Audio
  sub-features) and enables `WasapiCaptureSource`/`WasapiCaptureOptions`/
  `WasapiDevice`/`WasapiDeviceKind`. Independent of `dxgi-capture`/
  `d3d11-renderer`/`d3d12-renderer` — capturing audio needs none of them —
  but commonly turned on alongside `dxgi-capture` for a combined
  desktop+audio recording (see `screen_audio_record`). Windows-only (WASAPI
  itself is a Windows API). `audio_capture`/`screen_audio_record` turn it
  on in their own `Cargo.toml`.
- `wasapi-renderer` (on `media-pp`) — pulls in the same Windows Core Audio
  bindings and enables `WasapiRenderer`/`WasapiRendererOptions`. It shares
  `WasapiDevice`/`WasapiDeviceKind` with `wasapi-capture`, but is otherwise
  independent. `audio_playback` enables it and converts into the selected
  endpoint's returned `AudioFormat` with `AudioResampler`.
- `ort` (on `media-pp`) — pulls in the `ort` crate (ONNX Runtime bindings;
  downloads a prebuilt onnxruntime binary at build time) and `ndarray`, and
  enables `OrtDetector`. `detect` turns it on in its own `Cargo.toml`.
- `webrtc` (on `media-pp`) — pulls in `str0m` (sans-I/O WebRTC, `wincrypto`
  backend — native Windows crypto, no OpenSSL vendoring) and enables
  `WebRtcPeer`/`WebRtcHandle`/`WebRtcTrackSink`/`WebRtcTrackSource`. The initial SDP
  offer/answer and ICE candidate setup happen via str0m directly, in the
  caller's own code, *before* constructing a `WebRtcPeer`; there's no
  signaling server built in. `webrtc_loopback` turns it on in its own
  `Cargo.toml`.

## Requirements

- ffmpeg installed and discoverable by `ffmpeg-sys-next` (see that crate's
  build requirements). `D3d12vaDecoder`/`D3d11Decoder` additionally need an
  ffmpeg build with `d3d12va`/`d3d11va` hwaccel support respectively (check
  `ffmpeg -hwaccels`) and a GPU/driver that supports it.
- `rtsp_serve` and `rtsp_serve_seek` require an external RTSP server that
  accepts publishing at the supplied URL. MediaMTX is one compatible option,
  but it is not bundled or managed by `media-pp`.
- Windows-backed examples (`audio_capture`, `audio_playback`, the
  `examples/render/*` window/capture examples, and the current
  WebRTC loopbacks) keep their runtime dependencies behind `cfg(windows)`.
  They build as unsupported stubs on other targets and print a clear message
  when run; their actual pipelines still run only on Windows.
- Windows backend modules and public re-exports are guarded by both their
  Cargo feature and `target_os = "windows"`; enabling one of those features
  for another target does not expose the Windows-specific element types.
- `D3d12vaDecoder` hand-mirrors a few structs from FFmpeg's
  `libavutil/hwcontext_d3d12va.h` that `ffmpeg-sys-next` doesn't bind
  (see the doc comment at the top of
  `elements/filter/decoder/windows/d3d12va_decoder.rs`) — sourced from FFmpeg
  n8.0's header. A future FFmpeg version changing that header's layout
  would silently break this with no compile-time warning.
  `D3d11Decoder`/`D3d11Upload`/`DxgiCaptureSource`'s GPU mode do the same
  for a couple of small D3D11VA-specific structs
  (`elements/filter/decoder/windows/d3d11va_decoder.rs`),
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
  `D3d11Renderer`, `DxgiCaptureSource`'s GPU mode) must share exactly one
  `ID3D11Device` — that's what lets this stack skip explicit GPU-side
  fences entirely, unlike the D3D12 side (see `D3d11Renderer`'s own doc
  comment for why).

## License

Licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE)
or the [MIT License](LICENSE-MIT), at your option.

`media-pp` does not bundle FFmpeg. Users are responsible for complying with
the license of the FFmpeg build and optional codecs they link against.
