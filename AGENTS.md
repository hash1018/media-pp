# AGENTS.md

Repository guidance for AI-assisted and human development. Read `README.md`
first and the files it lists under `docs/` — the element inventory
(`docs/elements.md`), feature flags and what each needs
(`docs/features.md`), examples (`docs/examples.md`), and platform setup
(`docs/building/`) — the crate documentation in `lib/src/lib.rs` for how a pipeline
runs, and `CONTRIBUTING.md` for how to test. Treat the code and tests as the
final source of truth when documentation and implementation differ.

## Communication

- Reply in Korean by default in this repository, even when the prompt is in
  English, unless the user asks for another language.
- Lead with the result. Explain implementation details only as far as they help
  the user review or operate the change.

## Scope and design

- Prefer the smallest design that satisfies a current requirement. Do not add
  parameters, variants, compatibility aliases, or abstractions solely for a
  hypothetical future caller.
- Preserve existing public behavior unless the requested change intentionally
  replaces it. Do not add a deprecated compatibility shim automatically; first
  decide whether compatibility is actually a requirement for this project.
- Derive values that must agree instead of asking callers to supply both. For
  example, a D3D11 text layer must use its compositor's device, so it is created
  through `D3d11VideoCompositorHandle::add_text_layer` rather than a public
  constructor accepting an arbitrary device.
- Make fallible mutations atomic from the caller's perspective. Validate and
  allocate before replacing a same-name registration; an error must not leave a
  placeholder behind or invalidate the previous working registration.
- Follow the nearest existing API when its semantics match, but do not force a
  builder, handle, or module split merely because another type has one.

## Pipeline and error boundaries

- A direct `RawSink::consume` call is synchronous and may return `Err`. A `Queue`
  is the explicit thread and recovery boundary: it reports a downstream data
  error as `BusEvent::Error`, drops that buffer, and continues its worker.
  The one exception to "explicit": an element that waits on the clock
  inside `consume` — `Pacer`, `VideoSynchronizer` — answers
  `RawSink::own_queue`, and a chain runs it behind the `.queue()` straight in
  front of it or, where there is none, one it adds; a `Rack` refuses it.
  Its constructor takes no depth, so callers need not know.
- What a seek leaves behind is dropped between its `Flush` and its
  segment (`crate::stream`): the pipeline's own `Flush` puts every pad it
  passes and every `Queue` worker into flushing, and whatever reaches one
  before the flushed `Segment` the source begins after the seek is dropped
  there. No buffer carries a number saying which seek it is from. An
  element that hands buffers to a worker of its own drops what it holds on
  `Flush` and hands on nothing between the `Flush` and the segment.
- A source likewise goes on past one buffer's failure: the framework posts a
  failed pad push to the `Bus`, and a `Source` posts what fails for one part
  of what it makes — a compositor layer it cannot draw — itself. `produce`
  returns `Err` only when the source cannot meaningfully continue.
- Fan-in/fan-out and batched processing must isolate failures. A bad mixer input,
  Tee branch, or compositor layer must not prevent valid siblings from being
  processed. Avoid `try_for_each`, an unreviewed `?`, or an early return that
  turns one item's error into termination of the whole loop.
- Attribute bus errors to the most specific failing element/branch available,
  using the existing `PpLog` and stable graph identity conventions.
- Plain control-plane objects that are not `Element`s have no bus identity;
  their operations should return a typed error directly to the caller.

## Logging

- Library diagnostics must use `PpLog` and the `pp_info!`, `pp_debug!`,
  `pp_warn!`, `pp_error!`, and `pp_trace!` macros. Do not emit through or
  install a process-global `log` logger or `tracing` subscriber. The private
  file logger stays explicit and opt-in through `media_pp::log::init`, and the
  caller owns its `LogGuard` for the full period in which logs must be kept and
  flushed.
- Every attached element record must keep `pipeline_id`, element type, and
  caller-selected instance name as separate identity fields. Construct or
  update its `PpLog` through the existing pipeline helpers instead of packing
  identity into a free-form message. Use the stable graph element ID where a
  topology must disambiguate duplicate names. The originating thread is a
  record field the logger writes itself; never fold a thread id into a message.
- Keep levels intentional: `Error` for failed operations, `Warn` for degraded
  or recoverable conditions, `Info` for sparse lifecycle and topology changes,
  `Debug` for diagnostic state, and `Trace` for detailed EOS/control flow. Do
  not log ordinary video, audio, or packet buffers one record per buffer.
- A successful pipeline start logs one `run` record whose body is the complete
  multiline topology diagram; a successful dynamic `Tee` change logs the same
  kind of record under the `Tee`'s own identity, with `attach` or `detach` in
  place of `run`. Keep the diagram inside the event's own record — only lines
  within one record are guaranteed to stay adjacent, so a separate diagram
  record would merely tend to follow the event that caused it. The diagram
  shows stable `#id` values and source-pad labels; align each downstream
  connector under its upstream element so fan-out is visible at the actual
  branching point. Do not replace it with repeated root-to-leaf paths or add
  `reason`, revision, branch, element, or edge-count summaries without a new
  requirement.
- Trace EOS and control at every element/thread boundary with an explicit
  `event`, `phase`, and success/error `outcome` where applicable. Include the
  pad when the event is sent through a specific pad, so a log can show exactly
  where propagation stopped.
- Keep logging off hot paths when its level is disabled: check `enabled` before
  taking graph snapshots or doing non-trivial formatting. Queue a multiline
  diagram as one complete non-blocking-writer record, and never hold graph,
  branch, input, or pad locks while formatting or emitting a record.
- Every executable example initializes the private logger at `Trace`, writes to
  `./logs` with its Cargo package name as the prefix, and retains the returned
  guard until shutdown. For logging-format or propagation changes, update
  `lib/tests/flow_log.rs` and run an affected example end to end in addition to
  the normal library tests.

## Buffers, timestamps, and EOS

- Match the `MediaBuffer` variant before reading it and return a typed error for
  incompatible input. Before FFI or GPU calls, validate format, dimensions,
  plane/stride bounds, texture array index, and device ownership as applicable.
- Each variant holds a wrapper — `PacketBuffer`, `VideoBuffer`,
  `AudioBuffer` — that dereferences to the payload's `Arc` and carries the
  buffer's `Metadata`. Make one from its `Arc` with `.into()` (or
  `MediaBuffer::video/packet/audio`); take the `Arc` out with `payload()` or
  `into_payload()`. Forward the buffer you were handed, not a rewrap of its
  payload, wherever the picture is unchanged, so what it carries goes on.
- A `Filter`'s outputs at the input's timestamp, of the input's sort, are
  given what the input carried; nothing is carried onto another timestamp or
  sort. An element that stamps its outputs anew, or a direct `RawSink` that
  makes a buffer from one it was handed, carries `Metadata` itself, as
  `FrameRateLimiter` does — never onto a buffer made from another input.
- A filter that makes each buffer into none, one or several is a
  `Filter`: its media work, with `drain` for what it still holds at the
  end, `reset` for what a `Flush` lets go of and `stopping` for what a
  `Stop` lets go of beside it (by default the same). The framework's
  stage owns its pad, hands `Eos` on after the drain, keeps what the pad
  cannot take while a preroll holds the graph (`OutputStash`, which the
  decoders' gate, a bin and a rack keep too), and never shows it a control
  message. Write `RawSink` and `SrcPads` directly only for an element that
  routes the stream itself — `Queue`, `Tee`, a bin or a rack, a muxer, a
  compositor or mixer input — or reads the stream plane. An element of
  this crate moved onto one keeps its public name, constructors and
  `RawFilter` as a newtype over `FilterStage` (`filter_stage!`).
- A terminal that does something with each buffer — plays, shows, counts,
  hands it out — is a `Sink`: `render` for each buffer, `drain` before
  its end is taken, `reset` for a `Flush` and `stopping` for a `Stop` (by
  default the same), and `pausing` and
  `resuming` for a device it stops for a pause. The framework turns the
  control messages into those and never shows it one; what the terminal
  wrapper does — the preroll's counting, holding while paused — is
  unchanged by it. A muxer of several tracks, a compositor's or a mixer's
  input and an `AppSink` handing control to the application stay `RawSink`s.
  An element of this crate moved onto one keeps its public name as a
  newtype over `SinkStage` (`sink_stage!`).
- Declare a new element's link contract through `RawSink::input_contract` and
  `SrcPad::with_contract` — a `Filter`'s own `input_contract` and
  `output_contract` — limited to what construction already settles:
  the `MediaKind`s a port deals in, and for a decoded one the `MemoryDomain`s its
  frames may live in and the `PixelLayout`s (NV12, P010, BGRA, other) they may
  be in. `PortContract::Packets` has no domain or layout at all, since encoded
  media is always host memory; `PortContract::Frames` always states both, and
  an element that takes any backend says `MemoryDomainSet::ALL` rather than
  leaving a blank. State a layout set only where construction settles it — a
  scaler built for NV12 output, a renderer that presents NV12 — as every layout
  the runtime check lets through, never fewer; a layout that depends on the
  stream is every one it may be, one that follows the input is
  `OutputContract::SameLayout`, and in doubt it is `PixelLayoutSet::ALL`,
  which `PortContract::frame` starts from. This never replaces the runtime validation above
  — it only refuses wiring no buffer could have made work, before the pipeline
  starts. Both sides default to `Unknown`,
  which always links, so an element with a genuinely runtime-dependent contract
  simply leaves it alone rather than guessing.
- Whether a seek can be followed is declared the same way, at wiring: a
  sink whose output is a record of the stream as it ran — a file being
  written, a replay window — returns false from `RawSink::accepts_seek`, and a
  source that can reposition its input is a `SeekableSource`, saying so
  from `as_seekable`; one that cannot writes no `seek` at all. A live one
  also says so through `is_live`. `Pipeline::check_seek` answers from what
  the graph holds; nothing asks the running elements.
  Playing backwards likewise: a source that can is a `ReversibleSource`,
  and an element that turns a picture's packets into pictures a
  `ReversibleDecoder`, each saying so from `as_reversible`; the pipeline tells
  a sink where each stretch begins and ends, so no decoder works that out
  for itself. Everything else is handed the stream as it runs, and
  declares nothing.
- Preserve media metadata across transforms unless the element intentionally
  creates a new timeline: PTS, duration, packet `time_base`, and video
  color-space/range are part of the buffer contract, not optional decoration.
- The end of a stream is `StreamEvent::Eos`, not a buffer. Stateful codecs,
  resamplers, and muxers drain/flush delayed data on it, before the graph
  passes it on or the file is finalized. `Stop` means abandon, not natural EOS.
- Video frames from `UnboundObjectPool` travel as
  `Arc<UnboundObjectPoolRef<_>>`. Never mutate a frame after publishing it, and
  never return/reuse its backing resource while downstream `Arc` clones exist.
- A live graph emits at a rate, not on change: a capture of a still screen and a
  compositor with nothing to recompose both re-emit the picture they already
  have. An element that makes a new picture out of one frame — upload, readback,
  scale, convert, key — answers such a repeat with what it already produced,
  through `repeat::PerFrameTransform`. An element whose output depends on more
  than that frame (the compositors' layers, a capture's cursor) recognises its
  own repeats against its own state; one whose graph answers a frame with none
  or several (`CudaScaler`) has no single output to offer again; and one whose
  contract is a *rate* — an encoder, a muxer — is where repeats must keep
  flowing, since a picture identical to the last one is what an encoder is
  cheapest at.
- Whatever compares pictures by address holds the `Arc<UnboundObjectPoolRef<_>>`
  it was handed, never just an `av_frame_ref` of what is inside it. A frame
  reference keeps the buffer alive but leaves the producer's pool free to hand
  that slot out again and put new pixels at the same address. The same applies
  in reverse to a picture offered again: it stays checked out of its own pool
  until `buffer::picture_is_referenced` reads false for it, or the next output
  is written over pixels still queued downstream.

## Control, lifetime, and concurrency

- A source that makes one thing at a time is a `Source`: the framework
  runs its loop — takes each request the one way every source does, keeps
  a pause out of the clock `Wait::now` reads, and ends the stream after
  `Produced::End` — and it waits only through its `Wait`, which lets go the
  moment the pipeline has something for the thread. With several outputs
  (`Source::outputs`, `Produced::On`) the framework holds back what an
  output cannot take yet, within bounds, so the one read cursor keeps the
  others fed (`crate::parking`), and ends each output as soon as it owes
  nothing; a source that can be sought (`Source::as_seekable`) has its
  seeks taken between one thing made and the next and stays at its end
  until stopped or sought; a segment of its own — a lap, another input —
  is `Produced::Segment`. What
  a device sets up on the source's own thread — an apartment joined, a
  capture started — goes in `Source::starting` and is let go of in
  `stopping`, which the framework calls there however the loop ended;
  stopping and restarting its input for a pause goes in `pausing` and
  `resuming`. An element of this crate moved onto one keeps its public
  name as a newtype over `SourceStage` (`source_stage!`).
- Every source of this crate is a `Source`; a `RawSource` loop written
  by hand is left only in tests. Such a loop must remain responsive to
  Pause, Resume, Stop, and Seek: it drains its channel with `drain_control`
  (test-only), or hands each request it takes to `control::handle_request` —
  never applies one itself: a source that passed a `Pause` on without
  pausing deadlocked a seek (208af56).
- Control stays inside the crate: `ControlMsg` reaches this crate's
  elements through `RawSink::flow`, a hidden hook whose argument nothing
  outside can name (the crate calls it as `RawSinkExt::control`), and a sink
  outside the crate hears a pause, a resume and a stop through
  `RawSink::pausing`, `resuming` and `stopping`, which `flow`'s default
  calls, and a seek through the flushed segment. `flow` is an element's
  own reaction and nothing more. The graph
  passes each message on through a filter's `src_pads()` after it
  (`control::deliver` for a filter driven by hand), to every pad even where
  one fails; an element never forwards control itself, and `SrcPad::control`
  is crate-private so that it cannot hand everything after it a message
  twice. Only an element that routes control its own way — `Tee` to its
  branches, a bin to the line inside it — sends it on from its hook. A
  pipeline's requests reach every source and every `Queue` worker directly
  (`control::Direct`), so one passed on stops at the next queue; a queue
  carries across only what is sent without the pipeline — a `Stop` after a
  source failed, a `Flush` a bridge injects. A Queue control failure is
  reported without leaving the control cascade permanently blocked.
- What describes the stream travels in it (`crate::stream`): the
  `Segment` each stream begins with and each seek begins again — and a
  looping file at each lap, and a bridge where its feeding side flushed or
  another input begins, those on the timeline they come in — and the
  `Eos` it ends with, in order
  with the buffers. An element reacts through `RawSink::stream_event`, pushing
  from there whatever it answers the event with; the graph passes the event
  on through its pads, as it does control. Only an element that routes the
  stream its own way — `Queue`, `Tee`, a bin's or a rack's line — sends it
  on itself, in order with its buffers, never dropping one for room nor
  waiting for any; and whatever joins a stream under way — a branch
  attached, a line filled anew — is handed its last segment first. A time
  compared against the buffers — where an accurate seek shows from — is
  on their timeline already, put there by the source that begins the
  segment (`Segment::show_from`), never compared as the place in the media
  a caller named: a looping file keeps the two a lap or more apart.
- Where playback stands — paused, prerolling and for which seek, which
  timeline is current — is the pipeline's `PlaybackState`, given to every
  element in `attach_context` and written only by the pipeline, before the
  message that announces the change. An element that behaves differently
  while paused or while a preroll runs reads it there; it does not work the
  phase out from the `Pause`, `Resume` and `Preroll` it is handed, which
  remain for what an element does about a change — stopping a device,
  dropping a timeline's leftovers. Whatever an element holds back during a
  phase it hands on, in order, as soon as it sees the phase is over: from
  the next buffer as much as from the message behind it. The pipeline
  writes what restricts flow before its message; what lets flow go travels
  only as a message, which a paused source or queue waits for and passes
  on before any data — so a preroll always ends in a pause, even to play on.
- Dropping a running `Pipeline`, driver, Queue, or owned helper process must stop
  and join/collect the worker it owns. Retained handles must not accidentally
  keep an unrelated pipeline bus, graph, sink, or worker alive.
- Do not assume every `*Handle` is `Weak`-backed. Handles are thread-safe runtime
  control endpoints, but their ownership differs: some use `Weak`, some own an
  `Arc` control block, and some own channel endpoints. Document whether cloning
  is cheap, what it keeps alive, what happens after the target stops, and whether
  a call can block or perform expensive work. `D3d11TextLayerHandle`, for
  example, owns a device/font/pool and rasterizes/uploads on `set_text`.
- For dynamic same-name registrations, assign a stable registration ID. A stale
  Sink or handle from the replaced registration must be unable to update,
  remove, stop, or send EOS to its replacement.
- Snapshot shared registries under their lock, then release the lock before
  blocking downstream calls, GPU work, or user callbacks. Do not hold a global
  branch/input lock across code outside that registry.

## API and module organization

- Three layers, each built only on the ones below it: `core/` is the pipeline
  framework, `elements/` the parts that go into a pipeline (a bin such as
  `VideoDecodeBin` is one of them), and `app/` whole pipelines behind one type
  for a common job — `Player`. An `app/` type is built on the public elements
  and pipeline API; where it needs a crate-private item, that item documents
  why, so the layer could become a crate of its own without redesign. `core/`
  code never reaches up into `elements/` or `app/`; its docs and tests may
  name them.
- Put genuinely shared, backend-independent value types and math in a shared
  module; keep backend implementation and dependencies in the backend module.
  A type used by only one backend does not need to be made shared speculatively.
- Backend-specific public symbols carry the backend prefix (`D3d11*`, `D3d12*`).
  An unprefixed public type implies a deliberately backend-independent contract.
- Use a builder only when construction is genuinely multi-stage or collects
  configuration/branches. Use a handle for runtime control. Constructors that
  enforce cross-object invariants should stay private or `pub(crate)` and be
  exposed through the object that can supply the invariant correctly.
- When adding a feature-gated public type, keep its module declaration, imports,
  re-exports, error variants, and example dependency under compatible `cfg`/
  Cargo feature gates. Check both the feature-enabled and feature-disabled
  library build.
- Use `thiserror` enums for actionable component errors and wire them into the
  crate-level error only when callers need that conversion. Avoid panic/unwrap
  for invalid external media, missing codecs/devices, or other expected runtime
  failures.

## D3D11 and FFmpeg invariants

- Every interacting D3D11 element in a pipeline must use the same
  `ID3D11Device` and shared immediate context. Validate foreign textures before
  drawing/copying rather than relying on a later Windows API failure.
- Validate the FFmpeg frame's visible dimensions against the backing texture and
  preserve the selected texture-array slice. Do not assume padding rows or slice
  zero.
- Release/clear D3D11 bindings on every path after drawing so cached resources do
  not leak into the next frame's state.
- Do not reconstruct an FFmpeg D3D11 frames context from a hand-mirrored
  `AVD3D11VAFramesContext`. That approach previously caused memory corruption.
  The current upload/compositor path creates textures with `windows-rs`; the
  decoder only touches the small, already-initialized D3D11VA fields documented
  in its source. Read those comments and history before changing the FFI layout.

## Testing and verification

- Add a regression test for every fixed failure mode. Test the observable
  contract, including the state after an error, not just the returned variant.
- Start with targeted tests, then run the affected feature set. Typical checks:

  ```text
  cargo fmt --all -- --check
  cargo test -p media-pp
  cargo test -p media-pp --features d3d11
  ```

  Select other features according to the files changed. Also check the default
  build when changing feature-gated exports.
- Hardware-dependent tests use a `try_device()`-style helper and skip with a
  clear reason when the required device is unavailable. Detect absence rather
  than panicking whenever a system font, device, or media file is unavoidable.
- No media is checked into this repository, and the library's tests do not ask
  for any: `test_support::try_test_video` synthesizes a fixture through
  `test_support::synthesize`, built from this crate's own synthetic sources and
  the two encoders that are always present (`VideoCodec::OpenH264`,
  `AudioCodec::Aac`). Nothing there shells out to an `ffmpeg` binary; CI
  installs FFmpeg's development libraries and no command-line tool at all. So
  these tests run everywhere, and everywhere against the same file.
- `MEDIA_PP_TEST_VIDEO` is now read by `lib/tests/soak.rs` alone, where a real
  recording is the point — what a real workload costs is not what a 320x240
  clip costs. Do not reintroduce it in the library's own tests.
- Know what the fixture cannot show. It has no B-frames, no edit list, constant
  frame duration and a start time of exactly zero, so demuxing, seeking and
  decoding are exercised against this crate's own encoder output and no other.
  A test must assert a contract that holds for *any* fixture — never a
  particular file's codec, resolution, duration, or keyframe spacing — and
  never a fixed wait that assumes the source is still running, which is what a
  full-length recording used to hide.
- End-to-end audio and timing tests must pace what they feed. An unpaced branch
  delivers a file as fast as it decodes, which keeps every downstream buffer
  full and hides exactly the defects such a test is for — an input handing over
  less audio than it was given then only fills a queue more slowly. See
  `audio_mixer.rs`'s `against_a_file` tests, where removing the `Pacer` makes a
  test that reproduces an 8% shortfall pass against the bug.
- A change to how control travels — a new `ControlMsg`, the framework's
  source loop or a `Source`'s waits, an element that waits on the clock, a
  `Queue` or `Tee` path — runs the control conformance sequences pinned to
  two cores, a few hundred of them, alone and loaded, before it lands (see
  `CONTRIBUTING.md`). A new element that waits, holds buffers, or produces
  several outputs joins the matrix there as a choice on one of its axes, and
  the matrix pairs it with every other choice. These races do not show on a
  free machine. A bug the matrix finds that is not fixed at
  once goes into its `KNOWN_BROKEN` list with an ignored test that
  reproduces it — never into a looser promise.
- For stress tests, leak investigations, fitted-slope interpretation, or new
  per-cycle resource coverage, apply the repository skill
  `media-pp-soak-analysis`. Its hardware, fixture, and portal prerequisites must
  be measured rather than treating a skipped scenario as verified coverage.
- For a new example or a material example-pipeline change, apply the repository
  skill `media-pp-add-example`, including its cross-platform structure,
  documentation, and end-to-end output verification requirements.
- Run `git diff --check` and inspect `git status` before handoff. Do not commit
  build artifacts or verification media.

## Documentation and repository hygiene

- Update the documentation when public API, feature flags, requirements, or
  examples change: the element inventory in `docs/elements.md`, feature flags
  and their requirements in `docs/features.md`, examples in
  `docs/examples.md`, platform setup in `docs/building/`. `README.md` stays a
  short overview with a table of those files; detail goes into `docs/`. Keep volatile roadmap ideas out of agent instruction files.
- Doc comments should explain invariants, ownership, thread/error behavior, and
  non-obvious rationale. Do not repeat claims that can be read directly from a
  struct definition.
- Preserve unrelated user changes. Do not rewrite `Cargo.lock` unless dependency
  resolution actually changed, and never edit generated `target/` contents.

## Known historical hazards

- An early hardware encoder was implemented, tested on real hardware, and then
  intentionally reverted; the families now in the tree (`D3d11NvencEncoder`,
  `CudaEncoder`) were reintroduced deliberately afterwards. Read history before
  adding a third, but the question of whether hardware encoding belongs here at
  all is settled.
- D3D11VA decode surfaces are fixed-size. `extra_hw_frames` must cover the deepest
  downstream buffering, unlike the growable pools used elsewhere.
- NVDEC's pool is fixed-size *and* capped: at most 32 surfaces including the
  codec's own reference frames. Exceeding it fails `cuvidCreateDecoder`
  outright rather than degrading, so a downstream `Queue` depth has to fit that
  budget — see `CudaDecoder::new`.
- Before extending text overlays (for example multi-line layout, background
  boxes, or shared font caching), read the design and ownership rationale in
  `compositor/windows/d3d11_video_compositor/text_handle.rs` and
  `compositor/text_layer.rs`. The
  current split between backend-independent settings and the D3D11-owned
  rasterize/upload control object, including its deliberately constrained
  construction path, is intentional.
