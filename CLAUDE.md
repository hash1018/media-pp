# CLAUDE.md

Repository guidance for AI-assisted and human development. Read `README.md`
first for the architecture, element inventory, feature flags, examples, and
build requirements. Treat the code and tests as the final source of truth when
documentation and implementation differ.

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

- A direct `Sink::consume` call is synchronous and may return `Err`. A `Queue`
  is the explicit thread and recovery boundary: it reports a downstream data
  error as `BusEvent::Error`, drops that buffer, and continues its worker.
- A `SourceElement::run` implementation should likewise report a recoverable
  per-buffer pad-push error to its `Bus` and continue. Return `Err` only when the
  source cannot meaningfully continue.
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
- Preserve media metadata across transforms unless the element intentionally
  creates a new timeline: PTS, duration, packet `time_base`, and video
  color-space/range are part of the buffer contract, not optional decoration.
- Forward `Eos`. Stateful codecs, resamplers, and muxers must drain/flush delayed
  data before forwarding or finalizing it. `Stop` means abandon, not natural EOS.
- Video frames from `UnboundObjectPool` travel as
  `Arc<UnboundObjectPoolRef<_>>`. Never mutate a frame after publishing it, and
  never return/reuse its backing resource while downstream `Arc` clones exist.

## Control, lifetime, and concurrency

- Every `SourceElement` loop must remain responsive to Pause, Resume, Stop, and
  Seek using the established `drain_control` or select-on-control pattern.
  Wall-clock-driven sources must add `ControlOutcome::paused_for` back into their
  scheduling state so Resume does not emit a catch-up burst.
- `Sink::control` must consciously handle and, when it has downstream stages,
  propagate every control message. A Queue control failure is reported without
  leaving the control cascade permanently blocked.
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
- No media is checked into this repository. A test needing a real video calls
  `test_support::try_test_video`, which reads `MEDIA_PP_TEST_VIDEO` and skips
  with a reason when it is unset or unreadable; run the affected tests with that
  variable actually set, since a skipped test reports as passing. Such a test
  must assert a contract that holds for any fixture — never a particular file's
  codec, resolution, duration, or keyframe spacing.
- Examples take their media path as a required argument, print a `usage:` line
  to stderr when it is missing, and exit non-zero. Do not reintroduce a default
  path.
- For a new or changed example, run that actual example end to end. For recorded
  video/audio, inspect the result with `ffprobe`; for visual behavior, extract
  representative frames and verify the expected pixels/content instead of only
  checking exit status.
- Run `git diff --check` and inspect `git status` before handoff. Do not commit
  build artifacts or verification media.

## Documentation and repository hygiene

- Update `README.md` when public API, feature flags, requirements, or examples
  change. Keep volatile roadmap ideas out of agent instruction files.
- Doc comments should explain invariants, ownership, thread/error behavior, and
  non-obvious rationale. Do not repeat claims that can be read directly from a
  struct definition.
- Preserve unrelated user changes. Do not rewrite `Cargo.lock` unless dependency
  resolution actually changed, and never edit generated `target/` contents.

## Known historical hazards

- Hardware video encoding was implemented, tested on real hardware, and then
  intentionally reverted. `SwEncoder` is the only encoder family currently in
  the tree. Read history and ask before reintroducing a hardware encoder.
- D3D11VA decode surfaces are fixed-size. `extra_hw_frames` must cover the deepest
  downstream buffering, unlike the growable pools used elsewhere.
- Before extending text overlays (for example multi-line layout, background
  boxes, or shared font caching), read the design and ownership rationale in
  `compositor/windows/d3d11_video_compositor/text_handle.rs` and
  `compositor/text_layer.rs`. The
  current split between backend-independent settings and the D3D11-owned
  rasterize/upload control object, including its deliberately constrained
  construction path, is intentional.
