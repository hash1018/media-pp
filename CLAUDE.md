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
  using the existing `HLog` and stable graph identity conventions.
- Plain control-plane objects that are not `Element`s have no bus identity;
  their operations should return a typed error directly to the caller.

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
  cargo test -p media-pp --features d3d11-renderer
  ```

  Select other features according to the files changed. Also check the default
  build when changing feature-gated exports.
- Hardware-dependent tests use a `try_device()`-style helper and skip with a
  clear reason when the required device is unavailable. Prefer checked-in test
  data; if a system font/device is unavoidable, detect absence rather than
  panicking.
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
  `d3d11_video_compositor/text_handle.rs` and `compositor/text_layer.rs`. The
  current split between backend-independent settings and the D3D11-owned
  rasterize/upload control object, including its deliberately constrained
  construction path, is intentional.
