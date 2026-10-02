---
name: media-pp-add-element
description: Add a new source, filter, terminal, driver, or element-like media component to media-pp, including its contracts, exports, feature gates, tests, and documentation. Use when implementing a new pipeline element or promoting an internal media stage into a public element; do not activate for small fixes or refactors of an existing element.
---

# Add a media-pp element

Read `README.md`, `AGENTS.md`, the crate documentation in `lib/src/lib.rs`,
the core traits in `lib/src/core/element.rs`, `produce.rs`, `transform.rs`
and `render.rs`, and the nearest existing element before designing the new
type. Choose the analogue by matching its graph role, buffer type, backend,
threading, control behavior, and resource ownership — not merely by a
similar name.

## Classify it

Pick the smallest kind that fits; the framework then owns the pads, the
control messages, the stream events and the preroll for it.

- **`Produce`** — a source that makes one thing at a time (a file, a device,
  a network stream, a compositor tick). It waits only through its `Wait`,
  names its outputs in `outputs` and hands on with `Produced::On` when it has
  several, begins its own segments with `Produced::Segment`, and says it can
  be sought or played backwards through `as_seekable` / `as_reversible`.
  Device setup on the source thread goes in `starting`/`stopping`, a device
  it stops for a pause in `pausing`/`resuming`. Expose it as a newtype over
  `ProducingSource` with `produce_source!`. There is no hand-written
  `SourceElement` loop outside tests.
- **`Transform`** — a filter that turns each buffer into none, one or
  several: its media work in `transform`, `drain` for what it holds at the
  end, `reset` for what a `Flush` lets go of, `stopping` for a `Stop`.
  Expose it as a newtype with `transform_filter!`.
- **`Render`** — a terminal that does something with each buffer (plays,
  shows, counts, hands out): `render`, `drain`, `reset`, `stopping`,
  `pausing`/`resuming`. Expose it with `render_sink!`.
- **`Sink` (and `Source`) written directly** — only for an element that
  routes the stream itself: a queue, a tee, a bin or rack, a muxer of
  several tracks, a compositor or mixer input, an `AppSink`. It reacts to
  control through its own state only (`Sink::flow`, crate-private) and never
  forwards a message; the graph passes control and stream events on through
  its pads.
- A padless driver or a plain control-plane object when there is no stream
  through it. Do not force an `Element`, builder, handle, or module split onto
  a type whose behavior does not require it.

Define accepted and produced `MediaBuffer` variants, formats, dimensions,
metadata, number and meaning of pads, end-of-stream behavior, what it holds
across a `Flush`, error boundary, thread ownership, and teardown before
choosing the public API.

## Declare what wiring can check

- Link contract: `input_contract` and `output_contract` (a `Transform`'s own,
  or `Sink::input_contract` and `SrcPad::with_contract`). State only what
  construction settles — the `MediaKind`s, and for decoded frames the
  `MemoryDomainSet` and, where fixed, every `PixelLayout` the runtime check
  lets through. `OutputContract::SameLayout` for an output that follows its
  input; `PixelLayoutSet::ALL` where only a frame can tell. A contract
  narrower than the element really accepts refuses a working pipeline.
  `Any` with `OutputContract::Passthrough` only for an element that forwards
  every kind unchanged; a genuinely runtime-dependent one stays `Unknown`.
- Seeking: a sink recording the stream as it ran (a file, a replay window)
  returns false from `Sink::accepts_seek`. A source that can reposition is a
  `SeekableSource`; one that can play backwards a `ReversibleSource`, and a
  decoder that can a `ReversibleDecoder`. A live source says so in `is_live`.
- An element that waits on the playback clock inside `consume` answers
  `Sink::own_queue`, so a chain puts it behind a queue; a `Rack` refuses it.
  Its constructor takes no queue depth.
- Keep backend-specific public names prefixed; an unprefixed type has a
  deliberately backend-independent contract. Derive values that must agree,
  and keep constructors private when an owning object must supply a device.

## Implement the data path

- Store the name as `Arc<str>` and a `PpLog` from `element_pp_log`; pipeline
  wiring stamps the identity through `pp_log_mut`.
- Match the buffer variant before reading it; validate format, dimensions,
  bounds, device ownership and backend handles before FFI or GPU calls.
- Preserve PTS, duration, time base and colour metadata unless the element
  deliberately starts a new timeline, and then set the new time base.
- Read the phase — paused, prerolling, which timeline — from the
  `PlaybackState` given in `attach_context`, never from the messages. Hold
  back during a phase only what the framework does not already hold
  (`OutputStash`), and hand it on as soon as the phase is over.
- One buffer's or one input's failure is posted to the bus and the element
  goes on; fan-in, fan-out and batches isolate it from their siblings.
  Return `Err` from a source only when it cannot continue.
- A repeated picture on a live graph is answered with what was already made
  (`repeat::PerFrameTransform`); anything comparing pictures by address
  holds the pooled `Arc`, not a frame reference.
- Make fallible registrations and replacements atomic. Dropping a running
  element, worker, pool, device helper or child process releases or joins
  what it owns, and retained handles keep nothing unrelated alive.

## Integrate the public surface

- Put the implementation under `elements/source`, `filter`, `sink` or
  `driver` unless it is genuinely a platform backend type; `core/` never
  reaches into `elements/`.
- Keep module declarations, imports, flat `elements` re-exports, public error
  types, crate-level conversions, Cargo dependencies and feature gates
  compatible; check the enabled and disabled builds.
- Add an `ElementType` variant only for a built-in graph element.
- Document accepted buffers, metadata, ownership, threading, errors,
  end-of-stream and seek behavior, handle lifetime and backend requirements.
  Public docs may not link private items (CI runs rustdoc with `-D warnings`
  for every feature set).
- Update the inventory in `docs/elements.md` or the feature table in
  `docs/features.md` when the public surface changes. Extend an existing example rather than adding a parallel one.

## Verify the observable contract

- Focused tests for valid processing and every relevant failure mode,
  including the state after an error. Cover incompatible buffers, metadata
  preservation, draining at the end, what a `Flush` drops, recoverable
  downstream failure and teardown.
- For a link contract, test both the refused mismatch and a valid chain that
  must still link.
- Hardware tests detect absence and skip with a reason. Media comes from
  `test_support::try_test_video`; assert what holds for any fixture.
- An element that waits, holds buffers or produces on a loop of its own
  joins the control conformance matrix as a choice on one of its axes, and
  runs there pinned to two cores, loaded, before it lands (`CONTRIBUTING.md`).
- A per-cycle worker, codec context, pool, GPU object, file or helper process
  gets a lifecycle scenario through `$media-pp-soak-analysis`.
- Run the narrow tests, then the affected feature set and the default build;
  run a changed example end to end. Finish with `cargo fmt --all -- --check`,
  `git diff --check` and `git status --short`.
