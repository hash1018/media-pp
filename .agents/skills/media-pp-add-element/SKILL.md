---
name: media-pp-add-element
description: Add a new SourceElement, filter Sink, terminal Sink, driver, or element-like media component to media-pp, including its contracts, exports, feature gates, tests, and documentation. Use when implementing a new pipeline element or promoting an internal media stage into a public element; do not activate for small fixes or refactors of an existing element.
---

# Add a media-pp element

Read `README.md`, `AGENTS.md`, the core traits in
`lib/src/core/element.rs`, and the nearest existing element before designing the
new type. Choose the analogue by matching its graph role, buffer type, backend,
threading, control behavior, and resource ownership—not merely by a similar
name.

## Establish the contract

- Classify the component as a pure `SourceElement`, a `Sink` with output (a
  filter), a terminal `Sink`, a padless driver, or a plain control-plane object.
  Do not force an `Element`, builder, handle, or module split onto a type whose
  behavior does not require it.
- Define accepted and produced `MediaBuffer` variants, formats, dimensions,
  metadata, number and meaning of pads, EOS behavior, control behavior, error
  boundary, thread ownership, and teardown before choosing the public API.
- Keep backend-specific public names prefixed. An unprefixed type must have a
  deliberately backend-independent contract.
- Expose only construction and runtime controls required by the current use
  case. Derive values that must agree, and keep constructors private when an
  owning object must supply a device or other invariant.

## Implement the data and control paths

- Store the element name as `Arc<str>` and a `PpLog` built with
  `element_pp_log`; let pipeline wiring stamp the pipeline identity through
  `pp_log_mut`.
- Match the input buffer variant before reading it. Validate format, dimensions,
  bounds, device ownership, and backend handles before FFI or GPU calls.
- Preserve PTS, duration, packet time base, and video color metadata unless the
  element intentionally establishes a new timeline.
- A direct `Sink::consume` remains synchronous. Introduce a worker only when the
  contract requires one; `Queue` is the normal explicit downstream thread and
  recovery boundary.
- Implement every `Sink::control` case consciously and propagate control when
  the element has downstream pads. Flush delayed state before forwarding EOS;
  treat `Stop` as abandonment rather than natural EOS.
- In a `SourceElement` loop, remain responsive through the established control
  helpers. Report a recoverable per-buffer pad-push error to the source bus and
  continue; return `Err` only when the source cannot meaningfully continue.
- Isolate fan-in, fan-out, and batch failures so one bad item or branch cannot
  suppress valid siblings. Attribute errors to the most specific stable element
  or branch identity available.
- Make fallible registrations and resource replacement atomic. Dropping a
  running element, worker, pool, device helper, or child process must release or
  join what it owns without retained handles keeping unrelated state alive.

## Integrate the public surface

- Put the implementation under the matching `elements/source`, `filter`,
  `sink`, or `driver` module unless it is genuinely a platform backend type.
- Keep module declarations, imports, flat `elements` re-exports, public error
  types, crate-level conversions, Cargo dependencies, and target/feature gates
  compatible. Check both enabled and disabled configurations for gated code.
- Add an `ElementType` variant only for a built-in graph element that needs that
  stable type identity. Plain helpers and downstream custom elements do not
  require one.
- Document accepted buffers, metadata behavior, ownership, threading, error and
  recovery behavior, control/EOS semantics, handle lifetime, and non-obvious
  backend requirements.
- Update the README inventory or feature table when the public surface changes.
  Prefer extending an existing example over adding a parallel crate, and add an
  example only when it demonstrates behavior not already covered.

## Verify the observable contract

- Add focused tests for valid processing and every relevant failure mode. Test
  state after an error, not only the returned error variant.
- Cover incompatible buffers and invalid external dimensions or formats,
  metadata preservation, EOS draining and forwarding, control propagation,
  recoverable downstream failure, and teardown according to the element's
  contract.
- Make hardware tests detect unavailable devices and skip with a reason. Tests
  needing media use `test_support::try_test_video` and must not assume a
  particular fixture's codec, dimensions, duration, or keyframe layout.
- If the element introduces a per-cycle worker, codec context, pool, GPU object,
  file, or helper process, use `$media-pp-soak-analysis` to add or evaluate the
  corresponding lifecycle scenario.
- Run the narrow tests first, then the affected Cargo feature set and the default
  build where feature-gated exports changed. Run a changed example end to end
  and inspect recorded or visual output as required by `AGENTS.md`.
- Finish with `cargo fmt --all -- --check`, `git diff --check`, and
  `git status --short`, preserving unrelated worktree changes.
