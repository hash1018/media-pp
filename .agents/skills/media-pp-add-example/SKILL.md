---
name: media-pp-add-example
description: Add or substantially change a media-pp example, including platform branches, Cargo features, CLI behavior, README pipeline documentation, logging setup, and end-to-end output verification. Use for new example crates or material changes to an example pipeline; do not activate for library-only changes or trivial example formatting.
---

# Add or change a media-pp example

Read `README.md`, `AGENTS.md`, the target example's source and
README, and the nearest example with the same purpose before editing. Derive the
pipeline from its wiring closure rather than from imports or filenames.

## Choose the example shape

- Search all examples for the same user-facing purpose. Extend an existing
  crate instead of adding a parallel backend-specific example when its CLI,
  pipeline shape, and terminal behavior can remain the same.
- Put platform alternatives behind `cfg(target_os)` branches in one source file
  and select dependencies/features per target in that crate's `Cargo.toml`.
  Keep construction as the only platform difference unless the backend truly
  forces a behavioral divergence.
- Add a new crate only when no existing example can demonstrate the behavior
  without obscuring its original purpose. Place it in the category matching the
  purpose, not merely the backend.

## Preserve repository conventions

- Require media paths and other essential inputs as command-line arguments.
  Missing input prints a `usage:` line to stderr and exits non-zero; never add a
  repository-local default media path.
- Keep the pipeline, terminal sink, CLI, exit behavior, and output contract
  aligned across platform branches.
- Initialize the private `media_pp` logger at `Trace`, use `./logs` with the
  Cargo package name as prefix, and retain the returned guard through shutdown.
- Reuse `render_common` or another existing shared helper when its ownership and
  event-loop contract match. Do not create a helper abstraction for one caller.
- Keep target-specific dependencies and library features under compatible Cargo
  target tables and Rust `cfg`s so unsupported platforms do not resolve or
  compile the backend path.

## Document what runs

- Every new example crate includes `README.md` in the same change. Update the
  README whenever the pipeline, CLI, outputs, or observable behavior changes.
- Source the README from an existing crate-level doc comment where available,
  reformatting without changing its claims. Otherwise write only what the
  construction code demonstrates.
- State the actual graph using `SourceType -> Filter -> SinkType` notation and
  account for branches explicitly. Do not infer the graph from imports.
- Keep general feature tables, installation requirements, and repository-wide
  build instructions in the root README or type documentation, not the example
  README.
- Update the root README example inventory when a public example is added,
  removed, renamed, or materially changes its purpose.

## Verify end to end

- Build and run the actual example with the required feature set and realistic
  arguments. A successful compile or process exit is not sufficient.
- Exercise missing-argument behavior and confirm the usage text and non-zero
  exit status.
- For recorded audio or video, inspect streams, codecs, timestamps, duration,
  and finalization with `ffprobe`. Use assertions that follow from the example
  rather than from one particular media fixture.
- For visual behavior, extract representative frames and inspect the expected
  pixels or content. For live rendering, verify clean startup, completion,
  close, and worker-panic shutdown paths as applicable.
- Detect unavailable hardware, desktop sessions, portal tokens, fonts, and
  fixtures explicitly. Report a skipped prerequisite as unverified coverage,
  not as a successful end-to-end run.
- Remove verification media and other generated artifacts before handoff. Run
  targeted tests, the example's build, `cargo fmt --all -- --check`,
  `git diff --check`, and `git status --short`.
