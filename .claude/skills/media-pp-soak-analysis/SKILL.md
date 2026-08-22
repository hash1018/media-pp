---
name: media-pp-soak-analysis
description: Diagnose memory, GPU-resource, handle, or worker-lifetime growth in media-pp and add or evaluate isolated scenarios in lib/tests/soak.rs. Use for leak investigations, soak-test failures, fitted-slope interpretation, threshold calibration, or lifecycle stress coverage; do not activate for ordinary one-shot correctness tests without a resource-growth question.
---

# media-pp soak analysis

Read `README.md` and `AGENT_GUIDELINES.md`. Use the repository's existing soak
harness instead of inventing a parallel benchmark. Read the module
documentation and the nearest comparable scenario in `lib/tests/soak.rs`, then
read the relevant helpers in `lib/tests/common/mod.rs` and, for GPU resources,
`lib/tests/common/gpu.rs`.

## Establish the experiment

- Define the suspected resource and the complete lifecycle that should release
  it. A cycle normally includes construction, run or use, teardown, worker
  completion, and dropping every owner.
- Prefer extending the nearest existing scenario. Add a scenario only when the
  new element or ownership path introduces a lifecycle that existing coverage
  does not exercise.
- Start every cycle-driven scenario with `isolate!()`. Whole-process gauges
  cannot produce comparable trends after another scenario has disturbed the
  allocator or graphics driver.
- Preserve observable workload assertions and inspect bus errors. A flat trend
  from a workload that stopped doing useful work is not evidence of correct
  cleanup.
- Exercise every materially different teardown path, such as both ordered
  `finish` and abandoning `stop`, when the resource can be released through
  either path.

## Select gauges and a measurement window

- Measure process private bytes for Rust and C/FFmpeg allocations; a Rust
  allocator counter cannot see the C heap.
- Add the backend gauge for GPU resources: DXGI video-memory usage and D3D11
  live-object count where available, or NVIDIA per-process memory where the
  driver exposes it. Treat an unavailable gauge as unknown, not zero.
- Warm up lazy initialization and resource caches before collecting samples.
  Use `settle()` where the baseline can still be moving, but do not treat it as
  a substitute for process isolation.
- Fit and report a `Trend`; do not diagnose a leak from one before/after delta.
  Interpret a passing `assert_flat` only as no growth above the printed
  resolution for that sample window.
- If the window is too noisy, increase `MEDIA_PP_SOAK_ITERS` or
  `MEDIA_PP_SOAK_SECS`. Do not lower sensitivity or raise a threshold merely to
  obtain a pass.

## Diagnose before changing code

- Reproduce the same scenario more than once and retain its sample series,
  fitted slope, standard error, and resolution.
- Separate application ownership from allocator and driver caching. For a GPU
  trend, compare a mode that avoids the suspected allocation or a minimal raw
  backend loop when practical, and extend warm-up past demonstrated cache
  saturation.
- Form an ownership hypothesis from the implementation and teardown paths, but
  require a same-scenario A/B run before claiming that a code change fixed the
  growth.
- When the user requested diagnosis only, report the evidence and proposed fix;
  do not modify production code without authorization.

## Run and report

Run the narrow scenario first with `--ignored --nocapture`; select the platform
features it actually needs. Then run the affected soak group or feature set if
the targeted result supports the hypothesis. Hardware and portal scenarios must
skip with a clear reason when their documented prerequisites are unavailable;
never report a skipped scenario as measured coverage.

Record:

- exact command, feature set, fixture variables, hardware, and iteration count;
- warm-up and measured sample counts;
- each gauge's slope, standard error, threshold, and printed resolution;
- whether the run measured, skipped, or failed and why;
- baseline versus changed results from the same scenario;
- remaining gauges or environments that were not observable.

Before handoff, run formatting and the relevant ordinary regression tests in
addition to the soak scenario, followed by `git diff --check` and
`git status --short`.
