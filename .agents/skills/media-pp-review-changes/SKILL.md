---
name: media-pp-review-changes
description: Review recent commits or a diff in media-pp for correctness regressions, lifecycle failures, error-boundary violations, feature-gate mismatches, and missing tests. Use for review, audit, or "what should be fixed or checked" requests; do not modify code unless the user separately requests fixes.
---

# Review media-pp changes

Treat review as read-only. Read `README.md`, `AGENTS.md`, the
requested commit range or worktree diff, and the affected implementation and
tests. Do not infer permission to fix findings, stage files, or rewrite the
working tree.

## Establish scope and intent

- Determine the exact commits or diff being reviewed and inspect commit
  messages, file-level changes, and surrounding code. Preserve unrelated user
  changes and distinguish committed changes from the current worktree.
- Reconstruct the intended public and runtime behavior from callers, tests,
  documentation, and the nearest existing implementation. Code and observable
  tests outrank stale prose.
- Check whether a failure also occurs on the relevant parent or unchanged
  configuration before attributing it to the reviewed change. Separate local
  dependency, generated-binding, platform, fixture, and hardware failures from
  repository regressions.

## Trace the changed contracts

- Follow every changed success, error, early-return, panic, and drop path. Check
  state after partial allocation or registration and ensure a failed mutation
  preserves the previous working state.
- Trace ownership across `Arc`, `Weak`, channels, handles, workers, GPU objects,
  codec contexts, pools, files, and helper processes. Verify stop/join/drop and
  that retained handles do not keep unrelated pipelines or buses alive.
- For data paths, validate buffer variants and backend invariants, metadata
  preservation, EOS drain/forwarding, and `Stop` abandonment semantics.
- For a declared link contract, check that it states only what construction
  settles and no more. A contract narrower than what the element really accepts
  refuses a working pipeline at build time, which is a worse failure than the
  runtime error it was meant to pre-empt; confirm a passing chain is covered,
  not just a refused one.
- For concurrency, check lock scope, callbacks and blocking calls under locks,
  source control responsiveness, Queue recovery boundaries, and fan-in/fan-out
  failure isolation.
- For logging, verify stable graph identity, correct `PpLog` attribution,
  disabled-level hot-path cost, and complete topology/control records.
- For public and gated code, compare module declarations, imports, flat
  re-exports, error conversions, Cargo features/dependencies, platform cfgs,
  README inventory, examples, and both enabled and disabled builds.

## Evaluate evidence and coverage

- Search for tests that exercise the changed observable contract, especially
  invalid input, failure after partial progress, state after error, EOS,
  control, replacement, teardown, and duplicate/stale handles.
- Identify whether a new per-cycle resource requires
  `$media-pp-soak-analysis`; do not claim a leak from a one-shot memory delta.
- Run the narrowest relevant checks first, then broaden only where the changed
  feature or platform warrants it. Record skipped hardware or media fixtures as
  unverified, even when the test harness exits successfully.
- Do not repeatedly repair or work around a known external build failure during
  a review. Establish whether it is change-related, report the limitation, and
  continue with safe independent checks.

## Report findings

Lead with actionable findings ordered by severity. For each finding include:

- severity and concise failure statement;
- exact file and line or changed symbol;
- a concrete input, event sequence, or ownership path that triggers it;
- the observable consequence;
- why current tests do not prevent it and the regression test needed.

Use high severity for corruption, security impact, broadly reachable hangs or
data loss; medium for realistic correctness, resource, or lifecycle failures;
and low for narrow inconsistencies or maintainability defects with concrete
impact. Do not report style preferences, hypothetical future needs, or claims
without a reachable failure path.

If no findings survive verification, say so directly and list residual risks or
checks that could not run. Keep the summary secondary to findings and never
describe unexecuted validation as passed.
