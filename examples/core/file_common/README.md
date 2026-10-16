# file_common

Shared, unpublished support for `remux`, `transcode` and `resize`, not an
executable or part of media-pp's public API.

- `FileArgs` parses paths and `--overwrite`; `--` ends option parsing.
- `SafeOutput` refuses input/output aliases (including hard links), stages
  the export beside the destination, and removes a partial output on drop.
  Existing destinations require `--overwrite`. `commit` publishes only after
  the caller has closed the pipeline; without overwrite it never clobbers a
  file that appeared while the export ran. It rechecks aliases before publish.
- `run_to_end` owns and closes the pipeline, returning the first bus failure
  instead of reporting a partial export as success. The library's own
  per-buffer recovery policy is unchanged.

The staging file keeps the destination's extension for FFmpeg's container
selection. Publication uses `tempfile`'s same-filesystem persistence, without
deleting the old destination first; this is not a power-loss durability
guarantee or protection against a hostile process racing path replacement.

```sh
cargo test -p file_common
```
