# scripts

- **`setup-ffmpeg.ps1`** (Windows) and **`setup-ffmpeg.sh`** (Linux, macOS)
  build FFmpeg 8.0.1 — the release media-pp is built against — from vcpkg at
  the tag CI pins, into a folder of your own, and point the build at it
  through `FFMPEG_DIR` and the library search path. Run with `-Persist`
  (Windows) to keep the settings for your account, or source the `env.sh`
  the shell script writes. Then `cargo run -p doctor` checks the machine.
- **`crate-readme.sh`** writes `lib/README.md`, the README crates.io shows:
  `README.md` with each link to a repository file pointed at the tag of
  the version `lib/Cargo.toml` says, so a reader from crates.io opens the
  documentation of the release they installed rather than `main`'s. Edit
  `README.md`, then run it; `--check`, which CI runs, fails where the two
  disagree.
