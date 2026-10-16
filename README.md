# media-pp

A GStreamer-flavored media pipeline library for Rust, built on
[`ffmpeg-next`]. Decode, capture, composite, encode, stream and run object
detection — on the CPU, or end to end on the GPU with D3D11, D3D12, CUDA,
Vulkan or Metal.

```rust,no_run
use media_pp::player::{Player, PlayerOptions};

let player = Player::open("video.mp4", PlayerOptions::default())?;
player.play()?;
```

This is an excerpt, not a complete program: keep the player alive by
handling its events. [Getting started][getting-started] has the complete
program and the platform features it needs.

## What it does

- **Pipelines you can read.** Elements are connected in the order media
  flows; each runs as a plain function call, and threads exist only where
  you put a `Queue`. A failure comes back as a `Result`, naming the element.
- **Wiring checked before it runs.** Connecting things that cannot fit is
  refused up front, with what to put between them.
- **On the GPU end to end.** Decode, scale, composite, key, detect and
  encode without copying pictures back to the CPU, on whichever GPU API the
  machine has. `VideoDecodeBin`, `VideoEncodeBin` and `Player` choose
  hardware where it works and software where it does not.
- **Capture and output.** Screens, windows, cameras and sound on Windows,
  Linux and macOS; files, HLS, RTMP, RTSP, WebRTC and virtual cameras.
- **Playback control.** Pause, seek, step, change speed, play backwards.
- **Pipelines that change while running.** Add a recording to a running
  preview, add and move compositor layers, swap filters.
- **Vision.** ONNX detectors, classifiers and embedders, object tracking,
  counting, and drawing or hiding what was found.

## Getting started

1. Clone this repository, then install Rust, a C toolchain, libclang and
   **FFmpeg 8.0** — [`scripts/setup-ffmpeg`][scripts] builds the right
   FFmpeg. [Getting started][getting-started] gives the commands and where
   to run them. On Windows, open a new terminal after running the setup
   script with `-Persist`.
2. In the repository, run `cargo run -p doctor` to check the machine.
3. In your own Cargo project, add the crate. These platform-specific
   dependencies enable the `Player` shown above:

   ```toml
   [target.'cfg(target_os = "windows")'.dependencies]
   media-pp = { version = "0.4", features = ["d3d11", "wasapi-renderer"] }

   [target.'cfg(target_os = "linux")'.dependencies]
   media-pp = { version = "0.4", features = ["vulkan", "pipewire-audio-renderer"] }

   [target.'cfg(target_os = "macos")'.dependencies]
   media-pp = { version = "0.4", features = ["metal", "coreaudio-renderer"] }
   ```

   Linux playback also needs FFmpeg built with Vulkan and PipeWire's
   development files; the guide covers those before building the player.

[Getting started][getting-started] goes through it step by step,
starting with a software pipeline that needs no video file and no GPU
features, then reading a file and playing it.

## Documentation

| | |
|---|---|
| [Getting started][getting-started] | Installing, a first pipeline, playing a file |
| [How media-pp works][concepts] | Sources, filters and sinks, threads, errors, ending, seeking, GPUs |
| [Choosing a backend][choosing-a-backend] | Which GPU API to use on which machine |
| [Recipes][recipes] | Playing, converting, recording, streaming, compositing, detection |
| [Troubleshooting][troubleshooting] | Build, start-up and runtime problems |
| [Elements][elements] | Every element, by job and backend |
| [Features][features] | Every Cargo feature and what it needs |
| [Examples][examples] | The runnable examples |
| [`CHANGELOG.md`] | What changed between versions, and what to write instead |
| [All documentation][all-docs] | Building, design records, benchmarks, contributing |

API documentation: the [Windows], [macOS] and [Linux] API pages, each with
everything on that platform, document the latest release; [docs.rs] has what
every platform shares. Read on GitHub, this page's guides follow the
repository's `main`, which can be ahead of the release — what is not
released yet is under *Unreleased* in the changelog — and docs.rs shows the
latest release. Read on crates.io, the guides and docs.rs are the version
shown there.

The library is in `lib/`, and each directory under `examples/` is a crate of
its own. [`CONTRIBUTING.md`][contributing] says how to test a change, and
[`AGENTS.md`][agents] holds the repository's design and review rules, for
people and coding agents alike.

## License

Licensed under either the [Apache License, Version 2.0][apache] or the
[MIT License][mit], at your option.

`media-pp` does not bundle FFmpeg. You are responsible for complying with the
license of your FFmpeg build and the codecs you enable.

[scripts]: scripts
[getting-started]: docs/getting-started.md
[concepts]: docs/guide/concepts.md
[choosing-a-backend]: docs/guide/choosing-a-backend.md
[recipes]: docs/guide/recipes.md
[troubleshooting]: docs/guide/troubleshooting.md
[elements]: docs/elements.md
[features]: docs/features.md
[examples]: docs/examples.md
[all-docs]: docs/README.md
[contributing]: CONTRIBUTING.md
[agents]: AGENTS.md
[apache]: LICENSE-APACHE
[mit]: LICENSE-MIT
[`CHANGELOG.md`]: CHANGELOG.md
[`ffmpeg-next`]: https://github.com/zmwangx/rust-ffmpeg
[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
[Linux]: https://hash1018.github.io/media-pp/linux/media_pp/
