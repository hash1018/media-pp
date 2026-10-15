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

1. Install Rust, a C toolchain, libclang and **FFmpeg 8.0** —
   [`scripts/setup-ffmpeg`][scripts] builds the right FFmpeg.
2. Add the crate, with the features for your platform:

   ```toml
   [dependencies]
   media-pp = { version = "0.4", features = ["d3d11"] }
   ```

3. Check the machine with `cargo run -p doctor` in this repository.

[Getting started][getting-started] goes through it step by step,
with a first pipeline and a first player.

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
everything on that platform; [docs.rs] has what every platform shares. Both
document the latest release. The guides above follow the repository's
`main`, which can be ahead of it; what is not released yet, when there is
anything, is under *Unreleased* in the changelog.

The library is in `lib/`, and each directory under `examples/` is a crate of
its own. [`CONTRIBUTING.md`][contributing] says how to test a change, and
[`AGENTS.md`][agents] holds the repository's design and review rules, for
people and coding agents alike.

## License

Licensed under either the [Apache License, Version 2.0][apache] or the
[MIT License][mit], at your option.

`media-pp` does not bundle FFmpeg. You are responsible for complying with the
license of your FFmpeg build and the codecs you enable.

[scripts]: https://github.com/hash1018/media-pp/tree/main/scripts
[getting-started]: https://github.com/hash1018/media-pp/blob/main/docs/getting-started.md
[concepts]: https://github.com/hash1018/media-pp/blob/main/docs/guide/concepts.md
[choosing-a-backend]: https://github.com/hash1018/media-pp/blob/main/docs/guide/choosing-a-backend.md
[recipes]: https://github.com/hash1018/media-pp/blob/main/docs/guide/recipes.md
[troubleshooting]: https://github.com/hash1018/media-pp/blob/main/docs/guide/troubleshooting.md
[elements]: https://github.com/hash1018/media-pp/blob/main/docs/elements.md
[features]: https://github.com/hash1018/media-pp/blob/main/docs/features.md
[examples]: https://github.com/hash1018/media-pp/blob/main/docs/examples.md
[all-docs]: https://github.com/hash1018/media-pp/blob/main/docs/README.md
[contributing]: https://github.com/hash1018/media-pp/blob/main/CONTRIBUTING.md
[agents]: https://github.com/hash1018/media-pp/blob/main/AGENTS.md
[apache]: https://github.com/hash1018/media-pp/blob/main/LICENSE-APACHE
[mit]: https://github.com/hash1018/media-pp/blob/main/LICENSE-MIT
[`CHANGELOG.md`]: https://github.com/hash1018/media-pp/blob/main/CHANGELOG.md
[`ffmpeg-next`]: https://github.com/zmwangx/rust-ffmpeg
[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
[Linux]: https://hash1018.github.io/media-pp/linux/media_pp/
