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
   [`scripts/setup-ffmpeg`](scripts) builds the right FFmpeg.
2. Add the crate, with the features for your platform:

   ```toml
   [dependencies]
   media-pp = { version = "0.3", features = ["d3d11"] }
   ```

3. Check the machine with `cargo run -p doctor` in this repository.

[Getting started](docs/getting-started.md) goes through it step by step,
with a first pipeline and a first player.

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Installing, a first pipeline, playing a file |
| [How media-pp works](docs/guide/concepts.md) | Sources, filters and sinks, threads, errors, ending, seeking, GPUs |
| [Choosing a backend](docs/guide/choosing-a-backend.md) | Which GPU API to use on which machine |
| [Recipes](docs/guide/recipes.md) | Playing, converting, recording, streaming, compositing, detection |
| [Troubleshooting](docs/guide/troubleshooting.md) | Build, start-up and runtime problems |
| [Elements](docs/elements.md) | Every element, by job and backend |
| [Features](docs/features.md) | Every Cargo feature and what it needs |
| [Examples](docs/examples.md) | The runnable examples |
| [`CHANGELOG.md`] | What changed between versions, and what to write instead |
| [All documentation](docs/README.md) | Building, design records, benchmarks, contributing |

API documentation: [docs.rs] for the shared and Linux API, and the
[Windows] and [macOS] API pages for everything on those platforms.

The library is in `lib/`, and each directory under `examples/` is a crate of
its own. [`CONTRIBUTING.md`](CONTRIBUTING.md) says how to test a change, and
[`AGENTS.md`](AGENTS.md) holds the repository's design and review rules, for
people and coding agents alike.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.

`media-pp` does not bundle FFmpeg. You are responsible for complying with the
license of your FFmpeg build and the codecs you enable.

[`CHANGELOG.md`]: https://github.com/hash1018/media-pp/blob/main/CHANGELOG.md
[`ffmpeg-next`]: https://github.com/zmwangx/rust-ffmpeg
[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
