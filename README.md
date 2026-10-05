# media-pp

`media-pp` is a small, GStreamer-flavored media pipeline library for Rust,
built on [`ffmpeg-next`]. Stages are synchronous calls by default, and thread
boundaries are explicit bounded queues.

## What it does

- **Errors come back as `Result`.** A stage's failure returns straight up the
  call stack; only past a `Queue` does it become a bus event.
- **Running pipelines change shape.** Branches are added and finished while
  buffers flow, and filters are swapped without reopening the source.
- **GPU-resident end to end** on D3D11, D3D12, CUDA, Vulkan or
  VideoToolbox and Metal: decode, scale, composite, key and encode without
  copying pictures back to the CPU. `VideoDecodeBin` and `VideoEncodeBin`
  pick the hardware where it opens and fall back to software where it does
  not.
- **Capture** of screens, windows, cameras and audio on Windows, Linux and
  macOS; **output** to files, HLS, RTMP and RTSP servers, and WebRTC.
- **Playback control**: seek, pause, step, speed and reverse, with a preroll
  that shows the picture sought to before playing on. `Player` plays a file
  with its sound in a window in a dozen lines.

Every element, by what it does and which backend it runs on, is in
[docs/elements.md](docs/elements.md).

## Getting it

```toml
[dependencies]
media-pp = "0.3"
```

FFmpeg 8.0 or newer development libraries and Rust 1.88 or newer are
needed; the library has no default features, and each backend, capture and
output is one — see [docs/features.md](docs/features.md). A first pipeline
and a first `Player` are in
[docs/getting-started.md](docs/getting-started.md).

How a pipeline runs is the crate documentation's first page, on [docs.rs]
for the backend-independent and Linux API, and in the
[Windows API documentation] and the [macOS API documentation] for
everything on those platforms.

## Building

The library crate lives in `lib/`; each directory below `examples/` is a
crate of its own:

```sh
cargo run -p player -- path/to/video.mp4
```

Setting up FFmpeg and each platform's SDKs is in
[docs/building](docs/building/), and testing in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Adding the crate, a file's pipeline, playing a file, where to read on |
| [Elements](docs/elements.md) | Every element, by what it does and by backend |
| [Features](docs/features.md) | Every feature flag, its platform, and what it needs to build |
| [Examples](docs/examples.md) | The example crates and how to run them |
| [Building](docs/building/) | Setting up Windows, Linux and macOS to build and test |
| [Stream events](docs/stream-events.md) | Design record of how segments, flushes and the end of a stream travel |
| [`CHANGELOG.md`] | What changed between versions, and what to write instead |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Testing, the control conformance matrix, soak tests |

[`AGENTS.md`](AGENTS.md) holds the repository's design and review rules, for
people and coding agents alike.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT License](LICENSE-MIT), at your option.

`media-pp` does not bundle FFmpeg. Users are responsible for complying with
the license of their FFmpeg build and optional codecs.

[`CHANGELOG.md`]: https://github.com/hash1018/media-pp/blob/main/CHANGELOG.md
[`ffmpeg-next`]: https://github.com/zmwangx/rust-ffmpeg
[docs.rs]: https://docs.rs/media-pp
[Windows API documentation]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS API documentation]: https://hash1018.github.io/media-pp/macos/media_pp/
