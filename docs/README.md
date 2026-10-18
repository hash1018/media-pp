# media-pp documentation

These pages follow the repository's `main` branch, which can be ahead of the
release on crates.io; what is not released yet, when there is anything, is
under *Unreleased* in [`CHANGELOG.md`](../CHANGELOG.md). The API pages and
[docs.rs] document the latest release.

## Start here

| | |
|---|---|
| [Getting started](getting-started.md) | Installing, checking the machine, a first pipeline, playing a file |
| [How media-pp works](guide/concepts.md) | The ideas behind every pipeline: elements, branches, threads, errors, ending, seeking, buffers, GPUs |

## Using media-pp

| | |
|---|---|
| [Choosing a backend](guide/choosing-a-backend.md) | D3D11, D3D12, CUDA, Vulkan, Metal or software — which to use where |
| [Recipes](guide/recipes.md) | What to use for a job, and which example does it |
| [Troubleshooting](guide/troubleshooting.md) | Build, start-up, wiring and runtime problems, and their fixes |

## Reference

| | |
|---|---|
| [Elements](elements.md) | Every element, by job and by backend |
| [Features](features.md) | Every Cargo feature, its platform, and what it needs to build and run |
| [Examples](examples.md) | The example crates and how to run them |
| API documentation | The [Windows], [macOS] and [Linux] API pages, each with everything on that platform; [docs.rs] for what every platform shares |
| [`CHANGELOG.md`](../CHANGELOG.md) | What changed between versions, and what to write instead |

## Building and contributing

| | |
|---|---|
| [Building on Windows](building/windows.md), [Linux](building/linux.md), [macOS](building/macos.md) | Setting up a machine to build and test the repository |
| [`scripts/`](../scripts) | The FFmpeg setup scripts |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Tests, the control conformance matrix, soak tests |
| [`AGENTS.md`](../AGENTS.md) | The repository's design and review rules |

## Design records and measurements

| | |
|---|---|
| [Stream events](stream-events.md) | How segments, flushes and the end of a stream travel through a pipeline |
| [D3D12 inference](d3d12-inference.md) | Detection on any GPU through DirectML: what was measured and why it is built so |
| [Vision benchmarks](benchmarks/vision/) | How fast detection, tracking and classification run, on Linux, Windows and macOS |
| [Speech benchmarks](benchmarks/speech/) | How fast and how well the whisper transcribers hear, per model and backend, and how late live captions run |

[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
[Linux]: https://hash1018.github.io/media-pp/linux/media_pp/
