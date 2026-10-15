# media-pp documentation

These pages follow the repository's `main` branch, which can be ahead of the
release on crates.io. What is not released yet is under *Unreleased* in
[`CHANGELOG.md`](../CHANGELOG.md); [docs.rs] documents the released crate.

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
| API documentation | [docs.rs] for the shared and Linux API; the [Windows] and [macOS] API pages for the rest |
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

[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
