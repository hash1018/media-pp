# Getting started

From nothing to a program that reads a file and one that plays it.

1. [Install what media-pp needs](#1-install-what-media-pp-needs)
2. [Add the crate](#2-add-the-crate)
3. [Check the machine](#3-check-the-machine)
4. [A first pipeline](#4-a-first-pipeline)
5. [Playing a file](#5-playing-a-file)
6. [Where to go next](#6-where-to-go-next)

## 1. Install what media-pp needs

| | Why | How |
|---|---|---|
| **Rust 1.88** or newer | | [rustup](https://rustup.rs) |
| **A C toolchain** | FFmpeg is built from source | Windows: Visual Studio Build Tools, C++ workload. Linux: `build-essential`. macOS: `xcode-select --install` |
| **libclang** | The FFmpeg binding is generated at build time | Windows: LLVM, with `LIBCLANG_PATH` set to its `bin` if not found. Linux: `libclang-dev`. macOS: comes with Xcode's tools |
| **FFmpeg 8.0** development files — exactly 8.0, not 8.1 | media-pp is built on FFmpeg | The setup script below |

FFmpeg has to be **8.0**: the binding media-pp uses does not build against
8.1's headers, and no prebuilt 8.0 is published any more. The script in this
repository builds the one CI uses (tens of minutes, once) and points your
builds at it:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup-ffmpeg.ps1 -Persist
```

```sh
# Linux, macOS
scripts/setup-ffmpeg.sh
. ~/.local/share/media-pp/ffmpeg/env.sh   # in each shell, or in your profile
```

What the script does by hand, and what each platform needs beyond it, is in
building on [Windows](building/windows.md), [Linux](building/linux.md) or
[macOS](building/macos.md).

## 2. Add the crate

```toml
[dependencies]
media-pp = { version = "0.4", features = ["d3d11"] }
```

media-pp has **no default features**: the software elements are always
there, and each GPU backend, capture API and output is a feature you turn
on. [Choosing a backend](guide/choosing-a-backend.md) says which you want;
[features](features.md) lists them all with what each needs.

FFmpeg's Rust binding is re-exported as `media_pp::ffmpeg`. Use it rather
than adding `ffmpeg-next` yourself, so the two can never disagree.

## 3. Check the machine

The doctor, in this repository, checks the FFmpeg your builds find, its
encoders, and whether each GPU backend opens a device here:

```sh
git clone https://github.com/hash1018/media-pp
cd media-pp
cargo run -p doctor
```

```text
FFmpeg
  ok    libavcodec: 62.11.100
  ...
GPU backends in this build
  ok    D3D11: device made
  ok    D3D12: device made
  --    CUDA: none here (...)
```

Each problem it finds says what to do. More fixes are in
[troubleshooting](guide/troubleshooting.md).

## 4. A first pipeline

Count the video packets in a file:

```rust,no_run
use media_pp::{
    bus::BusEvent,
    elements::{FileDemuxer, PacketCounter},
    ffmpeg,
    pipeline::Pipeline,
};

fn main() -> media_pp::Result<()> {
    // The source: a file, split into its streams.
    let (source, _) = FileDemuxer::open("file", "video.mp4")?;
    let video = source.best(ffmpeg::media::Type::Video)?;

    // The sink: counts what reaches it.
    let (counter, packets) = PacketCounter::new("counter");

    // A branch from the video stream to the counter.
    let (pipeline, ()) = Pipeline::new("count", source, |source, ctx| {
        ctx.attach(source, video.index, ctx.branch().to(counter)?)?;
        Ok(())
    })?;
    pipeline.run()?;

    // A file's source waits at its end, in case of a seek, so the pipeline
    // is stopped once every sink has everything: `Finished`. An error does
    // not end it by itself either.
    for event in pipeline.bus().iter() {
        if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }
    println!("packets: {}", packets.get());
    Ok(())
}
```

What happens here, and everything a bigger pipeline adds — decoders, queues,
GPUs, seeking — is in [how media-pp works](guide/concepts.md).

## 5. Playing a file

`Player` builds the whole pipeline: decoding on the GPU where it can, a
window, the default audio output. It needs these features:

| Windows | Linux | macOS |
|---|---|---|
| `d3d11`, `wasapi-renderer` | `vulkan`, `pipewire-audio-renderer` | `metal`, `coreaudio-renderer` |

On macOS every window belongs to the main thread, so `main` runs the player
inside `media_pp::elements::run_with_windows`:

```rust,no_run
use media_pp::player::{Player, PlayerEvent, PlayerOptions};

#[cfg(not(target_os = "macos"))]
fn main() -> media_pp::Result<()> {
    play()
}

// AppKit's event loop runs on the main thread; the player runs beside it.
#[cfg(target_os = "macos")]
fn main() -> media_pp::Result<()> {
    media_pp::elements::run_with_windows(play)
}

fn play() -> media_pp::Result<()> {
    let player = Player::open("video.mp4", PlayerOptions::default())?;
    player.play()?;
    while let Some(event) = player.next_event() {
        match event {
            // Space, arrows, `.` `,` `-` `+`, F; `false` for Escape or a close.
            PlayerEvent::Window(event) if !player.respond_to(&event) => break,
            PlayerEvent::Ended => break,
            PlayerEvent::Error { name, error } => eprintln!("{name}: {error}"),
            _ => {}
        }
    }
    Ok(())
}
```

The [`player`](../examples/render/player) example is this program, ready to
run: `cargo run -p player -- video.mp4`.

## 6. Where to go next

| To | Read |
|---|---|
| Understand pipelines, threads, errors, ending and seeking | [How media-pp works](guide/concepts.md) |
| Pick a GPU backend | [Choosing a backend](guide/choosing-a-backend.md) |
| Do a particular job — record, stream, composite, detect | [Recipes](guide/recipes.md) |
| Find an element | [Elements](elements.md) |
| Run an example | [Examples](examples.md) |
| Fix a build or start-up problem | [Troubleshooting](guide/troubleshooting.md) |
| Look up a type | API documentation: [Windows], [macOS], [Linux]; [docs.rs] for what every platform shares |
| Upgrade between versions | [`CHANGELOG.md`](../CHANGELOG.md), which says what to write instead |

[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
[Linux]: https://hash1018.github.io/media-pp/linux/media_pp/
