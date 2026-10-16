# Getting started

From setting up the machine to a program that needs no input file, then one
that reads a file and one that plays it. The repository holds the setup
scripts and runnable examples; your own application will be a separate
Cargo project.

1. [Get the repository](#1-get-the-repository)
2. [Install what media-pp needs](#2-install-what-media-pp-needs)
3. [Check the machine](#3-check-the-machine)
4. [Create your project](#4-create-your-project)
5. [A first pipeline, without a file](#5-a-first-pipeline-without-a-file)
6. [Read a file](#6-read-a-file)
7. [Playing a file](#7-playing-a-file)
8. [Where to go next](#8-where-to-go-next)

## 1. Get the repository

Install [Git](https://git-scm.com) first. These commands put the checkout in
your home directory; if you choose another directory, use that path where
later steps return to the checkout. If it is already cloned, use the
existing checkout instead of cloning it again.

**Windows — PowerShell:**

```powershell
cd $env:USERPROFILE
git clone https://github.com/hash1018/media-pp
cd media-pp
```

**Linux, macOS — a terminal:**

```sh
cd ~
git clone https://github.com/hash1018/media-pp
cd media-pp
```

You are now in the **media-pp repository root**, which holds `scripts/` and
the example crates. Run the next step there.

## 2. Install what media-pp needs

| | Why | How |
|---|---|---|
| **Rust 1.88** or newer | Builds your program and the library | [rustup](https://rustup.rs) |
| **A C toolchain** | FFmpeg is built from source | Windows: Visual Studio Build Tools, C++ workload. Linux: `build-essential`. macOS: `xcode-select --install` |
| **libclang** | The FFmpeg binding is generated at build time | Windows: LLVM, with `LIBCLANG_PATH` set to its `bin` if not found. Linux: `libclang-dev`. macOS: comes with Xcode's tools |
| **FFmpeg 8.0** development files — exactly 8.0, not 8.1 | media-pp is built on FFmpeg | The setup script below |

The binding does not build against 8.1's headers. The setup scripts build
**8.0.1**, the version CI uses, and point your builds at it. The first build
takes tens of minutes. Install the tools listed for your platform in
building on [Windows](building/windows.md), [Linux](building/linux.md) or
[macOS](building/macos.md) before running the script.

**Windows — from the repository root:**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-ffmpeg.ps1 -Persist
```

`-Persist` saves `FFMPEG_DIR` and the DLL search path for your user account.
The command runs in a child PowerShell process: it **does not update the
terminal you ran it from**. Open a new terminal before step 3. If a new tab
still has the old environment, restart the terminal application or editor.

**Linux — from the repository root:**

```sh
scripts/setup-ffmpeg.sh
. ~/.local/share/media-pp/ffmpeg/env.sh
```

**macOS — from the repository root:**

```sh
scripts/setup-ffmpeg.sh
. ~/.local/share/media-pp/ffmpeg/env.sh
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$FFMPEG_DIR/lib"
```

On Linux and macOS, source `env.sh` in each new shell, or add that line to
your shell profile. On macOS keep the `RUSTFLAGS` setting too: the vcpkg
dylibs need an rpath on each executable, as [building on macOS](building/macos.md)
explains. If you passed a custom destination to the script, source the
`env.sh` at the path it printed instead of the default above.

## 3. Check the machine

**Windows — open a new PowerShell terminal**, then return to the checkout:

```powershell
cd "$env:USERPROFILE\media-pp"
cargo run -p doctor
```

Use your actual checkout path if you cloned elsewhere. If `FFMPEG_DIR` is
still empty, restart the terminal application or set `FFMPEG_DIR` and `PATH`
to the paths printed by the setup script before building.

**Linux, macOS — in the shell configured in step 2:**

```sh
cd ~/media-pp
cargo run -p doctor
```

The doctor checks the FFmpeg your builds find, its encoders, and whether
each GPU backend in its build opens a device. Output looks like this;
versions and available devices depend on the machine:

```text
media-pp doctor

FFmpeg
  ok    libavcodec: 62.11.100
  ...
GPU backends in this build
  ok    D3D11: device made
  ...
Nothing wrong found.
```

Each problem it finds says what to do. More fixes are in
[troubleshooting](guide/troubleshooting.md). You do not need a GPU for the
first pipeline below; the doctor also reports what the machine offers for
later hardware examples.

## 4. Create your project

Leave the repository and make a separate application beside it.

**Windows — in the new terminal from step 3:**

```powershell
cd $env:USERPROFILE
cargo new media-pp-hello
cd media-pp-hello
```

**Linux, macOS — in the shell configured in step 2:**

```sh
cd ~
cargo new media-pp-hello
cd media-pp-hello
```

You are now in **your application directory**, not the media-pp checkout.
Run steps 5–7 here. In its generated `Cargo.toml`, add this line under the
existing `[dependencies]` heading:

```toml
[dependencies]
media-pp = "0.4"
```

This uses the release on crates.io. media-pp has **no default features**:
the software elements are always there, and each GPU backend, capture API
and output is a feature you turn on later. No GPU feature is needed yet.

FFmpeg's Rust binding is re-exported as `media_pp::ffmpeg`. Use it rather
than adding `ffmpeg-next` yourself, so the two can never disagree.

## 5. A first pipeline, without a file

Replace **your application's `src/main.rs`** with this complete program.
It generates pictures and counts them: no video file, camera or window is
needed.

```text
TestVideoSource -> FrameCounter
```

```rust,no_run
use std::time::Duration;

use media_pp::{
    elements::{FrameCounter, TestVideoOptions, TestVideoSource},
    pipeline::Pipeline,
};

fn main() -> media_pp::Result<()> {
    let _log = media_pp::log::init("hello", "logs", media_pp::log::Level::Trace, 7)?;
    let source = TestVideoSource::new("source", TestVideoOptions::default());
    let (counter, frames) = FrameCounter::new("counter");

    let (pipeline, ()) = Pipeline::new("hello", source, |source, ctx| {
        ctx.attach(source, 0, ctx.branch().to(counter)?)?;
        Ok(())
    })?;
    pipeline.run()?;
    std::thread::sleep(Duration::from_millis(500));
    // This is a live source: like a camera, it runs until stopped.
    pipeline.stop();

    println!("frames: {}", frames.get());
    Ok(())
}
```

Run it **from your application directory**:

```sh
cargo run
```

After about half a second it exits with output such as:

```text
frames: 15
```

The exact count depends on scheduling; a positive count shows pictures
reached the counter. Diagnostics are written under your application's
`logs/` directory; the `_log` guard keeps the logger alive until `main`
returns. What each pipeline adds — decoders, queues, GPUs, seeking — is in
[how media-pp works](guide/concepts.md).

## 6. Read a file

Now use a local video file of your own. Nothing is downloaded and no media
is included in the repository. Replace **your application's `src/main.rs`**
with the following program; the Cargo dependency from step 4 is unchanged.
It counts encoded video packets, not decoded pictures.

```rust,no_run
use media_pp::{
    bus::BusEvent,
    elements::{FileDemuxer, PacketCounter},
    ffmpeg,
    pipeline::Pipeline,
};

fn main() -> media_pp::Result<()> {
    let _log = media_pp::log::init("hello", "logs", media_pp::log::Level::Trace, 7)?;
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: cargo run -- <video-path>");
        std::process::exit(1);
    };
    // The source: a file, split into its streams.
    let (source, _) = FileDemuxer::open("file", &path)?;
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
        match event {
            BusEvent::Finished => {
                pipeline.stop();
                break;
            }
            BusEvent::Error { error, .. } => {
                pipeline.stop();
                return Err(error);
            }
            _ => {}
        }
    }
    println!("packets: {}", packets.get());
    Ok(())
}
```

Run **from your application directory**, replacing the example path with
your video's path. Quote paths containing spaces. Cargo's `--` separates
its options from the path passed to your program.

```powershell
# Windows
cargo run -- "C:\path\to\video.mp4"
```

```sh
# Linux, macOS
cargo run -- "/path/to/video.mp4"
```

Successful output looks like `packets: 90`; the count depends on the file.
A missing file reports its path and an open error. An audio-only file has
no video stream to count. Running without a path prints the usage and exits
with a failure code.

## 7. Playing a file

`Player` builds the whole pipeline: decoding on the GPU where it can, a
window, and the default audio output. It needs both a window backend and
an audio renderer.

**Linux only:** before building this program, install PipeWire 0.3.50 or
newer development files (`libpipewire-0.3-dev` on Debian/Ubuntu), and enable
Vulkan in FFmpeg. From the **media-pp repository root** run:

```sh
cd ~/media-pp
scripts/setup-ffmpeg.sh --vulkan
. ~/.local/share/media-pp/ffmpeg/env.sh
cd ~/media-pp-hello
```

Use your actual paths if you chose other directories. See
[building on Linux](building/linux.md) for the other platform prerequisites.

In **your application's `Cargo.toml`**, replace the `media-pp = "0.4"`
dependency from step 4 with these target-specific sections. Keep the
generated `[package]` section unchanged:

```toml
[target.'cfg(target_os = "windows")'.dependencies]
media-pp = { version = "0.4", features = ["d3d11", "wasapi-renderer"] }

[target.'cfg(target_os = "linux")'.dependencies]
media-pp = { version = "0.4", features = ["vulkan", "pipewire-audio-renderer"] }

[target.'cfg(target_os = "macos")'.dependencies]
media-pp = { version = "0.4", features = ["metal", "coreaudio-renderer"] }
```

Then replace **your application's `src/main.rs`** with this complete
program. On macOS every window belongs to the main thread, so `main` runs
the player inside `media_pp::elements::run_with_windows`:

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
    let _log = media_pp::log::init("hello", "logs", media_pp::log::Level::Trace, 7)?;
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: cargo run -- <video-path>");
        std::process::exit(1);
    };
    let player = Player::open(&path, PlayerOptions::default())?;
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

Run **from your application directory** with the same command and video
path as step 6. A window should open, play the video with sound if the file
has it, and exit at the end; Escape or closing the window also ends it.

Alternatively, from the **media-pp repository root** the existing
[`player`](../examples/render/player) example is ready to run:
`cargo run -p player -- <video-path>`. Its own Cargo configuration already
selects the platform features.

## 8. Where to go next

| To | Read |
|---|---|
| Understand pipelines, threads, errors, ending and seeking | [How media-pp works](guide/concepts.md) |
| Pick a GPU backend | [Choosing a backend](guide/choosing-a-backend.md) |
| Do a particular job — record, stream, composite, detect | [Recipes](guide/recipes.md) |
| Find an element | [Elements](elements.md) |
| Check a feature's build and runtime requirements | [Features](features.md) |
| Run an example | [Examples](examples.md) |
| Fix a build or start-up problem | [Troubleshooting](guide/troubleshooting.md) |
| Look up a type | API documentation: [Windows], [macOS], [Linux]; [docs.rs] for what every platform shares |
| Upgrade between versions | [`CHANGELOG.md`](../CHANGELOG.md), which says what to write instead |

[docs.rs]: https://docs.rs/media-pp
[Windows]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS]: https://hash1018.github.io/media-pp/macos/media_pp/
[Linux]: https://hash1018.github.io/media-pp/linux/media_pp/
