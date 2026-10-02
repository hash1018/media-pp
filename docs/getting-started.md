# Getting started

```toml
[dependencies]
media-pp = "0.3"
```

FFmpeg 8.0 or newer development libraries must be installed — see
[building](building/) for each platform. `ffmpeg-next` is re-exported as
`media_pp::ffmpeg`; use that rather than depending on it separately. The
library has no default features: what each one adds is in
[features.md](features.md).

## A file's pipeline

A file's pipeline says when everything read has arrived —
`BusEvent::Finished` — and waits there, where a seek could still take it
back, until it is stopped:

```rust,no_run
use media_pp::{
    bus::BusEvent,
    elements::{FileDemuxer, PacketCounter},
    ffmpeg,
    pipeline::Pipeline,
};

fn main() -> media_pp::Result<()> {
    let (source, _) = FileDemuxer::open("file", "video.mp4")?;
    let video = source.best(ffmpeg::media::Type::Video)?;
    let (counter, packets) = PacketCounter::new("counter");
    let (pipeline, ()) = Pipeline::new("count", source, |source, ctx| {
        ctx.attach(source, video.index, ctx.branch().to(counter)?)?;
        Ok(())
    })?;
    pipeline.run()?;
    for event in pipeline.bus().iter() {
        // An error does not end a pipeline by itself either.
        if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }
    println!("packets: {}", packets.get());
    Ok(())
}
```

## Playing a file

`Player` builds the whole pipeline — decode on the GPU where it can, a
window, the default audio output — and reports what happens to it. It needs
`d3d11` and `wasapi-renderer` on Windows, `vulkan` and
`pipewire-audio-renderer` on Linux, and `metal` and `coreaudio-renderer` on
macOS, where every window is the main thread's, so `main` runs this inside
`media_pp::elements::run_with_windows`:

```rust,no_run
use media_pp::player::{Player, PlayerEvent, PlayerOptions};

fn main() -> media_pp::Result<()> {
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

## Where to read on

How a pipeline runs — buffers, threads, the end of a stream, seeking,
changing a running graph, what a link refuses before it runs, and how to
write an element of your own — is the crate documentation's first page;
each type's page says what it accepts, owns and how it fails. Both are on
[docs.rs] for the backend-independent and Linux API, and in the
[Windows API documentation] and the [macOS API documentation] for
everything on those platforms.

What is in the library is in [elements.md](elements.md), the runnable
examples in [examples.md](examples.md), and what changed between versions,
and what to write instead, in [`CHANGELOG.md`](../CHANGELOG.md).

[docs.rs]: https://docs.rs/media-pp
[Windows API documentation]: https://hash1018.github.io/media-pp/windows/media_pp/
[macOS API documentation]: https://hash1018.github.io/media-pp/macos/media_pp/
