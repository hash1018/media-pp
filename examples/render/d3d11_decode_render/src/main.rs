//! FileDemuxer -> D3d11Decoder -> Queue -> Pacer -> D3d11WindowRenderer:
//! decodes on the GPU via D3D11VA hardware acceleration and shows the frames
//! in a window at real playback speed, without ever copying the decoded
//! pixels back to system memory — the renderer draws straight from the
//! decoder's own D3D11 texture. The D3D11 sibling of `hw_decode_render`,
//! which decodes through `VideoDecodeBin` onto D3D12 with a software fallback.
//!
//! The window is the renderer's own: `D3d11WindowRenderer::open` opens it on
//! a thread of its own, the way a GStreamer video sink does, and reports what
//! happens to it — Space pauses and resumes, F or a double click fills the
//! screen and puts it back, Escape or closing the window stops, as do the
//! end of the file and an element's failure — and its `WindowControl`
//! changes it: the title shows where playback is. The one device every
//! element shares is a `D3d11Gpu`.
//!
//!     cargo run -p d3d11_decode_render -- path/to/video.mp4

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{sync::mpsc::RecvTimeoutError, time::Duration};

    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{
            D3d11Decoder, D3d11Gpu, D3d11WindowRenderer, FileDemuxer, Key, MouseButton, Pacer,
            WindowEvent, WindowOptions,
        },
        pipeline::Pipeline,
    };

    const TITLE: &str = "media-pp d3d11_decode_render";

    /// Decoded frames queued ahead of the pacer.
    const FRAMES: usize = 12;

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: d3d11_decode_render <video.mp4>");
            std::process::exit(1);
        };

        let (source, _) = FileDemuxer::open("demux", &path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        let gpu = D3d11Gpu::new()?;
        let (screen, window) = D3d11WindowRenderer::open(
            "screen",
            &gpu,
            WindowOptions {
                title: TITLE.into(),
                ..WindowOptions::default()
            },
        )?;
        // Taken before the renderer goes into the pipeline.
        let control = screen
            .window_control()
            .expect("a window the renderer opened is its own to change");

        let (pipeline, ()) = Pipeline::new("d3d11-decode-render", source, |source, ctx| {
            // On the renderer's device — required for the zero-copy path to
            // be valid at all (see D3d11Decoder::new). The decoder's
            // downstream-frame budget covers the `"frames"` queue and a few
            // more — the frame the pacer is waiting on, the one on screen —
            // since the D3D11VA pool does not grow. Its accurate-seek
            // candidate surface is reserved internally, so it is not
            // included in this value.
            let decoder = D3d11Decoder::new("decoder", params, &gpu, (FRAMES + 8) as i32)?;
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", FRAMES) // the pacer sleeps on its own thread; let decode run ahead into this
                .pipe(Pacer::new("pacer"))
                .to(screen)?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        pipeline.run()?;

        // Two things to watch: the pipeline's bus — the end of the stream is
        // `Finished`, and an error does not end a pipeline on its own — and
        // the window, whose keys and closing are this program's to act on.
        let mut paused = false;
        let mut shown = None;
        loop {
            // The title follows playback, once a second is enough to see.
            let seconds = pipeline.position().map(|position| position.as_secs());
            if seconds != shown {
                shown = seconds;
                if let Some(seconds) = seconds {
                    let _ = control.set_title(&format!(
                        "{TITLE} — {}:{:02}",
                        seconds / 60,
                        seconds % 60
                    ));
                }
            }
            match pipeline.bus().recv_timeout(Duration::from_millis(20)) {
                Ok(event) => {
                    println!("{event}");
                    if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {}
            }
            match window.try_recv() {
                Some(WindowEvent::Closed | WindowEvent::Key(Key::Escape)) => break,
                Some(WindowEvent::Key(Key::Space)) => {
                    paused = !paused;
                    if paused {
                        pipeline.pause();
                    } else {
                        pipeline.resume();
                    }
                }
                Some(
                    WindowEvent::Key(Key::Char('f'))
                    | WindowEvent::DoubleClick {
                        button: MouseButton::Left,
                        ..
                    },
                ) => {
                    let _ = control.set_fullscreen(!control.is_fullscreen());
                }
                _ => {}
            }
        }
        pipeline.stop();
        Ok(())
    }
}
