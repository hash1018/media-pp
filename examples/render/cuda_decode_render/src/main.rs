//! FileDemuxer -> CudaDecoder -> Queue -> Pacer -> VulkanWindowRenderer:
//! decodes on the GPU with NVDEC and shows the frames in a window at real
//! playback speed, without the decoded pixels ever leaving the GPU — the
//! renderer copies each one device to device into memory Vulkan draws from.
//! The Linux sibling of `d3d11_decode_render`.
//!
//! The window is the renderer's own: `VulkanWindowRenderer::open` opens it
//! on a thread of its own, the way a GStreamer video sink does, and reports
//! what happens to it — Space pauses and resumes, F or a double click fills
//! the screen and puts it back, Escape or closing the window stops — and its
//! `WindowControl` changes it: the title shows where playback is. It is an X11 window, so on a Wayland desktop it is an
//! XWayland one, and this program needs no `winit` or event loop of its own.
//!
//!     cargo run -p cuda_decode_render -- path/to/video.mp4

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("{} example only supports Linux", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "linux")]
fn main() -> media_pp::Result<()> {
    linux_example::run()
}

#[cfg(target_os = "linux")]
mod linux_example {
    use std::{sync::mpsc::RecvTimeoutError, time::Duration};

    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{
            CudaDecoder, CudaDevice, FileDemuxer, Key, MouseButton, Pacer, VulkanGpu,
            VulkanWindowRenderer, WindowEvent, WindowOptions,
        },
        pipeline::Pipeline,
    };

    const TITLE: &str = "media-pp cuda_decode_render";

    /// Decoded frames queued ahead of the pacer.
    const FRAMES: usize = 12;

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Debug,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: cuda_decode_render <video.mp4>");
            std::process::exit(1);
        };

        let (source, _) = FileDemuxer::open("demux", &path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        // The CUDA device first, and the Vulkan device on the same GPU: made
        // once, before anything decodes.
        let cuda = CudaDevice::new()?;
        let gpu = VulkanGpu::for_cuda(&cuda)?;
        let (screen, window) = VulkanWindowRenderer::open(
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

        let (pipeline, ()) = Pipeline::new("cuda-decode-render", source, |source, ctx| {
            // The decoder's surface pool does not grow, so its budget covers
            // the `"frames"` queue and a few more — the frame the pacer is
            // waiting on, the one being copied out.
            let decoder = CudaDecoder::new("decoder", params, &cuda, (FRAMES + 8) as i32)?;
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", FRAMES)
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
                Some(event) => println!("{event:?}"),
                None => {}
            }
        }
        pipeline.stop();
        Ok(())
    }
}
