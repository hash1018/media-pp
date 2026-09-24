//! Demux -> SwDecoder -> Queue -> Pacer -> VideoWindow: decodes a video file
//! in system memory and presents it in a native window at real playback
//! speed. `VideoWindow` is whichever window renderer the platform has —
//! `D3d11WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux — with a
//! GPU of its own; it uploads the decoded frames itself, and a `SwScaler`
//! goes in front only for a stream it cannot draw as it comes. One program
//! for both platforms, with no `#[cfg]` of its own.
//!
//!     cargo run -p sw_decode_render -- path/to/video.mp4

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("{} supports Windows and Linux only", env!("CARGO_PKG_NAME"));
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn main() -> impl std::process::Termination {
    example::run()
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod example {
    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{FileDemuxer, Pacer, SwDecoder, VideoWindow, WindowOptions},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: sw_decode_render <video.mp4>");
            std::process::exit(1);
        };
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, _) = FileDemuxer::open("demux", &path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        // Decoded frames stay in system memory; the window uploads them.
        let (screen, window) = VideoWindow::open(
            "screen",
            WindowOptions {
                title: "media-pp sw_decode_render".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);
        let to_drawable = render_common::to_drawable(&params, &screen)?;

        let (pipeline, ()) = Pipeline::new("sw-decode-render", source, |source, ctx| {
            let mut branch = ctx
                .branch()
                .pipe(SwDecoder::new("decoder", params)?)
                .queue("frames", 32)
                .pipe(Pacer::new("pacer"));
            if let Some(to_drawable) = to_drawable {
                branch = branch.pipe(to_drawable);
            }
            ctx.attach(source, video.index, branch.to(screen)?)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;

        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }
}
