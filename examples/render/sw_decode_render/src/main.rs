//! Demux -> SwDecoder -> Queue -> Pacer -> Renderer: decodes a video file in
//! system memory and presents it in a native window at real playback speed.
//! The renderer — `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on
//! Linux — uploads the decoded frames itself, in a window of its own, through
//! a `SwScaler` only for a stream it cannot draw as it comes.
//!
//!     cargo run -p sw_decode_render -- path/to/video.mp4

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("{} supports Windows and Linux only", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{D3d12Gpu, D3d12WindowRenderer, FileDemuxer, Pacer, SwDecoder, WindowOptions},
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

        // Decoded frames stay in system memory; the renderer uploads them.
        let gpu = D3d12Gpu::new()?;
        let (renderer, window) = D3d12WindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp sw_decode_render".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);
        let to_drawable = render_common::to_drawable(&params, &renderer)?;

        let (pipeline, ()) = Pipeline::new("sw-decode-render", source, |source, ctx| {
            let mut branch = ctx
                .branch()
                .pipe(SwDecoder::new("decoder", params)?)
                .queue("frames", 32)
                .pipe(Pacer::new("pacer"));
            if let Some(to_drawable) = to_drawable {
                branch = branch.pipe(to_drawable);
            }
            ctx.attach(source, video.index, branch.to(renderer)?)?;
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

#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{FileDemuxer, Pacer, SwDecoder, VulkanGpu, VulkanWindowRenderer, WindowOptions},
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

        // Decoded frames stay in system memory; the renderer uploads them.
        let gpu = VulkanGpu::new()?;
        let (renderer, window) = VulkanWindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp sw_decode_render".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);
        let to_drawable = render_common::to_drawable(&params, &renderer)?;

        let (pipeline, ()) = Pipeline::new("sw-decode-render", source, |source, ctx| {
            let mut branch = ctx
                .branch()
                .pipe(SwDecoder::new("decoder", params)?)
                .queue("frames", 32)
                .pipe(Pacer::new("pacer"));
            if let Some(to_drawable) = to_drawable {
                branch = branch.pipe(to_drawable);
            }
            ctx.attach(source, video.index, branch.to(renderer)?)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;
        drain_bus(&pipeline);
        Ok(())
    }

    fn drain_bus(pipeline: &Pipeline) {
        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
    }
}
