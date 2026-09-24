//! FileDemuxer -> D3d11Decoder -> D3d11Scaler (960x540) -> Queue ->
//! Pacer -> D3d11WindowRenderer: decodes and resizes video entirely on one
//! shared D3D11 device, then presents the fixed-size NV12 output in the
//! renderer's own window at real playback speed. Decoded array-texture
//! slices go directly through the D3D11 video processor, and neither
//! scaling nor rendering maps the pixels to system memory.
//!
//! The scaler sits before the queue deliberately. Once one synchronous
//! scale finishes, its decoded input surface can return to FFmpeg's fixed
//! D3D11VA pool; the queue retains the scaler's independent output
//! textures instead.
//!
//!     cargo run -p d3d11_scale_render -- path/to/video.mp4

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
    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{
            D3d11Decoder, D3d11Gpu, D3d11Scaler, D3d11ScalerFormat, D3d11WindowRenderer,
            FileDemuxer, Pacer, WindowOptions,
        },
        pipeline::Pipeline,
    };

    const OUTPUT_WIDTH: u32 = 960;
    const OUTPUT_HEIGHT: u32 = 540;

    pub(super) fn run() -> media_pp::Result<()> {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: d3d11_scale_render <video.mp4>");
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

        let gpu = D3d11Gpu::new()?;
        let (renderer, window) = D3d11WindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp d3d11_scale_render".into(),
                width: OUTPUT_WIDTH,
                height: OUTPUT_HEIGHT,
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);

        let (pipeline, ()) = Pipeline::new("d3d11-scale-render", source, |source, ctx| {
            // The scaler consumes each decoder surface synchronously before
            // the queue. At most that one in-flight frame can be retained if
            // the output queue is full, so one extra D3D11VA surface covers
            // the deepest downstream buffering of decoder-owned frames.
            let decoder = D3d11Decoder::new("decoder", params, &gpu, 1)?;
            let scaler = D3d11Scaler::new(
                "scaler",
                &gpu,
                // A pure resize: the decoder's NV12 surfaces stay NV12 all the
                // way to the renderer, which draws either format.
                D3d11ScalerFormat::Preserve,
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT,
            )?;
            let pacer = Pacer::new("pacer");
            let branch = ctx
                .branch()
                .pipe(decoder)
                // Scaling is synchronous and releases the decoder frame
                // before this queue retains the independent output texture.
                .pipe(scaler)
                .queue("scaled-frames", 8)
                .pipe(pacer)
                .to(renderer)?;
            ctx.attach(source, video.index, branch)?;
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
