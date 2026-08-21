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
    use ffmpeg_next::media;
    use media_pp::{
        Error,
        bus::BusEvent,
        elements::{D3d11Decoder, D3d11Scaler, D3d11ScalerFormat, FileDemuxer, Pacer},
        pipeline::Pipeline,
    };
    use render_common::{D3d11GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    const OUTPUT_WIDTH: u32 = 960;
    const OUTPUT_HEIGHT: u32 = 540;

    /// FileDemuxer -> D3d11Decoder -> D3d11Scaler (960x540) -> Queue ->
    /// Pacer -> D3d11Renderer: decodes and resizes video entirely on one
    /// shared D3D11 device, then presents the fixed-size NV12 output in a
    /// native window at real playback speed. Decoded array-texture slices go
    /// directly through the D3D11 video processor, and neither scaling nor
    /// rendering maps the pixels to system memory.
    ///
    /// The scaler sits before the queue deliberately. Once one synchronous
    /// scale finishes, its decoded input surface can return to FFmpeg's fixed
    /// D3D11VA pool; the queue retains the scaler's independent output
    /// textures instead.
    ///
    ///     cargo run -p d3d11_scale_render -- path/to/video.mp4
    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: d3d11_scale_render <video.mp4>");
            std::process::exit(1);
        };

        render_common::run_window(
            "media-pp d3d11_scale_render",
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
            move |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("d3d11_scale_render example only supports Windows");
                };
                play(
                    &path,
                    handle.hwnd.get(),
                    target.width,
                    target.height,
                    &shutdown,
                )
            },
        );
    }

    fn play(
        path: &str,
        hwnd: isize,
        width: u32,
        height: u32,
        shutdown: &Shutdown,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, streams) = FileDemuxer::open("demux", path)?;
        let video = streams
            .iter()
            .find(|stream| stream.kind == media::Type::Video)
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let params = source
            .stream_parameters(video.index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;
        let time_base = source
            .stream_time_base(video.index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;

        let gpu = D3d11GpuContext::new(None).map_err(|error| Error::Other(format!("{error:?}")))?;

        let pipeline = Pipeline::new("d3d11-scale-render", source, |source, ctx| {
            // The scaler consumes each decoder surface synchronously before
            // the queue. At most that one in-flight frame can be retained if
            // the output queue is full, so one extra D3D11VA surface covers
            // the deepest downstream buffering of decoder-owned frames.
            let decoder = D3d11Decoder::new("decoder", params, gpu.device(), 1)
                .expect("failed to open D3D11VA decoder");
            let scaler = D3d11Scaler::new(
                "scaler",
                gpu.device(),
                gpu.context(),
                // A pure resize: the decoder's NV12 surfaces stay NV12 all the
                // way to the renderer, which draws either format.
                D3d11ScalerFormat::Preserve,
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT,
            )?;
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
            let renderer =
                render_common::d3d11_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");
            let branch = ctx
                .branch()
                .pipe(decoder)
                // Scaling is synchronous and releases the decoder frame
                // before this queue retains the independent output texture.
                .pipe(scaler)
                .queue("scaled-frames", 8)
                .pipe(pacer)
                .to(Box::new(renderer))?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        pipeline.run();

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                BusEvent::Seeked {
                    name,
                    requested,
                    landed,
                    ..
                } => println!("[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"),
                _ => {}
            }
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }
}
