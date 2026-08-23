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
        Error,
        bus::BusEvent,
        elements::{D3d12Upload, FileDemuxer, Pacer, SwDecoder, SwScaler},
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    /// Demux -> SwDecoder -> Queue -> Pacer -> Renderer: decodes a video file
    /// and presents it in a native window at real playback speed, via
    /// `render_common`'s own `D3d12WindowRenderer` (wrapped as a
    /// `D3d12Renderer`).
    ///
    ///     cargo run -p sw_decode_render -- path/to/video.mp4
    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: sw_decode_render <video.mp4>");
            std::process::exit(1);
        };

        render_common::run_window(
            "media-pp sw_decode_render",
            1280,
            720,
            move |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("sw_decode_render example only supports Windows");
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
            .find(|s| s.kind == media::Type::Video)
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let params = source
            .stream_parameters(video.index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;
        let time_base = source
            .stream_time_base(video.index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;

        let gpu = D3d12GpuContext::new().map_err(|e| Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("sw-decode-render", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");
            // `D3d12Renderer` draws from a device resource only, so the
            // decoder's system-memory frames are converted to the NV12
            // layout `D3d12Upload` writes and uploaded here. Without this
            // pair the branch is refused as it is built, naming the
            // decoder — see `media_pp::contract`.
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = D3d12Upload::new("upload", gpu.device(), width, height)
                .expect("failed to create the D3D12 upload");
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", 32)
                .pipe(pacer)
                .pipe(scaler)
                .pipe(upload)
                .to(Box::new(renderer))?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        pipeline.run()?;

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
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }
}
