//! Demux -> D3d12Decoder -> Queue -> Pacer -> Renderer: decodes on the
//! GPU via D3D12VA hardware acceleration and presents the frames in a
//! native window at real playback speed, without ever copying the
//! decoded pixels back to system memory — `D3d12Renderer` draws straight
//! from the decoder's own D3D12 texture. Compare against
//! `sw_decode_render`, which uses `SwDecoder` (CPU decode) and a
//! CPU-upload submit path instead.
//!
//!     cargo run -p hw_decode_render -- path/to/video.mp4

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
        elements::{D3d12Decoder, FileDemuxer, Pacer},
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: hw_decode_render <video.mp4>");
            std::process::exit(1);
        };

        render_common::run_window(
            "media-pp hw_decode_render",
            1280,
            720,
            move |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("hw_decode_render example only supports Windows");
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

        let pipeline = Pipeline::new("hw-decode-render", source, |source, ctx| {
            // Same device the renderer draws with — required for the
            // zero-copy path to be valid at all (see D3d12Decoder::new).
            let decoder = D3d12Decoder::new("decoder", params, gpu.device())
                .expect("failed to open D3D12VA decoder");
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                .to(Box::new(renderer))?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — any failure (including the source's own) shows up as a
        // `BusEvent::Error` here instead of through a returned `Result`.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        pipeline.run()?;

        // Errors no longer end the pipeline on their own (see `BusEvent`'s
        // docs) — watch for one here and `stop()`, or this window would just
        // sit open (showing a frozen last frame) instead of closing after a
        // renderer failure. Single video stream, so `Eos` calling `stop()` is
        // a harmless no-op too.
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
