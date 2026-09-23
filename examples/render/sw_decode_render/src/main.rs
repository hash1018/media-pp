//! Demux -> SwDecoder -> Queue -> Pacer -> SwScaler -> GPU upload -> Renderer:
//! decodes a video file in system memory and presents it in a native window at
//! real playback speed. Windows uses D3D12; Linux uses CUDA/Vulkan.
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
        elements::{D3d12Upload, FileDemuxer, Pacer, SwDecoder, SwScaler},
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

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
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, _) = FileDemuxer::open("demux", path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        let gpu = D3d12GpuContext::new()?;

        let (pipeline, ()) = Pipeline::new("sw-decode-render", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params)?;
            let pacer = Pacer::new("pacer");
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)?;
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
            let upload = D3d12Upload::new("upload", gpu.device())?;
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", 32)
                .pipe(pacer)
                .pipe(scaler)
                .pipe(upload)
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

#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{
            CudaDevice, CudaFrameFormat, CudaUpload, FileDemuxer, Pacer, SwDecoder, SwScaler,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, VulkanGpuContext, WindowTarget};

    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: sw_decode_render <video.mp4>");
            std::process::exit(1);
        };
        render_common::run_window(
            "media-pp sw_decode_render",
            1280,
            720,
            move |target, shutdown| play(&path, target, &shutdown),
        );
    }

    fn play(path: &str, target: WindowTarget, shutdown: &Shutdown) -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (source, _) = FileDemuxer::open("demux", path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();
        let cuda = CudaDevice::new()?;
        let gpu = VulkanGpuContext::new(target.display)?;

        let (pipeline, ()) = Pipeline::new("sw-decode-render", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params)?;
            let pacer = Pacer::new("pacer");
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                target.width,
                target.height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new("upload", &cuda, CudaFrameFormat::Nv12);
            let renderer = render_common::cuda_window_renderer(
                "renderer",
                &gpu,
                &cuda,
                target.display,
                target.window,
                target.width,
                target.height,
            )?;
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", 32)
                .pipe(pacer)
                .pipe(scaler)
                .pipe(upload)
                .to(renderer)?;
            ctx.attach(source, video.index, branch)?;
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
