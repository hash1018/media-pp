//! Demux -> VideoDecodeBin -> Queue -> Pacer -> Renderer: decodes on the GPU
//! where it can, and in software onto the same device where it cannot, and
//! presents the frames at real playback speed. Windows decodes onto D3D12
//! (D3D12VA, or `SwDecoder` and an upload); Linux onto CUDA (NVDEC, or
//! `SwDecoder` and an upload) with Vulkan presentation, and puts a
//! `CudaConverter` before the renderer where `contract::check_elements`
//! says the bin's output does not fit the renderer's input — the bin hands
//! on BGRA for a stream with alpha, an odd side, BT.2020 or HDR colour, and
//! that renderer presents NV12. The check is asked of the two before they
//! are linked, rather than the example knowing what each one takes.
//! Compare against `sw_decode_render`, which always decodes on the CPU.
//!
//! Which way the bin decodes, and why where it is software, is printed when
//! it is opened; if the GPU refuses the stream part way, the bin goes on in
//! software and the example prints that too.
//!
//!     cargo run -p hw_decode_render -- path/to/video.mp4

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example only supports Windows and Linux",
        env!("CARGO_PKG_NAME")
    );
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
        Error,
        elements::{DecodeTarget, FileDemuxer, Pacer, VideoDecodeBin},
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
        let video = source
            .best_stream(media::Type::Video)
            .and_then(|index| streams.get(index))
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let time_base = video.time_base;

        let gpu = D3d12GpuContext::new().map_err(|e| Error::Other(format!("{e:?}")))?;

        // Opened out here rather than in the builder: a stream this build
        // cannot decode at all is an error to report, and which way it
        // decodes is known before anything runs. The same device the
        // renderer draws with — required for the zero-copy path.
        let decoder = VideoDecodeBin::open(
            "decoder",
            video.parameters.clone(),
            DecodeTarget::D3d12 {
                device: gpu.device().clone(),
            },
            None,
        )?;
        println!("decoding: {:?}", decoder.path());
        let decoding = decoder.handle();

        let pipeline = Pipeline::new("hw-decode-render", source, |source, ctx| {
            let pacer = Pacer::new("pacer", time_base)?;
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                .to(renderer)?;
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
        super::drain_bus(&pipeline);
        println!("decoded: {:?}", decoding.path());
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::ffmpeg::media;
    use media_pp::{
        Error,
        contract::check_elements,
        elements::{
            CudaConverter, CudaDevice, CudaFrameFormat, DecodeTarget, FileDemuxer, Pacer,
            VideoDecodeBin,
        },
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, VulkanGpuContext, WindowTarget};

    const VIDEO_QUEUE_DEPTH: usize = 8;

    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: hw_decode_render <video.mp4>");
            std::process::exit(1);
        };

        render_common::run_window(
            "media-pp hw_decode_render",
            1280,
            720,
            move |target, shutdown| play(&path, target, &shutdown),
        );
    }

    fn play(path: &str, target: WindowTarget, shutdown: &Shutdown) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, streams) = FileDemuxer::open("demux", path)?;
        let video = source
            .best_stream(media::Type::Video)
            .and_then(|index| streams.get(index))
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let time_base = video.time_base;

        let cuda = CudaDevice::new().map_err(|error| Error::Other(error.to_string()))?;
        let gpu = VulkanGpuContext::new(target.display).map_err(Error::Other)?;

        let mut decoder = VideoDecodeBin::open(
            "decoder",
            video.parameters.clone(),
            DecodeTarget::Cuda {
                device: cuda.clone(),
                downstream_hw_frames: VIDEO_QUEUE_DEPTH as i32,
            },
            None,
        )?;
        println!("decoding: {:?}", decoder.path());
        let decoding = decoder.handle();
        let renderer = render_common::cuda_window_renderer(
            "renderer",
            &gpu,
            &cuda,
            target.display,
            target.window,
            target.width,
            target.height,
        )
        .map_err(Error::Other)?;
        // Asked before linking rather than known: the renderer presents
        // NV12, and the bin hands on BGRA for alpha, an odd side or BT.2020
        // or HDR colour. Where the two do not fit, this crate's own kernel
        // brings BGRA back to NV12 — which needs even sides, as NV12 does.
        let fits = check_elements(&mut decoder, &renderer);
        let to_nv12 = if fits.is_refused() {
            println!("decoder and renderer: {fits}; converting to NV12");
            let context = media_pp::ffmpeg::codec::context::Context::from_parameters(
                video.parameters.clone(),
            )?;
            let size = context.decoder().video()?;
            Some(
                CudaConverter::new(
                    "to-nv12",
                    &cuda,
                    CudaFrameFormat::Nv12,
                    size.width(),
                    size.height(),
                )
                .map_err(|error| Error::Other(error.to_string()))?,
            )
        } else {
            None
        };

        let pipeline = Pipeline::new("hw-decode-render", source, |source, ctx| {
            let pacer = Pacer::new("pacer", time_base)?;
            let mut chain = ctx.branch().pipe(decoder);
            if let Some(to_nv12) = to_nv12 {
                chain = chain.pipe(to_nv12);
            }
            let branch = chain
                .queue("frames", VIDEO_QUEUE_DEPTH)
                .pipe(pacer)
                .to(renderer)?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;
        super::drain_bus(&pipeline);
        println!("decoded: {:?}", decoding.path());
        Ok(())
    }
}

/// Prints what the bus says until the stream ends or fails, then stops.
///
/// Errors no longer end the pipeline on their own (see `BusEvent`'s docs) —
/// this watches for one and `stop()`s, or the window would sit open on a
/// frozen last frame after a renderer failure. Single video stream, so
/// `Eos` calling `stop()` is a harmless no-op too.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn drain_bus(pipeline: &media_pp::pipeline::Pipeline) {
    use media_pp::bus::BusEvent;

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
            // `BusEvent` is `#[non_exhaustive]`; this example only acts on
            // the events above.
            _ => {}
        }
        if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }
}
