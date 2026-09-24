//! Demux -> VideoDecodeBin -> Queue -> Pacer -> Renderer: decodes on the GPU
//! where it can, and in software onto the same device where it cannot, and
//! presents the frames at real playback speed. Windows decodes onto D3D12
//! (D3D12VA, or `SwDecoder` and an upload) into `D3d12WindowRenderer`'s own
//! window; Linux onto CUDA (NVDEC, or
//! `SwDecoder` and an upload) into `VulkanWindowRenderer`'s own window, and
//! puts a `CudaConverter` before the renderer where
//! `contract::check_elements` says the bin's output does not fit the
//! renderer's input — the renderer draws NV12 and BGRA, and the bin may hand
//! on P010, or a layout it cannot name where the stream does not say. The
//! check is asked of the two before they are linked, rather than the example
//! knowing what each one takes.
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
        elements::{
            D3d12Gpu, D3d12WindowRenderer, DecodeTarget, FileDemuxer, Pacer, VideoDecodeBin,
            WindowOptions,
        },
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: hw_decode_render <video.mp4>");
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

        let gpu = D3d12Gpu::new()?;
        let (renderer, window) = D3d12WindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp hw_decode_render".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);

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

        let (pipeline, ()) = Pipeline::new("hw-decode-render", source, |source, ctx| {
            let pacer = Pacer::new("pacer");
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
        contract::check_elements,
        elements::{
            CudaConverter, CudaDevice, CudaFrameFormat, DecodeTarget, FileDemuxer, Pacer,
            VideoDecodeBin, VulkanGpu, VulkanWindowRenderer, WindowOptions,
        },
        pipeline::Pipeline,
    };

    const VIDEO_QUEUE_DEPTH: usize = 8;

    pub(super) fn run() -> media_pp::Result<()> {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: hw_decode_render <video.mp4>");
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

        // The CUDA device first, and the Vulkan device on the same GPU, which
        // is the one a CUDA frame can be copied to.
        let cuda = CudaDevice::new()?;
        let gpu = VulkanGpu::for_cuda(&cuda)?;
        let (renderer, window) = VulkanWindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp hw_decode_render".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);

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
        // Asked before linking rather than known: the renderer draws NV12
        // and BGRA, and the bin may hand on what it does not — P010, or a
        // surface in another layout, where the stream does not say what it
        // decodes to. Where the two do not fit, this crate's own kernel
        // brings it to NV12.
        let fits = check_elements(&mut decoder, &renderer);
        let to_nv12 = if fits.is_refused() {
            println!("decoder and renderer: {fits}; converting to NV12");
            Some(CudaConverter::new("to-nv12", &cuda, CudaFrameFormat::Nv12)?)
        } else {
            None
        };

        let (pipeline, ()) = Pipeline::new("hw-decode-render", source, |source, ctx| {
            let pacer = Pacer::new("pacer");
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
/// frozen last frame after a renderer failure. It stops on `Finished`, which
/// the pipeline posts once every terminal has ended, rather than on the
/// first `Eos` any element posts.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn drain_bus(pipeline: &media_pp::pipeline::Pipeline) {
    use media_pp::bus::BusEvent;

    for event in pipeline.bus().iter() {
        println!("{event}");
        if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }
}
