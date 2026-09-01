//! Starts with video only, then lets the terminal attach/detach a decoded
//! audio branch at runtime. `VideoSynchronizer` uses wall time while audio is
//! absent and automatically hands scheduling to the audio renderer's played-
//! sample position while the branch is attached.
//!
//!     cargo run -p av_playback -- path/to/video-with-audio.mp4
//!     audio on
//!     audio off
//!     pause
//!     resume
//!     seek 30
//!     seek 1:15
//!     keyseek 30
//!     q
//!
//! Both platforms hold the audio pad open with a dynamic `Tee` and run the same
//! audio branch — `SwDecoder -> AudioResampler -> Queue -> renderer`
//! (`WasapiRenderer` on Windows, `PipeWireAudioRenderer` on Linux). The video
//! branches differ in more than backend types, because only one of them decodes
//! on the GPU:
//!
//!     Windows: FileDemuxer -> SwDecoder -> Queue -> VideoSynchronizer
//!              -> SwScaler(NV12) -> D3d12Upload -> D3d12Renderer
//!     Linux:   FileDemuxer -> CudaDecoder -> Queue -> VideoSynchronizer
//!              -> CudaRenderer
//!
//! The Linux branch is the one that never brings decoded pixels to the CPU:
//! NVDEC keeps every frame in CUDA memory and the renderer copies it straight
//! into Vulkan-owned memory. Windows decodes in system memory, so it has to
//! convert and upload before the renderer can take it.

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("{} supports Windows and Linux only", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: av_playback <video-with-audio.mp4>");
        std::process::exit(2);
    };
    render_common::run_window(
        "media-pp A/V playback",
        1280,
        720,
        move |target, shutdown| windows_example::play(path, target, shutdown),
    );
}

#[cfg(target_os = "linux")]
fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: av_playback <video-with-audio.mp4>");
        std::process::exit(2);
    };
    render_common::run_window(
        "media-pp A/V playback",
        1280,
        720,
        move |target, shutdown| linux_example::play(path, target, shutdown),
    );
}

mod shell;

/// The parts of `play` that are the same on every backend: opening the file
/// and locating the two streams it needs.
#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common {
    use media_pp::ffmpeg::{Rational, codec::Parameters, media};
    use media_pp::{Error, elements::FileDemuxer};

    pub struct Streams {
        pub video_index: usize,
        pub video_params: Parameters,
        pub video_time_base: Rational,
        pub audio_index: usize,
        pub audio_params: Parameters,
        pub audio_time_base: Rational,
    }

    pub fn open(path: &str) -> media_pp::Result<(FileDemuxer, Streams)> {
        let (source, streams) = FileDemuxer::open("demux", path)?;
        let video = streams
            .iter()
            .find(|stream| stream.kind == media::Type::Video)
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let audio = streams
            .iter()
            .find(|stream| stream.kind == media::Type::Audio)
            .ok_or_else(|| Error::Other("no audio stream in file".into()))?;
        let gone = || Error::Other("stream disappeared".into());
        let streams = Streams {
            video_index: video.index,
            video_params: source.stream_parameters(video.index).ok_or_else(gone)?,
            video_time_base: source.stream_time_base(video.index).ok_or_else(gone)?,
            audio_index: audio.index,
            audio_params: source.stream_parameters(audio.index).ok_or_else(gone)?,
            audio_time_base: source.stream_time_base(audio.index).ok_or_else(gone)?,
        };
        Ok((source, streams))
    }
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::sync::Arc;

    use media_pp::{
        Error,
        bus::BusEvent,
        elements::{
            AudioResampler, D3d12Upload, SwDecoder, SwScaler, TeeBuilder, VideoSynchronizer,
            WasapiRenderer, WasapiRendererOptions,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown, WindowTarget};
    use winit::raw_window_handle::RawWindowHandle;

    use crate::common;

    pub fn play(
        path: String,
        target: WindowTarget,
        shutdown: Arc<Shutdown>,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (source, streams) = common::open(&path)?;

        let RawWindowHandle::Win32(handle) = target.window else {
            return Err(Error::Other("not a Win32 window".into()));
        };
        let hwnd = handle.hwnd.get();
        let gpu = D3d12GpuContext::new().map_err(|error| Error::Other(format!("{error:?}")))?;

        let mut audio_tee_handle = None;
        let pipeline = Pipeline::new("av-playback", source, |source, context| {
            let video_branch = context
                .branch()
                .pipe(SwDecoder::new(
                    "video-decoder",
                    streams.video_params.clone(),
                )?)
                .queue("video-frames", 32)
                .pipe(VideoSynchronizer::new(
                    "video-sync",
                    streams.video_time_base,
                )?)
                // After the synchronizer, not before: a frame it drops for
                // being late never pays for the conversion or the upload.
                // `D3d12Renderer` draws from a device resource only, so this
                // pair is what carries CPU-decoded frames to the GPU.
                .pipe(SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    target.width,
                    target.height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                ))
                .pipe(
                    D3d12Upload::new("video-upload", gpu.device(), target.width, target.height)
                        .map_err(|error| Error::Other(error.to_string()))?,
                )
                .to(Box::new(
                    render_common::d3d12_window_renderer(
                        "video-renderer",
                        &gpu,
                        hwnd,
                        target.width,
                        target.height,
                    )
                    .map_err(|error| Error::Other(format!("{error:?}")))?,
                ))?;
            context.attach(source, streams.video_index, video_branch)?;

            // Keep a stable insertion point on the demuxer's audio pad. With
            // no branches attached the Tee cheaply drops packets, so playback
            // starts video-only without decoding audio.
            let (audio_tee, handle) =
                TeeBuilder::new("audio-tee", context.clone()).build_dynamic()?;
            context.attach(source, streams.audio_index, audio_tee)?;
            audio_tee_handle = Some(handle);
            Ok(())
        })?;
        let audio_tee_handle =
            audio_tee_handle.ok_or_else(|| Error::Other("audio Tee was not initialized".into()))?;

        // Published before `run`, so a close that arrives from here on finds
        // the pipeline to stop. `true` means one already did.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;
        {
            let pipeline = pipeline.clone();
            let tee = audio_tee_handle.clone();
            let params = streams.audio_params.clone();
            let time_base = streams.audio_time_base;
            std::thread::spawn(move || {
                let attach_tee = tee.clone();
                crate::shell::read_commands(pipeline.clone(), tee, "WASAPI", move || {
                    attach_audio(&attach_tee, &params, time_base)
                });
            });
        }

        crate::drain_bus(&pipeline);
        Ok(())
    }

    fn attach_audio(
        audio_tee: &media_pp::elements::TeeHandle,
        audio_params: &media_pp::ffmpeg::codec::Parameters,
        audio_time_base: media_pp::ffmpeg::Rational,
    ) -> media_pp::Result<media_pp::graph::BranchId> {
        let device = WasapiRenderer::list_devices()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .find(|device| device.is_default)
            .ok_or_else(|| Error::Other("no default WASAPI render endpoint".into()))?;
        let device_name = device.name.clone();
        let (audio_renderer, output_format) =
            WasapiRenderer::open("speakers", WasapiRendererOptions { device })?;

        let branch = audio_tee
            .branch()
            .ok_or_else(|| Error::Other("audio Tee is no longer available".into()))?
            .pipe(SwDecoder::new("audio-decoder", audio_params.clone())?)
            .pipe(AudioResampler::new(
                "audio-resampler",
                output_format,
                audio_time_base,
            )?)
            .queue("audio-output", 8)
            .to(Box::new(audio_renderer))?;
        let branch_id = audio_tee.attach(branch)?;
        println!("audio on: {device_name}; video is synchronized to played audio");
        Ok(branch_id)
    }

    #[allow(unused_imports)]
    use BusEvent as _;
}

#[cfg(target_os = "linux")]
mod linux_example {
    use std::sync::Arc;

    use media_pp::{
        Error,
        elements::{
            AudioResampler, CudaDecoder, CudaDevice, PipeWireAudioRenderer,
            PipeWireAudioRendererOptions, SwDecoder, TeeBuilder, TeeHandle, VideoSynchronizer,
        },
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, VulkanGpuContext, WindowTarget};

    use crate::common;

    /// Matches the `video-frames` queue below. NVDEC's surface pool is fixed
    /// at open time. `CudaDecoder` reserves its accurate-seek candidate
    /// internally, so construction supplies only this downstream queue depth.
    /// NVDEC also caps the total pool at 32 surfaces, which is why this is far
    /// shallower than the Windows branch's software decoder can afford; see
    /// `CudaDecoder::new`'s docs. Eight frames are about 266 ms at 30 fps.
    const VIDEO_QUEUE_DEPTH: usize = 8;

    pub fn play(
        path: String,
        target: WindowTarget,
        shutdown: Arc<Shutdown>,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (source, streams) = common::open(&path)?;

        // One CUDA context for the whole stack: the decoder allocates frames
        // on it and the renderer imports its Vulkan memory into it. The
        // renderer element rejects any frame from a different one.
        let cuda = CudaDevice::new().map_err(|error| Error::Other(error.to_string()))?;
        let gpu = VulkanGpuContext::new(target.display).map_err(Error::Other)?;

        let mut audio_tee_handle = None;
        let pipeline = Pipeline::new("av-playback", source, |source, context| {
            let video_branch = context
                .branch()
                .pipe(
                    CudaDecoder::new(
                        "video-decoder",
                        streams.video_params.clone(),
                        &cuda,
                        VIDEO_QUEUE_DEPTH as i32,
                    )
                    .map_err(|error| Error::Other(error.to_string()))?,
                )
                .queue("video-frames", VIDEO_QUEUE_DEPTH)
                .pipe(VideoSynchronizer::new(
                    "video-sync",
                    streams.video_time_base,
                )?)
                .to(Box::new(
                    render_common::cuda_window_renderer(
                        "video-renderer",
                        &gpu,
                        &cuda,
                        target.display,
                        target.window,
                        target.width,
                        target.height,
                    )
                    .map_err(Error::Other)?,
                ))?;
            context.attach(source, streams.video_index, video_branch)?;

            let (audio_tee, handle) =
                TeeBuilder::new("audio-tee", context.clone()).build_dynamic()?;
            context.attach(source, streams.audio_index, audio_tee)?;
            audio_tee_handle = Some(handle);
            Ok(())
        })?;
        let audio_tee_handle =
            audio_tee_handle.ok_or_else(|| Error::Other("audio Tee was not initialized".into()))?;

        // Published before `run`, so a close that arrives from here on finds
        // the pipeline to stop. `true` means one already did.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;
        {
            let pipeline = pipeline.clone();
            let tee = audio_tee_handle.clone();
            let params = streams.audio_params.clone();
            let time_base = streams.audio_time_base;
            std::thread::spawn(move || {
                let attach_tee = tee.clone();
                crate::shell::read_commands(pipeline.clone(), tee, "PipeWire", move || {
                    attach_audio(&attach_tee, &params, time_base)
                });
            });
        }

        crate::drain_bus(&pipeline);
        Ok(())
    }

    fn attach_audio(
        audio_tee: &TeeHandle,
        audio_params: &media_pp::ffmpeg::codec::Parameters,
        audio_time_base: media_pp::ffmpeg::Rational,
    ) -> media_pp::Result<media_pp::graph::BranchId> {
        let device = PipeWireAudioRenderer::list_devices()
            .map_err(|error| Error::Other(error.to_string()))?
            .into_iter()
            .find(|device| device.is_default)
            .ok_or_else(|| Error::Other("no default PipeWire playback device".into()))?;
        let device_name = device.name.clone();
        let (audio_renderer, output_format) =
            PipeWireAudioRenderer::open("speakers", PipeWireAudioRendererOptions { device })
                .map_err(|error| Error::Other(error.to_string()))?;

        let branch = audio_tee
            .branch()
            .ok_or_else(|| Error::Other("audio Tee is no longer available".into()))?
            .pipe(SwDecoder::new("audio-decoder", audio_params.clone())?)
            .pipe(AudioResampler::new(
                "audio-resampler",
                output_format,
                audio_time_base,
            )?)
            .queue("audio-output", 8)
            .to(Box::new(audio_renderer))?;
        let branch_id = audio_tee.attach(branch)?;
        println!("audio on: {device_name}; video is synchronized to played audio");
        Ok(branch_id)
    }
}

/// The bus loop, identical on both backends.
#[cfg(any(target_os = "windows", target_os = "linux"))]
fn drain_bus(pipeline: &media_pp::pipeline::Pipeline) {
    use media_pp::bus::BusEvent;

    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer"),
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
    }
}
