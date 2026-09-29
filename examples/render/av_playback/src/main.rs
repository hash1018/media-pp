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
//! Every platform holds the audio pad open with a dynamic `Tee` and runs the
//! same audio branch — `SwDecoder -> AudioResampler -> Queue -> renderer`
//! (`WasapiRenderer` on Windows, `PipeWireAudioRenderer` on Linux,
//! `CoreAudioRenderer` on macOS). The video branches differ in more than
//! backend types, because only some of them decode on the GPU:
//!
//!     Windows: FileDemuxer -> SwDecoder -> Queue -> VideoSynchronizer
//!              -> D3d12WindowRenderer
//!     Linux:   FileDemuxer -> CudaDecoder -> Queue -> VideoSynchronizer
//!              -> VulkanWindowRenderer
//!     macOS:   FileDemuxer -> VideoToolboxDecoder -> Queue
//!              -> VideoSynchronizer -> MetalWindowRenderer
//!
//! The Linux and macOS branches never bring decoded pixels to the CPU: NVDEC
//! keeps every frame in CUDA memory and the renderer copies it straight into
//! Vulkan-owned memory; VideoToolbox decodes into pixel buffers the Metal
//! renderer draws as they are. Windows decodes in system memory and the renderer
//! uploads each frame itself — through a `SwScaler` after the synchronizer
//! only for a stream it cannot draw as it comes.

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn main() {
    eprintln!(
        "{} supports Windows, Linux and macOS only",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() -> media_pp::Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: av_playback <video-with-audio.mp4>");
        std::process::exit(2);
    };
    windows_example::play(path)
}

#[cfg(target_os = "linux")]
fn main() -> media_pp::Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: av_playback <video-with-audio.mp4>");
        std::process::exit(2);
    };
    linux_example::play(path)
}

/// A window is the main thread's on macOS: the main thread runs AppKit's
/// event loop, and the example runs beside it.
#[cfg(target_os = "macos")]
fn main() -> media_pp::Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: av_playback <video-with-audio.mp4>");
        std::process::exit(2);
    };
    media_pp::elements::run_with_windows(move || macos_example::play(path))
}

#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
mod shell;

/// The parts of `play` that are the same on every backend: opening the file
/// and locating the two streams it needs.
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
mod common {
    use media_pp::elements::FileDemuxer;
    use media_pp::ffmpeg::{codec::Parameters, media};

    pub struct Streams {
        pub video_index: usize,
        pub video_params: Parameters,
        pub audio_index: usize,
        pub audio_params: Parameters,
    }

    pub fn open(path: &str) -> media_pp::Result<(FileDemuxer, Streams)> {
        let (source, _) = FileDemuxer::open("demux", path)?;
        let video = source.best(media::Type::Video)?;
        let audio = source.best(media::Type::Audio)?;
        let streams = Streams {
            video_index: video.index,
            video_params: video.parameters.clone(),
            audio_index: audio.index,
            audio_params: audio.parameters.clone(),
        };
        Ok((source, streams))
    }
}

#[cfg(target_os = "windows")]
mod windows_example {
    use media_pp::{
        Error,
        bus::BusEvent,
        elements::{
            AudioResampler, D3d12Gpu, D3d12WindowRenderer, SwDecoder, VideoSynchronizer,
            WasapiRenderer, WasapiRendererOptions, WindowOptions,
        },
        pipeline::Pipeline,
    };

    use crate::common;

    pub fn play(path: String) -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (source, streams) = common::open(&path)?;

        // Decoded frames stay in system memory; the renderer uploads them.
        let gpu = D3d12Gpu::new()?;
        let (renderer, window) = D3d12WindowRenderer::open(
            "video-renderer",
            &gpu,
            WindowOptions {
                title: "media-pp A/V playback".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);
        let to_drawable = media_pp::elements::SwScaler::if_needed(
            "to-drawable",
            &streams.video_params,
            &renderer,
        )?;

        let (pipeline, audio_tee_handle) =
            Pipeline::new("av-playback", source, |source, context| {
                let mut video_branch = context
                    .branch()
                    .pipe(SwDecoder::new(
                        "video-decoder",
                        streams.video_params.clone(),
                    )?)
                    .queue("video-frames", 32)
                    .pipe(VideoSynchronizer::new("video-sync"));
                // After the synchronizer, not before: a frame it drops for
                // being late never pays for the conversion.
                if let Some(to_drawable) = to_drawable {
                    video_branch = video_branch.pipe(to_drawable);
                }
                context.attach(source, streams.video_index, video_branch.to(renderer)?)?;

                // Keep a stable insertion point on the demuxer's audio pad. With
                // no branches attached the Tee cheaply drops packets, so playback
                // starts video-only without decoding audio.
                let (audio_tee, handle) = context.tee("audio-tee").build_dynamic()?;
                context.attach(source, streams.audio_index, audio_tee)?;
                Ok(handle)
            })?;

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
            std::thread::spawn(move || {
                let attach_tee = tee.clone();
                crate::shell::read_commands(pipeline.clone(), tee, "WASAPI", move || {
                    attach_audio(&attach_tee, &params)
                });
            });
        }

        crate::drain_bus(&pipeline);
        Ok(())
    }

    fn attach_audio(
        audio_tee: &media_pp::elements::TeeHandle,
        audio_params: &media_pp::ffmpeg::codec::Parameters,
    ) -> media_pp::Result<media_pp::graph::BranchId> {
        let device = WasapiRenderer::list_devices()?
            .into_iter()
            .find(|device| device.is_default)
            .ok_or_else(|| Error::Other("no default WASAPI render endpoint".into()))?;
        let device_name = device.name.clone();
        let (audio_renderer, output_format) =
            WasapiRenderer::open("speakers", WasapiRendererOptions { device })?;

        let branch = audio_tee
            .branch()?
            .pipe(SwDecoder::new("audio-decoder", audio_params.clone())?)
            .pipe(AudioResampler::new("audio-resampler", output_format))
            .queue("audio-output", 8)
            .to(audio_renderer)?;
        let branch_id = audio_tee.attach(branch)?;
        println!("audio on: {device_name}; video is synchronized to played audio");
        Ok(branch_id)
    }

    #[allow(unused_imports)]
    use BusEvent as _;
}

#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::{
        Error,
        elements::{
            AudioResampler, CudaDecoder, CudaDevice, PipeWireAudioRenderer,
            PipeWireAudioRendererOptions, SwDecoder, TeeHandle, VideoSynchronizer, VulkanGpu,
            VulkanWindowRenderer, WindowOptions,
        },
        pipeline::Pipeline,
    };

    use crate::common;

    /// Matches the `video-frames` queue below. NVDEC's surface pool is fixed
    /// at open time. `CudaDecoder` reserves its accurate-seek candidate
    /// internally, so construction supplies only this downstream queue depth.
    /// NVDEC also caps the total pool at 32 surfaces, which is why this is far
    /// shallower than the Windows branch's software decoder can afford; see
    /// `CudaDecoder::new`'s docs. Eight frames are about 266 ms at 30 fps.
    const VIDEO_QUEUE_DEPTH: usize = 8;

    pub fn play(path: String) -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (source, streams) = common::open(&path)?;

        // One CUDA context for the whole stack: the decoder allocates frames
        // on it and the renderer copies out of it into memory its Vulkan
        // device allocated — which is why that device is made for this CUDA
        // one. The renderer rejects any frame from a different one.
        let cuda = CudaDevice::new()?;
        let gpu = VulkanGpu::for_cuda(&cuda)?;
        let (renderer, window) = VulkanWindowRenderer::open(
            "video-renderer",
            &gpu,
            WindowOptions {
                title: "media-pp A/V playback".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);

        let (pipeline, audio_tee_handle) =
            Pipeline::new("av-playback", source, |source, context| {
                let video_branch = context
                    .branch()
                    .pipe(CudaDecoder::new(
                        "video-decoder",
                        streams.video_params.clone(),
                        &cuda,
                        VIDEO_QUEUE_DEPTH as i32,
                    )?)
                    .queue("video-frames", VIDEO_QUEUE_DEPTH)
                    .pipe(VideoSynchronizer::new("video-sync"))
                    .to(renderer)?;
                context.attach(source, streams.video_index, video_branch)?;

                let (audio_tee, handle) = context.tee("audio-tee").build_dynamic()?;
                context.attach(source, streams.audio_index, audio_tee)?;
                Ok(handle)
            })?;

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
            std::thread::spawn(move || {
                let attach_tee = tee.clone();
                crate::shell::read_commands(pipeline.clone(), tee, "PipeWire", move || {
                    attach_audio(&attach_tee, &params)
                });
            });
        }

        crate::drain_bus(&pipeline);
        Ok(())
    }

    fn attach_audio(
        audio_tee: &TeeHandle,
        audio_params: &media_pp::ffmpeg::codec::Parameters,
    ) -> media_pp::Result<media_pp::graph::BranchId> {
        let device = PipeWireAudioRenderer::list_devices()?
            .into_iter()
            .find(|device| device.is_default)
            .ok_or_else(|| Error::Other("no default PipeWire playback device".into()))?;
        let device_name = device.name.clone();
        let (audio_renderer, output_format) =
            PipeWireAudioRenderer::open("speakers", PipeWireAudioRendererOptions { device })?;

        let branch = audio_tee
            .branch()?
            .pipe(SwDecoder::new("audio-decoder", audio_params.clone())?)
            .pipe(AudioResampler::new("audio-resampler", output_format))
            .queue("audio-output", 8)
            .to(audio_renderer)?;
        let branch_id = audio_tee.attach(branch)?;
        println!("audio on: {device_name}; video is synchronized to played audio");
        Ok(branch_id)
    }
}

#[cfg(target_os = "macos")]
mod macos_example {
    use media_pp::{
        Error,
        elements::{
            AudioResampler, CoreAudioRenderer, CoreAudioRendererOptions, MetalWindowRenderer,
            SwDecoder, TeeHandle, VideoSynchronizer, VideoToolboxDecoder, VideoToolboxDevice,
            WindowOptions,
        },
        pipeline::Pipeline,
    };

    use crate::common;

    /// Matches the `video-frames` queue below. VideoToolbox's pool grows as
    /// frames are held, so this is a latency choice rather than a budget:
    /// eight frames are about 266 ms at 30 fps, as on Linux.
    const VIDEO_QUEUE_DEPTH: usize = 8;

    pub fn play(path: String) -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (source, streams) = common::open(&path)?;

        // The decoder makes its pixel buffers on this context; the renderer
        // draws them as they are, since a pixel buffer belongs to no device.
        let device = VideoToolboxDevice::new()?;
        let (renderer, window) = MetalWindowRenderer::open(
            "video-renderer",
            WindowOptions {
                title: "media-pp A/V playback".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);

        let (pipeline, audio_tee_handle) =
            Pipeline::new("av-playback", source, |source, context| {
                let video_branch = context
                    .branch()
                    .pipe(VideoToolboxDecoder::new(
                        "video-decoder",
                        streams.video_params.clone(),
                        &device,
                    )?)
                    .queue("video-frames", VIDEO_QUEUE_DEPTH)
                    .pipe(VideoSynchronizer::new("video-sync"))
                    .to(renderer)?;
                context.attach(source, streams.video_index, video_branch)?;

                let (audio_tee, handle) = context.tee("audio-tee").build_dynamic()?;
                context.attach(source, streams.audio_index, audio_tee)?;
                Ok(handle)
            })?;

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
            std::thread::spawn(move || {
                let attach_tee = tee.clone();
                crate::shell::read_commands(pipeline.clone(), tee, "Core Audio", move || {
                    attach_audio(&attach_tee, &params)
                });
            });
        }

        crate::drain_bus(&pipeline);
        Ok(())
    }

    fn attach_audio(
        audio_tee: &TeeHandle,
        audio_params: &media_pp::ffmpeg::codec::Parameters,
    ) -> media_pp::Result<media_pp::graph::BranchId> {
        let device = CoreAudioRenderer::list_devices()?
            .into_iter()
            .find(|device| device.is_default)
            .ok_or_else(|| Error::Other("no default Core Audio output device".into()))?;
        let device_name = device.name.clone();
        let (audio_renderer, output_format) =
            CoreAudioRenderer::open("speakers", CoreAudioRendererOptions { device })?;

        let branch = audio_tee
            .branch()?
            .pipe(SwDecoder::new("audio-decoder", audio_params.clone())?)
            .pipe(AudioResampler::new("audio-resampler", output_format))
            .queue("audio-output", 8)
            .to(audio_renderer)?;
        let branch_id = audio_tee.attach(branch)?;
        println!("audio on: {device_name}; video is synchronized to played audio");
        Ok(branch_id)
    }
}

/// The bus loop, identical on every backend.
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
fn drain_bus(pipeline: &media_pp::pipeline::Pipeline) {
    use media_pp::bus::BusEvent;

    for event in pipeline.bus().iter() {
        // The file's source stays at its end, where a seek could still take
        // it back; the pipeline says when everything it read has played.
        if matches!(event, BusEvent::Finished) {
            println!("finished");
            pipeline.stop();
            continue;
        }
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
