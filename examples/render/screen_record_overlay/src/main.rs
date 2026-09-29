//! Records the desktop with a live overlay drawn on top, with every pixel
//! staying on the GPU from the moment it is captured to the moment it is
//! encoded.
//!
//! Two pipelines: `PipeWireScreenCaptureSource` (GPU mode) `-> Queue ->
//! CudaConverter ->` a `CudaVideoCompositor` input, and the compositor (with
//! a `CudaTextLayerHandle`) `-> Queue -> CudaEncoder -> FileMuxer`.
//!
//! The contrast with `screen_record_nvenc`, which records the capture
//! untouched, is the point: here something draws on it. The canvas is NV12,
//! and converting the BGRA capture up front lets it be copied in rather than
//! blended as BGRA. Nothing comes back to system memory: the capture is
//! imported as a CUDA surface, converted and composited by kernels, and
//! encoded by NVENC.
//!
//! The clock in the corner is redrawn once a second, so the recording proves
//! the overlay is live rather than a watermark baked in once. Any number of
//! further layers attach the same way — `add_source` for a video layer,
//! `add_text_layer` for another caption.
//!
//! On macOS the same graph is `ScreenCaptureKitSource` (VideoToolbox frames)
//! `-> Queue ->` a `MetalVideoCompositor` input, and the compositor (with a
//! `MetalTextLayerHandle`) `-> Queue -> VideoToolboxEncoder -> FileMuxer`,
//! with no conversion in it, since Metal composites the capture's BGRA
//! directly. The Windows shape is `DxgiCaptureSource` (GPU mode) `->
//! D3d11VideoCompositor -> D3d11VideoEncoder`, for the same reason.
//!
//! Needs an NVIDIA GPU and an ffmpeg build with NVENC on Linux.
//!
//!     cargo run -p screen_record_overlay -- <output.mp4> [seconds] [monitor|window] [restore-token]
//!     cargo run -p screen_record_overlay -- <output.mp4> [seconds] [monitor|window]   # macOS

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn main() {
    eprintln!(
        "{} example only supports Linux (PipeWire) and macOS (ScreenCaptureKit)",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "macos")]
fn main() -> impl std::process::Termination {
    macos_example::run()
}

#[cfg(target_os = "linux")]
mod linux_example {
    use std::time::{Duration, Instant};

    use media_pp::ffmpeg;
    use media_pp::{
        color::Color,
        elements::{
            CaptureSourceKind, CudaCodec, CudaConverter, CudaDevice, CudaEncoder,
            CudaEncoderOptions, CudaFrameFormat, CudaVideoCompositor, FileMuxer,
            PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource, TextLayer,
            VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
        },
        pipeline::Pipeline,
    };

    /// Fonts this crate does not bundle. The first one present wins; a system
    /// with none of them gets a clear error rather than an empty overlay.
    const FONT_CANDIDATES: [&str; 4] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ];

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!(
                "usage: screen_record_overlay <output.mp4> [seconds] [monitor|window] \
                 [restore-token]"
            );
            std::process::exit(2);
        };
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);
        // Monitor by default, matching every other capture example here.
        let source_kind = match std::env::args().nth(3).as_deref() {
            Some("window") => CaptureSourceKind::Window,
            _ => CaptureSourceKind::Monitor,
        };
        // Last so it can simply be left off: it is a long opaque string that
        // only a repeat run has.
        let restore_token = std::env::args().nth(4);
        if restore_token.is_none() {
            eprintln!("opening the portal — approve the screen-share dialog to continue...");
        }

        // One CUDA context for the whole stack: the capture imports its
        // DMA-BUFs onto it, the converter and compositor draw on it, and
        // NVENC encodes from it. Every element rejects a frame from another.
        let cuda = CudaDevice::new()?;
        let frame_rate = ffmpeg::Rational::new(30, 1);
        let (source, format, restore_token) = PipeWireScreenCaptureSource::open_gpu(
            "screen",
            PipeWireScreenCaptureOptions {
                frame_rate,
                source_kind,
                include_cursor: true,
                restore_token,
            },
            &cuda,
        )?;
        let (width, height) = (format.width, format.height);

        // Composited at the capture's own size, so the recording is the
        // desktop with something drawn on it rather than a rescaling of it.
        // Odd dimensions are refused by the converter at open — see its docs.
        let (compositor, handle) = CudaVideoCompositor::new(
            "compositor",
            &cuda,
            VideoCompositorOptions {
                width,
                height,
                frame_rate,
                // Only visible if the capture ever fails to fill the frame.
                background: Color::new(16, 16, 16),
                background_alpha: 255,
                mode: media_pp::elements::RenderMode::Live,
            },
        )?;

        let capture_input = handle.add_source(
            "desktop",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, width, height))
            },
        )?;
        let capture_sink = capture_input.sink;

        // The text layer receives no frames — no `Sink` to wire up, just a
        // handle driven by `set_text`.
        let (font_path, font_data) = FONT_CANDIDATES
            .iter()
            .find_map(|path| std::fs::read(path).ok().map(|data| (*path, data)))
            .ok_or_else(|| {
                media_pp::Error::Other(format!(
                    "no usable font found; looked for {FONT_CANDIDATES:?}"
                ))
            })?;
        println!("font: {font_path}");
        let mut text_layer = TextLayer::new(font_data);
        text_layer.font_size = 64.0;
        text_layer.x = 40;
        text_layer.y = 40;
        text_layer.color = Color::new(255, 220, 0);
        let clock = handle.add_text_layer("clock", text_layer)?;
        clock.set_text("rec 0s")?;

        let (capture_pipeline, ()) = Pipeline::new("desktop-capture", source, |source, ctx| {
            let converter =
                CudaConverter::new("convert", &cuda, media_pp::elements::CudaFrameFormat::Nv12)?;
            let branch = ctx
                .branch()
                // Thread boundary so conversion and compositing cannot stall
                // capture; the compositor keeps producing at its own rate.
                .queue("captured", 4)
                .pipe(converter)
                .to(capture_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        let encoder = CudaEncoder::new(
            "encoder",
            &cuda,
            CudaEncoderOptions {
                codec: CudaCodec::H264,
                // What the compositor produces, and what NVENC takes without
                // a conversion of its own.
                input_format: CudaFrameFormat::Nv12,
                width,
                height,
                frame_rate,
                bit_rate: 8_000_000,
                gop_size: 60, // ~2s @ 30fps
                max_b_frames: None,
            },
        )?;
        let mut muxer = FileMuxer::create(&path)?;
        let track = muxer.add_stream("video", &encoder)?;
        let muxer_sink = muxer.open()?.take(track)?;

        let (record_pipeline, ()) = Pipeline::new("overlay-record", compositor, |source, ctx| {
            let branch = ctx
                .branch()
                // Thread boundary so a slow encode cannot stall compositing.
                .queue("composited", 4)
                .pipe(encoder)
                .to(muxer_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        record_pipeline.run()?;
        capture_pipeline.run()?;

        println!("recording {seconds}s of {width}x{height} desktop with an overlay to {path} ...");
        let started = Instant::now();
        let duration = Duration::from_secs(seconds);
        let mut shown = 0;
        while started.elapsed() < duration {
            std::thread::sleep(Duration::from_millis(100));
            let elapsed = started.elapsed().as_secs().min(seconds);
            if elapsed != shown {
                shown = elapsed;
                // Redrawn every second, so a recording that shows the same
                // caption throughout is a broken overlay rather than a still
                // desktop.
                clock.set_text(&format!("rec {elapsed}s"))?;
            }
        }

        // Capture first: the compositor keeps its last frame per input, so
        // stopping it before the recorder cannot leave a gap.
        capture_pipeline.stop();
        record_pipeline.finish();

        for pipeline in [&capture_pipeline, &record_pipeline] {
            for event in pipeline.bus().iter() {
                println!("{event}");
            }
        }

        println!("wrote {path}");
        match restore_token {
            Some(token) => println!(
                "re-run without a dialog:\n  ... {path} {seconds} {} {token}",
                if matches!(source_kind, CaptureSourceKind::Window) {
                    "window"
                } else {
                    "monitor"
                }
            ),
            None => println!("the compositor issued no restore token; the next run will prompt"),
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod macos_example {
    use std::time::{Duration, Instant};

    use media_pp::ffmpeg;
    use media_pp::{
        color::Color,
        elements::{
            FileMuxer, MetalVideoCompositor, ScreenCaptureKitOptions, ScreenCaptureKitSource,
            ScreenCaptureKitTarget, TextLayer, VideoCompositorOptions, VideoFit, VideoLayer,
            VideoRect, VideoToolboxCodec, VideoToolboxDevice, VideoToolboxEncoder,
            VideoToolboxEncoderOptions, VideoToolboxFrameFormat,
        },
        pipeline::Pipeline,
    };

    /// Fonts macOS ships. The first one present wins; a system with none of
    /// them gets a clear error rather than an empty overlay.
    const FONT_CANDIDATES: [&str; 3] = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ];

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: screen_record_overlay <output.mp4> [seconds] [monitor|window]");
            std::process::exit(2);
        };
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);
        let target = match std::env::args().nth(3).as_deref() {
            Some("window") => {
                // An application's untitled windows are toolbars and strips
                // more often than documents.
                let window = ScreenCaptureKitSource::list_windows()?
                    .into_iter()
                    .find(|window| !window.title.is_empty())
                    .ok_or_else(|| {
                        media_pp::Error::Other("there is no window to capture".into())
                    })?;
                println!("capturing {:?} of {}", window.title, window.application);
                ScreenCaptureKitTarget::Window(window.id)
            }
            _ => {
                let display = ScreenCaptureKitSource::list_displays()?
                    .into_iter()
                    .find(|display| display.is_main)
                    .ok_or_else(|| media_pp::Error::Other("there is no main display".into()))?;
                ScreenCaptureKitTarget::Display(display.id)
            }
        };

        // The capture, the compositor's canvas and the encoder all make
        // their pixel buffers on this context; none of them belongs to it,
        // so each draws or reads the others' as they are.
        let device = VideoToolboxDevice::new()?;
        let frame_rate = ffmpeg::Rational::new(30, 1);
        let (source, format) = ScreenCaptureKitSource::open_videotoolbox(
            "screen",
            ScreenCaptureKitOptions {
                frame_rate,
                include_cursor: true,
                ..ScreenCaptureKitOptions::new(target)
            },
            &device,
        )?;
        // An NV12 canvas has even sides; a window can be any size at all, and
        // is stretched over the one pixel it loses.
        let (width, height) = (format.width & !1, format.height & !1);

        let (compositor, handle) = MetalVideoCompositor::with_format(
            "compositor",
            &device,
            VideoCompositorOptions {
                width,
                height,
                frame_rate,
                // Only visible if the capture ever fails to fill the frame.
                background: Color::new(16, 16, 16),
                background_alpha: 255,
                mode: media_pp::elements::RenderMode::Live,
            },
            VideoToolboxFrameFormat::Nv12,
        )?;

        let capture_input = handle.add_source(
            "desktop",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, width, height))
            },
        )?;
        let capture_sink = capture_input.sink;

        // The text layer receives no frames — no `Sink` to wire up, just a
        // handle driven by `set_text`.
        let (font_path, font_data) = FONT_CANDIDATES
            .iter()
            .find_map(|path| std::fs::read(path).ok().map(|data| (*path, data)))
            .ok_or_else(|| {
                media_pp::Error::Other(format!(
                    "no usable font found; looked for {FONT_CANDIDATES:?}"
                ))
            })?;
        println!("font: {font_path}");
        let mut text_layer = TextLayer::new(font_data);
        text_layer.font_size = 64.0;
        text_layer.x = 40;
        text_layer.y = 40;
        text_layer.color = Color::new(255, 220, 0);
        let clock = handle.add_text_layer("clock", text_layer)?;
        clock.set_text("rec 0s")?;

        let (capture_pipeline, ()) = Pipeline::new("desktop-capture", source, |source, ctx| {
            let branch = ctx
                .branch()
                // Thread boundary so compositing cannot stall capture; the
                // compositor keeps producing at its own rate.
                .queue("captured", 4)
                .to(capture_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        let encoder = VideoToolboxEncoder::new(
            "encoder",
            &device,
            VideoToolboxEncoderOptions {
                codec: VideoToolboxCodec::H264,
                // What the compositor produces, and what the media engine
                // takes as it is.
                format: VideoToolboxFrameFormat::Nv12,
                width,
                height,
                frame_rate,
                bit_rate: 8_000_000,
                gop_size: 60, // ~2s @ 30fps
                max_b_frames: None,
            },
        )?;
        let mut muxer = FileMuxer::create(&path)?;
        let track = muxer.add_stream("video", &encoder)?;
        let muxer_sink = muxer.open()?.take(track)?;

        let (record_pipeline, ()) = Pipeline::new("overlay-record", compositor, |source, ctx| {
            let branch = ctx
                .branch()
                // Thread boundary so a slow encode cannot stall compositing.
                .queue("composited", 4)
                .pipe(encoder)
                .to(muxer_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        record_pipeline.run()?;
        capture_pipeline.run()?;

        println!("recording {seconds}s of {width}x{height} screen with an overlay to {path} ...");
        let started = Instant::now();
        let duration = Duration::from_secs(seconds);
        let mut shown = 0;
        while started.elapsed() < duration {
            std::thread::sleep(Duration::from_millis(100));
            let elapsed = started.elapsed().as_secs().min(seconds);
            if elapsed != shown {
                shown = elapsed;
                // Redrawn every second, so a recording that shows the same
                // caption throughout is a broken overlay rather than a still
                // screen.
                clock.set_text(&format!("rec {elapsed}s"))?;
            }
        }

        // Capture first: the compositor keeps its last frame per input, so
        // stopping it before the recorder cannot leave a gap.
        capture_pipeline.stop();
        record_pipeline.finish();

        for pipeline in [&capture_pipeline, &record_pipeline] {
            for event in pipeline.bus().iter() {
                println!("{event}");
            }
        }

        println!("wrote {path}");
        Ok(())
    }
}
