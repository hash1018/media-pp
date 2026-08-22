//! Records the desktop with a live overlay drawn on top, with every pixel
//! staying on the GPU from the moment it is captured to the moment it is
//! encoded.
//!
//! `PipeWireScreenCaptureSource` (GPU mode) `-> Queue -> CudaConverter ->
//! CudaVideoCompositor` (+ `CudaTextLayerHandle`) `-> Queue -> CudaEncoder ->
//! Mp4Muxer`.
//!
//! The contrast with `screen_record_nvenc` is the point of the graph. That
//! example records the capture untouched, which needs no conversion at all:
//! the capture is BGRA and NVENC ingests BGRA directly. The moment anything
//! wants to *draw* on the capture, that stops being enough — the compositor
//! works in NV12, like everything else on the CUDA path that is not the
//! encoder — so `CudaConverter` sits between them. Nothing here comes back to
//! system memory: the capture is imported as a CUDA surface, converted by a
//! kernel, composited by a kernel, and encoded by NVENC.
//!
//! The clock in the corner is redrawn once a second, so the recording proves
//! the overlay is live rather than a watermark baked in once. Any number of
//! further layers attach the same way — `add_source` for a video layer,
//! `add_text_layer` for another caption.
//!
//! Linux only: it is the GPU screen capture that is Linux-specific here, not
//! the CUDA half. The Windows shape of the same graph is `DxgiCaptureSource`
//! (GPU mode) `-> D3d11VideoCompositor -> D3d11NvencEncoder`, with no
//! conversion in it, since D3D11 composites BGRA directly.
//!
//! Needs an NVIDIA GPU and an ffmpeg build with NVENC.
//!
//!     cargo run -p screen_overlay_record -- <output.mp4> [seconds] [monitor|window] [restore-token]

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!(
        "{} example only supports Linux (PipeWire screen capture)",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "linux")]
mod linux_example {
    use std::time::{Duration, Instant};

    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        color::Color,
        elements::{
            CaptureSourceKind, CudaCodec, CudaConverter, CudaDevice, CudaEncoder,
            CudaEncoderOptions, CudaFrameFormat, CudaVideoCompositor, Mp4Muxer,
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
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!(
                "usage: screen_overlay_record <output.mp4> [seconds] [monitor|window] \
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
        let cuda = CudaDevice::new().map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let frame_rate = ffmpeg::Rational::new(30, 1);
        let (source, format, restore_token) = PipeWireScreenCaptureSource::open_gpu(
            "screen",
            PipeWireScreenCaptureOptions {
                fps: 30,
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
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let time_base = compositor.time_base();

        let capture_input = handle
            .add_source(
                "desktop",
                VideoLayer {
                    fit: VideoFit::Stretch,
                    ..VideoLayer::new(VideoRect::new(0, 0, width, height))
                },
            )
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
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
        let clock = handle
            .add_text_layer("clock", text_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        clock
            .set_text("rec 0s")
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let capture_pipeline = Pipeline::new("desktop-capture", source, |source, ctx| {
            let converter = CudaConverter::new("convert", &cuda, width, height)
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
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
                time_base,
                frame_rate,
                bit_rate: 8_000_000,
                gop_size: 60, // ~2s @ 30fps
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let mut muxer = Mp4Muxer::create(&path)?;
        muxer.add_stream("video", encoder.parameters(), time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let record_pipeline = Pipeline::new("overlay-record", compositor, |source, ctx| {
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
                clock
                    .set_text(&format!("rec {elapsed}s"))
                    .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            }
        }

        // Capture first: the compositor keeps its last frame per input, so
        // stopping it before the recorder cannot leave a gap.
        capture_pipeline.stop();
        record_pipeline.finish();

        for pipeline in [&capture_pipeline, &record_pipeline] {
            for event in pipeline.bus().iter() {
                match event {
                    BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                    BusEvent::Dropped { name, .. } => {
                        eprintln!("[{name}] dropped a buffer (queue full)")
                    }
                    _ => {}
                }
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
