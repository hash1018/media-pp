//! A moving-gradient background composited with a live clock in front of it,
//! recorded to an mp4 — proof that dynamic text (not a static watermark)
//! really updates: the overlaid text changes once a second while recording, so
//! the output's frames differ over time only if `set_text` is actually
//! re-rasterizing and re-uploading each call.
//!
//! `TestVideoSource -> SwScaler(NV12) -> CudaUpload -> CudaVideoCompositor
//! (+ CudaTextLayerHandle) -> CudaDownload -> SwScaler(YUV420P) -> SwEncoder
//! -> Mp4Muxer`. Every composite and every text blend happens on the GPU; the
//! frame only comes back for the software encoder.
//!
//! Nothing about the graph is platform-specific — CUDA is a vendor backend,
//! not a Linux one, so this runs unchanged on Windows and Linux. `text_overlay`
//! is the D3D11 counterpart for the same graph. Only the raw-key terminal in
//! [`terminal`] and the system font path below differ per OS.
//!
//!     cargo run -p cuda_text_overlay -- [output.mp4] [seconds]
//!     (use the arrow keys to move the text, or `q` to stop early)

mod terminal;

use std::{
    sync::mpsc::RecvTimeoutError,
    thread,
    time::{Duration, Instant},
};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    color::Color,
    elements::{
        CudaDevice, CudaDownload, CudaFrameFormat, CudaUpload, CudaVideoCompositor, Mp4Muxer,
        SwEncoder, SwEncoderOptions, SwScaler, TestVideoOptions, TestVideoSource, TextLayer,
        VideoCodec, VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
    },
    pipeline::Pipeline,
};

use terminal::{MOVE_STEP, TerminalCommand};

/// Fonts this crate does not bundle. The first one present wins; a system with
/// none of them gets a clear error rather than an empty overlay.
#[cfg(windows)]
const FONT_CANDIDATES: [&str; 2] = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
];
#[cfg(unix)]
const FONT_CANDIDATES: [&str; 4] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
];

fn main() -> impl std::process::Termination {
    run()
}

fn run() -> media_pp::Result<()> {
    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cuda_text_overlay.mp4".into());
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    // One CUDA context for the whole stack: the upload allocates on it, the
    // compositor draws on it, and the download reads from it.
    let cuda = CudaDevice::new().map_err(|e| media_pp::Error::Other(e.to_string()))?;

    let output_width = 640;
    let output_height = 360;
    let frame_rate = ffmpeg::Rational::new(30, 1);
    let (compositor, compositor_handle) = CudaVideoCompositor::new(
        "compositor",
        &cuda,
        VideoCompositorOptions {
            width: output_width,
            height: output_height,
            frame_rate,
            background: Color::new(24, 24, 24),
        },
    )
    .map_err(|e| media_pp::Error::Other(e.to_string()))?;
    let time_base = compositor.time_base();

    let mut background_layer = VideoLayer::new(VideoRect::new(0, 0, output_width, output_height));
    background_layer.fit = VideoFit::Cover;
    let background_input = compositor_handle
        .add_source("background", background_layer)
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
    let background_sink = background_input.sink;

    // The text layer never receives `Pipeline` frames — no `Sink` to wire up,
    // just a handle driven directly by `set_text`. `add_text_layer` takes a
    // `TextLayer` the same way `add_source` takes a `VideoLayer`.
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
    text_layer.font_size = 48.0;
    text_layer.x = 20;
    text_layer.y = 20;
    let overlay = compositor_handle
        .add_text_layer("clock", text_layer)
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
    overlay
        .set_text("t=0s")
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;

    let background_source = TestVideoSource::new(
        "background-source",
        TestVideoOptions {
            width: output_width,
            height: output_height,
            framerate: frame_rate,
        },
    );
    let background_pipeline =
        Pipeline::new("background-input", background_source, |source, ctx| {
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                output_width,
                output_height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new(
                "upload",
                &cuda,
                CudaFrameFormat::Nv12,
                output_width,
                output_height,
            )
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            let branch = ctx.branch().pipe(scaler).pipe(upload).to(background_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

    let encoder = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: output_width,
            height: output_height,
            time_base,
            frame_rate,
            bit_rate: 2_000_000,
            gop_size: 60,
        },
    )?;
    let mut muxer = Mp4Muxer::create(&path)?;
    muxer.add_stream("video", encoder.parameters(), time_base)?;
    let muxer_sink = muxer.open()?.pop().expect("one video stream");

    let output_pipeline = Pipeline::new("composited-output", compositor, |source, ctx| {
        let download = CudaDownload::new(
            "download",
            &cuda,
            CudaFrameFormat::Nv12,
            output_width,
            output_height,
        );
        let to_yuv = SwScaler::new(
            "to-yuv",
            ffmpeg::format::Pixel::YUV420P,
            output_width,
            output_height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let branch = ctx
            .branch()
            .queue("record", 4)
            .pipe(download)
            .pipe(to_yuv)
            .queue("encode-frames", 8)
            .pipe(encoder)
            .to(muxer_sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;

    output_pipeline.run()?;
    background_pipeline.run()?;

    println!("controls: arrow keys move the text by {MOVE_STEP}px; q stops recording");
    let commands = terminal::commands();
    let started = Instant::now();
    let duration = Duration::from_secs(seconds);
    let mut next_text_update = Duration::from_secs(1);
    let (mut text_x, mut text_y) = (20, 20);
    let mut terminal_connected = true;

    while started.elapsed() < duration {
        let elapsed = started.elapsed();
        let wait = next_text_update
            .saturating_sub(elapsed)
            .min(duration.saturating_sub(elapsed));
        let command = if terminal_connected {
            commands.recv_timeout(wait)
        } else {
            thread::sleep(wait);
            Err(RecvTimeoutError::Timeout)
        };

        match command {
            Ok(TerminalCommand::Move { dx, dy }) => {
                text_x = (text_x + dx).clamp(0, output_width as i32);
                text_y = (text_y + dy).clamp(0, output_height as i32);
                overlay.set_position(text_x, text_y);
                println!("text position: ({text_x}, {text_y})");
            }
            Ok(TerminalCommand::Quit) => break,
            Err(RecvTimeoutError::Timeout) => {
                let elapsed_seconds = started.elapsed().as_secs().min(seconds);
                if elapsed_seconds > 0 && started.elapsed() >= next_text_update {
                    overlay
                        .set_text(&format!("t={elapsed_seconds}s"))
                        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
                    next_text_update = Duration::from_secs(elapsed_seconds.saturating_add(1));
                }
            }
            Err(RecvTimeoutError::Disconnected) => terminal_connected = false,
        }
    }

    background_pipeline.stop();
    output_pipeline.stop();

    for pipeline in [&background_pipeline, &output_pipeline] {
        for event in pipeline.bus().iter() {
            if let BusEvent::Error { name, error, .. } = event {
                eprintln!("[{name}] error: {error}");
            }
        }
    }

    println!("wrote {path}");
    Ok(())
}
