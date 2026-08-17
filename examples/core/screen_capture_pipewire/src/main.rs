use std::{thread, time::Duration};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    elements::{
        CaptureSourceKind, Mp4Muxer, PipeWireCaptureOptions, PipeWireCaptureSource, Scaler,
        SwEncoder, SwEncoderOptions, VideoCodec,
    },
    pipeline::Pipeline,
};

/// PipeWireCaptureSource -> Scaler -> SwEncoder -> Mp4Muxer: records the
/// Wayland desktop the user picks in the portal dialog.
///
///     cargo run -p screen_capture_pipewire -- <output.mp4> [seconds] [restore-token]
///
/// The first run shows the compositor's screen-share dialog. It prints the
/// restore token it was issued; passing that back as the third argument
/// reconnects to the same source with no dialog at all. There is no monitor
/// index or region to pass instead — see `PipeWireCaptureSource`'s own docs.
fn main() -> media_pp::Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: screen_capture_pipewire <output.mp4> [seconds] [restore-token]");
        std::process::exit(2);
    };
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let restore_token = std::env::args().nth(3);

    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

    let fps = 30;
    let frame_rate = ffmpeg::Rational::new(fps as i32, 1);
    let time_base = frame_rate.invert();

    if restore_token.is_none() {
        eprintln!("opening the portal — approve the screen-share dialog to continue...");
    }
    let (source, capture_width, capture_height, restore_token) = PipeWireCaptureSource::open(
        "screen",
        PipeWireCaptureOptions {
            fps,
            source_kind: CaptureSourceKind::Monitor,
            include_cursor: true,
            restore_token,
        },
    )?;
    println!("capturing {capture_width}x{capture_height}");

    // H.264 needs even dimensions, and the encoder wants YUV420P rather than
    // the capture's full-range BGRA.
    let output_width = capture_width & !1;
    let output_height = capture_height & !1;

    let encoder = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: output_width,
            height: output_height,
            time_base,
            frame_rate,
            bit_rate: 8_000_000,
            gop_size: fps * 2,
        },
    )?;
    let mut muxer = Mp4Muxer::create(&path)?;
    muxer.add_stream("video", encoder.parameters(), time_base)?;
    let muxer_sink = muxer.open()?.pop().expect("one video stream");

    let pipeline = Pipeline::new("screen-capture", source, |source, ctx| {
        let scaler = Scaler::new(
            "to-yuv",
            ffmpeg::format::Pixel::YUV420P,
            output_width,
            output_height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let branch = ctx
            .branch()
            .pipe(scaler)
            // A Queue keeps the capture loop's own pacing independent of how
            // long encoding a 1080p frame takes.
            .queue("encode-frames", 8)
            .pipe(encoder)
            .to(muxer_sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;

    pipeline.run();
    println!("recording for {seconds}s...");
    thread::sleep(Duration::from_secs(seconds));
    pipeline.finish();

    for event in pipeline.bus().iter() {
        if let BusEvent::Error { name, error, .. } = event {
            eprintln!("[{name}] error: {error}");
        }
    }

    println!("wrote {path}");
    match restore_token {
        Some(token) => println!("re-run without a dialog:\n  ... {path} {seconds} {token}"),
        None => println!("the compositor issued no restore token; the next run will prompt again"),
    }
    Ok(())
}
