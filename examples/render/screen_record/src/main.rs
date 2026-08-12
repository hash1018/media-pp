use std::{thread, time::Duration};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    elements::{
        CaptureMode, DxgiScreenOptions, DxgiScreenSource, Mp4Muxer, Scaler, SwEncoder,
        SwEncoderOptions, VideoCodec,
    },
    pipeline::Pipeline,
};

/// DxgiScreenSource -> Scaler -> SwEncoder -> Mp4Muxer: captures the
/// desktop live via DXGI Desktop Duplication and encodes it straight into
/// a playable `.mp4` file — no window, no renderer, just a headless
/// recording (compare `screen_capture`, which renders instead of encoding).
///
/// `DxgiScreenSource` never reaches `Eos` on its own (see its own docs);
/// this just captures for a fixed duration and then `pipeline.stop()`s,
/// which is also what finalizes the MP4's trailer — `Mp4Muxer` writes it on
/// `Stop` as well as `Eos`, unlike `RtspServer`, since an MP4 file needs a
/// valid trailer to be playable at all (see `Mp4Muxer`'s own docs).
///
///     cargo run -p screen_record -- [output.mp4] [seconds]
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "screen_record.mp4".into());
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let capture_options = DxgiScreenOptions {
        fps: 30,
        capture_mode: CaptureMode::Cpu {
            include_cursor: true,
        },
        ..DxgiScreenOptions::default()
    };
    let (source, width, height, _device) = DxgiScreenSource::open("screen", capture_options)?;
    let time_base = source.time_base();

    let encoder = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width,
            height,
            time_base,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 4_000_000,
            gop_size: 60, // ~2s @ 30fps
        },
    )
    .expect("failed to open encoder");
    // No container/demuxer in this loop to get these from — SwEncoder
    // exposes its own codec parameters for exactly this case (see
    // `transcode_render`'s own use of this, wiring a decoder instead).
    let mut muxer = Mp4Muxer::create(&path)?;
    muxer.add_stream("video", encoder.parameters(), time_base)?;
    let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

    let pipeline = Pipeline::new("screen-record", source, |source, ctx| {
        let scaler = Scaler::new(
            "to-yuv",
            ffmpeg::format::Pixel::YUV420P,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let branch = ctx
            .branch()
            .queue("captured", 4) // thread boundary so scaling doesn't block capture
            .pipe(scaler)
            .queue("frames", 8) // thread boundary so encoding doesn't block scaling
            .pipe(encoder)
            .to(muxer_sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;

    println!("recording {seconds}s of the desktop to {path} ...");
    pipeline.run();

    thread::sleep(Duration::from_secs(seconds));
    pipeline.stop();

    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            _ => {}
        }
    }

    println!("wrote {path}");
    Ok(())
}
