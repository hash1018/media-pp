use std::{path::PathBuf, thread, time::Duration};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    elements::{
        HlsMode, HlsMuxer, HlsOptions, SwEncoder, SwEncoderOptions, TestVideoOptions,
        TestVideoSource, VideoCodec,
    },
    pipeline::Pipeline,
};

/// TestVideoSource -> SwEncoder -> HlsMuxer: produces a live fMP4 media
/// playlist, initialization file, and keyframe-aligned media segments.
///
///     cargo run -p hls -- [output-directory] [seconds]
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let output_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("hls_output"));
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(6);
    std::fs::create_dir_all(&output_dir)
        .map_err(|error| media_pp::Error::Other(error.to_string()))?;

    let video_options = TestVideoOptions {
        width: 640,
        height: 360,
        framerate: ffmpeg::Rational::new(30, 1),
    };
    let source = TestVideoSource::new("video", video_options);
    let time_base = source.time_base();
    let encoder = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: video_options.width,
            height: video_options.height,
            time_base,
            frame_rate: video_options.framerate,
            bit_rate: 1_500_000,
            // Match the two-second HLS target so each requested boundary
            // has a keyframe available at 30 fps.
            gop_size: 60,
        },
    )?;

    let playlist_path = output_dir.join("index.m3u8");
    let mut hls_options = HlsOptions::new(&playlist_path, output_dir.join("segment_%05d.m4s"));
    hls_options.segment_duration = Duration::from_secs(2);
    hls_options.mode = HlsMode::Live {
        window_size: 6,
        delete_old_segments: true,
    };
    let mut muxer = HlsMuxer::create(hls_options)?;
    muxer.add_stream("video", encoder.parameters(), time_base)?;
    let muxer_sink = muxer.open()?.pop().expect("one stream was registered");

    let pipeline = Pipeline::new("hls", source, |source, context| {
        let branch = context.branch().pipe(encoder).to(muxer_sink)?;
        context.attach(source, 0, branch)?;
        Ok(())
    })?;

    println!(
        "writing a live fMP4 HLS stream for {seconds}s to {}",
        playlist_path.display()
    );
    pipeline.run();
    thread::sleep(Duration::from_secs(seconds));
    pipeline.stop();

    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer"),
            _ => {}
        }
    }
    println!("wrote {}", playlist_path.display());
    Ok(())
}
