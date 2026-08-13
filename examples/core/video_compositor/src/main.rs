use std::{thread, time::Duration};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    color::Color,
    elements::{
        Mp4Muxer, Scaler, SwEncoder, SwEncoderOptions, TestVideoOptions, TestVideoSource,
        VideoCodec, VideoCompositor, VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
    },
    pipeline::Pipeline,
};

/// Two TestVideoSource pipelines -> VideoCompositor -> Scaler ->
/// SwEncoder -> Mp4Muxer. The foreground layer moves at runtime through
/// its VideoLayerHandle while both source connections stay unchanged.
///
///     cargo run -p video_compositor -- [output.mp4] [seconds]
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "video_compositor.mp4".into());
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    let output_width = 640;
    let output_height = 360;
    let frame_rate = ffmpeg::Rational::new(30, 1);
    let (compositor, compositor_handle) = VideoCompositor::new(
        "compositor",
        VideoCompositorOptions {
            width: output_width,
            height: output_height,
            frame_rate,
            background: Color::new(24, 24, 24),
        },
    )?;
    let time_base = compositor.time_base();

    let mut background_layer = VideoLayer::new(VideoRect::new(0, 0, output_width, output_height));
    background_layer.fit = VideoFit::Cover;
    let background_input = compositor_handle
        .add_source("background", background_layer)?
        .expect("compositor is alive");
    let background_sink = background_input.sink;

    let foreground_width = 192;
    let foreground_height = 144;
    let mut foreground_layer = VideoLayer::new(VideoRect::new(
        0,
        output_height as i32 - foreground_height as i32,
        foreground_width,
        foreground_height,
    ));
    foreground_layer.z_index = 1;
    foreground_layer.opacity = 0.85;
    foreground_layer.fit = VideoFit::Cover;
    let foreground_input = compositor_handle
        .add_source("foreground", foreground_layer)?
        .expect("compositor is alive");
    let foreground_sink = foreground_input.sink;
    let foreground_handle = foreground_input.layer;

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
            let branch = ctx.branch().to(background_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

    // A different size and frame rate demonstrates that compositor inputs
    // are independent live pipelines; each sink retains only its latest
    // frame and the compositor emits on its own 30fps clock.
    let foreground_source = TestVideoSource::new(
        "foreground-source",
        TestVideoOptions {
            width: 320,
            height: 240,
            framerate: ffmpeg::Rational::new(15, 1),
        },
    );
    let foreground_pipeline =
        Pipeline::new("foreground-input", foreground_source, |source, ctx| {
            let branch = ctx.branch().to(foreground_sink)?;
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
            .queue("encode-frames", 8)
            .pipe(encoder)
            .to(muxer_sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;

    output_pipeline.run();
    background_pipeline.run();
    foreground_pipeline.run();

    let steps = seconds.saturating_mul(30);
    let travel = output_width - foreground_width;
    for step in 0..steps {
        let x = if steps <= 1 {
            0
        } else {
            (u64::from(travel) * step / (steps - 1)) as i32
        };
        foreground_handle.set_rect(VideoRect::new(
            x,
            output_height as i32 - foreground_height as i32,
            foreground_width,
            foreground_height,
        ))?;
        thread::sleep(Duration::from_millis(33));
    }

    background_pipeline.stop();
    foreground_pipeline.stop();
    output_pipeline.stop();

    for pipeline in [&background_pipeline, &foreground_pipeline, &output_pipeline] {
        for event in pipeline.bus().iter() {
            if let BusEvent::Error { name, error, .. } = event {
                eprintln!("[{name}] error: {error}");
            }
        }
    }

    println!("wrote {path}");
    Ok(())
}
