use std::{thread, time::Duration};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    color::Color,
    elements::{
        D3d11Download, D3d11Upload, D3d11VideoCompositor, Mp4Muxer, Scaler, SwEncoder,
        SwEncoderOptions, TestVideoOptions, TestVideoSource, TextLayer, VideoCodec,
        VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
    },
    pipeline::Pipeline,
};
use render_common::D3d11GpuContext;

/// A moving-gradient `TestVideoSource` background composited with a
/// `D3d11TextLayerHandle` clock in front of it, recorded to an mp4 — proves dynamic
/// text (not just a static watermark) actually updates on screen: the
/// overlaid text changes once a second while the recording runs, so the
/// output file's frames differ over time if `D3d11TextLayerHandle::set_text` is
/// really re-rasterizing and re-uploading each call.
///
///     cargo run -p text_overlay -- [output.mp4] [seconds]
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "text_overlay.mp4".into());
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);

    let gpu = D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

    let output_width = 640;
    let output_height = 360;
    let frame_rate = ffmpeg::Rational::new(30, 1);
    let (compositor, compositor_handle) = D3d11VideoCompositor::new(
        "compositor",
        gpu.device(),
        gpu.context(),
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
        .map_err(|e| media_pp::Error::Other(e.to_string()))?
        .expect("compositor is alive");
    let background_sink = background_input.sink;

    // `D3d11TextLayerHandle` never receives `Pipeline` frames — no `Sink` to wire up,
    // just a handle driven directly by `set_text`. `add_text_layer` takes a
    // `TextLayer` the same way `add_source` takes a `VideoLayer`, and
    // builds the `D3d11TextLayerHandle` in one call, always against this
    // compositor's own device.
    let font_data = std::fs::read(r"C:\Windows\Fonts\arial.ttf")
        .map_err(|e| media_pp::Error::Other(format!("failed to read font: {e}")))?;
    let mut text_layer = TextLayer::new(font_data);
    text_layer.font_size = 48.0;
    text_layer.x = 20;
    text_layer.y = 20;
    let overlay = compositor_handle
        .add_text_layer("clock", text_layer)
        .map_err(|e| media_pp::Error::Other(e.to_string()))?
        .expect("compositor is alive");
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
            let scaler = Scaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                output_width,
                output_height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = D3d11Upload::new("upload", gpu.device(), output_width, output_height);
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
        let download = D3d11Download::new(
            "download",
            gpu.device(),
            gpu.context(),
            output_width,
            output_height,
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let to_yuv = Scaler::new(
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

    output_pipeline.run();
    background_pipeline.run();

    for elapsed in 1..=seconds {
        thread::sleep(Duration::from_secs(1));
        overlay
            .set_text(&format!("t={elapsed}s"))
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
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
