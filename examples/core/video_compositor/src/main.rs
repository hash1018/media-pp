use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use media_pp::ffmpeg;
use media_pp::{
    buffer::MediaBuffer,
    bus::BusEvent,
    color::Color,
    elements::{
        AppSource, AppSourceHandle, ChromaKeyMethod, ChromaKeyOptions, Mp4Muxer, SwChromaKey,
        SwEncoder, SwEncoderOptions, SwScaler, SwVideoCompositor, TestVideoOptions,
        TestVideoSource, VideoCodec, VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
    },
    pipeline::Pipeline,
    pool::UnboundObjectPool,
};

/// A TestVideoSource background, and a green-screen foreground fed from an
/// `AppSource` through `SwChromaKey` — both into one `SwVideoCompositor` ->
/// `SwScaler` -> `SwEncoder` -> `Mp4Muxer`. The foreground layer's on-canvas
/// position moves at runtime through its `SwVideoLayerHandle`; the "figure"
/// inside its own frame stays put, so the only thing chroma-keying visibly
/// changes is that the green around it disappears to reveal the moving
/// background layer underneath.
///
///     cargo run -p video_compositor -- [output.mp4] [seconds]
fn main() -> media_pp::Result<()> {
    media_pp::init()?;
    let _log_guard = media_pp::log::init(
        env!("CARGO_PKG_NAME"),
        "logs",
        media_pp::log::Level::Trace,
        7,
    )?;

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
    let (compositor, compositor_handle) = SwVideoCompositor::new(
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

    // A different size and rate (15fps, fed by hand below) demonstrates
    // that compositor inputs are independent live pipelines; each sink
    // retains only its latest frame and the compositor emits on its own
    // 30fps clock regardless of when this one last updated.
    let foreground_source_width = 320;
    let foreground_source_height = 240;
    let (foreground_source, foreground_app_handle) = AppSource::new("foreground-source", 4);
    let foreground_pipeline =
        Pipeline::new("foreground-input", foreground_source, |source, ctx| {
            let chroma_key = SwChromaKey::new(
                "green-screen",
                ChromaKeyOptions {
                    method: ChromaKeyMethod::Green,
                    threshold: 0.2,
                    smoothing: 0.1,
                },
            );
            let branch = ctx.branch().pipe(chroma_key).to(foreground_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
    let feeding = Arc::new(AtomicBool::new(true));
    let foreground_feeder = spawn_green_screen_feeder(
        foreground_app_handle,
        foreground_source_width,
        foreground_source_height,
        feeding.clone(),
    );

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
        let scaler = SwScaler::new(
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

    output_pipeline.run()?;
    background_pipeline.run()?;
    foreground_pipeline.run()?;

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

    // Stop the feeder and let it push its own `Eos` while the pipeline is
    // still fully running, before `.stop()` below tears it down — avoids
    // racing a final `push` against a pipeline that already stopped.
    feeding.store(false, Ordering::Relaxed);
    foreground_feeder.join().expect("feeder thread panicked")?;

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

/// Draws a green background with a static, differently-colored "figure" —
/// a circular head over a rectangular body, the simplest shape that reads
/// as a person rather than an arbitrary block. `SwChromaKey` (with
/// `ChromaKeyMethod::Green`) then keys the green out, so only the figure
/// survives to composite over whatever the background layer is showing
/// underneath.
fn fill_green_screen_frame(frame: &mut ffmpeg::frame::Video, width: u32, height: u32) {
    let stride = frame.stride(0);
    let data = frame.data_mut(0);
    let green = [0u8, 255, 0, 255]; // BGRA: matches ChromaKeyMethod::Green exactly
    let figure = [60u8, 140, 220, 255]; // BGRA: a skin-toned color, far from green

    let body_left = i64::from(width) * 3 / 8;
    let body_right = i64::from(width) * 5 / 8;
    let body_top = i64::from(height) / 2;
    let body_bottom = i64::from(height) * 7 / 8;

    let head_center_x = i64::from(width) / 2;
    let head_center_y = i64::from(height) * 3 / 8;
    let head_radius = i64::from(height) / 6;
    let head_radius_sq = head_radius * head_radius;

    for y in 0..height as usize {
        let row = &mut data[y * stride..y * stride + width as usize * 4];
        for (x, pixel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (xi, yi) = (x as i64, y as i64);
            let in_body =
                (body_left..body_right).contains(&xi) && (body_top..body_bottom).contains(&yi);
            let dx = xi - head_center_x;
            let dy = yi - head_center_y;
            let in_head = dx * dx + dy * dy <= head_radius_sq;
            *pixel = if in_body || in_head { figure } else { green };
        }
    }
}

/// Stands in for a real external producer (a camera SDK's callback, say):
/// pushes a green-screen `BGRA` frame into `handle` at a nominal 15fps
/// until `running` goes false, then pushes `Eos`.
fn spawn_green_screen_feeder(
    handle: AppSourceHandle,
    width: u32,
    height: u32,
    running: Arc<AtomicBool>,
) -> thread::JoinHandle<media_pp::Result<()>> {
    thread::spawn(move || -> media_pp::Result<()> {
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        );
        let frame_interval = Duration::from_secs_f64(1.0 / 15.0);
        let mut next_due = Instant::now();
        let mut index: i64 = 0;
        while running.load(Ordering::Relaxed) {
            thread::sleep(next_due.saturating_duration_since(Instant::now()));
            let mut frame = pool.get();
            fill_green_screen_frame(&mut frame, width, height);
            frame.set_pts(Some(index));
            handle.push(MediaBuffer::Video(Arc::new(frame)))?;

            index += 1;
            next_due += frame_interval;
            let now = Instant::now();
            if next_due < now {
                next_due = now + frame_interval;
            }
        }
        handle.push(MediaBuffer::Eos)
    })
}
