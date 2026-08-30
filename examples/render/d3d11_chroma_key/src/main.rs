//! A green screen keyed out on the GPU and composited over live video, with
//! the frame never leaving video memory between the upload and the recording
//! branch's download.
//!
//! Three pipelines meet at one compositor:
//!
//!     AppSource(BGRA) -> D3d11Upload -> D3d11ChromaKey -> "keyed" layer
//!     TestVideoSource -> SwScaler(NV12) -> D3d11Upload -> "background" layer
//!     D3d11VideoCompositor -> D3d11Download -> SwScaler(YUV420P)
//!         -> SwEncoder -> FileMuxer
//!
//! `AppSource` stands in for a real external producer — a camera or capture
//! SDK's callback — handing over `BGRA` frames of a figure on a green
//! backdrop. `D3d11Upload` puts those on the GPU as BGRA rather than NV12,
//! which is what keeps the backdrop exactly the color
//! `ChromaKeyMethod::Green` keys: a YUV round trip would quantize it and
//! leave the threshold covering for the drift.
//!
//! `D3d11ChromaKey` writes that green into alpha, and the compositor blends
//! the result over its background layer. The keyed layer walks across the
//! canvas as it goes, so the recording shows the background passing behind a
//! figure with no green around it — where, without the key, an opaque green
//! rectangle would cover the background instead.
//!
//! `video_compositor` is the same graph on the CPU, with `SwChromaKey` and
//! `SwVideoCompositor`.
//!
//!     cargo run -p d3d11_chroma_key -- [output.mp4] [seconds]

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!(
        "{} supports Windows (D3D11) only; see `video_compositor` for the CPU backend",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
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
            AppSource, AppSourceHandle, ChromaKeyMethod, ChromaKeyOptions, D3d11ChromaKey,
            D3d11Download, D3d11Upload, D3d11VideoCompositor, FileMuxer, SwEncoder,
            SwEncoderOptions, SwScaler, TestVideoOptions, TestVideoSource, VideoCodec,
            VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
        },
        pipeline::Pipeline,
        pool::UnboundObjectPool,
    };
    use render_common::D3d11GpuContext;

    const CANVAS_WIDTH: u32 = 640;
    const CANVAS_HEIGHT: u32 = 360;
    /// The green-screen shot's own size, and therefore the keyed layer's.
    /// Smaller than the canvas so the background stays visible around it.
    const SHOT_WIDTH: u32 = 256;
    const SHOT_HEIGHT: u32 = 192;

    pub fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "d3d11_chroma_key.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);

        // One device and one shared immediate context for every D3D11 stage —
        // both uploads, the key, the compositor, and the download. Each of
        // them rejects a texture that came from a different device.
        let gpu =
            D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;
        let frame_rate = ffmpeg::Rational::new(30, 1);

        let (compositor, compositor_handle) = D3d11VideoCompositor::new(
            "compositor",
            gpu.device(),
            gpu.context(),
            VideoCompositorOptions {
                width: CANVAS_WIDTH,
                height: CANVAS_HEIGHT,
                frame_rate,
                background: Color::new(24, 24, 24),
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let time_base = compositor.time_base();

        let mut background_layer =
            VideoLayer::new(VideoRect::new(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT));
        background_layer.fit = VideoFit::Cover;
        let background_sink = compositor_handle
            .add_source("background", background_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?
            .expect("the compositor is alive")
            .sink;

        let mut keyed_layer = VideoLayer::new(VideoRect::new(
            0,
            (CANVAS_HEIGHT as i32 - SHOT_HEIGHT as i32) / 2,
            SHOT_WIDTH,
            SHOT_HEIGHT,
        ));
        keyed_layer.z_index = 1;
        let keyed_input = compositor_handle
            .add_source("keyed", keyed_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?
            .expect("the compositor is alive");
        let keyed_sink = keyed_input.sink;
        let keyed_handle = keyed_input.layer;

        // The green screen, keyed without a single format conversion:
        // AppSource hands over BGRA, D3d11Upload puts BGRA on the GPU, and
        // the key and the compositor both work in BGRA from there.
        let (green_screen, green_screen_handle) = AppSource::new("green-screen", 8);
        let keyed_pipeline = Pipeline::new("keyed-foreground", green_screen, |source, ctx| {
            let upload = D3d11Upload::new("upload", gpu.device(), SHOT_WIDTH, SHOT_HEIGHT);
            let key = D3d11ChromaKey::new(
                "key",
                gpu.device(),
                gpu.context(),
                ChromaKeyOptions {
                    method: ChromaKeyMethod::Green,
                    // The backdrop is exactly the key color here, so the
                    // threshold only has to cover the feathered edge the
                    // smoothing band creates around the figure.
                    threshold: 0.15,
                    smoothing: 0.1,
                },
            )
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            let branch = ctx.branch().pipe(upload).pipe(key).to(keyed_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // What the keyed figure is composited over. A compositor input has
        // to be GPU-resident, and this one comes from a decoder-shaped
        // source, so it takes the NV12 route through the same element.
        let background_pipeline = Pipeline::new(
            "background-feed",
            TestVideoSource::new(
                "background-source",
                TestVideoOptions {
                    width: CANVAS_WIDTH,
                    height: CANVAS_HEIGHT,
                    framerate: ffmpeg::Rational::new(15, 1),
                },
            ),
            |source, ctx| {
                let scaler = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    CANVAS_WIDTH,
                    CANVAS_HEIGHT,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload = D3d11Upload::new("upload", gpu.device(), CANVAS_WIDTH, CANVAS_HEIGHT);
                let branch = ctx.branch().pipe(scaler).pipe(upload).to(background_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            },
        )?;

        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: CANVAS_WIDTH,
                height: CANVAS_HEIGHT,
                time_base,
                frame_rate,
                bit_rate: 2_000_000,
                gop_size: 60,
            },
        )?;
        let mut muxer = FileMuxer::create(&path)?;
        muxer.add_stream("video", encoder.parameters(), time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let record_pipeline = Pipeline::new("record", compositor, |source, ctx| {
            let download = D3d11Download::new(
                "download",
                gpu.device(),
                gpu.context(),
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
            )
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            let to_yuv = SwScaler::new(
                "to-yuv",
                ffmpeg::format::Pixel::YUV420P,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
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

        println!("recording {seconds}s of {CANVAS_WIDTH}x{CANVAS_HEIGHT} to {path} ...");
        record_pipeline.run()?;
        background_pipeline.run()?;
        keyed_pipeline.run()?;

        let feeding = Arc::new(AtomicBool::new(true));
        let feeder = spawn_green_screen_feeder(green_screen_handle, feeding.clone());

        // Walk the keyed layer across the canvas. The figure inside the shot
        // stays put, so the only thing that moves is where the keyed layer
        // sits — and the background is visible right up to the figure's edge
        // the whole way.
        let steps = seconds.saturating_mul(30);
        let travel = CANVAS_WIDTH - SHOT_WIDTH;
        let top = (CANVAS_HEIGHT as i32 - SHOT_HEIGHT as i32) / 2;
        for step in 0..steps {
            let x = if steps <= 1 {
                0
            } else {
                (u64::from(travel) * step / (steps - 1)) as i32
            };
            keyed_handle
                .set_rect(VideoRect::new(x, top, SHOT_WIDTH, SHOT_HEIGHT))
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            thread::sleep(Duration::from_millis(33));
        }

        // The feeder pushes its own `Eos` while the pipeline is still fully
        // running, so nothing races a final `push` against a torn-down graph.
        feeding.store(false, Ordering::Relaxed);
        feeder
            .join()
            .expect("green-screen feeder thread panicked")?;

        // Drained in flow order, not abandoned: `finish` puts Eos behind each
        // source's already-produced frames, so the encoder flushes its delayed
        // packets and the muxer finalizes a playable file.
        keyed_pipeline.finish();
        background_pipeline.finish();
        record_pipeline.finish();

        let mut failed = false;
        for pipeline in [&keyed_pipeline, &background_pipeline, &record_pipeline] {
            for event in pipeline.bus().iter() {
                if let BusEvent::Error { name, error, .. } = event {
                    eprintln!("[{name}] error: {error}");
                    failed = true;
                }
            }
        }
        if failed {
            return Err(media_pp::Error::Other(
                "one or more elements reported an error; see the messages above".into(),
            ));
        }

        println!("wrote {path}");
        Ok(())
    }

    /// Draws a green backdrop with a static, differently-colored "figure" —
    /// a circular head over a rectangular body, the simplest shape that
    /// reads as a person rather than an arbitrary block. Everything that is
    /// not the figure is exactly `ChromaKeyMethod::Green`, so what survives
    /// the key is exactly the figure.
    fn fill_green_screen_frame(frame: &mut ffmpeg::frame::Video) {
        let (width, height) = (SHOT_WIDTH, SHOT_HEIGHT);
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
    /// pushes a green-screen `BGRA` frame into `handle` at a nominal 30fps
    /// until `feeding` goes false, then pushes `Eos`.
    fn spawn_green_screen_feeder(
        handle: AppSourceHandle,
        feeding: Arc<AtomicBool>,
    ) -> thread::JoinHandle<media_pp::Result<()>> {
        thread::spawn(move || -> media_pp::Result<()> {
            let pool = UnboundObjectPool::new(
                0,
                || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, SHOT_WIDTH, SHOT_HEIGHT),
                |_| {},
            );
            let frame_interval = Duration::from_secs_f64(1.0 / 30.0);
            let mut next_due = Instant::now();
            let mut index: i64 = 0;
            while feeding.load(Ordering::Relaxed) {
                thread::sleep(next_due.saturating_duration_since(Instant::now()));
                let mut frame = pool.get();
                fill_green_screen_frame(&mut frame);
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
}
