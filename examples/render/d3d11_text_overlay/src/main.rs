//! A moving-gradient background composited with a live clock in front of it,
//! recorded to an mp4 — proof that dynamic text (not a static watermark)
//! really updates: the overlaid text changes once a second while recording,
//! so the output's frames differ over time only if `set_text` is actually
//! re-rasterizing and re-uploading each call.
//!
//! `cuda_text_overlay` is the same graph on the CUDA backend, in its own crate
//! because CUDA is a vendor backend rather than a platform one and runs on
//! Windows too.
//!
//!     cargo run -p d3d11_text_overlay -- [output.mp4] [seconds]
//!     (use the arrow keys to move the text, or `q` to stop early)

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!(
        "{} supports Windows (D3D11) only; see `cuda_text_overlay` for the CUDA backend",
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
        sync::mpsc::{self, RecvTimeoutError},
        thread,
        time::{Duration, Instant},
    };

    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        color::Color,
        elements::{
            D3d11Download, D3d11Upload, D3d11VideoCompositor, FileMuxer, SwEncoder,
            SwEncoderOptions, SwScaler, TestVideoOptions, TestVideoSource, TextLayer, VideoCodec,
            VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
        },
        pipeline::Pipeline,
    };
    use render_common::D3d11GpuContext;
    use windows::Win32::System::Console::{
        GetStdHandle, INPUT_RECORD, KEY_EVENT, ReadConsoleInputW, STD_INPUT_HANDLE,
    };

    const MOVE_STEP: i32 = 10;

    enum TerminalCommand {
        Move { dx: i32, dy: i32 },
        Quit,
    }

    /// A moving-gradient `TestVideoSource` background composited with a
    /// `D3d11TextLayerHandle` clock in front of it, recorded to an mp4 — proves dynamic
    /// text (not just a static watermark) actually updates on screen: the
    /// overlaid text changes once a second while the recording runs, so the
    /// output file's frames differ over time if `D3d11TextLayerHandle::set_text` is
    /// really re-rasterizing and re-uploading each call.
    ///
    ///     cargo run -p d3d11_text_overlay -- [output.mp4] [seconds]
    ///     (use the arrow keys to move the text, or `q` to stop early)
    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "d3d11_text_overlay.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);

        let gpu =
            D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

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

        let mut background_layer =
            VideoLayer::new(VideoRect::new(0, 0, output_width, output_height));
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
                let scaler = SwScaler::new(
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
                max_b_frames: None,
            },
        )?;
        let mut muxer = FileMuxer::create(&path)?;
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
        let commands = terminal_commands();
        let started = Instant::now();
        let duration = Duration::from_secs(seconds);
        let mut next_text_update = Duration::from_secs(1);
        let (mut text_x, mut text_y) = (20, 20);
        let mut terminal_connected = true;

        while started.elapsed() < duration {
            let elapsed = started.elapsed();
            let wait_for_tick = next_text_update.saturating_sub(elapsed);
            let wait_for_end = duration.saturating_sub(elapsed);

            let wait = wait_for_tick.min(wait_for_end);
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
                    overlay
                        .set_position(text_x, text_y)
                        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
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

    fn terminal_commands() -> mpsc::Receiver<TerminalCommand> {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let input = match unsafe { GetStdHandle(STD_INPUT_HANDLE) } {
                Ok(input) => input,
                Err(error) => {
                    eprintln!("terminal controls unavailable: {error}");
                    return;
                }
            };

            loop {
                let mut record = INPUT_RECORD::default();
                let mut read = 0;
                if let Err(error) = unsafe {
                    ReadConsoleInputW(input, std::slice::from_mut(&mut record), &mut read)
                } {
                    // stdin may be redirected or detached in CI. Recording still
                    // follows its requested duration; only live controls are absent.
                    let _ = error;
                    return;
                }
                if read == 0 || u32::from(record.EventType) != KEY_EVENT {
                    continue;
                }

                // `EventType == KEY_EVENT` makes the matching union field active.
                let key = unsafe { record.Event.KeyEvent };
                if !key.bKeyDown.as_bool() {
                    continue;
                }

                let command = match key.wVirtualKeyCode {
                    0x25 => Some(TerminalCommand::Move {
                        dx: -MOVE_STEP,
                        dy: 0,
                    }),
                    0x26 => Some(TerminalCommand::Move {
                        dx: 0,
                        dy: -MOVE_STEP,
                    }),
                    0x27 => Some(TerminalCommand::Move {
                        dx: MOVE_STEP,
                        dy: 0,
                    }),
                    0x28 => Some(TerminalCommand::Move {
                        dx: 0,
                        dy: MOVE_STEP,
                    }),
                    0x51 => Some(TerminalCommand::Quit),
                    _ => None,
                };

                if command.is_some_and(|command| sender.send(command).is_err()) {
                    return;
                }
            }
        });
        receiver
    }
}
