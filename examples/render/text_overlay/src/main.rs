//! A moving-gradient background composited with a live clock in front of it,
//! recorded to an mp4 — proof that dynamic text (not a static watermark)
//! really updates: the overlaid text changes once a second while recording,
//! so the output's frames differ over time only if `set_text` is actually
//! re-rasterizing and re-uploading each call.
//!
//! Both platforms run the identical graph and CLI; only the GPU stack and the
//! terminal API differ — `D3d11Upload`/`D3d11VideoCompositor`/`D3d11Download`
//! with a Win32 console on Windows, `CudaUpload`/`CudaVideoCompositor`/
//! `CudaDownload` with a POSIX termios terminal on Linux.
//!
//!     cargo run -p text_overlay -- [output.mp4] [seconds]
//!     (use the arrow keys to move the text, or `q` to stop early)

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} supports Windows (D3D11) and Linux (CUDA) only",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{
        sync::mpsc::{self, RecvTimeoutError},
        thread,
        time::{Duration, Instant},
    };

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        color::Color,
        elements::{
            D3d11Download, D3d11Upload, D3d11VideoCompositor, Mp4Muxer, SwEncoder,
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
    ///     cargo run -p text_overlay -- [output.mp4] [seconds]
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
            .unwrap_or_else(|| "text_overlay.mp4".into());
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

        output_pipeline.run();
        background_pipeline.run();

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

/// The Linux half of the same example: the same graph, the same clock, the
/// same controls. Only the GPU stack and the terminal API differ — a text
/// layer is drawn by `CudaVideoCompositor`'s blend kernel rather than a D3D11
/// blend state, and raw keys come from termios rather than the Win32 console.
#[cfg(target_os = "linux")]
mod linux_example {
    use std::{
        io::Read,
        sync::mpsc::{self, RecvTimeoutError},
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

    const MOVE_STEP: i32 = 10;

    /// Fonts this crate does not bundle. The first one present wins; a system
    /// with none of them gets a clear error rather than an empty overlay.
    const FONT_CANDIDATES: [&str; 4] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ];

    enum TerminalCommand {
        Move { dx: i32, dy: i32 },
        Quit,
    }

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
            .unwrap_or_else(|| "text_overlay.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);

        // One CUDA context for the whole stack: the upload allocates on it,
        // the compositor draws on it, and the download reads from it.
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

        let mut background_layer =
            VideoLayer::new(VideoRect::new(0, 0, output_width, output_height));
        background_layer.fit = VideoFit::Cover;
        let background_input = compositor_handle
            .add_source("background", background_layer)
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let background_sink = background_input.sink;

        // The text layer never receives `Pipeline` frames — no `Sink` to wire
        // up, just a handle driven directly by `set_text`. `add_text_layer`
        // takes a `TextLayer` the same way `add_source` takes a `VideoLayer`.
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

        output_pipeline.run();
        background_pipeline.run();

        println!("controls: arrow keys move the text by {MOVE_STEP}px; q stops recording");
        let commands = terminal_commands();
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

    /// Arrow keys and `q`, read from a raw-mode terminal.
    ///
    /// Raw mode is what makes a single keypress arrive without Enter, the
    /// same thing `ReadConsoleInputW` gives the Windows branch for free. The
    /// terminal's original settings are restored when the reader thread
    /// exits; a redirected or absent stdin simply yields no commands, and the
    /// recording still follows its requested duration.
    fn terminal_commands() -> mpsc::Receiver<TerminalCommand> {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let Some(restore) = raw_mode() else {
                eprintln!("terminal controls unavailable: stdin is not a terminal");
                return;
            };
            let mut stdin = std::io::stdin();
            let mut buffer = [0u8; 3];
            while let Ok(read) = stdin.read(&mut buffer[..1]) {
                if read == 0 {
                    break;
                }
                let command = match buffer[0] {
                    b'q' | b'Q' => Some(TerminalCommand::Quit),
                    0x1b => {
                        // CSI sequence: ESC '[' followed by one final byte.
                        if stdin.read(&mut buffer[1..3]).unwrap_or(0) < 2 || buffer[1] != b'[' {
                            None
                        } else {
                            match buffer[2] {
                                b'A' => Some(TerminalCommand::Move {
                                    dx: 0,
                                    dy: -MOVE_STEP,
                                }),
                                b'B' => Some(TerminalCommand::Move {
                                    dx: 0,
                                    dy: MOVE_STEP,
                                }),
                                b'C' => Some(TerminalCommand::Move {
                                    dx: MOVE_STEP,
                                    dy: 0,
                                }),
                                b'D' => Some(TerminalCommand::Move {
                                    dx: -MOVE_STEP,
                                    dy: 0,
                                }),
                                _ => None,
                            }
                        }
                    }
                    _ => None,
                };
                if let Some(command) = command
                    && sender.send(command).is_err()
                {
                    break;
                }
            }
            restore();
        });
        receiver
    }

    /// Puts stdin in raw mode, returning what restores it. `None` when stdin
    /// is not a terminal at all.
    fn raw_mode() -> Option<impl FnOnce()> {
        use std::os::fd::AsRawFd;

        let fd = std::io::stdin().as_raw_fd();
        let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
        if unsafe { libc::tcgetattr(fd, &mut original) } != 0 {
            return None;
        }
        let mut raw = original;
        unsafe { libc::cfmakeraw(&mut raw) };
        // Leave output processing alone: this example keeps printing lines,
        // and full raw mode would strip their carriage returns.
        raw.c_oflag = original.c_oflag;
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
            return None;
        }
        Some(move || {
            unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original) };
        })
    }
}
