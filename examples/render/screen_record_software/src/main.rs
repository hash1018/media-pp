//! Records the desktop through the software-encoding path:
//! `capture -> Queue -> SwScaler -> Queue -> SwEncoder -> FileMuxer`.
//! Windows captures through DXGI; Linux uses PipeWire and the desktop portal;
//! macOS uses ScreenCaptureKit. All run for a fixed duration and finish the
//! pipeline to finalize the MP4.
//!
//! ```text
//! cargo run -p screen_record_software -- [output.mp4] [seconds]
//! cargo run -p screen_record_software -- [output.mp4] [seconds] [monitor|window] [restore-token] # Linux
//! cargo run -p screen_record_software -- [output.mp4] [seconds] [monitor|window] # macOS
//! ```

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn main() {
    eprintln!(
        "{} example supports Windows (DXGI), Linux (PipeWire) and macOS (ScreenCaptureKit)",
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

#[cfg(target_os = "macos")]
fn main() -> impl std::process::Termination {
    macos_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{thread, time::Duration};

    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            CaptureMode, DxgiCaptureOptions, DxgiCaptureSource, FileMuxer, SwEncoder,
            SwEncoderOptions, SwScaler, VideoCodec,
        },
        pipeline::Pipeline,
    };

    /// DxgiCaptureSource -> SwScaler -> SwEncoder -> FileMuxer: captures the
    /// desktop live via DXGI Desktop Duplication and encodes it straight into
    /// a playable `.mp4` file — no window, no renderer, just a headless
    /// recording (compare `screen_preview_cpu`, which renders
    /// instead of encoding).
    ///
    /// `DxgiCaptureSource` never reaches `Eos` on its own (see its own docs);
    /// this just captures for a fixed duration and then `pipeline.finish()`es:
    /// the capture places an `Eos` behind its last frame, the encoder flushes
    /// what it still holds, and the muxer writes the MP4's trailer after it.
    /// `stop()` would finalize a playable file too, but abandon those last
    /// frames.
    ///
    ///     cargo run -p screen_record_software -- [output.mp4] [seconds]
    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "screen_record_software.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        let frame_rate = ffmpeg::Rational::new(30, 1);
        let capture_options = DxgiCaptureOptions {
            frame_rate,
            capture_mode: CaptureMode::Cpu {
                include_cursor: true,
            },
            ..DxgiCaptureOptions::default()
        };
        let (source, format, _gpu) = DxgiCaptureSource::open("screen", capture_options)?;

        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: format.width,
                height: format.height,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate,
                bit_rate: 4_000_000,
                gop_size: 60, // ~2s @ 30fps
                max_b_frames: None,
            },
        )?;
        // No container/demuxer in this loop to get these from — SwEncoder
        // exposes its own codec parameters for exactly this case (see
        // `transcode_render`'s own use of this, wiring a decoder instead).
        let mut muxer = FileMuxer::create(&path)?;
        let track = muxer.add_stream("video", &encoder)?;
        let muxer_sink = muxer.open()?.take(track)?;

        let (pipeline, ()) = Pipeline::new("screen-record-software", source, |source, ctx| {
            let scaler = SwScaler::new(
                "to-yuv",
                ffmpeg::format::Pixel::YUV420P,
                format.width,
                format.height,
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
        pipeline.run()?;

        thread::sleep(Duration::from_secs(seconds));
        pipeline.finish();

        for event in pipeline.bus().iter() {
            println!("{event}");
        }

        println!("wrote {path}");
        Ok(())
    }
}

/// The Linux half of the same example. Deliberately the same pipeline as
/// `windows_example` — capture -> Queue -> SwScaler -> Queue -> SwEncoder ->
/// FileMuxer, same codec, same terminus — so only the capture source differs.
///
/// The one CLI difference is forced by the platform: Wayland has no way to
/// name a monitor, so the compositor prompts on the first run and hands back a
/// restore token that skips the prompt next time. See
/// `PipeWireScreenCaptureSource`'s own docs for why that is not something this
/// example can paper over.
#[cfg(target_os = "linux")]
mod linux_example {
    use std::{thread, time::Duration};

    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            CaptureSourceKind, FileMuxer, PipeWireScreenCaptureOptions,
            PipeWireScreenCaptureSource, SwEncoder, SwEncoderOptions, SwScaler, VideoCodec,
        },
        pipeline::Pipeline,
    };

    /// PipeWireScreenCaptureSource -> SwScaler -> SwEncoder -> FileMuxer: captures
    /// the desktop live through xdg-desktop-portal and encodes it straight into
    /// a playable `.mp4` file — no window, no renderer, just a headless
    /// recording.
    ///
    /// `PipeWireScreenCaptureSource` never reaches `Eos` on its own; like the
    /// Windows path this captures for a fixed duration and then
    /// `pipeline.finish()`es, draining the encoder into the MP4's trailer.
    ///
    ///     cargo run -p screen_record_software -- [output.mp4] [seconds] [monitor|window] [restore-token]
    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "screen_record_software.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        // Monitor by default, matching the Windows branch's whole-desktop
        // capture. `window` is worth reaching for when one application is the
        // subject: a monitor stream stalls while any client is fullscreen,
        // where a window stream does not — see `PipeWireScreenCaptureSource`.
        let source_kind = match std::env::args().nth(3).as_deref() {
            Some("window") => CaptureSourceKind::Window,
            _ => CaptureSourceKind::Monitor,
        };
        // Last so it can simply be left off, unlike the token-shaped argument
        // it replaces: it is a long opaque string that only a repeat run has.
        let restore_token = std::env::args().nth(4);

        if restore_token.is_none() {
            eprintln!("opening the portal — approve the screen-share dialog to continue...");
        }
        let frame_rate = ffmpeg::Rational::new(30, 1);
        let (source, capture_format, restore_token) = PipeWireScreenCaptureSource::open(
            "screen",
            PipeWireScreenCaptureOptions {
                frame_rate,
                source_kind,
                include_cursor: true,
                restore_token,
            },
        )?;

        // H.264 needs even dimensions. A monitor is even in practice, but the
        // portal's picker can hand back a window of any size at all, which the
        // DXGI path never has to consider.
        let (width, height) = (capture_format.width & !1, capture_format.height & !1);

        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width,
                height,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate,
                bit_rate: 4_000_000,
                gop_size: 60, // ~2s @ 30fps
                max_b_frames: None,
            },
        )?;
        let mut muxer = FileMuxer::create(&path)?;
        let track = muxer.add_stream("video", &encoder)?;
        let muxer_sink = muxer.open()?.take(track)?;

        let (pipeline, ()) = Pipeline::new("screen-record-software", source, |source, ctx| {
            let scaler = SwScaler::new(
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

        println!("recording {seconds}s of {width}x{height} desktop to {path} ...");
        pipeline.run()?;

        thread::sleep(Duration::from_secs(seconds));
        pipeline.finish();

        for event in pipeline.bus().iter() {
            println!("{event}");
        }

        println!("wrote {path}");
        match restore_token {
            Some(token) => println!(
                "re-run without a dialog:\n  ... {path} {seconds} {} {token}",
                if matches!(source_kind, CaptureSourceKind::Window) {
                    "window"
                } else {
                    "monitor"
                }
            ),
            None => println!("the compositor issued no restore token; the next run will prompt"),
        }
        Ok(())
    }
}

/// The macOS half of the same example: the same pipeline again, the capture
/// source the only difference.
///
/// A display is named by ID and a window by its own, so nothing here prompts
/// for a choice; `window` takes the frontmost window with a title. What does prompt
/// is macOS itself, once: recording the screen needs a permission given in
/// System Settings to the terminal this runs in, after which it has to be
/// started again.
#[cfg(target_os = "macos")]
mod macos_example {
    use std::{thread, time::Duration};

    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            FileMuxer, ScreenCaptureKitOptions, ScreenCaptureKitSource, ScreenCaptureKitTarget,
            SwEncoder, SwEncoderOptions, SwScaler, VideoCodec,
        },
        pipeline::Pipeline,
    };

    /// ScreenCaptureKitSource -> SwScaler -> SwEncoder -> FileMuxer: captures
    /// the main display, or the frontmost titled window, and encodes it straight
    /// into a playable `.mp4` file — no window, no renderer, just a headless
    /// recording.
    ///
    ///     cargo run -p screen_record_software -- [output.mp4] [seconds] [monitor|window]
    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "screen_record_software.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let target = match std::env::args().nth(3).as_deref() {
            Some("window") => {
                // An application's untitled windows are toolbars and strips
                // more often than documents.
                let window = ScreenCaptureKitSource::list_windows()?
                    .into_iter()
                    .find(|window| !window.title.is_empty())
                    .ok_or_else(|| {
                        media_pp::Error::Other("there is no window to capture".into())
                    })?;
                println!("capturing {:?} of {}", window.title, window.application);
                ScreenCaptureKitTarget::Window(window.id)
            }
            _ => {
                let display = ScreenCaptureKitSource::list_displays()?
                    .into_iter()
                    .find(|display| display.is_main)
                    .ok_or_else(|| media_pp::Error::Other("there is no main display".into()))?;
                ScreenCaptureKitTarget::Display(display.id)
            }
        };

        let frame_rate = ffmpeg::Rational::new(30, 1);
        let (source, capture_format) = ScreenCaptureKitSource::open(
            "screen",
            ScreenCaptureKitOptions {
                frame_rate,
                include_cursor: true,
                ..ScreenCaptureKitOptions::new(target)
            },
        )?;

        // H.264 needs even dimensions, and a window can be any size at all.
        let (width, height) = (capture_format.width & !1, capture_format.height & !1);

        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width,
                height,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate,
                bit_rate: 4_000_000,
                gop_size: 60, // ~2s @ 30fps
                max_b_frames: None,
            },
        )?;
        let mut muxer = FileMuxer::create(&path)?;
        let track = muxer.add_stream("video", &encoder)?;
        let muxer_sink = muxer.open()?.take(track)?;

        let (pipeline, ()) = Pipeline::new("screen-record-software", source, |source, ctx| {
            let scaler = SwScaler::new(
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

        println!("recording {seconds}s of {width}x{height} to {path} ...");
        pipeline.run()?;

        thread::sleep(Duration::from_secs(seconds));
        pipeline.finish();

        for event in pipeline.bus().iter() {
            println!("{event}");
        }

        println!("wrote {path}");
        Ok(())
    }
}
