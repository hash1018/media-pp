#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example supports Windows (DXGI) and Linux (PipeWire)",
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
    use std::{thread, time::Duration};

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            CaptureMode, DxgiCaptureOptions, DxgiCaptureSource, Mp4Muxer, SwEncoder,
            SwEncoderOptions, SwScaler, VideoCodec,
        },
        pipeline::Pipeline,
    };

    /// DxgiCaptureSource -> SwScaler -> SwEncoder -> Mp4Muxer: captures the
    /// desktop live via DXGI Desktop Duplication and encodes it straight into
    /// a playable `.mp4` file — no window, no renderer, just a headless
    /// recording (compare `screen_capture`, which renders instead of encoding).
    ///
    /// `DxgiCaptureSource` never reaches `Eos` on its own (see its own docs);
    /// this just captures for a fixed duration and then `pipeline.stop()`s,
    /// which is also what finalizes the MP4's trailer — `Mp4Muxer` writes it on
    /// `Stop` as well as `Eos`, unlike `RtspSink`, since an MP4 file needs a
    /// valid trailer to be playable at all (see `Mp4Muxer`'s own docs).
    ///
    ///     cargo run -p screen_record -- [output.mp4] [seconds]
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
            .unwrap_or_else(|| "screen_record.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        let capture_options = DxgiCaptureOptions {
            fps: 30,
            capture_mode: CaptureMode::Cpu {
                include_cursor: true,
            },
            ..DxgiCaptureOptions::default()
        };
        let (source, format, _device) = DxgiCaptureSource::open("screen", capture_options)?;

        let encoder = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: format.width,
                height: format.height,
                time_base: format.time_base,
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
        muxer.add_stream("video", encoder.parameters(), format.time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("screen-record", source, |source, ctx| {
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
        pipeline.run();

        thread::sleep(Duration::from_secs(seconds));
        pipeline.stop();

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
        }

        println!("wrote {path}");
        Ok(())
    }
}

/// The Linux half of the same example. Deliberately the same pipeline as
/// `windows_example` — capture -> Queue -> SwScaler -> Queue -> SwEncoder ->
/// Mp4Muxer, same codec, same terminus — so only the capture source differs.
///
/// The one CLI difference is forced by the platform: Wayland has no way to
/// name a monitor, so the compositor prompts on the first run and hands back a
/// restore token that skips the prompt next time. See
/// `PipeWireScreenCaptureSource`'s own docs for why that is not something this
/// example can paper over.
#[cfg(target_os = "linux")]
mod linux_example {
    use std::{thread, time::Duration};

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            CaptureSourceKind, Mp4Muxer, PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource,
            SwEncoder, SwEncoderOptions, SwScaler, VideoCodec,
        },
        pipeline::Pipeline,
    };

    /// PipeWireScreenCaptureSource -> SwScaler -> SwEncoder -> Mp4Muxer: captures
    /// the desktop live through xdg-desktop-portal and encodes it straight into
    /// a playable `.mp4` file — no window, no renderer, just a headless
    /// recording.
    ///
    /// `PipeWireScreenCaptureSource` never reaches `Eos` on its own; like the
    /// Windows path this captures for a fixed duration and then
    /// `pipeline.stop()`s, which is also what finalizes the MP4's trailer.
    ///
    ///     cargo run -p screen_record -- [output.mp4] [seconds] [monitor|window] [restore-token]
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
            .unwrap_or_else(|| "screen_record.mp4".into());
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
        let (source, capture_format, restore_token) = PipeWireScreenCaptureSource::open(
            "screen",
            PipeWireScreenCaptureOptions {
                fps: 30,
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
                time_base: capture_format.time_base,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 4_000_000,
                gop_size: 60, // ~2s @ 30fps
            },
        )
        .expect("failed to open encoder");
        let mut muxer = Mp4Muxer::create(&path)?;
        muxer.add_stream("video", encoder.parameters(), capture_format.time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("screen-record", source, |source, ctx| {
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
        pipeline.run();

        thread::sleep(Duration::from_secs(seconds));
        pipeline.stop();

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
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
