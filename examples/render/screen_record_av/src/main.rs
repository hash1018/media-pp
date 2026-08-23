//! Records desktop video and system audio into one MP4 from two independent
//! live capture sources sharing one `PipelineBuilder` pipeline.
//! Windows uses DXGI + WASAPI; Linux uses PipeWire screen capture and a
//! PipeWire sink monitor. Enter `q` in the terminal to stop both sources and
//! finalize the file.
//!
//! ```text
//! cargo run -p screen_record_av -- [output.mp4]
//! cargo run -p screen_record_av -- [output.mp4] [monitor|window] [restore-token] # Linux
//! ```

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example supports Windows (DXGI + WASAPI) and Linux (PipeWire)",
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
        io::{self, BufRead},
        thread,
    };

    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            AudioCodec, CaptureMode, DxgiCaptureOptions, DxgiCaptureSource, Mp4Muxer,
            SwAudioEncoder, SwAudioEncoderOptions, SwEncoder, SwEncoderOptions, SwScaler,
            VideoCodec, WasapiCaptureOptions, WasapiCaptureSource, WasapiDeviceKind,
        },
        pipeline::PipelineBuilder,
    };

    /// DxgiCaptureSource + WasapiCaptureSource (system-audio loopback — whatever
    /// the default playback device is putting out, i.e. "PC 소리") -> one
    /// Mp4Muxer: records the desktop and its system audio together into a
    /// single playable `.mp4`. Two independent live sources sharing one
    /// `Pipeline` via `PipelineBuilder` (see its own docs) — each on its own
    /// thread, but one `pipeline.stop()` reaches both.
    ///
    /// Neither capture source ever reaches a natural `Eos` (same as
    /// `screen_record_software`'s own docs) — this runs until `q` + Enter in the same
    /// terminal, which is also what finalizes the MP4's trailer (`Mp4Muxer`
    /// writes it once *every* track — video and audio both — reports done via
    /// `Eos` *or* `Stop`, not on whichever finishes first; see `Mp4Muxer::open`'s
    /// own docs, and `PipelineBuilder`'s for why one `stop()` call is enough to
    /// reach both tracks even though they're two independent sources).
    ///
    ///     cargo run -p screen_record_av -- [output.mp4]
    ///     (then in the same terminal: `q` + Enter to stop and finalize)
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
            .unwrap_or_else(|| "screen_record_av.mp4".into());

        let capture_options = DxgiCaptureOptions {
            fps: 30,
            capture_mode: CaptureMode::Cpu {
                include_cursor: true,
            },
            ..DxgiCaptureOptions::default()
        };
        let (video_source, video_format, _device) =
            DxgiCaptureSource::open("screen", capture_options)?;
        let video_time_base = video_format.time_base;

        let devices = WasapiCaptureSource::list_devices()
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let device = devices
            .into_iter()
            .find(|d| d.kind == WasapiDeviceKind::Render && d.is_default)
            .ok_or_else(|| media_pp::Error::Other("no default playback device found".into()))?;
        println!("capturing system audio from: {}", device.name);
        let (audio_source, audio_format) =
            WasapiCaptureSource::open("system-audio", WasapiCaptureOptions { device })
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let audio_time_base = audio_source.time_base();

        let video_encoder = SwEncoder::new(
            "video-encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: video_format.width,
                height: video_format.height,
                time_base: video_time_base,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 4_000_000,
                gop_size: 60, // ~2s @ 30fps
            },
        )
        .expect("failed to open video encoder");
        let audio_encoder = SwAudioEncoder::new(
            "audio-encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: audio_format.sample_rate,
                channels: audio_format.channels,
                time_base: audio_time_base,
                bit_rate: 128_000,
            },
        )
        .expect("failed to open audio encoder");

        // No container/demuxer in this loop to get these from — each encoder
        // exposes its own codec parameters for exactly this case.
        let mut muxer = Mp4Muxer::create(&path)?;
        muxer.add_stream("video", video_encoder.parameters(), video_time_base)?;
        muxer.add_stream("audio", audio_encoder.parameters(), audio_time_base)?;
        let mut sinks = muxer.open()?;
        let audio_sink = sinks.pop().expect("two streams were added");
        let video_sink = sinks.pop().expect("two streams were added");

        let pipeline = PipelineBuilder::new("screen-record-av")
            .add_source(video_source, |source, ctx| {
                let scaler = SwScaler::new(
                    "to-yuv",
                    ffmpeg::format::Pixel::YUV420P,
                    video_format.width,
                    video_format.height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let branch = ctx
                    .branch()
                    .queue("captured", 4) // thread boundary so scaling doesn't block capture
                    .pipe(scaler)
                    .queue("frames", 8) // thread boundary so encoding doesn't block scaling
                    .pipe(video_encoder)
                    .to(video_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?
            .add_source(audio_source, |source, ctx| {
                let branch = ctx.branch().pipe(audio_encoder).to(audio_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?
            .build();

        println!("recording desktop + system audio to {path} — type `q` + Enter to stop");
        pipeline.run()?;

        {
            let pipeline = pipeline.clone();
            thread::spawn(move || {
                for line in io::stdin().lock().lines() {
                    let Ok(line) = line else { break };
                    if line.trim().eq_ignore_ascii_case("q") {
                        pipeline.stop();
                        break;
                    }
                }
            });
        }

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
            // A capture that failed will not come back, and recording audio
            // against a frozen video track is not worth continuing — stop so
            // the muxer finalizes what it has.
            if matches!(event, BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("wrote {path}");
        Ok(())
    }
}

/// The Linux half of the same example. Deliberately the same shape as
/// `windows_example`: two independent live capture sources sharing one
/// `PipelineBuilder`, one `Mp4Muxer` with a video and an audio track, and one
/// `stop()` reaching both.
///
/// The one CLI difference is forced by the platform — Wayland cannot name a
/// monitor, so the compositor prompts for the screen on the first run and
/// hands back a restore token that skips the prompt next time. Audio needs no
/// portal at all and is selected programmatically, which is why only the video
/// half has a token.
#[cfg(target_os = "linux")]
mod linux_example {
    use std::{
        io::{self, BufRead},
        thread,
    };

    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            AudioCodec, CaptureSourceKind, Mp4Muxer, PipeWireAudioCaptureOptions,
            PipeWireAudioCaptureSource, PipeWireAudioDeviceKind, PipeWireScreenCaptureOptions,
            PipeWireScreenCaptureSource, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder,
            SwEncoderOptions, SwScaler, VideoCodec,
        },
        pipeline::PipelineBuilder,
    };

    /// PipeWireScreenCaptureSource + PipeWireAudioCaptureSource (a sink's
    /// monitor — whatever the system is playing) -> one Mp4Muxer: records the
    /// desktop and its system audio together into a single playable `.mp4`.
    ///
    /// Neither capture source ever reaches a natural `Eos`, so this runs until
    /// `q` + Enter in the same terminal, which is also what finalizes the MP4's
    /// trailer.
    ///
    ///     cargo run -p screen_record_av -- [output.mp4] [monitor|window] [restore-token]
    ///     (then in the same terminal: `q` + Enter to stop and finalize)
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
            .unwrap_or_else(|| "screen_record_av.mp4".into());
        // Monitor by default, matching the Windows branch's whole-desktop
        // capture. `window` is worth reaching for when one application is the
        // subject: a monitor stream stalls while any client is fullscreen,
        // where a window stream does not — see `PipeWireScreenCaptureSource`.
        let source_kind = match std::env::args().nth(2).as_deref() {
            Some("window") => CaptureSourceKind::Window,
            _ => CaptureSourceKind::Monitor,
        };
        // Last so it can simply be left off: it is a long opaque string that
        // only a repeat run has.
        let restore_token = std::env::args().nth(3);

        if restore_token.is_none() {
            eprintln!("opening the portal — approve the screen-share dialog to continue...");
        }
        let (video_source, video_format, restore_token) = PipeWireScreenCaptureSource::open(
            "screen",
            PipeWireScreenCaptureOptions {
                fps: 30,
                source_kind,
                include_cursor: true,
                restore_token,
            },
        )?;
        let video_time_base = video_format.time_base;
        // H.264 needs even dimensions; the portal's picker can hand back a
        // window of any size at all.
        let (width, height) = (video_format.width & !1, video_format.height & !1);

        let devices = PipeWireAudioCaptureSource::list_devices()
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let device = devices
            .iter()
            .find(|d| d.kind == PipeWireAudioDeviceKind::Sink && d.is_default)
            // A session need not designate a default; any sink still records
            // system audio through its monitor.
            .or_else(|| {
                devices
                    .iter()
                    .find(|d| d.kind == PipeWireAudioDeviceKind::Sink)
            })
            .cloned()
            .ok_or_else(|| media_pp::Error::Other("no playback device found".into()))?;
        println!("capturing system audio from: {}", device.name);
        let (audio_source, audio_format) = PipeWireAudioCaptureSource::open(
            "system-audio",
            PipeWireAudioCaptureOptions { device },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let audio_time_base = audio_source.time_base();

        let video_encoder = SwEncoder::new(
            "video-encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width,
                height,
                time_base: video_time_base,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 4_000_000,
                gop_size: 60, // ~2s @ 30fps
            },
        )
        .expect("failed to open video encoder");
        let audio_encoder = SwAudioEncoder::new(
            "audio-encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: audio_format.sample_rate,
                channels: audio_format.channels,
                time_base: audio_time_base,
                bit_rate: 128_000,
            },
        )
        .expect("failed to open audio encoder");

        let mut muxer = Mp4Muxer::create(&path)?;
        muxer.add_stream("video", video_encoder.parameters(), video_time_base)?;
        muxer.add_stream("audio", audio_encoder.parameters(), audio_time_base)?;
        let mut sinks = muxer.open()?;
        let audio_sink = sinks.pop().expect("two streams were added");
        let video_sink = sinks.pop().expect("two streams were added");

        let pipeline = PipelineBuilder::new("screen-record-av")
            .add_source(video_source, |source, ctx| {
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
                    .pipe(video_encoder)
                    .to(video_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?
            .add_source(audio_source, |source, ctx| {
                let branch = ctx.branch().pipe(audio_encoder).to(audio_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?
            .build();

        println!("recording desktop + system audio to {path} — type `q` + Enter to stop");
        pipeline.run()?;

        {
            let pipeline = pipeline.clone();
            thread::spawn(move || {
                for line in io::stdin().lock().lines() {
                    let Ok(line) = line else { break };
                    if line.trim().eq_ignore_ascii_case("q") {
                        pipeline.stop();
                        break;
                    }
                }
            });
        }

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
            // A capture that failed will not come back, and recording audio
            // against a frozen video track is not worth continuing — stop so
            // the muxer finalizes what it has.
            if matches!(event, BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("wrote {path}");
        match restore_token {
            Some(token) => println!(
                "re-run without a dialog:\n  ... {path} {} {token}",
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
