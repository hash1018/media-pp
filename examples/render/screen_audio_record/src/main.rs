#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{
        io::{self, BufRead},
        thread,
    };

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            AudioCodec, CaptureMode, DxgiCaptureOptions, DxgiCaptureSource, Mp4Muxer, Scaler,
            SwAudioEncoder, SwAudioEncoderOptions, SwEncoder, SwEncoderOptions, VideoCodec,
            WasapiCaptureOptions, WasapiCaptureSource, WasapiDeviceKind,
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
    /// `screen_record`'s own docs) — this runs until `q` + Enter in the same
    /// terminal, which is also what finalizes the MP4's trailer (`Mp4Muxer`
    /// writes it once *every* track — video and audio both — reports done via
    /// `Eos` *or* `Stop`, not on whichever finishes first; see `Mp4Muxer::open`'s
    /// own docs, and `PipelineBuilder`'s for why one `stop()` call is enough to
    /// reach both tracks even though they're two independent sources).
    ///
    ///     cargo run -p screen_audio_record -- [output.mp4]
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
            .unwrap_or_else(|| "screen_audio_record.mp4".into());

        let capture_options = DxgiCaptureOptions {
            fps: 30,
            capture_mode: CaptureMode::Cpu {
                include_cursor: true,
            },
            ..DxgiCaptureOptions::default()
        };
        let (video_source, width, height, _device) =
            DxgiCaptureSource::open("screen", capture_options)?;
        let video_time_base = video_source.time_base();

        let devices = WasapiCaptureSource::list_devices()
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let device = devices
            .into_iter()
            .find(|d| d.kind == WasapiDeviceKind::Render && d.is_default)
            .ok_or_else(|| media_pp::Error::Other("no default playback device found".into()))?;
        println!("capturing system audio from: {}", device.name);
        let (audio_source, sample_rate, channels) =
            WasapiCaptureSource::open("system-audio", WasapiCaptureOptions { device })
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
                sample_rate,
                channels,
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

        let pipeline = PipelineBuilder::new("screen-audio-record")
            .add_source(video_source, |source, ctx| {
                let scaler = Scaler::new(
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
        pipeline.run();

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
        }

        println!("wrote {path}");
        Ok(())
    }
}
