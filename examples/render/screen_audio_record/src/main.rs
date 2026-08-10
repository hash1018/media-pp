use std::{
    io::{self, BufRead},
    thread,
};

use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    element::Source,
    elements::{
        AudioCaptureOptions, AudioCaptureSource, AudioCodec, AudioDeviceKind, CaptureMode,
        DxgiScreenOptions, DxgiScreenSource, Mp4Muxer, Scaler, SwAudioEncoder,
        SwAudioEncoderOptions, SwEncoder, SwEncoderOptions, VideoCodec,
    },
    pipeline::{ChainBuilder, PipelineBuilder},
};

/// DxgiScreenSource + AudioCaptureSource (system-audio loopback — whatever
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
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "screen_audio_record.mp4".into());

    let capture_options = DxgiScreenOptions {
        fps: 30,
        capture_mode: CaptureMode::Cpu {
            include_cursor: true,
        },
        ..DxgiScreenOptions::default()
    };
    let (video_source, width, height, _device) = DxgiScreenSource::open("screen", capture_options)?;
    let video_time_base = video_source.time_base();

    let devices =
        AudioCaptureSource::list_devices().map_err(|e| media_pp::Error::Other(e.to_string()))?;
    let device = devices
        .into_iter()
        .find(|d| d.kind == AudioDeviceKind::Render && d.is_default)
        .ok_or_else(|| media_pp::Error::Other("no default playback device found".into()))?;
    println!("capturing system audio from: {}", device.name);
    let (audio_source, sample_rate, channels) =
        AudioCaptureSource::open("system-audio", AudioCaptureOptions { device })
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
            let branch = ChainBuilder::new(ctx.clone())
                .queue("captured", 4) // thread boundary so scaling doesn't block capture
                .pipe(scaler)
                .queue("frames", 8) // thread boundary so encoding doesn't block scaling
                .pipe(video_encoder)
                .build(video_sink);
            source.src_pads()[0].link(branch);
        })
        .add_source(audio_source, |source, ctx| {
            let branch = ChainBuilder::new(ctx.clone())
                .pipe(audio_encoder)
                .build(audio_sink);
            source.src_pads()[0].link(branch);
        })
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
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            _ => {}
        }
    }

    println!("wrote {path}");
    Ok(())
}
