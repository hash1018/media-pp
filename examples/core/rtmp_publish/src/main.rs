//! TestVideoSource + TestAudioSource -> SwEncoder/SwAudioEncoder ->
//! RtmpMuxer: publishes a live H.264 + AAC broadcast to an RTMP server,
//! which is what Twitch and YouTube receive.
//!
//! Two sources into one muxer is the point: a broadcast is video *and*
//! audio in one FLV container, so this is `RtmpMuxer`'s two-track path
//! rather than `rtsp_serve`'s single stream. Both sources are synthetic, so
//! nothing has to be captured, downloaded, or granted a permission to run
//! it.
//!
//! The server is somebody else's — this publishes and does not listen.
//! [MediaMTX] accepts RTMP on port 1935 with its shipped configuration:
//!
//! ```text
//! ./mediamtx                                          # in another terminal
//! cargo run -p rtmp_publish -- [rtmp://host/app/key] [seconds]
//! ffplay -fflags nobuffer rtmp://127.0.0.1:1935/live/stream   # in a third
//! ```
//!
//! [MediaMTX]: https://github.com/bluenviron/mediamtx

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{thread, time::Duration};

    use media_pp::ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            AudioCodec, RtmpMuxer, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder,
            SwEncoderOptions, TestAudioOptions, TestAudioSource, TestVideoOptions, TestVideoSource,
            VideoCodec,
        },
        pipeline::PipelineBuilder,
    };

    /// Two seconds at 30 fps. A viewer joining a live stream cannot start
    /// until a keyframe arrives, so this is how long `ffplay` may sit black
    /// before the picture appears — not a container constraint like
    /// `hls`'s, just the wait this example is willing to show.
    const GOP_SIZE: u32 = 60;

    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let url = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "rtmp://127.0.0.1:1935/live/stream".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|value| value.parse().ok())
            .unwrap_or(10);

        let video_options = TestVideoOptions {
            width: 640,
            height: 360,
            framerate: ffmpeg::Rational::new(30, 1),
        };
        let video_source = TestVideoSource::new("video", video_options);
        let video_time_base = video_source.time_base();
        let video_encoder = SwEncoder::new(
            "video-encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: video_options.width,
                height: video_options.height,
                time_base: video_time_base,
                frame_rate: video_options.framerate,
                bit_rate: 1_500_000,
                gop_size: GOP_SIZE,
                // FLV carries no reordered timestamps well, and a live
                // broadcast has no use for the latency B-frames buy.
                max_b_frames: None,
            },
        )?;

        let audio_options = TestAudioOptions {
            sample_rate: 48_000,
            channels: 2,
            frequency: 440.0,
        };
        let audio_source = TestAudioSource::new("audio", audio_options);
        let audio_time_base = audio_source.time_base();
        let audio_encoder = SwAudioEncoder::new(
            "audio-encoder",
            SwAudioEncoderOptions {
                // H.264 and AAC are the pair every RTMP service accepts.
                codec: AudioCodec::Aac,
                sample_rate: audio_options.sample_rate,
                channels: audio_options.channels,
                time_base: audio_time_base,
                bit_rate: 128_000,
            },
        )?;

        // Connects here: an unreachable server or a rejected stream key
        // fails before a single frame has been encoded.
        let mut muxer = RtmpMuxer::create(&url)?;
        muxer.add_stream("video", video_encoder.parameters(), video_time_base)?;
        muxer.add_stream("audio", audio_encoder.parameters(), audio_time_base)?;
        // Held before `open` consumes the muxer — the URL itself is not
        // printed, since a real one ends in a credential.
        let shown_url = muxer.redacted_url().to_string();
        let mut sinks = muxer.open()?;
        let audio_sink = sinks.pop().expect("two streams were added");
        let video_sink = sinks.pop().expect("two streams were added");

        // One pipeline, two live sources — neither can be the other's
        // upstream, and one `stop()` ends both so the muxer sees every
        // track finish (see `RtmpMuxer::open`'s own docs).
        let pipeline = PipelineBuilder::new("rtmp-publish")
            .add_source(video_source, |source, ctx| {
                let branch = ctx
                    .branch()
                    // Thread boundary: encoding must not stall the source's
                    // wall clock, or the broadcast falls behind real time.
                    .queue("frames", 8)
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

        println!("publishing to {shown_url} for {seconds}s");
        pipeline.run()?;

        {
            let pipeline = pipeline.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_secs(seconds));
                pipeline.stop();
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
            // A lost connection does not come back on its own — this type
            // does not reconnect — so publishing audio into a broken socket
            // is not worth continuing.
            if matches!(event, BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("stopped publishing to {shown_url}");
        Ok(())
    }
}
