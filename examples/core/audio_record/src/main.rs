//! TestAudioSource -> SwAudioEncoder -> FileMuxer: encodes a generated sine
//! tone straight into a playable `.mp4` file — the audio-only counterpart
//! to `screen_record_software`'s video path, and `FileMuxer`'s single-track path
//! (see `screen_record_av` for a video+audio track combined into one
//! file instead).
//!
//!     cargo run -p audio_record -- [output.mp4] [seconds]

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{thread, time::Duration};

    use media_pp::{
        elements::{
            AudioCodec, FileMuxer, SwAudioEncoder, SwAudioEncoderOptions, TestAudioOptions,
            TestAudioSource,
        },
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "audio_record.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        let audio_options = TestAudioOptions {
            sample_rate: 48000,
            channels: 2,
            frequency: 440.0,
        };
        let source = TestAudioSource::new("audio", audio_options);

        let encoder = SwAudioEncoder::new(
            "encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: audio_options.sample_rate,
                channels: audio_options.channels,
                bit_rate: 128_000,
            },
        )?;
        let mut muxer = FileMuxer::create(&path)?;
        let track = muxer.add_stream("audio", encoder.parameters(), encoder.time_base())?;
        let muxer_sink = muxer.open()?.take(track)?;

        let (pipeline, ()) = Pipeline::new("audio-record", source, |source, ctx| {
            let branch = ctx.branch().pipe(encoder).to(muxer_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        println!(
            "recording {seconds}s of a {}Hz tone to {path} ...",
            audio_options.frequency
        );
        pipeline.run()?;

        thread::sleep(Duration::from_secs(seconds));
        pipeline.stop();

        for event in pipeline.bus().iter() {
            println!("{event}");
        }

        println!("wrote {path}");
        Ok(())
    }
}
