use std::{thread, time::Duration};

use media_pp::{
    bus::BusEvent,
    elements::{
        AudioCodec, Mp4Muxer, SwAudioEncoder, SwAudioEncoderOptions, TestAudioOptions,
        TestAudioSource,
    },
    pipeline::Pipeline,
};

/// TestAudioSource -> SwAudioEncoder -> Mp4Muxer: encodes a generated sine
/// tone straight into a playable `.mp4` file — the audio-only counterpart
/// to `screen_record`'s video path, and `Mp4Muxer`'s single-track path
/// (see `screen_audio_record` for a video+audio track combined into one
/// file instead).
///
///     cargo run -p audio_record -- [output.mp4] [seconds]
fn main() -> media_pp::Result<()> {
    media_pp::init()?;
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
    let time_base = source.time_base();

    let encoder = SwAudioEncoder::new(
        "encoder",
        SwAudioEncoderOptions {
            codec: AudioCodec::Aac,
            sample_rate: audio_options.sample_rate,
            channels: audio_options.channels,
            time_base,
            bit_rate: 128_000,
        },
    )
    .expect("failed to open aac encoder");
    let mut muxer = Mp4Muxer::create(&path)?;
    muxer.add_stream("audio", encoder.parameters(), time_base)?;
    let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

    let pipeline = Pipeline::new("audio-record", source, |source, ctx| {
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
        match &event {
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            _ => {}
        }
    }

    println!("wrote {path}");
    Ok(())
}
