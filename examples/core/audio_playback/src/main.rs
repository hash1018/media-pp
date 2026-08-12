use std::{thread, time::Duration};

use media_pp::{
    Result,
    bus::BusEvent,
    elements::{
        AudioResampler, TestAudioOptions, TestAudioSource, WasapiRenderer, WasapiRendererOptions,
    },
    pipeline::Pipeline,
};

/// TestAudioSource -> AudioResampler -> Queue -> WasapiRenderer: plays a
/// 440Hz tone for three seconds through a selected WASAPI render endpoint.
///
///     cargo run -p audio_playback
///     cargo run -p audio_playback -- list
///     cargo run -p audio_playback -- <device-name-substring>
fn main() -> Result<()> {
    media_pp::init()?;

    let devices = WasapiRenderer::list_devices()
        .map_err(|error| media_pp::Error::Other(error.to_string()))?;
    let argument = std::env::args().nth(1);
    if argument.as_deref() == Some("list") {
        for device in &devices {
            println!(
                "{}{}",
                device.name,
                if device.is_default { " (default)" } else { "" }
            );
        }
        return Ok(());
    }

    let device = devices
        .into_iter()
        .find(|device| match &argument {
            Some(name) => device.name.contains(name),
            None => device.is_default,
        })
        .ok_or_else(|| media_pp::Error::Other("no matching render device found".into()))?;
    println!("selected: {}", device.name);

    let (renderer, output_format) =
        WasapiRenderer::open("speakers", WasapiRendererOptions { device })
            .map_err(|error| media_pp::Error::Other(error.to_string()))?;
    println!(
        "output: {}Hz, {} channel(s), {:?}",
        output_format.sample_rate, output_format.channels, output_format.sample_format
    );

    // Deliberately differs from whatever the endpoint reports on systems
    // configured to 44.1kHz or mono, proving AudioResampler owns the format
    // conversion rather than WasapiRenderer doing it implicitly.
    let source = TestAudioSource::new("tone", TestAudioOptions::default());
    let resampler = AudioResampler::new("resampler", output_format);
    let pipeline = Pipeline::new("audio-playback", source, |source, context| {
        let branch = context
            .branch()
            .pipe(resampler)
            .queue("audio-output", 8)
            .to(Box::new(renderer))?;
        context.attach(source, 0, branch)?;
        Ok(())
    })?;

    pipeline.run();
    thread::sleep(Duration::from_secs(3));
    pipeline.stop();

    for event in pipeline.bus().iter() {
        if let BusEvent::Error { name, error, .. } = event {
            eprintln!("[{name}] error: {error}");
        }
    }
    Ok(())
}
