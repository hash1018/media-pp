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
    use std::{thread, time::Duration};

    use media_pp::{
        Result,
        bus::BusEvent,
        elements::{
            AudioResampler, AudioVolume, TestAudioOptions, TestAudioSource, WasapiRenderer,
            WasapiRendererOptions,
        },
        pipeline::Pipeline,
    };

    /// TestAudioSource -> AudioResampler -> AudioVolume -> Queue ->
    /// WasapiRenderer: plays a 440Hz tone for three seconds and demonstrates
    /// click-free runtime gain/mute changes.
    ///
    ///     cargo run -p audio_playback
    ///     cargo run -p audio_playback -- list
    ///     cargo run -p audio_playback -- <device-name-substring>
    pub(super) fn run() -> Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

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

        let (mut renderer, output_format) =
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
        let resampler = AudioResampler::new("resampler", output_format, source.time_base())?;
        let (volume, volume_handle) = AudioVolume::new("volume");
        let pipeline = Pipeline::new("audio-playback", source, |source, context| {
            renderer.bind_playback_clock(context.playback_clock.clone())?;
            let branch = context
                .branch()
                .pipe(resampler)
                .pipe(volume)
                .queue("audio-output", 8)
                .to(Box::new(renderer))?;
            context.attach(source, 0, branch)?;
            Ok(())
        })?;

        pipeline.run();
        thread::sleep(Duration::from_secs(1));
        println!("volume: -12 dB");
        volume_handle.set_gain_db(-12.0)?;
        thread::sleep(Duration::from_secs(1));
        println!("muted");
        volume_handle.set_muted(true);
        thread::sleep(Duration::from_millis(500));
        println!("unmuted");
        volume_handle.set_muted(false);
        thread::sleep(Duration::from_millis(500));
        pipeline.stop();

        for event in pipeline.bus().iter() {
            if let BusEvent::Error { name, error, .. } = event {
                eprintln!("[{name}] error: {error}");
            }
        }
        Ok(())
    }
}
