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
    use std::{sync::atomic::Ordering, thread, time::Duration};

    use media_pp::{
        Result,
        bus::BusEvent,
        elements::{FrameCounter, WasapiCaptureOptions, WasapiCaptureSource, WasapiDeviceKind},
        pipeline::Pipeline,
    };

    /// Lists every audio endpoint, picks one, captures ~3 seconds from it and
    /// reports how many buffers came through — a smoke test for
    /// `WasapiCaptureSource`'s list-then-pick device API.
    ///
    ///     cargo run -p audio_capture              # default render device (system audio / loopback)
    ///     cargo run -p audio_capture -- mic        # default capture device (microphone)
    ///     cargo run -p audio_capture -- list       # just print every device and exit
    ///     cargo run -p audio_capture -- <name>     # first device whose name contains <name>
    pub(super) fn run() -> Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let devices = WasapiCaptureSource::list_devices()
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let arg = std::env::args().nth(1);
        if arg.as_deref() == Some("list") {
            for device in &devices {
                println!(
                    "{:?} {}{}",
                    device.kind,
                    device.name,
                    if device.is_default { " (default)" } else { "" }
                );
            }
            return Ok(());
        }

        let wanted_kind = if arg.as_deref() == Some("mic") {
            Some(WasapiDeviceKind::Capture)
        } else {
            None
        };
        let device = devices
            .into_iter()
            .find(|d| match &wanted_kind {
                Some(kind) => d.kind == *kind && d.is_default,
                None => match &arg {
                    Some(name) => d.name.contains(name.as_str()),
                    None => d.kind == WasapiDeviceKind::Render && d.is_default,
                },
            })
            .ok_or_else(|| media_pp::Error::Other("no matching device found".into()))?;
        println!("selected: {:?} {}", device.kind, device.name);

        let (source, sample_rate, channels) =
            WasapiCaptureSource::open("audio-capture", WasapiCaptureOptions { device })
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        println!("opened: {sample_rate}Hz, {channels} channel(s)");

        let (counter, count) = FrameCounter::new("frame-counter");
        let pipeline = Pipeline::new("audio-capture", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
        pipeline.run();

        thread::sleep(Duration::from_secs(3));
        pipeline.stop();

        for event in pipeline.bus().iter() {
            if let BusEvent::Error { name, error, .. } = &event {
                eprintln!("[{name}] error: {error}");
            }
        }

        println!("buffers captured: {}", count.load(Ordering::Relaxed));
        Ok(())
    }
}
