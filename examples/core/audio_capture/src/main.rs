//! Lists audio devices, selects one, captures about three seconds, and counts
//! the buffers received.
//!
//! - Windows: `WasapiCaptureSource -> FrameCounter`
//! - Linux: `PipeWireAudioCaptureSource -> FrameCounter`
//!
//! ```text
//! cargo run -p audio_capture
//! cargo run -p audio_capture -- mic|list|<device-name-substring>
//! ```

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example supports Windows (WASAPI) and Linux (PipeWire)",
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

        let (source, format) =
            WasapiCaptureSource::open("audio-capture", WasapiCaptureOptions { device })
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        println!(
            "opened: {}Hz, {} channel(s)",
            format.sample_rate, format.channels
        );

        let (counter, count) = FrameCounter::new("frame-counter");
        let pipeline = Pipeline::new("audio-capture", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
        pipeline.run()?;

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

/// The Linux half of the same example. Deliberately the same shape as
/// `windows_example`: same CLI, same pipeline, same `FrameCounter` terminus —
/// only the source element and its device type differ, which is the whole
/// point of showing them side by side.
///
/// The device APIs really do line up, because audio capture needs no portal on
/// either platform: `PipeWireAudioDeviceKind::Sink` is captured through its
/// monitor ports and so plays the role `WasapiDeviceKind::Render` does with
/// loopback, and `Source` matches `Capture`. Screen capture is where the two
/// platforms genuinely diverge — see `PipeWireScreenCaptureSource`'s docs.
#[cfg(target_os = "linux")]
mod linux_example {
    use std::{sync::atomic::Ordering, thread, time::Duration};

    use media_pp::{
        Result,
        bus::BusEvent,
        elements::{
            FrameCounter, PipeWireAudioCaptureOptions, PipeWireAudioCaptureSource,
            PipeWireAudioDeviceKind,
        },
        pipeline::Pipeline,
    };

    /// Lists every audio node, picks one, captures ~3 seconds from it and
    /// reports how many buffers came through — a smoke test for
    /// `PipeWireAudioCaptureSource`'s list-then-pick device API.
    ///
    ///     cargo run -p audio_capture              # default sink's monitor (system audio)
    ///     cargo run -p audio_capture -- mic        # default source (microphone)
    ///     cargo run -p audio_capture -- list       # just print every node and exit
    ///     cargo run -p audio_capture -- <name>     # first node whose name contains <name>
    pub(super) fn run() -> Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let devices = PipeWireAudioCaptureSource::list_devices()
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
            Some(PipeWireAudioDeviceKind::Source)
        } else {
            None
        };
        // Prefer the default node of the wanted kind, but fall back to any
        // node of that kind. PipeWire's `default.audio.source` metadata often
        // names a *sink* — that is how "use this output's monitor as my input"
        // is expressed — which legitimately leaves no `Source` flagged default.
        let pick = |only_default: bool| {
            devices.iter().find(|d| {
                let matches_default = !only_default || d.is_default;
                match &wanted_kind {
                    Some(kind) => d.kind == *kind && matches_default,
                    None => match &arg {
                        Some(name) => d.name.contains(name.as_str()),
                        None => d.kind == PipeWireAudioDeviceKind::Sink && matches_default,
                    },
                }
            })
        };
        let device = pick(true)
            .or_else(|| pick(false))
            .cloned()
            .ok_or_else(|| media_pp::Error::Other("no matching device found".into()))?;
        println!("selected: {:?} {}", device.kind, device.name);

        let (source, format) = PipeWireAudioCaptureSource::open(
            "audio-capture",
            PipeWireAudioCaptureOptions { device },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;
        println!(
            "opened: {}Hz, {} channel(s)",
            format.sample_rate, format.channels
        );

        let (counter, count) = FrameCounter::new("frame-counter");
        let pipeline = Pipeline::new("audio-capture", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
        pipeline.run()?;

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
