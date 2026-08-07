use std::{sync::atomic::Ordering, thread, time::Duration};

use media_pp::{
    Result,
    bus::BusEvent,
    element::Source,
    elements::{AudioCaptureOptions, AudioCaptureSource, AudioDeviceKind, FrameCounter},
    pipeline::{ChainBuilder, Pipeline},
};

/// Lists every audio endpoint, picks one, captures ~3 seconds from it and
/// reports how many buffers came through — a smoke test for
/// `AudioCaptureSource`'s list-then-pick device API.
///
///     cargo run -p audio_capture              # default render device (system audio / loopback)
///     cargo run -p audio_capture -- mic        # default capture device (microphone)
///     cargo run -p audio_capture -- list       # just print every device and exit
///     cargo run -p audio_capture -- <name>     # first device whose name contains <name>
fn main() -> Result<()> {
    media_pp::init()?;

    let devices =
        AudioCaptureSource::list_devices().map_err(|e| media_pp::Error::Other(e.to_string()))?;

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
        Some(AudioDeviceKind::Capture)
    } else {
        None
    };
    let device = devices
        .into_iter()
        .find(|d| match &wanted_kind {
            Some(kind) => d.kind == *kind && d.is_default,
            None => match &arg {
                Some(name) => d.name.contains(name.as_str()),
                None => d.kind == AudioDeviceKind::Render && d.is_default,
            },
        })
        .ok_or_else(|| media_pp::Error::Other("no matching device found".into()))?;
    println!("selected: {:?} {}", device.kind, device.name);

    let (source, sample_rate, channels) =
        AudioCaptureSource::open("audio-capture", AudioCaptureOptions { device })
            .map_err(|e| media_pp::Error::Other(e.to_string()))?;
    println!("opened: {sample_rate}Hz, {channels} channel(s)");

    let (counter, count) = FrameCounter::new("frame-counter");
    let pipeline = Pipeline::new("audio-capture", source, |source, ctx| {
        let branch = ChainBuilder::new(ctx.clone()).build(Box::new(counter));
        source.src_pads()[0].link(branch);
    });
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
