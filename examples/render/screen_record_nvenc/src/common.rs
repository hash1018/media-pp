//! Everything about this example that is not the GPU stack: argument
//! parsing and the fixed-duration recording loop. Both backends drive the
//! identical shell, so the only thing that differs between platforms is how
//! the capture source, upload, and encoder are constructed.

use std::{
    thread,
    time::{Duration, Instant},
};

use media_pp::{bus::BusEvent, pipeline::Pipeline};

pub struct Recording {
    pub path: String,
    pub seconds: u64,
}

/// `usage_tail` is whatever the platform adds after `[seconds]` — Wayland
/// needs a source kind and a restore token that DXGI has no equivalent for.
pub fn parse_args(usage_tail: &str) -> media_pp::Result<Recording> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| env!("CARGO_PKG_NAME").into());
    let usage = format!("usage: {program} <output.mp4> [seconds]{usage_tail}");

    let Some(path) = args.next() else {
        eprintln!("{usage}");
        return Err(media_pp::Error::Other("missing output path".into()));
    };
    let seconds = args
        .next()
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| media_pp::Error::Other(format!("invalid seconds: {value}")))
        })
        .transpose()?
        .unwrap_or(5);
    if seconds == 0 {
        return Err(media_pp::Error::Other(
            "recording duration must be greater than zero".into(),
        ));
    }
    Ok(Recording { path, seconds })
}

/// Records for `seconds`, then finishes the pipeline and drains its bus.
///
/// Neither capture source ever reaches `Eos` on its own, so the duration is
/// what ends the recording. `Pipeline::finish` sends ordered EOS through the
/// encoder and muxer so delayed frames are drained before the MP4 trailer is
/// finalized; a pipeline that already failed is stopped instead, since there
/// is nothing worth draining.
pub fn record(pipeline: &Pipeline, seconds: u64) -> media_pp::Result<()> {
    pipeline.run();

    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut pipeline_error = None;
    while Instant::now() < deadline && pipeline_error.is_none() {
        while let Some(event) = pipeline.bus().try_recv() {
            match event {
                BusEvent::Error { name, error, .. } => {
                    eprintln!("[{name}] error: {error}");
                    pipeline_error = Some(error);
                    break;
                }
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
        }
        thread::sleep(
            deadline
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(20)),
        );
    }

    if pipeline_error.is_some() {
        pipeline.stop();
    } else {
        pipeline.finish();
    }

    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Error { name, error, .. } => {
                eprintln!("[{name}] error: {error}");
                if pipeline_error.is_none() {
                    pipeline_error = Some(error);
                }
            }
            BusEvent::Dropped { name, .. } => {
                eprintln!("[{name}] dropped a buffer (queue full)")
            }
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            _ => {}
        }
    }

    match pipeline_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}
