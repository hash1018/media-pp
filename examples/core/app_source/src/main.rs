//! A background thread demuxes a file with plain `ffmpeg` — standing
//! in for whatever a real caller would push from (a network receive loop,
//! a camera SDK's callback, ...) — and pushes its video packets into an
//! `AppSource` one by one via `AppSourceHandle::push`. The pipeline itself
//! (`AppSource -> SwDecoder -> FrameCounter`) never touches the file
//! directly; proves `AppSource` (GStreamer's `appsrc` equivalent) actually
//! carries externally-produced buffers all the way through decode.
//!
//!     cargo run -p app_source -- path/to/video.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{sync::Arc, thread};

    use media_pp::ffmpeg;
    use media_pp::{
        Result,
        buffer::MediaBuffer,
        bus::BusEvent,
        elements::{AppSource, FrameCounter, SwDecoder},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: app_source <video.mp4>");
            std::process::exit(1);
        };

        // Opened once up front just to learn the video stream's decoder
        // parameters — same reason every other example does this before
        // wiring — then closed; the feeder thread below opens its own handle
        // on the file for the actual read loop.
        let (video_index, params) = {
            let input = ffmpeg::format::input(&path)?;
            let stream = input
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or_else(|| media_pp::Error::Other("no video stream in file".into()))?;
            (stream.index(), stream.parameters())
        };

        let (app_source, handle) = AppSource::new("app-source", 8);
        let (frame_counter, count) = FrameCounter::new("frame-counter");

        let pipeline = Pipeline::new("app-source", app_source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as `AppSource::run` — cheap enough not to need a queue
                .to(Box::new(frame_counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
        pipeline.run()?;

        // The "external producer": wholly separate from the thread
        // `Pipeline::run` spawned for `app_source` above.
        let feeder = thread::spawn(move || -> Result<()> {
            let mut input = ffmpeg::format::input(&path)?;
            for (stream, packet) in input.packets() {
                if stream.index() == video_index {
                    handle.push(MediaBuffer::Packet(Arc::new(packet)))?;
                }
            }
            handle.push(MediaBuffer::Eos)
        });

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        feeder.join().expect("feeder thread panicked")?;
        println!(
            "decoded frames: {}",
            count.load(std::sync::atomic::Ordering::Relaxed)
        );
        Ok(())
    }
}
