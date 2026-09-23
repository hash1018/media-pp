//! Demux -> SwDecoder -> AppSink: same shape as `decode`, but the
//! terminal sink is a plain closure instead of a bespoke `FrameCounter`
//! — proves `AppSink` lets a caller consume frames without writing a
//! dedicated `Element`/`Sink` impl at all (the GStreamer `appsink`
//! equivalent).
//!
//!     cargo run -p app_sink -- path/to/video.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use media_pp::ffmpeg::media;
    use media_pp::{
        buffer::MediaBuffer,
        bus::BusEvent,
        elements::{AppSink, FileDemuxer, SwDecoder},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: app_sink <video.mp4>");
            std::process::exit(1);
        };

        let (source, _) = FileDemuxer::open("demux", &path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        let count = Arc::new(AtomicUsize::new(0));
        let sink = {
            let count = count.clone();
            AppSink::new("counter", move |buf: MediaBuffer| {
                if matches!(buf, MediaBuffer::Video(_)) {
                    count.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            })
        };

        let (pipeline, ()) = Pipeline::new("app-sink", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params)?;
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .to(sink)?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        pipeline.run()?;

        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("decoded frames: {}", count.load(Ordering::Relaxed));
        Ok(())
    }
}
