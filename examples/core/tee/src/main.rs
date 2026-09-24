//! Demux -> Tee, fanning the same packets out to two independent
//! branches:
//!   - SwDecoder -> FrameCounter: decodes and counts frames
//!   - PacketCounter: counts the raw (still-encoded) packets
//!
//! Proves `Tee` delivers every packet to both branches — same source
//! data, two unrelated consumers.
//!
//!     cargo run -p tee -- path/to/video.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {

    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{FileDemuxer, FrameCounter, PacketCounter, SwDecoder},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: tee <video.mp4>");
            std::process::exit(1);
        };

        let (source, streams) = FileDemuxer::open("demux", &path)?;

        println!("stream count: {}", streams.len());
        for s in &streams {
            println!("  [{}] {:?}", s.index, s.kind);
        }

        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        let (frame_counter, frame_count) = FrameCounter::new("frame-counter");
        let (packet_counter, packet_count) = PacketCounter::new("packet-counter");

        let (pipeline, ()) = Pipeline::new("tee", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params)?;
            let decode_branch = ctx.branch().pipe(decoder).to(frame_counter)?;
            let packet_branch = ctx.branch().to(packet_counter)?;

            let tee_branch = ctx
                .tee("tee")
                .branch(decode_branch)
                .branch(packet_branch)
                .build()?;
            ctx.attach(source, video.index, tee_branch)?;
            Ok(())
        })?;

        pipeline.run()?;
        // Printed as `log_events()` would print it, stopping on `Finished` —
        // everything read has reached both counters — or on an error. A file's
        // source waits at its end until stopped, so the bus does not close by
        // itself.
        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("decoded frames: {}", frame_count.get());
        println!("raw packets: {}", packet_count.get());
        Ok(())
    }
}
