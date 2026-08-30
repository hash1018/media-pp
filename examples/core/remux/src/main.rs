//! FileDemuxer -> FileMuxer: remuxes every video/audio stream in a file
//! straight into a new `.mp4` container — no decode/re-encode, just
//! repackaging. Packets pass through byte-for-byte; only their timestamps
//! get rescaled to whatever time_base the output container actually
//! assigns each stream (see `FileMuxer::open`'s own docs).
//!
//! `FileDemuxer` is a single source with one `src_pad` per container
//! stream, so — unlike combining two independent *live* sources (see
//! `screen_record_av`, which needs `PipelineBuilder` for exactly that)
//! — this only ever needs one `Pipeline`: `Eos` reaches every kept
//! stream's `FileMuxer` sink from that same source thread, no multi-source
//! coordination needed.
//!
//!     cargo run -p remux -- [input.mp4] [output.mp4]

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{FileDemuxer, FileMuxer},
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

        let Some(input_path) = std::env::args().nth(1) else {
            eprintln!("usage: remux <video.mp4> [output.mp4]");
            std::process::exit(1);
        };
        let output_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "remuxed.mp4".into());

        let (source, streams) = FileDemuxer::open("demux", &input_path)?;

        // Only video/audio streams can be muxed into an MP4 this way (see
        // `FileMuxer::add_stream`) — skip anything else (subtitles, data
        // streams, ...) rather than failing the whole remux over one stream
        // this can't carry.
        let mut muxer = FileMuxer::create(&output_path)?;
        let mut kept_indices = Vec::new();
        for stream in &streams {
            if !matches!(stream.kind, media::Type::Video | media::Type::Audio) {
                println!(
                    "skipping stream {} ({:?}) — not video/audio",
                    stream.index, stream.kind
                );
                continue;
            }
            let parameters = source
                .stream_parameters(stream.index)
                .expect("stream disappeared");
            let time_base = source
                .stream_time_base(stream.index)
                .expect("stream disappeared");
            muxer.add_stream(format!("{:?}", stream.kind), parameters, time_base)?;
            kept_indices.push(stream.index);
        }
        let sinks = muxer.open()?;
        let mut remaining = kept_indices.len();

        let pipeline = Pipeline::new("remux", source, |source, ctx| {
            for (stream_index, sink) in kept_indices.into_iter().zip(sinks) {
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, stream_index, branch)?;
            }
            Ok(())
        })?;

        println!("remuxing {input_path} -> {output_path} ...");
        pipeline.run()?;

        // Multiple tracks means multiple `BusEvent::Eos` (one per stream's own
        // `FileMuxer` sink) — only `stop()` once every kept stream has reported
        // its own `Eos`, not the first one (see `FileMuxer::open`'s own docs on
        // why finalizing early would truncate whichever track is still going).
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => {
                    println!("[{name}] eos");
                    remaining = remaining.saturating_sub(1);
                }
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
            if remaining == 0 || matches!(event, BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }

        println!("wrote {output_path}");
        Ok(())
    }
}
