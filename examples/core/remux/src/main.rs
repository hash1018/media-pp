//! FileDemuxer -> FileMuxer: remuxes every stream this crate has a kind
//! for — video, audio and subtitles — straight into a new `.mp4`
//! container, with no decode/re-encode, just repackaging. Packets pass
//! through byte-for-byte; only their timestamps get rescaled to whatever
//! time_base the output container actually assigns each stream.
//!
//! Which streams those are is asked of `MediaKind::packet_for` rather than
//! listed here, so this does not go stale as the crate learns to carry
//! more: subtitles started travelling when `MediaKind::SubtitlePacket` was
//! added, and this file needed no edit to start keeping them.
//!
//! `FileDemuxer` is a single source with one `src_pad` per container
//! stream, so — unlike combining two independent *live* sources (see
//! `screen_record_av`, which needs `PipelineBuilder` for exactly that)
//! — this only ever needs one `Pipeline`: `Eos` reaches every kept
//! stream's `FileMuxer` sink from that same source thread, no multi-source
//! coordination needed.
//!
//! The destination is staged and published only on success. Existing outputs
//! need `--overwrite`; input/output aliases are always refused.
//!
//!     cargo run -p remux -- input.mp4 [output.mp4] [--overwrite]

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use file_common::{FileArgs, SafeOutput, run_to_end};
    use media_pp::contract::MediaKind;
    use media_pp::{
        elements::{FileDemuxer, FileMuxer, Mp4Layout},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let args =
            FileArgs::parse(std::env::args_os().skip(1), Some("remuxed.mp4")).map_err(|e| {
                file_common::message(format!(
                    "{e}\nusage: remux <input> [output.mp4] [--overwrite]"
                ))
            })?;
        let output = SafeOutput::new(&args)?;

        let (source, streams) = FileDemuxer::open("demux", &args.input)?;

        // Whichever streams this crate has a kind for, which is what
        // `MediaKind::packet_for` answers: video, audio and subtitles.
        // Asking the library rather than listing the kinds here is what
        // keeps this from going stale — subtitles only started travelling
        // when `SubtitlePacket` was added, and this needed no edit to
        // start carrying them.
        //
        // A stream with no kind — a data stream, an attachment — is
        // skipped rather than failing the whole remux over one track
        // nothing here can describe.
        let mut muxer = FileMuxer::create(output.path(), Mp4Layout::Standard)?;
        let mut kept = Vec::new();
        for stream in &streams {
            if MediaKind::packet_for(stream.kind()).is_none() {
                println!(
                    "skipping stream {} ({:?}) — not a kind this can carry",
                    stream.index(),
                    stream.kind()
                );
                continue;
            }
            let track = muxer.add_stream(format!("{:?}", stream.kind()), stream)?;
            kept.push((stream.index(), track));
        }
        let mut sinks = muxer.open()?;
        let routes = kept
            .into_iter()
            .map(|(stream_index, track)| Ok((stream_index, sinks.take(track)?)))
            .collect::<media_pp::Result<Vec<_>>>()?;

        let (pipeline, ()) = Pipeline::new("remux", source, |source, ctx| {
            for (stream_index, sink) in routes {
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, stream_index, branch)?;
            }
            Ok(())
        })?;

        println!(
            "remuxing {} -> {} ...",
            args.input.display(),
            args.output.display()
        );
        run_to_end(pipeline)?;
        output.commit()?;
        println!("wrote {}", args.output.display());
        Ok(())
    }
}
