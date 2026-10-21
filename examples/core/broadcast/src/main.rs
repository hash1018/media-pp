//! One input written once into a `Broadcast` and read twice: recorded to a
//! file through a `BroadcastSource`, and counted by a `BroadcastReader` of
//! the program's own — while the input drops and comes back.
//!
//! The input is a file played at its own speed, standing in for a camera.
//! It plays twice, a second apart, each time as a pipeline of its own that
//! publishes into the same broadcast — what an input opened again after it
//! dropped does. The recording and the reader go on across the drop: the
//! reader is told one run ended and another began, and the recording puts
//! the second run after the first, a second's gap between them, on one
//! timeline from zero.
//!
//! ```text
//! FileDemuxer -> Pacer -> BroadcastSink        (one per track, each run)
//!                              |
//!                          Broadcast
//!                         /         \
//! BroadcastSource -> FileMuxer     BroadcastReader (counts, on a thread)
//! ```
//!
//! Each run plays at most five seconds of the input.
//!
//!     cargo run -p broadcast -- input.mp4 [output.mp4] [--overwrite]

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::thread;
    use std::time::{Duration, Instant};

    use file_common::{FileArgs, SafeOutput, message};
    use media_pp::bus::BusEvent;
    use media_pp::contract::MediaKind;
    use media_pp::elements::{
        Broadcast, BroadcastItem, BroadcastOptions, BroadcastSource, FileDemuxer, FileMuxer,
        Mp4Layout, Pacer, TrackFormat,
    };
    use media_pp::pipeline::Pipeline;

    /// How much of the input each run plays.
    const RUN: Duration = Duration::from_secs(5);
    /// How long the input is gone between its two runs.
    const DROP: Duration = Duration::from_secs(1);

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let args =
            FileArgs::parse(std::env::args_os().skip(1), Some("broadcast.mp4")).map_err(|e| {
                message(format!(
                    "{e}\nusage: broadcast <input> [output.mp4] [--overwrite]"
                ))
            })?;
        let output = SafeOutput::new(&args)?;

        let camera = Broadcast::new("camera", BroadcastOptions::default());

        // The first run is published before anything reads, so the
        // recording knows the tracks it makes outputs for.
        let first = Input::open(&camera, &args.input, 1)?;

        let (source, tracks) = BroadcastSource::new("recorder", &camera)?;
        let mut muxer = FileMuxer::create(output.path(), Mp4Layout::Standard)?;
        let ids = tracks
            .iter()
            .map(|track| muxer.add_stream(format!("{:?}", track.kind), track))
            .collect::<media_pp::Result<Vec<_>>>()?;
        let mut files = muxer.open()?;
        let ends = ids
            .into_iter()
            .map(|id| files.take(id))
            .collect::<media_pp::Result<Vec<_>>>()?;
        let (recording, ()) = Pipeline::new("recording", source, |source, ctx| {
            for (index, end) in ends.into_iter().enumerate() {
                let branch = ctx
                    .branch()
                    .queue(format!("recorded-{index}"), 64)
                    .to(end)?;
                ctx.attach(source, index, branch)?;
            }
            Ok(())
        })?;
        recording.run()?;

        let mut reader = camera.subscribe();
        let counter = thread::spawn(move || {
            let mut runs = Vec::new();
            loop {
                match reader.next_timeout(Duration::from_secs(1)) {
                    Some(BroadcastItem::Started(tracks)) => {
                        println!(
                            "reader: run {} began, {} tracks",
                            runs.len() + 1,
                            tracks.len()
                        );
                        runs.push(0usize);
                    }
                    Some(BroadcastItem::Packet { .. }) => {
                        if let Some(count) = runs.last_mut() {
                            *count += 1;
                        }
                    }
                    Some(BroadcastItem::Skipped) => println!("reader: fell behind, skipped"),
                    Some(BroadcastItem::Ended) => println!(
                        "reader: run {} ended after {} packets",
                        runs.len(),
                        runs.last().copied().unwrap_or(0)
                    ),
                    Some(BroadcastItem::Closed) => break,
                    None => {}
                }
            }
            runs
        });

        println!("run 1: playing {} as a camera ...", args.input.display());
        first.play()?;
        println!("the camera dropped; back in {DROP:?}");
        thread::sleep(DROP);
        println!("run 2: playing it again into the same broadcast ...");
        Input::open(&camera, &args.input, 2)?.play()?;

        // Closing ends the recording's stream, which finishes the file.
        camera.close();
        wait_for_finished(&recording)?;
        recording.stop();
        output.commit()?;
        let runs = counter
            .join()
            .map_err(|_| message("the reader's thread panicked"))?;
        println!("reader: {} runs, {:?} packets", runs.len(), runs);
        println!("wrote {}", args.output.display());
        Ok(())
    }

    /// One run of the input: its pipeline, wired to the sinks a
    /// publication handed out.
    struct Input(std::sync::Arc<Pipeline>);

    impl Input {
        fn open(camera: &Broadcast, path: &std::path::Path, run: usize) -> media_pp::Result<Self> {
            let (demuxer, streams) = FileDemuxer::open("demux", path)?;
            let kept: Vec<_> = streams
                .iter()
                .filter(|stream| MediaKind::packet_for(stream.kind).is_some())
                .collect();
            let formats: Vec<TrackFormat> = kept
                .iter()
                .map(|stream| TrackFormat::from(*stream))
                .collect();
            let sinks = camera.publish(&formats)?;
            let (pipeline, ()) = Pipeline::new(format!("camera-{run}"), demuxer, |source, ctx| {
                for (stream, sink) in kept.iter().zip(sinks) {
                    let branch = ctx
                        .branch()
                        .queue(format!("packets-{}", stream.index), 64)
                        .pipe(Pacer::new(format!("pacer-{}", stream.index)))
                        .to(sink)?;
                    ctx.attach(source, stream.index, branch)?;
                }
                Ok(())
            })?;
            Ok(Self(pipeline))
        }

        /// Plays it for [`RUN`], or to its end, then stops it.
        fn play(self) -> media_pp::Result<()> {
            let pipeline = self.0;
            pipeline.run()?;
            let deadline = Instant::now() + RUN;
            let mut result = Ok(());
            loop {
                let left = deadline.saturating_duration_since(Instant::now());
                match pipeline.bus().recv_timeout(left) {
                    Ok(BusEvent::Finished) | Err(_) => break,
                    Ok(BusEvent::Error { error, .. }) => {
                        result = Err(error);
                        break;
                    }
                    Ok(_) => {}
                }
            }
            pipeline.stop();
            result
        }
    }

    fn wait_for_finished(pipeline: &Pipeline) -> media_pp::Result<()> {
        for event in pipeline.bus().iter() {
            match event {
                BusEvent::Finished => return Ok(()),
                BusEvent::Error { error, .. } => return Err(error),
                _ => {}
            }
        }
        Err(message("the recording ended without finishing"))
    }
}
