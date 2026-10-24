//! An RTSP server: plays a file to any number of RTSP players as a camera
//! would, and takes a publisher's push on a second path, which players play
//! too.
//!
//! ```text
//! FileDemuxer -> Pacer -> BroadcastSink -> Broadcast "file"    (the file, looped, at its own speed)
//! publisher -> RtspServer -> RtspSource -> BroadcastSink -> Broadcast "live"
//!
//! Broadcast "file" -> BroadcastSource -> RtspMuxer (path "file") -> RtspServer -> players
//! Broadcast "live" -> BroadcastSource -> RtspMuxer (path "live") -> RtspServer -> players
//!
//! with --rtmp:  OBS -> RtmpServer "live/stream" -> RtmpSource -> BroadcastSink -> Broadcast "live"
//!
//! with --record:  Broadcast "file" -> BroadcastSource -> HlsMuxer (event playlist)
//! ```
//!
//! The server hands each publisher out as a source of what it sends, and a
//! pipeline of the example's own writes it into the broadcast `live`. Each
//! broadcast is the example's own: the server's path of the same name reads
//! it, through the `RtspMuxer` its path makes, as a recording would.
//!
//! Each player is sent the broadcast's packets as they are — nothing is
//! decoded or encoded for it — from the newest keyframe on. `live` takes one
//! publisher at a time; one that leaves ends its run, and the next one's
//! players go on.
//!
//!     cargo run -p rtsp_server -- input.mp4 [address] [--seconds N] [--threads N] [--record DIR] [--rtmp ADDRESS]
//!     ffplay rtsp://127.0.0.1:8554/file
//!     ffmpeg -re -i video.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/live
//!     ffplay rtsp://127.0.0.1:8554/live
//!
//! The address defaults to `127.0.0.1:8554`, this machine alone; give
//! `0.0.0.0:8554` to serve the network. It runs until Enter is pressed, or
//! for `--seconds`; `--threads` is how many threads the server runs its
//! connections on, two by default. `--record` also records what `file`
//! plays, as it plays it, into an HLS event playlist under `DIR` —
//! `index.m3u8` and its fMP4 segments, playable while it grows and finished
//! as the server stops. `--rtmp` also takes an RTMP publisher at
//! `rtmp://ADDRESS/live` with the stream key `stream` — OBS's "Custom"
//! service — into `live`, for RTSP players as an RTSP publisher's would be.

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use media_pp::bus::BusEvent;
    use media_pp::driver::DriverRunner;
    use media_pp::element::{BoxSink, IntoSource};
    use media_pp::elements::{
        Broadcast, BroadcastError, BroadcastOptions, BroadcastSource, FileDemuxer, HlsMode,
        HlsMuxer, HlsOptions, Pacer, Publishers, RtmpServer, RtmpServerOptions, RtspMuxer,
        RtspServer, RtspServerOptions, StreamInfo,
    };
    use media_pp::pipeline::Pipeline;

    const USAGE: &str = "usage: rtsp_server <input> [address] [--seconds N] [--threads N] [--record DIR] [--rtmp ADDRESS]";

    struct Args {
        input: String,
        address: String,
        seconds: Option<u64>,
        threads: Option<usize>,
        record: Option<String>,
        rtmp: Option<String>,
    }

    fn args() -> Option<Args> {
        let mut input = None;
        let mut address = None;
        let mut seconds = None;
        let mut threads = None;
        let mut record = None;
        let mut rtmp = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--seconds" {
                seconds = Some(args.next()?.parse().ok()?);
            } else if arg == "--record" {
                record = Some(args.next()?);
            } else if arg == "--rtmp" {
                rtmp = Some(args.next()?);
            } else if arg == "--threads" {
                threads = Some(args.next()?.parse().ok().filter(|&n| n > 0)?);
            } else if input.is_none() {
                input = Some(arg);
            } else if address.is_none() {
                address = Some(arg);
            } else {
                return None;
            }
        }
        Some(Args {
            input: input?,
            address: address.unwrap_or_else(|| "127.0.0.1:8554".to_owned()),
            seconds,
            threads,
            record,
            rtmp,
        })
    }

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let Some(args) = args() else {
            eprintln!("{USAGE}");
            std::process::exit(1);
        };

        let mut options = RtspServerOptions::default();
        if let Some(threads) = args.threads {
            options.threads = threads;
        }
        let (server, handle) = RtspServer::bind("server", &args.address, options)?;
        let file = Broadcast::new("file", BroadcastOptions::default());
        let live = Broadcast::new("live", BroadcastOptions::default());
        // What each broadcast carries, written into the server's path its
        // players play.
        let file_path = handle.path("file")?;
        let live_path = handle.path("live")?;
        forward(file.clone(), move |streams| {
            muxed(file_path.muxer(), streams)
        });
        forward(live.clone(), move |streams| {
            muxed(live_path.muxer(), streams)
        });
        // A publisher of `live`, into what its players play.
        relay(handle.take_publishers("live")?, live.clone());
        // RTMP publishers into the same broadcast: OBS's server
        // `rtmp://ADDRESS/live`, its stream key `stream`.
        let rtmp = match &args.rtmp {
            Some(address) => {
                let (rtmp, ingest) =
                    RtmpServer::bind("rtmp", address, RtmpServerOptions::default())?;
                relay(ingest.take_publishers("live/stream")?, live.clone());
                let rtmp = DriverRunner::new(rtmp);
                rtmp.run()?;
                println!(
                    "an RTMP publisher may push to {} (server rtmp://{}/live, key stream)",
                    ingest.url("live/stream"),
                    ingest.local_addr()
                );
                Some((rtmp, ingest))
            }
            None => None,
        };

        // The file as a camera: looped, at its own speed, into its broadcast.
        let (demuxer, streams) = FileDemuxer::open("file", &args.input)?;
        demuxer.looping_handle().set_looping(true);
        let tracks: Vec<StreamInfo> = streams.iter().map(StreamInfo::from).collect();
        let sinks = file.publish(&tracks)?;
        let (camera, ()) = Pipeline::new("file", demuxer, |source, ctx| {
            for (stream, sink) in streams.iter().zip(sinks) {
                let branch = ctx
                    .branch()
                    .queue(format!("packets-{}", stream.index()), 64)
                    .pipe(Pacer::new(format!("pacer-{}", stream.index())))
                    .to(sink)?;
                ctx.attach(source, stream.index(), branch)?;
            }
            Ok(())
        })?;
        camera.run()?;
        let recorder = match &args.record {
            Some(dir) => {
                println!("recording into {dir}/index.m3u8");
                Some(record(&file, dir)?)
            }
            None => None,
        };

        let server = DriverRunner::new(server);
        server.run()?;
        println!("playing {} at {}", args.input, handle.url("file"));
        println!("a publisher may push to {}", handle.url("live"));
        match args.seconds {
            Some(seconds) => println!("serving for {seconds} s"),
            None => println!("press Enter to stop"),
        }

        let stop = until_asked(args.seconds);
        let mut next_report = Instant::now();
        loop {
            match stop.recv_timeout(Duration::from_millis(200)) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            if Instant::now() >= next_report {
                next_report += Duration::from_secs(5);
                println!(
                    "players: {}, live published: {}, RTMP publishers: {}",
                    handle.players(),
                    live.is_live(),
                    rtmp.as_ref().map_or(0, |(_, ingest)| ingest.publishers())
                );
            }
            if let Some(event) = camera.bus().try_recv() {
                println!("{event}");
            }
        }

        if let Some(recorder) = recorder {
            // Finished, not stopped: the playlist is closed with its end.
            recorder.finish();
        }
        // The paths' forwarders end with what they read.
        file.close();
        live.close();
        server.stop();
        camera.stop();
        if let Some((rtmp, _)) = &rtmp {
            rtmp.stop();
            wait_stopped(rtmp);
        }
        wait_stopped(&server);
        println!("stopped");
        Ok(())
    }

    /// Writes what `broadcast` carries into the runs `open` begins — a
    /// server path's muxer's — on a thread of its own: a run each time the
    /// broadcast begins with tracks the last could not go on with, until it
    /// is closed. `open` answers each output to write and its sink.
    fn forward<F>(broadcast: Broadcast, open: F)
    where
        F: Fn(&[StreamInfo]) -> media_pp::Result<Vec<(usize, BoxSink)>> + Send + 'static,
    {
        std::thread::spawn(move || {
            loop {
                let (source, streams) =
                    match BroadcastSource::new(broadcast.name().to_string(), &broadcast) {
                        Ok(opened) => opened,
                        Err(BroadcastError::NotPublished(_)) => {
                            std::thread::sleep(Duration::from_millis(50));
                            continue;
                        }
                        Err(_) => return,
                    };
                let Ok(sinks) = open(&streams) else {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                };
                let wired = Pipeline::new(broadcast.name().to_string(), source, |source, ctx| {
                    for (index, sink) in sinks {
                        ctx.attach(source, index, ctx.branch().to(sink)?)?;
                    }
                    Ok(())
                });
                let Ok((pipeline, ())) = wired else {
                    return;
                };
                if pipeline.run().is_err() {
                    return;
                }
                for event in pipeline.bus().iter() {
                    if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                        break;
                    }
                }
                pipeline.stop();
            }
        });
    }

    /// Opens `muxer` with every one of `streams`, and answers each one's output and
    /// sink.
    fn muxed(
        mut muxer: RtspMuxer,
        streams: &[StreamInfo],
    ) -> media_pp::Result<Vec<(usize, BoxSink)>> {
        let tracks = streams
            .iter()
            .map(|stream| {
                Ok((
                    stream.index(),
                    muxer.add_stream(format!("track-{}", stream.index()), stream)?,
                ))
            })
            .collect::<media_pp::Result<Vec<_>>>()?;
        let mut sinks = muxer.open()?;
        tracks
            .into_iter()
            .map(|(index, track)| Ok((index, sinks.take(track)?)))
            .collect()
    }

    /// Writes each publisher `publishers` hands out into `broadcast`, a
    /// pipeline of its source each, on a thread of its own until the server
    /// stops. A publisher's source ends as it leaves; the broadcast's players
    /// go on to the next one.
    fn relay<S, M>(publishers: Publishers<S>, broadcast: Broadcast)
    where
        S: IntoSource<M> + Send + 'static,
    {
        std::thread::spawn(move || {
            while let Some((source, streams)) = publishers.next() {
                let Ok(sinks) = broadcast.publish(&streams) else {
                    continue;
                };
                let wired = Pipeline::new(publishers.path(), source, |source, ctx| {
                    for (stream, sink) in streams.iter().zip(sinks) {
                        ctx.attach(source, stream.index(), ctx.branch().to(sink)?)?;
                    }
                    Ok(())
                });
                let Ok((pipeline, ())) = wired else {
                    continue;
                };
                if pipeline.run().is_err() {
                    continue;
                }
                println!("a publisher began on {}", publishers.path());
                for event in pipeline.bus().iter() {
                    if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                        break;
                    }
                }
                pipeline.stop();
                println!("the publisher of {} left", publishers.path());
            }
        });
    }

    /// Records `broadcast` while it is served, as an HLS event playlist
    /// under `dir` — `index.m3u8`, its fMP4 segments of about two seconds,
    /// cut at keyframes, and `init.mp4`: playable as it grows, back to its
    /// start, and finished as the server stops.
    fn record(broadcast: &Broadcast, dir: &str) -> media_pp::Result<Arc<Pipeline>> {
        std::fs::create_dir_all(dir).map_err(|error| media_pp::Error::Other(error.to_string()))?;
        let dir = std::path::Path::new(dir);
        let mut options = HlsOptions::new(dir.join("index.m3u8"), dir.join("segment_%05d.m4s"));
        options.mode = HlsMode::Event;
        let (source, streams) = BroadcastSource::new("recorder", broadcast)?;
        let mut muxer = HlsMuxer::create(options)?;
        let mut tracks = Vec::new();
        for stream in &streams {
            let track = muxer.add_stream(
                format!("track-{}", stream.index()),
                StreamInfo::from(stream),
            )?;
            tracks.push((stream.index(), track));
        }
        let mut sinks = muxer.open()?;
        let (recorder, ()) = Pipeline::new("recorder", source, |source, ctx| {
            for (index, track) in tracks {
                ctx.attach(source, index, ctx.branch().to(sinks.take(track)?)?)?;
            }
            Ok(())
        })?;
        recorder.run()?;
        Ok(recorder)
    }

    /// A channel that gets a message once Enter is pressed, or `seconds`
    /// have gone by.
    fn until_asked(seconds: Option<u64>) -> mpsc::Receiver<()> {
        let (tx, rx) = mpsc::channel();
        match seconds {
            Some(seconds) => {
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(seconds));
                    let _ = tx.send(());
                });
            }
            None => {
                std::thread::spawn(move || {
                    let mut line = String::new();
                    let _ = std::io::stdin().read_line(&mut line);
                    let _ = tx.send(());
                });
            }
        }
        rx
    }

    /// Waits for the server's thread to end: its bus closes then.
    fn wait_stopped(server: &Arc<DriverRunner>) {
        for event in server.bus().iter() {
            println!("{event}");
        }
    }
}
