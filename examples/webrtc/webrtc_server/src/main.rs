//! A WebRTC server a browser plays: a file played to any number of WebRTC
//! players as a camera would, and a second path a WHIP publisher pushes
//! into, which players play too.
//!
//! ```text
//! FileDemuxer -> Pacer -> BroadcastSink -> Broadcast "file"    (the file, looped, at its own speed)
//! publisher (WHIP) -> WebRtcServer -> WebRtcSource -> BroadcastSink -> Broadcast "live"
//!
//! Broadcast "file" -> BroadcastSource -> WebRtcMuxer (path "file") -> WebRtcServer -> players (WHEP)
//! Broadcast "live" -> BroadcastSource -> WebRtcMuxer (path "live") -> WebRtcServer -> players (WHEP)
//!
//! with --rtmp:  OBS -> RtmpServer "live/stream" -> RtmpSource -> BroadcastSink -> Broadcast "live"
//!
//! with --record:  Broadcast "file" -> BroadcastSource -> HlsMuxer (event playlist)
//! ```
//!
//! The server hands each publisher out as a source of what it sends, and a
//! pipeline of the example's own writes it into the broadcast `live`. Each
//! broadcast is the example's own: the server's path of the same name reads
//! it, through the `WebRtcMuxer` its path makes, as a recording would.
//!
//! Open `http://127.0.0.1:8080/file` in a browser: the page offers to play
//! the path through WHEP, and the server answers. Each player is sent the
//! broadcast's packets as they are, from the newest keyframe; WebRTC
//! carries H.264, VP8, VP9 and AV1 pictures and Opus sound, so a file's AAC
//! is left out. Plays of the file without B-frames, as a browser takes
//! H.264.
//!
//!     cargo run -p webrtc_server -- input.mp4 [--udp 127.0.0.1:8189] [--http 127.0.0.1:8080] [--threads N] [--seconds N] [--record DIR] [--rtmp ADDRESS] [--candidate ADDRESS]...
//!
//! Both addresses default to this machine alone. To serve the network,
//! give the address the network reaches this machine at to `--udp` — a
//! browser is told to send there — and `0.0.0.0:8080` to `--http`. It runs
//! until Enter is pressed, or for `--seconds`, printing what each session
//! does. `--threads` is how many threads — each with a UDP port of its own,
//! from the one `--udp` names on — it runs its sessions on, two by default.
//! `--candidate` is an address the server tells browsers to send to in place
//! of the one it is bound to — the public address of a router that forwards
//! the UDP ports to this machine; a bare IP keeps the bound port; give it
//! once for each address a browser may reach it at.
//! `--record` also records what `file` plays, as it plays it, into an HLS
//! event playlist under `DIR` — `index.m3u8` and its fMP4 segments, playable
//! while it grows and finished as the server stops. `--rtmp` also takes an
//! RTMP publisher at `rtmp://ADDRESS/live` with the stream key `stream` (OBS's
//! "Custom" service) into `live`: its H.264 plays in a browser where the
//! encoder sends no B-frames (OBS: x264 or NVENC with B-frames 0), and its
//! AAC, which WebRTC does not carry, is left out.

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::net::{IpAddr, SocketAddr};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use media_pp::bus::BusEvent;
    use media_pp::driver::DriverRunner;
    use media_pp::element::{BoxSink, IntoSource};
    use media_pp::elements::{
        Broadcast, BroadcastError, BroadcastOptions, BroadcastSource, FileDemuxer, HlsMode,
        HlsMuxer, HlsOptions, Pacer, Publishers, RtmpServer, RtmpServerOptions, StreamInfo,
        WebRtcMuxer, WebRtcServer, WebRtcServerOptions,
    };
    use media_pp::pipeline::Pipeline;

    const USAGE: &str = "usage: webrtc_server <input> [--udp ADDRESS] [--http ADDRESS] [--threads N] [--seconds N] [--record DIR] [--rtmp ADDRESS] [--candidate ADDRESS]...";

    struct Args {
        input: String,
        udp: String,
        http: String,
        seconds: Option<u64>,
        threads: Option<usize>,
        record: Option<String>,
        rtmp: Option<String>,
        candidates: Vec<SocketAddr>,
    }

    /// `ADDRESS`, or an IP alone, which then keeps the port the server is
    /// bound to.
    fn candidate(text: &str) -> Option<SocketAddr> {
        text.parse()
            .ok()
            .or_else(|| text.parse::<IpAddr>().ok().map(|ip| SocketAddr::new(ip, 0)))
    }

    fn args() -> Option<Args> {
        let mut input = None;
        let mut udp = "127.0.0.1:8189".to_owned();
        let mut http = "127.0.0.1:8080".to_owned();
        let mut seconds = None;
        let mut threads = None;
        let mut record = None;
        let mut rtmp = None;
        let mut candidates = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--udp" => udp = args.next()?,
                "--http" => http = args.next()?,
                "--seconds" => seconds = Some(args.next()?.parse().ok()?),
                "--threads" => threads = Some(args.next()?.parse().ok().filter(|&n| n > 0)?),
                "--record" => record = Some(args.next()?),
                "--rtmp" => rtmp = Some(args.next()?),
                "--candidate" => candidates.push(candidate(&args.next()?)?),
                _ if input.is_none() => input = Some(arg),
                _ => return None,
            }
        }
        Some(Args {
            input: input?,
            udp,
            http,
            seconds,
            threads,
            record,
            rtmp,
            candidates,
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

        let mut options = WebRtcServerOptions::default();
        if let Some(threads) = args.threads {
            options.threads = threads;
        }
        options.candidates = args.candidates;
        let (server, handle) = WebRtcServer::bind("server", &args.udp, options)?;
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
        // A WHIP publisher of `live`, into what its players play.
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
        let http = handle.serve_http(&args.http)?;
        println!("open http://{}/file in a browser", http.local_addr());
        println!("WHEP: {}", http.whep_url("file"));
        println!(
            "WHIP: {} (then play http://{}/live)",
            http.whip_url("live"),
            http.local_addr()
        );
        if let Some((_, ingest)) = &rtmp {
            println!(
                "RTMP: {} (server rtmp://{}/live, key stream; then play http://{}/live)",
                ingest.url("live/stream"),
                ingest.local_addr(),
                http.local_addr()
            );
        }
        match args.seconds {
            Some(seconds) => println!("serving for {seconds} s"),
            None => println!("press Enter to stop"),
        }

        let stop = until_asked(args.seconds);
        loop {
            match stop.recv_timeout(Duration::from_millis(100)) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            while let Some(event) = handle.next_event(Duration::ZERO) {
                println!("{event:?}");
            }
        }

        if let Some(recorder) = recorder {
            // Finished, not stopped: the playlist is closed with its end.
            recorder.finish();
        }
        drop(http);
        // The paths' forwarders end with what they read.
        file.close();
        live.close();
        server.stop();
        camera.stop();
        if let Some((rtmp, _)) = &rtmp {
            rtmp.stop();
            for event in rtmp.bus().iter() {
                println!("{event}");
            }
        }
        for event in server.bus().iter() {
            println!("{event}");
        }
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

    /// Opens `muxer` with the tracks WebRTC carries of `streams`, and answers each one's output and
    /// sink.
    fn muxed(
        mut muxer: WebRtcMuxer,
        streams: &[StreamInfo],
    ) -> media_pp::Result<Vec<(usize, BoxSink)>> {
        let tracks = streams
            .iter()
            .filter(|stream| WebRtcServer::carries(stream))
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
        std::thread::spawn(move || {
            match seconds {
                Some(seconds) => std::thread::sleep(Duration::from_secs(seconds)),
                None => {
                    let mut line = String::new();
                    let _ = std::io::stdin().read_line(&mut line);
                }
            }
            let _ = tx.send(());
        });
        rx
    }
}
