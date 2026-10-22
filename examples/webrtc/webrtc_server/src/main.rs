//! A WebRTC server a browser plays: a file played to any number of WebRTC
//! players as a camera would, and a second path a WHIP publisher pushes
//! into, which players play too.
//!
//! ```text
//! FileDemuxer -> Pacer -> BroadcastSink    (the file, looped, at its own speed)
//!                              |
//!                      Broadcast "file" ----\
//!                                            WebRtcServer -> players (WHEP)
//! publisher (WHIP) -> Broadcast "live" -----/
//!                            ^
//!   with --rtmp: OBS -> RtmpServer "live/stream"
//!
//! with --record:  Broadcast "file" -> BroadcastSource -> HlsMuxer (event playlist)
//! ```
//!
//! Open `http://127.0.0.1:8080/file` in a browser: the page offers to play
//! the path through WHEP, and the server answers. Each player is sent the
//! broadcast's packets as they are, from the newest keyframe; WebRTC
//! carries H.264, VP8, VP9 and AV1 pictures and Opus sound, so a file's AAC
//! is left out. Plays of the file without B-frames, as a browser takes
//! H.264.
//!
//!     cargo run -p webrtc_server -- input.mp4 [--udp 127.0.0.1:8189] [--http 127.0.0.1:8080] [--threads N] [--seconds N] [--record DIR] [--rtmp ADDRESS]
//!
//! Both addresses default to this machine alone. To serve the network,
//! give the address the network reaches this machine at to `--udp` — a
//! browser is told to send there — and `0.0.0.0:8080` to `--http`. It runs
//! until Enter is pressed, or for `--seconds`, printing what each session
//! does. `--threads` is how many threads — each with a UDP port of its own,
//! from the one `--udp` names on — it runs its sessions on, two by default.
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
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use media_pp::driver::DriverRunner;
    use media_pp::elements::{
        Broadcast, BroadcastOptions, BroadcastSource, FileDemuxer, HlsMode, HlsMuxer, HlsOptions,
        Pacer, RtmpServer, RtmpServerOptions, StreamInfo, WebRtcPathMode, WebRtcServer,
        WebRtcServerOptions,
    };
    use media_pp::pipeline::Pipeline;

    const USAGE: &str = "usage: webrtc_server <input> [--udp ADDRESS] [--http ADDRESS] [--threads N] [--seconds N] [--record DIR] [--rtmp ADDRESS]";

    struct Args {
        input: String,
        udp: String,
        http: String,
        seconds: Option<u64>,
        threads: Option<usize>,
        record: Option<String>,
        rtmp: Option<String>,
    }

    fn args() -> Option<Args> {
        let mut input = None;
        let mut udp = "127.0.0.1:8189".to_owned();
        let mut http = "127.0.0.1:8080".to_owned();
        let mut seconds = None;
        let mut threads = None;
        let mut record = None;
        let mut rtmp = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--udp" => udp = args.next()?,
                "--http" => http = args.next()?,
                "--seconds" => seconds = Some(args.next()?.parse().ok()?),
                "--threads" => threads = Some(args.next()?.parse().ok().filter(|&n| n > 0)?),
                "--record" => record = Some(args.next()?),
                "--rtmp" => rtmp = Some(args.next()?),
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
        let (server, handle) = WebRtcServer::bind("server", &args.udp, options)?;
        let file = Broadcast::new("file", BroadcastOptions::default());
        let live = Broadcast::new("live", BroadcastOptions::default());
        handle.add_path("file", &file, WebRtcPathMode::Play)?;
        handle.add_path("live", &live, WebRtcPathMode::Publish)?;
        // RTMP publishers into the same broadcast: OBS's server
        // `rtmp://ADDRESS/live`, its stream key `stream`.
        let rtmp = match &args.rtmp {
            Some(address) => {
                let (rtmp, ingest) =
                    RtmpServer::bind("rtmp", address, RtmpServerOptions::default())?;
                ingest.add_path("live/stream", &live)?;
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
