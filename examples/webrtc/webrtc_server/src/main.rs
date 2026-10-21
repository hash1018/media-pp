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
//! ```
//!
//! Open `http://127.0.0.1:8080/file` in a browser: the page offers to play
//! the path through WHEP, and the server answers. Each player is sent the
//! broadcast's packets as they are, from the newest keyframe; WebRTC
//! carries H.264, VP8, VP9 and AV1 pictures and Opus sound, so a file's AAC
//! is left out. Plays of the file without B-frames, as a browser takes
//! H.264.
//!
//!     cargo run -p webrtc_server -- input.mp4 [--udp 127.0.0.1:8189] [--http 127.0.0.1:8080] [--seconds N]
//!
//! Both addresses default to this machine alone. To serve the network,
//! give the address the network reaches this machine at to `--udp` — a
//! browser is told to send there — and `0.0.0.0:8080` to `--http`. It runs
//! until Enter is pressed, or for `--seconds`, printing what each session
//! does.

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::sync::mpsc;
    use std::time::Duration;

    use media_pp::driver::DriverRunner;
    use media_pp::elements::{
        Broadcast, BroadcastOptions, FileDemuxer, Pacer, TrackFormat, WebRtcPathMode, WebRtcServer,
        WebRtcServerOptions,
    };
    use media_pp::pipeline::Pipeline;

    const USAGE: &str =
        "usage: webrtc_server <input> [--udp ADDRESS] [--http ADDRESS] [--seconds N]";

    struct Args {
        input: String,
        udp: String,
        http: String,
        seconds: Option<u64>,
    }

    fn args() -> Option<Args> {
        let mut input = None;
        let mut udp = "127.0.0.1:8189".to_owned();
        let mut http = "127.0.0.1:8080".to_owned();
        let mut seconds = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--udp" => udp = args.next()?,
                "--http" => http = args.next()?,
                "--seconds" => seconds = Some(args.next()?.parse().ok()?),
                _ if input.is_none() => input = Some(arg),
                _ => return None,
            }
        }
        Some(Args {
            input: input?,
            udp,
            http,
            seconds,
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

        let (server, handle) =
            WebRtcServer::bind("server", &args.udp, WebRtcServerOptions::default())?;
        let file = Broadcast::new("file", BroadcastOptions::default());
        let live = Broadcast::new("live", BroadcastOptions::default());
        handle.add_path("file", &file, WebRtcPathMode::Play)?;
        handle.add_path("live", &live, WebRtcPathMode::Publish)?;

        // The file as a camera: looped, at its own speed, into its broadcast.
        let (demuxer, streams) = FileDemuxer::open("file", &args.input)?;
        demuxer.looping_handle().set_looping(true);
        let tracks: Vec<TrackFormat> = streams.iter().map(TrackFormat::from).collect();
        let sinks = file.publish(&tracks)?;
        let (camera, ()) = Pipeline::new("file", demuxer, |source, ctx| {
            for (stream, sink) in streams.iter().zip(sinks) {
                let branch = ctx
                    .branch()
                    .queue(format!("packets-{}", stream.index), 64)
                    .pipe(Pacer::new(format!("pacer-{}", stream.index)))
                    .to(sink)?;
                ctx.attach(source, stream.index, branch)?;
            }
            Ok(())
        })?;
        camera.run()?;

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

        drop(http);
        server.stop();
        camera.stop();
        for event in server.bus().iter() {
            println!("{event}");
        }
        println!("stopped");
        Ok(())
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
