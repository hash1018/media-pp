//! Waits for a publisher to push an RTSP stream into it — an encoder, a
//! camera that pushes, `ffmpeg -f rtsp`, or the `rtsp_serve` example — and
//! counts the packets of every stream the publisher announced, once a
//! second, until it finishes: `RtspSource::listen -> PacketCounter`, one
//! counter per stream.
//!
//! It receives one publisher and then ends: when the publisher closes its
//! connection the counts stop, `finished` is printed and it exits. A
//! publisher that goes silent, or none arriving within a minute, is an
//! error and a non-zero exit. Nothing is decoded, so any codec the
//! publisher sends is counted.
//!
//!     cargo run -p rtsp_listen -- rtsp://0.0.0.0:8554/live
//!     ffmpeg -re -i video.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live
//!
//! The publisher picks the transport. For `-rtsp_transport udp`, listen at
//! the address the publisher reaches this machine at —
//! `rtsp://127.0.0.1:8554/live` for one on the same machine — not
//! `0.0.0.0`, which FFmpeg's listener would hand the publisher as where to
//! send its packets.

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::sync::mpsc::RecvTimeoutError;
    use std::time::{Duration, Instant};

    use media_pp::{
        bus::BusEvent,
        elements::{CounterHandle, PacketCounter, RtspListenOptions, RtspSource},
        pipeline::Pipeline,
    };

    /// How long to wait for a publisher before giving up.
    const WAIT: Duration = Duration::from_secs(60);

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(url) = std::env::args().nth(1) else {
            eprintln!("usage: rtsp_listen <rtsp://host:port/path>");
            std::process::exit(1);
        };

        println!("listening at {url} for a publisher, for up to {WAIT:?} ...");
        let options = RtspListenOptions {
            wait: WAIT,
            ..Default::default()
        };
        let (source, streams) = RtspSource::listen("ingest", &url, options)?;
        println!("publisher connected, {} stream(s):", streams.len());
        for stream in &streams {
            println!(
                "  [{}] {:?} {:?}, time base {}",
                stream.index,
                stream.kind,
                stream.parameters.id(),
                stream.time_base
            );
        }

        let (pipeline, counts) = Pipeline::new("rtsp-listen", source, |source, ctx| {
            let mut counts = Vec::new();
            for stream in &streams {
                let (counter, count) = PacketCounter::new(format!("count-{}", stream.index));
                let branch = ctx.branch().to(counter)?;
                ctx.attach(source, stream.index, branch)?;
                counts.push(count);
            }
            Ok(counts)
        })?;
        pipeline.run()?;

        // The source ends when the publisher closes its connection, and the
        // pipeline posts `Finished` once every counter has had its end; a
        // publisher gone silent is an error instead. Neither reconnects:
        // `RtspSource::listen` again would take the next publisher.
        let mut failure = None;
        let mut reported = Instant::now();
        loop {
            match pipeline.bus().recv_timeout(Duration::from_millis(200)) {
                Ok(BusEvent::Finished) => {
                    println!("finished: the publisher closed its connection");
                    break;
                }
                Ok(BusEvent::Error { name, error, .. }) => {
                    eprintln!("[{name}] error: {error}");
                    failure = Some(error);
                    break;
                }
                Ok(event) => println!("{event}"),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            if reported.elapsed() >= Duration::from_secs(1) {
                report(&counts);
                reported = Instant::now();
            }
        }
        pipeline.stop();
        report(&counts);

        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn report(counts: &[CounterHandle]) {
        let counts: Vec<String> = counts
            .iter()
            .enumerate()
            .map(|(index, count)| format!("[{index}] {}", count.get()))
            .collect();
        println!("packets: {}", counts.join(", "));
    }
}
