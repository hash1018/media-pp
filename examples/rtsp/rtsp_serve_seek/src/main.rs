//! Same Demux -> Queue -> Pacer -> RtspMuxer chain as `rtsp_serve`, plus a
//! terminal prompt (same as `seek_render`'s) that reads timestamps and calls
//! `Pipeline::seek` with them while the stream is live — lets a viewer
//! jump around a live-served RTSP stream instead of only watching it play
//! straight through.
//!
//!     cargo run -p rtsp_serve_seek -- path/to/video.mp4 rtsp://127.0.0.1:8554/stream
//!     ffplay rtsp://127.0.0.1:8554/stream                    # in another terminal
//!     (then use `pause`, `resume`, `seek 30`, `seek 1:15`, or `q`)

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{
        io::{self, BufRead},
        thread,
        time::Duration,
    };

    use media_pp::ffmpeg::media;
    use media_pp::{
        Error,
        bus::BusEvent,
        element::ElementType,
        elements::{FileDemuxer, Pacer, RtspMuxer, RtspTransport},
        pipeline::{Pipeline, SeekMode},
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
            eprintln!("usage: rtsp_serve_seek <video.mp4> [rtsp://host:port/path]");
            std::process::exit(1);
        };
        let url = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "rtsp://127.0.0.1:8554/stream".into());

        let (source, streams) = FileDemuxer::open("demux", &path)?;
        let video = streams
            .iter()
            .find(|s| s.kind == media::Type::Video)
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let video_index = video.index;
        let video_params = source
            .stream_parameters(video_index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;
        let video_time_base = source
            .stream_time_base(video_index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;

        // Optional, as in `rtsp_serve`. Two tracks are also what makes the
        // seek below worth watching here: each rebases its own published
        // timestamps onto its own last ones.
        let audio_track = match streams.iter().find(|s| s.kind == media::Type::Audio) {
            Some(audio) => {
                let params = source
                    .stream_parameters(audio.index)
                    .ok_or_else(|| Error::Other("stream disappeared".into()))?;
                let time_base = source
                    .stream_time_base(audio.index)
                    .ok_or_else(|| Error::Other("stream disappeared".into()))?;
                Some((audio.index, params, time_base))
            }
            None => None,
        };
        let tracks = 1 + usize::from(audio_track.is_some());

        println!("publishing to {url} (the RTSP server must already be running) ...");

        let pipeline = Pipeline::new("rtsp-publish-seek", source, |source, ctx| {
            let mut muxer = RtspMuxer::create(&url, RtspTransport::Tcp)?;
            muxer.add_stream("video", video_params, video_time_base)?;
            let audio = match audio_track {
                Some((index, params, time_base)) => {
                    muxer.add_stream("audio", params, time_base)?;
                    Some((index, time_base))
                }
                None => None,
            };
            let mut sinks = muxer.open()?;
            let audio_sink = audio.map(|_| sinks.pop().expect("audio was registered second"));
            let video_sink = sinks.pop().expect("video was registered first");

            let branch = ctx
                .branch()
                .queue("video-packets", 32) // pacer sleeps on its own thread; let demux run ahead into this
                .pipe(Pacer::new("video-pacer", video_time_base)?)
                .to(video_sink)?;
            ctx.attach(source, video_index, branch)?;

            if let (Some((index, time_base)), Some(audio_sink)) = (audio, audio_sink) {
                let branch = ctx
                    .branch()
                    .queue("audio-packets", 32)
                    .pipe(Pacer::new("audio-pacer", time_base)?)
                    .to(audio_sink)?;
                ctx.attach(source, index, branch)?;
            }
            Ok(())
        })?;

        println!("publishing — connect a viewer to `{url}`");
        // `run()` starts publishing on a background thread and returns right
        // away — that's what makes this terminal prompt possible on the same
        // thread that would otherwise just be blocked waiting for it.
        pipeline.run()?;

        // Reads seek requests for as long as the process lives, on its own
        // thread — a blocked stdin read can't also notice natural playback
        // completion, so it doesn't try to; the whole process (this thread
        // included) exits once the bus loop below returns.
        {
            let pipeline = pipeline.clone();
            thread::spawn(move || read_seek_commands(&pipeline));
        }

        // Errors no longer end the pipeline on their own (see `BusEvent`'s
        // docs) — watch for one here and `stop()`, or this would just keep
        // trying to publish into a broken server connection forever
        // instead of exiting.
        //
        // Natural completion waits for *every* track, not the first: the
        // session's trailer is only written once all of them report `Eos`,
        // so stopping on whichever finishes first would cut the other off.
        let mut finished = 0;
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos {
                    name,
                    element_type: ElementType::RtspMuxer,
                } => {
                    finished += 1;
                    println!("[{name}] eos ({finished}/{tracks})");
                }
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                BusEvent::Seeked {
                    name,
                    requested,
                    landed,
                    ..
                } => println!("[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"),
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            if finished == tracks || matches!(event, BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }

    fn read_seek_commands(pipeline: &Pipeline) {
        print_help();
        for line in io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.eq_ignore_ascii_case("q") {
                pipeline.stop();
                break;
            }
            if line.eq_ignore_ascii_case("pause") {
                pipeline.pause();
                println!("paused");
                continue;
            }
            if line.eq_ignore_ascii_case("resume") {
                pipeline.resume();
                println!("resumed");
                continue;
            }
            if line.eq_ignore_ascii_case("help") {
                print_help();
                continue;
            }
            let value = line.strip_prefix("seek ").unwrap_or(line).trim();
            match parse_timestamp(value) {
                Some(target) => {
                    println!("seeking to {target:.2?}...");
                    if let Err(error) = pipeline.seek(target, SeekMode::Accurate) {
                        eprintln!("seek rejected: {error}");
                    }
                }
                None => eprintln!("couldn't parse {line:?} — use seconds (`30`) or mm:ss (`1:15`)"),
            }
        }
    }

    fn print_help() {
        println!("commands:");
        println!("  pause             pause publishing");
        println!("  resume            resume publishing");
        println!("  seek <seconds>    seek, for example `seek 30` or `seek 1:15`");
        println!("  help              print this help");
        println!("  q                 stop publishing");
    }

    /// `"90"` (plain seconds) or `"1:30"` (mm:ss) -> `Duration`. Fractional
    /// seconds work in both forms (`"1.5"`, `"1:01.5"`).
    fn parse_timestamp(s: &str) -> Option<Duration> {
        let secs = match s.split_once(':') {
            Some((min, sec)) => min.parse::<f64>().ok()? * 60.0 + sec.parse::<f64>().ok()?,
            None => s.parse::<f64>().ok()?,
        };
        if secs.is_finite() && secs >= 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    }
}
