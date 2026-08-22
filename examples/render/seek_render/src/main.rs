#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{
        io::{self, BufRead},
        thread,
        time::Duration,
    };

    use ffmpeg_next::media;
    use media_pp::{
        Error,
        bus::BusEvent,
        elements::{FileDemuxer, Pacer, SwDecoder},
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, Shutdown};
    use winit::raw_window_handle::RawWindowHandle;

    /// Demux -> SwDecoder -> Queue -> Pacer -> Renderer, same chain as
    /// `sw_decode_render`, plus a terminal prompt that reads timestamps and
    /// calls `Pipeline::seek` with them while the window is open — proves
    /// `seek` actually changes what's on screen, not just that it compiles.
    ///
    ///     cargo run -p seek_render -- path/to/video.mp4
    ///     (then use `pause`, `resume`, `seek 30`, `seek 1:15`, or `q`)
    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: seek_render <video.mp4>");
            std::process::exit(1);
        };

        render_common::run_window(
            "media-pp seek_render",
            1280,
            720,
            move |target, shutdown| {
                let RawWindowHandle::Win32(handle) = target.window else {
                    panic!("seek_render example only supports Windows");
                };
                play(
                    &path,
                    handle.hwnd.get(),
                    target.width,
                    target.height,
                    &shutdown,
                )
            },
        );
    }

    fn play(
        path: &str,
        hwnd: isize,
        width: u32,
        height: u32,
        shutdown: &Shutdown,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, streams) = FileDemuxer::open("demux", path)?;
        let video = streams
            .iter()
            .find(|s| s.kind == media::Type::Video)
            .ok_or_else(|| Error::Other("no video stream in file".into()))?;
        let params = source
            .stream_parameters(video.index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;
        let time_base = source
            .stream_time_base(video.index)
            .ok_or_else(|| Error::Other("stream disappeared".into()))?;

        let gpu = D3d12GpuContext::new().map_err(|e| Error::Other(format!("{e:?}")))?;

        let pipeline = Pipeline::new("seek-render", source, |source, ctx| {
            let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
            let pacer = Pacer::new("pacer", time_base, ctx.clock.clone())?;
            let renderer =
                render_common::d3d12_window_renderer("renderer", &gpu, hwnd, width, height)
                    .expect("failed to create renderer");
            let branch = ctx
                .branch()
                .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(pacer)
                .to(Box::new(renderer))?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — that's what makes this terminal prompt possible on the same
        // thread that would otherwise just be blocked waiting for it.
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        pipeline.run()?;

        // Reads seek requests for as long as the process lives, on its own
        // thread — a blocked stdin read can't also notice natural playback
        // completion, so it doesn't try to; the shared shell stops the
        // pipeline and exits once the playback worker has returned.
        {
            let pipeline = pipeline.clone();
            thread::spawn(move || read_seek_commands(&pipeline));
        }

        // Same output `log_events()` would print, but also calls `stop()` on
        // `Eos`/`Error` — errors no longer end the pipeline on their own (see
        // `BusEvent`'s docs), so without this an error here (e.g. the
        // renderer's GPU upload ring running out of slots) would just get
        // printed forever instead of ending playback. `Eos` calling `stop()`
        // too is a harmless no-op in this example (single video stream, one
        // `Eos` means everything's already finished) — a multi-stream
        // pipeline would need to wait for every branch's `Eos`, not stop on
        // the first one.
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
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
            if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
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
                    pipeline.seek(target);
                }
                None => eprintln!("couldn't parse {line:?} — use seconds (`30`) or mm:ss (`1:15`)"),
            }
        }
    }

    fn print_help() {
        println!("commands:");
        println!("  pause             pause playback");
        println!("  resume            resume playback");
        println!("  seek <seconds>    seek, for example `seek 30` or `seek 1:15`");
        println!("  help              print this help");
        println!("  q                 stop playback");
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
