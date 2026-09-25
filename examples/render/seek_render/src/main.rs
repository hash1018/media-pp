//! Demux -> SwDecoder -> Queue -> Pacer -> Renderer, the same chain as
//! `sw_decode_render`, plus a terminal prompt that reads timestamps and calls
//! `Pipeline::seek` with them while the window is open — proves `seek`
//! actually changes what's on screen, not just that it compiles. The renderer
//! — `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux —
//! uploads the decoded frames itself, through a `SwScaler` only for a stream
//! it cannot draw as it comes.
//!
//!     cargo run -p seek_render -- path/to/video.mp4
//!     (then use `pause`, `resume`, `seek 30`, `seek 1:15`, or `q`)

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example only supports Windows and Linux",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{
        io::{self, BufRead},
        thread,
        time::Duration,
    };

    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{D3d12Gpu, D3d12WindowRenderer, FileDemuxer, Pacer, SwDecoder, WindowOptions},
        pipeline::{Pipeline, SeekMode},
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: seek_render <video.mp4>");
            std::process::exit(1);
        };
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, _) = FileDemuxer::open("demux", &path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        // Decoded frames stay in system memory; the renderer uploads them.
        let gpu = D3d12Gpu::new()?;
        let (renderer, window) = D3d12WindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp seek_render".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);
        let to_drawable =
            media_pp::elements::SwScaler::if_needed("to-drawable", &params, &renderer)?;

        let (pipeline, ()) = Pipeline::new("seek-render", source, |source, ctx| {
            let mut branch = ctx
                .branch()
                .pipe(SwDecoder::new("decoder", params)?) // same thread as the demux — cheap enough not to need a queue
                .queue("frames", 32) // pacer sleeps on its own thread; let decode run ahead into this
                .pipe(Pacer::new("pacer"));
            if let Some(to_drawable) = to_drawable {
                branch = branch.pipe(to_drawable);
            }
            ctx.attach(source, video.index, branch.to(renderer)?)?;
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
        // completion, so it doesn't try to; the bus loop below ends when the
        // pipeline stops, and the process exits with it.
        {
            let pipeline = pipeline.clone();
            thread::spawn(move || read_seek_commands(&pipeline));
        }

        // Same output `log_events()` would print, but also calls `stop()` on
        // `Finished`/`Error` — errors no longer end the pipeline on their own (see
        // `BusEvent`'s docs), so without this an error here (e.g. the
        // renderer's GPU upload ring running out of slots) would just get
        // printed forever instead of ending playback. `Finished` rather than
        // `Eos`: every element that ends posts one, so the first `Eos` is
        // only the first thing to end.
        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
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

#[cfg(target_os = "linux")]
mod linux_example {
    use std::{
        io::{self, BufRead},
        thread,
        time::Duration,
    };

    use media_pp::ffmpeg::media;
    use media_pp::{
        bus::BusEvent,
        elements::{FileDemuxer, Pacer, SwDecoder, VulkanGpu, VulkanWindowRenderer, WindowOptions},
        pipeline::{Pipeline, SeekMode},
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: seek_render <video.mp4>");
            std::process::exit(1);
        };
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (source, _) = FileDemuxer::open("demux", &path)?;
        let video = source.best(media::Type::Video)?;
        let params = video.parameters.clone();

        let gpu = VulkanGpu::new()?;
        let (renderer, window) = VulkanWindowRenderer::open(
            "renderer",
            &gpu,
            WindowOptions {
                title: "media-pp seek_render".into(),
                ..WindowOptions::default()
            },
        )?;
        let shutdown = render_common::stop_on_close([window]);
        let to_drawable =
            media_pp::elements::SwScaler::if_needed("to-drawable", &params, &renderer)?;

        let (pipeline, ()) = Pipeline::new("seek-render", source, |source, ctx| {
            let mut branch = ctx
                .branch()
                .pipe(SwDecoder::new("decoder", params)?)
                .queue("frames", 32)
                .pipe(Pacer::new("pacer"));
            if let Some(to_drawable) = to_drawable {
                branch = branch.pipe(to_drawable);
            }
            ctx.attach(source, video.index, branch.to(renderer)?)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;
        {
            let pipeline = pipeline.clone();
            thread::spawn(move || read_seek_commands(&pipeline));
        }
        drain_bus(&pipeline);
        Ok(())
    }

    fn drain_bus(pipeline: &Pipeline) {
        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
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
        println!("  pause             pause playback");
        println!("  resume            resume playback");
        println!("  seek <seconds>    seek, for example `seek 30` or `seek 1:15`");
        println!("  help              print this help");
        println!("  q                 stop playback");
    }

    fn parse_timestamp(value: &str) -> Option<Duration> {
        let seconds = match value.split_once(':') {
            Some((minutes, seconds)) => {
                minutes.parse::<f64>().ok()? * 60.0 + seconds.parse::<f64>().ok()?
            }
            None => value.parse::<f64>().ok()?,
        };
        if seconds.is_finite() && seconds >= 0.0 {
            Some(Duration::from_secs_f64(seconds))
        } else {
            None
        }
    }
}
