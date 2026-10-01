//! Shows a pipeline's pictures as a camera other applications open — Teams,
//! Zoom, a browser, Windows' Camera app list it as "media-pp Windows Virtual
//! Camera" — until `q` + Enter:
//!
//! - with no file: `TestVideoSource -> Queue -> MfVirtualCamera`;
//! - with one: `FileDemuxer -> SwDecoder -> Queue -> Pacer -> MfVirtualCamera`,
//!   looping, so the file plays at its own speed for as long as this runs.
//!
//! `MfVirtualCamera` converts each picture to the size the reading
//! application picked. Windows 11 only, and the camera's DLL — the `vcam`
//! crate — must be registered once for the machine, from an elevated
//! prompt (`vcam/install.ps1`); without it this says so and exits.
//!
//!     cargo run -p virtual_camera -- [video.mp4]

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
    use std::io::BufRead;

    use media_pp::{
        elements::{
            FileDemuxer, MfVirtualCamera, Pacer, SwDecoder, TestVideoOptions, TestVideoSource,
        },
        ffmpeg,
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let camera = MfVirtualCamera::new("camera", "media-pp")?;
        let pipeline = match std::env::args().nth(1) {
            Some(path) => {
                let (source, _) = FileDemuxer::open("demux", &path)?;
                source.looping_handle().set_looping(true);
                let video = source.best(ffmpeg::media::Type::Video)?;
                let decoder = SwDecoder::new("decoder", video.parameters.clone())?;
                let (pipeline, ()) = Pipeline::new("virtual-camera", source, |source, ctx| {
                    let branch = ctx
                        .branch()
                        .pipe(decoder)
                        .queue("frames", 8)
                        .pipe(Pacer::new("pacer"))
                        .to(camera)?;
                    ctx.attach(source, video.index, branch)?;
                    Ok(())
                })?;
                println!("showing {path} as the camera, looping");
                pipeline
            }
            None => {
                let source = TestVideoSource::new(
                    "pattern",
                    TestVideoOptions {
                        width: 1280,
                        height: 720,
                        frame_rate: ffmpeg::Rational::new(30, 1),
                    },
                );
                let (pipeline, ()) = Pipeline::new("virtual-camera", source, |source, ctx| {
                    let branch = ctx.branch().queue("frames", 4).to(camera)?;
                    ctx.attach(source, 0, branch)?;
                    Ok(())
                })?;
                println!("showing a test pattern as the camera");
                pipeline
            }
        };
        pipeline.run()?;
        println!(
            "open \"media-pp Windows Virtual Camera\" in any application; \
             type `q` + Enter to stop"
        );

        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            if line.map_or(true, |line| line.trim() == "q") {
                break;
            }
        }
        pipeline.stop();
        while let Some(event) = pipeline.bus().try_recv() {
            println!("{event}");
        }
        Ok(())
    }
}
