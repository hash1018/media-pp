//! Shows a pipeline's pictures as a camera other applications open — a
//! browser, Zoom, Teams, Windows' Camera app — until `q` + Enter:
//!
//! - with no file: `TestVideoSource -> Queue -> camera`;
//! - with one: `FileDemuxer -> SwDecoder -> Queue -> Pacer -> camera`,
//!   looping, so the file plays at its own speed for as long as this runs.
//!
//! The camera is the platform's:
//!
//! - Windows: `MfVirtualCamera`, listed as "media-pp Windows Virtual
//!   Camera", which converts each picture to the size the reading
//!   application picked. Windows 11 only, and the camera's DLL — the `vcam`
//!   crate — must be registered once for the machine, from an elevated
//!   prompt (`vcam/install.ps1`).
//! - Linux: `V4l2VirtualCamera`, writing 1280x720 at 30 fps into the first
//!   free v4l2loopback device, listed under the label the module was loaded
//!   with. The module must be loaded first — for one,
//!   `sudo modprobe v4l2loopback exclusive_caps=1 card_label=media-pp`.
//!
//! Without its camera this says what is missing and exits.
//!
//!     cargo run -p virtual_camera -- [video.mp4]

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example supports Windows and Linux",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn main() -> impl std::process::Termination {
    example::run()
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod example {
    use std::io::BufRead;

    use media_pp::{
        element::{BoxSink, IntoTerminal},
        elements::{FileDemuxer, Pacer, SwDecoder, TestVideoOptions, TestVideoSource},
        ffmpeg,
        pipeline::Pipeline,
    };

    /// The camera, and the name applications list it under.
    #[cfg(target_os = "windows")]
    fn camera() -> media_pp::Result<(BoxSink, String)> {
        let camera = media_pp::elements::MfVirtualCamera::new("camera", "media-pp")?;
        Ok((
            camera.into_terminal(),
            "media-pp Windows Virtual Camera".to_owned(),
        ))
    }

    /// The camera, and the name applications list it under.
    #[cfg(target_os = "linux")]
    fn camera() -> media_pp::Result<(BoxSink, String)> {
        use media_pp::elements::V4l2VirtualCamera;

        let Some(device) = V4l2VirtualCamera::list_devices()
            .map_err(|error| media_pp::Error::Other(format!("listing cameras: {error}")))?
            .into_iter()
            .next()
        else {
            return Err(media_pp::Error::Other(
                "no free v4l2loopback device: load the module first, for one \
                 `sudo modprobe v4l2loopback exclusive_caps=1 card_label=media-pp`"
                    .to_owned(),
            ));
        };
        let camera = V4l2VirtualCamera::new(
            "camera",
            &device.id,
            1280,
            720,
            ffmpeg::Rational::new(30, 1),
        )?;
        Ok((
            camera.into_terminal(),
            format!("{} ({})", device.name, device.id),
        ))
    }

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let (camera, listed_as) = camera()?;
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
        println!("open \"{listed_as}\" in any application; type `q` + Enter to stop");

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
