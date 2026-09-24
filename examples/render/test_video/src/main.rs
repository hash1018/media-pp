//! TestVideoSource -> Queue -> Renderer: a synthetic moving-gradient stream,
//! no file/camera/decoder involved at all, presented in a native window. The
//! renderer — `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on
//! Linux — draws the source's YUV420P as it comes and uploads it itself, in a
//! window of its own. This proves the source, upload, and presentation path
//! works end to end without needing a real video source.
//!
//! No `Pacer` here, deliberately, as an experiment: `TestVideoSource`
//! self-paces with a drift-free absolute schedule (see its own docs) and
//! only a queue sits between it and the renderer. Testing
//! confirmed that schedule is enough on its own
//! for a vsync-locked renderer to stay smooth without a separate pacing
//! stage; `screen_preview_cpu` reached the same result after its source moved
//! from variable-rate emission to the same absolute scheduling scheme.
//!
//!     cargo run -p test_video

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("{} supports Windows and Linux only", env!("CARGO_PKG_NAME"));
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
    use media_pp::{
        bus::BusEvent,
        elements::{
            D3d12Gpu, D3d12WindowRenderer, TestVideoOptions, TestVideoSource, WindowOptions,
        },
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // Frames in system memory, which the renderer uploads itself: YUV420P
        // as the source makes it, with nothing in between.
        let gpu = D3d12Gpu::new()?;
        let window_options = WindowOptions {
            title: "media-pp test_video".into(),
            ..WindowOptions::default()
        };
        let (width, height) = (window_options.width, window_options.height);
        let (renderer, window) = D3d12WindowRenderer::open("renderer", &gpu, window_options)?;
        let shutdown = render_common::stop_on_close([window]);

        let options = TestVideoOptions {
            width,
            height,
            ..TestVideoOptions::default()
        };
        let source = TestVideoSource::new("test-video", options);

        let (pipeline, ()) = Pipeline::new("test-video", source, |source, ctx| {
            let branch = ctx
                .branch()
                .queue("frames", 8) // thread boundary so rendering doesn't block generation
                .to(renderer)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        // `run()` starts playback on a background thread and returns right
        // away — any failure (e.g. an unsupported pixel format from
        // `Renderer`) shows up as a `BusEvent::Error` here instead of through
        // a returned `Result`. `TestVideoSource` never reaches `Eos` on its
        // own — closing the window is what ends this (see `Ok(())` below,
        // reached when closing the window stops this published pipeline, or
        // when an error ends it below).
        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }

        pipeline.run()?;

        for event in pipeline.bus().iter() {
            println!("{event}");
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::{
        bus::BusEvent,
        elements::{
            TestVideoOptions, TestVideoSource, VulkanGpu, VulkanWindowRenderer, WindowOptions,
        },
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // Frames in system memory, which the renderer uploads itself: YUV420P
        // as the source makes it, with nothing in between.
        let gpu = VulkanGpu::new()?;
        let options = WindowOptions {
            title: "media-pp test_video".into(),
            ..WindowOptions::default()
        };
        let (width, height) = (options.width, options.height);
        let (renderer, window) = VulkanWindowRenderer::open("renderer", &gpu, options)?;
        let shutdown = render_common::stop_on_close([window]);

        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width,
                height,
                ..TestVideoOptions::default()
            },
        );
        let (pipeline, ()) = Pipeline::new("test-video", source, |source, ctx| {
            let branch = ctx.branch().queue("frames", 8).to(renderer)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        if shutdown.publish(std::slice::from_ref(&pipeline)) {
            return Ok(());
        }
        pipeline.run()?;
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
}
