//! Keeps one fixed branch from initial wiring, then adds and removes a
//! second branch through `TeeHandle` while synthetic video is flowing.
//!
//!     cargo run -p dynamic_tee

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{sync::atomic::Ordering, thread, time::Duration};

    use media_pp::{
        Error,
        elements::{FrameCounter, TeeBuilder, TestVideoOptions, TestVideoSource},
        pipeline::Pipeline,
    };

    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let source = TestVideoSource::new("video", TestVideoOptions::default());
        let (initial_counter, initial_count) = FrameCounter::new("initial-counter");
        let mut tee_handle = None;

        let pipeline = Pipeline::new("dynamic-tee", source, |source, ctx| {
            let initial_branch = ctx.branch().to(Box::new(initial_counter))?;
            let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone())
                .branch(initial_branch)
                .build_dynamic()?;
            ctx.attach(source, 0, tee_branch)?;
            tee_handle = Some(handle);
            Ok(())
        })?;
        let tee_handle = tee_handle.expect("wire closure must provide the TeeHandle");

        pipeline.run()?;
        thread::sleep(Duration::from_millis(500));

        let (dynamic_counter, dynamic_count) = FrameCounter::new("dynamic-counter");
        let dynamic_branch = tee_handle
            .branch()
            .ok_or_else(|| Error::Other("tee is no longer alive".into()))?
            .to(Box::new(dynamic_counter))?;
        let branch_id = tee_handle.attach(dynamic_branch)?;
        println!("attached runtime branch {branch_id}");

        thread::sleep(Duration::from_secs(1));
        tee_handle.detach(branch_id)?;
        println!("detached runtime branch {branch_id}");

        thread::sleep(Duration::from_millis(500));
        pipeline.stop();
        pipeline.bus().log_events();

        println!(
            "frames: initial={}, dynamic={}",
            initial_count.load(Ordering::Relaxed),
            dynamic_count.load(Ordering::Relaxed)
        );
        Ok(())
    }
}
