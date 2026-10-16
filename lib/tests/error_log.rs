//! What the log records for a failure reported by an element other than the
//! one that failed. In a test binary of its own, as the private logger is
//! installed once per process.

use std::{
    fs,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use media_pp::{
    buffer::MediaBuffer,
    bus::BusEvent,
    elements::{AppSink, AppSource},
    ffmpeg,
    log::{self, Level},
    pipeline::Pipeline,
};

/// A sink's failure crosses a queue, which posts it. The bus has always named
/// the sink; the log named the queue alone, so a reader of the log could not
/// tell what had failed. The record now goes under the sink, and says the
/// queue reported it.
#[test]
fn a_failure_reported_by_a_queue_is_logged_under_the_element_that_failed() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_nanos();
    let log_dir = std::env::temp_dir().join(format!(
        "media_pp_error_log_{}_{}",
        std::process::id(),
        unique
    ));
    let guard =
        log::init("error", &log_dir, Level::Error, 1).expect("private file logger must initialize");

    let failed = Arc::new(AtomicBool::new(false));
    let fail_once = failed.clone();
    let (source, handle) = AppSource::new("source", 4);
    let (pipeline, ()) = Pipeline::new("error-test", source, |source, context| {
        let branch = context
            .branch()
            .queue("queue", 4)
            .to(AppSink::new("sink", move |_buf| {
                if fail_once.swap(true, Ordering::SeqCst) {
                    Ok(())
                } else {
                    Err(media_pp::Error::Other("the sink broke".into()))
                }
            }))?;
        context.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline must build");

    pipeline.run().unwrap();
    for _ in 0..2 {
        handle
            .push(MediaBuffer::packet(ffmpeg::Packet::empty()))
            .expect("source must take the buffer");
    }
    handle.finish().expect("source must take the end");
    drop(handle);
    let events: Vec<_> = pipeline.bus().iter().collect();
    assert!(
        failed.load(Ordering::SeqCst),
        "the sink was never handed a buffer"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            BusEvent::Error { name, .. } if &**name == "sink"
        )),
        "the bus names the sink: {events:?}"
    );

    assert_eq!(guard.dropped_lines(), 0);
    drop(guard);
    let log_file = fs::read_dir(&log_dir)
        .expect("temporary log directory must remain readable")
        .map(|entry| entry.expect("log directory entry must be readable").path())
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("error."))
        })
        .expect("the rolling appender must create a log file");
    let contents = fs::read_to_string(log_file).expect("log file must contain UTF-8 text");
    let record = contents
        .lines()
        .find(|line| line.contains("the sink broke"))
        .unwrap_or_else(|| panic!("the failure was not logged:\n{contents}"));
    assert!(
        record.contains("[element=AppSink] [name=sink]"),
        "logged under the sink: {record}"
    );
    assert!(
        record.contains("(reported by Queue(queue))"),
        "and says which element reported it: {record}"
    );
    fs::remove_dir_all(&log_dir).ok();
}
