//! `media_pp::log::init` succeeds at most once per process, so this file owns
//! its own test binary: any other test emitting through `PpLog` in the same
//! process could otherwise land records in the file these assertions read.

use std::{
    fs,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use ::log as host_log;
use media_pp::log::{self, Level, LogInitError};
use media_pp::pp_log::{PpLog, pp_debug, pp_info};

static HOST_LOG_COUNT: AtomicUsize = AtomicUsize::new(0);
static HOST_LOGGER: CountingHostLogger = CountingHostLogger;

struct CountingHostLogger;

impl host_log::Log for CountingHostLogger {
    fn enabled(&self, _metadata: &host_log::Metadata<'_>) -> bool {
        true
    }

    fn log(&self, _record: &host_log::Record<'_>) {
        HOST_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    fn flush(&self) {}
}

#[test]
fn private_file_logger_filters_and_flushes_on_guard_drop() {
    host_log::set_logger(&HOST_LOGGER).expect("host test logger must install exactly once");
    host_log::set_max_level(host_log::LevelFilter::Trace);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_nanos();
    let log_dir = std::env::temp_dir().join(format!(
        "media_pp_private_log_{}_{}",
        std::process::id(),
        unique
    ));
    let log_path = log_dir.to_string_lossy();
    let guard = log::init("isolated", &log_path, Level::Info, 2)
        .expect("private file logger must initialize");
    let pp_log = PpLog::new("TestElement", "test-name", Some("test-pipeline"));
    let debug_argument_evaluated = AtomicBool::new(false);

    pp_info!(pp_log: &pp_log, "private-info-marker");
    pp_debug!(
        pp_log: &pp_log,
        "private-debug-marker {}",
        debug_argument_evaluated.swap(true, Ordering::Relaxed)
    );

    assert_eq!(HOST_LOG_COUNT.load(Ordering::Relaxed), 0);
    assert!(!debug_argument_evaluated.load(Ordering::Relaxed));
    assert_eq!(guard.dropped_lines(), 0);
    drop(guard);

    // A log call that begins after the guard is dropped is ignored: it must
    // neither reach the file nor fault on the writer the drop released.
    pp_info!(pp_log: &pp_log, "after-guard-marker");

    assert!(matches!(
        log::init("isolated", &log_path, Level::Info, 2),
        Err(LogInitError::AlreadyInitialized)
    ));

    let log_file = fs::read_dir(&log_dir)
        .expect("temporary log directory must remain readable")
        .map(|entry| entry.expect("log directory entry must be readable").path())
        .find(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("isolated."))
        })
        .expect("the rolling appender must create a log file");
    let contents = fs::read_to_string(log_file).expect("log file must contain UTF-8 text");

    // The thread field carries a per-process sequence number, so match the
    // two halves around it rather than pinning the number itself.
    assert!(contents.contains(" INFO [thread="));
    assert!(contents.contains(
        "] [pipeline_id=test-pipeline] [element=TestElement] [name=test-name] private-info-marker"
    ));
    assert!(!contents.contains("private-debug-marker"));
    assert!(!contents.contains("after-guard-marker"));

    fs::remove_dir_all(log_dir).expect("temporary log directory must be removable");
}
