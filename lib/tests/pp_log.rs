use std::{
    fs,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use ::log as host_log;
use media_pp::log::{self, Level, LogInitError};
use media_pp::pp_log::{PpLog, pp_debug, pp_error, pp_info, pp_trace, pp_warn};

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

struct LoggedValue {
    pp_log: PpLog,
}

impl LoggedValue {
    fn emit(&self) {
        pp_info!(self, "self form");
    }
}

#[test]
fn public_pp_log_macros_accept_every_supported_target_form() {
    let pp_log = PpLog::new("CustomElement", "element", Some("pipeline"));

    pp_info!(pp_log: &pp_log, "context form");
    pp_debug!(pp_log: &pp_log, "debug form");
    pp_warn!(pp_log: &pp_log, "warn form");
    pp_error!(pp_log: &pp_log, "error form");
    pp_trace!(pp_log: &pp_log, "trace form");

    LoggedValue { pp_log }.emit();
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

    assert!(contents.contains(
        " INFO [pipeline_id=test-pipeline] [element=TestElement] [name=test-name] private-info-marker"
    ));
    assert!(!contents.contains("ThreadId"));
    assert!(!contents.contains("private-debug-marker"));

    fs::remove_dir_all(log_dir).expect("temporary log directory must be removable");
}
