//! Opt-in file logging owned exclusively by `media-pp`.
//!
//! [`crate::pp_log`]'s macros write directly to a private non-blocking file
//! writer. They never install or emit through the process-global `log` or
//! `tracing` facilities, so an embedding application's logs cannot enter these
//! files and `media-pp` records cannot enter the application's logger.

use std::{
    fmt::{self, Write as _},
    fs,
    io::Write as _,
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use thiserror::Error as ThisError;
use time::OffsetDateTime;
use tracing_appender::{
    non_blocking::{ErrorCounter, NonBlocking, NonBlockingBuilder, WorkerGuard},
    rolling::{InitError, RollingFileAppender, Rotation},
};

use crate::pp_log::PpLog;

const BUFFERED_LINES_LIMIT: usize = 4096;

static LOGGER: OnceLock<PrivateLogger> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Severity threshold for the private `media-pp` file logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        })
    }
}

#[derive(Debug, ThisError)]
pub enum LogInitError {
    #[error("the media-pp file logger has already been initialized")]
    AlreadyInitialized,

    #[error("failed to create log directory `{path}`: {source}")]
    LogDirectory {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to create log file appender: {0}")]
    FileAppender(#[from] InitError),
}

/// Owns the private logging worker installed by [`init`].
///
/// Keep this value alive for as long as logging should remain active. Dropping
/// it first disables new records, then flushes queued records and joins the
/// worker. It is deliberately not stored in a static because Rust does not drop
/// static values at process exit, which would make final-record flushing
/// unreliable.
pub struct LogGuard {
    active: Arc<AtomicBool>,
    error_counter: ErrorCounter,
    worker: Option<WorkerGuard>,
}

impl LogGuard {
    /// Number of complete log records discarded because the bounded writer
    /// queue was full.
    pub fn dropped_lines(&self) -> usize {
        self.error_counter.dropped_lines()
    }
}

impl Drop for LogGuard {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
        self.worker.take();
    }
}

struct PrivateLogger {
    level: Level,
    active: Arc<AtomicBool>,
    writer: NonBlocking,
}

/// Starts the private `media-pp` file logger.
///
/// Records are appended to `{log_prefix}.{date}.log` in `log_path`. Files
/// rotate daily and only the newest `max_log_files` are retained. The bounded
/// writer is lossy by design: if disk output falls behind a burst of records,
/// the producing media thread drops that record instead of blocking. Use
/// [`LogGuard::dropped_lines`] to inspect the count.
///
/// This does not install a global `log` logger or `tracing` subscriber. It can
/// coexist with any logger installed by the embedding application.
///
/// The returned [`LogGuard`] must be retained for as long as logging is needed.
/// Dropping it flushes queued records and permanently stops this one-shot
/// logger; a process cannot initialize it again.
pub fn init(
    log_prefix: &str,
    log_path: &str,
    level: Level,
    max_log_files: usize,
) -> Result<LogGuard, LogInitError> {
    let _init_guard = INIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    if LOGGER.get().is_some() {
        return Err(LogInitError::AlreadyInitialized);
    }

    fs::create_dir_all(log_path).map_err(|source| LogInitError::LogDirectory {
        path: log_path.into(),
        source,
    })?;

    let file_appender = RollingFileAppender::builder()
        .filename_prefix(log_prefix)
        .filename_suffix("log")
        .rotation(Rotation::DAILY)
        .max_log_files(max_log_files)
        .build(log_path)?;

    let (writer, worker) = NonBlockingBuilder::default()
        .buffered_lines_limit(BUFFERED_LINES_LIMIT)
        .lossy(true)
        .thread_name("media-pp-log")
        .finish(file_appender);
    let error_counter = writer.error_counter();
    let active = Arc::new(AtomicBool::new(true));

    let logger = PrivateLogger {
        level,
        active: active.clone(),
        writer,
    };
    if LOGGER.set(logger).is_err() {
        return Err(LogInitError::AlreadyInitialized);
    }

    Ok(LogGuard {
        active,
        error_counter,
        worker: Some(worker),
    })
}

#[doc(hidden)]
#[inline]
pub fn enabled(level: Level) -> bool {
    LOGGER
        .get()
        .is_some_and(|logger| logger.active.load(Ordering::Acquire) && level <= logger.level)
}

#[doc(hidden)]
pub fn emit(level: Level, pp_log: &PpLog, args: fmt::Arguments<'_>) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    if !logger.active.load(Ordering::Acquire) || level > logger.level {
        return;
    }

    // Build one complete line before calling `NonBlocking::write_all`.
    // `NonBlocking` treats each write as an independent queued message, so
    // writing the prefix and message separately could interleave fragments
    // emitted concurrently by different media threads.
    let timestamp = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    let mut line = String::with_capacity(256);
    write_timestamp(&mut line, timestamp);
    let _ = write!(line, " {level}");
    if let Some(pipeline_id) = pp_log.pipeline_id() {
        let _ = write!(line, " [pipeline_id={pipeline_id}]");
    }
    let _ = write!(
        line,
        " [element={}] [name={}] ",
        pp_log.element(),
        pp_log.name()
    );
    let _ = line.write_fmt(args);
    line.push('\n');

    let mut writer = logger.writer.clone();
    let _ = writer.write_all(line.as_bytes());
}

fn write_timestamp(output: &mut String, timestamp: OffsetDateTime) {
    let offset_seconds = timestamp.offset().whole_seconds();
    let offset_sign = if offset_seconds < 0 { '-' } else { '+' };
    let offset_seconds = offset_seconds.unsigned_abs();
    let offset_hours = offset_seconds / 3_600;
    let offset_minutes = (offset_seconds % 3_600) / 60;

    let _ = write!(
        output,
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}{offset_sign}{offset_hours:02}:{offset_minutes:02}",
        timestamp.year(),
        u8::from(timestamp.month()),
        timestamp.day(),
        timestamp.hour(),
        timestamp.minute(),
        timestamp.second(),
        timestamp.millisecond(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::{Date, Month, Time, UtcOffset};

    #[test]
    fn timestamp_is_iso_8601_with_milliseconds_and_numeric_offset() {
        let timestamp = Date::from_calendar_date(2026, Month::August, 15)
            .unwrap()
            .with_time(Time::from_hms_milli(15, 52, 24, 68).unwrap())
            .assume_offset(UtcOffset::from_hms(9, 0, 0).unwrap());
        let mut output = String::new();

        write_timestamp(&mut output, timestamp);

        assert_eq!(output, "2026-08-15T15:52:24.068+09:00");
    }
}
