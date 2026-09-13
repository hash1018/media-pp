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
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};

use arc_swap::ArcSwapOption;
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
static NEXT_THREAD_NUMBER: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// Built once per thread, on that thread's first record. Formatting it
    /// per record would mean a `thread::current()` handle clone and a
    /// `String` build on every line, and the value never changes.
    static THREAD_TAG: String = thread_tag();
}

/// Numbers threads in the order they first log, in the `name#number` shape
/// the topology diagram already uses for elements. A thread name alone does
/// not identify a thread — a pipeline with two sources has two threads both
/// named `pipeline:source` — and `ThreadId`'s own value is not readable on
/// stable Rust, so the number is assigned here.
fn thread_tag() -> String {
    let number = NEXT_THREAD_NUMBER.fetch_add(1, Ordering::Relaxed);
    match thread::current().name() {
        Some(name) => format!("{name}#{number}"),
        None => format!("#{number}"),
    }
}

/// Severity threshold for the private `media-pp` file logger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Error conditions that prevent an operation from completing.
    Error,
    /// Recoverable problems or unexpected conditions.
    Warn,
    /// High-level lifecycle and topology events.
    Info,
    /// Detailed diagnostic events useful during development.
    Debug,
    /// Fine-grained per-operation tracing.
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
/// Why [`init`] could not install the file logger.
pub enum LogInitError {
    /// The process already installed media-pp's global private logger.
    #[error("the media-pp file logger has already been initialized")]
    AlreadyInitialized,

    /// The configured log directory could not be created.
    #[error("failed to create log directory `{path}`: {source}")]
    LogDirectory {
        /// Directory that could not be created.
        path: PathBuf,
        /// Operating-system error returned while creating the directory.
        source: std::io::Error,
    },

    /// The rolling file appender could not be initialized.
    #[error("failed to create log file appender: {0}")]
    FileAppender(#[from] InitError),
}

/// Owns the private logging worker installed by [`init`].
///
/// Keep this value alive for as long as logging should remain active.
///
/// Dropping it rejects log calls that begin afterwards. A record already being
/// emitted concurrently may complete or be discarded — the guard does not join
/// the worker thread, the same trade the lossy writer already makes on a full
/// queue.
///
/// The final flush is attempted, not promised. The drop asks [`WorkerGuard`] to
/// shut the worker down, which enqueues a shutdown message on the same bounded
/// channel the records use — waiting at most 100ms — and then waits at most one
/// second for the worker's acknowledgement, sent only after it has flushed
/// everything already queued. So under normal conditions queued records do reach
/// the file, but a file writer stalled long enough to keep that channel full
/// makes the drop give up and return with records still queued. On that path
/// `tracing-appender` also prints one line to the process's stdout, which this
/// crate cannot suppress. The worker still terminates and flushes on its own
/// afterwards (see this type's `Drop`), just with nothing waiting for it.
///
/// It is deliberately not stored in a static because Rust does not drop static
/// values at process exit, which would make even that attempt impossible.
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
        // Release this process-wide writer *before* running the worker
        // guard. That guard gets 100ms to enqueue its shutdown message on
        // the same bounded channel the records use; a stalled file writer
        // can make that time out. While the static still held a sender the
        // worker could then never observe a disconnect either, and would
        // outlive this guard for the rest of the process. Dropping ours
        // first leaves the worker guard's own sender as the last one, so
        // that path terminates the worker through disconnect instead —
        // after it drains and flushes what is already queued.
        if let Some(logger) = LOGGER.get() {
            logger.writer.store(None);
        }
        self.worker.take();
    }
}

struct PrivateLogger {
    level: Level,
    active: Arc<AtomicBool>,
    /// Cleared by [`LogGuard::drop`], which is the only thing that makes the
    /// worker's channel reach zero senders — this value lives in a `static`
    /// that Rust never drops. See that `Drop` impl for why the worker's
    /// termination depends on it.
    writer: ArcSwapOption<NonBlocking>,
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
/// Dropping it permanently stops this one-shot logger and makes a bounded
/// attempt to flush what is still queued; see [`LogGuard`] for what that does
/// and does not promise. Initialization is per process, not per guard: a second
/// call returns
/// [`LogInitError::AlreadyInitialized`] whether or not the first guard is still
/// alive, so an application cannot re-enable logging or move it to a different
/// directory afterwards. Integration tests that need this logger therefore need
/// one test binary each, since `cargo test` runs a binary's tests in one
/// process.
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
        writer: ArcSwapOption::from_pointee(writer),
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
    let Some(writer) = logger.writer.load_full() else {
        return;
    };

    // Build one complete line before calling `NonBlocking::write_all`.
    // `NonBlocking` treats each write as an independent queued message, so
    // writing the prefix and message separately could interleave fragments
    // emitted concurrently by different media threads.
    let timestamp = OffsetDateTime::now_local().unwrap_or_else(|_| OffsetDateTime::now_utc());
    let mut line = String::with_capacity(256);
    write_timestamp(&mut line, timestamp);
    let _ = write!(line, " {level}");
    // Ahead of the element identity, not between it and the message: a
    // reader grepping for one element's records should get its message on
    // the same match, and the thread is a property of the record's origin
    // like the timestamp and level, not part of who the element is.
    // `try_with` because a record emitted from a `Drop` running during
    // thread teardown would find this thread-local already destroyed; the
    // field still gets written so every record has the same shape.
    let tagged = THREAD_TAG.try_with(|tag| {
        let _ = write!(line, " [thread={tag}]");
    });
    if tagged.is_err() {
        let _ = line.write_str(" [thread=?]");
    }
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

    let mut writer = NonBlocking::clone(&writer);
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
