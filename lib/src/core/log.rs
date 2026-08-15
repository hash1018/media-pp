//! Opt-in file logging backend for [`crate::pp_log`]. Elements emit their own
//! lifecycle/error records, and [`crate::bus::Bus::post`] additionally logs
//! each event it publishes. Inert until [`init`]
//! is called: this is a library, and writing to disk by default —
//! including from the crate's own tests, or anything embedding this crate
//! that never asked for log files — would be a surprising side effect.
//!
//! `crate::pp_log`'s `pp_info!`/`pp_warn!`/`pp_error!` macros go through the plain
//! `log` facade, so this bridges those records into a `tracing`
//! subscriber (via `tracing-log`) writing through a daily-rotating
//! [`tracing_appender`] file — same backend shape as this crate's own
//! sibling projects, not a bespoke one.

use thiserror::Error as ThisError;
use tracing_appender::{
    non_blocking::WorkerGuard,
    rolling::{InitError, RollingFileAppender, Rotation},
};
use tracing_subscriber::fmt::time::LocalTime;

#[derive(Debug, ThisError)]
pub enum LogInitError {
    #[error("failed to create log file appender: {0}")]
    FileAppender(#[from] InitError),
}

/// Starts file logging: every `pp_info!`/`pp_warn!`/`pp_error!` this crate makes
/// (via `crate::pp_log`, see [`crate::bus::Bus::post`]) from this point on is
/// appended to a rotating file named `{log_prefix}.{date}.log` in
/// `log_path` — one new file per day, oldest ones beyond `max_log_files`
/// deleted automatically. Writes happen on a dedicated background thread,
/// so nothing calling into `Bus::post` ever blocks on file I/O.
///
/// The returned [`WorkerGuard`] must be kept alive for as long as logging
/// should keep working — dropping it stops the background writer (and
/// flushes whatever's queued), so hold it for the program's lifetime
/// (e.g. a local in `main`), not something to let fall out of scope early.
///
/// # Panics
///
/// Panics if a global tracing subscriber has already been installed.
/// Call this at most once, before another part of the application installs
/// its own subscriber.
pub fn init(
    log_prefix: &str,
    log_path: &str,
    level: tracing::Level,
    max_log_files: usize,
) -> Result<WorkerGuard, LogInitError> {
    let file_appender = RollingFileAppender::builder()
        .filename_prefix(log_prefix)
        .filename_suffix("log")
        .rotation(Rotation::DAILY)
        .max_log_files(max_log_files)
        .build(log_path)?;

    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // `tracing-subscriber`'s `tracing-log` feature (on by default) already
    // bridges the plain `log` facade into whatever subscriber `.init()`
    // installs below — which is what `crate::pp_log`'s `pp_info!`/`pp_warn!`/
    // `pp_error!` macros go through — so no separate `tracing_log::LogTracer`
    // setup is needed (an explicit second one collides: `log::set_logger`
    // only ever succeeds once).
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_ansi(false)
        .with_max_level(level)
        .with_thread_ids(true)
        .with_timer(LocalTime::rfc_3339())
        .init();

    Ok(guard)
}
