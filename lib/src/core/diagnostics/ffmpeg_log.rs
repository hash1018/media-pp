//! FFmpeg's own messages, taken into the private logger while it runs.
//!
//! FFmpeg writes what it has to say — an encoder's closing statistics, a
//! codec library's warnings — through `av_log`, which prints to stderr by
//! default: a console line with no time, no thread and nothing to say which
//! element's codec it came from, and nowhere near the `media-pp` log a
//! problem is being read in. [`crate::log::init`] is the caller asking for
//! that log, so it installs [`route`] as FFmpeg's log callback, and
//! [`crate::log::LogGuard`] puts FFmpeg's own back when it is dropped.
//!
//! The callback is process-wide FFmpeg state, which is why it is only ever
//! installed by that explicit, once-per-process call and never by merely
//! using an element. What reaches it is what FFmpeg would have printed:
//! FFmpeg's own threshold (`av_log_set_level`, `INFO` by default) still
//! applies first, and the logger's level after it.

use std::{
    borrow::Cow,
    ffi::{CStr, c_char, c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
};

use ffmpeg_next::ffi;

use super::log::{Level, emit, enabled, is_active};
use crate::pp_log::PpLog;

/// The longest message kept whole. FFmpeg's own default callback uses the
/// same bound; anything longer is cut, not dropped.
const LINE_BYTES: usize = 1024;

/// Makes FFmpeg's messages this logger's.
pub(super) fn install() {
    // SAFETY: registers a function with the signature `av_log` calls; the
    // callback is `'static` and handles being called from any thread.
    unsafe { ffi::av_log_set_callback(Some(route)) }
}

/// Gives FFmpeg's messages back to its own default, stderr.
pub(super) fn uninstall() {
    // SAFETY: FFmpeg's own default callback, as FFmpeg starts with.
    unsafe { ffi::av_log_set_callback(Some(ffi::av_log_default_callback)) }
}

/// The logger level an FFmpeg level is recorded at, or `None` for
/// `AV_LOG_QUIET`, which is never printed.
fn level_of(av_level: c_int) -> Option<Level> {
    Some(match av_level {
        // AV_LOG_QUIET
        i32::MIN..=-1 => return None,
        // AV_LOG_PANIC, AV_LOG_FATAL, AV_LOG_ERROR
        0..=16 => Level::Error,
        // AV_LOG_WARNING
        17..=24 => Level::Warn,
        // AV_LOG_INFO
        25..=32 => Level::Info,
        // AV_LOG_VERBOSE, AV_LOG_DEBUG
        33..=48 => Level::Debug,
        // AV_LOG_TRACE
        _ => Level::Trace,
    })
}

/// FFmpeg's log callback while the logger runs.
///
/// A level the logger does not keep is dropped before anything is
/// formatted. Once the logger has stopped — a message racing the guard's
/// drop — it goes where FFmpeg would have sent it.
unsafe extern "C" fn route(
    avcl: *mut c_void,
    av_level: c_int,
    fmt: *const c_char,
    vl: ffi::va_list,
) {
    // A panic must not unwind into C; a message lost to one is only a
    // message lost.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(level) = level_of(av_level) else {
            return;
        };
        // FFmpeg's own threshold is checked in its default callback, not by
        // `av_log`, so a replacement has to apply it itself — or every
        // codec's per-frame debugging would arrive.
        // SAFETY: reads FFmpeg's global log level.
        if av_level > unsafe { ffi::av_log_get_level() } {
            return;
        }
        if !enabled(level) {
            if !is_active() {
                // SAFETY: the arguments exactly as FFmpeg passed them.
                unsafe { ffi::av_log_default_callback(avcl, av_level, fmt, vl) };
            }
            return;
        }
        let mut line = [0 as c_char; LINE_BYTES];
        // No `[aac @ 0x…]` prefix: the context is recorded as the name.
        let mut print_prefix: c_int = 0;
        // SAFETY: `avcl`, `fmt` and `vl` are what `av_log` was called with,
        // and `line` is `LINE_BYTES` long, as the size says.
        let written = unsafe {
            ffi::av_log_format_line2(
                avcl,
                av_level,
                fmt,
                vl,
                line.as_mut_ptr(),
                LINE_BYTES as c_int,
                &mut print_prefix,
            )
        };
        if written < 0 {
            return;
        }
        // SAFETY: `av_log_format_line2` always NUL-terminates `line`.
        let text = unsafe { CStr::from_ptr(line.as_ptr()) }.to_string_lossy();
        let text = text.trim_end();
        if text.is_empty() {
            return;
        }
        // SAFETY: `avcl` is `av_log`'s context argument.
        let name = unsafe { context_name(avcl) };
        emit(
            level,
            &PpLog::new("FFmpeg", &name, None),
            format_args!("{text}"),
        );
    }));
}

/// What FFmpeg calls the context a message came from — the codec's name for
/// a codec context (`aac`, `libopenh264`), the format's for a format one —
/// or `ffmpeg` for a message with no context.
///
/// # Safety
///
/// `avcl` is null or points to a struct whose first field is a pointer to
/// its `AVClass`, which is what `av_log` requires of it.
unsafe fn context_name(avcl: *mut c_void) -> Cow<'static, str> {
    const NONE: &str = "ffmpeg";
    if avcl.is_null() {
        return Cow::Borrowed(NONE);
    }
    // SAFETY: as documented above.
    let class = unsafe { *avcl.cast::<*const ffi::AVClass>() };
    if class.is_null() {
        return Cow::Borrowed(NONE);
    }
    // SAFETY: a live `AVClass`, whose `item_name` takes the context it
    // describes and returns a static or context-owned C string.
    let name = unsafe {
        match (*class).item_name {
            Some(item_name) => item_name(avcl),
            None => (*class).class_name,
        }
    };
    if name.is_null() {
        return Cow::Borrowed(NONE);
    }
    // SAFETY: a NUL-terminated string that outlives this call.
    Cow::Owned(
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ffmpeg_level_has_its_counterpart() {
        assert_eq!(level_of(-8), None);
        assert_eq!(level_of(0), Some(Level::Error));
        assert_eq!(level_of(8), Some(Level::Error));
        assert_eq!(level_of(16), Some(Level::Error));
        assert_eq!(level_of(24), Some(Level::Warn));
        assert_eq!(level_of(32), Some(Level::Info));
        assert_eq!(level_of(40), Some(Level::Debug));
        assert_eq!(level_of(48), Some(Level::Debug));
        assert_eq!(level_of(56), Some(Level::Trace));
    }
}
