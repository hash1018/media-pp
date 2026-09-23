//! FFmpeg's own messages go into the private logger while it runs.
//!
//! A test binary of its own, since the logger initializes once per process.

use std::{
    ffi::{c_int, c_void},
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use ffmpeg_next as ffmpeg;
use media_pp::log::{self, Level};

const AV_LOG_WARNING: c_int = 24;
const AV_LOG_INFO: c_int = 32;
const AV_LOG_DEBUG: c_int = 48;

/// What FFmpeg would have printed to stderr is recorded instead, at the
/// matching level and named after the context it came from — the codec's
/// name for a codec context, `ffmpeg` for none — without FFmpeg's own
/// `[aac @ 0x…]` prefix. What FFmpeg's own threshold holds back — its
/// debugging, at `INFO` — it still holds back, though the logger keeps
/// everything down to `Trace`. And once the guard is dropped, nothing more is recorded.
#[test]
fn ffmpeg_messages_are_recorded_as_the_codec_that_said_them() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "media_pp_ffmpeg_log_{}_{unique}",
        std::process::id()
    ));
    let guard = log::init("ffmpeg", &dir, Level::Trace, 2).expect("the logger initializes");

    let codec = ffmpeg::encoder::find(ffmpeg::codec::Id::AAC).expect("AAC is built into FFmpeg");
    let context = ffmpeg::codec::context::Context::new_with_codec(codec);
    // SAFETY: a live codec context, whose first field is its AVClass, and
    // format strings matching their arguments.
    unsafe {
        ffmpeg::ffi::av_log(
            context.as_ptr() as *mut c_void,
            AV_LOG_WARNING,
            c"codec-marker %d\n".as_ptr(),
            7 as c_int,
        );
        ffmpeg::ffi::av_log(std::ptr::null_mut(), AV_LOG_INFO, c"bare-marker\n".as_ptr());
        ffmpeg::ffi::av_log(
            std::ptr::null_mut(),
            AV_LOG_DEBUG,
            c"debug-marker\n".as_ptr(),
        );
    }
    drop(guard);
    // SAFETY: as above; FFmpeg prints this one itself now.
    unsafe {
        ffmpeg::ffi::av_log(
            std::ptr::null_mut(),
            AV_LOG_WARNING,
            c"after-guard-marker\n".as_ptr(),
        );
    }

    let file = fs::read_dir(&dir)
        .expect("the log directory is readable")
        .filter_map(Result::ok)
        .find(|entry| entry.file_name().to_string_lossy().starts_with("ffmpeg."))
        .expect("a log file was written");
    let text = fs::read_to_string(file.path()).expect("the log file is readable");
    let _ = fs::remove_dir_all(&dir);

    let line = |marker: &str| text.lines().find(|line| line.contains(marker));
    let codec_line = line("codec-marker").expect("the codec's warning was recorded");
    assert!(codec_line.contains(" WARN "), "{codec_line}");
    assert!(
        codec_line.contains("[element=FFmpeg] [name=aac] codec-marker 7"),
        "{codec_line}"
    );
    let bare_line = line("bare-marker").expect("a message with no context was recorded");
    assert!(bare_line.contains(" INFO "), "{bare_line}");
    assert!(
        bare_line.contains("[element=FFmpeg] [name=ffmpeg] bare-marker"),
        "{bare_line}"
    );
    assert!(!text.contains(" @ 0x"), "FFmpeg's own prefix is left out");
    assert!(
        line("debug-marker").is_none(),
        "FFmpeg's threshold still applies"
    );
    assert!(
        line("after-guard-marker").is_none(),
        "nothing after the guard"
    );
}
