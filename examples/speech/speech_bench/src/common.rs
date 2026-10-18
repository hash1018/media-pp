//! What both binaries need: the process's CPU time, the clip as Whisper
//! reads it, and the transcript written where `bench.py` scores it.

/// The CPU time this process has spent, user and system together.
#[cfg(unix)]
pub fn cpu_seconds() -> Option<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a whole rusage into the pointer it is given,
    // which points at one, and is read only when it says it succeeded.
    let usage = unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
            return None;
        }
        usage.assume_init()
    };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    Some(seconds(usage.ru_utime) + seconds(usage.ru_stime))
}

/// The CPU time this process has spent, user and kernel together.
#[cfg(windows)]
pub fn cpu_seconds() -> Option<f64> {
    use windows::Win32::{
        Foundation::FILETIME,
        System::Threading::{GetCurrentProcess, GetProcessTimes},
    };
    let (mut created, mut exited, mut kernel, mut user) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    // SAFETY: the pseudo-handle of the current process needs no closing, and
    // each pointer is to a FILETIME of this frame, read only once the call
    // has said it succeeded.
    unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        )
        .ok()?;
    }
    // Hundreds of nanoseconds.
    let seconds = |t: FILETIME| {
        ((u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime)) as f64 / 1e7
    };
    Some(seconds(kernel) + seconds(user))
}

#[cfg(not(any(unix, windows)))]
pub fn cpu_seconds() -> Option<f64> {
    None
}

/// The registry name whisper.cpp gives `backend`'s GPUs, or `None` for the
/// CPU — as each transcriber names its own.
pub fn registry_name(backend: &str) -> Option<Option<&'static str>> {
    match backend {
        "cpu" => Some(None),
        #[cfg(feature = "cuda")]
        "cuda" => Some(Some("CUDA")),
        #[cfg(feature = "vulkan")]
        "vulkan" => Some(Some("Vulkan")),
        #[cfg(all(target_os = "macos", feature = "metal"))]
        "metal" => Some(Some("Metal")),
        _ => None,
    }
}

/// Writes what was heard, one line each, where `--transcript` says, and
/// exits when it cannot: a run whose transcript is lost cannot be scored.
pub fn write_transcript(path: Option<&str>, lines: &[String]) {
    if let Some(path) = path
        && let Err(error) = std::fs::write(path, lines.join("\n") + "\n")
    {
        eprintln!("could not write {path}: {error}");
        std::process::exit(1);
    }
}

/// Exits with a usage line when `value` is missing or does not parse.
pub fn parse<T: std::str::FromStr>(flag: &str, value: Option<String>) -> T {
    match value.as_deref().map(str::parse) {
        Some(Ok(value)) => value,
        _ => {
            eprintln!("{flag} takes a number");
            std::process::exit(2);
        }
    }
}
