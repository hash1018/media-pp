//! `media_pp::log::init` takes the directory as a path, not as text, so a
//! directory whose name is not valid UTF-8 is the one the logs go to.
//!
//! A test binary of its own, since the logger initializes once per process.

use std::{ffi::OsString, fs, path::PathBuf};

use media_pp::log::{self, Level};

/// A directory name no `&str` can hold: a lone surrogate on Windows, a byte
/// that is not UTF-8 anywhere else.
fn not_utf8(prefix: &str) -> OsString {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStringExt;
        let mut wide: Vec<u16> = prefix.encode_utf16().collect();
        wide.push(0xD800);
        OsString::from_wide(&wide)
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let mut bytes = prefix.as_bytes().to_vec();
        bytes.push(0xFF);
        OsString::from_vec(bytes)
    }
}

/// Converting such a path to text first — the only way the old `&str`
/// parameter could be called with it — replaced the character with U+FFFD
/// and logged into a different directory from the one asked for.
#[test]
fn logs_go_to_a_directory_whose_name_is_not_utf8() {
    let parent = std::env::temp_dir().join(format!("media_pp_log_path_{}", std::process::id()));
    let directory: PathBuf = parent.join(not_utf8("logs-"));
    assert!(directory.to_str().is_none(), "the name must not be UTF-8");
    if let Err(error) = fs::create_dir_all(&directory) {
        // Some filesystems (APFS) refuse such a name outright.
        eprintln!("skipping: this filesystem refuses a non-UTF-8 name: {error}");
        let _ = fs::remove_dir_all(&parent);
        return;
    }

    let guard = log::init("path", &directory, Level::Info, 2).expect("the logger initializes");
    drop(guard);

    let written = fs::read_dir(&directory)
        .expect("the directory is readable")
        .filter_map(Result::ok)
        .any(|entry| entry.file_name().to_string_lossy().starts_with("path."));
    let others = fs::read_dir(&parent)
        .expect("the parent is readable")
        .filter_map(Result::ok)
        .count();
    let _ = fs::remove_dir_all(&parent);

    assert!(
        written,
        "the log file is in the directory that was asked for"
    );
    assert_eq!(others, 1, "no second, lossily named directory was made");
}
