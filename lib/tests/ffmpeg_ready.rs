//! FFmpeg is readied by the library itself: there is no `init` for a caller
//! to forget.
//!
//! A test binary of its own, so it is the first thing in its process to reach
//! FFmpeg — which is exactly the situation a forgotten `init` used to leave a
//! program in.

use media_pp::elements::FileDemuxer;

/// FFmpeg describes its own errors only once they are registered; before
/// that, one displays as an empty string. A file that is not media, opened
/// before anything else in the process has touched FFmpeg, has to fail with
/// FFmpeg's own words for why.
#[test]
fn the_first_ffmpeg_error_in_a_process_says_what_went_wrong() {
    let path = std::env::temp_dir().join(format!("media_pp_not_media_{}.mp4", std::process::id()));
    std::fs::write(&path, b"this is not a media file, whatever its name says")
        .expect("write the file");

    let error = match FileDemuxer::open("demux", path.to_str().expect("a UTF-8 path")) {
        Ok(_) => panic!("text is not a container FFmpeg can open"),
        Err(error) => error.to_string(),
    };
    std::fs::remove_file(&path).ok();

    assert!(
        error.contains("Invalid data"),
        "FFmpeg's description of the failure is missing: {error:?}"
    );
}
