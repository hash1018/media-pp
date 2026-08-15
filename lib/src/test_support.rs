//! Shared fixtures for this crate's own tests.

/// Path to a real video file for tests that need one, from
/// `MEDIA_PP_TEST_VIDEO`. Returns `None` — after printing why — when the
/// variable is unset or names something that is not a readable file, the same
/// way a hardware test's `try_device()` skips on a machine without the device.
///
/// No media is checked into this repository, so there is no default to fall
/// back to:
///
/// ```text
/// MEDIA_PP_TEST_VIDEO=/path/to/video.mp4 cargo test -p media-pp
/// ```
///
/// Any container `FileDemuxer` can open works, as long as it holds a video
/// stream and runs for at least a few seconds — the seek tests pace playback
/// and then reposition, so a clip shorter than that finishes before they get
/// to it. Nothing depends on a particular codec, resolution, or keyframe
/// spacing; a test that would need one must assert the contract instead (see
/// `pipeline::tests::seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe`).
///
/// Tests using this must still assert real behavior when it does return a
/// path — skipping is for the machine that has no fixture, not a way to make
/// a failing assertion optional.
pub(crate) fn try_test_video() -> Option<String> {
    let Ok(path) = std::env::var("MEDIA_PP_TEST_VIDEO") else {
        eprintln!(
            "skipping: set MEDIA_PP_TEST_VIDEO to a video file to run this test \
             (no media is checked into this repository)"
        );
        return None;
    };
    if !std::path::Path::new(&path).is_file() {
        eprintln!("skipping: MEDIA_PP_TEST_VIDEO=`{path}` is not a readable file");
        return None;
    }
    eprintln!("using test video: {path}");
    Some(path)
}
