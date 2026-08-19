//! Rejects an FFmpeg older than 8.0 at build time.
//!
//! `ffmpeg-sys-next` (`links = "ffmpeg"`) probes the installed libraries and
//! republishes what it found as build metadata. Cargo forwards that metadata
//! only to the build script of a crate that depends on it *directly*, which is
//! the sole reason `Cargo.toml` names `ffmpeg-sys-next` — no code in this
//! crate uses it, the FFI comes through `ffmpeg::ffi`.
//!
//! The check reads what `ffmpeg-sys-next` detected on *its* last run. That
//! build script declares no `rerun-if-env-changed`, so repointing
//! `PKG_CONFIG_PATH` at a different FFmpeg leaves both its generated bindings
//! and this gate stale until `cargo clean -p ffmpeg-sys-next`.

/// Set to `"true"` when the detected libavcodec is 62.8 or newer, which is
/// FFmpeg 8.0. `ffmpeg-sys-next` publishes the key either way, so an empty
/// value means "detected, and older".
const DETECTED_FFMPEG_8_0: &str = "DEP_FFMPEG_FFMPEG_8_0";

fn main() {
    println!("cargo:rerun-if-env-changed={DETECTED_FFMPEG_8_0}");

    // docs.rs documents rather than links, and its image supplies whatever
    // FFmpeg it supplies; failing there would break the published docs without
    // protecting anyone's runtime.
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    if std::env::var(DETECTED_FFMPEG_8_0).as_deref() != Ok("true") {
        println!(
            "cargo::error=media-pp requires FFmpeg 8.0 or newer (libavcodec 62.8+), but \
             ffmpeg-sys-next found an older installation. Point PKG_CONFIG_PATH at an FFmpeg 8 \
             build, then run `cargo clean -p ffmpeg-sys-next` so its bindings are regenerated."
        );
    }
}
