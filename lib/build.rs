use std::{env, path::PathBuf};

/// When the `rtsp-server` feature is enabled, copies the vendored
/// `mediamtx` binary (`third_party/mediamtx/`) next to whatever binary
/// ends up depending on this crate. Windows resolves a bare command name
/// (`Command::new("mediamtx")`, as `RtspServer` uses) by searching the
/// calling executable's own directory before `PATH` — so dropping it
/// there is enough for `RtspServer` to find it with no `PATH` setup.
fn main() {
    if env::var_os("CARGO_FEATURE_RTSP_SERVER").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendored = manifest_dir.join("../third_party/mediamtx/mediamtx.exe");
    println!("cargo:rerun-if-changed={}", vendored.display());

    // OUT_DIR is target/<profile>/build/media-pp-<hash>/out; the actual
    // binary output directory (where the final example .exe lands) is
    // three levels up, at target/<profile>. Cargo has no stable env var
    // for that directory, so this is the standard workaround (also used
    // by crates that need to place a DLL next to a test binary).
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR should be target/<profile>/build/<pkg>/out")
        .to_path_buf();

    std::fs::copy(&vendored, target_dir.join("mediamtx.exe"))
        .expect("failed to copy vendored mediamtx.exe into the target directory");
}
