use std::{fs, process::Command};

#[path = "../../file_common/test_support/mod.rs"]
mod support;

#[test]
fn missing_arguments_fail_with_usage() {
    let dir = tempfile::tempdir().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_transcode"))
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage: transcode"));
}

#[test]
fn rejects_input_aliases_and_existing_outputs_before_opening_hardware() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.mp4");
    support::fixture(&input);
    let original = fs::read(&input).unwrap();
    for overwrite in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_transcode"));
        command.current_dir(dir.path()).arg(&input).arg(&input);
        if overwrite {
            command.arg("--overwrite");
        }
        assert!(!command.output().unwrap().status.success());
        assert_eq!(fs::read(&input).unwrap(), original);
    }
    let output = dir.path().join("out.mp4");
    fs::write(&output, b"old output").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_transcode"))
        .current_dir(dir.path())
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(fs::read(&output).unwrap(), b"old output");
    fs::write(&input, b"not a media container").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_transcode"))
        .current_dir(dir.path())
        .arg(&input)
        .arg(&output)
        .arg("--overwrite")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!String::from_utf8_lossy(&result.stdout).contains("wrote "));
    assert_eq!(fs::read(output).unwrap(), b"old output");
}

#[cfg(windows)]
#[test]
fn decoder_bus_failure_is_not_success_or_an_overwritten_destination() {
    use media_pp::ffmpeg;
    if let Err(error) = media_pp::elements::D3d11Gpu::new() {
        eprintln!("skipping runtime transcode check: no D3D11 device ({error})");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("corrupt.mp4");
    support::fixture(&input);
    let (offset, size) = {
        let mut media = ffmpeg::format::input(&input).unwrap();
        let (_, packet) = media.packets().last().unwrap();
        (usize::try_from(packet.position()).unwrap(), packet.size())
    };
    let mut bytes = fs::read(&input).unwrap();
    bytes[offset..offset + size].fill(0xff);
    fs::write(&input, &bytes).unwrap();
    let output = dir.path().join("out.mp4");
    fs::write(&output, b"old output").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_transcode"))
        .current_dir(dir.path())
        .arg(&input)
        .arg(&output)
        .arg("--overwrite")
        .output()
        .unwrap();
    assert!(!result.status.success());
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("transcoding "),
        "pipeline setup did not complete: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!stdout.contains("wrote "));
    assert_eq!(fs::read(output).unwrap(), b"old output");
    assert_eq!(fs::read(input).unwrap(), bytes);
}
