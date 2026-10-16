use std::{fs, process::Command};

#[path = "../../file_common/test_support/mod.rs"]
mod support;

#[test]
fn missing_arguments_fail_with_usage() {
    let dir = tempfile::tempdir().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_remux"))
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage: remux"));
}

#[test]
fn protects_input_and_destination_and_publishes_only_with_permission() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.mp4");
    let output = dir.path().join("out.mp4");
    support::fixture(&input);
    let original = fs::read(&input).unwrap();
    for overwrite in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_remux"));
        command.current_dir(dir.path()).arg(&input).arg(&input);
        if overwrite {
            command.arg("--overwrite");
        }
        assert!(!command.output().unwrap().status.success());
        assert_eq!(fs::read(&input).unwrap(), original);
    }
    fs::write(&output, b"old output").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_remux"))
        .current_dir(dir.path())
        .arg(&input)
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert_eq!(fs::read(&output).unwrap(), b"old output");
    let result = Command::new(env!("CARGO_BIN_EXE_remux"))
        .current_dir(dir.path())
        .arg(&input)
        .arg(&output)
        .arg("--overwrite")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(media_pp::ffmpeg::format::input(&output).is_ok());
    assert_eq!(fs::read(&input).unwrap(), original);
    fs::write(&input, b"not a media container").unwrap();
    let exported = fs::read(&output).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_remux"))
        .current_dir(dir.path())
        .arg(&input)
        .arg(&output)
        .arg("--overwrite")
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!String::from_utf8_lossy(&result.stdout).contains("wrote "));
    assert_eq!(fs::read(output).unwrap(), exported);
}
