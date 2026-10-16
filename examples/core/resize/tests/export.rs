//! Real CLI regression tests. Fixtures are encoded through media-pp itself;
//! neither a checked-in video nor the ffmpeg/ffprobe executables are needed.

use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

use media_pp::ffmpeg;
#[path = "../../file_common/test_support/mod.rs"]
mod support;
use support::fixture;

fn resize(input: &Path, output: &Path, overwrite: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_resize"));
    command
        .current_dir(input.parent().unwrap())
        .arg(input)
        .arg(output)
        .args(["--width", "160"]);
    if overwrite {
        command.arg("--overwrite");
    }
    command.output().unwrap()
}

fn assert_failed(output: &Output) {
    assert!(
        !output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(!String::from_utf8_lossy(&output.stdout).contains("wrote "));
}

fn assert_no_stage(directory: &Path) {
    assert!(!fs::read_dir(directory).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".media-pp-")
    }));
}

#[test]
fn missing_arguments_report_usage_and_nonzero_status() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_resize"))
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert_failed(&output);
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage: resize"));
}

#[test]
fn exports_a_decodable_file_with_the_requested_size_and_all_frames() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input with spaces.mp4");
    let output = dir.path().join("output with spaces.mp4");
    fixture(&input);
    let original = fs::read(&input).unwrap();
    let result = resize(&input, &output, false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("wrote "));
    let mut media = ffmpeg::format::input(&output).unwrap();
    let stream = media.streams().best(ffmpeg::media::Type::Video).unwrap();
    let index = stream.index();
    let params = stream.parameters().clone();
    let mut decoder = ffmpeg::codec::context::Context::from_parameters(params)
        .unwrap()
        .decoder()
        .video()
        .unwrap();
    assert_eq!((decoder.width(), decoder.height()), (160, 120));
    let mut count = 0;
    let mut frame = ffmpeg::frame::Video::empty();
    for (stream, packet) in media.packets() {
        if stream.index() == index {
            decoder.send_packet(&packet).unwrap();
            while decoder.receive_frame(&mut frame).is_ok() {
                count += 1;
            }
        }
    }
    decoder.send_eof().unwrap();
    while decoder.receive_frame(&mut frame).is_ok() {
        count += 1;
    }
    assert_eq!(count, 15);
    assert_eq!(fs::read(input).unwrap(), original);
    assert_no_stage(dir.path());
}

#[test]
fn refuses_aliases_and_requires_explicit_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    let output = dir.path().join("output.mp4");
    fixture(&input);
    let original = fs::read(&input).unwrap();
    for overwrite in [false, true] {
        assert_failed(&resize(&input, &input, overwrite));
        assert_eq!(fs::read(&input).unwrap(), original);
    }
    fs::write(&output, b"existing output").unwrap();
    assert_failed(&resize(&input, &output, false));
    assert_eq!(fs::read(&output).unwrap(), b"existing output");
    assert!(resize(&input, &output, true).status.success());
    assert!(ffmpeg::format::input(&output).is_ok());
    assert_no_stage(dir.path());
}

#[test]
fn runtime_decoder_error_neither_publishes_nor_replaces_an_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("corrupt.mp4");
    fixture(&input);
    // The container and its first frames remain valid. Corrupt only the last
    // packet, so opening succeeds and the failure occurs in the running decoder.
    let (offset, size) = {
        let mut media = ffmpeg::format::input(&input).unwrap();
        let (_, packet) = media.packets().last().unwrap();
        (usize::try_from(packet.position()).unwrap(), packet.size())
    };
    let mut bytes = fs::read(&input).unwrap();
    bytes[offset..offset + size].fill(0xff);
    fs::write(&input, &bytes).unwrap();
    let output = dir.path().join("out.mp4");
    let result = resize(&input, &output, false);
    assert_failed(&result);
    assert!(String::from_utf8_lossy(&result.stdout).contains("resizing "));
    assert!(!output.exists());
    fs::write(&output, b"old destination").unwrap();
    let result = resize(&input, &output, true);
    assert_failed(&result);
    assert!(String::from_utf8_lossy(&result.stdout).contains("resizing "));
    assert_eq!(fs::read(output).unwrap(), b"old destination");
    assert_eq!(fs::read(input).unwrap(), bytes);
    assert_no_stage(dir.path());
}
