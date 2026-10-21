use std::{
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant},
};

use media_pp::{
    buffer::{MediaBuffer, set_time_base},
    elements::{AppSource, Pacer, RtspMuxer, RtspOptions, SwEncoder, SwEncoderOptions, VideoCodec},
    ffmpeg::{self, format::Pixel},
    pipeline::Pipeline,
};

#[test]
fn missing_arguments_fail_with_usage() {
    let dir = tempfile::tempdir().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_rtsp_listen"))
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage: rtsp_listen"));
}

/// The example receives a publisher pushing into it, counts what it sends,
/// and exits successfully once the publisher finishes.
#[test]
fn counts_a_publisher_and_exits_when_it_finishes() {
    let dir = tempfile::tempdir().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .unwrap()
        .port();
    let url = format!("rtsp://127.0.0.1:{port}/live");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rtsp_listen"))
        .current_dir(dir.path())
        .arg(&url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Retried until the example has bound the port.
    let deadline = Instant::now() + Duration::from_secs(30);
    let publisher = loop {
        match publish(&url) {
            Ok(publisher) => break publisher,
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => {
                let _ = child.kill();
                panic!("could not publish to the example: {error}");
            }
        }
    };

    let deadline = Instant::now() + Duration::from_secs(30);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("the example did not exit after the publisher finished");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    publisher.stop();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("publisher connected, 1 stream(s)"),
        "{stdout}"
    );
    assert!(stdout.contains("finished"), "{stdout}");
    let last = stdout
        .lines()
        .rev()
        .find(|line| line.starts_with("packets: [0] "))
        .unwrap_or_else(|| panic!("no count printed: {stdout}"));
    let counted: u64 = last["packets: [0] ".len()..].trim().parse().unwrap();
    assert!(counted > 0, "{stdout}");
}

/// Two seconds of pictures, encoded and published to `url` at the pace
/// they play at — running once this returns, and finishing by itself.
fn publish(url: &str) -> media_pp::Result<Arc<Pipeline>> {
    const FRAMES: i64 = 60;
    // Room for every picture and the end behind them, so that none of
    // this waits on a pipeline not yet running.
    let (source, handle) = AppSource::new("pictures", FRAMES as usize + 1);
    for index in 0..FRAMES {
        let mut frame = ffmpeg::frame::Video::new(Pixel::YUV420P, 320, 240);
        frame.data_mut(0).fill(60 + index as u8);
        frame.data_mut(1).fill(100);
        frame.data_mut(2).fill(160);
        frame.set_pts(Some(index));
        set_time_base(&mut frame, ffmpeg::Rational(1, 30));
        handle.push(MediaBuffer::video(frame))?;
    }
    handle.finish()?;
    let encoder = SwEncoder::new(
        "encode",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: 320,
            height: 240,
            pixel_format: Pixel::YUV420P,
            frame_rate: ffmpeg::Rational(30, 1),
            bit_rate: 500_000,
            gop_size: 15,
            max_b_frames: None,
        },
    )?;
    let options = RtspOptions {
        timeout: Duration::from_secs(2),
        ..Default::default()
    };
    let mut muxer = RtspMuxer::create(url, options)?;
    let track = muxer.add_stream("video", &encoder)?;
    let sink = muxer.open()?.take(track)?;
    let (pipeline, ()) = Pipeline::new("publisher", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(encoder)
            .pipe(Pacer::new("pace"))
            .to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;
    pipeline.run()?;
    Ok(pipeline)
}
