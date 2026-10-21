use std::{
    path::Path,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use media_pp::{
    buffer::{MediaBuffer, set_time_base},
    bus::BusEvent,
    element::Sink,
    elements::{
        AppSource, FileMuxer, Mp4Layout, RtspOptions, RtspSource, SwEncoder, SwEncoderOptions,
        VideoCodec,
    },
    ffmpeg::{self, format::Pixel},
    pipeline::Pipeline,
};

#[test]
fn missing_arguments_fail_with_usage() {
    let dir = tempfile::tempdir().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_rtsp_server"))
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage: rtsp_server"));
}

/// The example plays a file to a player from a keyframe, and stops by
/// itself after the seconds it was given.
#[test]
fn plays_a_file_to_a_player_and_stops() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    encode(&input).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .unwrap()
        .port();
    let child = Command::new(env!("CARGO_BIN_EXE_rtsp_server"))
        .current_dir(dir.path())
        .arg(&input)
        .arg(format!("127.0.0.1:{port}"))
        .args(["--seconds", "8"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Retried until the example is listening.
    let url = format!("rtsp://127.0.0.1:{port}/file");
    let deadline = Instant::now() + Duration::from_secs(30);
    let (source, streams) = loop {
        let options = RtspOptions {
            timeout: Duration::from_secs(2),
            ..Default::default()
        };
        match RtspSource::open("player", &url, options) {
            Ok(opened) => break opened,
            Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(200)),
            Err(error) => panic!("could not play the example's file: {error}"),
        }
    };
    assert_eq!(streams.len(), 1);
    let keys = Arc::new(Mutex::new(Vec::new()));
    let (player, ()) = Pipeline::new("player", source, |source, ctx| {
        let branch = ctx.branch().to(Keys(Arc::clone(&keys)))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .unwrap();
    player.run().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while keys.lock().unwrap().len() < 30 {
        assert!(Instant::now() < deadline, "too little arrived");
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        keys.lock().unwrap()[0],
        "the first picture played is a keyframe"
    );

    // The example stops by itself, and its player sees it go.
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains(&url), "{stdout}");
    assert!(stdout.contains("stopped"), "{stdout}");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match player.bus().recv_timeout(Duration::from_millis(100)) {
            Ok(BusEvent::Finished | BusEvent::Error { .. }) => break,
            _ => assert!(
                Instant::now() < deadline,
                "the player did not see the server go"
            ),
        }
    }
    player.stop();
}

/// Whether each picture played was a keyframe.
struct Keys(Arc<Mutex<Vec<bool>>>);

impl Sink for Keys {
    fn render(&mut self, buf: MediaBuffer) -> media_pp::Result<()> {
        if let MediaBuffer::Packet(packet) = buf {
            self.0.lock().unwrap().push(packet.is_key());
        }
        Ok(())
    }
}

/// Two seconds of pictures, encoded into an MP4 at `path`.
fn encode(path: &Path) -> media_pp::Result<()> {
    const FRAMES: i64 = 60;
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
    let mut muxer = FileMuxer::create(path, Mp4Layout::Standard)?;
    let track = muxer.add_stream("video", &encoder)?;
    let sink = muxer.open()?.take(track)?;
    let (pipeline, ()) = Pipeline::new("encode", source, |source, ctx| {
        let branch = ctx.branch().pipe(encoder).to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;
    pipeline.run()?;
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Finished => break,
            BusEvent::Error { error, .. } => return Err(error),
            _ => {}
        }
    }
    pipeline.stop();
    Ok(())
}
