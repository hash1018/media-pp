use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream, UdpSocket},
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use media_pp::{
    buffer::{MediaBuffer, set_time_base},
    bus::BusEvent,
    elements::{AppSource, FileMuxer, Mp4Layout, SwEncoder, SwEncoderOptions, VideoCodec},
    ffmpeg::{self, format::Pixel},
    pipeline::Pipeline,
};
use str0m::{
    Candidate, Rtc,
    media::{Direction, MediaKind},
};

#[test]
fn missing_arguments_fail_with_usage() {
    let dir = tempfile::tempdir().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_webrtc_server"))
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage: webrtc_server"));
}

/// The example serves the page that plays a path, answers a player's offer
/// for it, and stops by itself after the seconds it was given.
#[test]
fn serves_its_page_answers_an_offer_and_stops() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input.mp4");
    encode(&input).unwrap();
    let free = |udp: bool| -> SocketAddr {
        if udp {
            UdpSocket::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
        } else {
            TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
        }
    };
    let (udp, http) = (free(true), free(false));
    let child = Command::new(env!("CARGO_BIN_EXE_webrtc_server"))
        .current_dir(dir.path())
        .arg(&input)
        .args([
            "--udp",
            &udp.to_string(),
            "--http",
            &http.to_string(),
            "--seconds",
            "6",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Retried until the example is listening.
    let deadline = Instant::now() + Duration::from_secs(30);
    let page = loop {
        match request(http, "GET /file HTTP/1.1\r\nHost: x\r\n\r\n") {
            Ok(page) => break page,
            Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(200)),
            Err(error) => panic!("the example never served its page: {error}"),
        }
    };
    assert!(page.starts_with("HTTP/1.1 200"), "{page}");
    assert!(
        page.contains("/file/whep"),
        "the page plays the path through WHEP"
    );

    let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let mut rtc = Rtc::builder().build(Instant::now());
    rtc.add_local_candidate(Candidate::host(socket.local_addr().unwrap(), "udp").unwrap());
    let mut api = rtc.sdp_api();
    api.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
    let (offer, _pending) = api.apply().unwrap();
    let offer = offer.to_sdp_string();
    let answer = request(
        http,
        &format!(
            "POST /file/whep HTTP/1.1\r\nHost: x\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{offer}",
            offer.len()
        ),
    )
    .unwrap();
    assert!(answer.starts_with("HTTP/1.1 201"), "{answer}");
    assert!(answer.contains("Location: /session/"), "{answer}");
    assert!(answer.contains("a=candidate:"), "{answer}");

    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains(&format!("open http://{http}/file")),
        "{stdout}"
    );
    assert!(stdout.contains("stopped"), "{stdout}");
}

fn request(addr: SocketAddr, text: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.write_all(text.as_bytes())?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
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
