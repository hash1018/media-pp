use std::{process::Command, sync::Arc, time::Duration};

use media_pp::{
    buffer::{MediaBuffer, set_time_base},
    driver::DriverRunner,
    elements::{
        AppSource, Broadcast, BroadcastOptions, Pacer, RtspPathMode, RtspServer, RtspServerOptions,
        SwEncoder, SwEncoderOptions, TrackFormat, VideoCodec,
    },
    ffmpeg::{self, format::Pixel},
    pipeline::Pipeline,
};

#[test]
fn missing_arguments_fail_with_usage() {
    let dir = tempfile::tempdir().unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_rtsp_load"))
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("usage: rtsp_load"));
}

/// Players over TCP and over UDP each play a server's stream to the end of
/// the run, lose nothing, and are summed up.
#[test]
fn players_play_and_are_summed_up() {
    let (server, handle) =
        RtspServer::bind("server", "127.0.0.1:0", RtspServerOptions::default()).unwrap();
    let broadcast = Broadcast::new("camera", BroadcastOptions::default());
    handle
        .add_path("cam", &broadcast, RtspPathMode::Play)
        .unwrap();
    let server = DriverRunner::new(server);
    server.run().unwrap();
    let camera = camera(&broadcast).unwrap();

    for udp in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_rtsp_load"));
        command
            .current_dir(dir.path())
            .args([handle.url("cam").as_str(), "3", "--seconds", "5"]);
        if udp {
            command.arg("--udp");
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("players: 3 played, 0 failed"),
            "udp={udp}: {stdout}"
        );
        assert!(stdout.contains("packets lost: 0 of"), "udp={udp}: {stdout}");
        assert!(stdout.contains("lag spread: median"), "udp={udp}: {stdout}");
    }
    camera.stop();
    server.stop();
}

/// A live camera — pictures encoded as they are due — written into
/// `broadcast` until stopped.
fn camera(broadcast: &Broadcast) -> media_pp::Result<Arc<Pipeline>> {
    const FRAMES: i64 = 30 * 20;
    let (source, handle) = AppSource::new("pictures", FRAMES as usize + 1);
    for index in 0..FRAMES {
        let mut frame = ffmpeg::frame::Video::new(Pixel::YUV420P, 320, 240);
        frame.data_mut(0).fill((index % 200) as u8 + 30);
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
            gop_size: 30,
            max_b_frames: None,
        },
    )?;
    let sink = broadcast.publish(&[TrackFormat::from(&encoder)])?.remove(0);
    let (pipeline, ()) = Pipeline::new("camera", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(Pacer::new("pace"))
            .pipe(encoder)
            .to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })?;
    pipeline.run()?;
    // A keyframe for the players to start at.
    std::thread::sleep(Duration::from_millis(500));
    Ok(pipeline)
}
