//! Start and stop a recording on a live pipeline, the way an application's
//! record button does.
//!
//! A preview branch runs the whole time; a recording branch joins a running
//! `Tee` and later leaves it. The property under test is the one a user sees:
//! the file that comes out is playable, which it only is if the branch was
//! ended with an ordered EOS rather than dropped — a muxer writes its trailer
//! on EOS, and an MP4 without one has no `moov` atom at all.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use media_pp::{
    buffer::MediaBuffer,
    bus::BusEvent,
    elements::{
        AppSource, FrameCounter, Mp4Muxer, SwEncoder, SwEncoderOptions, SwScaler, TeeBuilder,
        TestVideoOptions, TestVideoSource, VideoCodec,
    },
    pipeline::Pipeline,
    pool::UnboundObjectPool,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn temp_path(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("media-pp-tee-recording");
    std::fs::create_dir_all(&dir).expect("create the temp directory");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

#[test]
fn finishing_a_recording_branch_leaves_a_playable_file_and_a_running_preview() {
    media_pp::init().expect("initialize FFmpeg");
    let path = temp_path("recording.mp4");
    let path_str = path.to_string_lossy().to_string();
    let time_base = ffmpeg_next::Rational::new(1, 30);

    let source = TestVideoSource::new(
        "test-video",
        TestVideoOptions {
            width: WIDTH,
            height: HEIGHT,
            framerate: ffmpeg_next::Rational::new(30, 1),
        },
    );
    let (preview, preview_frames) = FrameCounter::new("preview");

    let mut tee_handle = None;
    let pipeline = Pipeline::new("tee-recording", source, |source, ctx| {
        let preview_branch = ctx.branch().to(Box::new(preview))?;
        let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone())
            .branch(preview_branch)
            .build_dynamic()?;
        ctx.attach(source, 0, tee_branch)?;
        tee_handle = Some(handle);
        Ok(())
    })
    .expect("wire the preview pipeline");
    let tee = tee_handle.expect("the wire closure provides the TeeHandle");

    pipeline.run().expect("start the preview pipeline");
    wait_until(|| preview_frames.load(Ordering::Relaxed) > 0);

    // --- the record button ---
    let encoder = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: WIDTH,
            height: HEIGHT,
            time_base,
            frame_rate: ffmpeg_next::Rational::new(30, 1),
            bit_rate: 1_000_000,
            gop_size: 30,
        },
    )
    .expect("open the encoder");
    let mut muxer = Mp4Muxer::create(&path_str).expect("create the MP4");
    muxer
        .add_stream("video", encoder.parameters(), time_base)
        .expect("add the video stream");
    let muxer_sink = muxer
        .open()
        .expect("open the MP4")
        .pop()
        .expect("exactly one stream was added");

    let recording = tee
        .branch()
        .expect("the tee is alive")
        .queue("captured", 8)
        .pipe(SwScaler::new(
            "to-yuv",
            ffmpeg_next::format::Pixel::YUV420P,
            WIDTH,
            HEIGHT,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        ))
        .queue("frames", 8)
        .pipe(encoder)
        .to(muxer_sink)
        .expect("build the recording branch");
    let recording_id = tee.attach(recording).expect("attach the recording branch");

    let frames_at_start = preview_frames.load(Ordering::Relaxed);
    wait_until(|| preview_frames.load(Ordering::Relaxed) >= frames_at_start + 20);

    // --- the stop button ---
    // Returns immediately; the drain runs on a thread the Tee owns.
    tee.finish_branch(recording_id)
        .expect("finish the recording branch");
    assert_eq!(
        tee.sink_count(),
        1,
        "only the preview branch is left attached"
    );

    // The preview is unaffected — that is the whole reason this is a Tee
    // branch and not the pipeline's own EOS.
    let frames_after_stop = preview_frames.load(Ordering::Relaxed);
    wait_until(|| preview_frames.load(Ordering::Relaxed) > frames_after_stop);

    // The file is complete once the terminal reports its EOS, not when
    // `finish_branch` returns.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut finalized = false;
    while !finalized && Instant::now() < deadline {
        match pipeline.bus().try_recv() {
            Some(BusEvent::Eos { name, .. }) if name.as_ref() == "video" => finalized = true,
            Some(BusEvent::Error { name, error, .. }) => {
                panic!("unexpected pipeline error from {name}: {error}")
            }
            Some(_) => {}
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    assert!(finalized, "the muxer must report EOS on the bus");

    pipeline.stop();

    let input = ffmpeg_next::format::input(&path_str)
        .expect("the recording must be playable — a missing trailer fails here");
    assert_eq!(input.streams().count(), 1, "one video track was recorded");
    assert!(
        input.duration() > 0,
        "a finalized MP4 carries a real duration"
    );
    let _ = std::fs::remove_file(&path);
}

/// Every frame the `Tee` handed the recording branch has to reach the file.
///
/// Opening the MP4 only proves a trailer was written. An EOS that raced past
/// the buffers still sitting in the branch's `Queue`, or one that reached the
/// encoder without draining whatever it was still holding, both truncate the
/// recording and both still leave a perfectly playable file — counting
/// packets is what tells a complete one from a short one. Feeding the source
/// by hand is what makes the expected number knowable at all; a
/// wall-clock-paced source has no exact frame count to assert.
#[test]
fn finishing_a_recording_branch_loses_no_frame_that_reached_it() {
    const FRAMES: usize = 24;

    media_pp::init().expect("initialize FFmpeg");
    let path = temp_path("frame-count.mp4");
    let path_str = path.to_string_lossy().to_string();
    let time_base = ffmpeg_next::Rational::new(1, 30);

    let (source, feed) = AppSource::new("app-source", 4);
    let (preview, preview_frames) = FrameCounter::new("preview");

    let mut tee_handle = None;
    let pipeline = Pipeline::new("tee-frame-count", source, |source, ctx| {
        let preview_branch = ctx.branch().to(Box::new(preview))?;
        let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone())
            .branch(preview_branch)
            .build_dynamic()?;
        ctx.attach(source, 0, tee_branch)?;
        tee_handle = Some(handle);
        Ok(())
    })
    .expect("wire the preview pipeline");
    let tee = tee_handle.expect("the wire closure provides the TeeHandle");
    pipeline.run().expect("start the pipeline");

    let encoder = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: WIDTH,
            height: HEIGHT,
            time_base,
            frame_rate: ffmpeg_next::Rational::new(30, 1),
            bit_rate: 1_000_000,
            gop_size: 30,
        },
    )
    .expect("open the encoder");
    let mut muxer = Mp4Muxer::create(&path_str).expect("create the MP4");
    muxer
        .add_stream("video", encoder.parameters(), time_base)
        .expect("add the video stream");
    let muxer_sink = muxer
        .open()
        .expect("open the MP4")
        .pop()
        .expect("exactly one stream was added");

    let recording = tee
        .branch()
        .expect("the tee is alive")
        .queue("frames", 32)
        .pipe(encoder)
        .to(muxer_sink)
        .expect("build the recording branch");
    let recording_id = tee.attach(recording).expect("attach the recording branch");

    let pool = UnboundObjectPool::new(0, ffmpeg_next::frame::Video::empty, |_| {});
    for index in 0..FRAMES {
        let mut frame =
            ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::YUV420P, WIDTH, HEIGHT);
        frame.set_pts(Some(index as i64));
        // Varying content, so the encoder cannot collapse the stream into
        // something whose packet count says nothing.
        frame.data_mut(0).fill((index * 8) as u8);
        frame.data_mut(1).fill(128);
        frame.data_mut(2).fill(128);
        let mut slot = pool.get();
        *slot = frame;
        feed.push(MediaBuffer::Video(Arc::new(slot)))
            .expect("push a frame");
    }

    // The preview counter is the proof that the Tee has actually handed all
    // of them to its branches: `push` only reaches the source's channel.
    wait_until(|| preview_frames.load(Ordering::Relaxed) >= FRAMES);

    tee.finish_branch(recording_id)
        .expect("finish the recording branch");
    wait_for_muxer_eos(&pipeline);
    pipeline.stop();

    let mut input = ffmpeg_next::format::input(&path_str).expect("the recording must be playable");
    let packets = input.packets().count();
    assert_eq!(
        packets, FRAMES,
        "every frame the branch received must be in the file, including the encoder's delayed tail"
    );
    let _ = std::fs::remove_file(&path);
}

/// Blocks until the muxer's own terminal reports EOS, which is when the
/// trailer has been written and the file is complete.
fn wait_for_muxer_eos(pipeline: &Pipeline) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        match pipeline.bus().try_recv() {
            Some(BusEvent::Eos { name, .. }) if name.as_ref() == "video" => return,
            Some(BusEvent::Error { name, error, .. }) => {
                panic!("unexpected pipeline error from {name}: {error}")
            }
            Some(_) => {}
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    panic!("the muxer never reported EOS on the bus");
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        condition(),
        "timed out waiting for the pipeline to progress"
    );
}
