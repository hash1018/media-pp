//! Synthetic, finite media for CLI tests, made without an ffmpeg executable.

use std::path::Path;

use file_common::run_to_end;
use media_pp::{
    buffer::{MediaBuffer, set_time_base},
    elements::{AppSource, FileMuxer, Mp4Layout, SwEncoder, SwEncoderOptions, VideoCodec},
    ffmpeg::{self, format::Pixel},
    pipeline::Pipeline,
};

pub fn fixture(path: &Path) {
    let (source, handle) = AppSource::new("pictures", 32);
    for index in 0..15 {
        let mut frame = ffmpeg::frame::Video::new(Pixel::YUV420P, 320, 240);
        frame.data_mut(0).fill(60 + index as u8 * 5);
        frame.data_mut(1).fill(100);
        frame.data_mut(2).fill(160);
        frame.set_pts(Some(index));
        set_time_base(&mut frame, ffmpeg::Rational(1, 30));
        handle.push(MediaBuffer::video(frame)).unwrap();
    }
    handle.finish().unwrap();
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
    )
    .unwrap();
    let mut muxer = FileMuxer::create(path, Mp4Layout::FastStart).unwrap();
    let track = muxer.add_stream("video", &encoder).unwrap();
    let sink = muxer.open().unwrap().take(track).unwrap();
    let (pipeline, ()) = Pipeline::new("fixture", source, |source, ctx| {
        ctx.attach(source, 0, ctx.branch().pipe(encoder).to(sink)?)?;
        Ok(())
    })
    .unwrap();
    run_to_end(pipeline).unwrap();
}
