//! Everything about this example that is not the GPU stack: argument
//! parsing, the synthetic frame generator, and the shutdown bookkeeping.
//! `cuda_record` drives the identical shell, so the only thing that differs
//! between the two is how the upload and encoder are constructed.

use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use media_pp::ffmpeg;
use media_pp::{
    buffer::MediaBuffer, bus::BusEvent, elements::AppSourceHandle, pipeline::Pipeline,
    pool::UnboundObjectPool,
};

pub struct Recording {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: ffmpeg::Rational,
    pub time_base: ffmpeg::Rational,
    pub frame_count: i64,
}

pub fn parse_args() -> media_pp::Result<Recording> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "nvenc_record.mp4".into());
    let seconds: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let frame_count: i64 = seconds
        .checked_mul(30)
        .and_then(|count| count.try_into().ok())
        .ok_or_else(|| media_pp::Error::Other("recording duration is too large".into()))?;
    Ok(Recording {
        path,
        width: 1280,
        height: 720,
        frame_rate: ffmpeg::Rational::new(30, 1),
        time_base: ffmpeg::Rational::new(1, 30),
        frame_count,
    })
}

fn fill_test_frame(frame: &mut ffmpeg::frame::Video, index: i64, width: u32) {
    let y_stride = frame.stride(0);
    let y_height = frame.plane_height(0) as usize;
    let y_plane = frame.data_mut(0);
    for row in 0..y_height {
        for col in 0..width as usize {
            y_plane[row * y_stride + col] = ((col as i64 + row as i64 + index) % 256) as u8;
        }
    }
    for plane in [1usize, 2usize] {
        frame.data_mut(plane).fill(128);
    }
    frame.set_pts(Some(index));
}

/// Pushes a moving gradient at the nominal frame rate, then `Eos`. Paced in
/// wall time so the recording's duration matches what the CLI asked for.
pub fn spawn_feeder(
    handle: AppSourceHandle,
    width: u32,
    height: u32,
    frame_count: i64,
) -> thread::JoinHandle<media_pp::Result<()>> {
    thread::spawn(move || -> media_pp::Result<()> {
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width, height),
            |_| {},
        );
        let frame_interval = Duration::from_secs_f64(1.0 / 30.0);
        let mut next_due = Instant::now();
        for index in 0..frame_count {
            thread::sleep(next_due.saturating_duration_since(Instant::now()));
            let mut frame = pool.get();
            fill_test_frame(&mut frame, index, width);
            handle.push(MediaBuffer::Video(Arc::new(frame)))?;

            next_due += frame_interval;
            let now = Instant::now();
            if next_due < now {
                next_due = now + frame_interval;
            }
        }
        handle.push(MediaBuffer::Eos)
    })
}

/// Drains the bus until the pipeline finishes, joins the feeder, and reports
/// whichever error came first.
pub fn finish(
    pipeline: &Pipeline,
    feeder: thread::JoinHandle<media_pp::Result<()>>,
    path: &str,
) -> media_pp::Result<()> {
    let mut pipeline_error = None;
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Error { name, error, .. } => {
                eprintln!("[{name}] error: {error}");
                if pipeline_error.is_none() {
                    pipeline_error = Some(error);
                }
                pipeline.stop();
            }
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            // `BusEvent` is `#[non_exhaustive]`; this example only acts on
            // the events above.
            _ => {}
        }
    }

    let feeder_result = feeder.join().expect("NVENC feeder thread panicked");
    if let Some(error) = pipeline_error {
        return Err(error);
    }
    feeder_result?;

    println!("wrote {path}");
    Ok(())
}
