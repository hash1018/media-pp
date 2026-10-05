use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    control::ControlMsg,
    element::{ElementType, RawSinkExt, element_pp_log},
    elements::{AppSource, AppSourceHandle, source::render_mode::RenderMode},
    ffmpeg,
    pipeline::Pipeline,
    stream::StreamEvent,
};

use super::*;

/// A small picture, `pts` in a 30 fps time base.
fn picture(pts: i64) -> MediaBuffer {
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 16, 16);
    frame.set_pts(Some(pts));
    crate::buffer::set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
    MediaBuffer::video(frame)
}

fn pts(buf: &MediaBuffer) -> i64 {
    match buf {
        MediaBuffer::Video(frame) => frame.pts().unwrap(),
        _ => panic!("a picture"),
    }
}

fn offline() -> StreamMuxOptions {
    StreamMuxOptions {
        mode: RenderMode::Offline { end: None },
        ..StreamMuxOptions::default()
    }
}

/// One input with a pipeline of its own: an `AppSource`, and a `Queue` for
/// an offline mux to hold back.
fn feed(handle: &StreamMuxHandle, name: &str) -> (Arc<Pipeline>, AppSourceHandle, StreamId) {
    let (sink, id) = handle.add_source(name).unwrap();
    let (source, pusher) = AppSource::new(name, 64);
    let (pipeline, ()) = Pipeline::new(format!("{name}-feed"), source, |source, ctx| {
        let branch = ctx.branch().queue(format!("{name}-queue"), 2).to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    (pipeline, pusher, id)
}

/// What a mux handed on, whether it finished, and its pipeline.
type Run = (Vec<MediaBuffer>, bool, Arc<Pipeline>);

/// Runs `mux` into a capturing sink until it finishes, `enough` says so, or
/// `limit` passes.
fn run(mux: StreamMux, limit: Duration, enough: impl Fn(&[MediaBuffer]) -> bool) -> Run {
    let received = Arc::new(Mutex::new(Vec::new()));
    let sink = crate::test_support::CapturingSink {
        received: received.clone(),
        pp_log: element_pp_log(ElementType::Other, "capture", None),
    };
    let (pipeline, ()) = Pipeline::new("mux", mux, |source, ctx| {
        let branch = ctx.branch().to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    let deadline = Instant::now() + limit;
    let mut finished = false;
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        if enough(&received.lock().unwrap()) {
            break;
        }
        if let Ok(BusEvent::Finished) = pipeline
            .bus()
            .recv_timeout(left.min(Duration::from_millis(20)))
        {
            finished = true;
            break;
        }
    }
    let received = received.lock().unwrap().clone();
    (received, finished, pipeline)
}

fn origin(buf: &MediaBuffer) -> StreamOrigin {
    buf.metadata()
        .unwrap()
        .get::<StreamOrigin>()
        .unwrap()
        .clone()
}

fn slot(buf: &MediaBuffer) -> BatchSlot {
    *buf.metadata().unwrap().get::<BatchSlot>().unwrap()
}

/// Offline, a batch is one picture of every input, in slot order, and each
/// stream's pictures keep their order and timestamps.
#[test]
fn an_offline_batch_is_a_picture_of_every_input() {
    let (mux, handle) = StreamMux::new("mux", offline()).unwrap();
    let feeds: Vec<_> = ["a", "b", "c"]
        .iter()
        .map(|name| feed(&handle, name))
        .collect();
    for (_, pusher, _) in &feeds {
        for pts in 0..5 {
            pusher.push(picture(pts)).unwrap();
        }
        pusher.finish().unwrap();
    }
    let (received, finished, _pipeline) = run(mux, Duration::from_secs(10), |_| false);
    assert!(finished, "an offline mux ends with its inputs");
    assert_eq!(received.len(), 15);
    for (batch, pictures) in received.chunks(3).enumerate() {
        let mut streams: Vec<_> = pictures.iter().map(|buf| origin(buf).id).collect();
        streams.sort();
        streams.dedup();
        assert_eq!(
            streams.len(),
            3,
            "one picture of each input in batch {batch}"
        );
        for (index, buf) in pictures.iter().enumerate() {
            assert_eq!(
                slot(buf),
                BatchSlot {
                    batch: batch as u64,
                    index,
                    size: 3
                }
            );
            assert_eq!(pts(buf), batch as i64, "its own timestamp");
        }
    }
}

/// An input that ends early is not waited for, and its end is recorded
/// after the last batch it had a picture in.
#[test]
fn an_input_that_ends_early_leaves_the_batches() {
    let (mux, handle) = StreamMux::new("mux", offline()).unwrap();
    let shared = handle.shared.upgrade().unwrap();
    let (_a, long, _) = feed(&handle, "long");
    let (_b, short, short_id) = feed(&handle, "short");
    for pts in 0..4 {
        long.push(picture(pts)).unwrap();
    }
    for pts in 0..2 {
        short.push(picture(pts)).unwrap();
    }
    long.finish().unwrap();
    short.finish().unwrap();
    let (received, finished, _pipeline) = run(mux, Duration::from_secs(10), |_| false);
    assert!(finished);
    let sizes: Vec<_> = received.iter().map(|buf| slot(buf).size).collect();
    assert_eq!(sizes, [2, 2, 2, 2, 1, 1]);
    let ended = shared.take_ended();
    assert!(
        ended.contains(&Ended {
            id: short_id,
            after: Some(1)
        }),
        "the short stream ends after batch 1: {ended:?}"
    );
}

/// With more inputs than `max_batch`, the inputs take turns.
#[test]
fn inputs_take_turns_past_max_batch() {
    let options = StreamMuxOptions {
        max_batch: 2,
        ..offline()
    };
    let (mux, handle) = StreamMux::new("mux", options).unwrap();
    let feeds: Vec<_> = ["a", "b", "c"]
        .iter()
        .map(|name| feed(&handle, name))
        .collect();
    for (_, pusher, _) in &feeds {
        for pts in 0..4 {
            pusher.push(picture(pts)).unwrap();
        }
        pusher.finish().unwrap();
    }
    let (received, finished, _pipeline) = run(mux, Duration::from_secs(10), |_| false);
    assert!(finished);
    assert_eq!(received.len(), 12);
    assert!(received.iter().all(|buf| slot(buf).size <= 2));
    // Over the first three batches, every input gave the same.
    let mut count = HashMap::new();
    for buf in &received[..6] {
        *count.entry(origin(buf).id).or_insert(0) += 1;
    }
    assert!(count.values().all(|&n| n == 2), "fair turns: {count:?}");
    // And each stream's pictures stay in order.
    for id in count.keys() {
        let mine: Vec<_> = received
            .iter()
            .filter(|buf| origin(buf).id == *id)
            .map(pts)
            .collect();
        assert_eq!(mine, [0, 1, 2, 3]);
    }
}

/// Live, an input with nothing holds the others back by the timeout and no
/// more.
#[test]
fn a_live_batch_goes_without_a_stalled_input() {
    let options = StreamMuxOptions {
        batch_timeout: Duration::from_millis(50),
        ..StreamMuxOptions::default()
    };
    let (mux, handle) = StreamMux::new("mux", options).unwrap();
    let (_a, a, a_id) = feed(&handle, "a");
    let (_b, _stalled, _) = feed(&handle, "stalled");
    a.push(picture(0)).unwrap();
    let started = Instant::now();
    let (received, _, _pipeline) = run(mux, Duration::from_secs(5), |got| !got.is_empty());
    assert_eq!(received.len(), 1);
    assert_eq!(origin(&received[0]).id, a_id);
    assert_eq!(slot(&received[0]).size, 1);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "not held past the timeout"
    );
}

/// Live, a full queue lets go of its oldest picture.
#[test]
fn a_full_live_queue_drops_its_oldest() {
    let options = StreamMuxOptions {
        queue: 2,
        ..StreamMuxOptions::default()
    };
    let (_mux, handle) = StreamMux::new("mux", options).unwrap();
    let shared = handle.shared.upgrade().unwrap();
    let (mut sink, id) = handle.add_source("camera").unwrap();
    for pts in 0..5 {
        sink.consume(picture(pts)).unwrap();
    }
    let kept = shared
        .with_input(id.0, |input| {
            input.pictures.iter().map(pts).collect::<Vec<_>>()
        })
        .unwrap();
    assert_eq!(kept, [3, 4]);
}

/// A flush — its pipeline was sought — empties the queue and starts a new
/// generation; its end marks it ended.
#[test]
fn a_flush_starts_a_new_generation() {
    let (_mux, handle) = StreamMux::new("mux", StreamMuxOptions::default()).unwrap();
    let shared = handle.shared.upgrade().unwrap();
    let (mut sink, id) = handle.add_source("file").unwrap();
    sink.consume(picture(0)).unwrap();
    sink.control(&ControlMsg::Flush).unwrap();
    let state = shared
        .with_input(id.0, |input| (input.pictures.len(), input.generation))
        .unwrap();
    assert_eq!(state, (0, 1));
    sink.stream_event(&StreamEvent::Eos).unwrap();
    assert_eq!(shared.with_input(id.0, |input| input.ended), Some(true));
}

/// A new input under a name in use is a new stream, and the old sink can no
/// longer touch it.
#[test]
fn a_replaced_input_is_a_new_stream_and_its_old_sink_is_inert() {
    let (_mux, handle) = StreamMux::new("mux", StreamMuxOptions::default()).unwrap();
    let shared = handle.shared.upgrade().unwrap();
    let (mut old, old_id) = handle.add_source("cam").unwrap();
    let (_new, new_id) = handle.add_source("cam").unwrap();
    assert_ne!(old_id, new_id);
    assert_eq!(handle.source_count(), 1);
    assert_eq!(
        shared.take_ended(),
        [Ended {
            id: old_id,
            after: None
        }]
    );
    old.consume(picture(0)).unwrap();
    old.control(&ControlMsg::Stop).unwrap();
    assert_eq!(
        handle.source_count(),
        1,
        "a stop from the replaced sink leaves the new input"
    );
    assert_eq!(
        shared.with_input(new_id.0, |input| input.pictures.len()),
        Some(0),
        "the replaced sink's picture went nowhere"
    );
}

#[test]
fn options_no_mux_can_run_with_are_refused() {
    for options in [
        StreamMuxOptions {
            max_batch: 0,
            ..StreamMuxOptions::default()
        },
        StreamMuxOptions {
            queue: 0,
            ..StreamMuxOptions::default()
        },
    ] {
        assert!(matches!(
            StreamMux::new("mux", options),
            Err(StreamMuxError::InvalidOptions(_))
        ));
    }
}

#[test]
fn an_input_refuses_what_is_not_a_picture() {
    let (_mux, handle) = StreamMux::new("mux", StreamMuxOptions::default()).unwrap();
    let (mut sink, _) = handle.add_source("x").unwrap();
    let packet = MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty()).into());
    assert!(sink.consume(packet).is_err());
}

#[test]
fn a_handle_outliving_its_mux_says_so() {
    let (mux, handle) = StreamMux::new("mux", StreamMuxOptions::default()).unwrap();
    drop(mux);
    assert!(matches!(
        handle.add_source("x"),
        Err(StreamMuxError::Stopped)
    ));
    assert_eq!(handle.source_count(), 0);
}
