//! The stream plane: every element is handed the segment its buffers belong
//! to before them, whether it was there from the start or joined the stream
//! as it ran — see `crate::stream`. What a seek does to it is judged across
//! every shape by the conformance sequences.

use super::*;

use crate::elements::{Rack, TeeHandle};
use crate::stream::StreamEvent;

/// What an element on the stream was handed, in the order it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seen {
    Segment {
        id: u64,
        flushed: bool,
        start: Duration,
    },
    Buffer,
}

type Record = Arc<Mutex<Vec<Seen>>>;

/// Writes down what it is handed, and passes buffers on through its pad —
/// nowhere, as a terminal; to what follows, in a rack.
struct Watcher {
    pp_log: PpLog,
    name: Arc<str>,
    pad: SrcPad,
    seen: Record,
}

impl Watcher {
    fn new(name: &str) -> (Self, Record) {
        let seen = Record::default();
        (
            Self {
                pp_log: element_pp_log(ElementType::Other, name, None),
                name: name.into(),
                pad: SrcPad::new("src"),
                seen: Arc::clone(&seen),
            },
            seen,
        )
    }
}

impl Element for Watcher {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Other
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl SrcPads for Watcher {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl RawSink for Watcher {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.seen.lock().unwrap().push(Seen::Buffer);
        self.pad.push(buf)
    }

    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        let StreamEvent::Segment(segment) = event else {
            return Ok(());
        };
        self.seen.lock().unwrap().push(Seen::Segment {
            id: segment.id,
            flushed: segment.flushed,
            start: segment.start,
        });
        Ok(())
    }
}

/// The stream's segment as `seen` has it, having checked that it came
/// first and that nothing but buffers came after it.
fn segment_of(name: &str, seen: &Record) -> u64 {
    let seen = seen.lock().unwrap();
    let Some(Seen::Segment { id, flushed, .. }) = seen.first().copied() else {
        panic!("{name} was handed {:?} before any segment", seen.first());
    };
    assert!(
        !flushed,
        "{name}: nothing was flushed before a stream began"
    );
    for (at, entry) in seen.iter().enumerate().skip(1) {
        assert_eq!(
            *entry,
            Seen::Buffer,
            "{name}, entry {at}: one segment, and then the stream"
        );
    }
    id
}

/// Waits until `seen` holds a buffer.
fn wait_for_a_buffer(name: &str, seen: &Record) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !seen
        .lock()
        .unwrap()
        .iter()
        .any(|entry| matches!(entry, Seen::Buffer))
    {
        assert!(Instant::now() < deadline, "{name} was handed no buffer");
        thread::sleep(Duration::from_millis(5));
    }
}

fn small() -> TestVideoSource {
    TestVideoSource::new(
        "gen",
        TestVideoOptions {
            width: 64,
            height: 48,
            ..TestVideoOptions::default()
        },
    )
}

/// Every element is handed the segment ahead of the stream — behind a queue,
/// straight after a `Tee`, and inside a rack, where the elements are the
/// rack's rather than the graph's — and the element after a rack just once,
/// from the rack's own pad rather than out of the line inside it as well.
#[test]
fn every_element_begins_with_the_segment_its_buffers_are_on() {
    let (queued, queued_seen) = Watcher::new("behind-a-queue");
    let (direct, direct_seen) = Watcher::new("straight-after-the-tee");
    let (inside, inside_seen) = Watcher::new("in-a-rack");
    let (after, after_seen) = Watcher::new("after-the-rack");
    let (rack, rack_handle) = Rack::new("rack", InputContract::Unknown, OutputContract::Unknown);
    rack_handle
        .replace(vec![crate::element::AnyFilter::new(inside)])
        .expect("one pad");
    let (pipeline, ()) = Pipeline::new("segments", small(), |source, ctx| {
        let tee = ctx
            .tee("tee")
            .branch(ctx.branch().queue("q", 2).to(queued)?)
            .branch(ctx.branch().to(direct)?)
            .branch(ctx.branch().pipe(rack).to(after)?)
            .build()?;
        ctx.attach(source, 0, tee)?;
        Ok(())
    })
    .expect("wiring succeeds");
    pipeline.run().expect("run");
    let watched = [
        ("behind-a-queue", &queued_seen),
        ("straight-after-the-tee", &direct_seen),
        ("in-a-rack", &inside_seen),
        ("after-the-rack", &after_seen),
    ];
    for (name, seen) in watched {
        wait_for_a_buffer(name, seen);
    }
    pipeline.stop();
    let ids: Vec<u64> = watched
        .iter()
        .map(|(name, seen)| segment_of(name, seen))
        .collect();
    assert!(
        ids.windows(2).all(|pair| pair[0] == pair[1]),
        "one stream, one segment: {ids:?}"
    );
}

/// A branch attached to a `Tee` while the stream runs never saw the segment
/// go past, and is handed it before its first buffer.
#[test]
fn a_branch_attached_as_the_stream_runs_is_handed_its_segment_first() {
    let (first, first_seen) = Watcher::new("first");
    let (pipeline, tee) = Pipeline::new("attached", small(), |source, ctx| {
        let (tee, handle): (_, TeeHandle) = ctx
            .tee("tee")
            .branch(ctx.branch().to(first)?)
            .build_dynamic()?;
        ctx.attach(source, 0, tee)?;
        Ok(handle)
    })
    .expect("wiring succeeds");
    pipeline.run().expect("run");
    wait_for_a_buffer("first", &first_seen);

    let (later, later_seen) = Watcher::new("later");
    let branch = tee
        .branch()
        .expect("the tee is running")
        .queue("q", 2)
        .to(later)
        .expect("a branch");
    tee.attach(branch).expect("attached");
    wait_for_a_buffer("later", &later_seen);
    pipeline.stop();

    assert_eq!(
        segment_of("later", &later_seen),
        segment_of("first", &first_seen),
        "the stream's own segment, not one of its own"
    );
}

/// What a rack is filled with while the stream runs is handed the segment
/// before its first buffer, as a branch attached late is.
#[test]
fn a_rack_filled_anew_hands_what_it_holds_the_segment_first() {
    let (first, first_seen) = Watcher::new("first");
    let (after, after_seen) = Watcher::new("after");
    let (rack, rack_handle) = Rack::new("rack", InputContract::Unknown, OutputContract::Unknown);
    rack_handle
        .replace(vec![crate::element::AnyFilter::new(first)])
        .expect("one pad");
    let (pipeline, ()) = Pipeline::new("refilled", small(), |source, ctx| {
        let branch = ctx.branch().pipe(rack).to(after)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wiring succeeds");
    pipeline.run().expect("run");
    wait_for_a_buffer("first", &first_seen);

    let (second, second_seen) = Watcher::new("second");
    rack_handle
        .replace(vec![crate::element::AnyFilter::new(second)])
        .expect("one pad");
    wait_for_a_buffer("second", &second_seen);
    pipeline.stop();

    let id = segment_of("after", &after_seen);
    assert_eq!(segment_of("first", &first_seen), id);
    assert_eq!(segment_of("second", &second_seen), id);
}

/// A source begins its stream before it has looked at its control, so its
/// segment has to go past a queue that a pause started with has already
/// stopped. Waiting there for room, as a buffer would, it never looked:
/// `run` waited for good on the source to take the pause.
#[test]
fn a_segment_goes_past_a_queue_a_pause_has_stopped() {
    let (terminal, seen) = Watcher::new("terminal");
    let (pipeline, ()) = Pipeline::new("paused-start", small(), |source, ctx| {
        let branch = ctx.branch().queue("q", 0).to(terminal)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wiring succeeds");
    pipeline.pause();
    let (ran, run_returned) = mpsc::channel();
    let running = Arc::clone(&pipeline);
    thread::spawn(move || {
        let _ = ran.send(running.run());
    });
    run_returned
        .recv_timeout(Duration::from_secs(5))
        .expect("run returns, paused")
        .expect("run");
    pipeline.resume();
    wait_for_a_buffer("terminal", &seen);
    pipeline.stop();
    segment_of("terminal", &seen);
}

/// A looping file begins each lap as a segment of its own: on the same
/// timeline, since no seek began it, not flushed, and starting a lap on
/// from the one before — with the lap's buffers between each two.
#[test]
fn each_lap_of_a_looping_file_begins_a_segment() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let index = streams
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream")
        .index;
    source.looping_handle().set_looping(true);
    let (terminal, seen) = Watcher::new("terminal");
    let (pipeline, ()) = Pipeline::new("laps", source, |source, ctx| {
        let branch = ctx.branch().to(terminal)?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("wiring succeeds");
    pipeline.run().expect("run");
    let segments = || {
        seen.lock()
            .unwrap()
            .iter()
            .filter(|entry| matches!(entry, Seen::Segment { .. }))
            .count()
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while segments() < 3 {
        assert!(Instant::now() < deadline, "two laps did not begin");
        thread::sleep(Duration::from_millis(5));
    }
    pipeline.stop();

    let seen = seen.lock().unwrap();
    let starts: Vec<_> = seen
        .iter()
        .filter_map(|entry| match *entry {
            Seen::Segment { id, flushed, start } => Some((id, flushed, start)),
            Seen::Buffer => None,
        })
        .collect();
    let (id, _, first) = starts[0];
    assert_eq!(first, Duration::ZERO, "the stream starts at the start");
    for (at, pair) in starts.windows(2).enumerate() {
        let ((_, _, before), (lap_id, flushed, start)) = (pair[0], pair[1]);
        assert_eq!(lap_id, id, "lap {}: no seek began it", at + 1);
        assert!(!flushed, "lap {}: nothing is flushed", at + 1);
        assert!(start > before, "lap {}: a lap on, at {start:?}", at + 1);
    }
    let mut between = 0;
    for entry in seen.iter().skip(1) {
        match entry {
            Seen::Buffer => between += 1,
            Seen::Segment { .. } => {
                assert!(between > 0, "a lap with nothing in it");
                between = 0;
            }
        }
    }
}
