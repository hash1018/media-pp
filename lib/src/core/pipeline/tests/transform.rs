//! A `Filter` in a running pipeline: piped as any filter is, or racked
//! through `into_filter`, and told of a seek by the framework rather than by
//! a message of its own.

use super::*;

use crate::element::{Filter, IntoFilter, Output};
use crate::elements::Rack;

/// Passes everything on, counting what it passed and each reset.
struct Counting {
    pp_log: PpLog,
    name: Arc<str>,
    passed: Arc<AtomicUsize>,
    resets: Arc<AtomicUsize>,
}

impl Counting {
    fn new(name: &str) -> (Self, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let passed = Arc::new(AtomicUsize::new(0));
        let resets = Arc::new(AtomicUsize::new(0));
        (
            Self {
                pp_log: element_pp_log(ElementType::Other, name, None),
                name: name.into(),
                passed: Arc::clone(&passed),
                resets: Arc::clone(&resets),
            },
            passed,
            resets,
        )
    }
}

impl Element for Counting {
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

impl Filter for Counting {
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        self.passed.fetch_add(1, Ordering::SeqCst);
        out.push(buf);
        Ok(())
    }

    fn reset(&mut self) {
        self.resets.fetch_add(1, Ordering::SeqCst);
    }
}

fn wait_for(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        thread::sleep(Duration::from_millis(5));
    }
}

/// A transform goes where a filter does — straight into `pipe`, and into a
/// rack as `into_filter` makes one of it — and a seek resets it once, with
/// the stream going on through it from the new position.
#[test]
fn a_transform_follows_a_seek_piped_or_racked() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let index = streams
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream")
        .index;

    let (piped, piped_passed, piped_resets) = Counting::new("piped");
    let (racked, racked_passed, racked_resets) = Counting::new("racked");
    let (rack, rack_handle) = Rack::new("rack", InputContract::Unknown, OutputContract::Unknown);
    rack_handle
        .replace(vec![racked.into_filter()])
        .expect("one pad");
    let count = Arc::new(AtomicUsize::new(0));
    let sink = CountingSink {
        pp_log: element_pp_log(ElementType::Other, "sink", None),
        name: "sink".into(),
        count: Arc::clone(&count),
    };
    // Paced, so the file is still playing when the seek comes.
    let (pipeline, ()) = Pipeline::new("transform", source, |source, ctx| {
        let branch = ctx
            .branch()
            .queue("q", 4)
            .pipe(Pacer::new("pacer"))
            .pipe(piped)
            .pipe(rack)
            .to(sink)?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("wiring succeeds");

    pipeline.run().expect("run");
    wait_for("nothing reached the sink", || {
        count.load(Ordering::SeqCst) > 0
    });
    assert_eq!(piped_resets.load(Ordering::SeqCst), 0, "reset unasked");
    assert_eq!(racked_resets.load(Ordering::SeqCst), 0, "reset unasked");

    pipeline
        .seek(Duration::from_secs(1), SeekMode::Accurate)
        .expect("seek");
    assert_eq!(
        piped_resets.load(Ordering::SeqCst),
        1,
        "one seek, one reset"
    );
    assert_eq!(
        racked_resets.load(Ordering::SeqCst),
        1,
        "one seek, one reset"
    );
    let (piped_before, racked_before) = (
        piped_passed.load(Ordering::SeqCst),
        racked_passed.load(Ordering::SeqCst),
    );
    wait_for("nothing went through after the seek", || {
        piped_passed.load(Ordering::SeqCst) > piped_before
            && racked_passed.load(Ordering::SeqCst) > racked_before
    });
    pipeline.stop();

    let errors: Vec<_> = pipeline
        .bus()
        .iter()
        .filter(|event| matches!(event, BusEvent::Error { .. }))
        .collect();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
}
