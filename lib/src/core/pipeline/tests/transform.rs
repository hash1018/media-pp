//! A `Filter` in a running pipeline: piped as any filter is, or racked
//! through `into_filter`, and told of a seek by the framework rather than by
//! a message of its own.

use super::*;

use crate::element::{Attached, Filter, IntoFilter, Output};
use crate::elements::Rack;

/// Passes everything on, counting what it passed and each reset.
struct Counting {
    pp_log: PpLog,
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
                passed: Arc::clone(&passed),
                resets: Arc::clone(&resets),
            },
            passed,
            resets,
        )
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

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
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

/// Each name an element of its own was told it had, and the pipeline it
/// was told it was in, in the order it was told.
type Told = Arc<Mutex<Vec<(String, Option<String>)>>>;

/// Passes everything on, writing down what each wiring told it.
struct Telling(Told);

impl Filter for Telling {
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        out.push(buf);
        Ok(())
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.0.lock().unwrap().push((
            cx.name().to_owned(),
            cx.pp_log().pipeline_id().map(str::to_owned),
        ));
    }
}

/// An element of its own keeps no identity, and is told it as it is wired:
/// its name — the one it was given, or its type's — and the pipeline it is
/// in, which its log records then name; and again when a rack wires it anew.
#[test]
fn a_filter_is_told_who_it_is_where_it_is_wired() {
    let piped = Told::default();
    let racked = Told::default();
    let (rack, rack_handle) = Rack::new("rack", InputContract::Unknown, OutputContract::Unknown);
    let source = TestVideoSource::new("source", TestVideoOptions::default());
    let sink = CountingSink {
        pp_log: element_pp_log(ElementType::Other, "sink", None),
        name: "sink".into(),
        count: Arc::new(AtomicUsize::new(0)),
    };
    let (pipeline, ()) = Pipeline::new("told", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(Telling(Arc::clone(&piped)).named("piped"))
            .pipe(rack)
            .to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wiring succeeds");
    rack_handle
        .replace(vec![Telling(Arc::clone(&racked)).into_filter()])
        .expect("racked");

    let told = |name: &str| vec![(name.to_owned(), Some("told".to_owned()))];
    assert_eq!(
        *piped.lock().unwrap(),
        told("piped"),
        "told as it was piped"
    );
    // A rack fills its line as the next buffer comes through it.
    pipeline.run().expect("run");
    wait_for("the rack was not filled", || {
        !racked.lock().unwrap().is_empty()
    });
    pipeline.stop();
    assert_eq!(
        *racked.lock().unwrap(),
        told("Telling"),
        "told as it was racked"
    );
}
