//! `BusEvent::Finished`: when a pipeline has ended, as opposed to when the
//! first thing in it did.

use super::*;
use crate::bus::BusMessage;

/// Pushes `buffers` packets and then `Eos`, and does both again after every
/// seek — a file played to its end, and played to it again from wherever it
/// was sent back to.
struct EndingSource {
    pp_log: PpLog,
    pad: SrcPad,
    buffers: usize,
    /// Whether the stream from the last start or seek still has to be sent.
    owed: bool,
}

impl EndingSource {
    fn new(buffers: usize) -> Self {
        Self {
            pp_log: element_pp_log(ElementType::Other, "ending", None),
            pad: SrcPad::new("ending_src"),
            buffers,
            owed: true,
        }
    }
}

impl Element for EndingSource {
    fn name(&self) -> Arc<str> {
        "ending".into()
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

impl Source for EndingSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for EndingSource {
    fn is_live(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            if drain_control(control, self, bus)?.stopped {
                return Ok(());
            }
            if self.owed {
                self.owed = false;
                for _ in 0..self.buffers {
                    self.pad
                        .push(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))?;
                }
                self.pad.push(MediaBuffer::Eos)?;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn as_seekable(&mut self) -> Option<&mut dyn crate::element::SeekableSource> {
        Some(self)
    }
}

impl crate::element::SeekableSource for EndingSource {
    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.owed = true;
        Ok(target)
    }
}

/// A terminal that can be told to take its time over `Eos`, or to fail it.
struct EndingSink {
    pp_log: PpLog,
    name: Arc<str>,
    eos_delay: Duration,
    fail_eos: bool,
}

impl EndingSink {
    fn new(name: &str) -> Self {
        Self {
            pp_log: element_pp_log(ElementType::Other, name, None),
            name: name.into(),
            eos_delay: Duration::ZERO,
            fail_eos: false,
        }
    }
}

impl Element for EndingSink {
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

impl Sink for EndingSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if buf.is_eos() {
            thread::sleep(self.eos_delay);
            if self.fail_eos {
                return Err(crate::error::Error::Other("trailer failed".into()));
            }
        }
        Ok(())
    }
}

/// Reads the bus until `Finished` arrives or `timeout` passes, returning
/// everything read. Stopping is the caller's.
fn read_until_finished(pipeline: &Pipeline, timeout: Duration) -> Vec<BusMessage> {
    let deadline = Instant::now() + timeout;
    let mut messages = Vec::new();
    while Instant::now() < deadline {
        match pipeline.bus().try_recv_message() {
            Some(message) => {
                let finished = matches!(message.event, BusEvent::Finished);
                messages.push(message);
                if finished {
                    break;
                }
            }
            None => thread::sleep(Duration::from_millis(1)),
        }
    }
    messages
}

fn eos_from(messages: &[BusMessage], name: &str) -> Option<usize> {
    messages
        .iter()
        .position(|message| matches!(&message.event, BusEvent::Eos { name: n, .. } if &**n == name))
}

fn finished_count(messages: &[BusMessage]) -> usize {
    messages
        .iter()
        .filter(|message| matches!(message.event, BusEvent::Finished))
        .count()
}

/// The contract in one: the first `Eos` is one branch ending, and
/// `Finished` waits for the slow one — the muxer still writing its trailer
/// — before it says the pipeline is done. It comes once, after both, as
/// the pipeline's rather than any element's.
#[test]
fn finished_waits_for_every_terminal_of_a_fan_out() {
    let (pipeline, ()) = Pipeline::new("finished-fan-out", EndingSource::new(4), |source, ctx| {
        let fast = ctx.branch().to(EndingSink::new("fast"))?;
        let slow = ctx.branch().queue("slow-queue", 8).to(EndingSink {
            eos_delay: Duration::from_millis(150),
            ..EndingSink::new("slow")
        })?;
        let tee = ctx.tee("tee").branch(fast).branch(slow).build()?;
        ctx.attach(source, 0, tee)?;
        Ok(())
    })
    .expect("wiring");

    pipeline.run().unwrap();
    let messages = read_until_finished(&pipeline, Duration::from_secs(5));
    pipeline.stop();
    let messages: Vec<_> = messages
        .into_iter()
        .chain(pipeline.bus().iter_with_ids())
        .collect();

    let fast = eos_from(&messages, "fast").expect("the fast branch ended");
    let slow = eos_from(&messages, "slow").expect("the slow branch ended");
    let finished = messages
        .iter()
        .position(|message| matches!(message.event, BusEvent::Finished))
        .expect("the pipeline finished");
    assert!(fast < slow, "the fast branch ends first: {messages:?}");
    assert!(
        finished > slow,
        "Finished follows the last terminal, not the first: {messages:?}"
    );
    assert_eq!(finished_count(&messages), 1, "and comes once");
    assert_eq!(
        messages[finished].element_id, None,
        "posted by the pipeline, not an element"
    );
}

/// A branch whose trailer failed will never end; detaching it is how a
/// caller gives up on it, and the rest of the pipeline — already ended —
/// is then finished. Before the detach it is not: a failed terminal does
/// not count as one that ended.
#[test]
fn detaching_the_branch_that_failed_to_end_finishes_the_rest() {
    let (pipeline, (tee, broken)) =
        Pipeline::new("finished-detach", EndingSource::new(2), |source, ctx| {
            let (tee, tee_handle) = ctx.tee("tee").build_dynamic()?;
            ctx.attach(source, 0, tee)?;
            tee_handle.attach(ctx.branch().to(EndingSink::new("fine"))?)?;
            let broken =
                tee_handle.attach(ctx.branch().queue("broken-queue", 8).to(EndingSink {
                    fail_eos: true,
                    ..EndingSink::new("broken")
                })?)?;
            Ok((tee_handle, broken))
        })
        .expect("wiring");

    pipeline.run().unwrap();
    // The broken branch reports its failure; nothing reports Finished.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut before = Vec::new();
    while Instant::now() < deadline
        && !before
            .iter()
            .any(|message: &BusMessage| matches!(message.event, BusEvent::Error { .. }))
    {
        match pipeline.bus().try_recv_message() {
            Some(message) => before.push(message),
            None => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(eos_from(&before, "fine").is_some(), "{before:?}");
    assert!(eos_from(&before, "broken").is_none(), "{before:?}");
    thread::sleep(Duration::from_millis(50));
    while let Some(message) = pipeline.bus().try_recv_message() {
        before.push(message);
    }
    assert_eq!(
        finished_count(&before),
        0,
        "a terminal that failed its Eos has not ended: {before:?}"
    );

    tee.detach(broken).expect("detach the broken branch");
    let after = read_until_finished(&pipeline, Duration::from_secs(5));
    pipeline.stop();
    assert_eq!(
        finished_count(&after),
        1,
        "what is left has all ended: {after:?}"
    );
}

/// A seek sends a new stream through terminals that already ended the old
/// one, so the pipeline is not finished again until they end that one too
/// — and then it is, and says so a second time.
#[test]
fn a_seek_after_the_end_finishes_again_at_the_new_end() {
    let (pipeline, ()) = Pipeline::new("finished-seek", EndingSource::new(3), |source, ctx| {
        let branch = ctx.branch().queue("queue", 8).to(EndingSink::new("sink"))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wiring");

    pipeline.run().unwrap();
    let first = read_until_finished(&pipeline, Duration::from_secs(5));
    assert_eq!(finished_count(&first), 1, "{first:?}");

    pipeline
        .seek(Duration::ZERO, SeekMode::Keyframe)
        .expect("seek back to the start");
    let second = read_until_finished(&pipeline, Duration::from_secs(5));
    pipeline.stop();
    assert!(
        eos_from(&second, "sink").is_some(),
        "the new stream ended: {second:?}"
    );
    assert_eq!(
        finished_count(&second),
        1,
        "and the pipeline finished again: {second:?}"
    );
}

/// Finishing a pipeline while it plays hands every terminal its `Eos`, paced
/// or synchronized alike.
///
/// It did not: `finish` interrupts the clock so a long wait lets go of its
/// thread, a `Pacer` or `VideoSynchronizer` in one gave up and kept its
/// picture, and each took an interrupt as answered only by a control message
/// of its own. `finish` sends none downstream — its `Eos` travels as data —
/// so every wait after it counted as interrupted, the kept picture never
/// went, and the `Eos` stayed behind it. A muxer at the end of such a branch
/// never wrote its trailer. Found by the conformance sequences.
#[test]
fn finishing_while_playing_delivers_eos_through_a_timed_branch() {
    use crate::elements::AppSink;

    for synchronized in [false, true] {
        let Some(path) = try_test_video() else { return };
        let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("the fixture has a picture")
            .clone();
        let pictures = Arc::new(AtomicUsize::new(0));
        let ended = Arc::new(AtomicBool::new(false));
        let screen = {
            let pictures = Arc::clone(&pictures);
            let ended = Arc::clone(&ended);
            AppSink::new("screen", move |buffer| {
                match buffer {
                    MediaBuffer::Eos => ended.store(true, Ordering::SeqCst),
                    _ => {
                        pictures.fetch_add(1, Ordering::SeqCst);
                    }
                }
                Ok(())
            })
        };
        let (pipeline, ()) = Pipeline::new("finish-while-timed", source, |source, ctx| {
            let decoded = ctx
                .branch()
                .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
                .queue("video-frames", 4);
            let branch = if synchronized {
                decoded
                    .pipe(VideoSynchronizer::new("video-sync"))
                    .to(screen)?
            } else {
                decoded.pipe(Pacer::new("video-pacer")).to(screen)?
            };
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })
        .expect("wire the pipeline");
        pipeline.run().unwrap();
        // Playing, with a picture waiting for its turn.
        let waited = Instant::now();
        while pictures.load(Ordering::SeqCst) < 3 {
            assert!(
                waited.elapsed() < Duration::from_secs(10),
                "playback never started"
            );
            thread::sleep(Duration::from_millis(10));
        }

        pipeline.finish();
        assert!(
            ended.load(Ordering::SeqCst),
            "finished with {} pictures shown and no Eos (synchronized: {synchronized})",
            pictures.load(Ordering::SeqCst)
        );
    }
}

/// A seek past the end of a file, through a `Tee`, ends every branch.
///
/// The seek's preroll takes the file's last picture on each branch; the
/// `Eos` right behind it reached the `Tee` once both branches had theirs,
/// and a `Tee` skips a branch whose preroll is done — the `Eos` along with
/// the pictures. It was dropped, not kept, so neither branch ever ended, and
/// a `finish` afterwards could not end them either. Found by the conformance
/// sequences.
#[test]
fn a_seek_past_the_end_through_a_tee_ends_every_branch() {
    use crate::elements::AppSink;

    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
    let duration = source.duration().expect("the fixture says how long it is");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("the fixture has a picture")
        .clone();
    let ended: Vec<Arc<AtomicBool>> = (0..2).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let sinks: Vec<_> = ended
        .iter()
        .enumerate()
        .map(|(index, ended)| {
            let ended = Arc::clone(ended);
            AppSink::new(format!("screen-{index}"), move |buffer| {
                if buffer.is_eos() {
                    ended.store(true, Ordering::SeqCst);
                }
                Ok(())
            })
        })
        .collect();
    let (pipeline, ()) = Pipeline::new("seek-past-end-tee", source, |source, ctx| {
        let mut tee = ctx.tee("tee");
        for (index, sink) in sinks.into_iter().enumerate() {
            tee = tee.branch(ctx.branch().queue(format!("branch-{index}"), 2).to(sink)?);
        }
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
            .queue("video-frames", 4)
            .to_branch(tee.build()?)?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })
    .expect("wire the pipeline");
    pipeline.run().unwrap();
    pipeline
        .seek(duration + Duration::from_millis(500), SeekMode::Accurate)
        .expect("a seek past the end");

    let waited = Instant::now();
    while !ended.iter().all(|ended| ended.load(Ordering::SeqCst)) {
        assert!(
            waited.elapsed() < Duration::from_secs(5),
            "a branch never ended: {:?}",
            ended
                .iter()
                .map(|ended| ended.load(Ordering::SeqCst))
                .collect::<Vec<_>>()
        );
        thread::sleep(Duration::from_millis(10));
    }
    pipeline.stop();
}
