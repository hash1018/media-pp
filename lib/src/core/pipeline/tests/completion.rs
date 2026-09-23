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

    fn is_seekable(&self) -> bool {
        true
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

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
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
    let pipeline = Pipeline::new("finished-fan-out", EndingSource::new(4), |source, ctx| {
        let fast = ctx.branch().to(EndingSink::new("fast"))?;
        let slow = ctx.branch().queue("slow-queue", 8).to(EndingSink {
            eos_delay: Duration::from_millis(150),
            ..EndingSink::new("slow")
        })?;
        let tee = TeeBuilder::new("tee", ctx.clone())
            .branch(fast)
            .branch(slow)
            .build()?;
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
    let handle = Arc::new(Mutex::new(None));
    let failing = Arc::new(Mutex::new(None));
    let pipeline = Pipeline::new("finished-detach", EndingSource::new(2), {
        let handle = Arc::clone(&handle);
        let failing = Arc::clone(&failing);
        move |source, ctx| {
            let (tee, tee_handle) = TeeBuilder::new("tee", ctx.clone()).build_dynamic()?;
            ctx.attach(source, 0, tee)?;
            tee_handle.attach(ctx.branch().to(EndingSink::new("fine"))?)?;
            let broken =
                tee_handle.attach(ctx.branch().queue("broken-queue", 8).to(EndingSink {
                    fail_eos: true,
                    ..EndingSink::new("broken")
                })?)?;
            *failing.lock().unwrap() = Some(broken);
            *handle.lock().unwrap() = Some(tee_handle);
            Ok(())
        }
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

    let tee = handle.lock().unwrap().take().expect("tee handle");
    let broken = failing.lock().unwrap().take().expect("branch id");
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
    let pipeline = Pipeline::new("finished-seek", EndingSource::new(3), |source, ctx| {
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
