use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ffmpeg_next as ffmpeg;

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::{MediaBuffer, release_picture},
    contract::{InputContract, MediaKind, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// Stops a branch taking frames, and takes the paused span out of the
/// timeline of what it does take.
///
/// # What it is for
///
/// Pausing a recording. The file has to stay open and then continue, with no
/// trace of the pause in it — a still picture lasting however long it was
/// paused is not what anybody means by the word. So dropping the frames is
/// only half of it: the frames that follow have to move back by as much as
/// was skipped, or the gap simply moves downstream.
///
/// # Not `ControlMsg::Pause`
///
/// That control stops queues pulling, which backpressures whatever feeds
/// them. On a [`Tee`](crate::elements::Tee) branch it would reach back
/// through the fan-out and stall the source for every other branch — pausing
/// a recording would freeze the preview beside it. This drops frames instead,
/// which nothing upstream can tell.
///
/// # How long the pause was
///
/// Measured in the source's own timestamps: the first frame dropped marks
/// where the pause began, and the first frame forwarded after it marks where
/// it ended. That is what makes this composable — it needs to know nothing
/// about the rate, and it is correct even for a source whose rate varies.
///
/// It does assume frames keep arriving while paused, which is what it counts.
/// A source that stops entirely cannot be measured this way — but neither
/// does its own timeline advance while it is stopped, so the span this would
/// have subtracted is one that was never there. The two agree.
///
/// # Where it goes
///
/// In front of the encoder, on frames. Behind one, the encoder would spend
/// the whole pause compressing frames that are about to be thrown away.
///
/// Directly in front of it, with nothing that buffers in between. What this
/// forwards is a wrapper holding a *reference* to the picture rather than the
/// pooled reference it was handed, and a frame reference keeps the buffer
/// allocated while leaving the producer's pool free to hand that slot out
/// again — the distinction [`ChangeGate`](crate::elements::ChangeGate) and
/// both video compositors keep retire lists for. This needs none, because the
/// wrapper never outlives the call that made it: the push downstream is
/// synchronous, the encoder copies what it is given before returning, and the
/// pooled reference this was handed is alive for the whole of that. A `Queue`
/// between the two would end that — the wrapper would outlive the pooled
/// reference, and the compositor would be free to draw into the slot it still
/// names.
///
/// # Cost
///
/// A reference to the frames it forwards and nothing at all for the ones it
/// drops. No pixels are read, copied or converted.
pub struct PauseGate {
    pp_log: PpLog,
    name: Arc<str>,
    /// Where the current pause began, in the source's timestamps. `None`
    /// while running.
    paused_from: Option<i64>,
    /// How much of the source's timeline has been paused away in total, and
    /// so how far back every frame after it is moved.
    skipped: i64,
    /// Empty frames to point at the pictures this forwards, so each carries
    /// its own timestamp without touching the one the source stamped — a
    /// sibling branch off the same `Tee` must keep seeing that.
    ///
    /// `release_picture` when one comes back, so an idle wrapper does not pin
    /// a picture the producer's pool wants returned.
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
    control: Arc<PauseControl>,
    pad: SrcPad,
}

/// What [`PauseGateHandle`] and the element share.
#[derive(Default)]
struct PauseControl {
    paused: AtomicBool,
}

/// Pauses and resumes a [`PauseGate`] from another thread.
///
/// Cloneable and cheap: the thread that owns a button and the one that owns
/// the element are never the same.
#[derive(Clone)]
pub struct PauseGateHandle {
    control: Arc<PauseControl>,
}

impl PauseGateHandle {
    /// Stops or resumes taking frames.
    pub fn set_paused(&self, paused: bool) {
        self.control.paused.store(paused, Ordering::Relaxed);
    }

    /// Whether it is currently paused.
    pub fn is_paused(&self) -> bool {
        self.control.paused.load(Ordering::Relaxed)
    }
}

impl PauseGate {
    /// Starts running. Nothing is paused until a handle says so.
    pub fn new(name: impl Into<String>) -> (Self, PauseGateHandle) {
        let name: Arc<str> = name.into().into();
        let pad = SrcPad::with_contract(format!("{name}_src"), OutputContract::Passthrough);
        let pp_log = element_pp_log(ElementType::PauseGate, &name, None);
        let control = Arc::new(PauseControl::default());
        let handle = PauseGateHandle {
            control: Arc::clone(&control),
        };
        let gate = Self {
            pp_log,
            name,
            paused_from: None,
            skipped: 0,
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, release_picture),
            control,
            pad,
        };
        (gate, handle)
    }
}

impl Element for PauseGate {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::PauseGate
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for PauseGate {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for PauseGate {
    /// Video frames. It reads a timestamp and forwards a reference, so where
    /// the pixels live is not its business.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::any_frame(MediaKind::VideoFrame))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            // `Eos` above all: whatever follows finalizes on it, and it
            // carries no timeline to move.
            return self.pad.push(buf);
        };
        // Nothing can be placed on the output timeline without one, and
        // guessing would put it wherever the last one happened to land.
        let Some(pts) = frame.pts() else {
            return Ok(());
        };

        if self.control.paused.load(Ordering::Relaxed) {
            // The first frame dropped is where this pause began. Later ones
            // only confirm it is still going.
            if self.paused_from.is_none() {
                self.paused_from = Some(pts);
                pp_info!(self, "paused at {pts}");
            }
            return Ok(());
        }
        if let Some(from) = self.paused_from.take() {
            // This frame is the first past the pause, so the span between is
            // what has to come out of everything from here on.
            let span = pts.saturating_sub(from).max(0);
            self.skipped = self.skipped.saturating_add(span);
            pp_info!(
                self,
                "resumed at {pts}, {span} skipped ({} total)",
                self.skipped
            );
        }

        // A wrapper of its own rather than the buffer it was handed: this
        // moves a timestamp, and the `Arc` may be shared with a sibling branch
        // off the same `Tee` which must keep the source's. What the wrapper
        // takes is a reference to the picture, not a copy of it.
        let mut moved = self.wrappers.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, unreferenced
        // before it is given a new one, and the source is the frame this call
        // was handed — both live, and distinct from each other.
        let code = unsafe {
            let ptr = moved.as_mut_ptr();
            ffmpeg::ffi::av_frame_unref(ptr);
            ffmpeg::ffi::av_frame_ref(ptr, frame.as_ref().as_ptr())
        };
        if code < 0 {
            return Err(ffmpeg::Error::from(code).into());
        }
        // After the reference, which copies the source's properties over
        // whatever the wrapper carried.
        moved.set_pts(Some(pts.saturating_sub(self.skipped)));
        self.pad.push(MediaBuffer::Video(Arc::new(moved)))
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // `Stop` abandons this run, so a pipeline started again owes nothing
        // to what the last one skipped.
        //
        // `Flush` deliberately keeps the total: it announces a new position in
        // the same timeline, and forgetting how much had been paused away
        // would send the output back by that much — timestamps going
        // backwards, which is not something a muxer accepts. A pause that was
        // open when it arrived is closed, because the position it was measured
        // from is no longer where the source is.
        match msg {
            ControlMsg::Stop => {
                self.paused_from = None;
                self.skipped = 0;
            }
            ControlMsg::Flush => self.paused_from = None,
            _ => {}
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
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

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn capture(element: &mut PauseGate) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// One tiny frame carrying `pts`, with the same value in its first pixel
    /// so a test can say *which* input frames came out.
    fn frame(pts: Option<i64>) -> MediaBuffer {
        let mut video = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::GRAY8, 2, 2);
        video.set_pts(pts);
        if let Some(pts) = pts {
            let stride = video.stride(0);
            video.data_mut(0)[..stride].fill(pts as u8);
        }
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut pooled = pool.get();
        *pooled = video;
        MediaBuffer::Video(Arc::new(pooled))
    }

    fn stamps(received: &Arc<Mutex<Vec<MediaBuffer>>>) -> Vec<i64> {
        received
            .lock()
            .unwrap()
            .iter()
            .filter_map(|buf| match buf {
                MediaBuffer::Video(frame) => frame.pts(),
                _ => None,
            })
            .collect()
    }

    fn kept(received: &Arc<Mutex<Vec<MediaBuffer>>>) -> Vec<u8> {
        received
            .lock()
            .unwrap()
            .iter()
            .filter_map(|buf| match buf {
                MediaBuffer::Video(frame) => Some(frame.data(0)[0]),
                _ => None,
            })
            .collect()
    }

    /// Running, it changes nothing. Anything else would make a gate that is
    /// never paused cost something.
    #[test]
    fn timestamps_pass_through_untouched_while_running() {
        let (mut gate, _handle) = PauseGate::new("gate");
        let received = capture(&mut gate);

        for pts in [10, 11, 12] {
            gate.consume(frame(Some(pts))).unwrap();
        }

        assert_eq!(stamps(&received), vec![10, 11, 12]);
    }

    /// The whole point: what follows a pause is moved back by exactly as much
    /// as was paused, so the output timeline has no gap in it.
    #[test]
    fn the_paused_span_comes_out_of_the_timeline() {
        let (mut gate, handle) = PauseGate::new("gate");
        let received = capture(&mut gate);

        for pts in 0..4 {
            gate.consume(frame(Some(pts))).unwrap();
        }
        handle.set_paused(true);
        for pts in 4..64 {
            gate.consume(frame(Some(pts))).unwrap();
        }
        handle.set_paused(false);
        for pts in 64..68 {
            gate.consume(frame(Some(pts))).unwrap();
        }

        assert_eq!(
            stamps(&received),
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            "the output must continue where it left off, with the pause gone"
        );
        assert_eq!(
            kept(&received),
            vec![0, 1, 2, 3, 64, 65, 66, 67],
            "the frames themselves are the ones that were live on either side"
        );
    }

    /// Two pauses accumulate rather than the second replacing the first.
    #[test]
    fn a_second_pause_is_taken_out_as_well() {
        let (mut gate, handle) = PauseGate::new("gate");
        let received = capture(&mut gate);

        gate.consume(frame(Some(0))).unwrap();
        handle.set_paused(true);
        for pts in 1..11 {
            gate.consume(frame(Some(pts))).unwrap();
        }
        handle.set_paused(false);
        gate.consume(frame(Some(11))).unwrap();
        handle.set_paused(true);
        for pts in 12..32 {
            gate.consume(frame(Some(pts))).unwrap();
        }
        handle.set_paused(false);
        gate.consume(frame(Some(32))).unwrap();

        // 10 skipped, then 20 more.
        assert_eq!(stamps(&received), vec![0, 1, 2]);
    }

    /// A muxer finalizes on it, so it must arrive however the gate is set.
    #[test]
    fn end_of_stream_is_forwarded_even_while_paused() {
        let (mut gate, handle) = PauseGate::new("gate");
        let received = capture(&mut gate);
        handle.set_paused(true);

        gate.consume(MediaBuffer::Eos).unwrap();

        assert!(matches!(
            received.lock().unwrap().first(),
            Some(MediaBuffer::Eos)
        ));
    }

    /// `Stop` ends this run, so a pipeline started again owes nothing to what
    /// the last one skipped. `Flush` is a new position in the same timeline,
    /// and forgetting the total there would send the output backwards.
    #[test]
    fn stop_forgets_what_was_skipped_and_flush_does_not() {
        let (mut gate, handle) = PauseGate::new("gate");
        let received = capture(&mut gate);

        gate.consume(frame(Some(0))).unwrap();
        handle.set_paused(true);
        gate.consume(frame(Some(1))).unwrap();
        handle.set_paused(false);
        gate.consume(frame(Some(11))).unwrap();
        assert_eq!(stamps(&received), vec![0, 1], "ten ticks were paused away");

        gate.control(ControlMsg::Flush).unwrap();
        gate.consume(frame(Some(21))).unwrap();
        assert_eq!(
            stamps(&received).last(),
            Some(&11),
            "a flush must not give back the ten it had skipped"
        );

        gate.control(ControlMsg::Stop).unwrap();
        gate.consume(frame(Some(31))).unwrap();
        assert_eq!(
            stamps(&received).last(),
            Some(&31),
            "a stopped run starts owing nothing"
        );
    }

    #[test]
    fn the_handle_reports_what_it_set() {
        let (_gate, handle) = PauseGate::new("gate");

        assert!(!handle.is_paused());
        handle.set_paused(true);
        assert!(handle.is_paused());
    }
}
