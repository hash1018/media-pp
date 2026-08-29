use std::sync::Arc;
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::{MediaBuffer, picture_id},
    contract::InputContract,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPoolRef,
};

/// Forwards a video frame only when its picture is not the one it last
/// forwarded, and never more often than `min_interval`.
///
/// # What it is for
///
/// A live source produces frames at a rate, not on change. A screen capture
/// of a still desktop captures nothing and re-emits the picture it already
/// has; a compositor with nothing to recompose does the same. Everything
/// downstream of that then works at the full rate to produce a result
/// identical to the last one — and for a terminal that repaints a window per
/// frame, that is the whole cost of an idle scene.
///
/// Put this in front of such a terminal and the idle scene costs nothing: it
/// is not called at all until the picture changes.
///
/// # Why the rate limit lives here too
///
/// These two are one decision, and separating them breaks the second one.
///
/// A terminal that drops frames of its own — most do, to hold a preview to a
/// rate a window can usefully repaint at — can drop a frame that *did* carry
/// a change. The repeats that follow carry that same new picture, so a gate
/// placed before such a terminal will suppress every one of them, and the
/// display stays on the picture before the change until something else
/// changes. Dropping to a rate and dropping repeats have to happen in that
/// order, in one place, or the second silently defeats itself.
///
/// So the contract this offers a terminal is the useful one: *the newest
/// picture, no more often than this, and never the same one twice*. A
/// terminal behind this one should draw everything it receives.
///
/// # What it compares
///
/// Which buffer the pixels live in, not the frame around them — see
/// `buffer::picture_id`. It holds the frame it forwarded, which is what makes that
/// identity sound: a picture still held cannot be handed out again with
/// something else in it.
///
/// What it holds is the *pooled* reference it was given, not an
/// `av_frame_ref` of the picture inside it. A frame reference keeps the
/// buffer allocated but leaves the producer's pool free to hand that slot
/// out again — and a producer that composites or scales into its pooled
/// frames then puts new pixels at the very address this is comparing
/// against, so a real change reads as "unchanged" and, if the scene goes
/// still right after, the display stays on the picture before it. Holding
/// the pooled reference keeps that slot checked out for exactly as long as
/// this names it. It costs the producer one frame, which its pool simply
/// grows by.
///
/// It reads no pixels, so it works the same on a GPU frame as on one in
/// system memory, and costs a pointer comparison either way.
pub struct ChangeGate {
    pp_log: PpLog,
    name: Arc<str>,
    min_interval: Duration,
    forwarded: Option<Forwarded>,
    pad: SrcPad,
}

/// The last picture forwarded, and when.
struct Forwarded {
    picture: (usize, usize),
    /// The pooled frame itself, held so nothing can be composed or scaled
    /// into that slot while `picture` still names it — see this element's
    /// own docs on why a frame reference is not enough.
    _frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    at: Instant,
}

impl ChangeGate {
    /// `min_interval` is the shortest time between two forwarded frames.
    /// [`Duration::ZERO`] forwards every change as it arrives.
    pub fn new(name: impl Into<String>, min_interval: Duration) -> Self {
        let name: Arc<str> = name.into().into();
        let pad = SrcPad::new(format!("{name}_src"));
        let pp_log = element_pp_log(ElementType::ChangeGate, &name, None);
        pp_info!(pp_log: &pp_log, "created: at most one frame per {min_interval:?}");
        Self {
            pp_log,
            name,
            min_interval,
            forwarded: None,
            pad,
        }
    }

    /// Whether this frame is one the terminal downstream has not seen.
    fn is_new(&self, frame: &ffmpeg::frame::Video, now: Instant) -> bool {
        match &self.forwarded {
            // The rate first, so that a change dropped here is still carried
            // by the repeats that follow it — see this type's own docs.
            Some(forwarded) => {
                now.duration_since(forwarded.at) >= self.min_interval
                    && forwarded.picture != picture_id(frame)
            }
            None => true,
        }
    }

    fn remember(&mut self, frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>, now: Instant) {
        self.forwarded = Some(Forwarded {
            picture: picture_id(frame),
            _frame: Arc::clone(frame),
            at: now,
        });
    }
}

impl Element for ChangeGate {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::ChangeGate
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for ChangeGate {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for ChangeGate {
    /// Whatever arrives, wherever its pixels live: this compares pointers and
    /// reads nothing, so it passes the upstream contract straight through.
    fn input_contract(&self) -> InputContract {
        InputContract::Any
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            // Everything that is not a picture — end of stream above all —
            // has nothing to be unchanged about.
            return self.pad.push(buf);
        };
        let now = Instant::now();
        if !self.is_new(frame, now) {
            return Ok(());
        }
        self.remember(frame, now);
        self.pad.push(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            // Whatever comes next is new by definition: downstream has been
            // reset, or is about to stop.
            self.forwarded = None;
        }
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::pool::UnboundObjectPool;

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

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// One frame with its own picture. System memory, because what this gate
    /// compares is the same either way and a test needs no GPU for it.
    fn picture(pts: i64) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16);
        frame.set_pts(Some(pts));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        MediaBuffer::Video(Arc::new(slot))
    }

    /// Another `AVFrame` over the same picture — what a source with nothing
    /// new to show hands over on every tick.
    fn repeat_of(buffer: &MediaBuffer, pts: i64) -> MediaBuffer {
        let MediaBuffer::Video(frame) = buffer else {
            panic!("expected a Video buffer");
        };
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        // SAFETY: both are live `AVFrame`s and distinct — the slot is the
        // empty one just taken from the pool.
        unsafe {
            assert!(ffmpeg::ffi::av_frame_ref(slot.as_mut_ptr(), frame.as_ptr()) >= 0);
        }
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    fn timestamps(received: &Arc<Mutex<Vec<MediaBuffer>>>) -> Vec<Option<i64>> {
        received
            .lock()
            .unwrap()
            .iter()
            .map(|buf| match buf {
                MediaBuffer::Video(frame) => frame.pts(),
                other => panic!("expected a Video buffer, got {}", other.kind()),
            })
            .collect()
    }

    #[test]
    fn a_repeated_picture_is_not_forwarded_twice() {
        let mut gate = ChangeGate::new("gate", Duration::ZERO);
        let forwarded = capture(&mut gate);
        let first = picture(1);

        gate.consume(repeat_of(&first, 1)).expect("first");
        gate.consume(repeat_of(&first, 2)).expect("repeat");
        gate.consume(repeat_of(&first, 3)).expect("repeat");

        assert_eq!(
            timestamps(&forwarded),
            vec![Some(1)],
            "only the first arrival carried a picture downstream had not seen"
        );
    }

    /// A producer that composites into its pooled frames puts new pixels at
    /// an address this gate may already be naming. Holding the pooled frame
    /// is what stops that: while the gate names a slot, the pool cannot hand
    /// it back, so the same address never comes to mean a different picture.
    #[test]
    fn a_recycled_pool_slot_cannot_come_back_as_an_unchanged_picture() {
        let mut gate = ChangeGate::new("gate", Duration::ZERO);
        let forwarded = capture(&mut gate);
        // One frame's worth of pool, as a producer that reuses its output
        // buffer has: whatever it hands out next is this same slot, unless
        // something still holds it.
        let pool = UnboundObjectPool::new(
            1,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16),
            |_| {},
        );
        let produce = |pts: i64| {
            let mut slot = pool.get();
            slot.set_pts(Some(pts));
            MediaBuffer::Video(Arc::new(slot))
        };

        // Forwarded, then consumed downstream — nothing but the gate is left
        // holding it.
        gate.consume(produce(1)).expect("first");
        forwarded.lock().unwrap().clear();

        // A second composite. It only lands at the same address if the pool
        // handed the slot back, which is the case being pinned.
        gate.consume(produce(2)).expect("a new picture");

        assert_eq!(
            timestamps(&forwarded),
            vec![Some(2)],
            "a new picture was suppressed because it reused a recycled address"
        );
    }

    #[test]
    fn a_changed_picture_is_forwarded() {
        let mut gate = ChangeGate::new("gate", Duration::ZERO);
        let forwarded = capture(&mut gate);
        let first = picture(1);

        gate.consume(repeat_of(&first, 1)).expect("first");
        gate.consume(repeat_of(&first, 2)).expect("repeat");
        gate.consume(picture(3)).expect("changed");

        assert_eq!(timestamps(&forwarded), vec![Some(1), Some(3)]);
    }

    #[test]
    fn nothing_is_forwarded_faster_than_the_interval() {
        let mut gate = ChangeGate::new("gate", Duration::from_secs(60));
        let forwarded = capture(&mut gate);

        gate.consume(picture(1)).expect("first");
        gate.consume(picture(2)).expect("too soon");
        gate.consume(picture(3)).expect("too soon");

        assert_eq!(
            timestamps(&forwarded),
            vec![Some(1)],
            "the first frame sets the clock; the rest are inside the interval"
        );
    }

    /// The order the two rules are applied in, and the reason they are one
    /// element: a change held back by the interval must still reach
    /// downstream through the repeats that follow it.
    #[test]
    fn a_change_held_back_by_the_interval_arrives_on_its_next_repeat() {
        let mut gate = ChangeGate::new("gate", Duration::from_millis(30));
        let forwarded = capture(&mut gate);

        gate.consume(picture(1)).expect("first");
        let changed = picture(2);
        gate.consume(repeat_of(&changed, 2)).expect("too soon");
        std::thread::sleep(Duration::from_millis(35));
        gate.consume(repeat_of(&changed, 3)).expect("repeat");

        assert_eq!(
            timestamps(&forwarded),
            vec![Some(1), Some(3)],
            "the new picture must not be lost because its first arrival was early"
        );
    }

    #[test]
    fn end_of_stream_is_forwarded_whatever_came_before() {
        let mut gate = ChangeGate::new("gate", Duration::from_secs(60));
        let forwarded = capture(&mut gate);

        gate.consume(picture(1)).expect("first");
        gate.consume(MediaBuffer::Eos).expect("eos");

        let received = forwarded.lock().unwrap();
        assert!(matches!(received[1], MediaBuffer::Eos));
    }

    #[test]
    fn a_flush_makes_the_next_picture_new_again() {
        let mut gate = ChangeGate::new("gate", Duration::ZERO);
        let forwarded = capture(&mut gate);
        let first = picture(1);

        gate.consume(repeat_of(&first, 1)).expect("first");
        gate.control(ControlMsg::Flush).expect("flush");
        gate.consume(repeat_of(&first, 2)).expect("after flush");

        assert_eq!(
            timestamps(&forwarded),
            vec![Some(1), Some(2)],
            "downstream was reset, so it has not seen this picture"
        );
    }
}
