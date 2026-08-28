//! Answering an unchanged input with the output already made from it.
//!
//! A live graph produces frames at a rate rather than on change: a screen
//! capture of a still desktop re-emits the picture it already has, and a
//! compositor with nothing to recompose does the same. Every per-frame
//! element downstream then does its work again — an upload, a readback, a
//! scale, a shader pass — to produce a result indistinguishable from the
//! last one.
//!
//! [`PerFrameTransform`] is the shape of an element that can answer such a
//! repeat with what it already produced, and [`RepeatedOutput`] is what it
//! keeps in order to. What goes downstream is an empty wrapper referencing
//! the picture already produced, under this frame's own timestamp: no pixels
//! are copied and the rate downstream sees does not change.
//!
//! # What is *not* here
//!
//! An element whose output depends on more than the frame it was handed —
//! the video compositors, with their layers, text and several inputs, and
//! `DxgiCaptureSource`, with its cursor — recognises its own repeats
//! against its own state. This is for the ordinary case: one frame in, one
//! frame out, the output a function of the input alone.
//!
//! `CudaScaler` is the other exclusion, and for the other reason: it scales
//! through a libavfilter graph, which answers one frame with none or several
//! and holds frames of its own to flush at end of stream. "The output" is not
//! a single frame there, so there is nothing for this to hand out again.
//!
//! Nothing here belongs in front of an element whose contract is a *rate*.
//! An encoder or a muxer needs the frame per tick, and an encoder given a
//! picture identical to the last one is already at its cheapest.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};

use crate::{
    buffer::{picture_id, picture_is_referenced, release_picture},
    error::Result,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

/// Everything about an input frame that can change what an element makes of
/// it: which buffer its pixels live in, and how they are to be read.
///
/// The pixels are identified by address rather than by content — see
/// [`picture_id`], including why that is only sound while the frame is held.
/// The properties are part of it because they travel separately from the
/// pixels: one texture can arrive tagged BT.601 and then BT.709, and a
/// consumer that reads the tag makes two different pictures out of it.
pub(crate) type InputIdentity = (
    (usize, usize),
    ffmpeg::format::Pixel,
    u32,
    u32,
    ffmpeg::color::Space,
    ffmpeg::color::Range,
);

/// See [`InputIdentity`].
pub(crate) fn input_identity(frame: &ffmpeg::frame::Video) -> InputIdentity {
    (
        picture_id(frame),
        frame.format(),
        frame.width(),
        frame.height(),
        frame.color_space(),
        frame.color_range(),
    )
}

/// One frame in, one frame out, the output a function of the input alone.
///
/// Implementing this is how such an element answers a repeated input with
/// what it already produced instead of producing it again. The order the
/// steps have to happen in lives in [`PerFrameTransform::transform`], not in
/// each element: recognise the repeat, or produce and *remember*. Forgetting
/// the last of those is what would leave an element doing the work every
/// time while looking like it does not.
///
/// An element supplies the three things only it can know: where its cache
/// lives, how it makes a frame, and what to call the failure. What counts as
/// the same input is [`input_identity`]'s answer, which is the picture plus
/// the properties that decide how it is read — an element needing more than
/// that (a runtime setting of its own, say) does not fit this trait and
/// should compare its own state, as the compositors do.
pub(crate) trait PerFrameTransform {
    /// This element's own [`RepeatedOutput`] field.
    fn repeated(&mut self) -> &mut RepeatedOutput;

    /// Makes one output frame from `input` — the work this element exists to
    /// do, including whatever properties it stamps on the result. Called
    /// only when the input is not one already answered.
    ///
    /// Handed the pooled reference rather than the frame inside it, because
    /// an element that leaves GPU work running past this call has to keep
    /// the input checked out until it completes — `D3d12Scaler` holds it in
    /// the command slot it just queued. One that reads the pixels and is
    /// done simply derefs.
    fn produce(
        &mut self,
        input: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>>;

    /// This element's own error for a failed `av_frame_ref`, so that a
    /// repeat that cannot be referenced is reported as this element's
    /// failure rather than a generic one.
    fn frame_ref_failed(&self, code: i32) -> crate::error::Error;

    /// The whole sequence, in the one order that is correct.
    fn transform(
        &mut self,
        input: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>> {
        if self.repeated().made_from(input) {
            return match self.repeated().repeat(input) {
                Ok(output) => Ok(Arc::new(output)),
                Err(code) => Err(self.frame_ref_failed(code)),
            };
        }
        let produced = Arc::new(self.produce(input)?);
        self.repeated().store(input, Arc::clone(&produced));
        Ok(produced)
    }
}

/// The output an element last produced, and the input it was made from.
///
/// # What it holds, and why each of them
///
/// The **input** is held as the pooled reference the element was handed, not
/// as an `av_frame_ref` of the picture inside it. A frame reference keeps
/// the buffer allocated but leaves the producer's pool free to hand that
/// slot out again, and a producer that composites or scales into its pooled
/// frames then puts new pixels at the very address being compared — so a
/// changed picture would read as unchanged. Holding the pooled reference
/// keeps that slot checked out for as long as this names it, at the cost of
/// one frame the producer's pool grows by.
///
/// The **output** is held for the mirror-image reason: a repeat shares its
/// buffer but not its pool slot, so the slot must stay checked out while any
/// repeat is still in flight or the element's next output would be written
/// over pixels still queued downstream. A replaced output therefore moves to
/// `retired` and is let go only once [`picture_is_referenced`] reads false
/// for it.
pub(crate) struct RepeatedOutput {
    held: Option<Held>,
    retired: Vec<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
    /// Empty frames, never produced into: each one is pointed at the held
    /// output and carries only this frame's own timestamp.
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
}

struct Held {
    input: InputIdentity,
    /// See the type docs: this is what makes `input` an identity.
    _input: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    output: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
}

impl RepeatedOutput {
    pub(crate) fn new() -> Self {
        Self {
            held: None,
            retired: Vec::new(),
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, release_picture),
        }
    }

    /// Whether `input` is the very picture the held output was made from, so
    /// that making it again would produce what is already in hand.
    pub(crate) fn made_from(&self, input: &ffmpeg::frame::Video) -> bool {
        self.held
            .as_ref()
            .is_some_and(|held| held.input == input_identity(input))
    }

    /// A wrapper around the held output, under `input`'s own timing.
    /// Returns the FFmpeg error code on failure, for the caller to report as
    /// its own error type.
    ///
    /// Everything but the timing comes from the held output rather than from
    /// `input`, because it is the *output's* description that has to travel
    /// with these pixels: an element that converts color hands downstream
    /// the tags it converted to, and a repeat of that frame is still that
    /// frame. Only call this where [`RepeatedOutput::made_from`] returned
    /// true.
    pub(crate) fn repeat(
        &self,
        input: &ffmpeg::frame::Video,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, i32> {
        let held = self
            .held
            .as_ref()
            .expect("only reached with a previous output");
        let mut wrapper = self.wrappers.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, unreferenced
        // before it is given a new one; the source of that reference is the
        // output this still holds. Both are live and distinct, so this adds
        // a reference rather than copying any pixels — and carries the
        // output's own properties with it.
        unsafe {
            let ptr = wrapper.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            let code = ffi::av_frame_ref(ptr, held.output.as_ptr());
            if code < 0 {
                return Err(code);
            }
            // This frame's own place in the timeline, not the one the held
            // output was produced at.
            (*ptr).pts = (*input.as_ptr()).pts;
            (*ptr).pkt_dts = (*input.as_ptr()).pkt_dts;
            (*ptr).duration = (*input.as_ptr()).duration;
        }
        Ok(wrapper)
    }

    /// Remembers what `input` produced, so the next frame carrying the same
    /// picture can be answered with it.
    pub(crate) fn store(
        &mut self,
        input: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
        output: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) {
        if let Some(previous) = self.held.take() {
            self.retired.push(previous.output);
        }
        self.retired.retain(|output| picture_is_referenced(output));
        self.held = Some(Held {
            input: input_identity(input),
            _input: Arc::clone(input),
            output,
        });
    }

    /// Forgets everything held, so the next frame is produced afresh.
    ///
    /// For a `Flush` or a `Stop`: what comes next may be a different stream,
    /// and nothing downstream should be answered out of a previous one.
    pub(crate) fn clear(&mut self) {
        self.held = None;
        self.retired.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pooled(frame: ffmpeg::frame::Video) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = frame;
        Arc::new(slot)
    }

    fn picture(pts: i64) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16);
        frame.set_pts(Some(pts));
        pooled(frame)
    }

    /// Another `AVFrame` over the same picture, with its own timestamp —
    /// what a producer with nothing new to show hands over on every tick.
    fn repeat_of(
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
        pts: i64,
    ) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let mut kept = ffmpeg::frame::Video::empty();
        // SAFETY: both are live `AVFrame`s and distinct — `kept` is the empty
        // local above.
        unsafe {
            assert!(ffi::av_frame_ref(kept.as_mut_ptr(), frame.as_ptr()) >= 0);
        }
        kept.set_pts(Some(pts));
        pooled(kept)
    }

    #[test]
    fn a_repeated_input_is_answered_with_the_output_it_already_made() {
        let mut cache = RepeatedOutput::new();
        let input = picture(1);
        let output = picture(1);
        assert!(!cache.made_from(&input), "nothing produced yet");

        cache.store(&input, Arc::clone(&output));
        let again = repeat_of(&input, 2);
        assert!(cache.made_from(&again));

        let repeated = cache.repeat(&again).expect("reference the held output");
        assert_eq!(
            picture_id(&repeated),
            picture_id(&output),
            "a repeat points at the output already produced"
        );
        assert_eq!(
            repeated.pts(),
            Some(2),
            "under this frame's own timestamp, not the held output's"
        );
    }

    /// What describes the pixels is the output's own tags: an element that
    /// converted color hands those downstream, and a repeat is still that
    /// frame.
    #[test]
    fn a_repeat_carries_the_output_description_not_the_input_one() {
        let mut cache = RepeatedOutput::new();
        let input = picture(1);
        let mut output = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16);
        output.set_color_space(ffmpeg::color::Space::BT709);
        output.set_color_range(ffmpeg::color::Range::MPEG);
        cache.store(&input, pooled(output));

        let repeated = cache.repeat(&repeat_of(&input, 2)).expect("repeat");
        assert_eq!(repeated.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(repeated.color_range(), ffmpeg::color::Range::MPEG);
    }

    /// The identity is the picture *and* how it is to be read: the same
    /// pixels tagged differently are a different output.
    #[test]
    fn a_re_tagged_picture_is_not_the_input_it_was_made_from() {
        let mut cache = RepeatedOutput::new();
        let input = picture(1);
        cache.store(&input, picture(1));

        let mut re_tagged = repeat_of(&input, 2);
        Arc::get_mut(&mut re_tagged)
            .expect("sole owner")
            .set_color_space(ffmpeg::color::Space::BT470BG);
        assert!(!cache.made_from(&re_tagged));
    }

    /// A held output must stay out of its pool while a repeat of it is still
    /// in flight, or the element's next output is produced over pixels still
    /// queued downstream.
    #[test]
    fn an_output_still_under_a_repeat_is_not_released() {
        let pool = UnboundObjectPool::new(
            1,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16),
            |_| {},
        );
        let mut cache = RepeatedOutput::new();
        let first_input = picture(1);
        cache.store(&first_input, Arc::new(pool.get()));

        let in_flight = cache.repeat(&first_input).expect("repeat");
        // The output itself goes back, as it would once downstream consumed
        // it; only the repeat above still refers to those pixels.
        cache.store(&picture(2), Arc::new(pool.get()));

        assert_eq!(
            pool.size(),
            0,
            "the pool handed back a frame a repeat is still showing"
        );
        // Dropping the repeat returns its wrapper to the pool it came from,
        // which is where it lets go of the picture — see `release_picture`.
        drop(in_flight);
        cache.store(&picture(3), Arc::new(pool.get()));
        assert_eq!(
            pool.size(),
            2,
            "a picture nobody holds must be reusable — both the one the repeat \
             was showing and the one that replaced it"
        );
    }
}
