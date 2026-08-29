use std::sync::Arc;

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

/// Forwards video frames at a fixed rate lower than the one they arrive at,
/// stamping what it forwards as a constant-rate stream of its own.
///
/// # What it is for
///
/// A branch that wants a different rate from the source everything else on
/// the `Tee` shares. A compositor runs at one rate because the Preview and
/// the reported figure are made of it; a recording made from the same frames
/// need not be written at that rate, and at 60 into 30 it is half the encode
/// and half the file for a result most viewers cannot tell apart.
///
/// Lowering the compositor instead would lower everything — the Preview, and
/// the rate the status bar reports a recording *would* be made at. So the
/// choice belongs on the branch, which is what this is.
///
/// # It re-stamps rather than passing timestamps through
///
/// Output `pts` is this element's own count of what it has forwarded, in
/// `1/rate`. That makes the output constant-rate by construction, which is
/// what a recording wants: converting the input timeline instead would leave
/// uneven gaps wherever the ratio is not a whole number — 60 into 24 is two
/// input frames for one output frame and then three — and write a
/// variable-rate file out of a source that was perfectly regular.
///
/// The consequence is that a source which stalls is elided rather than
/// represented: ten seconds of nothing becomes a cut, not ten seconds of
/// still picture. That is the same choice a constant frame rate always makes,
/// and it is why the rate this is given has to be the rate the encoder after
/// it is configured for.
///
/// Because the count starts at zero, so does the output — a
/// [`TimestampOrigin`](crate::elements::TimestampOrigin) after the encoder
/// then has nothing left to shift, and is only needed on a branch that has no
/// limiter.
///
/// # Which frames it keeps
///
/// The first frame whose own timestamp has reached each output tick, so the
/// spacing follows the input's timeline rather than its frame count. A source
/// that drops a frame, or pauses and resumes, still comes out at the rate
/// asked for; counting arrivals instead would let a stall shift every frame
/// after it.
///
/// # Cost
///
/// A reference to the frames it keeps and nothing at all for the ones it
/// drops. No pixels are read, copied or converted here.
pub struct FrameRateLimiter {
    pp_log: PpLog,
    name: Arc<str>,
    /// The unit incoming `pts` are counted in.
    input_time_base: ffmpeg::Rational,
    /// Frames per second out.
    rate: ffmpeg::Rational,
    /// The next output tick that has not been filled, in `1/rate`. `None`
    /// until the first timestamped frame decides where this stream starts.
    next_tick: Option<i64>,
    /// How many frames have been forwarded, which is also the next output
    /// `pts` — see this type's own docs on why it is a count rather than a
    /// conversion.
    forwarded: i64,
    /// Empty frames to point at the pictures this forwards, so each carries
    /// its own timestamp without touching the one the source stamped — a
    /// sibling branch off the same `Tee` must keep seeing that.
    ///
    /// `release_picture` when one comes back, so an idle wrapper does not
    /// pin a picture the producer's pool wants returned.
    wrappers: UnboundObjectPool<ffmpeg::frame::Video>,
    pad: SrcPad,
}

impl FrameRateLimiter {
    /// `input_time_base` is the unit the incoming frames' `pts` are in — the
    /// producer's own, such as `CudaVideoCompositor::time_base`. `rate` is
    /// frames per second out, and must be the rate whatever follows this is
    /// configured for: the timestamps this writes mean nothing else.
    pub fn new(
        name: impl Into<String>,
        input_time_base: ffmpeg::Rational,
        rate: ffmpeg::Rational,
    ) -> Self {
        let name: Arc<str> = name.into().into();
        let pad = SrcPad::with_contract(format!("{name}_src"), OutputContract::Passthrough);
        let pp_log = element_pp_log(ElementType::FrameRateLimiter, &name, None);
        Self {
            pp_log,
            name,
            input_time_base,
            rate,
            next_tick: None,
            forwarded: 0,
            wrappers: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, release_picture),
            pad,
        }
    }

    /// The unit this element's output `pts` are in, which is what an encoder
    /// or muxer after it has to be told.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(self.rate.denominator(), self.rate.numerator())
    }

    /// Which output tick a frame's own timestamp falls in.
    ///
    /// Worked out in `i128` because the multiplication is a timestamp by a
    /// rate by a time base, and at a fine input unit — microseconds, or a
    /// codec's own 1/90000 — that leaves `i64` well within reach of a
    /// recording that runs for hours.
    fn tick_of(&self, pts: i64) -> i64 {
        let numerator = i128::from(pts)
            * i128::from(self.input_time_base.numerator())
            * i128::from(self.rate.numerator());
        let denominator =
            i128::from(self.input_time_base.denominator()) * i128::from(self.rate.denominator());
        // Floor division, not truncation: a negative timestamp belongs to the
        // tick before zero rather than to zero itself, and rounding it the
        // wrong way would let two frames share a tick.
        let tick = numerator.div_euclid(denominator.max(1));
        tick.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
    }
}

impl Element for FrameRateLimiter {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::FrameRateLimiter
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for FrameRateLimiter {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for FrameRateLimiter {
    /// Video frames. It reads a timestamp and forwards a reference, so where
    /// the pixels live is not its business — but packets are not what it
    /// handles, and audio has no frame rate to limit.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::any_frame(MediaKind::VideoFrame))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            // `Eos` above all: whatever follows finalizes on it, and it
            // carries no timestamp to place.
            return self.pad.push(buf);
        };
        // A frame with no timestamp cannot be placed on the output timeline,
        // and guessing would put it wherever the last one happened to land.
        let Some(pts) = frame.pts() else {
            return Ok(());
        };
        let tick = self.tick_of(pts);
        let next = *self.next_tick.get_or_insert_with(|| {
            pp_info!(
                self,
                "limiting to {}/{} fps from {}",
                self.rate.numerator(),
                self.rate.denominator(),
                self.input_time_base
            );
            tick
        });
        if tick < next {
            // Still inside a tick already filled. Dropped rather than
            // forwarded: two frames on one output timestamp is not something
            // a muxer accepts.
            return Ok(());
        }
        // The tick this frame lands in is filled by it, and the next one is
        // what the following frame has to reach. Set from `tick` rather than
        // `next + 1` so a gap in the source does not leave this owing every
        // tick it skipped.
        self.next_tick = Some(tick + 1);

        // A wrapper of its own rather than the buffer it was handed: this
        // stamps a timestamp, and the `Arc` may be shared with a sibling
        // branch off the same `Tee` which must keep the source's. What the
        // wrapper takes is a reference to the picture, not a copy of it.
        let mut stamped = self.wrappers.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, unreferenced
        // before it is given a new one, and the source is the frame this call
        // was handed — both live, and distinct from each other.
        let code = unsafe {
            let ptr = stamped.as_mut_ptr();
            ffmpeg::ffi::av_frame_unref(ptr);
            ffmpeg::ffi::av_frame_ref(ptr, frame.as_ref().as_ptr())
        };
        if code < 0 {
            return Err(ffmpeg::Error::from(code).into());
        }
        // After the reference, which copies the source's properties over
        // whatever the wrapper carried.
        stamped.set_pts(Some(self.forwarded));
        self.forwarded += 1;
        self.pad.push(MediaBuffer::Video(Arc::new(stamped)))
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // `Stop` abandons this run, so a pipeline started again begins a new
        // output timeline at zero like the first one did.
        //
        // `Flush` deliberately does not reset the count: it announces a new
        // position in the same output stream, and restarting the timestamps
        // would send them backwards — which is not something a muxer accepts.
        // The tick does go, because the position it was tracking is no longer
        // where the source is.
        match msg {
            ControlMsg::Stop => {
                self.next_tick = None;
                self.forwarded = 0;
            }
            ControlMsg::Flush => self.next_tick = None,
            _ => {}
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

    fn capture(element: &mut FrameRateLimiter) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// One tiny frame carrying `pts`. The luma value is `pts` too, so a test
    /// can say *which* input frames came out rather than only how many.
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

    /// The output `pts` of everything that reached the sink.
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

    /// Which *input* frame each output carries, read out of the picture.
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

    fn sixty_into(rate: i32) -> FrameRateLimiter {
        FrameRateLimiter::new(
            "limit",
            ffmpeg::Rational::new(1, 60),
            ffmpeg::Rational::new(rate, 1),
        )
    }

    /// The whole point: half the frames out, and the ones kept are evenly
    /// spaced across the input rather than clustered.
    #[test]
    fn sixty_into_thirty_keeps_every_other_frame() {
        let mut limiter = sixty_into(30);
        let received = capture(&mut limiter);

        for pts in 0..10 {
            limiter.consume(frame(Some(pts))).unwrap();
        }

        assert_eq!(kept(&received), vec![0, 2, 4, 6, 8]);
    }

    /// What option (a) buys: the output is constant-rate by construction, so
    /// a muxer writes a file that plays at the rate it was told.
    #[test]
    fn output_timestamps_count_from_zero_whatever_the_input_did() {
        let mut limiter = sixty_into(30);
        let received = capture(&mut limiter);

        // A source already well into its own timeline, as a branch attached
        // to a running `Tee` always is.
        for pts in 1_000..1_010 {
            limiter.consume(frame(Some(pts))).unwrap();
        }

        assert_eq!(stamps(&received), vec![0, 1, 2, 3, 4]);
    }

    /// A ratio that is not a whole number is where counting arrivals would
    /// go wrong: 60 into 24 is two input frames for one output and then
    /// three, and the spacing has to follow the timeline.
    #[test]
    fn sixty_into_twenty_four_keeps_the_right_frames() {
        let mut limiter = sixty_into(24);
        let received = capture(&mut limiter);

        for pts in 0..60 {
            limiter.consume(frame(Some(pts))).unwrap();
        }

        let kept = kept(&received);
        assert_eq!(
            kept.len(),
            24,
            "a second of input must be a second of output"
        );
        assert_eq!(&kept[..6], &[0, 3, 5, 8, 10, 13]);
        // Consecutive all the way, which is what makes the file constant-rate.
        assert_eq!(stamps(&received), (0..24).collect::<Vec<_>>());
    }

    /// The rate asked for is what comes out even when the source misses its
    /// own: the frames that do arrive are placed by their timestamps, so a
    /// gap stays a gap instead of shifting everything after it.
    #[test]
    fn a_source_that_drops_frames_still_lands_on_its_own_timeline() {
        let mut limiter = sixty_into(30);
        let received = capture(&mut limiter);

        // 0..4 arrive, 4..8 never do, 8..12 resume.
        for pts in (0..4).chain(8..12) {
            limiter.consume(frame(Some(pts))).unwrap();
        }

        assert_eq!(
            kept(&received),
            vec![0, 2, 8, 10],
            "the frames after the gap must be placed where their timestamps say"
        );
    }

    /// Nothing else can place it on the output timeline, and letting it
    /// through would put it wherever the last one happened to land.
    #[test]
    fn a_frame_without_a_timestamp_is_dropped() {
        let mut limiter = sixty_into(30);
        let received = capture(&mut limiter);

        limiter.consume(frame(None)).unwrap();

        assert!(stamps(&received).is_empty());
    }

    /// A muxer finalizes on it, so it must arrive whatever the rate is doing.
    #[test]
    fn end_of_stream_is_forwarded() {
        let mut limiter = sixty_into(30);
        let received = capture(&mut limiter);

        limiter.consume(MediaBuffer::Eos).unwrap();

        assert!(matches!(
            received.lock().unwrap().first(),
            Some(MediaBuffer::Eos)
        ));
    }

    /// `Stop` ends this run, so a pipeline started again writes a new stream
    /// from zero. `Flush` is a new position in the *same* stream, and
    /// restarting the count there would send timestamps backwards.
    #[test]
    fn stop_restarts_the_count_and_flush_does_not() {
        let mut limiter = sixty_into(30);
        let received = capture(&mut limiter);

        limiter.consume(frame(Some(0))).unwrap();
        limiter.consume(frame(Some(2))).unwrap();
        limiter.control(ControlMsg::Flush).unwrap();
        limiter.consume(frame(Some(100))).unwrap();
        assert_eq!(
            stamps(&received),
            vec![0, 1, 2],
            "a flush must not send the output back to zero"
        );

        limiter.control(ControlMsg::Stop).unwrap();
        limiter.consume(frame(Some(200))).unwrap();
        assert_eq!(
            stamps(&received).last(),
            Some(&0),
            "a stopped run starts over"
        );
    }
}
