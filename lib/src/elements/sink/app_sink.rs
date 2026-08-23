use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::InputContract,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
};

/// Terminal sink that hands every buffer (and, optionally, every control
/// message) to a plain closure instead of requiring a bespoke `struct` +
/// `Element`/`Sink` impl — the equivalent of GStreamer's `appsink`: the
/// pipeline's job ends here, and whatever the caller does with the data
/// (run inference, forward it to a channel, write it out, ...) is none of
/// this crate's concern.
///
/// `FrameCounter`/`PacketCounter` are what a one-off consumer looked
/// like *before* this existed — this is the general case of the same
/// pattern, for when a whole new type per use site is more ceremony than
/// the actual logic warrants:
///
/// ```
/// # use media_pp::{buffer::MediaBuffer, elements::AppSink};
/// let mut count = 0usize;
/// let sink = AppSink::new("counter", move |buf: MediaBuffer| {
///     if matches!(buf, MediaBuffer::Video(_)) {
///         count += 1;
///     }
///     Ok(())
/// });
/// ```
pub struct AppSink<F, C> {
    pp_log: PpLog,
    name: Arc<str>,
    consume: F,
    control: C,
}

impl<F> AppSink<F, fn(ControlMsg) -> Result<()>>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
{
    /// `consume` is the only thing this reacts to — every `ControlMsg`
    /// (`Pause`/`Resume`/`Stop`/`Seek`) is silently ignored, the same as
    /// `FrameCounter`/`PacketCounter`. Reach for
    /// [`AppSink::with_control`] instead if the closure needs to know
    /// about one of those — e.g. resetting a tracker's history, or a
    /// batch buffer, on `Seek`, the same way `SwDecoder`/`Pacer` react to
    /// it internally.
    pub fn new(name: impl Into<String>, consume: F) -> Self {
        Self::with_control(name, consume, |_| Ok(()))
    }
}

impl<F, C> AppSink<F, C>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
    C: FnMut(ControlMsg) -> Result<()> + Send + 'static,
{
    /// Same as [`AppSink::new`], but also hands every [`ControlMsg`] to
    /// `control` instead of silently dropping it.
    ///
    /// ```
    /// # use media_pp::{control::ControlMsg, elements::AppSink};
    /// let sink = AppSink::with_control(
    ///     "detector",
    ///     |_buf| Ok(()),
    ///     |msg| {
    ///         if let ControlMsg::Seek(_) = msg {
    ///             // e.g. clear a tracker's history here
    ///         }
    ///         Ok(())
    ///     },
    /// );
    /// ```
    pub fn with_control(name: impl Into<String>, consume: F, control: C) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AppSink, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name,
            pp_log,
            consume,
            control,
        }
    }
}

impl<F, C> Element for AppSink<F, C>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
    C: FnMut(ControlMsg) -> Result<()> + Send + 'static,
{
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AppSink
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl<F, C> Sink for AppSink<F, C>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
    C: FnMut(ControlMsg) -> Result<()> + Send + 'static,
{
    /// Every buffer reaches the closure verbatim, so this element never
    /// rejects one itself. It is a claim about this sink, not about the
    /// closure: one that only understands packets still returns its own
    /// error for a frame, which is application behavior a link check
    /// cannot see.
    fn input_contract(&self) -> InputContract {
        InputContract::Any
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        (self.consume)(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        (self.control)(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    use ffmpeg_next as ffmpeg;

    use super::*;

    fn control_messages() -> [ControlMsg; 4] {
        [
            ControlMsg::Pause,
            ControlMsg::Resume,
            ControlMsg::Stop,
            ControlMsg::Seek(Duration::from_secs(1)),
        ]
    }

    /// `AppSink::new`'s docs promise every `ControlMsg` is *silently*
    /// ignored — accepting it and doing nothing, not failing the control
    /// cascade the way an `Err` here would.
    #[test]
    fn new_accepts_and_ignores_every_control_message() {
        let mut sink = AppSink::new("counter", |_buf| Ok(()));

        for msg in control_messages() {
            sink.control(msg).unwrap();
        }
    }

    /// The whole point of `with_control` over `new`: no variant is
    /// filtered out on the way to the closure.
    #[test]
    fn with_control_forwards_every_control_message() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let mut sink = AppSink::with_control(
            "detector",
            |_buf| Ok(()),
            move |msg| {
                recorded.lock().unwrap().push(msg);
                Ok(())
            },
        );

        for msg in control_messages() {
            sink.control(msg).unwrap();
        }

        assert_eq!(&*seen.lock().unwrap(), &control_messages());
    }

    /// A terminal `Sink`'s error has to come back out of `consume`
    /// unchanged: that return value is what a direct caller propagates
    /// with `?`, and what a `Queue` worker turns into `BusEvent::Error`.
    /// Swallowing it here would make both silently impossible.
    #[test]
    fn consume_error_propagates_to_the_caller() {
        let mut sink = AppSink::new("failing", |_buf| {
            Err(crate::error::Error::Other("closure failed".into()))
        });

        let error = sink.consume(MediaBuffer::Eos).unwrap_err();

        assert!(error.to_string().contains("closure failed"));
    }

    /// `Eos` reaches the closure like any other buffer rather than being
    /// consumed by the sink itself — a caller that finalizes on EOS (a
    /// muxer wrapper, a channel it closes) only ever learns about it here.
    #[test]
    fn every_buffer_including_eos_reaches_the_closure() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let mut sink = AppSink::new("recorder", move |buf| {
            recorded
                .lock()
                .unwrap()
                .push(matches!(buf, MediaBuffer::Eos));
            Ok(())
        });

        sink.consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())))
            .unwrap();
        sink.consume(MediaBuffer::Eos).unwrap();

        assert_eq!(&*seen.lock().unwrap(), &[false, true]);
    }
}
