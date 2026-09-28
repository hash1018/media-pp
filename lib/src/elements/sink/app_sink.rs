use std::sync::Arc;

use crate::pp_log::{PpLog, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::InputContract,
    element::{Element, ElementType, Sink, element_pp_log},
    error::Result,
    stream::StreamEvent,
};

/// Terminal sink that hands every buffer (and, optionally, every event in
/// the stream) to a plain closure instead of requiring a bespoke `struct` +
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
pub struct AppSink<F, E> {
    pp_log: PpLog,
    name: Arc<str>,
    consume: F,
    events: E,
}

impl<F> AppSink<F, fn(&StreamEvent) -> Result<()>>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
{
    /// `consume` is the only thing this reacts to — the events in the
    /// stream, its segments and its end, are silently let by, the same as
    /// by `FrameCounter`/`PacketCounter`. Reach for [`AppSink::with_events`]
    /// instead if the closure needs to know about one of those — e.g.
    /// closing what it writes at the end, or resetting a tracker's history,
    /// or a batch buffer, where a seek begins a new segment.
    pub fn new(name: impl Into<String>, consume: F) -> Self {
        Self::with_events(name, consume, |_| Ok(()))
    }
}

impl<F, E> AppSink<F, E>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
    E: FnMut(&StreamEvent) -> Result<()> + Send + 'static,
{
    /// Same as [`AppSink::new`], but also hands every [`StreamEvent`] to
    /// `events`, in order with the buffers, instead of letting it by.
    ///
    /// ```
    /// # use media_pp::{elements::AppSink, stream::StreamEvent};
    /// let sink = AppSink::with_events(
    ///     "detector",
    ///     |_buf| Ok(()),
    ///     |event| {
    ///         match event {
    ///             // A seek's: e.g. clear a tracker's history here.
    ///             StreamEvent::Segment(segment) if segment.flushed => {}
    ///             // The end: e.g. close what the buffers went to.
    ///             StreamEvent::Eos => {}
    ///             _ => {}
    ///         }
    ///         Ok(())
    ///     },
    /// );
    /// ```
    pub fn with_events(name: impl Into<String>, consume: F, events: E) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::AppSink, &name, None);
        pp_info!(pp_log: &pp_log, "created");
        Self {
            name,
            pp_log,
            consume,
            events,
        }
    }
}

impl<F, E> Element for AppSink<F, E>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
    E: FnMut(&StreamEvent) -> Result<()> + Send + 'static,
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

impl<F, E> Sink for AppSink<F, E>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
    E: FnMut(&StreamEvent) -> Result<()> + Send + 'static,
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

    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        (self.events)(event)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ffmpeg_next as ffmpeg;

    use super::*;

    /// `AppSink::new`'s docs promise every event is *silently* let by —
    /// accepted and nothing done, not failing the stream the way an `Err`
    /// here would.
    #[test]
    fn new_lets_every_event_by() {
        let mut sink = AppSink::new("counter", |_buf| Ok(()));

        sink.stream_event(&StreamEvent::Eos).unwrap();
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

        let error = sink
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .unwrap_err();

        assert!(error.to_string().contains("closure failed"));
    }

    /// The end reaches the events closure, in order with the buffers,
    /// rather than being consumed by the sink itself — a caller that
    /// finalizes on it (a muxer wrapper, a channel it closes) only ever
    /// learns about it there; and its error comes back as the sink's.
    #[test]
    fn with_events_hands_the_closure_the_end_after_the_buffers() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut sink = AppSink::with_events(
            "recorder",
            {
                let seen = Arc::clone(&seen);
                move |_buf| {
                    seen.lock().unwrap().push("buffer");
                    Ok(())
                }
            },
            {
                let seen = Arc::clone(&seen);
                move |event: &StreamEvent| {
                    seen.lock().unwrap().push("end");
                    match event {
                        StreamEvent::Eos => {
                            Err(crate::error::Error::Other("trailer failed".into()))
                        }
                        _ => Ok(()),
                    }
                }
            },
        );

        sink.consume(MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())))
            .unwrap();
        let error = sink.stream_event(&StreamEvent::Eos).unwrap_err();

        assert_eq!(&*seen.lock().unwrap(), &["buffer", "end"]);
        assert!(error.to_string().contains("trailer failed"));
    }
}
