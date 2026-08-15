use std::sync::Arc;

use crate::clog::{CLog, cinfo};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_clog},
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
    clog: CLog,
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
        let clog = element_clog(ElementType::AppSink, &name, None);
        cinfo!(clog: &clog, "created");
        Self {
            name,
            clog,
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

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
    }
}

impl<F, C> Sink for AppSink<F, C>
where
    F: FnMut(MediaBuffer) -> Result<()> + Send + 'static,
    C: FnMut(ControlMsg) -> Result<()> + Send + 'static,
{
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        (self.consume)(buf)
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        (self.control)(msg)
    }
}
