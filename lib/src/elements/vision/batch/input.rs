//! One input of a [`StreamMux`](super::StreamMux): the sink at the end of
//! the pipeline that feeds it.

use std::sync::{Arc, Weak};

use crate::pp_log::{PpLog, pp_debug, pp_error};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, PortContract},
    control::ControlMsg,
    element::{Context, Element, ElementType, Flow, RawSink},
    error::Result,
    stream::StreamEvent,
};

use super::{MuxShared, StreamMuxError};

/// One [`StreamMux`](super::StreamMux) input, returned by
/// [`StreamMuxHandle::add_source`](super::StreamMuxHandle::add_source).
///
/// Queues each picture for the mux to take into a batch. Live, a queue
/// already full lets go of its oldest picture for the new one — a camera
/// faster than the batches are made loses pictures rather than falling
/// behind; offline, a full queue holds its feeding pipeline back instead.
///
/// `consume` runs on the feeding pipeline's thread. Every change goes through
/// the mux's lock, and is made only while this sink's registration is still
/// the one under its name: a sink whose input was replaced or removed
/// changes nothing.
pub struct StreamMuxInput {
    pub(super) pp_log: PpLog,
    pub(super) name: Arc<str>,
    pub(super) id: u64,
    pub(super) shared: Weak<MuxShared>,
}

impl Element for StreamMuxInput {
    /// Remembers the feeding pipeline, which an offline mux wakes as it takes
    /// pictures and so makes room.
    fn attach_context(&mut self, context: &Arc<Context>) {
        if let Some(shared) = self.shared.upgrade() {
            shared.with_input(self.id, |input| {
                input.upstream = Arc::downgrade(&context.state);
            });
        }
    }

    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::StreamMuxInput
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl RawSink for StreamMuxInput {
    /// Decoded pictures, wherever they live: the mux hands them on as they
    /// came, and only what runs a model on them has to read them.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::any_frame(MediaKind::VideoFrame))
    }

    /// Always, live. Offline, not while the queue is full.
    fn ready_consume(&mut self) -> bool {
        let Some(shared) = self.shared.upgrade() else {
            return true;
        };
        if shared.options.mode.is_live() {
            return true;
        }
        let capacity = shared.options.queue;
        shared
            .with_input(self.id, |input| input.pictures.len() < capacity)
            .unwrap_or(true)
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let Some(shared) = self.shared.upgrade() else {
            return Ok(()); // the mux is gone: nothing to feed
        };
        if !matches!(buf, MediaBuffer::Video(_)) {
            pp_error!(self, "unsupported buffer: expected Video");
            return Err(StreamMuxError::UnsupportedBuffer(buf.kind()).into());
        }
        let live = shared.options.mode.is_live();
        let capacity = shared.options.queue;
        let dropped = shared.with_input(self.id, |input| {
            // Live, the newest picture wins a full queue: what is behind is
            // what a batch would show late.
            let dropped = live && input.pictures.len() >= capacity;
            if dropped {
                input.pictures.pop_front();
                input.dropped += 1;
            }
            input.pictures.push_back(buf);
            dropped.then_some(input.dropped)
        });
        if let Some(Some(dropped)) = dropped {
            pp_debug!(
                self,
                "queue full: dropped its oldest picture ({dropped} so far)"
            );
        }
        shared.arrived.ring();
        Ok(())
    }

    /// Its stream's end: once its queue is empty it is not waited for, and
    /// whatever follows the mux is told nothing more of it comes.
    fn stream_event(&mut self, event: &StreamEvent) -> Result<()> {
        let (StreamEvent::Eos, Some(shared)) = (event, self.shared.upgrade()) else {
            return Ok(());
        };
        shared.with_input(self.id, |input| input.ended = true);
        shared.arrived.ring();
        Ok(())
    }

    /// `Flush` — its pipeline was sought — empties its queue and starts a
    /// new generation of the stream. `Stop` removes the input, as
    /// [`StreamMuxHandle::remove_source`](super::StreamMuxHandle::remove_source)
    /// does: a live capture is stopped rather than ended, and a stopped input
    /// must not be waited for.
    fn flow(&mut self, Flow(msg): Flow<'_>) -> Result<()> {
        let Some(shared) = self.shared.upgrade() else {
            return Ok(());
        };
        match msg {
            ControlMsg::Flush => {
                shared.with_input(self.id, |input| {
                    input.pictures.clear();
                    input.ended = false;
                    input.generation += 1;
                });
            }
            ControlMsg::Stop => shared.remove(self.id),
            _ => return Ok(()),
        }
        shared.arrived.ring();
        Ok(())
    }
}
