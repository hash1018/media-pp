//! Splitting a [`StreamMux`](super::StreamMux)'s streams out again:
//! [`StreamMuxHandle::demux`] — DeepStream's `nvstreamdemux`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use crate::pp_log::{PpLog, pp_warn};
use crate::{
    buffer::MediaBuffer,
    element::{Context, ElementType, element_pp_log},
    elements::filter::tee::{Observer, Route, TeeBuilder, TeeHandle},
    error::Result,
    graph::BranchId,
    pipeline::{ChainBuilder, DetachedBranch},
    stream::{Item, StreamEvent},
};

use super::{BatchSlot, Ended, MuxShared, StreamId, StreamMuxError, StreamMuxHandle, StreamOrigin};

/// The branches a demux has, by the stream each is for.
type Streams = Arc<Mutex<HashMap<StreamId, BranchId>>>;

impl StreamMuxHandle {
    /// Begins the element that splits this mux's streams out again, in
    /// `context` — the pipeline the mux's pictures reach it in — returning
    /// the branch to end that pipeline's chain with and the handle that
    /// gives each stream a branch of its own.
    ///
    /// It is a [`Tee`](crate::elements::Tee) that hands each picture only to
    /// the branch attached for its [`StreamOrigin`], and drops a picture
    /// whose stream has none; stream events — the segment, the end — reach
    /// every branch, a branch attached late given the segment first, and one
    /// branch failing leaves the others going. Made from the mux's handle
    /// because one thing only the mux knows has to reach it: when an input
    /// ends there is no picture left to mark as its last, so the mux records
    /// after which batch nothing more of that stream comes, and the demux
    /// ends that stream's branch — an `Eos` behind its last picture, which
    /// closes a file being written for it — once that batch has gone by.
    pub fn demux(
        &self,
        context: &Arc<Context>,
        name: impl Into<String>,
    ) -> Result<(DetachedBranch, StreamDemuxHandle)> {
        let name: String = name.into();
        let streams: Streams = Arc::default();
        let pp_log = element_pp_log(ElementType::StreamDemux, &name, Some(&context.pipeline_id));
        let mux = self.shared.clone();
        let finishing = Arc::clone(&streams);
        let builder =
            TeeBuilder::routed(name, Arc::clone(context), ElementType::StreamDemux, |tee| {
                finisher(mux, finishing, tee, pp_log)
            });
        let (branch, tee) = builder.build_dynamic()?;
        Ok((branch, StreamDemuxHandle { tee, streams }))
    }
}

/// Ends the branch of each stream the mux has recorded as over, once the
/// last batch it had a picture in has gone by.
fn finisher(mux: Weak<MuxShared>, streams: Streams, tee: TeeHandle, pp_log: PpLog) -> Observer {
    let mut waiting: Vec<Ended> = Vec::new();
    Box::new(move |item: &Item| {
        if let Some(mux) = mux.upgrade() {
            waiting.extend(mux.take_ended());
        }
        if waiting.is_empty() {
            return;
        }
        let now = match item {
            Item::Buffer(buf) => slot(buf).map(|slot| slot.batch),
            _ => None,
        };
        // The end of everything: whatever has ended has gone by.
        let over = matches!(item, Item::Event(StreamEvent::Eos));
        let mut due = Vec::new();
        waiting.retain(|ended| {
            let gone_by = over
                || match (ended.after, now) {
                    (None, _) => true,
                    (Some(after), Some(now)) => now > after,
                    (Some(_), None) => false,
                };
            if gone_by {
                due.push(ended.id);
            }
            !gone_by
        });
        for id in due {
            // Taken out of the map before the branch is ended: ending it
            // pushes an `Eos` through it, which is not done under a lock.
            let branch = streams.lock().unwrap().remove(&id);
            if let Some(branch) = branch
                && let Err(error) = tee.finish_branch(branch)
            {
                pp_warn!(pp_log: &pp_log, "could not end {id}'s branch: {error}");
            }
        }
    })
}

fn slot(buf: &MediaBuffer) -> Option<BatchSlot> {
    buf.metadata()?.get::<BatchSlot>().copied()
}

fn stream(buf: &MediaBuffer) -> Option<StreamId> {
    buf.metadata()?
        .get::<StreamOrigin>()
        .map(|origin| origin.id)
}

/// Gives each of a [`StreamMux`](super::StreamMux)'s streams a branch of its
/// own after a demux — see [`StreamMuxHandle::demux`]. Cheap to clone, and
/// keeps only a weak reference to the demux, as a
/// [`TeeHandle`](crate::elements::TeeHandle) does: a call once it is gone
/// fails rather than keeping anything alive.
#[derive(Clone)]
pub struct StreamDemuxHandle {
    tee: TeeHandle,
    streams: Streams,
}

impl StreamDemuxHandle {
    /// A chain to build a stream's branch from, in the demux's pipeline.
    pub fn branch(&self) -> Result<ChainBuilder> {
        self.tee.branch()
    }

    /// Attaches `branch` for `stream`: from now on it is handed that
    /// stream's pictures, and ended once the stream is. A stream has one
    /// branch at a time; another for it is refused with
    /// [`StreamMuxError::AlreadyAttached`] and changes nothing.
    pub fn attach(&self, stream: StreamId, branch: DetachedBranch) -> Result<BranchId> {
        let mut streams = self.streams.lock().unwrap();
        if streams.contains_key(&stream) {
            return Err(StreamMuxError::AlreadyAttached(stream).into());
        }
        let route: Route = Arc::new(move |buf| self::stream(buf) == Some(stream));
        let id = self.tee.attach_routed(branch, Some(route))?;
        streams.insert(stream, id);
        Ok(id)
    }

    /// Ends `stream`'s branch as a recording ends — an `Eos` behind what it
    /// was already handed — and detaches it. Nothing if it has none.
    pub fn finish(&self, stream: StreamId) -> Result<()> {
        let branch = self.streams.lock().unwrap().remove(&stream);
        match branch {
            Some(branch) => self.tee.finish_branch(branch),
            None => Ok(()),
        }
    }

    /// Detaches `stream`'s branch, abandoning what it held. Nothing if it
    /// has none.
    pub fn detach(&self, stream: StreamId) -> Result<()> {
        let branch = self.streams.lock().unwrap().remove(&stream);
        match branch {
            Some(branch) => self.tee.detach(branch),
            None => Ok(()),
        }
    }

    /// The streams with a branch.
    pub fn streams(&self) -> Vec<StreamId> {
        let mut streams: Vec<_> = self.streams.lock().unwrap().keys().copied().collect();
        streams.sort();
        streams
    }
}
