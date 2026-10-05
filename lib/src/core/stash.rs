//! What a stage keeps back from what follows it while a preroll holds the
//! graph — see [`OutputStash`].

use std::{collections::VecDeque, sync::Arc};

use crate::{buffer::MediaBuffer, error::Result, pad::SrcPad, playback_state::PlaybackState};

/// What one stage made that its pad could not take while a preroll held
/// the graph, kept in order to go first once it can.
///
/// A terminal takes the sample a preroll asks of it and from then answers
/// not ready, and wherever a thread asks before it hands something on —
/// a queue's worker, a source — that holds everything behind it. Where
/// nothing asks between one output and the next it does not: a decoder
/// handed one packet can answer it with several frames, a tempo stretch one
/// frame with several pieces, all pushed from one call. Pushed into a
/// terminal that had its sample they were taken all the same, so a
/// preroll's terminal was handed more than it asked for. What such a stage
/// makes once its pad is not ready waits here, and the stage answers not
/// ready while it does, so its own input is held in turn.
///
/// Only while a preroll runs, or the pause that followed one keeps what it
/// held: otherwise a pad that cannot take a buffer yet is waited on inside
/// the push, as it always was, and nothing is kept.
#[derive(Default)]
pub(crate) struct OutputStash {
    held: VecDeque<MediaBuffer>,
    /// The pipeline's playback state, which says whether a preroll holds
    /// the graph — `None` for a stage wired into no pipeline, which never
    /// keeps anything.
    state: Option<Arc<PlaybackState>>,
}

impl OutputStash {
    /// Reads from `state` whether a preroll holds the graph.
    pub(crate) fn attach(&mut self, state: &Arc<PlaybackState>) {
        self.state = Some(Arc::clone(state));
    }

    /// Whether a preroll, or the pause that followed one, holds the graph.
    fn holding(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|state| state.preroll().is_some())
    }

    /// Hands `buf` on through `pad` after whatever is kept — or keeps it
    /// too, where a preroll holds the graph and `pad` cannot take it now.
    pub(crate) fn push(&mut self, pad: &mut SrcPad, buf: MediaBuffer) -> Result<()> {
        let released = self.release(pad);
        if !self.held.is_empty() || (self.holding() && !pad.ready_consume()) {
            self.held.push_back(buf);
            return released;
        }
        released.and(pad.push(buf))
    }

    /// Hands on what is kept, oldest first: as far as `pad` takes it while a
    /// preroll holds the graph, and all of it once none does. Every one is
    /// pushed however the one before fared, and the first failure answered.
    pub(crate) fn release(&mut self, pad: &mut SrcPad) -> Result<()> {
        let mut first = Ok(());
        while !self.held.is_empty() {
            if self.holding() && !pad.ready_consume() {
                break;
            }
            let buf = self.held.pop_front().expect("not empty");
            let pushed = pad.push(buf);
            if first.is_ok() {
                first = pushed;
            }
        }
        first
    }

    /// Hands on everything kept, whether `pad` can take it or not — ahead of
    /// the end of the stream, after which nothing would take it, and ahead
    /// of an event, which must not overtake it.
    pub(crate) fn release_all(&mut self, pad: &mut SrcPad) -> Result<()> {
        let mut first = Ok(());
        for buf in self.held.drain(..) {
            let pushed = pad.push(buf);
            if first.is_ok() {
                first = pushed;
            }
        }
        first
    }

    /// Whether the stage can take its next input: nothing is kept, or what
    /// is can go on — no preroll holds the graph, or `pad` takes it now.
    pub(crate) fn ready(&mut self, pad: &mut SrcPad) -> bool {
        self.held.is_empty() || !self.holding() || pad.ready_consume()
    }

    /// Lets go of what is kept — a seek's `Flush`, a `Stop`.
    pub(crate) fn clear(&mut self) {
        self.held.clear();
    }

    /// How many buffers are kept.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.held.len()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    };

    use ffmpeg_next as ffmpeg;

    use super::*;
    use crate::{
        control::{ControlMsg, PrerollContext},
        element::{Element, ElementType, RawSink, element_pp_log},
        pp_log::PpLog,
    };

    fn packet(tag: u8) -> MediaBuffer {
        MediaBuffer::Packet(Arc::new(ffmpeg::Packet::copy(&[tag])).into())
    }

    /// Takes what it is handed while its gate is open, writing down each
    /// packet's tag.
    struct Gated {
        pp_log: PpLog,
        open: Arc<AtomicBool>,
        taken: Arc<Mutex<Vec<u8>>>,
    }

    impl Element for Gated {
        fn name(&self) -> Arc<str> {
            "gated".into()
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

    impl RawSink for Gated {
        fn ready_consume(&mut self) -> bool {
            self.open.load(Ordering::SeqCst)
        }
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Packet(packet) = &buf {
                self.taken
                    .lock()
                    .unwrap()
                    .push(packet.data().expect("a payload")[0]);
            }
            Ok(())
        }
    }

    fn gated_pad() -> (SrcPad, Arc<AtomicBool>, Arc<Mutex<Vec<u8>>>) {
        let open = Arc::new(AtomicBool::new(true));
        let taken = Arc::new(Mutex::new(Vec::new()));
        let mut pad = SrcPad::new("src");
        pad.link(Box::new(Gated {
            pp_log: element_pp_log(ElementType::Other, "gated", None),
            open: Arc::clone(&open),
            taken: Arc::clone(&taken),
        }));
        (pad, open, taken)
    }

    /// While a preroll holds the graph, what the pad cannot take is kept
    /// and the stage answers not ready; once it can, what was kept goes
    /// first, in order.
    #[test]
    fn in_a_preroll_what_the_pad_cannot_take_waits_its_turn() {
        let state = PlaybackState::new();
        let mut stash = OutputStash::default();
        stash.attach(&state);
        let (mut pad, open, taken) = gated_pad();
        state.observe(&ControlMsg::Preroll(Arc::new(PrerollContext::new([]))));

        stash.push(&mut pad, packet(1)).unwrap();
        open.store(false, Ordering::SeqCst);
        stash.push(&mut pad, packet(2)).unwrap();
        stash.push(&mut pad, packet(3)).unwrap();
        assert_eq!(*taken.lock().unwrap(), [1], "the sample, and nothing more");
        assert_eq!(stash.len(), 2);
        assert!(!stash.ready(&mut pad), "its input is held in turn");

        open.store(true, Ordering::SeqCst);
        assert!(stash.ready(&mut pad));
        stash.push(&mut pad, packet(4)).unwrap();
        assert_eq!(*taken.lock().unwrap(), [1, 2, 3, 4], "in order");
        assert_eq!(stash.len(), 0);
    }

    /// Kept through the pause a preroll ends in; once playback goes on it
    /// all goes, the pad waited on inside the push as ever.
    #[test]
    fn what_a_preroll_kept_goes_once_playback_goes_on() {
        let state = PlaybackState::new();
        let mut stash = OutputStash::default();
        stash.attach(&state);
        let (mut pad, open, taken) = gated_pad();
        state.observe(&ControlMsg::Preroll(Arc::new(PrerollContext::new([]))));
        open.store(false, Ordering::SeqCst);
        stash.push(&mut pad, packet(1)).unwrap();

        state.observe(&ControlMsg::Pause);
        stash.release(&mut pad).unwrap();
        assert!(taken.lock().unwrap().is_empty(), "the pause keeps it");

        state.observe(&ControlMsg::Resume);
        assert!(stash.ready(&mut pad), "nothing holds it back now");
        stash.release(&mut pad).unwrap();
        assert_eq!(*taken.lock().unwrap(), [1]);
    }

    /// Outside a preroll nothing is kept: a buffer goes into its pad as it
    /// always did, whether the pad is ready or not.
    #[test]
    fn outside_a_preroll_nothing_is_kept() {
        let state = PlaybackState::new();
        let mut stash = OutputStash::default();
        stash.attach(&state);
        let (mut pad, open, taken) = gated_pad();
        open.store(false, Ordering::SeqCst);
        stash.push(&mut pad, packet(1)).unwrap();
        assert_eq!(*taken.lock().unwrap(), [1]);
        assert_eq!(stash.len(), 0);
    }

    /// Ahead of the end of the stream or an event, everything kept goes
    /// whatever the pad says; a `Flush` lets go of it instead.
    #[test]
    fn everything_goes_ahead_of_the_end_and_nothing_after_a_flush() {
        let state = PlaybackState::new();
        let mut stash = OutputStash::default();
        stash.attach(&state);
        let (mut pad, open, taken) = gated_pad();
        state.observe(&ControlMsg::Preroll(Arc::new(PrerollContext::new([]))));
        open.store(false, Ordering::SeqCst);
        stash.push(&mut pad, packet(1)).unwrap();
        stash.release_all(&mut pad).unwrap();
        assert_eq!(*taken.lock().unwrap(), [1]);

        stash.push(&mut pad, packet(2)).unwrap();
        stash.clear();
        stash.release_all(&mut pad).unwrap();
        assert_eq!(*taken.lock().unwrap(), [1], "the flush let it go");
    }
}
