//! The audio a transcriber holds until the speech in it is settled.
//!
//! Kept apart from the model so that which buffer carries which line, and
//! when each is let go of, is tested without one.

use std::collections::VecDeque;

use crate::buffer::MediaBuffer;
use crate::elements::speech::meta::{Transcript, Utterance};

/// Audio waiting for its words, and words waiting for their audio.
///
/// A buffer goes on once everything up to its end has been transcribed —
/// [`Held::release`] with where that is — carrying the lines that begin in
/// it. The lines come in order, each once, so each is put on the first
/// buffer going on that ends after it begins.
#[derive(Default)]
pub(super) struct Held {
    /// Each buffer with where it starts and ends, in milliseconds.
    buffers: VecDeque<(MediaBuffer, i64, i64)>,
    /// Lines settled and not yet put on a buffer.
    lines: VecDeque<Utterance>,
}

impl Held {
    /// Holds `buf`, which runs from `start_ms` to `end_ms`.
    pub(super) fn hold(&mut self, buf: MediaBuffer, start_ms: i64, end_ms: i64) {
        self.buffers.push_back((buf, start_ms, end_ms));
    }

    /// Lines that have been settled, in the order they were said.
    pub(super) fn heard(&mut self, lines: impl IntoIterator<Item = Utterance>) {
        self.lines.extend(lines);
    }

    /// Every buffer that ends by `settled_ms`, in order, with the lines
    /// beginning in it.
    pub(super) fn release(&mut self, settled_ms: i64) -> Vec<MediaBuffer> {
        let mut going = Vec::new();
        while self
            .buffers
            .front()
            .is_some_and(|&(_, _, end)| end <= settled_ms)
        {
            let (buf, _, end) = self.buffers.pop_front().expect("one is there");
            going.push(self.carrying(buf, |line| line.start_ms < end));
        }
        going
    }

    /// Every buffer, at the end of the stream: nothing more is coming, so
    /// whatever lines remain go on the last of them.
    ///
    /// Answers the lines left over when no buffer was held to carry them —
    /// which there cannot be, since every line was heard in held audio, but
    /// is said rather than silently dropped.
    pub(super) fn release_all(&mut self) -> (Vec<MediaBuffer>, usize) {
        let mut going = Vec::new();
        while let Some((buf, _, end)) = self.buffers.pop_front() {
            let last = self.buffers.is_empty();
            going.push(self.carrying(buf, |line| last || line.start_ms < end));
        }
        (going, std::mem::take(&mut self.lines).len())
    }

    /// Lets go of everything, for a seek or a stop.
    pub(super) fn clear(&mut self) {
        self.buffers.clear();
        self.lines.clear();
    }

    /// `buf`, with the lines at the front that `belongs` takes put on it
    /// beside whatever it already carried.
    fn carrying(&mut self, buf: MediaBuffer, belongs: impl Fn(&Utterance) -> bool) -> MediaBuffer {
        let mut utterances = Vec::new();
        while self.lines.front().is_some_and(&belongs) {
            utterances.push(self.lines.pop_front().expect("one is there"));
        }
        if utterances.is_empty() {
            return buf;
        }
        let metadata = buf
            .metadata()
            .cloned()
            .unwrap_or_default()
            .with(Transcript { utterances });
        buf.with_metadata(metadata)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ffmpeg_next as ffmpeg;

    use super::*;
    use crate::buffer::Metadata;

    /// A buffer of sound stamped `pts`, so each can be told apart.
    fn sound(pts: i64) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Audio::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            16,
            ffmpeg::channel_layout::ChannelLayout::MONO,
        );
        frame.set_pts(Some(pts));
        MediaBuffer::Audio(Arc::new(frame).into())
    }

    fn line(start_ms: i64, end_ms: i64, text: &str) -> Utterance {
        Utterance {
            start_ms,
            end_ms,
            text: text.to_owned(),
            words: Vec::new(),
        }
    }

    /// The lines a buffer carries, by their text.
    fn said(buf: &MediaBuffer) -> Vec<String> {
        buf.metadata()
            .and_then(Metadata::get::<Transcript>)
            .map(|transcript| {
                transcript
                    .utterances
                    .iter()
                    .map(|u| u.text.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Nothing goes on until the speech in it is settled, and then it goes
    /// in order.
    #[test]
    fn a_buffer_waits_for_its_end_to_be_settled() {
        let mut held = Held::default();
        held.hold(sound(0), 0, 100);
        held.hold(sound(1), 100, 200);
        assert!(held.release(150).len() == 1, "only the first is settled");
        let rest = held.release(200);
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].pts(), Some(1));
    }

    /// Each line rides on the buffer it begins in, once.
    #[test]
    fn a_line_rides_on_the_buffer_it_begins_in() {
        let mut held = Held::default();
        held.hold(sound(0), 0, 100);
        held.hold(sound(1), 100, 200);
        held.hold(sound(2), 200, 300);
        held.heard([line(120, 260, "둘째"), line(250, 290, "셋째")]);
        let going = held.release(300);
        assert_eq!(said(&going[0]), Vec::<String>::new());
        assert_eq!(said(&going[1]), ["둘째"]);
        assert_eq!(
            said(&going[2]),
            ["셋째"],
            "a line running past its buffer stays on the one it began in"
        );
    }

    /// At the end every buffer goes, and whatever is left rides on the last.
    #[test]
    fn the_end_lets_everything_go() {
        let mut held = Held::default();
        held.hold(sound(0), 0, 100);
        held.heard([line(150, 180, "꼬리")]);
        let (going, stray) = held.release_all();
        assert_eq!(said(&going[0]), ["꼬리"]);
        assert_eq!(stray, 0);
    }

    /// What a buffer already carried is kept beside the transcript.
    #[test]
    fn what_a_buffer_carried_is_kept() {
        #[derive(Debug, PartialEq)]
        struct Mark(u8);
        let mut held = Held::default();
        let marked = sound(0).with_metadata(Metadata::default().with(Mark(7)));
        held.hold(marked, 0, 100);
        held.heard([line(10, 50, "말")]);
        let going = held.release(100);
        let metadata = going[0].metadata().expect("carries both");
        assert_eq!(metadata.get::<Mark>(), Some(&Mark(7)));
        assert_eq!(said(&going[0]), ["말"]);
    }

    /// A seek lets go of what was held, without handing it on.
    #[test]
    fn clearing_hands_nothing_on() {
        let mut held = Held::default();
        held.hold(sound(0), 0, 100);
        held.heard([line(10, 50, "버려짐")]);
        held.clear();
        assert!(held.release(1_000).is_empty());
        assert_eq!(held.release_all().0.len(), 0);
    }
}
