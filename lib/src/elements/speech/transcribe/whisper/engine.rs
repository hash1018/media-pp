//! The whisper.cpp end of every transcriber: what to run, and what of the
//! answer to keep.
//!
//! Kept apart from the elements so that the awkward part — the arithmetic
//! of overlapping windows — is tested without a model, and so that every
//! backend runs exactly the same loop.

use super::{ChunkPolicy, SAMPLE_RATE, WhisperTranscriberError};
use crate::elements::speech::meta::{Utterance, Word};

/// One token of what the model said, and when, in stream milliseconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Heard {
    /// Its text, as bytes: a token need not end on a character boundary.
    pub(super) bytes: Vec<u8>,
    pub(super) start_ms: i64,
    pub(super) end_ms: i64,
}

/// Whether a token begins partway through a character — the rest of one
/// the token before it started.
fn starts_mid_character(token: &Heard) -> bool {
    token
        .bytes
        .first()
        .is_some_and(|byte| byte & 0b1100_0000 == 0b1000_0000)
}

/// What the [`Chunker`] wants run, and which of the answer to keep.
#[derive(Debug, PartialEq)]
pub(super) struct Inference {
    /// Where `audio` starts, in milliseconds from the first sample ever
    /// given to the chunker. Whisper reports its segments relative to the
    /// audio it was handed, so this is what turns those into stream time.
    pub(super) offset_ms: i64,
    /// Everything to feed the model: the chunk to transcribe, with the one
    /// before it in front as context.
    pub(super) audio: Vec<f32>,
    /// What starts before this was reported by an earlier round.
    accept_from_ms: i64,
    /// What starts at or after this is too near the edge to trust; the next
    /// round will cover it.
    accept_until_ms: i64,
    /// Where `audio` ends. Nothing the model says can have been said later.
    audio_end_ms: i64,
}

/// Decides what to transcribe and what of the result to believe.
///
/// Separated from the model so that the awkward part — the arithmetic of
/// overlapping windows — can be tested without one. Every decision this
/// makes is a function of sample counts, and none of it needs whisper.cpp
/// to be present, let alone a multi-gigabyte model file.
///
/// # The overlap
///
/// Each inference is given the previous chunk as well as the current one.
/// A word spanning the boundary is otherwise seen half by one round and
/// half by the next, and comes out mangled or twice. Feeding the context
/// costs a second pass over audio already transcribed — roughly doubling
/// the work — and buys words that survive the seam.
pub(super) struct Chunker {
    policy: ChunkPolicy,
    /// New audio, not yet transcribed.
    pending: Vec<f32>,
    /// Where `pending` starts on the timeline of the audio itself.
    pending_start: i64,
    /// The chunk before `pending`, kept to feed as context. Empty when what
    /// came before did not join onto this.
    context: Vec<f32>,
    /// Where `context` starts on the same timeline.
    context_start: i64,
    /// Everything before this has been reported.
    accepted_until: i64,
    /// Where the next sample is expected to land, so one arriving anywhere
    /// else is recognised as a gap. `None` before the first.
    next_pts: Option<i64>,
    /// Where the last reported segment ended, in milliseconds — see
    /// [`Chunker::place`].
    reported_until_ms: i64,
}

impl Chunker {
    pub(super) fn new(policy: ChunkPolicy) -> Self {
        Self {
            policy,
            pending: Vec::new(),
            pending_start: 0,
            context: Vec::new(),
            context_start: 0,
            accepted_until: 0,
            next_pts: None,
            reported_until_ms: i64::MIN,
        }
    }

    fn samples_to_ms(samples: i64) -> i64 {
        (samples * 1000) / SAMPLE_RATE as i64
    }

    /// Takes more audio, and answers the inference it now wants run.
    ///
    /// `pts` places the samples on the timeline of the audio itself, in that
    /// timeline's own units — which for the
    /// [`WHISPER_SAMPLE_RATE`](SAMPLE_RATE) mono this accepts is a sample
    /// index. `None` means "where the last frame ended", which is what a source
    /// that stamps nothing amounts to.
    ///
    /// Answers `None` while there is not yet a full chunk. One call yields
    /// at most one inference, so a caller handing over more than a chunk at
    /// a time keeps calling [`Chunker::take_chunk`] until it answers `None`.
    ///
    /// # A gap throws away what was pending
    ///
    /// Audio that does not continue where the last left off — a queue that
    /// dropped buffers because this is behind, a source that jumped —
    /// leaves what is buffered unjoinable to what follows. Transcribing
    /// across the seam would put one sentence over two moments that are not
    /// adjacent, so the buffer goes and the timeline picks up where the new
    /// audio actually is.
    ///
    /// The alternative, counting samples and letting positions slide, is
    /// worse in the way that matters: every line after the loss would be
    /// early by what was lost, and the error would accumulate instead of
    /// staying where it happened.
    pub(super) fn push(&mut self, pts: Option<i64>, samples: &[f32]) -> Option<Inference> {
        let pts = pts.or(self.next_pts).unwrap_or(0);
        match self.next_pts {
            None => {
                self.pending_start = pts;
                self.accepted_until = pts;
            }
            Some(expected) if pts != expected => {
                self.pending.clear();
                self.context.clear();
                self.pending_start = pts;
                self.accepted_until = self.accepted_until.max(pts);
            }
            Some(_) => {}
        }
        self.pending.extend_from_slice(samples);
        self.next_pts = Some(pts + samples.len() as i64);
        self.take_chunk()
    }

    pub(super) fn take_chunk(&mut self) -> Option<Inference> {
        let chunk = self.policy.chunk_samples();
        if self.pending.len() < chunk {
            return None;
        }
        let rest = self.pending.split_off(chunk);
        let current = std::mem::replace(&mut self.pending, rest);
        let current_start = self.pending_start;
        self.pending_start = current_start + chunk as i64;

        // The edge is where the audio runs out, and the last stretch before
        // it is what the model is least sure of.
        let current_end = current_start + current.len() as i64;
        let accept_until = current_end - self.policy.live_edge_samples() as i64;

        let inference = self.inference(current_start, &current, accept_until);

        self.context_start = current_start;
        self.context = current;
        self.accepted_until = accept_until.max(self.accepted_until);
        Some(inference)
    }

    /// The last inference, once no more audio is coming.
    ///
    /// Whatever is left is transcribed however short it is, and the live
    /// edge is not held back: there is no later round to pick it up, and
    /// nothing more will arrive to change what the model hears. `None` when
    /// everything has already been reported.
    pub(super) fn flush(&mut self) -> Option<Inference> {
        let current = std::mem::take(&mut self.pending);
        let current_start = self.pending_start;
        let current_end = current_start + current.len() as i64;
        if current_end <= self.accepted_until {
            return None;
        }
        let inference = self.inference(current_start, &current, current_end);
        self.pending_start = current_end;
        self.context_start = current_start;
        self.context = current;
        self.accepted_until = current_end;
        Some(inference)
    }

    /// Builds the request: context then the chunk, and the window of the
    /// answer that is neither already said nor too new to trust.
    fn inference(&self, current_start: i64, current: &[f32], accept_until: i64) -> Inference {
        // There is no context before the first chunk, nor after a gap, and
        // then what the model is handed begins where this chunk does.
        let audio_start = if self.context.is_empty() {
            current_start
        } else {
            self.context_start
        };
        let mut audio = Vec::with_capacity(self.context.len() + current.len());
        audio.extend_from_slice(&self.context);
        audio.extend_from_slice(current);
        let audio_end = audio_start + audio.len() as i64;
        Inference {
            offset_ms: Self::samples_to_ms(audio_start),
            audio,
            accept_from_ms: Self::samples_to_ms(self.accepted_until),
            accept_until_ms: Self::samples_to_ms(accept_until),
            audio_end_ms: Self::samples_to_ms(audio_end),
        }
    }

    /// The part of one segment that belongs to this round, as the line to
    /// report — or `None` when none of it does.
    ///
    /// Cut by token, because cutting by segment lost speech and repeated it.
    /// A segment is whatever stretch the model chose to call one, and it
    /// often opens in the context — audio already reported — and runs on
    /// into the new window. Kept by where it started, such a segment was
    /// dropped every round, and a minute of Korean lost its last fifteen
    /// seconds that way; kept whole, it would say again what the last round
    /// said. So each token is judged by where *it* starts: the ones in the
    /// window are this round's, the ones before it were reported, and the
    /// ones at the edge are left for the next round, which hears them again
    /// in its context.
    ///
    /// The window starts at whichever is later of where this round's window
    /// does and where the last line ended, so what reaches the caller is in
    /// order and never overlaps — an MP4's text track cannot hold overlap,
    /// and its muxer wrote negative durations for it. A line ends no later
    /// than the audio it was heard in.
    ///
    /// A character can be split between tokens — a Korean syllable is three
    /// bytes, and a token boundary can fall inside one — so a cut that lands
    /// mid-character is moved to include the whole of it.
    pub(super) fn place(&mut self, inference: &Inference, tokens: &[Heard]) -> Option<Utterance> {
        let from = inference.accept_from_ms.max(self.reported_until_ms);
        let until = inference.accept_until_ms;
        let inside = |token: &Heard| token.start_ms >= from && token.start_ms < until;
        let mut first = tokens.iter().position(inside)?;
        let mut last = tokens.iter().rposition(inside)?;
        while first > 0 && starts_mid_character(&tokens[first]) {
            first -= 1;
        }
        while last + 1 < tokens.len() && starts_mid_character(&tokens[last + 1]) {
            last += 1;
        }

        let kept = &tokens[first..=last];
        let bytes: Vec<u8> = kept
            .iter()
            .flat_map(|token| token.bytes.iter().copied())
            .collect();
        let text = String::from_utf8_lossy(&bytes).trim().to_owned();
        if text.is_empty() {
            return None;
        }
        let start = kept[0].start_ms.max(from);
        let end = kept[kept.len() - 1]
            .end_ms
            .min(inference.audio_end_ms)
            .max(start);
        self.reported_until_ms = end;
        Some(Utterance {
            start_ms: start,
            end_ms: end,
            text,
            words: words(kept, start, end),
        })
    }

    /// Where everything before has been transcribed and reported, in
    /// milliseconds: what audio up to here holds will not change.
    pub(super) fn settled_until_ms(&self) -> i64 {
        Self::samples_to_ms(self.accepted_until)
    }

    /// Forgets everything, for a seek or a stop.
    ///
    /// The timeline is not reset: positions stay where they were, because
    /// what the caller does with a seek is its own business and inventing a
    /// new zero here would only disagree with it.
    pub(super) fn reset(&mut self) {
        self.pending.clear();
        self.context.clear();
    }
}

/// A line's tokens as words: a word begins at each token that starts with a
/// space, which is how whisper's tokens mark one — and runs on through the
/// tokens after it that do not.
///
/// The times are kept inside the line's own, `start` to `end`, since those
/// are what was cut to the window and the audio.
fn words(tokens: &[Heard], start: i64, end: i64) -> Vec<Word> {
    let mut groups: Vec<Vec<&Heard>> = Vec::new();
    for token in tokens {
        match groups.last_mut() {
            Some(group) if token.bytes.first() != Some(&b' ') => group.push(token),
            _ => groups.push(vec![token]),
        }
    }
    groups
        .into_iter()
        .filter_map(|group| {
            let bytes: Vec<u8> = group
                .iter()
                .flat_map(|token| token.bytes.iter().copied())
                .collect();
            let text = String::from_utf8_lossy(&bytes).trim().to_owned();
            if text.is_empty() {
                return None;
            }
            let word_start = group[0].start_ms.clamp(start, end);
            let word_end = group[group.len() - 1].end_ms.clamp(word_start, end);
            Some(Word {
                start_ms: word_start,
                end_ms: word_end,
                text,
            })
        })
        .collect()
}

/// The dimensions a whisper.cpp model file states at its start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ModelShape {
    n_vocab: i32,
    n_audio_state: i32,
    n_audio_layer: i32,
    n_text_layer: i32,
    n_mels: i32,
}

/// Reads a model file's header: the `ggml` magic, then eleven
/// little-endian `i32` hyperparameters, of which these are the ones that
/// tell one model from another. `None` for anything else.
pub(super) fn model_shape(path: &std::path::Path) -> Option<ModelShape> {
    use std::io::Read;

    let mut header = [0u8; 4 * 12];
    std::fs::File::open(path)
        .ok()?
        .read_exact(&mut header)
        .ok()?;
    let word = |index: usize| {
        let bytes = header[index * 4..index * 4 + 4]
            .try_into()
            .expect("four bytes");
        i32::from_le_bytes(bytes)
    };
    if word(0) as u32 != 0x6767_6d6c {
        return None;
    }
    // n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
    // n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype.
    Some(ModelShape {
        n_vocab: word(1),
        n_audio_state: word(3),
        n_audio_layer: word(5),
        n_text_layer: word(9),
        n_mels: word(10),
    })
}

/// Which of whisper.cpp's alignment-head presets fits a model of this
/// shape, or `None` for one it has no preset for.
///
/// Told apart by width and depth, and English-only from multilingual by the
/// vocabulary, which is one token shorter. `large-v3-turbo` is `large-v3`'s
/// encoder with four decoder layers instead of thirty-two. `large-v1` and
/// `large-v2` share every dimension, so both are read as `large-v2` — the
/// one still in use.
pub(super) fn alignment_heads(shape: ModelShape) -> Option<whisper_rs::DtwModelPreset> {
    use whisper_rs::DtwModelPreset as Preset;

    let english = shape.n_vocab == 51_864;
    Some(
        match (shape.n_audio_state, shape.n_audio_layer, shape.n_text_layer) {
            (384, 4, 4) if english => Preset::TinyEn,
            (384, 4, 4) => Preset::Tiny,
            (512, 6, 6) if english => Preset::BaseEn,
            (512, 6, 6) => Preset::Base,
            (768, 12, 12) if english => Preset::SmallEn,
            (768, 12, 12) => Preset::Small,
            (1024, 24, 24) if english => Preset::MediumEn,
            (1024, 24, 24) => Preset::Medium,
            (1280, 32, 4) if shape.n_mels == 128 => Preset::LargeV3Turbo,
            (1280, 32, 32) if shape.n_mels == 128 => Preset::LargeV3,
            (1280, 32, 32) => Preset::LargeV2,
            _ => return None,
        },
    )
}

/// whisper.cpp's own code for a language given as a code or an English name,
/// or why it has none.
///
/// Checked here because whisper.cpp would take an unknown one and fail every
/// chunk with it, and whisper-rs panics outright on an interior NUL.
pub(super) fn language_code(
    language: &str,
) -> std::result::Result<&'static str, WhisperTranscriberError> {
    let unknown = || WhisperTranscriberError::UnknownLanguage(language.to_owned());
    if language.contains('\0') {
        return Err(unknown());
    }
    let id = whisper_rs::get_lang_id(language).ok_or_else(unknown)?;
    whisper_rs::get_lang_str(id).ok_or_else(unknown)
}

#[cfg(test)]
mod tests {
    use super::super::TokenTiming;
    use super::*;

    const POLICY: ChunkPolicy = ChunkPolicy {
        chunk_ms: 4_000,
        live_edge_ms: 1_000,
        token_timing: TokenTiming::Estimated,
    };

    fn samples(ms: usize) -> Vec<f32> {
        vec![0.0; ms * SAMPLE_RATE as usize / 1000]
    }

    fn at(ms: usize) -> i64 {
        (ms * SAMPLE_RATE as usize / 1000) as i64
    }

    /// Pushes `ms` of audio continuing wherever the last push ended, which
    /// is the ordinary case and what most of these tests want.
    fn push_on(chunker: &mut Chunker, ms: usize) -> Option<Inference> {
        chunker.push(None, &samples(ms))
    }

    /// Nothing is transcribed until a whole chunk has arrived, because a
    /// shorter one is padded to thirty seconds and the model was not
    /// trained to hear the padding as silence at the end of a sentence.
    #[test]
    fn a_partial_chunk_is_not_transcribed() {
        let mut chunker = Chunker::new(POLICY);
        assert_eq!(push_on(&mut chunker, 3_999), None);
        assert!(push_on(&mut chunker, 1).is_some(), "and now it is full");
    }

    /// The window that is believed starts where the last one stopped and
    /// ends short of the edge, so every millisecond is reported once.
    #[test]
    fn the_believed_window_stops_short_of_the_edge_and_resumes_there() {
        let mut chunker = Chunker::new(POLICY);

        let first = push_on(&mut chunker, 4_000).expect("a full chunk");
        assert_eq!(first.offset_ms, 0);
        assert_eq!(first.accept_from_ms, 0);
        assert_eq!(
            first.accept_until_ms, 3_000,
            "the last second is not trusted"
        );

        let second = push_on(&mut chunker, 4_000).expect("another");
        assert_eq!(
            second.accept_from_ms, 3_000,
            "the second held back before is picked up here"
        );
        assert_eq!(second.accept_until_ms, 7_000);
    }

    /// A word across the seam is heard whole because the round that reports
    /// it was given the audio before it too.
    #[test]
    fn each_inference_carries_the_previous_chunk_as_context() {
        let mut chunker = Chunker::new(POLICY);

        let first = push_on(&mut chunker, 4_000).expect("a full chunk");
        assert_eq!(
            first.audio.len(),
            samples(4_000).len(),
            "nothing precedes the first"
        );

        let second = push_on(&mut chunker, 4_000).expect("another");
        assert_eq!(
            second.audio.len(),
            samples(8_000).len(),
            "four seconds of context and four of new audio"
        );
        assert_eq!(second.offset_ms, 0, "and it starts where the context does");
    }

    /// A caller handing over more than a chunk at once must not have the
    /// surplus swallowed.
    #[test]
    fn a_long_push_yields_one_inference_per_chunk() {
        let mut chunker = Chunker::new(POLICY);
        assert!(push_on(&mut chunker, 12_000).is_some());
        assert!(chunker.take_chunk().is_some());
        assert!(chunker.take_chunk().is_some());
        assert_eq!(chunker.take_chunk(), None, "three chunks and no more");
    }

    /// At the end there is no later round, so the tail is transcribed
    /// however short it is and the live edge is not held back.
    #[test]
    fn the_tail_is_transcribed_at_eos_without_holding_the_edge_back() {
        let mut chunker = Chunker::new(POLICY);
        push_on(&mut chunker, 4_000).expect("a full chunk");
        push_on(&mut chunker, 500);

        let last = chunker.flush().expect("the tail is worth hearing");
        assert_eq!(
            last.accept_from_ms, 3_000,
            "starting where the chunk stopped"
        );
        assert_eq!(last.accept_until_ms, 4_500, "and running to the very end");
        assert_eq!(
            last.audio.len(),
            samples(4_500).len(),
            "the whole chunk before it, as context, then the tail"
        );
    }

    /// Flushing twice must not transcribe the tail again and report every
    /// line of it a second time.
    #[test]
    fn a_second_flush_has_nothing_left_to_say() {
        let mut chunker = Chunker::new(POLICY);
        push_on(&mut chunker, 4_000);
        push_on(&mut chunker, 500);
        assert!(chunker.flush().is_some());
        assert_eq!(chunker.flush(), None);
    }

    /// A stream that ends on an exact chunk boundary has already had every
    /// millisecond transcribed except the second held at the edge.
    #[test]
    fn a_stream_ending_on_a_boundary_still_reports_its_held_back_edge() {
        let mut chunker = Chunker::new(POLICY);
        push_on(&mut chunker, 4_000).expect("a full chunk");

        let last = chunker.flush().expect("the held-back second is still owed");
        assert_eq!(last.accept_from_ms, 3_000);
        assert_eq!(last.accept_until_ms, 4_000);
    }
    /// The whole reason positions come from the timestamps: audio that
    /// arrives late in its own timeline, because a queue dropped what was
    /// between, must not drag everything after it earlier.
    #[test]
    fn a_gap_leaves_a_hole_rather_than_shifting_what_follows() {
        let mut chunker = Chunker::new(POLICY);
        push_on(&mut chunker, 4_000).expect("a full chunk");

        // Ten seconds went missing. What arrives is stamped where it really
        // is, not where counting would have put it.
        let after = chunker
            .push(Some(at(14_000)), &samples(4_000))
            .expect("a full chunk again");
        assert_eq!(
            after.offset_ms, 14_000,
            "the audio is placed where its timestamps say"
        );
        assert_eq!(
            after.accept_from_ms, 14_000,
            "and nothing before it is claimed"
        );
        assert_eq!(after.accept_until_ms, 17_000);
    }

    /// Across a gap the audio before it is not context for the audio after:
    /// they are not adjacent, and feeding them together would let one
    /// sentence run over a hole.
    #[test]
    fn audio_before_a_gap_is_not_context_for_what_follows() {
        let mut chunker = Chunker::new(POLICY);
        push_on(&mut chunker, 4_000).expect("a full chunk");

        let after = chunker
            .push(Some(at(14_000)), &samples(4_000))
            .expect("a full chunk again");
        assert_eq!(
            after.audio.len(),
            samples(4_000).len(),
            "the new chunk alone, with nothing in front of it"
        );
    }

    /// A source whose audio does not start at zero — a file seeked into, a
    /// mixer that has been running — reports where it actually is.
    #[test]
    fn a_timeline_that_does_not_start_at_zero_is_reported_as_it_is() {
        let mut chunker = Chunker::new(POLICY);
        let first = chunker
            .push(Some(at(60_000)), &samples(4_000))
            .expect("a full chunk");
        assert_eq!(first.offset_ms, 60_000);
        assert_eq!(first.accept_from_ms, 60_000);
        assert_eq!(first.accept_until_ms, 63_000);
    }

    /// The rounds that produced the overlapping lines, as a minute of Korean
    /// speech had them: the chunk ending at 36 s, heard with the one before
    /// it, and then the next.
    fn rounds_to_36_and_40() -> (Chunker, Inference, Inference) {
        let mut chunker = Chunker::new(POLICY);
        push_on(&mut chunker, 32_000);
        while chunker.take_chunk().is_some() {}
        let to_36 = push_on(&mut chunker, 4_000).expect("the chunk ending at 36 s");
        let to_40 = push_on(&mut chunker, 4_000).expect("and the one after");
        (chunker, to_36, to_40)
    }

    /// Tokens as the model hands them over: text, then start and end in
    /// stream milliseconds.
    fn tokens(spec: &[(&str, i64, i64)]) -> Vec<Heard> {
        spec.iter()
            .map(|&(text, start_ms, end_ms)| Heard {
                bytes: text.as_bytes().to_vec(),
                start_ms,
                end_ms,
            })
            .collect()
    }

    /// The loss this exists to stop. A segment opening in the context and
    /// running into the new window used to be dropped whole, every round,
    /// because of where it started; the small model lost the last fifteen
    /// seconds of a minute of Korean that way. Its words in the window are
    /// this round's.
    #[test]
    fn a_segment_opening_in_the_context_keeps_what_falls_in_the_window() {
        let (mut chunker, _, to_40) = rounds_to_36_and_40();
        assert_eq!(
            (to_40.accept_from_ms, to_40.accept_until_ms),
            (35_000, 39_000)
        );

        let line = chunker
            .place(
                &to_40,
                &tokens(&[
                    (" 이번에는", 33_000, 34_000),
                    (" 패키지", 34_000, 35_500),
                    (" 여행으로", 35_500, 37_000),
                    (" 다녀왔어요", 37_000, 38_500),
                    (".", 39_000, 39_200),
                ]),
            )
            .expect("the words inside the window are reported");
        assert_eq!(line.text, "여행으로 다녀왔어요");
        assert_eq!((line.start_ms, line.end_ms), (35_500, 38_500));
    }

    /// And the other half of cutting by segment: words said before the
    /// window were reported last round and are not said again. A seam used
    /// to read "한 5일에서 / 한 5일에서 열흘 정도".
    #[test]
    fn what_was_already_reported_is_not_said_again() {
        let (mut chunker, to_36, to_40) = rounds_to_36_and_40();
        let first = chunker
            .place(
                &to_36,
                &tokens(&[(" 한", 33_000, 33_800), (" 5일에서", 33_800, 34_900)]),
            )
            .expect("reported");
        assert_eq!(first.text, "한 5일에서");

        let second = chunker
            .place(
                &to_40,
                &tokens(&[
                    (" 한", 33_000, 33_800),
                    (" 5일에서", 33_800, 34_900),
                    (" 열흘", 35_100, 35_800),
                    (" 정도", 35_800, 36_400),
                ]),
            )
            .expect("reported");
        assert_eq!(second.text, "열흘 정도");
    }

    /// What starts at the edge is left alone — and the next round, which
    /// hears it again as context, is the one that reports it.
    #[test]
    fn the_edge_is_left_for_the_next_round_and_reported_there() {
        let (mut chunker, to_36, to_40) = rounds_to_36_and_40();
        let at_the_edge = tokens(&[(" 사진이", 35_300, 36_000)]);
        assert_eq!(chunker.place(&to_36, &at_the_edge), None);
        assert_eq!(
            chunker.place(&to_40, &at_the_edge).map(|line| line.text),
            Some("사진이".to_owned())
        );
    }

    /// A line cannot end after the audio it was heard in. The model said
    /// 32.8 to 40 of audio that ended at 36.
    #[test]
    fn a_line_ending_past_its_audio_is_cut_where_the_audio_ends() {
        let (mut chunker, to_36, _) = rounds_to_36_and_40();
        assert_eq!(to_36.audio_end_ms, 36_000);
        let line = chunker
            .place(&to_36, &tokens(&[(" 그런데", 32_800, 40_000)]))
            .expect("reported");
        assert_eq!(line.end_ms, 36_000);
    }

    /// Consecutive lines are in order and never overlap — the overlap that
    /// made the MP4 muxer write negative durations.
    #[test]
    fn lines_from_consecutive_rounds_never_overlap() {
        let (mut chunker, to_36, to_40) = rounds_to_36_and_40();
        let first = chunker
            .place(&to_36, &tokens(&[(" 그런데", 33_000, 36_500)]))
            .expect("reported");
        let second = chunker
            .place(
                &to_40,
                &tokens(&[(" 강이나", 35_500, 36_200), (" 호수도", 36_200, 37_000)]),
            )
            .expect("reported");
        assert!(
            second.start_ms >= first.end_ms,
            "{second:?} starts before {first:?} ends"
        );
        assert_eq!(
            second.text, "호수도",
            "and what starts inside the last line is its, not this one's"
        );
    }

    /// A syllable split between two tokens is kept whole rather than cut
    /// through: 스 is three bytes, and here the token boundary falls after
    /// the second of them.
    #[test]
    fn a_character_split_between_tokens_is_kept_whole() {
        let (mut chunker, _, to_40) = rounds_to_36_and_40();
        let syllable = "스".as_bytes();
        assert_eq!(syllable.len(), 3);
        let split = vec![
            Heard {
                bytes: [b" ".as_slice(), &syllable[..2]].concat(),
                start_ms: 34_600,
                end_ms: 35_100,
            },
            Heard {
                bytes: [&syllable[2..], "위스는".as_bytes()].concat(),
                start_ms: 35_100,
                end_ms: 36_000,
            },
        ];
        let line = chunker.place(&to_40, &split).expect("reported");
        assert_eq!(line.text, "스위스는");
        assert_eq!(
            line.start_ms, 35_000,
            "though it starts no earlier than the window"
        );
    }

    /// Nothing in the window is nothing to report — and it leaves nothing
    /// behind that would hold the next line back.
    #[test]
    fn a_segment_with_nothing_in_the_window_reports_nothing() {
        let (mut chunker, _, to_40) = rounds_to_36_and_40();
        assert_eq!(
            chunker.place(&to_40, &tokens(&[(" 알프스", 33_000, 34_500)])),
            None
        );
        assert_eq!(chunker.place(&to_40, &tokens(&[])), None);
        assert!(
            chunker
                .place(&to_40, &tokens(&[(" 산맥이", 35_000, 36_000)]))
                .is_some()
        );
    }

    /// A line the model gave no length keeps none, rather than being lost
    /// or made to run backwards.
    #[test]
    fn a_line_with_no_length_is_kept_with_none() {
        let (mut chunker, to_36, _) = rounds_to_36_and_40();
        let line = chunker
            .place(&to_36, &tokens(&[(" 네", 33_000, 33_000)]))
            .expect("reported");
        assert_eq!((line.start_ms, line.end_ms), (33_000, 33_000));
    }

    /// A model file's header, as whisper.cpp writes it.
    fn header(hparams: [i32; 11]) -> Vec<u8> {
        std::iter::once(0x6767_6d6c_u32 as i32)
            .chain(hparams)
            .flat_map(i32::to_le_bytes)
            .collect()
    }

    /// The header of the turbo model this was measured with, exactly as
    /// whisper.cpp printed it on loading, read back into the shape that
    /// picks its heads.
    #[test]
    fn a_model_s_shape_is_read_from_its_header() {
        let dir =
            std::env::temp_dir().join(format!("media-pp-whisper-shape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ggml-large-v3-turbo.bin");
        // n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
        // n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype
        std::fs::write(
            &path,
            header([51_866, 1500, 1280, 20, 32, 448, 1280, 20, 4, 128, 1]),
        )
        .unwrap();

        let shape = model_shape(&path).expect("a whisper.cpp header");
        assert_eq!(
            shape,
            ModelShape {
                n_vocab: 51_866,
                n_audio_state: 1280,
                n_audio_layer: 32,
                n_text_layer: 4,
                n_mels: 128,
            }
        );
        assert!(matches!(
            alignment_heads(shape),
            Some(whisper_rs::DtwModelPreset::LargeV3Turbo)
        ));

        std::fs::write(&path, b"not a model at all, and long enough to read").unwrap();
        assert_eq!(model_shape(&path), None, "no magic, no shape");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The models told apart by more than size: turbo from large-v3 by its
    /// decoder, large-v3 from large-v2 by its mel bins, English-only from
    /// multilingual by one vocabulary entry.
    #[test]
    fn alignment_heads_follow_the_model_s_shape() {
        use whisper_rs::DtwModelPreset as Preset;

        let shape = |n_vocab, n_audio_state, layers: (i32, i32), n_mels| ModelShape {
            n_vocab,
            n_audio_state,
            n_audio_layer: layers.0,
            n_text_layer: layers.1,
            n_mels,
        };
        assert!(matches!(
            alignment_heads(shape(51_865, 768, (12, 12), 80)),
            Some(Preset::Small)
        ));
        assert!(matches!(
            alignment_heads(shape(51_864, 768, (12, 12), 80)),
            Some(Preset::SmallEn)
        ));
        assert!(matches!(
            alignment_heads(shape(51_866, 1280, (32, 32), 128)),
            Some(Preset::LargeV3)
        ));
        assert!(matches!(
            alignment_heads(shape(51_865, 1280, (32, 32), 80)),
            Some(Preset::LargeV2)
        ));
        assert!(
            alignment_heads(shape(51_866, 1280, (32, 2), 128)).is_none(),
            "a distilled model with a decoder of its own has no preset"
        );
    }

    /// Codes and names are both accepted and come back as whisper.cpp's own
    /// code; anything else is refused before it can reach whisper.cpp, which
    /// would otherwise fail every chunk — or, for an interior NUL, panic.
    #[test]
    fn a_language_is_checked_against_whisper_s_own_list() {
        assert_eq!(language_code("ko").ok(), Some("ko"));
        assert_eq!(language_code("korean").ok(), Some("ko"));
        for refused in ["klingon", "", "ko\0"] {
            assert!(
                matches!(
                    language_code(refused),
                    Err(WhisperTranscriberError::UnknownLanguage(_))
                ),
                "{refused:?} must be refused"
            );
        }
    }
}
