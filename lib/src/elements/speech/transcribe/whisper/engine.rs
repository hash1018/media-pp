//! The whisper.cpp end of every transcriber: what to run, and what of the
//! answer to keep.
//!
//! Kept apart from the elements so that the awkward part — deciding which
//! words two rounds agree on — is tested without a model, and so that every
//! backend runs exactly the same loop.

use super::{ChunkPolicy, SAMPLE_RATE, WhisperTranscriberError};
use crate::elements::speech::meta::{Utterance, Word};

/// Once the audio being heard again is this long, what is committed in it
/// is cut away, at the end of a segment where there is one.
const TRIM_AFTER_MS: i64 = 15_000;

/// The longest the audio being heard again may grow: past it, a round
/// commits everything it heard, agreed or not. Whisper hears thirty seconds
/// at most, and audio two rounds keep disagreeing about — music, noise, a
/// model looping — must not grow until it cannot be heard at all.
const MAX_AUDIO_MS: i64 = 25_000;

/// How much of what was committed before the audio being heard is given to
/// the model as what came before it, in characters.
const PROMPT_CHARS: usize = 200;

/// How far apart, in the model's own times, a committed word and one heard
/// again may be and be the same word: well past how far its guesses wander,
/// and short of a word said again a sentence later.
const SAME_WORD_MS: i64 = 1_500;

/// How many of the words last cut away are still looked for at the start
/// of what is heard: their times were guesses, and the sound of the last
/// of them may still be in the audio.
const CUT_AWAY_WORDS: usize = 8;

/// How many of the last words committed a round's are checked against for
/// the model repeating itself, and how long a run of words repeated is a
/// loop rather than speech.
const LOOP_WINDOW: usize = 60;
const LOOP_RUN: usize = 4;

/// A line ends at a pause this long between two words.
const LINE_PAUSE_MS: i64 = 600;
/// A line ends once it is this long, in time or in characters, whatever
/// else: a subtitle any longer is not read before it is gone.
const LINE_MAX_MS: i64 = 6_000;
const LINE_MAX_CHARS: usize = 42;

/// How far either side of where the model says two words part a cut looks
/// for the quietest moment, and how long a moment is.
const CUT_SEARCH_MS: i64 = 600;
const CUT_WINDOW_MS: i64 = 40;

/// How long a stretch of sound is measured over, and how far below the
/// loudest stretch heard so far one may be and still be sound rather than
/// silence: −30 dB.
const SOUND_WINDOW_MS: i64 = 100;
const SOUND_FLOOR: f32 = 0.0316;

/// How much sound after the last word the model heard, short of the edge,
/// makes the tail worth hearing again on its own: whisper now and then
/// ends a window early, with seconds of speech still in it.
const TAIL_GAP_MS: i64 = 2_000;

/// How far short of the end of what was heard silence is believed: a word
/// only just begun at the very end is not heard as one yet.
const QUIET_EDGE_MS: i64 = 1_000;

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

/// Whether a character belongs to a script written without spaces, where
/// each token is a unit of its own rather than a piece of a word.
fn unspaced(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30FF}' // Hiragana, Katakana
        | '\u{3400}'..='\u{4DBF}' // CJK extension A
        | '\u{4E00}'..='\u{9FFF}' // CJK unified ideographs
        | '\u{F900}'..='\u{FAFF}' // CJK compatibility ideographs
    )
}

/// The RMS of some samples.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Whether two words' keys are the same word heard twice: the same, or
/// differing in one character, or a third of the longer's — "업기" and
/// "없기". A word of one character has to be the same.
fn alike(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let longer = a.len().max(b.len());
    if longer < 2 {
        return false;
    }
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut next = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            next[j + 1] = (row[j] + usize::from(x != y))
                .min(row[j + 1] + 1)
                .min(next[j] + 1);
        }
        row = next;
    }
    row[b.len()] <= (longer / 3).max(1)
}

/// How many words two hearings agree on from their start: words the same,
/// and a word heard otherwise in one of them where the word after it is the
/// same in both — one word misheard, not the two hearings parting.
///
/// Word for word, by place, and nothing else: a round that passed over a
/// stretch the other heard is out of step with it from there, and agrees
/// with it on nothing after.
fn agreement(now: &[HeardWord], before: &[HeardWord]) -> usize {
    let same = |at: usize| match (now.get(at), before.get(at)) {
        (Some(a), Some(b)) => a.key == b.key,
        _ => false,
    };
    let mut agreed = 0;
    while agreed < now.len().min(before.len()) {
        if !same(agreed) && !same(agreed + 1) {
            break;
        }
        agreed += 1;
    }
    agreed
}

/// One word of what the model heard, on the stream's timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HeardWord {
    /// As the model wrote it, with the space in front of it.
    raw: String,
    /// What two rounds compare it by: its letters and digits, lower-cased,
    /// so a comma or a capital the model changed its mind about is the same
    /// word still.
    key: String,
    start_ms: i64,
    end_ms: i64,
    /// Whether the model ended a segment with it — where a line and a cut
    /// of the audio are best made.
    ends_segment: bool,
}

impl HeardWord {
    fn new(raw: String, start_ms: i64, end_ms: i64) -> Self {
        let key = raw
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        Self {
            raw,
            key,
            start_ms,
            end_ms,
            ends_segment: false,
        }
    }

    /// Whether it ends a sentence.
    fn ends_sentence(&self) -> bool {
        self.raw
            .trim_end()
            .ends_with(['.', '?', '!', '。', '？', '！', '…'])
    }
}

/// Each segment's tokens as words: a word begins at each token that starts
/// with a space, which is how whisper's tokens mark one, and runs on
/// through the tokens after it that do not — except in a script written
/// without spaces, where each token is a word of its own. A token that is
/// only punctuation, or the rest of a character split between tokens, goes
/// with the word before it.
fn words_of(segments: &[Vec<Heard>]) -> Vec<HeardWord> {
    let mut words = Vec::new();
    for segment in segments {
        let mut groups: Vec<Vec<&Heard>> = Vec::new();
        for token in segment {
            let text = String::from_utf8_lossy(&token.bytes);
            let begins = match groups.last() {
                None => true,
                Some(_) if starts_mid_character(token) => false,
                Some(_) => {
                    token.bytes.first() == Some(&b' ') || text.chars().next().is_some_and(unspaced)
                }
            };
            let punctuation = !text.chars().any(char::is_alphanumeric);
            match groups.last_mut() {
                Some(group) if !begins || (punctuation && !text.starts_with(' ')) => {
                    group.push(token)
                }
                _ => groups.push(vec![token]),
            }
        }
        let before = words.len();
        for group in groups {
            let bytes: Vec<u8> = group
                .iter()
                .flat_map(|token| token.bytes.iter().copied())
                .collect();
            let raw = String::from_utf8_lossy(&bytes).into_owned();
            let start = group[0].start_ms;
            let end = group[group.len() - 1].end_ms.max(start);
            let word = HeardWord::new(raw, start, end);
            if !word.key.is_empty() {
                words.push(word);
            } else if words.len() > before {
                // Punctuation on its own, after a space: the end of the
                // word before it, not a word.
                let last = words.last_mut().expect("a word of this segment");
                last.raw.push_str(word.raw.trim_start());
                last.end_ms = last.end_ms.max(word.end_ms);
            }
        }
        if words.len() > before
            && let Some(last) = words.last_mut()
        {
            last.ends_segment = true;
        }
    }
    words
}

/// What the [`Chunker`] wants run.
#[derive(Debug, PartialEq)]
pub(super) struct Inference {
    /// Where `audio` starts, in milliseconds from the first sample ever
    /// given to the chunker. Whisper reports its segments relative to the
    /// audio it was handed, so this is what turns those into stream time.
    pub(super) offset_ms: i64,
    /// Everything not yet cut away: what was committed in it, heard again
    /// so that the words after it are heard in their sentence, and what
    /// was not.
    pub(super) audio: Vec<f32>,
    /// What was committed before `audio`, the end of it, for the model to
    /// read as what came before.
    pub(super) prompt: String,
    /// Whether this is the last round, once nothing more is coming: what it
    /// hears is committed whether or not a round agrees with it.
    last: bool,
}

/// What one round did, for the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Round {
    /// Words committed.
    pub(super) committed: usize,
    /// Words heard and left for the next round to agree with.
    pub(super) tentative: usize,
    /// Whether the round committed what it heard without agreement — the
    /// last round, or audio grown to [`MAX_AUDIO_MS`].
    pub(super) forced: bool,
    /// Whether what the round would have committed repeated what was just
    /// committed, and was refused as the model looping.
    pub(super) looped: bool,
}

/// Decides what to transcribe and which of the answer to believe.
///
/// Separated from the model so that the awkward part can be tested without
/// one. Every decision it makes is a function of sample counts and of the
/// words it is handed, and none of it needs whisper.cpp to be present.
///
/// # Agreement
///
/// The audio not yet committed is kept and heard again every round, a
/// [`ChunkPolicy::chunk_ms`] more of it each time. A word is committed —
/// reported, never to change — once two rounds in a row hear the same
/// words up to it: the words both rounds begin with, after what is
/// committed already, compared by their text and not by when the model
/// says they were said ("LocalAgreement", from Macháček et al.'s
/// whisper_streaming). What the two disagree about is left for the next
/// round, which hears it again with more after it.
///
/// Cutting each round by the times the model gives its words lost them and
/// repeated them: the same word heard in two rounds is given two times
/// hundreds of milliseconds apart, so one at the seam fell before both
/// rounds' windows, or inside both. Text that two rounds agree on is said
/// once, wherever its times fall.
///
/// # What is heard again
///
/// What was committed stays in the audio heard, and the model says it
/// again each round; it is found at the start of what the model says by
/// its text and passed over. Once the audio is longer than
/// [`TRIM_AFTER_MS`] it is cut after a committed word that ended a
/// segment, two committed words left after the cut, at the quietest moment
/// near where the model says the words part — its times are a second out
/// in a small model, and a round begun inside a word passed over the
/// sentence it began. The text before the cut is given as the prompt
/// instead, and the last [`CUT_AWAY_WORDS`] words cut away are still passed
/// over where the model says them again. It stays shorter than Whisper's
/// thirty seconds by committing what it heard at [`MAX_AUDIO_MS`], agreed
/// or not.
///
/// # The model repeating itself
///
/// A line the model makes up, committed, goes into the prompt, and the
/// model then says it again, every round, with growing confidence — and
/// every round agrees. A run of [`LOOP_RUN`] words the last
/// [`LOOP_WINDOW`] committed words already hold is passed over, and the
/// prompt is dropped so that the next round hears the audio alone.
///
/// # Lines
///
/// Committed words are gathered into lines, which end at a sentence's end,
/// a segment's end, a pause of [`LINE_PAUSE_MS`], or past
/// [`LINE_MAX_MS`] or [`LINE_MAX_CHARS`]. Lines come in order and never
/// overlap. The times are the model's, kept in order: a word starts no
/// earlier than the one before it ends.
pub(super) struct Chunker {
    /// How much new audio each round waits for, in samples and in
    /// milliseconds.
    step: usize,
    step_ms: i64,
    /// The audio heard each round, from `audio_start`.
    audio: Vec<f32>,
    /// Where `audio` starts on the timeline of the audio itself.
    audio_start: i64,
    /// Samples arrived since the last round.
    fresh: usize,
    /// Where the next sample is expected to land, so one arriving anywhere
    /// else is recognised as a gap. `None` before the first.
    next_pts: Option<i64>,
    /// Words committed whose audio is still in `audio`: what each round
    /// says again and is not to report twice.
    committed: Vec<HeardWord>,
    /// What was committed before `audio`, its end, for the prompt.
    history: String,
    /// The last words cut away with the audio — see [`CUT_AWAY_WORDS`].
    cut_away: Vec<HeardWord>,
    /// The keys of the last words committed, for telling the model
    /// repeating itself from speech — see [`LOOP_WINDOW`].
    recent: std::collections::VecDeque<String>,
    /// What the last round heard after what is committed: what this round
    /// has to agree with.
    tentative: Vec<HeardWord>,
    /// Committed words not yet made a line.
    line: Vec<HeardWord>,
    /// Lines finished and not yet taken.
    lines: Vec<Utterance>,
    /// Where the last committed word ends — or, where nothing was heard,
    /// how far the silence is believed to run — in milliseconds.
    committed_until_ms: i64,
    /// Whether the next audio begins the timeline again, after a reset.
    restart: bool,
    /// The loudest [`SOUND_WINDOW_MS`] stretch heard, as RMS: what sound is
    /// measured against.
    loudest: f32,
    /// Where the last round first heard something not committed, or short
    /// of the end of its audio where it heard nothing: how far it believed
    /// the silence ran. `None` before a round, which one round alone does
    /// not settle.
    heard_from: Option<i64>,
}

impl Chunker {
    pub(super) fn new(policy: ChunkPolicy) -> Self {
        Self {
            step: (policy.chunk_ms as usize * SAMPLE_RATE as usize / 1000).max(1),
            step_ms: i64::from(policy.chunk_ms),
            audio: Vec::new(),
            audio_start: 0,
            fresh: 0,
            next_pts: None,
            committed: Vec::new(),
            history: String::new(),
            cut_away: Vec::new(),
            recent: std::collections::VecDeque::new(),
            tentative: Vec::new(),
            line: Vec::new(),
            lines: Vec::new(),
            committed_until_ms: i64::MIN,
            restart: false,
            heard_from: None,
            loudest: 0.0,
        }
    }

    fn samples_to_ms(samples: i64) -> i64 {
        (samples * 1000) / SAMPLE_RATE as i64
    }

    fn audio_end_ms(&self) -> i64 {
        Self::samples_to_ms(self.audio_start + self.audio.len() as i64)
    }

    /// Takes more audio, and answers the round it now wants run.
    ///
    /// `pts` places the samples on the timeline of the audio itself, in that
    /// timeline's own units — which for the
    /// [`WHISPER_SAMPLE_RATE`](SAMPLE_RATE) mono this accepts is a sample
    /// index. `None` means "where the last frame ended", which is what a source
    /// that stamps nothing amounts to.
    ///
    /// Answers `None` until another [`ChunkPolicy::chunk_ms`] has arrived.
    ///
    /// # A gap ends what was being heard
    ///
    /// Audio that does not continue where the last left off — a queue that
    /// dropped buffers because this is behind, a source that jumped —
    /// cannot be heard as one with what came before: one sentence would run
    /// over two moments that are not adjacent. What the last round heard is
    /// committed as it was heard, the line in progress ends, and the
    /// timeline picks up where the new audio actually is.
    ///
    /// The alternative, counting samples and letting positions slide, is
    /// worse in the way that matters: every line after the loss would be
    /// early by what was lost, and the error would accumulate instead of
    /// staying where it happened.
    pub(super) fn push(&mut self, pts: Option<i64>, samples: &[f32]) -> Option<Inference> {
        let pts = pts.or(self.next_pts).unwrap_or(0);
        match self.next_pts {
            _ if self.restart => {
                self.restart = false;
                self.audio_start = pts;
                self.committed_until_ms = Self::samples_to_ms(pts);
            }
            None => {
                self.audio_start = pts;
                self.committed_until_ms = Self::samples_to_ms(pts);
            }
            Some(expected) if pts != expected => {
                self.break_off();
                self.audio_start = pts;
                self.committed_until_ms = self.committed_until_ms.max(Self::samples_to_ms(pts));
            }
            Some(_) => {}
        }
        let window = (SAMPLE_RATE as i64 * SOUND_WINDOW_MS / 1000) as usize;
        for stretch in samples.chunks(window) {
            self.loudest = self.loudest.max(rms(stretch));
        }
        self.audio.extend_from_slice(samples);
        self.fresh += samples.len();
        self.next_pts = Some(pts + samples.len() as i64);
        if self.fresh < self.step {
            return None;
        }
        self.fresh = 0;
        Some(self.inference(false))
    }

    /// The last round, once no more audio is coming: what it hears is
    /// committed as it is heard, since no later round will hear more.
    /// `None` when there is nothing left to hear — and then the line in
    /// progress is finished here.
    pub(super) fn flush(&mut self) -> Option<Inference> {
        if self.audio.is_empty() || (self.fresh == 0 && self.tentative.is_empty()) {
            self.finish_line();
            return None;
        }
        self.fresh = 0;
        Some(self.inference(true))
    }

    fn inference(&self, last: bool) -> Inference {
        Inference {
            offset_ms: Self::samples_to_ms(self.audio_start),
            audio: self.audio.clone(),
            prompt: self.history.clone(),
            last,
        }
    }

    /// Takes what the model heard in `inference`, its segments' tokens
    /// placed on the stream's timeline, and commits what this round agrees
    /// with the last on.
    pub(super) fn heard(&mut self, inference: &Inference, segments: &[Vec<Heard>]) -> Round {
        let audio_end = self.audio_end_ms();
        let mut hypothesis = words_of(segments);
        for word in &mut hypothesis {
            word.start_ms = word.start_ms.clamp(inference.offset_ms, audio_end);
            word.end_ms = word.end_ms.clamp(word.start_ms, audio_end);
        }
        let fresh = self.after_committed(hypothesis);

        let forced = inference.last || audio_end - inference.offset_ms >= MAX_AUDIO_MS;
        // The model repeating itself — a line it made up, then said again
        // and again because the prompt now holds it — is agreed on by every
        // round, since every round hears it the same. What it repeats is
        // passed over, and the prompt that keeps it going is dropped.
        let (fresh, looped) = self.without_repeats(fresh);
        if looped {
            self.history.clear();
        }
        let agreed = if inference.last {
            // The fuller of this hearing and the last, then whatever this one
            // heard after it: a last round that passed over the end of the
            // stream would otherwise take with it what the round before
            // heard there.
            let count = self.commit_longest(fresh, i64::MAX);
            let rest = std::mem::take(&mut self.tentative);
            let more = rest.len();
            self.commit_all(rest);
            count + more
        } else if forced {
            self.commit_longest(fresh, audio_end - self.step_ms)
        } else if fresh.is_empty() {
            // A round that heard nothing new leaves what the last one heard
            // to be agreed with: it more likely passed over the speech than
            // heard it end.
            0
        } else {
            let agreed = agreement(&fresh, &self.tentative);
            let mut fresh = fresh.into_iter();
            for word in fresh.by_ref().take(agreed) {
                self.commit(word);
            }
            self.tentative = fresh.collect();
            agreed
        };

        // Silence is agreed on as words are: what this round and the last
        // both heard nothing new in, up to where either heard something, is
        // settled. One round alone is not believed — whisper now and then
        // passes over a stretch of speech whole, and the next round hears
        // it — and what one round's silence cut away was lost.
        let heard_from = match self.tentative.first() {
            Some(word) => word.start_ms - QUIET_EDGE_MS / 2,
            None => audio_end - QUIET_EDGE_MS,
        };
        // And only where the audio is quiet too: two rounds both passed
        // over a sentence begun near the end of what they heard, nine
        // seconds of it, and it was settled as silence.
        let quiet = self.heard_from.map(|before| {
            let quiet = before.min(heard_from);
            match self.sound_from(self.committed_until_ms, quiet) {
                Some(sound) => quiet.min(sound - QUIET_EDGE_MS / 2),
                None => quiet,
            }
        });
        self.heard_from = Some(heard_from);
        if let Some(quiet) = quiet {
            self.committed_until_ms = self.committed_until_ms.max(quiet.min(audio_end));
        }

        if inference.last {
            self.finish_line();
            self.committed_until_ms = self.committed_until_ms.max(audio_end);
            self.cut(audio_end);
        } else {
            self.trim(quiet);
        }
        Round {
            committed: agreed,
            tentative: self.tentative.len(),
            forced,
            looped,
        }
    }

    /// Commits everything heard, once nothing more will be: the last round.
    fn commit_all(&mut self, fresh: Vec<HeardWord>) {
        for word in fresh {
            self.commit(word);
        }
        self.tentative.clear();
    }

    /// Commits, at the longest the audio may grow, what is older than
    /// `edge` in whichever of this round's hearing and the last one's holds
    /// more of it — the one less likely to have passed something over —
    /// and keeps this round's newer words in doubt. Answers how many were
    /// committed.
    fn commit_longest(&mut self, fresh: Vec<HeardWord>, edge: i64) -> usize {
        let older = |words: &[HeardWord]| words.iter().take_while(|w| w.end_ms <= edge).count();
        let (now, before) = (older(&fresh), older(&self.tentative));
        let (committing, until) = if before > now {
            let words: Vec<HeardWord> = self.tentative.drain(..before).collect();
            let until = words.last().map_or(i64::MIN, |w| w.end_ms);
            (words, until)
        } else {
            let until = fresh[..now].last().map_or(i64::MIN, |w| w.end_ms);
            (fresh[..now].to_vec(), until)
        };
        let count = committing.len();
        for word in committing {
            self.commit(word);
        }
        // What this round heard after what was committed, by time: the two
        // hearings are not in step, which is why this round is forced.
        self.tentative = fresh.into_iter().filter(|w| w.start_ms >= until).collect();
        count
    }

    /// What the model heard after what is committed: the committed words
    /// it said again, found at its start by their text, passed over — and
    /// with them whatever it said before them, its prompt said again
    /// included.
    ///
    /// Found as the shortest start of what was heard that holds as many of
    /// the committed words, in order, as any — each heard the same and near
    /// where it was before (`SAME_WORD_MS`) — so a committed word the model
    /// now hears otherwise, or not at all, still lines the rest up, and a
    /// round that says none of them again passes over nothing.
    ///
    /// Only a word heard the same counts, but for the committed words after
    /// the last of those, which a round may have committed as it heard them
    /// — "업기" where the next hears "없기" — and which are otherwise
    /// reported again: each heard alike next is passed over too. Counting a
    /// word heard alike anywhere lined a committed word the cut had left
    /// out against the new word after it, which was then lost.
    fn after_committed(&self, mut heard: Vec<HeardWord>) -> Vec<HeardWord> {
        let reference: Vec<&HeardWord> = self.cut_away.iter().chain(&self.committed).collect();
        if reference.is_empty() {
            return heard;
        }
        let near = |a: &HeardWord, b: &HeardWord| (a.start_ms - b.start_ms).abs() <= SAME_WORD_MS;
        let same = |a: &HeardWord, b: &HeardWord| a.key == b.key && near(a, b);
        // The longest common subsequence of every start of the committed
        // words and every start of what was heard.
        let (done, heard_len) = (&reference, heard.len());
        let mut table = vec![vec![0usize; heard_len + 1]; done.len() + 1];
        for i in 0..done.len() {
            for j in 0..heard_len {
                table[i + 1][j + 1] = if same(done[i], &heard[j]) {
                    table[i][j] + 1
                } else {
                    table[i][j + 1].max(table[i + 1][j])
                };
            }
        }
        let most = table[done.len()][heard_len];
        let mut skip = table[done.len()]
            .iter()
            .position(|&found| found == most)
            .unwrap_or(0);
        // The committed words after the last one heard the same, which a
        // round may have committed as it heard them: each heard alike next
        // is passed over too.
        let mut matched = 0;
        let (mut i, mut j) = (done.len(), skip);
        while i > 0 && j > 0 {
            if same(done[i - 1], &heard[j - 1]) && table[i][j] == table[i - 1][j - 1] + 1 {
                matched = i;
                break;
            } else if table[i - 1][j] >= table[i][j - 1] {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        // Only where some committed word was heard the same: where none
        // was, the cut may have taken them all, and a new word alike one of
        // them would be passed over in its place.
        if most > 0 {
            for &word in &done[matched..] {
                if heard
                    .get(skip)
                    .is_some_and(|next| alike(&next.key, &word.key) && near(next, word))
                {
                    skip += 1;
                }
            }
        }
        heard.drain(..skip);
        heard
    }

    /// `heard` without the words that repeat what was just committed or
    /// said before them in it — each run of [`LOOP_RUN`] words that the
    /// last [`LOOP_WINDOW`] committed words, or the words before it, already
    /// hold — and whether there were any.
    ///
    /// Only the repeats go: refusing a whole round for one lost the words
    /// around it, and on the last round the end of the stream with them.
    fn without_repeats(&self, heard: Vec<HeardWord>) -> (Vec<HeardWord>, bool) {
        let keys: Vec<&str> = self
            .recent
            .iter()
            .map(String::as_str)
            .chain(heard.iter().map(|w| w.key.as_str()))
            .collect();
        let new_from = self.recent.len();
        let mut repeated = vec![false; heard.len()];
        let mut seen = std::collections::HashSet::new();
        for (at, run) in keys.windows(LOOP_RUN).enumerate() {
            if !seen.insert(run) {
                for index in at.max(new_from)..at + LOOP_RUN {
                    repeated[index - new_from] = true;
                }
            }
        }
        let looped = repeated.contains(&true);
        let kept = heard
            .into_iter()
            .zip(repeated)
            .filter_map(|(word, repeated)| (!repeated).then_some(word))
            .collect();
        (kept, looped)
    }

    /// Reports `word`, never to change, as part of the line in progress.
    fn commit(&mut self, mut word: HeardWord) {
        word.start_ms = word.start_ms.max(self.committed_until_ms);
        word.end_ms = word.end_ms.max(word.start_ms);
        if let Some(last) = self.line.last()
            && word.start_ms - last.end_ms >= LINE_PAUSE_MS
        {
            self.finish_line();
        }
        self.committed_until_ms = word.end_ms;
        if self.recent.len() == LOOP_WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(word.key.clone());
        self.committed.push(word.clone());
        let ends = word.ends_sentence() || word.ends_segment;
        self.line.push(word);

        let chars: usize = self.line.iter().map(|w| w.raw.chars().count()).sum();
        let span = self.line[self.line.len() - 1].end_ms - self.line[0].start_ms;
        if ends || chars >= LINE_MAX_CHARS || span >= LINE_MAX_MS {
            self.finish_line();
        }
    }

    /// Makes the words gathered so far a line.
    fn finish_line(&mut self) {
        if self.line.is_empty() {
            return;
        }
        let words = std::mem::take(&mut self.line);
        let text: String = words.iter().map(|w| w.raw.as_str()).collect();
        let start = words[0].start_ms;
        let end = words[words.len() - 1].end_ms.max(start);
        self.lines.push(Utterance {
            start_ms: start,
            end_ms: end,
            text: text.trim().to_owned(),
            words: words
                .into_iter()
                .map(|w| Word {
                    start_ms: w.start_ms,
                    end_ms: w.end_ms,
                    text: w.raw.trim().to_owned(),
                })
                .collect(),
        });
    }

    /// Cuts away what is committed once the audio has grown long: between
    /// two committed words, after the last one that ended a segment where
    /// one did — or, where nothing is in doubt, as far as two rounds agree
    /// there was silence (`quiet`), and at the longest the audio may grow,
    /// past whatever is committed.
    ///
    /// Never in the last two committed words: the times are the model's
    /// guesses, and a cut on one can fall inside the word after it. With
    /// committed words after the cut, whatever of a word the cut leaves to
    /// be heard again is heard among words already committed and passed
    /// over with them — two, so that one the cut took out entirely leaves
    /// another to line up by.
    fn trim(&mut self, quiet: Option<i64>) {
        let audio_end = self.audio_end_ms();
        if audio_end - Self::samples_to_ms(self.audio_start) < TRIM_AFTER_MS {
            return;
        }
        let between = self.committed.len().saturating_sub(2);
        let cut = self.committed[..between]
            .iter()
            .rposition(|word| word.ends_segment)
            .or(between.checked_sub(1))
            .map(|at| {
                let (end, next) = (self.committed[at].end_ms, self.committed[at + 1].start_ms);
                // Where the model says the words part, moved to where the
                // sound is quietest near it — but never into the second word
                // kept, which the cut must leave whole to line up by.
                let guess = end + (next - end).max(0) / 2;
                let low = self.committed[at].start_ms + CUT_SEARCH_MS / 4;
                let high = self.committed[at + 2].start_ms;
                self.quietest(guess, low, high)
            })
            .or_else(|| {
                // At the longest, past whatever is committed, however little:
                // the audio must not outgrow what Whisper can hear.
                let longest = audio_end - Self::samples_to_ms(self.audio_start) >= MAX_AUDIO_MS;
                match (self.tentative.first(), self.committed.last()) {
                    // Nothing in doubt: what was heard is all committed, and
                    // the silence after it as far as two rounds agree on it.
                    (None, last) => quiet.max(last.filter(|_| longest).map(|w| w.end_ms)),
                    (Some(first), Some(last)) if longest => {
                        Some(last.end_ms + (first.start_ms - last.end_ms).max(0) / 2)
                    }
                    (Some(first), None) if longest => Some(first.start_ms - QUIET_EDGE_MS / 2),
                    _ => None,
                }
            });
        // At the longest, back to what a round may grow from, whatever the
        // cut found: everything older is committed by then, and only the
        // newest chunk's words are in doubt.
        let longest = audio_end - Self::samples_to_ms(self.audio_start) >= MAX_AUDIO_MS;
        let cut = if longest {
            let floor = audio_end - TRIM_AFTER_MS;
            let ceiling = self
                .tentative
                .first()
                .map_or(audio_end, |first| first.start_ms - QUIET_EDGE_MS / 2);
            Some(cut.unwrap_or(floor).max(floor).min(ceiling))
        } else {
            cut
        };
        if let Some(cut) = cut {
            self.cut(cut);
        }
    }

    /// Where sound begins between `from_ms` and `to_ms` in the audio held:
    /// the first [`SOUND_WINDOW_MS`] stretch within [`SOUND_FLOOR`] of the
    /// loudest heard.
    fn sound_from(&self, from_ms: i64, to_ms: i64) -> Option<i64> {
        let rate = SAMPLE_RATE as i64;
        let window = (rate * SOUND_WINDOW_MS / 1000) as usize;
        let at = |ms: i64| {
            (ms * rate / 1000 - self.audio_start).clamp(0, self.audio.len() as i64) as usize
        };
        let (from, to) = (at(from_ms), at(to_ms));
        let floor = self.loudest * SOUND_FLOOR;
        (from..to.saturating_sub(window).max(from))
            .step_by(window)
            .find(|&start| rms(&self.audio[start..(start + window).min(self.audio.len())]) > floor)
            .map(|start| (self.audio_start + start as i64) * 1000 / rate)
    }

    /// The tail of `inference` to hear again on its own, where the model
    /// heard nothing in seconds of sound after its last word: whisper ends a
    /// window early now and then, and hearing what it left out without the
    /// rest — and without the prompt — is what gets it heard.
    pub(super) fn tail(&self, inference: &Inference, segments: &[Vec<Heard>]) -> Option<Inference> {
        let audio_end =
            inference.offset_ms + inference.audio.len() as i64 * 1000 / SAMPLE_RATE as i64;
        let heard_until = segments
            .iter()
            .flatten()
            .map(|token| token.end_ms)
            .max()
            .unwrap_or(inference.offset_ms)
            .clamp(inference.offset_ms, audio_end);
        let edge = if inference.last {
            audio_end
        } else {
            audio_end - QUIET_EDGE_MS
        };
        if edge - heard_until < TAIL_GAP_MS {
            return None;
        }
        let sound = self.sound_from(heard_until, edge)?;
        let from = (sound - QUIET_EDGE_MS / 2).max(heard_until);
        let skip = ((from - inference.offset_ms) * SAMPLE_RATE as i64 / 1000) as usize;
        Some(Inference {
            offset_ms: inference.offset_ms + skip as i64 * 1000 / SAMPLE_RATE as i64,
            audio: inference.audio.get(skip..)?.to_vec(),
            prompt: String::new(),
            last: inference.last,
        })
    }

    /// The quietest moment within [`CUT_SEARCH_MS`] of `guess_ms`, and
    /// between `low_ms` and `high_ms`: where to cut so that what is heard
    /// next begins in a pause rather than inside a word.
    ///
    /// The model's times for its words are guesses, a second out in a
    /// small model; a cut on one left the round after it to begin with the
    /// end of a word, and whisper, handed that, passed over the sentence it
    /// began — two rounds running, so that it was agreed away.
    fn quietest(&self, guess_ms: i64, low_ms: i64, high_ms: i64) -> i64 {
        let rate = SAMPLE_RATE as i64;
        let window = (rate * CUT_WINDOW_MS / 1000) as usize;
        let at = |ms: i64| {
            (ms * rate / 1000 - self.audio_start).clamp(0, self.audio.len() as i64) as usize
        };
        let from = at(low_ms.max(guess_ms - CUT_SEARCH_MS));
        let to = at(high_ms.min(guess_ms + CUT_SEARCH_MS));
        if to < from + window {
            return guess_ms;
        }
        let step = window / 4;
        let energy = |start: usize| -> f32 {
            self.audio[start..start + window]
                .iter()
                .map(|s| s * s)
                .sum()
        };
        let mut best = (f32::INFINITY, guess_ms);
        let mut start = from;
        while start + window <= to {
            let e = energy(start);
            let centre = (self.audio_start + (start + window / 2) as i64) * 1000 / rate;
            // The nearer to the guess of two alike, so silence is cut in the
            // middle rather than at its edge.
            let closer = (centre - guess_ms).abs() < (best.1 - guess_ms).abs();
            if e < best.0 * 0.999 || (e <= best.0 * 1.001 && closer) {
                best = (e, centre);
            }
            start += step.max(1);
        }
        best.1
    }

    /// Lets go of the audio before `cut_ms`, the committed words in it
    /// going into the prompt.
    fn cut(&mut self, cut_ms: i64) {
        let cut = (cut_ms * SAMPLE_RATE as i64 / 1000)
            .clamp(self.audio_start, self.audio_start + self.audio.len() as i64);
        self.audio.drain(..(cut - self.audio_start) as usize);
        self.audio_start = cut;
        let cut_ms = Self::samples_to_ms(cut);
        let kept = self.committed.iter().position(|w| w.end_ms > cut_ms);
        let gone: Vec<HeardWord> = match kept {
            Some(at) => self.committed.drain(..at).collect(),
            None => std::mem::take(&mut self.committed),
        };
        for word in gone {
            self.history.extend(word.raw.chars().filter(|&c| c != '\0'));
            self.cut_away.push(word);
        }
        let excess = self.cut_away.len().saturating_sub(CUT_AWAY_WORDS);
        self.cut_away.drain(..excess);
        let excess = self.history.chars().count().saturating_sub(PROMPT_CHARS);
        if excess > 0 {
            let from = self
                .history
                .char_indices()
                .nth(excess)
                .map_or(self.history.len(), |(at, _)| at);
            self.history.drain(..from);
        }
    }

    /// The lines finished since the last call, in order.
    pub(super) fn take_lines(&mut self) -> Vec<Utterance> {
        std::mem::take(&mut self.lines)
    }

    /// Where everything before is committed and made a line, in
    /// milliseconds: what audio up to here holds will not change.
    pub(super) fn settled_until_ms(&self) -> i64 {
        match self.line.first() {
            Some(word) => self.committed_until_ms.min(word.start_ms),
            None => self.committed_until_ms,
        }
    }

    /// Ends what was being heard at a gap: what the last round heard is
    /// committed as it was heard, since nothing more will be heard with it.
    fn break_off(&mut self) {
        for word in std::mem::take(&mut self.tentative) {
            self.commit(word);
        }
        self.finish_line();
        self.audio.clear();
        self.committed.clear();
        self.history.clear();
        self.cut_away.clear();
        self.recent.clear();
        self.heard_from = None;
        self.fresh = 0;
    }

    /// Forgets everything, for a seek or a stop — what was heard and not
    /// yet taken included, since `Stop` abandons and a seek's audio is from
    /// elsewhere.
    ///
    /// The timeline is not reset: the next audio is placed where its own
    /// timestamps say, wherever that is, because what the caller does with
    /// a seek is its own business and inventing a new zero here would only
    /// disagree with it.
    pub(super) fn reset(&mut self) {
        self.restart = true;
        self.audio.clear();
        self.committed.clear();
        self.history.clear();
        self.tentative.clear();
        self.line.clear();
        self.lines.clear();
        self.cut_away.clear();
        self.recent.clear();
        self.heard_from = None;
        self.fresh = 0;
    }
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

    /// One segment's tokens as the model hands them over: text, then start
    /// and end in stream milliseconds.
    fn segment(spec: &[(&str, i64, i64)]) -> Vec<Heard> {
        spec.iter()
            .map(|&(text, start_ms, end_ms)| Heard {
                bytes: text.as_bytes().to_vec(),
                start_ms,
                end_ms,
            })
            .collect()
    }

    fn texts(lines: &[Utterance]) -> Vec<&str> {
        lines.iter().map(|line| line.text.as_str()).collect()
    }

    /// Nothing is heard until a whole chunk has arrived, and then the
    /// round hears everything not yet cut away.
    #[test]
    fn a_round_waits_for_a_chunk_and_hears_all_that_is_not_cut_away() {
        let mut chunker = Chunker::new(POLICY);
        assert_eq!(push_on(&mut chunker, 3_999), None);
        let first = push_on(&mut chunker, 1).expect("a whole chunk");
        assert_eq!(
            (first.offset_ms, first.audio.len()),
            (0, samples(4_000).len())
        );
        chunker.heard(&first, &[]);

        let second = push_on(&mut chunker, 4_000).expect("another");
        assert_eq!(
            (second.offset_ms, second.audio.len()),
            (0, samples(8_000).len()),
            "nothing was committed, so nothing was cut away"
        );
    }

    /// A word is reported once two rounds in a row hear the words up to
    /// it; what they disagree about waits for the next.
    #[test]
    fn what_two_rounds_agree_on_is_committed_and_the_rest_waits() {
        let mut chunker = Chunker::new(POLICY);
        let first = push_on(&mut chunker, 4_000).unwrap();
        let round = chunker.heard(
            &first,
            &[segment(&[
                (" 재입국", 200, 900),
                (" 충격은", 900, 1_600),
                (" 신혼", 1_600, 3_900),
            ])],
        );
        assert_eq!(
            (round.committed, round.tentative),
            (0, 3),
            "nothing to agree with yet"
        );

        let second = push_on(&mut chunker, 4_000).unwrap();
        let round = chunker.heard(
            &second,
            &[segment(&[
                (" 재입국", 300, 1_000),
                (" 충격은", 1_000, 1_700),
                (" 신혼", 1_700, 2_300),
                (" 단계가", 2_300, 3_000),
                (" 업기", 3_000, 3_500),
            ])],
        );
        assert_eq!((round.committed, round.tentative), (3, 2));
        assert!(chunker.take_lines().is_empty(), "the line runs on");
        assert!(
            chunker.settled_until_ms() <= 300,
            "and the audio it began in is held until it is a line"
        );
    }

    /// The loss this exists to stop. The same words heard in two rounds
    /// are given times hundreds of milliseconds apart; cut by time, a word
    /// at the seam fell outside both rounds or inside both. Agreed by text,
    /// each is said once.
    #[test]
    fn times_that_move_between_rounds_lose_and_repeat_nothing() {
        let mut chunker = Chunker::new(POLICY);
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(
            &first,
            &[segment(&[
                (" 빠르게", 2_400, 3_100),
                (" 발생", 3_100, 3_900),
            ])],
        );
        let second = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(
            &second,
            &[segment(&[
                (" 빠르게", 2_900, 3_600),
                (" 발생하며", 3_600, 4_400),
                (" 더", 4_400, 4_600),
            ])],
        );
        let third = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(
            &third,
            &[segment(&[
                (" 빠르게", 2_600, 3_300),
                (" 발생하며", 3_300, 4_100),
                (" 더", 4_100, 4_300),
                (" 오래", 4_300, 4_900),
                (" 지속하고.", 4_900, 6_000),
            ])],
        );
        let last = chunker.flush().expect("the tail is heard");
        chunker.heard(
            &last,
            &[segment(&[
                (" 빠르게", 2_700, 3_400),
                (" 발생하며", 3_400, 4_200),
                (" 더", 4_200, 4_400),
                (" 오래", 4_400, 5_000),
                (" 지속하고.", 5_000, 6_100),
            ])],
        );
        assert_eq!(
            texts(&chunker.take_lines()).join(" "),
            "빠르게 발생하며 더 오래 지속하고."
        );
    }

    /// A tiny, seeded generator: the same run every time, and no crate for
    /// it.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.0 >> 33
        }

        fn below(&mut self, n: u64) -> i64 {
            (self.next() % n) as i64
        }
    }

    /// Three hundred words, said one after another with pauses, heard
    /// round after round the way a model hears them: the times off by up
    /// to 300 ms each time, the words near the end of the audio missing or
    /// misheard, now and then a word already committed heard differently,
    /// and now and then a round passing over all it heard or a segment of
    /// it — though never what the round before it passed over: two rounds
    /// that agree a stretch was not said are believed. Every word comes out once, in order, and every line
    /// after the one before it — for each of twenty seeded runs: one seed
    /// alone passed while others lost a sentence. Over three hundred seeds,
    /// three runs of three hundred words still lose or repeat some.
    #[test]
    fn a_long_stream_heard_unsteadily_comes_out_word_for_word() {
        for seed in 1..=20 {
            heard_unsteadily(seed);
        }
    }

    fn heard_unsteadily(seed: u64) {
        let mut rng = Lcg(seed);
        let mut said = Vec::new();
        let mut t = 300;
        for n in 0..300 {
            let length = 300 + rng.below(400);
            said.push((format!(" w{n}"), t, t + length));
            t += length + if rng.below(8) == 0 { 700 } else { 60 };
        }
        let total_ms = t + 1_000;

        let mut chunker = Chunker::new(POLICY);
        let mut lines = Vec::new();
        // What the last round passed over, as indexes into `said`: the next
        // does not pass over it too. Two rounds that both pass over the
        // same words agree they were not said, and agreement is final.
        let mut passed_over = 0..0;
        let mut hear = |chunker: &mut Chunker, inference: &Inference, rng: &mut Lcg| {
            let end = inference.offset_ms + inference.audio.len() as i64 * 1000 / 16_000;
            let edge = if inference.last { 0 } else { rng.below(900) };
            let mut segments = Vec::new();
            let mut current = Vec::new();
            let mut indexes = Vec::new();
            for (n, (text, start, stop)) in said.iter().enumerate() {
                if *start < inference.offset_ms || *stop > end - edge {
                    continue;
                }
                indexes.push(n);
                let jitter = |rng: &mut Lcg| rng.below(601) - 300;
                let mut text = text.clone();
                // A word near the end of what was heard, guessed at.
                if !inference.last && *stop > end - 1_500 && rng.below(2) == 0 {
                    text = format!(" guess{}", rng.below(1_000));
                }
                // A word heard before, heard otherwise this time.
                if !inference.last && *stop < end - 4_000 && rng.below(25) == 0 {
                    text.push('x');
                }
                current.push(Heard {
                    bytes: text.into_bytes(),
                    start_ms: start + jitter(rng),
                    end_ms: stop + jitter(rng),
                });
                if n % 7 == 6 {
                    segments.push(std::mem::take(&mut current));
                }
            }
            if !current.is_empty() {
                segments.push(current);
            }
            // Now and then a round passes over speech whole: all of it, or
            // one segment — never what the round before passed over.
            let mut spans = Vec::new();
            let mut at = 0;
            for tokens in &segments {
                spans.push(indexes[at]..indexes[at + tokens.len() - 1] + 1);
                at += tokens.len();
            }
            let free = |span: &std::ops::Range<usize>| {
                span.end <= passed_over.start || span.start >= passed_over.end
            };
            let mut skipped = 0..0;
            if !inference.last && !segments.is_empty() {
                match rng.below(12) {
                    0 if spans.iter().all(free) => {
                        skipped = spans[0].start..spans[spans.len() - 1].end;
                        segments.clear();
                    }
                    1 => {
                        let at = rng.below(segments.len() as u64) as usize;
                        if free(&spans[at]) {
                            skipped = spans[at].clone();
                            segments.remove(at);
                        }
                    }
                    _ => {}
                }
            }
            passed_over = skipped;
            chunker.heard(inference, &segments);
        };

        let mut fed = 0;
        while fed < total_ms {
            let step = 1_000.min(total_ms - fed) as usize;
            fed += step as i64;
            if let Some(inference) = push_on(&mut chunker, step) {
                assert!(
                    inference.audio.len() <= samples(MAX_AUDIO_MS as usize + 4_000).len(),
                    "the audio heard stays inside Whisper's window"
                );
                hear(&mut chunker, &inference, &mut rng);
            }
            lines.extend(chunker.take_lines());
        }
        if let Some(inference) = chunker.flush() {
            hear(&mut chunker, &inference, &mut rng);
        }
        lines.extend(chunker.take_lines());

        // A word heard otherwise in a round that agreed with the one before
        // is committed as that round heard it: a word misheard, not lost.
        let heard: Vec<&str> = lines
            .iter()
            .flat_map(|line| {
                line.words
                    .iter()
                    .map(|word| word.text.trim_end_matches('x'))
            })
            .collect();
        let expected: Vec<String> = (0..300).map(|n| format!("w{n}")).collect();
        assert_eq!(heard, expected, "seed {seed}");
        for pair in lines.windows(2) {
            assert!(
                pair[1].start_ms >= pair[0].end_ms,
                "{:?} overlaps {:?}",
                pair[1],
                pair[0]
            );
        }
    }

    /// Lines end where a reader would want them to: at the end of a
    /// sentence, at a pause, and before they grow too long to read.
    #[test]
    fn lines_end_at_a_sentence_a_pause_or_a_length() {
        let mut chunker = Chunker::new(POLICY);
        let words = segment(&[
            (" 안녕하세요.", 0, 800),
            (" 오늘은", 900, 1_300),
            (" 비가", 2_500, 2_900),
            (" 옵니다", 2_900, 3_300),
        ]);
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(&first, std::slice::from_ref(&words));
        let last = chunker.flush().unwrap();
        chunker.heard(&last, &[words]);
        assert_eq!(
            texts(&chunker.take_lines()),
            ["안녕하세요.", "오늘은", "비가 옵니다"],
            "the sentence ends one, the pause of 1.2 s another"
        );

        let mut chunker = Chunker::new(POLICY);
        let long: Vec<(String, i64, i64)> = (0..12)
            .map(|n| (format!(" word{n:02}"), n * 300, n * 300 + 250))
            .collect();
        let spec: Vec<(&str, i64, i64)> =
            long.iter().map(|(t, s, e)| (t.as_str(), *s, *e)).collect();
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(&first, &[segment(&spec)]);
        let last = chunker.flush().unwrap();
        chunker.heard(&last, &[segment(&spec)]);
        let lines = chunker.take_lines();
        assert!(lines.len() > 1, "{:?}", texts(&lines));
        assert!(
            lines
                .iter()
                .all(|line| line.text.chars().count() <= LINE_MAX_CHARS + 7),
            "{:?}",
            texts(&lines)
        );
    }

    /// What a model hears of `segments` in audio ending at `end_ms`: the
    /// words said by then.
    fn by(segments: &[Vec<Heard>], end_ms: i64) -> Vec<Vec<Heard>> {
        segments
            .iter()
            .map(|tokens| {
                tokens
                    .iter()
                    .filter(|token| token.end_ms <= end_ms)
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .filter(|tokens| !tokens.is_empty())
            .collect()
    }

    /// Once the audio is long, what is committed is cut away after a
    /// segment's end, two committed words left after the cut, and given
    /// as the prompt instead — so the audio stays short and the model
    /// still knows what came before.
    #[test]
    fn long_audio_is_cut_inside_what_is_committed_and_becomes_the_prompt() {
        let mut chunker = Chunker::new(POLICY);
        let said = [
            segment(&[(" 하나", 1_000, 1_500), (" 둘.", 1_500, 2_000)]),
            segment(&[(" 셋", 5_000, 5_500), (" 넷.", 5_500, 6_000)]),
            segment(&[(" 다섯", 9_000, 9_500), (" 여섯", 9_500, 10_000)]),
        ];
        for _ in 0..4 {
            let round = push_on(&mut chunker, 4_000).unwrap();
            assert_eq!(round.offset_ms, 0, "not yet long enough to cut");
            let end = round.offset_ms + round.audio.len() as i64 * 1000 / 16_000;
            chunker.heard(&round, &by(&said, end));
        }
        let next = push_on(&mut chunker, 4_000).unwrap();
        assert_eq!(
            next.offset_ms, 7_500,
            "halfway between the segment's end and the word after it"
        );
        assert_eq!(next.prompt, " 하나 둘. 셋 넷.");
        assert_eq!(
            texts(&chunker.take_lines()),
            ["하나 둘.", "셋 넷.", "다섯 여섯"]
        );
    }

    /// The model sometimes says its prompt again before the committed words
    /// it hears again; all of it is passed over, and only what is new is
    /// reported.
    #[test]
    fn the_prompt_said_again_is_not_reported_again() {
        let mut chunker = Chunker::new(POLICY);
        let said = [
            segment(&[(" 알프스", 1_000, 1_600), (" 산맥.", 1_600, 2_200)]),
            segment(&[(" 스위스는", 9_000, 9_800), (" 작은", 9_800, 10_300)]),
        ];
        for _ in 0..4 {
            let round = push_on(&mut chunker, 4_000).unwrap();
            chunker.heard(&round, &said);
        }
        let round = push_on(&mut chunker, 4_000).unwrap();
        assert!(round.offset_ms > 2_200, "what was said first is cut away");
        assert_eq!(round.prompt, " 알프스 산맥.");
        chunker.take_lines();

        let echo = [segment(&[
            (" 알프스", round.offset_ms, round.offset_ms + 100),
            (" 산맥.", round.offset_ms + 100, round.offset_ms + 200),
            (" 스위스는", 9_000, 9_800),
            (" 작은", 9_800, 10_300),
            (" 나라", 10_300, 10_900),
        ])];
        chunker.heard(&round, &echo);
        let last = chunker.flush().unwrap();
        chunker.heard(&last, &echo);
        assert_eq!(texts(&chunker.take_lines()), ["나라"]);
    }

    /// Rounds that never agree — music, noise, a model looping — still
    /// commit at the longest the audio may grow, so it stays inside the
    /// thirty seconds Whisper can hear.
    #[test]
    fn rounds_that_never_agree_commit_at_the_longest_the_audio_may_grow() {
        let mut chunker = Chunker::new(POLICY);
        let mut forced = false;
        let unlike = [
            "사과", "하늘", "구름", "나무", "파도", "기차", "우산", "연필", "시계", "의자",
        ];
        for word in unlike {
            let round = push_on(&mut chunker, 4_000).unwrap();
            assert!(round.audio.len() <= samples(MAX_AUDIO_MS as usize + 4_000).len());
            let text = format!(" {word}");
            let words = [segment(&[(
                text.as_str(),
                round.offset_ms + 500,
                round.offset_ms + 900,
            )])];
            forced |= chunker.heard(&round, &words).forced;
        }
        assert!(forced);
        assert!(!chunker.take_lines().is_empty());
    }

    /// At the end there is no later round: what the last one hears is
    /// committed as it is heard, and a second flush has nothing to say.
    #[test]
    fn the_last_round_commits_what_it_hears_and_only_once() {
        let mut chunker = Chunker::new(POLICY);
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(&first, &[segment(&[(" 감사합니다", 3_000, 3_900)])]);
        push_on(&mut chunker, 500);
        let last = chunker.flush().expect("the tail is worth hearing");
        assert!(last.last);
        chunker.heard(&last, &[segment(&[(" 감사합니다", 3_000, 4_000)])]);
        assert_eq!(texts(&chunker.take_lines()), ["감사합니다"]);
        assert_eq!(
            chunker.settled_until_ms(),
            4_500,
            "everything heard is settled"
        );
        assert_eq!(chunker.flush(), None);
    }

    /// The audio a line in progress began in is not settled until the line
    /// is finished, so the line still has a buffer to go on.
    #[test]
    fn a_line_in_progress_holds_the_audio_it_began_in() {
        let mut chunker = Chunker::new(POLICY);
        let words = [segment(&[
            (" 그런데", 1_000, 1_500),
            (" 강이나", 1_500, 2_100),
        ])];
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(&first, &words);
        let second = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(
            &second,
            &[segment(&[
                (" 그런데", 1_000, 1_500),
                (" 강이나", 1_500, 2_100),
                (" 호수", 2_100, 2_600),
            ])],
        );
        assert!(chunker.take_lines().is_empty());
        assert_eq!(chunker.settled_until_ms(), 1_000);
    }

    /// Silence is agreed on as words are: two rounds that hear nothing let
    /// the audio go up to a second short of where the first of them ended,
    /// and one alone lets nothing go.
    #[test]
    fn silence_is_settled_once_two_rounds_hear_it() {
        let mut chunker = Chunker::new(POLICY);
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(&first, &[]);
        assert_eq!(chunker.settled_until_ms(), 0, "one round is not believed");
        let second = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(&second, &[]);
        assert_eq!(chunker.settled_until_ms(), 3_000);
    }

    /// The loss one round's silence made: whisper passed over a stretch of
    /// speech whole, the round heard nothing, and what the round before had
    /// heard there was cut away as silence. The next round hears it again,
    /// and it is reported.
    #[test]
    fn a_round_that_hears_nothing_loses_nothing() {
        let mut chunker = Chunker::new(POLICY);
        let said = [segment(&[
            (" 합금은", 13_000, 13_600),
            (" 혼합물이다.", 13_600, 14_500),
        ])];
        let mut lines = Vec::new();
        for n in 0..6 {
            let round = push_on(&mut chunker, 4_000).unwrap();
            let end = round.offset_ms + round.audio.len() as i64 * 1000 / 16_000;
            // The round ending at 20 s passes over everything.
            let heard = if n == 4 { Vec::new() } else { by(&said, end) };
            chunker.heard(&round, &heard);
            lines.extend(chunker.take_lines());
        }
        assert_eq!(
            texts(&lines),
            ["합금은 혼합물이다."],
            "agreed across the round that heard nothing"
        );
    }

    /// The whole reason positions come from the timestamps: audio that
    /// arrives late in its own timeline, because a queue dropped what was
    /// between, must not drag everything after it earlier. What the last
    /// round heard before the gap is reported as it was heard.
    #[test]
    fn a_gap_ends_what_was_heard_and_places_what_follows_where_it_is() {
        let mut chunker = Chunker::new(POLICY);
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(&first, &[segment(&[(" 사진이", 1_000, 1_700)])]);

        // Ten seconds went missing.
        let after = chunker
            .push(Some(at(14_000)), &samples(4_000))
            .expect("a whole chunk again");
        assert_eq!(texts(&chunker.take_lines()), ["사진이"]);
        assert_eq!(after.offset_ms, 14_000, "placed where its timestamps say");
        assert_eq!(
            after.audio.len(),
            samples(4_000).len(),
            "with nothing from before the gap in front of it"
        );
        assert!(chunker.settled_until_ms() >= 14_000);
    }

    /// After a seek the audio is placed where its own timestamps say, even
    /// earlier than before, and nothing heard before it is said.
    #[test]
    fn a_reset_forgets_and_the_next_audio_places_itself() {
        let mut chunker = Chunker::new(POLICY);
        let first = chunker.push(Some(at(60_000)), &samples(4_000)).unwrap();
        chunker.heard(&first, &[segment(&[(" 패키지", 61_000, 61_800)])]);
        chunker.reset();
        assert!(chunker.take_lines().is_empty());

        let after = chunker.push(Some(at(10_000)), &samples(4_000)).unwrap();
        assert_eq!(after.offset_ms, 10_000);
        assert!(after.prompt.is_empty());
        chunker.heard(&after, &[segment(&[(" 여행", 10_500, 11_000)])]);
        let last = chunker.flush().unwrap();
        chunker.heard(&last, &[segment(&[(" 여행", 10_500, 11_000)])]);
        let lines = chunker.take_lines();
        assert_eq!(texts(&lines), ["여행"]);
        assert_eq!(lines[0].start_ms, 10_500);
    }

    /// The model's word times are guesses, and a word cut away with the
    /// audio may still be heard at the start of what is left: it is passed
    /// over, not reported twice.
    #[test]
    fn a_word_cut_away_and_heard_again_is_not_reported_again() {
        let mut chunker = Chunker::new(POLICY);
        let said = [
            segment(&[(" 하나", 1_000, 1_500), (" 둘.", 1_500, 2_000)]),
            segment(&[(" 셋", 5_000, 5_500), (" 넷.", 5_500, 6_000)]),
            segment(&[(" 다섯", 9_000, 9_500), (" 여섯", 9_500, 10_000)]),
        ];
        for _ in 0..4 {
            let round = push_on(&mut chunker, 4_000).unwrap();
            let end = round.offset_ms + round.audio.len() as i64 * 1000 / 16_000;
            chunker.heard(&round, &by(&said, end));
        }
        let round = push_on(&mut chunker, 4_000).unwrap();
        assert!(round.offset_ms > 6_000, "넷 is cut away");
        chunker.take_lines();
        // The end of 넷 is still in the audio, and heard as the word.
        let heard = [
            segment(&[(" 넷.", round.offset_ms, round.offset_ms + 200)]),
            segment(&[
                (" 다섯", 9_000, 9_500),
                (" 여섯", 9_500, 10_000),
                (" 일곱", 18_000, 18_500),
            ]),
        ];
        chunker.heard(&round, &heard);
        let last = chunker.flush().unwrap();
        chunker.heard(&last, &heard);
        assert_eq!(texts(&chunker.take_lines()), ["일곱"]);
    }

    /// A model that makes up a line and then says it again, round after
    /// round, because the prompt now holds it: every round agrees, and the
    /// repeats are passed over instead, the prompt dropped.
    #[test]
    fn the_model_repeating_itself_is_passed_over_and_its_prompt_dropped() {
        let mut chunker = Chunker::new(POLICY);
        let words = ["이", "인공위성을", "찾을", "수", "있는", "기술을"];
        let looping = |offset: i64, times: usize| -> Vec<Vec<Heard>> {
            (0..times)
                .map(|n| {
                    let at = offset + 300 + n as i64 * 2_000;
                    words
                        .iter()
                        .enumerate()
                        .map(|(k, w)| Heard {
                            bytes: format!(" {w}").into_bytes(),
                            start_ms: at + k as i64 * 300,
                            end_ms: at + k as i64 * 300 + 280,
                        })
                        .collect()
                })
                .collect()
        };
        let mut looped = false;
        for n in 0..6 {
            let round = push_on(&mut chunker, 4_000).unwrap();
            looped |= chunker
                .heard(&round, &looping(round.offset_ms, 1 + n))
                .looped;
        }
        assert!(looped);
        let said: Vec<String> = chunker.take_lines().into_iter().map(|l| l.text).collect();
        assert!(said.len() <= 1, "said once at most: {said:?}");
        let next = push_on(&mut chunker, 4_000).unwrap();
        assert!(
            next.prompt.is_empty(),
            "the prompt that kept it going is gone"
        );
    }

    /// `ms` of sound, as loud as speech.
    fn loud(ms: usize) -> Vec<f32> {
        (0..ms * SAMPLE_RATE as usize / 1000)
            .map(|n| if n % 2 == 0 { 0.3 } else { -0.3 })
            .collect()
    }

    /// Two rounds that hear nothing in sound — whisper passing over a
    /// sentence begun near the end of what it heard — do not settle it as
    /// silence: the next round that hears it reports it where it is.
    #[test]
    fn rounds_that_hear_nothing_in_sound_do_not_settle_it() {
        let mut chunker = Chunker::new(POLICY);
        let first = chunker.push(None, &loud(4_000)).unwrap();
        chunker.heard(&first, &[]);
        let second = chunker.push(None, &loud(4_000)).unwrap();
        chunker.heard(&second, &[]);
        assert!(
            chunker.settled_until_ms() <= 0,
            "sound is not silence: {}",
            chunker.settled_until_ms()
        );
        let said = [segment(&[(" 미국", 1_000, 1_400), (" 체조", 1_400, 1_900)])];
        let third = chunker.push(None, &loud(4_000)).unwrap();
        chunker.heard(&third, &said);
        let last = chunker.flush().unwrap();
        chunker.heard(&last, &said);
        let lines = chunker.take_lines();
        assert_eq!(texts(&lines), ["미국 체조"]);
        assert_eq!(lines[0].start_ms, 1_000, "where it was said");
    }

    /// Seconds of sound after the last word a round heard are heard again on
    /// their own, without the prompt; a round that heard to its edge, or
    /// silence after its last word, is not.
    #[test]
    fn the_unheard_tail_of_sound_is_heard_again_on_its_own() {
        let mut chunker = Chunker::new(POLICY);
        let first = chunker.push(None, &loud(8_000)).unwrap();
        let heard = [segment(&[
            (" 입증되지", 500, 1_200),
            (" 않았습니다.", 1_200, 2_000),
        ])];
        let tail = chunker
            .tail(&first, &heard)
            .expect("six seconds of sound unheard");
        assert!(
            tail.offset_ms >= 2_000 && tail.offset_ms < 3_000,
            "{}",
            tail.offset_ms
        );
        assert!(tail.prompt.is_empty());
        assert_eq!(
            tail.audio.len() as i64,
            (8_000 - tail.offset_ms) * SAMPLE_RATE as i64 / 1000
        );
        let to_the_edge = [segment(&[(" 말", 500, 6_800)])];
        assert_eq!(chunker.tail(&first, &to_the_edge), None);

        let mut quiet = Chunker::new(POLICY);
        let mut audio = loud(2_000);
        audio.extend(samples(6_000));
        let round = quiet.push(None, &audio).unwrap();
        assert_eq!(
            quiet.tail(&round, &heard),
            None,
            "silence after the last word"
        );
    }

    /// A last round that passes over the end of the stream does not take
    /// with it what the round before heard there.
    #[test]
    fn a_last_round_hearing_less_keeps_what_the_round_before_heard() {
        let mut chunker = Chunker::new(POLICY);
        let first = push_on(&mut chunker, 4_000).unwrap();
        chunker.heard(
            &first,
            &[segment(&[
                (" 선수들이", 1_000, 1_600),
                (" 안전하고", 1_600, 2_400),
            ])],
        );
        push_on(&mut chunker, 500);
        let last = chunker.flush().unwrap();
        chunker.heard(&last, &[segment(&[(" 선수들이", 1_000, 1_600)])]);
        assert_eq!(texts(&chunker.take_lines()), ["선수들이 안전하고"]);
    }

    /// Words are cut where whisper's tokens mark them: at a leading space,
    /// with punctuation and the rest of a split character kept on the word
    /// before — and, in a script written without spaces, at every token.
    #[test]
    fn words_are_whole_and_unspaced_scripts_go_token_by_token() {
        let syllable = "스".as_bytes();
        let words = words_of(&[vec![
            Heard {
                bytes: [b" ".as_slice(), &syllable[..2]].concat(),
                start_ms: 0,
                end_ms: 100,
            },
            Heard {
                bytes: [&syllable[2..], "위스는".as_bytes()].concat(),
                start_ms: 100,
                end_ms: 400,
            },
            Heard {
                bytes: b",".to_vec(),
                start_ms: 400,
                end_ms: 450,
            },
            Heard {
                bytes: " 東京".as_bytes().to_vec(),
                start_ms: 500,
                end_ms: 700,
            },
            Heard {
                bytes: "です".as_bytes().to_vec(),
                start_ms: 700,
                end_ms: 900,
            },
        ]]);
        let raw: Vec<&str> = words.iter().map(|w| w.raw.as_str()).collect();
        assert_eq!(raw, [" 스위스는,", " 東京", "です"]);
        assert!(words[2].ends_segment && !words[0].ends_segment);
    }

    /// Two hearings agree past a word misheard in either, but not past two
    /// in a row, nor past a stretch one passed over.
    #[test]
    fn hearings_agree_past_one_misheard_word_only() {
        let words = |text: &str| -> Vec<HeardWord> {
            text.split(' ')
                .map(|word| HeardWord::new(format!(" {word}"), 0, 0))
                .collect()
        };
        let before = words("a b c d e f");
        assert_eq!(agreement(&words("a b c d e f g"), &before), 6);
        assert_eq!(agreement(&words("a x c d e f"), &before), 6, "b misheard");
        assert_eq!(agreement(&words("a x y d e f"), &before), 1, "two in a row");
        assert_eq!(
            agreement(&words("a b e f"), &before),
            2,
            "c and d passed over"
        );
        assert_eq!(
            agreement(&words("a x c y e f"), &before),
            6,
            "b and d, apart"
        );
        assert_eq!(
            agreement(&words("a b c d e x"), &before),
            5,
            "nothing after to agree"
        );
    }

    /// A word heard a little differently the second time is still the
    /// word it was; a different short word is not.
    #[test]
    fn words_heard_slightly_differently_are_alike() {
        assert!(alike("없기", "업기"));
        assert!(alike("발생하며", "발생하면"));
        assert!(!alike("더", "또"));
        assert!(!alike("오래", "지속하고"));
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
