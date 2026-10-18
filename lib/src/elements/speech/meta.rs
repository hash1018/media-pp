//! What a transcriber heard, carried on the audio it heard it in.

/// What was said, carried on the audio it was said in.
///
/// A transcriber puts one on each buffer of sound that begins a stretch of
/// speech, holding the sound until the speech in it is settled; anything
/// after it reads it back by its type. Read it with
/// `buffer.metadata().and_then(|m| m.get::<Transcript>())`.
///
/// # Which buffer carries a line
///
/// Each [`Utterance`] rides on the buffer its first word starts in, once
/// and only once, so an element that writes every line it sees writes each
/// line once. A line runs on past the end of that buffer, as speech does;
/// its own times, not the buffer's, say where it is. An element that acts on
/// the sound where a word is — a bleep — keeps the words it has seen until
/// the sound reaches them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    /// The lines that begin in this buffer, in the order they were said.
    pub utterances: Vec<Utterance>,
}

/// One line of speech, and when it was said.
///
/// The times are milliseconds on the timeline of the audio itself, taken
/// from the frames' own timestamps rather than counted from the first
/// sample that arrived. That is what keeps a line where it belongs when
/// audio is lost: a queue that drops buffers because the model is behind
/// leaves a hole rather than shifting everything after it earlier, for
/// ever, by what was dropped.
///
/// So a line muxed into the file the audio came from needs no offset. One
/// muxed beside a *different* timeline — a recording whose tracks are
/// rebased to start at zero, say — needs the same rebase applied here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utterance {
    /// When the line starts, in milliseconds.
    pub start_ms: i64,
    /// When it ends, in milliseconds. Never before `start_ms`; equal to it
    /// for a line the model gave no length.
    pub end_ms: i64,
    /// What was said, trimmed. Never empty — a line with nothing in it is
    /// not reported.
    pub text: String,
    /// The line's words, each with its own time, in order. A word is what
    /// lies between two spaces, so a language written without them comes
    /// out as one word per stretch the model wrote without one.
    pub words: Vec<Word>,
}

/// One word of an [`Utterance`], and when it was said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// When the word starts, in milliseconds on the audio's timeline.
    pub start_ms: i64,
    /// When it ends, in milliseconds. Never before `start_ms`.
    pub end_ms: i64,
    /// The word, without the spaces around it.
    pub text: String,
}
