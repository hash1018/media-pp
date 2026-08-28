//! What travels between elements.
//!
//! [`MediaBuffer`] is an enum rather than one opaque buffer type, because
//! `ffmpeg-next` already hands back strongly-typed packets and frames and
//! collapsing them would only mean unwrapping again downstream. Its own
//! documentation covers why each payload is shared rather than copied, and
//! why a video frame arrives through a pool reference.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};

use crate::pool::UnboundObjectPoolRef;

/// The unit of data that flows between elements.
///
/// Compressed and uncompressed data are kept as distinct variants (rather
/// than a single opaque `Buffer` type like GStreamer) because ffmpeg-next
/// already gives us strongly-typed `Packet`/`Frame` types — collapsing them
/// into one type would just mean unwrapping again downstream.
///
/// Payloads are `Arc`-wrapped so `MediaBuffer` is cheaply `Clone` —
/// duplicating a buffer (e.g. [`crate::elements::Tee`] fanning packets out
/// to a decode branch and a remux branch) is a refcount bump, never a copy
/// of the encoded/decoded data.
///
/// `Video` specifically wraps an [`UnboundObjectPoolRef`], not a plain
/// `ffmpeg::frame::Video` — that's what lets whichever element produced it
/// (see [`crate::pool::UnboundObjectPool`], owned as that element's own
/// struct field) get the underlying buffer back automatically once every
/// `Arc` clone downstream has been dropped, instead of it just being freed.
#[derive(Clone)]
pub enum MediaBuffer {
    /// Encoded media packet produced by a demuxer or encoder.
    ///
    /// Its PTS, DTS, duration, stream index, and time base remain part of the
    /// packet contract while it travels through packet-level elements.
    Packet(Arc<ffmpeg::Packet>),

    /// Decoded video frame whose backing storage returns to its producer's
    /// [`crate::pool::UnboundObjectPool`] after the last `Arc` clone drops.
    ///
    /// Treat the published frame as immutable. A transforming element creates
    /// a replacement frame and preserves PTS, duration, and color metadata
    /// unless it intentionally establishes a new timeline.
    Video(Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>),

    /// Decoded audio frame shared immutably between downstream branches.
    ///
    /// Sample format, sample rate, channel layout, PTS, and duration describe
    /// the audio contract a transforming element must either preserve or
    /// deliberately replace.
    Audio(Arc<ffmpeg::frame::Audio>),

    /// Ordered end-of-stream marker.
    ///
    /// Stateful elements flush delayed output before forwarding it, and
    /// muxers finalize their output after receiving it. Unlike
    /// [`crate::control::ControlMsg::Stop`], this requests natural completion
    /// rather than abandoning buffered work.
    Eos,
}

impl MediaBuffer {
    /// Returns whether this buffer is the ordered [`MediaBuffer::Eos`] marker.
    pub fn is_eos(&self) -> bool {
        matches!(self, MediaBuffer::Eos)
    }

    /// Stable, human-readable variant name for diagnostics emitted when
    /// elements are wired to an incompatible media type.
    pub fn kind(&self) -> &'static str {
        match self {
            MediaBuffer::Packet(_) => "Packet",
            MediaBuffer::Video(_) => "Video",
            MediaBuffer::Audio(_) => "Audio",
            MediaBuffer::Eos => "Eos",
        }
    }
}

/// Which buffer a video frame's pixels live in.
///
/// Not the frame: a producer with nothing new to show re-emits a fresh
/// `AVFrame` referencing the same picture every tick — a screen capture of a
/// still desktop does exactly that — so comparing frames, or the `Arc`s
/// around them, answers "changed" every time while the pixels have not
/// moved. The plane pointers do not.
///
/// The first two planes are enough for every layout this crate carries:
/// packed formats use one, and the semi-planar and planar ones this crate
/// composites in differ in the first two whenever they differ at all.
///
/// Only sound as an identity while the frame it came from is still
/// referenced. A picture whose buffer has been released can be handed out
/// again at the same address, so every caller here holds that reference for
/// as long as it holds the identity.
pub(crate) fn picture_id(frame: &ffmpeg::frame::Video) -> (usize, usize) {
    // SAFETY: `as_ptr` is a live `AVFrame`. Only the values of the first two
    // plane pointers are read; nothing dereferences them, which for GPU
    // memory would not be valid from the host anyway.
    unsafe {
        let ptr = frame.as_ptr();
        ((*ptr).data[0] as usize, (*ptr).data[1] as usize)
    }
}

/// Lets go of the picture a pooled wrapper was pointing at, as it returns to
/// its pool.
///
/// The `release` an [`UnboundObjectPool`](crate::pool::UnboundObjectPool) of
/// *wrappers* wants: an empty frame is given a picture with `av_frame_ref` on
/// every checkout, and without this it would keep that reference until its
/// next one. Nothing reads those pixels in the meantime, but everything that
/// asks whether a picture is still in use — [`picture_is_referenced`] — would
/// go on answering yes, so a producer would keep a buffer, or a screen-sized
/// texture, alive for an idle wrapper.
///
/// Only for a pool whose items own no picture of their own. A pool of real
/// frames composited or decoded into would be emptied by this.
pub(crate) fn release_picture(frame: &mut ffmpeg::frame::Video) {
    // SAFETY: `as_mut_ptr` is this frame's own live `AVFrame`, and the pool
    // has taken it back, so nothing else refers to it.
    unsafe { ffi::av_frame_unref(frame.as_mut_ptr()) }
}

/// Whether anything besides this frame itself still points at its picture.
///
/// The companion to [`picture_id`], for an element that offers an unchanged
/// picture again by pointing an empty wrapper at it with `av_frame_ref`.
/// Such a wrapper shares the picture's *buffer*, not the pool slot the frame
/// came from, so an [`UnboundObjectPoolRef`] that has gone back to its pool
/// says nothing about whether a wrapper downstream is still showing those
/// pixels — the buffer's own reference count is the only record of it, and
/// a producer that recycles its frames has to keep one out of the pool until
/// this reads false for it.
///
/// Only the first buffer is examined, for the same reason [`picture_id`]
/// reads only the first plane pointers: a wrapper references either all of a
/// frame's buffers or none of them.
pub(crate) fn picture_is_referenced(frame: &ffmpeg::frame::Video) -> bool {
    // SAFETY: `as_ptr` is a live `AVFrame`. `buf[0]` is either null (a frame
    // that owns no picture) or a live `AVBufferRef`, and reading its count
    // does not touch the picture itself — which for GPU memory would not be
    // valid from the host anyway.
    unsafe {
        let buf = (*frame.as_ptr()).buf[0];
        !buf.is_null() && ffi::av_buffer_get_ref_count(buf) > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_reports_each_variant() {
        assert_eq!(
            MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())).kind(),
            "Packet"
        );
        assert_eq!(
            MediaBuffer::Audio(Arc::new(ffmpeg::frame::Audio::empty())).kind(),
            "Audio"
        );
        assert_eq!(MediaBuffer::Eos.kind(), "Eos");
    }
}
