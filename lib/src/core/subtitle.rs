//! Timed text as this crate carries it: `mov_text` packets for an MP4
//! subtitle track.
//!
//! Kept in `core` rather than under an element because nothing here is one.
//! A subtitle is already text by the time it is a packet, so there is
//! nothing to decode and nothing to transform — what this module does is
//! build the two things a muxer needs, and both are pure functions of what
//! the caller already has.
//!
//! # Why `mov_text` and nothing else
//!
//! MP4 does not take SRT or ASS. Its subtitle track is 3GPP Timed Text,
//! written as a `tx3g` sample entry — FFmpeg calls the codec `mov_text`,
//! which is the same thing under the container's name. Converting an ASS
//! subtitle into one throws away nearly all of its styling, which is why a
//! caller that has styled subtitles and wants to keep them should be
//! writing Matroska instead.
//!
//! Only MP4 is served here because only MP4 needs help: Matroska takes SRT
//! and ASS as they are, so a caller muxing one has nothing to build.
//!
//! # What a sample looks like
//!
//! A length and the text, and that is the whole format:
//!
//! ```text
//! 00 12   "첫 번째 자막입니다"
//!  ↑      ↑
//! u16be   UTF-8
//! ```
//!
//! Style boxes may follow the text — `styl` for bold/italic/colour runs —
//! and this module writes none, because the defaults in the sample entry
//! already say what every line should look like.
//!
//! # Gaps are the muxer's to fill
//!
//! A `tx3g` track is continuous: every instant between the first sample and
//! the last belongs to some sample. That does not mean a caller emits
//! "clear the subtitle" packets — FFmpeg's MP4 muxer inserts an empty
//! sample wherever one line ends before the next begins. Push a packet when
//! there is something to say and nothing when there is not.

use std::sync::Arc;

use ffmpeg_next as ffmpeg;

use crate::buffer::MediaBuffer;

/// The `tx3g` sample entry body, byte for byte what FFmpeg's own `mov_text`
/// encoder writes.
///
/// This is the track's defaults — where lines are placed and what they look
/// like — stated once in the header rather than on every sample. Centred
/// along the bottom, white 16pt Arial on an opaque black background, which
/// is what a caption looks like everywhere.
///
/// A muxer that receives no extradata writes only its own `btrt` box and
/// leaves all of this unstated, which players are then free to interpret as
/// they like. Supplying it is what makes the track say how it should look.
///
/// The trailing `btrt` box FFmpeg appends itself, so this stops before it.
#[rustfmt::skip]
const TX3G: [u8; 48] = [
    // displayFlags: no scroll-in, no scroll-out, not written vertically.
    0x00, 0x00, 0x00, 0x00,
    // horizontal-justification: centre. vertical-justification: bottom.
    0x01, 0xff,
    // background-color-rgba: opaque black.
    0x00, 0x00, 0x00, 0xff,
    // default-text-box: top, left, bottom, right — all zero, meaning the
    // whole of the video track's own frame.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // default style: startChar and endChar, both zero because it applies to
    // every character; then font-ID 1, which the table below names.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    // face-style-flags: none. font-size: 16.
    0x00, 0x10,
    // text-color-rgba: opaque white.
    0xff, 0xff, 0xff, 0xff,
    // FontTableBox: 18 bytes, one entry, font 1, five characters, "Arial".
    0x00, 0x00, 0x00, 0x12, b'f', b't', b'a', b'b',
    0x00, 0x01, 0x00, 0x01, 0x05, b'A', b'r', b'i', b'a', b'l',
];

/// The stream description an MP4 subtitle track is registered with.
///
/// Hand this to [`FileMuxer::add_stream`](crate::elements::FileMuxer::add_stream)
/// alongside the time base the packets' timestamps are in, the same way a
/// video or audio track is registered with its encoder's parameters. There
/// is no encoder here to ask, because [`packet`] is the encoder: a
/// `mov_text` sample is a length and some text.
///
/// The track's defaults travel with it as extradata — centred along the
/// bottom, white on black — so the track says how its lines should be drawn
/// rather than leaving it to the player.
#[must_use]
pub fn mov_text_parameters() -> ffmpeg::codec::Parameters {
    let mut parameters = ffmpeg::codec::Parameters::new();
    // SAFETY: `Parameters::new` allocates a zeroed `AVCodecParameters` and
    // owns it for its whole life. `codec_type`/`codec_id` are plain scalar
    // fields of that struct. `extradata` is allocated with FFmpeg's own
    // allocator and handed over, which is how FFmpeg expects to receive it:
    // `avcodec_parameters_free` releases it along with the parameters, so
    // ownership passes with the write and nothing here has to free it.
    //
    // The padding is `AV_INPUT_BUFFER_PADDING_SIZE`, which every FFmpeg
    // extradata buffer carries so that a reader may over-read the end
    // without leaving the allocation.
    unsafe {
        let raw = parameters.as_mut_ptr();
        (*raw).codec_type = ffmpeg::sys::AVMediaType::AVMEDIA_TYPE_SUBTITLE;
        (*raw).codec_id = ffmpeg::sys::AVCodecID::AV_CODEC_ID_MOV_TEXT;

        let padding = ffmpeg::sys::AV_INPUT_BUFFER_PADDING_SIZE as usize;
        let buffer = ffmpeg::sys::av_mallocz(TX3G.len() + padding).cast::<u8>();
        if !buffer.is_null() {
            std::ptr::copy_nonoverlapping(TX3G.as_ptr(), buffer, TX3G.len());
            (*raw).extradata = buffer;
            (*raw).extradata_size = TX3G.len() as i32;
        }
    }
    parameters
}

/// One line of timed text, as a packet a muxer will accept.
///
/// `pts` and `duration` are in whatever time base the track was registered
/// with — this does not know it, and does not convert. A line that is meant
/// to stay on screen for two seconds has a `duration` saying so; a `tx3g`
/// sample with no duration is a line that vanishes immediately.
///
/// `dts` is set equal to `pts`, which is the only thing it can be: text
/// does not reference other text, so there is nothing to reorder and no
/// decode order distinct from presentation order.
///
/// The text is written as UTF-8 with no style boxes after it, so every line
/// is drawn the way the track's own defaults say.
#[must_use]
pub fn packet(text: &str, pts: i64, duration: i64) -> MediaBuffer {
    let bytes = text.as_bytes();
    // A `tx3g` sample states its own length before its text, and that
    // length is 16-bit, so a line longer than 65535 bytes cannot be
    // expressed. Truncating on a character boundary keeps the text valid
    // UTF-8; a subtitle that long is a caller's mistake rather than a
    // condition worth an error, since nothing could display it either.
    let bytes = if bytes.len() > u16::MAX as usize {
        let mut end = u16::MAX as usize;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        &text.as_bytes()[..end]
    } else {
        bytes
    };

    let mut payload = Vec::with_capacity(2 + bytes.len());
    payload.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    payload.extend_from_slice(bytes);

    let mut packet = ffmpeg::Packet::copy(&payload);
    packet.set_pts(Some(pts));
    packet.set_dts(Some(pts));
    packet.set_duration(duration);
    MediaBuffer::Packet(Arc::new(packet))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(buffer: &MediaBuffer) -> Vec<u8> {
        let MediaBuffer::Packet(packet) = buffer else {
            panic!("a subtitle is a packet");
        };
        packet.data().expect("a packet with a payload").to_vec()
    }

    #[test]
    fn a_sample_is_its_length_then_its_text() {
        let buffer = packet("hi", 0, 1000);
        assert_eq!(payload(&buffer), vec![0x00, 0x02, b'h', b'i']);
    }

    /// The length is in bytes and not in characters, which for anything
    /// outside ASCII is not the same number.
    #[test]
    fn the_length_counts_bytes_rather_than_characters() {
        let buffer = packet("한글", 0, 1000);
        let payload = payload(&buffer);
        assert_eq!(&payload[..2], &[0x00, 0x06], "two characters, six bytes");
        assert_eq!(
            std::str::from_utf8(&payload[2..]).expect("valid utf-8"),
            "한글"
        );
    }

    /// Text does not reference other text, so there is no decode order
    /// distinct from presentation order and a muxer must not be handed a
    /// packet that implies otherwise.
    #[test]
    fn a_line_carries_its_own_time_and_no_reordering() {
        let buffer = packet("hi", 4_000, 2_500);
        let MediaBuffer::Packet(packet) = &buffer else {
            panic!("a subtitle is a packet");
        };
        assert_eq!(packet.pts(), Some(4_000));
        assert_eq!(packet.dts(), Some(4_000));
        assert_eq!(packet.duration(), 2_500);
    }

    /// A 16-bit length cannot describe a longer line, and cutting it in the
    /// middle of a character would leave the sample holding text no reader
    /// can decode.
    #[test]
    fn an_over_long_line_is_cut_on_a_character_boundary() {
        // Three bytes apiece, so the limit falls inside a character.
        let long = "한".repeat(30_000);
        let buffer = packet(&long, 0, 1000);
        let payload = payload(&buffer);

        let length = u16::from_be_bytes([payload[0], payload[1]]) as usize;
        assert_eq!(length, payload.len() - 2, "the stated length is the truth");
        assert!(length <= u16::MAX as usize);
        std::str::from_utf8(&payload[2..]).expect("still valid utf-8");
    }

    /// The defaults are what make a player draw the line where a caption
    /// belongs, so a track registered without them is a track that looks
    /// like whatever the player felt like.
    #[test]
    fn the_stream_description_carries_the_tx3g_defaults() {
        let parameters = mov_text_parameters();
        assert_eq!(parameters.medium(), ffmpeg::media::Type::Subtitle);
        assert_eq!(
            parameters.id(),
            ffmpeg::codec::Id::MOV_TEXT,
            "MP4 takes 3GPP Timed Text and nothing else"
        );

        // SAFETY: reading back the two fields written above, from
        // parameters this test owns and has not moved.
        let extradata = unsafe {
            let raw = parameters.as_ptr();
            std::slice::from_raw_parts((*raw).extradata, (*raw).extradata_size as usize)
        };
        assert_eq!(
            extradata, TX3G,
            "byte for byte what a mov_text encoder writes"
        );
    }
}
