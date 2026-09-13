//! Timed text as this crate carries it: packets for a subtitle track, in
//! whichever of three codecs the destination takes.
//!
//! Kept in `core` rather than under an element because nothing here is one.
//! A subtitle is already text by the time it is a packet, so there is
//! nothing to decode and nothing to transform — what this module does is
//! build the two things a muxer needs, and both are pure functions of what
//! the caller already has.
//!
//! # Where each one goes
//!
//! | [`Codec`] | written to |
//! |---|---|
//! | [`MovText`](Codec::MovText) | an MP4 track — MP4 takes nothing else |
//! | [`SubRip`](Codec::SubRip) | an `.srt` file, or a Matroska track |
//! | [`WebVtt`](Codec::WebVtt) | a `.vtt` file, or a Matroska track |
//!
//! A file of its own is a [`FileMuxer`](crate::elements::FileMuxer) of its
//! own, with this one track in it: FFmpeg picks the SubRip or WebVTT writer
//! from the extension the way it picks MP4 from `.mp4`, and each of them
//! takes exactly one text stream. One line, one packet, whichever it is —
//! the codec is chosen where the track is registered and nowhere else.
//!
//! A sidecar file is written as the lines arrive, where an MP4's text track
//! is indexed only when the file is finished. A recording that dies
//! halfway leaves an `.srt` holding every line up to that moment.
//!
//! # `mov_text`: MP4's own
//!
//! MP4 does not take SRT or ASS. Its subtitle track is 3GPP Timed Text,
//! written as a `tx3g` sample entry — FFmpeg calls the codec `mov_text`,
//! which is the same thing under the container's name. Converting an ASS
//! subtitle into one throws away nearly all of its styling, which is why a
//! caller that has styled subtitles and wants to keep them should be
//! writing Matroska instead.
//!
//! A sample is a length and the text, and that is the whole format:
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
//! A `tx3g` track is continuous: every instant between the first sample and
//! the last belongs to some sample. That does not mean a caller emits
//! "clear the subtitle" packets — FFmpeg's MP4 muxer inserts an empty
//! sample wherever one line ends before the next begins. Push a packet when
//! there is something to say and nothing when there is not.
//!
//! # SubRip and WebVTT: the text itself
//!
//! A packet of either is the line's text and nothing else; the writer puts
//! the cue number and the times around it from the packet's own timestamps.
//! Each cue ends at the first blank line, so a blank line inside the text
//! would end it early and leave the rest as a cue with no times — they are
//! taken out. WebVTT also reads its text as markup, so `&`, `<` and `>` are
//! written as the entities that stand for them.

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

/// The codec a subtitle track is written in — which is settled by where it
/// is going. See the [module](self) for which goes where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// 3GPP Timed Text, for an MP4 track.
    MovText,
    /// SubRip, for an `.srt` file or a Matroska track.
    SubRip,
    /// WebVTT, for a `.vtt` file or a Matroska track.
    WebVtt,
}

impl Codec {
    /// The stream description a track in this codec is registered with.
    ///
    /// Hand this to
    /// [`FileMuxer::add_stream`](crate::elements::FileMuxer::add_stream)
    /// alongside the time base the packets' timestamps are in, the same way
    /// a video or audio track is registered with its encoder's parameters.
    /// There is no encoder here to ask, because [`Codec::packet`] is the
    /// encoder.
    ///
    /// A `mov_text` track's defaults travel with it as extradata — centred
    /// along the bottom, white on black — so the track says how its lines
    /// should be drawn rather than leaving it to the player. SubRip and
    /// WebVTT have no such header to fill.
    #[must_use]
    pub fn parameters(self) -> ffmpeg::codec::Parameters {
        use ffmpeg::sys::AVCodecID;

        let mut parameters = ffmpeg::codec::Parameters::new();
        // SAFETY: `Parameters::new` allocates a zeroed `AVCodecParameters`
        // and owns it for its whole life. `codec_type`/`codec_id` are plain
        // scalar fields of that struct. `extradata` is allocated with
        // FFmpeg's own allocator and handed over, which is how FFmpeg
        // expects to receive it: `avcodec_parameters_free` releases it along
        // with the parameters, so ownership passes with the write and
        // nothing here has to free it.
        //
        // The padding is `AV_INPUT_BUFFER_PADDING_SIZE`, which every FFmpeg
        // extradata buffer carries so that a reader may over-read the end
        // without leaving the allocation.
        unsafe {
            let raw = parameters.as_mut_ptr();
            (*raw).codec_type = ffmpeg::sys::AVMediaType::AVMEDIA_TYPE_SUBTITLE;
            (*raw).codec_id = match self {
                Self::MovText => AVCodecID::AV_CODEC_ID_MOV_TEXT,
                Self::SubRip => AVCodecID::AV_CODEC_ID_SUBRIP,
                Self::WebVtt => AVCodecID::AV_CODEC_ID_WEBVTT,
            };

            if self == Self::MovText {
                let padding = ffmpeg::sys::AV_INPUT_BUFFER_PADDING_SIZE as usize;
                let buffer = ffmpeg::sys::av_mallocz(TX3G.len() + padding).cast::<u8>();
                if !buffer.is_null() {
                    std::ptr::copy_nonoverlapping(TX3G.as_ptr(), buffer, TX3G.len());
                    (*raw).extradata = buffer;
                    (*raw).extradata_size = TX3G.len() as i32;
                }
            }
        }
        parameters
    }

    /// One line of timed text, as a packet a muxer will accept.
    ///
    /// `pts` and `duration` are in whatever time base the track was
    /// registered with — this does not know it, and does not convert. A line
    /// meant to stay on screen for two seconds has a `duration` saying so;
    /// one with none is a line that vanishes as it appears.
    ///
    /// `dts` is set equal to `pts`, which is the only thing it can be: text
    /// does not reference other text, so there is nothing to reorder and no
    /// decode order distinct from presentation order.
    #[must_use]
    pub fn packet(self, text: &str, pts: i64, duration: i64) -> MediaBuffer {
        let payload = match self {
            Self::MovText => mov_text_sample(text),
            Self::SubRip => cue_text(text).into_bytes(),
            Self::WebVtt => escape_markup(&cue_text(text)).into_bytes(),
        };
        let mut packet = ffmpeg::Packet::copy(&payload);
        packet.set_pts(Some(pts));
        packet.set_dts(Some(pts));
        packet.set_duration(duration);
        MediaBuffer::Packet(Arc::new(packet))
    }
}

/// A `mov_text` sample: the text's length in bytes, then the text, with no
/// style boxes after it, so every line is drawn the way the track's own
/// defaults say.
fn mov_text_sample(text: &str) -> Vec<u8> {
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
        &bytes[..end]
    } else {
        bytes
    };

    let mut payload = Vec::with_capacity(2 + bytes.len());
    payload.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    payload.extend_from_slice(bytes);
    payload
}

/// The text of one SubRip or WebVTT cue: its lines with every blank one
/// taken out, since the first blank line is where a cue ends.
fn cue_text(text: &str) -> String {
    text.lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// WebVTT's cue text is markup — `<` opens a tag and `&` an entity — so the
/// three characters that could be taken for either are written as the
/// entities that stand for them. That also covers `-->`, which WebVTT
/// forbids in a cue's text.
fn escape_markup(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            other => escaped.push(other),
        }
    }
    escaped
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
        let buffer = Codec::MovText.packet("hi", 0, 1000);
        assert_eq!(payload(&buffer), vec![0x00, 0x02, b'h', b'i']);
    }

    /// The length is in bytes and not in characters, which for anything
    /// outside ASCII is not the same number.
    #[test]
    fn the_length_counts_bytes_rather_than_characters() {
        let buffer = Codec::MovText.packet("한글", 0, 1000);
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
        let buffer = Codec::MovText.packet("hi", 4_000, 2_500);
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
        let buffer = Codec::MovText.packet(&long, 0, 1000);
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
        let parameters = Codec::MovText.parameters();
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

    /// A SubRip or WebVTT packet is the text itself, and the codec each is
    /// registered as is the one its writer asks for.
    #[test]
    fn subrip_and_webvtt_packets_are_the_text_itself() {
        assert_eq!(
            payload(&Codec::SubRip.packet("한글", 0, 1000)),
            "한글".as_bytes()
        );
        assert_eq!(
            payload(&Codec::WebVtt.packet("한글", 0, 1000)),
            "한글".as_bytes()
        );
        assert_eq!(Codec::SubRip.parameters().id(), ffmpeg::codec::Id::SUBRIP);
        assert_eq!(Codec::WebVtt.parameters().id(), ffmpeg::codec::Id::WEBVTT);
    }

    /// A blank line is where a cue ends, so one inside the text would cut
    /// the cue short and leave the rest outside any cue at all.
    #[test]
    fn a_blank_line_inside_a_cue_is_taken_out() {
        let text = "first line\n\n  \r\nsecond line\r\n";
        assert_eq!(
            payload(&Codec::SubRip.packet(text, 0, 1000)),
            b"first line\nsecond line"
        );
    }

    /// WebVTT reads its cue text as markup, so the characters that could be
    /// taken for a tag or an entity are written as entities — `-->` among
    /// them, which WebVTT forbids in a cue's text.
    #[test]
    fn webvtt_text_is_escaped_as_markup() {
        assert_eq!(
            payload(&Codec::WebVtt.packet("A & B <i> --> C", 0, 1000)),
            b"A &amp; B &lt;i&gt; --&gt; C"
        );
        assert_eq!(
            payload(&Codec::SubRip.packet("A & B", 0, 1000)),
            b"A & B",
            "while SubRip has no escaping to do"
        );
    }

    /// A sidecar file, written the way a video file is: a `FileMuxer` of its
    /// own, one track registered, one packet a line. What lands on disk is
    /// what a player reads, so that is what is checked.
    fn write_sidecar(extension: &str, codec: Codec) -> String {
        let path = std::env::temp_dir().join(format!(
            "media_pp_subtitle_{}_{:?}.{extension}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut muxer = crate::elements::FileMuxer::create(&path).expect("the muxer must open");
        let text = muxer
            .add_stream("text", codec.parameters(), ffmpeg::Rational::new(1, 1000))
            .expect("one text track");
        let mut sink = muxer
            .open()
            .expect("open must write the header")
            .take(text)
            .expect("its sink");
        sink.consume(codec.packet("첫 번째 자막", 1_000, 1_660))
            .expect("first line");
        sink.consume(codec.packet("second line", 2_660, 1_340))
            .expect("second line");
        sink.consume(MediaBuffer::Eos).expect("eos closes the file");

        let written = std::fs::read_to_string(&path).expect("a text file");
        std::fs::remove_file(&path).ok();
        written
    }

    #[test]
    fn an_srt_file_is_written_through_a_file_muxer() {
        crate::init().expect("ffmpeg");
        let written = write_sidecar("srt", Codec::SubRip);
        for expected in [
            "1\n00:00:01,000 --> 00:00:02,660\n첫 번째 자막\n",
            "2\n00:00:02,660 --> 00:00:04,000\nsecond line\n",
        ] {
            assert!(
                written.contains(expected),
                "{expected:?} not in:\n{written}"
            );
        }
    }

    #[test]
    fn a_vtt_file_is_written_through_a_file_muxer() {
        crate::init().expect("ffmpeg");
        let written = write_sidecar("vtt", Codec::WebVtt);
        assert!(
            written.starts_with("WEBVTT"),
            "not a WebVTT file:\n{written}"
        );
        for expected in [
            "00:01.000 --> 00:02.660\n첫 번째 자막\n",
            "00:02.660 --> 00:04.000\nsecond line\n",
        ] {
            assert!(
                written.contains(expected),
                "{expected:?} not in:\n{written}"
            );
        }
    }

    /// Matroska takes SubRip as a track of its own, which is the claim this
    /// module's table makes — so it is read back rather than taken on trust.
    #[test]
    fn matroska_carries_a_subrip_track() {
        crate::init().expect("ffmpeg");
        let path = std::env::temp_dir().join(format!(
            "media_pp_subtitle_{}_{:?}.mkv",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut muxer = crate::elements::FileMuxer::create(&path).expect("the muxer must open");
        let text = muxer
            .add_stream(
                "text",
                Codec::SubRip.parameters(),
                ffmpeg::Rational::new(1, 1000),
            )
            .expect("one text track");
        let mut sink = muxer.open().expect("header").take(text).expect("its sink");
        sink.consume(Codec::SubRip.packet("첫 번째 자막", 1_000, 1_660))
            .expect("a line");
        sink.consume(MediaBuffer::Eos).expect("eos");

        let mut input = ffmpeg::format::input(&path).expect("a readable file");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Subtitle)
            .expect("a subtitle track");
        assert_eq!(stream.parameters().id(), ffmpeg::codec::Id::SUBRIP);
        let index = stream.index();
        let texts: Vec<Vec<u8>> = input
            .packets()
            .filter(|(stream, _)| stream.index() == index)
            .filter_map(|(_, packet)| packet.data().map(<[u8]>::to_vec))
            .collect();
        assert_eq!(texts, vec!["첫 번째 자막".as_bytes().to_vec()]);
        std::fs::remove_file(&path).ok();
    }
}
