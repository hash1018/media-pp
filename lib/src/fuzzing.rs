//! What the fuzz targets in `lib/fuzz` reach: the parsers that take bytes
//! from the network, fed whatever bytes. None may panic, loop without end
//! or ask for memory without bound on any input: a server reading a camera
//! or a publisher it does not control would stop, for everyone, on one.
//! Built only by `cargo fuzz`, which sets `--cfg fuzzing`; no part of the
//! API.

use crate::elements::parameter_sets;
use crate::elements::rtsp::depacketize::{self, Depacketizer};
use crate::elements::rtsp::sdp::{self, Payload};
use crate::elements::rtsp::{digest, message};

/// RTSP messages and interleaved packets, one after another, as a
/// connection reads them.
pub fn rtsp_messages(data: &[u8]) {
    let mut at = 0;
    while at < data.len() {
        match message::parse(&data[at..]) {
            Ok(Some((incoming, taken))) => {
                assert!(taken > 0, "a message that takes nothing is read forever");
                if let message::Incoming::Request(request) = &incoming {
                    let _ = request.header("CSeq");
                    #[cfg(feature = "rtsp-server")]
                    let _ = request.session();
                }
                at += taken;
            }
            Ok(None) | Err(_) => break,
        }
    }
}

/// What RTSP's header values and URLs are taken apart into.
pub fn rtsp_values(data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    let _ = message::transport(&text);
    let _ = message::delivery(&text);
    let _ = message::session(&text);
    let _ = message::address(&text);
    #[cfg(feature = "rtsp-server")]
    {
        let _ = message::target(&text);
        let _ = message::query_token(&text);
    }
    let _ = message::without_query(&text);
    let half = (0..=text.len() / 2)
        .rev()
        .find(|at| text.is_char_boundary(*at))
        .unwrap_or(0);
    let (base, control) = text.split_at(half);
    let _ = message::resolve(base, control);
    let _ = digest::parameters(&text);
}

/// A session description, as an RTSP server's `DESCRIBE` answers with or
/// a publisher's `ANNOUNCE` sends.
pub fn sdp(data: &[u8]) {
    let text = String::from_utf8_lossy(data);
    if let Ok(tracks) = sdp::parse(&text) {
        for track in &tracks {
            let _ = track.parameters();
            let _ = track.time_base();
        }
    }
    let _ = sdp::describe(&text);
}

/// RTP packets of one track taken apart: the first byte picks the
/// payload, the second AAC's header sizes, and each packet after is
/// behind a two-byte length.
pub fn rtp(data: &[u8]) {
    let [kind, sizes, rest @ ..] = data else {
        return;
    };
    let payload = match kind % 6 {
        0 => Payload::H264,
        1 => Payload::H265,
        2 => Payload::Aac {
            size_length: sizes >> 4,
            index_length: sizes & 0x0f,
        },
        3 => Payload::Whole,
        4 => Payload::Vp8,
        _ => Payload::Vp9,
    };
    let mut depacketizer = Depacketizer::new(payload);
    let mut rest = rest;
    while let [high, low, after @ ..] = rest {
        let length = usize::from(u16::from_be_bytes([*high, *low])).min(after.len());
        let (packet, more) = after.split_at(length);
        for unit in depacketizer.push(packet) {
            if let Some(data) = unit.data() {
                for nal in depacketize::nal_units(data) {
                    let _ = depacketize::key_unit(nal, payload == Payload::H265);
                }
            }
        }
        rest = more;
    }
}

/// An H.264 or H.265 sequence parameter set, as a stream carries it in
/// band, and an Annex-B stream split into its units.
pub fn parameter_sets(data: &[u8]) {
    let _ = parameter_sets::h264_shape(data);
    let _ = parameter_sets::h264_sequence(data);
    let _ = parameter_sets::hevc_shape(data);
    let _ = parameter_sets::hevc_sequence(data);
    let _ = parameter_sets::rbsp(data);
    for unit in parameter_sets::annex_b(data) {
        let _ = parameter_sets::h264_shape(unit);
    }
}

#[cfg(feature = "rtmp-server")]
pub use crate::elements::driver::rtmp_fuzz::rtmp;
