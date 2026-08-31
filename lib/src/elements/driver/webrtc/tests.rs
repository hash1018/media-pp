use std::{
    net::UdpSocket,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::PpLog;
use crossbeam_channel::{Receiver, bounded, unbounded};
use ffmpeg_next as ffmpeg;
use str0m::{
    Candidate, Rtc,
    change::SdpOffer,
    format::{Codec, CodecSpec, FormatParams},
    media::{Direction, Frequency, MediaKind, Mid},
};

use super::command::Command;
use super::peer::packet_rtp_time;
use super::{
    AttachedTrack, TrackEndpoints, TrackId, WebRtcError, WebRtcHandle, WebRtcPeer,
    WebRtcStreamInfo, WebRtcTrackSink, WebRtcTrackSource,
};
use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    control::ControlMsg,
    driver::DriverRunner,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::{
        FrameCounter, SwDecoder, SwEncoder, SwEncoderOptions, TestVideoOptions, TestVideoSource,
        VideoCodec,
    },
    error::Result,
    pipeline::Pipeline,
};

fn command_only_handle(capacity: usize) -> (WebRtcHandle, Receiver<Command>) {
    let (command_tx, command_rx) = bounded(capacity);
    let (_new_track_tx, new_track_rx) = unbounded();
    (
        WebRtcHandle {
            next_id: Arc::new(AtomicU64::new(0)),
            command_tx,
            new_track_rx,
        },
        command_rx,
    )
}

#[test]
fn add_track_returns_the_id_that_was_enqueued() {
    let (handle, command_rx) = command_only_handle(1);

    let returned = handle
        .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
        .expect("live command receiver should accept AddTrack");
    let Command::AddTrack(enqueued, kind, direction, codec) =
        command_rx.recv().expect("AddTrack should be queued")
    else {
        panic!("expected AddTrack command");
    };

    assert_eq!(returned, enqueued);
    assert_eq!(kind, MediaKind::Video);
    assert_eq!(direction, Direction::SendRecv);
    assert_eq!(codec, Codec::Vp8);
}

#[test]
fn add_track_blocks_for_backpressure_then_unblocks_when_capacity_opens() {
    let (handle, command_rx) = command_only_handle(1);
    handle
        .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
        .expect("first command should fill the queue");

    let (entered_tx, entered_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let blocked_handle = handle.clone();
    let worker = thread::spawn(move || {
        entered_tx.send(()).expect("test receiver alive");
        let result = blocked_handle.add_track(MediaKind::Audio, Direction::SendOnly, Codec::Opus);
        done_tx.send(result).expect("test receiver alive");
    });

    entered_rx.recv().expect("worker should start add_track");
    assert!(
        done_rx.recv_timeout(Duration::from_millis(50)).is_err(),
        "second AddTrack must wait while the bounded queue is full"
    );

    let _first = command_rx.recv().expect("free one queue slot");
    let second_id = done_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("add_track should unblock once capacity opens")
        .expect("command receiver is still alive");
    let Command::AddTrack(enqueued_id, ..) =
        command_rx.recv().expect("second AddTrack should be queued")
    else {
        panic!("expected AddTrack command");
    };
    assert_eq!(second_id, enqueued_id);
    worker.join().expect("worker should finish cleanly");
}

#[test]
fn add_track_returns_closed_instead_of_a_phantom_id() {
    let (handle, command_rx) = command_only_handle(1);
    drop(command_rx);

    let error = handle
        .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
        .expect_err("closed peer must not yield a TrackId");
    assert!(matches!(
        error,
        crate::Error::WebRtcError(WebRtcError::Closed)
    ));
}

#[test]
fn a_remote_track_sink_rejects_packets_until_its_codec_is_declared() {
    let (handle, _command_rx) = command_only_handle(1);
    let negotiated = Arc::new(std::sync::Mutex::new(vec![Codec::Vp8]));
    let mut sink = WebRtcTrackSink::new(
        TrackId(7),
        MediaKind::Video,
        None,
        negotiated,
        handle.command_tx.clone(),
    );
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
    packet.set_pts(Some(0));

    let error = sink
        .consume(MediaBuffer::Packet(Arc::new(packet)))
        .expect_err("a remote endpoint must not guess its outbound payload type");

    assert!(matches!(
        error,
        crate::Error::WebRtcError(WebRtcError::OutboundCodecNotDeclared(TrackId(7)))
    ));
}

/// Codec parameters naming `id` and carrying `extradata`, the way an
/// encoder's or a demuxer's `parameters()` does.
fn parameters_for(id: ffmpeg::codec::Id, extradata: &[u8]) -> ffmpeg::codec::Parameters {
    let mut parameters = ffmpeg::codec::Parameters::new();
    // SAFETY: `parameters` is a live `AVCodecParameters` this test owns, and
    // the allocation it is given is FFmpeg's to free with it.
    unsafe {
        let raw = parameters.as_mut_ptr();
        (*raw).codec_id = id.into();
        if !extradata.is_empty() {
            let padded = extradata.len() + ffmpeg::ffi::AV_INPUT_BUFFER_PADDING_SIZE as usize;
            let allocation = ffmpeg::ffi::av_mallocz(padded) as *mut u8;
            std::ptr::copy_nonoverlapping(extradata.as_ptr(), allocation, extradata.len());
            (*raw).extradata = allocation;
            (*raw).extradata_size = extradata.len() as i32;
        }
    }
    parameters
}

/// An H.264 video sink for a track whose negotiation retained H.264, with
/// the receiver its pushes land in.
fn h264_sink(id: u64, capacity: usize) -> (WebRtcTrackSink, Receiver<Command>) {
    video_sink(id, capacity, Codec::H264)
}

fn video_sink(id: u64, capacity: usize, codec: Codec) -> (WebRtcTrackSink, Receiver<Command>) {
    let (handle, command_rx) = command_only_handle(capacity);
    let sink = WebRtcTrackSink::new(
        TrackId(id),
        MediaKind::Video,
        Some(codec),
        Arc::new(std::sync::Mutex::new(vec![codec])),
        handle.command_tx.clone(),
    );
    (sink, command_rx)
}

/// One encoded video packet with the time base str0m needs to build a
/// `MediaTime` from.
fn video_packet(payload: &[u8], key: bool) -> MediaBuffer {
    let mut packet = ffmpeg::Packet::copy(payload);
    packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
    packet.set_pts(Some(0));
    if key {
        packet.set_flags(ffmpeg::packet::Flags::KEY);
    }
    MediaBuffer::Packet(Arc::new(packet))
}

/// An encoder that keeps its SPS/PPS in `parameters()` sends a bitstream with
/// none, and nothing downstream of RTP can decode that — so the sender puts
/// them back, on every keyframe and only on keyframes.
///
/// The time base is the part worth guarding: rebuilding a packet drops it to
/// 0/0, and str0m answers a packet it cannot build a `MediaTime` from by
/// dropping it rather than refusing it, so getting this wrong is a peer that
/// silently receives nothing.
#[test]
fn parameter_sets_go_in_front_of_every_keyframe_and_nothing_else() {
    let (mut sink, command_rx) = h264_sink(9, 4);

    const HEADERS: [u8; 8] = [0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1f];
    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::H264, &HEADERS))
        .expect("H.264 is negotiated for this track");

    // A keyframe without them, one that already carries them, and a
    // non-keyframe.
    let payloads: [(&[u8], bool); 3] = [
        (&[0, 0, 0, 1, 0x65, 0x88], true),
        (
            &[0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1f, 0, 0, 0, 1, 0x65, 0x88],
            true,
        ),
        (&[0, 0, 0, 1, 0x41, 0x9a], false),
    ];
    for (payload, key) in payloads {
        sink.consume(video_packet(payload, key)).expect("push");
    }

    let sent: Vec<Vec<u8>> = (0..3)
        .map(|_| {
            let Command::Push(TrackId(9), _, MediaBuffer::Packet(packet)) =
                command_rx.recv().expect("packet was queued")
            else {
                panic!("expected a packet command");
            };
            assert_eq!(
                packet.time_base(),
                ffmpeg::Rational::new(1, 90_000),
                "a rebuilt packet keeps the time base str0m needs"
            );
            packet.data().expect("packet has a payload").to_vec()
        })
        .collect();

    assert_eq!(
        sent[0],
        [&HEADERS[..], &[0, 0, 0, 1, 0x65, 0x88][..]].concat(),
        "a keyframe without the headers gets them"
    );
    assert_eq!(
        sent[1], payloads[1].0,
        "a keyframe that already carries them is left alone"
    );
    assert_eq!(
        sent[2], payloads[2].0,
        "a non-keyframe is left alone: the headers are only useful where          decoding can start"
    );
}

/// An `AVCDecoderConfigurationRecord` as a container demuxer reports one:
/// version, profile/compatibility/level, four-byte NAL lengths (`0xff`), one
/// SPS (`0xe1`) and its bytes, then one PPS and its bytes.
const AVCC: [u8; 19] = [
    0x01, 0x42, 0xc0, 0x1f, 0xff, 0xe1, 0x00, 0x04, 0x67, 0x42, 0xc0, 0x1f, 0x01, 0x00, 0x04, 0x68,
    0xce, 0x3c, 0x80,
];

/// The same parameter sets in the form RTP carries.
const AVCC_AS_ANNEX_B: [u8; 16] = [
    0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1f, 0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80,
];

/// A demuxer describes H.264 with an `avcC` record and emits length-prefixed
/// packets to match. Both facts come out of the one record, so handing it
/// over is all a caller relaying such packets has to do.
#[test]
fn a_demuxers_avcc_configuration_rewrites_its_packets_as_annex_b() {
    let (mut sink, command_rx) = h264_sink(11, 2);
    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::H264, &AVCC))
        .expect("H.264 is negotiated for this track");

    // One four-byte length, then the NAL unit it introduces.
    let payloads: [(&[u8], bool); 2] = [
        (&[0, 0, 0, 2, 0x65, 0x88], true),
        (&[0, 0, 0, 2, 0x41, 0x9a], false),
    ];
    for (payload, key) in payloads {
        sink.consume(video_packet(payload, key))
            .expect("an avcC-configured sink accepts the packets that record describes");
    }

    let sent: Vec<Vec<u8>> = (0..2)
        .map(|_| {
            let Command::Push(TrackId(11), _, MediaBuffer::Packet(packet)) =
                command_rx.recv().expect("packet was queued")
            else {
                panic!("expected a packet command");
            };
            assert_eq!(
                packet.time_base(),
                ffmpeg::Rational::new(1, 90_000),
                "a rewritten packet keeps the time base str0m needs"
            );
            packet.data().expect("packet has a payload").to_vec()
        })
        .collect();

    assert_eq!(
        sent[0],
        [&AVCC_AS_ANNEX_B[..], &[0, 0, 0, 1, 0x65, 0x88][..]].concat(),
        "a keyframe is rewritten and gets the record's parameter sets in Annex-B"
    );
    assert_eq!(
        sent[1],
        [0, 0, 0, 1, 0x41, 0x9a],
        "every packet is rewritten, not only the ones the headers go in front of"
    );
}

/// The record says the payloads are length-prefixed, and this one is not.
///
/// A caller can have reason to pass a demuxer's parameters — the parameter
/// sets are only there — while something upstream has already converted the
/// packets. Read as length-prefixed, an Annex-B payload's leading start code
/// parses as a one-byte NAL unit and the packet is refused as malformed,
/// which names neither the cause nor the fix.
#[test]
fn a_sink_declared_from_an_avcc_record_still_takes_annex_b_packets() {
    let (mut sink, command_rx) = h264_sink(15, 1);
    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::H264, &AVCC))
        .expect("H.264 is negotiated for this track");

    let payload: &[u8] = &[0, 0, 0, 1, 0x65, 0x88];
    sink.consume(video_packet(payload, true))
        .expect("a payload already in the form RTP carries is not a malformed one");

    let Command::Push(TrackId(15), _, MediaBuffer::Packet(packet)) =
        command_rx.recv().expect("packet was queued")
    else {
        panic!("expected a packet command");
    };
    assert_eq!(
        packet.data().expect("packet has a payload"),
        [&AVCC_AS_ANNEX_B[..], payload].concat(),
        "the payload is passed through and still gets the record's parameter sets"
    );
}

/// `set_codec` says the codec and nothing else, including after a
/// declaration that said more.
///
/// Both halves of what the earlier call left have to go. The length prefix
/// would refuse every Annex-B packet that follows, and the headers would be
/// put in front of keyframes that are no longer the ones they describe.
#[test]
fn set_codec_forgets_what_a_previous_declaration_said() {
    let (mut sink, command_rx) = h264_sink(16, 1);
    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::H264, &AVCC))
        .expect("H.264 is negotiated for this track");
    sink.set_codec(Codec::H264)
        .expect("H.264 is negotiated for this track");

    // Its own parameter sets, and deliberately not the record's: headers that
    // survived the redeclaration would be prepended to this, where the ones
    // it carries would have hidden that by matching.
    let payload: &[u8] = &[
        0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1f, 0, 0, 0, 1, 0x68, 0xee, 0x3c, 0x80, 0, 0, 0, 1, 0x65,
        0x88,
    ];
    sink.consume(video_packet(payload, true))
        .expect("a self-assembled Annex-B keyframe carrying its own parameter sets is accepted");

    let Command::Push(TrackId(16), _, MediaBuffer::Packet(packet)) =
        command_rx.recv().expect("packet was queued")
    else {
        panic!("expected a packet command");
    };
    assert_eq!(
        packet.data().expect("packet has a payload"),
        payload,
        "nothing of the previous declaration is left to rewrite or prepend"
    );
}

/// An `avcC` holding no parameter sets at all, which is legal: such a file
/// keeps them in the bitstream. Its packets are length-prefixed all the
/// same, and that is the half of the record the sink still needs — refusing
/// it outright would leave nothing able to state the prefix size.
const AVCC_WITHOUT_PARAMETER_SETS: [u8; 7] = [0x01, 0x64, 0x00, 0x28, 0xff, 0xe0, 0x00];

#[test]
fn an_avcc_with_no_parameter_sets_still_rewrites_its_packets() {
    let (mut sink, command_rx) = h264_sink(17, 1);
    sink.set_source_parameters(&parameters_for(
        ffmpeg::codec::Id::H264,
        &AVCC_WITHOUT_PARAMETER_SETS,
    ))
    .expect("a record with only a prefix size is still a record");

    // In-band, which is where a file like this keeps them.
    let payload: &[u8] = &[
        0, 0, 0, 4, 0x67, 0x64, 0x00, 0x1f, 0, 0, 0, 4, 0x68, 0xee, 0x3c, 0x80, 0, 0, 0, 2, 0x65,
        0x88,
    ];
    sink.consume(video_packet(payload, true))
        .expect("a keyframe carrying its own parameter sets is accepted");

    let Command::Push(TrackId(17), _, MediaBuffer::Packet(packet)) =
        command_rx.recv().expect("packet was queued")
    else {
        panic!("expected a packet command");
    };
    assert_eq!(
        packet.data().expect("packet has a payload"),
        [
            0, 0, 0, 1, 0x67, 0x64, 0x00, 0x1f, 0, 0, 0, 1, 0x68, 0xee, 0x3c, 0x80, 0, 0, 0, 1,
            0x65, 0x88
        ],
        "the prefix size the record did carry is used, and nothing is prepended"
    );
}

/// And the guard the empty record leaves in place. With no parameter sets
/// declared, a keyframe that does not carry its own is the failure this
/// whole declaration exists to prevent, caught where it can still be seen.
#[test]
fn an_avcc_with_no_parameter_sets_still_refuses_a_keyframe_that_has_none() {
    let (mut sink, _command_rx) = h264_sink(18, 1);
    sink.set_source_parameters(&parameters_for(
        ffmpeg::codec::Id::H264,
        &AVCC_WITHOUT_PARAMETER_SETS,
    ))
    .expect("a record with only a prefix size is still a record");

    let error = sink
        .consume(video_packet(&[0, 0, 0, 2, 0x65, 0x88], true))
        .expect_err("a keyframe with no parameter sets anywhere cannot be sent");
    assert!(
        matches!(
            error,
            crate::Error::WebRtcError(WebRtcError::MissingParameterSets(TrackId(18)))
        ),
        "unexpected error: {error:?}"
    );
}

/// VVC moved the NAL unit type into the second byte of its header, so a
/// reader written for H.264 and HEVC finds nothing there and every keyframe
/// looks like one carrying parameter sets. The guard has to see this codec's
/// keyframes as they are, not conclude they are fine because it cannot read
/// them.
#[test]
fn a_vvc_keyframe_is_read_by_its_own_nal_header() {
    // Type in the top five bits of the second byte, temporal id in the rest:
    // SPS 15, PPS 16, and an IDR slice 7.
    const SPS: [u8; 6] = [0, 0, 0, 1, 0x00, 0x79];
    const PPS: [u8; 6] = [0, 0, 0, 1, 0x00, 0x81];
    const IDR: [u8; 7] = [0, 0, 0, 1, 0x00, 0x39, 0xaa];

    let (mut refused, _rx) = video_sink(19, 1, Codec::H266);
    let error = refused
        .consume(video_packet(&IDR, true))
        .expect_err("a VVC keyframe with no parameter sets cannot be sent either");
    assert!(
        matches!(
            error,
            crate::Error::WebRtcError(WebRtcError::MissingParameterSets(TrackId(19)))
        ),
        "unexpected error: {error:?}"
    );

    let (mut accepted, command_rx) = video_sink(20, 1, Codec::H266);
    accepted
        .consume(video_packet(&[&SPS[..], &PPS[..], &IDR[..]].concat(), true))
        .expect("a VVC keyframe that carries its own parameter sets is accepted");
    assert!(
        command_rx.recv().is_ok(),
        "the accepted keyframe reaches the peer"
    );
}

/// Without the record there is no prefix size to rewrite by, and str0m would
/// packetize the length bytes as a NAL header — well-formed RTP that decodes
/// to nothing, reported nowhere. Refusing says which end is wrong.
#[test]
fn a_length_prefixed_packet_is_refused_when_no_avcc_configuration_was_given() {
    let (mut sink, _command_rx) = h264_sink(12, 2);

    for _ in 0..2 {
        let error = sink
            .consume(video_packet(&[0, 0, 0, 2, 0x65, 0x88], true))
            .expect_err("a length-prefixed packet cannot be sent as Annex-B");
        assert!(
            matches!(
                error,
                crate::Error::WebRtcError(WebRtcError::NotAnnexB(TrackId(12)))
            ),
            "every such packet is refused, not just the first: {error}"
        );
    }
}

/// A prefix size that does not match the payload means the two disagree
/// about what is being sent. Emitting whatever the bytes happen to split
/// into would be the same invisible failure by another route.
#[test]
fn a_malformed_length_prefixed_packet_is_refused() {
    let (mut sink, _command_rx) = h264_sink(13, 2);
    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::H264, &AVCC))
        .expect("H.264 is negotiated for this track");

    // A length of nine with two bytes behind it.
    let error = sink
        .consume(video_packet(&[0, 0, 0, 9, 0x65, 0x88], true))
        .expect_err("a truncated access unit cannot be rewritten");
    assert!(matches!(
        error,
        crate::Error::WebRtcError(WebRtcError::MalformedLengthPrefixedPacket(TrackId(13)))
    ));
}

/// The failure the whole declaration exists to prevent: an encoder whose
/// headers went to `parameters()` sends keyframes with none, and a peer waits
/// for configuration that is never coming. Nothing about the packets is
/// malformed, so this is the only place it can be caught.
#[test]
fn a_keyframe_with_no_parameter_sets_and_none_declared_is_refused() {
    let (mut sink, _command_rx) = h264_sink(14, 2);

    // An IDR slice, correctly Annex-B, with no SPS or PPS in front of it.
    for _ in 0..2 {
        let error = sink
            .consume(video_packet(&[0, 0, 0, 1, 0x65, 0x88], true))
            .expect_err("a keyframe no receiver could configure a decoder from");
        assert!(
            matches!(
                error,
                crate::Error::WebRtcError(WebRtcError::MissingParameterSets(TrackId(14)))
            ),
            "every such keyframe is refused, not just the first: {error}"
        );
    }

    // In-band parameter sets are the other way to satisfy it, and need no
    // declaration at all.
    sink.consume(video_packet(
        &[
            0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1f, 0, 0, 0, 1, 0x68, 0xce, 0, 0, 0, 1, 0x65, 0x88,
        ],
        true,
    ))
    .expect("a keyframe carrying its own SPS/PPS needs nothing prepended");
}

/// HEVC keeps its configuration in `hvcC`, which is neither Annex-B nor the
/// `avcC` this sink converts. Storing it would put a decoder configuration
/// record into the bitstream in front of every keyframe.
#[test]
fn hevc_configuration_that_is_not_annex_b_is_refused() {
    let (handle, _command_rx) = command_only_handle(1);
    let mut sink = WebRtcTrackSink::new(
        TrackId(15),
        MediaKind::Video,
        Some(Codec::H265),
        Arc::new(std::sync::Mutex::new(vec![Codec::H265])),
        handle.command_tx.clone(),
    );

    let error = sink
        .set_source_parameters(&parameters_for(
            ffmpeg::codec::Id::HEVC,
            &[0x01, 0x01, 0x60],
        ))
        .expect_err("hvcC cannot be prepended and cannot be converted");
    assert!(matches!(
        error,
        crate::Error::WebRtcError(WebRtcError::ParameterSetsNotSupported {
            track_id: TrackId(15),
            codec: Codec::H265,
        })
    ));
}

/// Only the Annex-B codecs prepend anything. `OpusHead` in front of every
/// Opus packet — and libavcodec flags them all as keyframes — would be
/// corruption, so an audio source's parameters declare the codec and nothing
/// else.
#[test]
fn audio_parameters_declare_the_codec_without_leaving_headers_to_prepend() {
    let (handle, command_rx) = command_only_handle(1);
    let mut sink = WebRtcTrackSink::new(
        TrackId(16),
        MediaKind::Audio,
        None,
        Arc::new(std::sync::Mutex::new(vec![Codec::Opus])),
        handle.command_tx.clone(),
    );
    let opus_head = [b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 2];

    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::OPUS, &opus_head))
        .expect("Opus is negotiated for this track");

    let payload: &[u8] = &[0xfc, 0xff, 0xfe];
    sink.consume(video_packet(payload, true))
        .expect("an Opus packet needs no bitstream handling");
    let Command::Push(TrackId(16), Some(Codec::Opus), MediaBuffer::Packet(packet)) =
        command_rx.recv().expect("packet was queued")
    else {
        panic!("expected an Opus packet command");
    };
    assert_eq!(
        packet.data().expect("packet has a payload"),
        payload,
        "the payload must reach str0m exactly as the encoder produced it"
    );
}

/// A codec WebRTC has no payload type for cannot be sent at all, and saying
/// so beats letting str0m label it as whatever was negotiated.
#[test]
fn parameters_naming_a_codec_webrtc_does_not_carry_are_refused() {
    let (mut sink, _command_rx) = h264_sink(17, 1);

    let error = sink
        .set_source_parameters(&parameters_for(ffmpeg::codec::Id::AAC, &[]))
        .expect_err("WebRTC negotiates no AAC payload type");
    assert!(matches!(
        error,
        crate::Error::WebRtcError(WebRtcError::SourceCodecUnsupported(_))
    ));
}

#[test]
fn track_sink_shifts_a_negative_encoder_delay_without_changing_packet_spacing() {
    let (handle, command_rx) = command_only_handle(2);
    let negotiated = Arc::new(std::sync::Mutex::new(vec![Codec::Opus]));
    let mut sink = WebRtcTrackSink::new(
        TrackId(8),
        MediaKind::Audio,
        Some(Codec::Opus),
        negotiated,
        handle.command_tx.clone(),
    );

    for pts in [-312, 648] {
        let mut packet = ffmpeg::Packet::copy(&[1, 2, 3]);
        packet.set_time_base(ffmpeg::Rational::new(1, 48_000));
        packet.set_pts(Some(pts));
        packet.set_dts(Some(pts));
        sink.consume(MediaBuffer::Packet(Arc::new(packet)))
            .expect("negative encoder delay should be normalized before RTP");
    }

    for expected in [0, 960] {
        let Command::Push(TrackId(8), Some(Codec::Opus), MediaBuffer::Packet(packet)) =
            command_rx.recv().expect("normalized packet was queued")
        else {
            panic!("expected an Opus packet command");
        };
        assert_eq!(packet.pts(), Some(expected));
        assert_eq!(packet.dts(), Some(expected));
    }
}

#[test]
fn selecting_an_unnegotiated_codec_preserves_the_previous_selection() {
    let (handle, command_rx) = command_only_handle(1);
    let negotiated = Arc::new(std::sync::Mutex::new(vec![Codec::Vp8]));
    let mut sink = WebRtcTrackSink::new(
        TrackId(7),
        MediaKind::Video,
        None,
        negotiated,
        handle.command_tx.clone(),
    );

    sink.set_codec(Codec::Vp8)
        .expect("VP8 is negotiated for this track");
    let error = sink
        .set_codec(Codec::H264)
        .expect_err("H.264 must be rejected without changing the VP8 selection");
    assert!(matches!(
        error,
        crate::Error::WebRtcError(WebRtcError::OutboundCodecNotNegotiated {
            track_id: TrackId(7),
            codec: Codec::H264,
            negotiated,
        }) if negotiated == vec![Codec::Vp8]
    ));

    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
    packet.set_pts(Some(0));
    sink.consume(MediaBuffer::Packet(Arc::new(packet)))
        .expect("the previous VP8 selection should still be usable");
    let Command::Push(TrackId(7), codec, _) = command_rx.recv().expect("packet should be queued")
    else {
        panic!("expected Push command");
    };
    assert_eq!(codec, Some(Codec::Vp8));
}

#[test]
fn wait_stream_info_can_retry_after_timeout_and_caches_the_result() {
    let (_data_tx, data_rx) = bounded(1);
    let (info_tx, info_rx) = bounded(1);
    let source = WebRtcTrackSource::new(
        TrackId(9),
        MediaKind::Video,
        "track-in",
        data_rx,
        Arc::new(std::sync::Mutex::new(None)),
        Arc::new(std::sync::Mutex::new(vec![Codec::H264])),
        info_rx,
    );
    let timeout = Duration::from_millis(10);

    let error = source
        .wait_stream_info(timeout)
        .expect_err("no media should time out");
    assert!(matches!(
        error,
        crate::Error::WebRtcError(WebRtcError::StreamInfoTimeout {
            track_id: TrackId(9),
            timeout: actual,
        }) if actual == timeout
    ));

    let expected = WebRtcStreamInfo::from(CodecSpec {
        codec: Codec::H264,
        clock_rate: Frequency::NINETY_KHZ,
        channels: None,
        format: FormatParams::default(),
    });
    info_tx
        .send(expected.clone())
        .expect("source still owns receiver");
    assert_eq!(
        source
            .wait_stream_info(Duration::from_secs(1))
            .expect("retry should receive the first payload info"),
        expected
    );
    drop(info_tx);
    assert_eq!(
        source
            .wait_stream_info(Duration::ZERO)
            .expect("cached info should not depend on the channel"),
        expected
    );
}

#[test]
fn wait_stream_info_returns_closed_if_the_peer_ends_before_media() {
    let (_data_tx, data_rx) = bounded(1);
    let (info_tx, info_rx) = bounded(1);
    let source = WebRtcTrackSource::new(
        TrackId(10),
        MediaKind::Video,
        "track-in",
        data_rx,
        Arc::new(std::sync::Mutex::new(None)),
        Arc::new(std::sync::Mutex::new(vec![Codec::H264])),
        info_rx,
    );
    drop(info_tx);

    assert!(matches!(
        source.wait_stream_info(Duration::from_secs(1)),
        Err(crate::Error::WebRtcError(WebRtcError::Closed))
    ));
}

struct CountingSink {
    pp_log: PpLog,
    count: Arc<AtomicUsize>,
}

impl Element for CountingSink {
    fn name(&self) -> Arc<str> {
        "counter".into()
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

impl Sink for CountingSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if matches!(buf, MediaBuffer::Packet(_)) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// Two `Rtc`s, connected via a throwaway data channel — just enough to
/// get ICE/DTLS established with zero media, mirroring how a real
/// caller does the initial signaling before ever touching
/// `WebRtcPeer` (see its own docs).
fn connected_pair() -> (Rtc, UdpSocket, Rtc, UdpSocket) {
    let socket_a = UdpSocket::bind("127.0.0.1:0").expect("bind a");
    let socket_b = UdpSocket::bind("127.0.0.1:0").expect("bind b");
    let addr_a = socket_a.local_addr().expect("addr a");
    let addr_b = socket_b.local_addr().expect("addr b");

    let mut rtc_a = Rtc::builder().build(Instant::now());
    rtc_a
        .add_local_candidate(Candidate::host(addr_a, "udp").expect("candidate a"))
        .expect("add candidate a");
    let mut rtc_b = Rtc::builder().build(Instant::now());
    rtc_b
        .add_local_candidate(Candidate::host(addr_b, "udp").expect("candidate b"))
        .expect("add candidate b");

    let mut changes = rtc_a.sdp_api();
    changes.add_channel("bootstrap".to_string());
    let (offer, pending) = changes.apply().expect("adding a channel always offers");
    let answer = rtc_b.sdp_api().accept_offer(offer).expect("b accepts");
    rtc_a
        .sdp_api()
        .accept_answer(pending, answer)
        .expect("a accepts answer");

    (rtc_a, socket_a, rtc_b, socket_b)
}

/// Both halves of a `Direction::SendRecv` track, panicking if the track
/// attached with anything else — the direction is what the test asked
/// `add_track` for, so a different one is a failure of the code under
/// test rather than a case to handle.
fn send_recv(track: AttachedTrack) -> (WebRtcTrackSink, WebRtcTrackSource) {
    let TrackEndpoints::SendRecv(sink, source) = track.endpoints else {
        panic!("expected a SendRecv track");
    };
    (sink, source)
}

/// The inbound half of a receive-only track — what the peer that did
/// *not* call `add_track` sees for a `Direction::SendOnly` one.
fn recv_only(track: AttachedTrack) -> WebRtcTrackSource {
    let TrackEndpoints::Recv(source) = track.endpoints else {
        panic!("expected a RecvOnly track");
    };
    source
}

fn push_packets(sink: &mut WebRtcTrackSink) {
    for i in 0..5 {
        // `write_track` needs a real time base to build str0m's own
        // `MediaTime` from — a bare `Packet::copy` defaults to 0/0,
        // which is silently unusable (dropped, not an error).
        let payload: &[u8] = if i == 0 {
            // A complete Annex-B access unit with SPS/PPS. H.264 stream info
            // intentionally does not become ready from a merely-labeled
            // payload; the actual parameter sets must cross RTP first.
            &[
                0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1f, 0x1a, 0x32, 0x35, 0x01, 0x40, 0x7a, 0x40, 0x3c,
                0x22, 0x11, 0xa8, 0, 0, 0, 1, 0x68, 0x1a, 0x34, 0xe3, 0xc8, 0, 0, 0, 1, 0x65, 0x88,
                0x84,
            ]
        } else {
            &[1, 2, 3, 4]
        };
        let mut packet = ffmpeg::Packet::copy(payload);
        packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
        packet.set_pts(Some(i * 3_000));
        sink.consume(MediaBuffer::Packet(Arc::new(packet)))
            .expect("push");
    }
}

fn wait_for_frames(frames: &AtomicUsize) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if frames.load(Ordering::SeqCst) > 0 {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("expected at least one decoded frame");
}

/// Wires `source` into its own `Pipeline`, forwarding every packet it
/// produces into a `CountingSink` that increments `count` — standing in
/// for the real, independent downstream pipeline a `WebRtcTrackSource`
/// is meant to be driven from.
fn wire_counting(source: WebRtcTrackSource, count: Arc<AtomicUsize>) -> Arc<Pipeline> {
    let sink = CountingSink {
        count,
        pp_log: element_pp_log(ElementType::Other, "counter", None),
    };
    Pipeline::new("test", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(sink))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed")
}

/// One H.264 `Direction::SendRecv` track, opened by `WebRtcHandle::add_track`
/// on one peer and relayed to the other by hand (standing in for a
/// real signaling transport — see `on_offer`'s docs), carries data
/// *both* ways: peer-a pushes into the `WebRtcTrackSink` its own
/// `next_track()` returned for the track it just added (str0m never
/// fires `Event::MediaAdded` for that — see the type docs), and peer-b
/// pushes back on the exact same `Mid` via the `WebRtcTrackSink` its
/// own `next_track()` returned for the resulting `Event::MediaAdded` —
/// no second `add_track`/renegotiation needed for the reverse
/// direction. Also exercises real ICE/DTLS-SRTP over loopback UDP, with
/// each `WebRtcTrackSource` driven by its own independent `Pipeline`.
#[test]
fn one_h264_sendrecv_track_carries_data_both_ways_with_the_declared_payload_type() {
    let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();

    let (offer_tx, offer_rx) = crossbeam_channel::unbounded::<SdpOffer>();

    let (peer_a, handle_a) = WebRtcPeer::new(
        "peer-a",
        rtc_a,
        socket_a,
        move |offer| {
            let _ = offer_tx.send(offer);
        },
        |_id| {},
    );
    let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});

    // `WebRtcPeer` is a `Driver`, not a `Pipeline` source — no wiring
    // closure needed; inbound tracks attach dynamically via
    // `next_track()` instead, each into its own `Pipeline`.
    let driver_a = DriverRunner::new(peer_a);
    let driver_b = DriverRunner::new(peer_b);
    driver_a.run().unwrap();
    driver_b.run().unwrap();

    // Let ICE/DTLS actually finish over the real loopback sockets
    // before renegotiating.
    thread::sleep(Duration::from_millis(200));

    let track_id = handle_a
        .add_track(MediaKind::Video, Direction::SendRecv, Codec::H264)
        .expect("running peer should accept AddTrack");
    // `next_track()` returns for peer-a's own track the moment
    // `add_track`'s negotiation mints a `Mid` — before the offer even
    // leaves this process, let alone before an answer comes back.
    let attached_a = handle_a
        .next_track()
        .expect("peer-a's own track should attach");
    assert_eq!(
        track_id, attached_a.id,
        "next_track should report the TrackId add_track just returned"
    );
    let (mut sink_a, source_a) = send_recv(attached_a);

    let offer = offer_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("peer-a should generate a renegotiation offer");
    let answer = handle_b
        .accept_remote_offer(offer)
        .expect("peer-b should accept the offer");
    handle_a.set_answer(answer);
    // peer-b's track attaches as part of accepting the offer (its
    // `Event::MediaAdded`), independent of the answer round-trip.
    let (mut sink_b, source_b) = send_recv(
        handle_b
            .next_track()
            .expect("peer-b's remote track should attach"),
    );
    // The negotiated capability list is available when peer-b's endpoints
    // are created, while the actual inbound codec remains unknown until the
    // first RTP packet arrives. Peer-b then selects the codec its own encoder
    // produces, validated against that list.
    assert!(sink_b.negotiated_codecs().contains(&Codec::H264));
    assert_eq!(sink_b.negotiated_codecs(), source_b.negotiated_codecs());
    assert_eq!(source_b.codec(), None);

    // Let the answer actually apply before pushing media through it.
    thread::sleep(Duration::from_millis(100));

    push_packets(&mut sink_a);

    // peer-b's return direction uses a real OpenH264 encoder. Besides
    // checking the observed RTP codec below, peer-a decodes this stream and
    // must produce a frame; a mislabeled VP8 payload would fail there.
    let video_options = TestVideoOptions {
        width: 160,
        height: 120,
        framerate: ffmpeg::Rational::new(15, 1),
    };
    let video_source = TestVideoSource::new("peer-b-video", video_options);
    let encoder = SwEncoder::new(
        "peer-b-h264",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: video_options.width,
            height: video_options.height,
            time_base: video_source.time_base(),
            frame_rate: video_options.framerate,
            bit_rate: 250_000,
            gop_size: 15,
        },
    )
    .expect("OpenH264 encoder should open");
    // One declaration covers peer-b's whole outbound half: the payload type
    // (validated against the negotiated list asserted above) and the SPS/PPS
    // the encoder keeps in `parameters()` rather than in the bitstream.
    // Without the latter the peer never sees them and `wait_stream_info`
    // below times out.
    sink_b
        .set_source_parameters(&encoder.parameters())
        .expect("H.264 should be negotiated for peer-b's outbound half");
    let send_b = Pipeline::new("peer-b-h264-send", video_source, |source, ctx| {
        let branch = ctx.branch().pipe(encoder).to(Box::new(sink_b))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("peer-b send pipeline should wire");
    send_b.run().expect("peer-b send pipeline should start");

    // Keep each source out of its receive pipeline until the peer has seen a
    // packet. The returned specs come from payload types str0m actually
    // received, not from either sink's declared encoder choice. The receive
    // pipelines below also prove those first packets remain queued.
    let info_a = source_a
        .wait_stream_info(Duration::from_secs(2))
        .expect("peer-a should observe peer-b's H.264 payload");
    let info_b = source_b
        .wait_stream_info(Duration::from_secs(2))
        .expect("peer-b should observe peer-a's H.264 payload");
    assert_eq!(info_a.codec(), Codec::H264);
    assert_eq!(info_b.codec(), Codec::H264);
    assert_eq!(source_a.codec(), Some(Codec::H264));
    assert_eq!(source_b.codec(), Some(Codec::H264));

    let received_by_b = Arc::new(AtomicUsize::new(0));
    let decoder = SwDecoder::new(
        "peer-a-h264-decode",
        info_a
            .codec_parameters()
            .expect("actual H.264 info should create codec parameters"),
    )
    .expect("peer-a H.264 decoder should open");
    let (counter, decoded_by_a) = FrameCounter::new("decoded-by-a");
    let track_pipeline_a = Pipeline::new("peer-a-h264-recv", source_a, |source, ctx| {
        let branch = ctx.branch().pipe(decoder).to(Box::new(counter))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("peer-a receive pipeline should wire");
    let track_pipeline_b = wire_counting(source_b, received_by_b.clone());
    track_pipeline_a.run().unwrap();
    track_pipeline_b.run().unwrap();
    wait_for_frames(&decoded_by_a);
    thread::sleep(Duration::from_millis(100));
    assert_eq!(
        received_by_b.load(Ordering::SeqCst),
        5,
        "peer-b should receive everything peer-a pushed"
    );
    assert!(
        decoded_by_a.load(Ordering::SeqCst) > 0,
        "peer-a should decode peer-b's reverse H.264 stream"
    );

    // Trivial now — `DriverRunner::stop` just flips a flag, no
    // rendezvous ack to race (see `StopReceiver`'s own docs).
    send_b.stop();
    driver_a.stop();
    driver_b.stop();
    // Deliberately also stopped even though each `WebRtcTrackSource`'s
    // `Pipeline` may already be ending on its own by this point (the
    // `driver_a/b.stop()` above disconnects its data channel, which
    // ends it with a natural `Eos` — see that type's own docs): this is
    // a regression guard for a real `Pipeline::stop` bug this exact
    // race used to hit, now fixed at its source (see `Pipeline`'s own
    // `control_rx` field docs).
    track_pipeline_a.stop();
    track_pipeline_b.stop();

    let events_a: Vec<_> = driver_a.bus().iter().collect();
    let events_b: Vec<_> = driver_b.bus().iter().collect();
    let send_events_b: Vec<_> = send_b.bus().iter().collect();
    let track_events_a: Vec<_> = track_pipeline_a.bus().iter().collect();
    let track_events_b: Vec<_> = track_pipeline_b.bus().iter().collect();
    assert!(
        !events_a.iter().any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s) on peer-a: {events_a:?}"
    );
    assert!(
        !events_b.iter().any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s) on peer-b: {events_b:?}"
    );
    assert!(
        !send_events_b
            .iter()
            .any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s) on peer-b's outbound encoder: {send_events_b:?}"
    );
    assert!(
        !track_events_a
            .iter()
            .any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s) on peer-a's inbound track: {track_events_a:?}"
    );
    assert!(
        !track_events_b
            .iter()
            .any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s) on peer-b's inbound track: {track_events_b:?}"
    );
}

/// Regression test for the claim in `Driver::run`'s own docs: stopping
/// one side must clear its `tracks_in` immediately, so an
/// already-attached `WebRtcTrackSource` sees its data channel
/// disconnect and ends with a final `Eos` on its own — not left
/// hanging until some other, unrelated thing (e.g. a caller's own
/// `Pipeline::stop`) intervenes. There's no "reconnect" concept here
/// to test instead: `WebRtcPeer::run` only ever reacts to its `Rtc`
/// going not-alive or an explicit `Stop`, treating both the same way
/// (see that `if` right before `tracks_in.clear()`); `Stop` is the
/// deterministic one of the two to trigger from a test. Deliberately
/// omits the `track_pipeline.stop()` safety net
/// `one_h264_sendrecv_track_carries_data_both_ways_with_the_declared_payload_type`
/// uses — the whole point
/// here is to prove the `Pipeline` ends on its own.
#[test]
fn stopping_a_peer_ends_its_inbound_track_source_with_a_clean_eos() {
    let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();

    let (offer_tx, offer_rx) = crossbeam_channel::unbounded::<SdpOffer>();
    let (peer_a, handle_a) = WebRtcPeer::new(
        "peer-a",
        rtc_a,
        socket_a,
        move |offer| {
            let _ = offer_tx.send(offer);
        },
        |_id| {},
    );
    let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});

    let driver_a = DriverRunner::new(peer_a);
    let driver_b = DriverRunner::new(peer_b);
    driver_a.run().unwrap();
    driver_b.run().unwrap();

    thread::sleep(Duration::from_millis(200));

    handle_a
        .add_track(MediaKind::Video, Direction::SendOnly, Codec::Vp8)
        .expect("running peer should accept AddTrack");
    let offer = offer_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("peer-a should generate a renegotiation offer");
    let answer = handle_b
        .accept_remote_offer(offer)
        .expect("peer-b should accept the offer");
    handle_a.set_answer(answer);

    // peer-a asked for `SendOnly`, so peer-b sees the inverse: a track it
    // can only receive on, and one this crate hands out without a sink at
    // all rather than with an inert one.
    let source_b = recv_only(
        handle_b
            .next_track()
            .expect("peer-b's remote track should attach"),
    );

    let received = Arc::new(AtomicUsize::new(0));
    let track_pipeline_b = wire_counting(source_b, received);
    track_pipeline_b.run().unwrap();

    // Let the answer actually apply before tearing the connection down.
    thread::sleep(Duration::from_millis(100));

    driver_b.stop();

    // No `track_pipeline_b.stop()` — draining the bus to completion is
    // itself the proof: it only returns once every `Bus` sender,
    // including the one `WebRtcTrackSource`'s own `Pipeline::run`
    // thread holds, has actually dropped.
    let track_events_b: Vec<_> = track_pipeline_b.bus().iter().collect();
    assert!(
        track_events_b
            .iter()
            .any(|e| matches!(e, BusEvent::Eos { .. })),
        "expected the inbound track's own Pipeline to reach Eos on its \
             own once peer-b stopped, without an explicit Pipeline::stop; \
             got {track_events_b:?}"
    );
    assert!(
        !track_events_b
            .iter()
            .any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s): {track_events_b:?}"
    );

    driver_a.stop();
}

/// Regression test for the RTP-timestamp bug where `write_track` built
/// str0m's `MediaTime` from `pts / time_base.denominator()`, silently
/// dropping `time_base`'s numerator. That was invisible with the
/// numerator-1 time bases used elsewhere in this codebase (e.g.
/// `1/90_000`) but wrong for an NTSC-style time base like `1001/30_000`,
/// where it made the RTP clock run ~1001x too fast.
#[test]
fn packet_rtp_time_accounts_for_the_time_base_numerator() {
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(1001, 30_000));
    packet.set_pts(Some(30));

    let media_time = packet_rtp_time(&packet).expect("packet has a usable time base");

    // 30 ticks of 1001/30_000s each = 1.001s = 30_030/30_000.
    assert_eq!(media_time.numer(), 30_030);
    assert_eq!(media_time.denom(), 30_000);
    assert!(
        (media_time.as_seconds() - 1.001).abs() < 1e-9,
        "expected ~1.001s, got {}",
        media_time.as_seconds()
    );
}

/// The common case (numerator 1) must still come out exactly right.
#[test]
fn packet_rtp_time_handles_unit_numerator_time_bases() {
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
    packet.set_pts(Some(3_000));

    let media_time = packet_rtp_time(&packet).expect("packet has a usable time base");

    assert_eq!(media_time.numer(), 3_000);
    assert_eq!(media_time.denom(), 90_000);
}

#[test]
fn packet_rtp_time_rejects_missing_pts() {
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(1, 90_000));

    assert!(matches!(
        packet_rtp_time(&packet),
        Err(WebRtcError::MissingPacketPts)
    ));
}

#[test]
fn packet_rtp_time_rejects_negative_pts() {
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
    packet.set_pts(Some(-1));

    assert!(matches!(
        packet_rtp_time(&packet),
        Err(WebRtcError::NegativePacketPts(-1))
    ));
}

#[test]
fn packet_rtp_time_rejects_invalid_time_base() {
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(0, 0));
    packet.set_pts(Some(0));

    assert!(matches!(
        packet_rtp_time(&packet),
        Err(WebRtcError::InvalidPacketTimeBase { .. })
    ));
}

#[test]
fn packet_rtp_time_rejects_timestamp_overflow() {
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(i32::MAX, 1));
    packet.set_pts(Some(i64::MAX));

    assert!(matches!(
        packet_rtp_time(&packet),
        Err(WebRtcError::PacketTimestampOverflow { .. })
    ));
}

/// The endpoints handed out have to be exactly what the negotiated
/// direction allows, on *both* sides of one `SendOnly` track: the side
/// that added it gets a sink and nothing to receive on, and the side that
/// only receives gets a source and no sink it could push into to no
/// effect. This is the contract that replaced handing out an inert half.
#[test]
fn a_send_only_track_yields_a_sink_on_one_side_and_a_source_on_the_other() {
    let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();

    let (offer_tx, offer_rx) = crossbeam_channel::unbounded::<SdpOffer>();
    let (peer_a, handle_a) = WebRtcPeer::new(
        "peer-a",
        rtc_a,
        socket_a,
        move |offer| {
            let _ = offer_tx.send(offer);
        },
        |_id| {},
    );
    let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});
    let driver_a = DriverRunner::new(peer_a);
    let driver_b = DriverRunner::new(peer_b);
    driver_a.run().unwrap();
    driver_b.run().unwrap();
    thread::sleep(Duration::from_millis(200));

    handle_a
        .add_track(MediaKind::Video, Direction::SendOnly, Codec::Vp8)
        .expect("running peer should accept AddTrack");
    let attached_a = handle_a
        .next_track()
        .expect("peer-a's own track should attach");
    assert!(
        matches!(attached_a.endpoints, TrackEndpoints::Send(_)),
        "a SendOnly track must not hand its adder a source"
    );

    let offer = offer_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("peer-a should generate a renegotiation offer");
    let answer = handle_b
        .accept_remote_offer(offer)
        .expect("peer-b should accept the offer");
    handle_a.set_answer(answer);

    let attached_b = handle_b
        .next_track()
        .expect("peer-b's remote track should attach");
    assert!(
        matches!(attached_b.endpoints, TrackEndpoints::Recv(_)),
        "the receiving side of a SendOnly track must not get a sink"
    );

    driver_a.stop();
    driver_b.stop();
}

/// The mirror of the `SendOnly` case, so neither direction is covered
/// only by inference from the other: asking for `RecvOnly` gives the
/// *adder* the inbound half, and the remote peer — which sees the
/// inverse — the outbound one.
#[test]
fn a_recv_only_track_yields_a_source_on_one_side_and_a_sink_on_the_other() {
    let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();

    let (offer_tx, offer_rx) = crossbeam_channel::unbounded::<SdpOffer>();
    let (peer_a, handle_a) = WebRtcPeer::new(
        "peer-a",
        rtc_a,
        socket_a,
        move |offer| {
            let _ = offer_tx.send(offer);
        },
        |_id| {},
    );
    let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});
    let driver_a = DriverRunner::new(peer_a);
    let driver_b = DriverRunner::new(peer_b);
    driver_a.run().unwrap();
    driver_b.run().unwrap();
    thread::sleep(Duration::from_millis(200));

    handle_a
        .add_track(MediaKind::Video, Direction::RecvOnly, Codec::Vp8)
        .expect("running peer should accept AddTrack");
    let attached_a = handle_a
        .next_track()
        .expect("peer-a's own track should attach");
    assert!(
        matches!(attached_a.endpoints, TrackEndpoints::Recv(_)),
        "a RecvOnly track must not hand its adder a sink"
    );

    let offer = offer_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("peer-a should generate a renegotiation offer");
    let answer = handle_b
        .accept_remote_offer(offer)
        .expect("peer-b should accept the offer");
    handle_a.set_answer(answer);

    let attached_b = handle_b
        .next_track()
        .expect("peer-b's remote track should attach");
    assert!(
        matches!(attached_b.endpoints, TrackEndpoints::Send(_)),
        "the sending side of a RecvOnly track must not get a source"
    );

    driver_a.stop();
    driver_b.stop();
}

/// A `WebRtcPeer` with nothing running on it, for exercising the
/// attach/event handling directly. Driving a *real* direction change
/// end to end would mean hand-rolling one side's whole ICE/DTLS poll
/// loop just to reach `SdpApi::set_direction`; what actually needs
/// covering is this element's own reaction to the event, so the event is
/// what the tests below supply.
fn idle_peer(name: &str) -> (WebRtcPeer, WebRtcHandle) {
    let (rtc, socket, _rtc_b, _socket_b) = connected_pair();
    WebRtcPeer::new(name, rtc, socket, |_offer| {}, |_id| {})
}

/// `Event::Closed` is the deterministic signal produced by str0m for a
/// received DTLS `close_notify`. Unlike the end-to-end loopback test below,
/// this does not rely on a later UDP send eliciting an ICMP error: the event
/// itself must end the driver and disconnect its inbound source.
#[test]
fn a_remote_closed_event_ends_the_peer_and_its_inbound_source() {
    let (mut peer, handle) = idle_peer("peer");
    peer.attach_track(
        TrackId(0),
        "0".into(),
        MediaKind::Video,
        Direction::RecvOnly,
    );
    let source = recv_only(
        handle
            .next_track()
            .expect("the inbound track should attach"),
    );
    let received = Arc::new(AtomicUsize::new(0));
    let track_pipeline = wire_counting(source, received);
    track_pipeline.run().unwrap();

    let (bus, _bus_rx) = crate::bus::Bus::new();
    peer.handle_event(str0m::Event::Closed, &bus);
    let driver = DriverRunner::new(peer);
    driver.run().unwrap();

    let driver_events: Vec<_> = driver.bus().iter().collect();
    assert!(
        !driver_events
            .iter()
            .any(|event| matches!(event, BusEvent::Error { .. })),
        "remote close should end cleanly, got {driver_events:?}"
    );
    let track_events: Vec<_> = track_pipeline.bus().iter().collect();
    assert!(
        track_events
            .iter()
            .any(|event| matches!(event, BusEvent::Eos { .. })),
        "the inbound source should reach Eos, got {track_events:?}"
    );
    assert!(matches!(
        handle.add_track(MediaKind::Video, Direction::SendOnly, Codec::Vp8),
        Err(crate::Error::WebRtcError(WebRtcError::Closed))
    ));
}

/// `TrackEndpoints` is minted once, from the direction the track
/// attached with, so a remote peer renegotiating a different one leaves
/// the caller holding endpoints that no longer describe the connection.
/// Nothing can be re-issued at that point, which is exactly why it has
/// to be reported rather than silently tolerated.
#[test]
fn a_direction_change_after_attaching_is_reported_on_the_bus() {
    let (mut peer, handle) = idle_peer("peer");
    let (bus, bus_rx) = crate::bus::Bus::new();
    let mid: Mid = "0".into();

    peer.attach_track(TrackId(0), mid, MediaKind::Video, Direction::SendOnly);
    let attached = handle.next_track().expect("the track should attach");
    assert!(matches!(attached.endpoints, TrackEndpoints::Send(_)));

    peer.handle_event(
        str0m::Event::MediaChanged(str0m::media::MediaChanged {
            mid,
            direction: Direction::RecvOnly,
        }),
        &bus,
    );

    let event = bus_rx
        .try_recv()
        .expect("a changed direction should reach the bus");
    let BusEvent::Error { error, .. } = event else {
        panic!("expected BusEvent::Error");
    };
    let crate::Error::WebRtcError(WebRtcError::DirectionChanged { mid: at, from, to }) = error
    else {
        panic!("expected WebRtcError::DirectionChanged, got {error}");
    };
    assert_eq!(at, mid);
    assert_eq!(from, Direction::SendOnly);
    assert_eq!(to, Direction::RecvOnly);
}

/// `MediaChanged` also fires when a renegotiation restates the direction
/// already in force. Reporting that would turn every unrelated
/// renegotiation into a bus error, so only an actual change counts.
#[test]
fn restating_the_direction_a_track_already_has_is_not_reported() {
    let (mut peer, handle) = idle_peer("peer");
    let (bus, bus_rx) = crate::bus::Bus::new();
    let mid: Mid = "0".into();

    peer.attach_track(TrackId(0), mid, MediaKind::Video, Direction::SendRecv);
    let _attached = handle.next_track().expect("the track should attach");

    peer.handle_event(
        str0m::Event::MediaChanged(str0m::media::MediaChanged {
            mid,
            direction: Direction::SendRecv,
        }),
        &bus,
    );

    assert!(
        bus_rx.try_recv().is_none(),
        "an unchanged direction must not be reported as a change"
    );
}

/// `MediaChanged` can name a `mid` this element never attached — data
/// channels and media it does not track go through the same event
/// stream. There is nothing to compare against and nothing the caller
/// holds, so there is nothing to report either.
#[test]
fn a_direction_change_on_an_unattached_track_is_ignored() {
    let (mut peer, _handle) = idle_peer("peer");
    let (bus, bus_rx) = crate::bus::Bus::new();

    peer.handle_event(
        str0m::Event::MediaChanged(str0m::media::MediaChanged {
            mid: "99".into(),
            direction: Direction::RecvOnly,
        }),
        &bus,
    );

    assert!(
        bus_rx.try_recv().is_none(),
        "a mid this element never attached has nothing to report"
    );
}

/// `Direction::Inactive` still attaches: the track exists and its `mid`
/// is negotiated, so a caller matching attachments against its own
/// `add_track` calls has to see it — it just has nothing to send or
/// receive on yet.
#[test]
fn an_inactive_track_attaches_with_no_endpoints() {
    let (mut peer, handle) = idle_peer("peer");

    peer.attach_track(
        TrackId(7),
        "0".into(),
        MediaKind::Audio,
        Direction::Inactive,
    );

    let attached = handle
        .next_track()
        .expect("an inactive track should still attach");
    assert_eq!(attached.id, TrackId(7));
    assert_eq!(attached.kind, MediaKind::Audio);
    assert!(matches!(attached.endpoints, TrackEndpoints::Inactive));
}

/// The other half of `stopping_a_peer_ends_its_inbound_track_source_with_a_clean_eos`:
/// what the peer that *survives* observes. Both endpoints it was handed
/// have to learn that the connection is over — the sink because there is
/// nowhere left to send, the source because its stream has ended — and
/// each says so in its own terms rather than going quiet.
///
/// This remains the end-to-end counterpart to
/// `a_remote_closed_event_ends_the_peer_and_its_inbound_source`. Loopback
/// can also fail a later UDP receive through ICMP, so the direct event test
/// above is what proves the close-notify mechanism independently of that
/// network side effect.
#[test]
fn a_peer_that_dies_notifies_the_surviving_peer_on_both_endpoints() {
    let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();

    let (offer_tx, offer_rx) = crossbeam_channel::unbounded::<SdpOffer>();
    let (peer_a, handle_a) = WebRtcPeer::new(
        "peer-a",
        rtc_a,
        socket_a,
        move |offer| {
            let _ = offer_tx.send(offer);
        },
        |_id| {},
    );
    let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});
    let driver_a = DriverRunner::new(peer_a);
    let driver_b = DriverRunner::new(peer_b);
    driver_a.run().unwrap();
    driver_b.run().unwrap();
    thread::sleep(Duration::from_millis(200));

    handle_a
        .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
        .expect("running peer should accept AddTrack");
    let (mut sink_a, source_a) = send_recv(
        handle_a
            .next_track()
            .expect("peer-a's own track should attach"),
    );
    let offer = offer_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("peer-a should generate a renegotiation offer");
    let answer = handle_b
        .accept_remote_offer(offer)
        .expect("peer-b should accept the offer");
    handle_a.set_answer(answer);
    let (_sink_b, _source_b) = send_recv(
        handle_b
            .next_track()
            .expect("peer-b's remote track should attach"),
    );
    thread::sleep(Duration::from_millis(200));

    // peer-a's inbound half runs in its own `Pipeline`, the way a real
    // caller drives it, so an `Eos` shows up on that pipeline's bus.
    let received = Arc::new(AtomicUsize::new(0));
    let track_pipeline_a = wire_counting(source_a, received);
    track_pipeline_a.run().unwrap();

    driver_b.stop();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut sink_errored = false;
    let mut source_ended = false;
    while Instant::now() < deadline && !(sink_errored && source_ended) {
        if !sink_errored {
            let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
            packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
            packet.set_pts(Some(0));
            sink_errored = sink_a
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .is_err();
        }
        while let Some(event) = track_pipeline_a.bus().try_recv() {
            if matches!(event, BusEvent::Eos { .. }) {
                source_ended = true;
            }
        }
        thread::sleep(Duration::from_millis(50));
    }

    assert!(
        sink_errored,
        "the surviving peer's sink kept accepting buffers with nowhere to send them"
    );
    assert!(
        source_ended,
        "the surviving peer's source never ended, so its pipeline would wait forever"
    );

    track_pipeline_a.stop();
    driver_a.stop();
}
