use std::{
    net::UdpSocket,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crate::pp_log::PpLog;
use crossbeam_channel::{Receiver, bounded, unbounded};
use ffmpeg_next as ffmpeg;
use str0m::{
    Candidate, Event, Rtc,
    change::SdpOffer,
    format::{Codec, CodecSpec, FormatParams},
    media::{Direction, Frequency, MediaAdded, MediaChanged, MediaKind, Mid},
};

use super::command::Command;
use super::inbox::{Delivered, Inbox};
use super::outbox::{Outbox, Outgoing};
use super::peer::{Arriving, packet_rtp_time};
use super::stream_info::WebRtcStreamInfo;
use super::track::WebRtcTrackSink;
use super::{TrackId, WebRtcError, WebRtcHandle, WebRtcMuxer, WebRtcPeer, WebRtcSource};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    driver::DriverRunner,
    element::{BoxSink, Element, ElementType, IntoSource, RawSink, element_pp_log},
    elements::{
        CounterHandle, FileDemuxer, FrameCounter, MuxerSinks, StreamInfo, SwDecoder, SwEncoder,
        SwEncoderOptions, TestVideoOptions, TestVideoSource, VideoCodec,
    },
    error::Result,
    pipeline::Pipeline,
};

/// A handle whose peer is the test: what it asks lands on the returned
/// receiver, and it has no remote tracks to hand out.
fn command_only_handle(capacity: usize) -> (WebRtcHandle, Receiver<Command>) {
    let (command_tx, command_rx) = bounded(capacity);
    let (_sources_tx, sources_rx) = unbounded();
    (
        WebRtcHandle {
            next_id: Arc::new(AtomicU64::new(0)),
            command_tx,
            sources_rx,
        },
        command_rx,
    )
}

/// What a sink put in its outbox, one at a time, as the peer takes it.
struct Pushes(
    Outbox,
    std::cell::RefCell<std::collections::VecDeque<Outgoing>>,
);

/// One packet a sink pushed: its track, its codec, the packet.
struct Push(TrackId, Option<Codec>, MediaBuffer);

impl Pushes {
    fn recv(&self) -> std::result::Result<Push, &'static str> {
        let mut taken = self.1.borrow_mut();
        if taken.is_empty() {
            taken.extend(self.0.take_all());
        }
        let outgoing = taken.pop_front().ok_or("nothing was pushed")?;
        // As the connection writes it.
        let packet = super::track::rewritten_packet(&outgoing.packet, &outgoing.payload());
        Ok(Push(outgoing.track, Some(outgoing.codec), packet))
    }
}

/// An outbox for a sink, and how a test reads what it pushes.
fn outbox_of(capacity: usize) -> (Outbox, Pushes) {
    let outbox = Outbox::new(capacity);
    (outbox.clone(), Pushes(outbox, Default::default()))
}

/// A stream of `id`, of `kind`, with no extradata: what a sender that
/// writes its parameter sets in-band declares — as `push_packets`' do.
fn stream_of(id: ffmpeg::codec::Id, kind: ffmpeg::media::Type) -> StreamInfo {
    let mut parameters = parameters_for(id, &[]);
    // SAFETY: `parameters` is a live `AVCodecParameters` this test owns.
    unsafe {
        (*parameters.as_mut_ptr()).codec_type = kind.into();
    }
    StreamInfo::new(parameters, ffmpeg::Rational::new(1, 90_000))
}

fn h264_stream() -> StreamInfo {
    stream_of(ffmpeg::codec::Id::H264, ffmpeg::media::Type::Video)
}

/// A muxer's tracks reach the peer as one request — one renegotiation —
/// each with its kind, its codec and the name a keyframe request for it is
/// reported by; the muxer opens with the sinks the peer answers.
#[test]
fn a_muxers_tracks_are_asked_for_together() {
    let (handle, command_rx) = command_only_handle(1);
    let mut muxer = handle.muxer("camera");
    let video = muxer.add_stream("video", h264_stream()).unwrap();
    let audio = muxer
        .add_stream(
            "audio",
            stream_of(ffmpeg::codec::Id::OPUS, ffmpeg::media::Type::Audio),
        )
        .unwrap();
    let opening = thread::spawn(move || muxer.open());

    let Command::OpenTracks(tracks, reply) = command_rx.recv().expect("the muxer asks the peer")
    else {
        panic!("expected OpenTracks");
    };
    let asked: Vec<_> = tracks
        .iter()
        .map(|track| (track.kind, track.codec, track.name.to_string()))
        .collect();
    assert_eq!(
        asked,
        [
            (MediaKind::Video, Codec::H264, "video".to_owned()),
            (MediaKind::Audio, Codec::Opus, "audio".to_owned()),
        ]
    );
    assert_ne!(tracks[0].id, tracks[1].id);
    assert!(
        command_rx.try_recv().is_err(),
        "two tracks are one request, not two renegotiations"
    );

    let (outbox, _pushes) = outbox_of(4);
    let sinks = tracks
        .iter()
        .map(|track| {
            WebRtcTrackSink::new(
                track.id,
                track.kind,
                Some(track.codec),
                Arc::new(std::sync::Mutex::new(vec![track.codec])),
                outbox.clone(),
            )
        })
        .collect();
    reply.send(Ok(sinks)).unwrap();
    let mut sinks = opening
        .join()
        .unwrap()
        .expect("the muxer opens with the peer's sinks");
    let _video = sinks.take(video).expect("the video track has its sink");
    let _audio = sinks.take(audio).expect("the audio track has its sink");
}

#[test]
fn a_muxer_of_a_stopped_peer_fails_closed() {
    let (handle, command_rx) = command_only_handle(1);
    drop(command_rx);
    let mut muxer = handle.muxer("camera");
    let _track = muxer.add_stream("video", h264_stream()).unwrap();

    assert!(matches!(
        muxer.open(),
        Err(crate::Error::WebRtcError(WebRtcError::Closed))
    ));
}

/// The peer stopping between taking a muxer's request and attaching its
/// tracks lets go of the reply, which the muxer hears as the end it is.
#[test]
fn a_muxer_hears_of_a_peer_that_stops_while_it_opens() {
    let (handle, command_rx) = command_only_handle(1);
    let mut muxer = handle.muxer("camera");
    let _track = muxer.add_stream("video", h264_stream()).unwrap();
    let opening = thread::spawn(move || muxer.open());

    drop(command_rx.recv().expect("the muxer asks the peer"));

    assert!(matches!(
        opening.join().unwrap(),
        Err(crate::Error::WebRtcError(WebRtcError::Closed))
    ));
}

#[test]
fn next_source_returns_closed_once_the_peer_has_stopped() {
    let (handle, _command_rx) = command_only_handle(1);

    assert!(matches!(
        handle.next_source("remote"),
        Err(crate::Error::WebRtcError(WebRtcError::Closed))
    ));
}

/// One track of a batch: its inbox, and a description sent already or —
/// `None` — never coming.
fn arriving(info: Option<WebRtcStreamInfo>) -> (Inbox, Arriving) {
    let inbox = Inbox::new(4);
    let (described_tx, described) = bounded(1);
    if let Some(info) = info {
        described_tx.send(info).unwrap();
    }
    (inbox.clone(), Arriving { inbox, described })
}

fn vp8_info() -> WebRtcStreamInfo {
    WebRtcStreamInfo::from(CodecSpec {
        codec: Codec::Vp8,
        clock_rate: Frequency::NINETY_KHZ,
        channels: None,
        format: FormatParams::default(),
    })
}

/// A source of the remote peer's tracks has an output for each that said
/// what it carries, and lets go of the inbox of one that did not, which
/// the peer then stops filling.
#[test]
fn a_source_of_a_batch_leaves_out_a_track_that_never_said_what_it_carries() {
    let (silent_inbox, silent) = arriving(None);
    let (said_inbox, said) = arriving(Some(vp8_info()));

    let (_source, streams) = WebRtcSource::arriving("remote".into(), vec![silent, said])
        .expect("one of the tracks said what it carries");

    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].index(), 0);
    assert_eq!(streams[0].parameters().id(), ffmpeg::codec::Id::VP8);
    assert_eq!(
        silent_inbox.push(video_packet(&[1, 2, 3, 4], true)),
        Delivered::Gone
    );
    assert_eq!(
        said_inbox.push(video_packet(&[1, 2, 3, 4], true)),
        Delivered::Taken
    );
}

#[test]
fn a_batch_none_of_which_said_what_it_carries_is_no_source() {
    let (_inbox, silent) = arriving(None);

    assert!(matches!(
        WebRtcSource::arriving("remote".into(), vec![silent]),
        Err(crate::Error::WebRtcError(WebRtcError::NothingDescribed(_)))
    ));
}

#[test]
fn a_track_sink_rejects_packets_until_its_codec_is_declared() {
    let (outbox, _command_rx) = outbox_of(1);
    let negotiated = Arc::new(std::sync::Mutex::new(vec![Codec::Vp8]));
    let mut sink = WebRtcTrackSink::new(
        TrackId(7),
        MediaKind::Video,
        None,
        negotiated,
        outbox.clone(),
    );
    let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
    packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
    packet.set_pts(Some(0));

    let error = sink
        .consume(MediaBuffer::Packet(Arc::new(packet).into()))
        .expect_err("a sink must not guess its outbound payload type");

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
fn h264_sink(id: u64, capacity: usize) -> (WebRtcTrackSink, Pushes) {
    video_sink(id, capacity, Codec::H264)
}

fn video_sink(id: u64, capacity: usize, codec: Codec) -> (WebRtcTrackSink, Pushes) {
    let (outbox, command_rx) = outbox_of(capacity);
    let sink = WebRtcTrackSink::new(
        TrackId(id),
        MediaKind::Video,
        Some(codec),
        Arc::new(std::sync::Mutex::new(vec![codec])),
        outbox,
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
    MediaBuffer::Packet(Arc::new(packet).into())
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
            let Push(TrackId(9), _, MediaBuffer::Packet(packet)) =
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
            let Push(TrackId(11), _, MediaBuffer::Packet(packet)) =
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

/// An MP4's HEVC comes the same way, its configuration in `hvcC`: the
/// record's parameter sets go in front of a keyframe, and every packet is
/// rewritten as Annex-B — what the WebRTC server did for itself before it
/// sent through this sink.
#[test]
fn a_demuxers_hvcc_configuration_rewrites_its_packets_as_annex_b() {
    let (mut sink, command_rx) = video_sink(21, 2, Codec::H265);
    // An hvcC with four-byte lengths and one array: a VPS of two bytes.
    let mut record = vec![1u8; 22];
    record[21] = 0x03;
    record.push(1);
    record.extend_from_slice(&[0x20, 0, 1, 0, 2, 0x40, 1]);
    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::HEVC, &record))
        .expect("H.265 is negotiated for this track");

    // A keyframe that carries its SPS and PPS, as a demuxer's first one does.
    let keyframe: &[u8] = &[
        0, 0, 0, 2, 0x42, 1, 0, 0, 0, 2, 0x44, 1, 0, 0, 0, 2, 0x26, 1,
    ];
    sink.consume(video_packet(keyframe, true))
        .expect("an hvcC-configured sink accepts the packets that record describes");
    sink.consume(video_packet(&[0, 0, 0, 2, 0x02, 1], false))
        .expect("and the ones between keyframes");

    let sent: Vec<Vec<u8>> = (0..2)
        .map(|_| {
            let Push(TrackId(21), _, MediaBuffer::Packet(packet)) =
                command_rx.recv().expect("packet was queued")
            else {
                panic!("expected a packet");
            };
            packet.data().expect("packet has a payload").to_vec()
        })
        .collect();
    assert_eq!(
        sent[0],
        [
            0, 0, 0, 1, 0x40, 1, 0, 0, 0, 1, 0x42, 1, 0, 0, 0, 1, 0x44, 1, 0, 0, 0, 1, 0x26, 1
        ]
    );
    assert_eq!(sent[1], [0, 0, 0, 1, 0x02, 1]);
}

/// A NAL unit of 256 to 511 bytes has a four-byte length that opens
/// `00 00 01` — a start code to the eye. Taken for an Annex-B payload, such a
/// packet was sent as it was, and a receiver read its length as a NAL unit
/// of type 25: one picture in several of an MP4's, found when the WebRTC
/// server began sending through this sink.
#[test]
fn a_length_that_looks_like_a_start_code_is_still_a_length() {
    let (mut sink, command_rx) = h264_sink(22, 1);
    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::H264, &AVCC))
        .expect("H.264 is negotiated for this track");
    let mut nalu = vec![0x41u8];
    nalu.resize(300, 0x9a);
    let payload = [&(nalu.len() as u32).to_be_bytes()[..], &nalu[..]].concat();
    assert_eq!(
        &payload[..3],
        &[0, 0, 1],
        "a length that reads as a start code"
    );
    sink.consume(video_packet(&payload, false))
        .expect("an avcC-configured sink takes the packets that record describes");
    let Push(TrackId(22), _, MediaBuffer::Packet(packet)) =
        command_rx.recv().expect("packet was queued")
    else {
        panic!("expected a packet");
    };
    assert_eq!(
        packet.data().unwrap(),
        [&[0, 0, 0, 1][..], &nalu[..]].concat(),
        "rewritten as Annex-B"
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

    let Push(TrackId(15), _, MediaBuffer::Packet(packet)) =
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

    let Push(TrackId(17), _, MediaBuffer::Packet(packet)) =
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

/// HEVC configuration that is neither Annex-B nor an `hvcC` record this
/// sink can read is refused: storing it would put a decoder configuration
/// into the bitstream in front of every keyframe.
#[test]
fn hevc_configuration_that_is_not_annex_b_is_refused() {
    let (outbox, _command_rx) = outbox_of(1);
    let mut sink = WebRtcTrackSink::new(
        TrackId(15),
        MediaKind::Video,
        Some(Codec::H265),
        Arc::new(std::sync::Mutex::new(vec![Codec::H265])),
        outbox.clone(),
    );

    let error = sink
        .set_source_parameters(&parameters_for(
            ffmpeg::codec::Id::HEVC,
            &[0x01, 0x01, 0x60],
        ))
        .expect_err("a record too short to be an hvcC is neither prepended nor converted");
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
    let (outbox, command_rx) = outbox_of(1);
    let mut sink = WebRtcTrackSink::new(
        TrackId(16),
        MediaKind::Audio,
        None,
        Arc::new(std::sync::Mutex::new(vec![Codec::Opus])),
        outbox.clone(),
    );
    let opus_head = [b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', 1, 2];

    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::OPUS, &opus_head))
        .expect("Opus is negotiated for this track");

    let payload: &[u8] = &[0xfc, 0xff, 0xfe];
    sink.consume(video_packet(payload, true))
        .expect("an Opus packet needs no bitstream handling");
    let Push(TrackId(16), Some(Codec::Opus), MediaBuffer::Packet(packet)) =
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
    let (outbox, command_rx) = outbox_of(2);
    let negotiated = Arc::new(std::sync::Mutex::new(vec![Codec::Opus]));
    let mut sink = WebRtcTrackSink::new(
        TrackId(8),
        MediaKind::Audio,
        Some(Codec::Opus),
        negotiated,
        outbox.clone(),
    );

    for pts in [-312, 648] {
        let mut packet = ffmpeg::Packet::copy(&[1, 2, 3]);
        packet.set_time_base(ffmpeg::Rational::new(1, 48_000));
        packet.set_pts(Some(pts));
        packet.set_dts(Some(pts));
        sink.consume(MediaBuffer::Packet(Arc::new(packet).into()))
            .expect("negative encoder delay should be normalized before RTP");
    }

    for expected in [0, 960] {
        let Push(TrackId(8), Some(Codec::Opus), MediaBuffer::Packet(packet)) =
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
    let (outbox, command_rx) = outbox_of(1);
    let negotiated = Arc::new(std::sync::Mutex::new(vec![Codec::Vp8]));
    let mut sink = WebRtcTrackSink::new(
        TrackId(7),
        MediaKind::Video,
        None,
        negotiated,
        outbox.clone(),
    );

    sink.set_source_parameters(&parameters_for(ffmpeg::codec::Id::VP8, &[]))
        .expect("VP8 is negotiated for this track");
    let error = sink
        .set_source_parameters(&parameters_for(ffmpeg::codec::Id::H264, &[]))
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
    sink.consume(MediaBuffer::Packet(Arc::new(packet).into()))
        .expect("the previous VP8 selection should still be usable");
    let Push(TrackId(7), codec, _) = command_rx.recv().expect("packet should be queued") else {
        panic!("expected Push command");
    };
    assert_eq!(codec, Some(Codec::Vp8));
}

/// The other kind of source a sink is fed from, end to end: a container's
/// own packets, which no example sends and nothing but a unit test had ever
/// put on the wire.
///
/// Everything about this path differs from an encoder's. The packets are
/// length-prefixed rather than Annex-B, so each one is rewritten; the
/// parameter sets come out of an `avcC` record rather than from Annex-B
/// extradata; and the far side has to end up with a stream it can decode
/// all the same. `next_source` returning at all is the load-bearing
/// assertion — for H.264 it waits for real SPS/PPS to cross RTP, which is
/// precisely what a mis-framed payload would never deliver.
#[test]
fn a_demuxers_own_packets_reach_a_peer_and_decode() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let pair = running_pair();

    let (demuxer, streams) = FileDemuxer::open("demux", &path).expect("test video should open");
    let video = streams
        .iter()
        .find(|stream| stream.kind() == ffmpeg::media::Type::Video)
        .expect("test video has a video stream")
        .clone();
    let index = video.index();
    // The whole declaration, from the container rather than an encoder: the
    // payload type, the parameter sets the record holds, and the fact that
    // the packets behind it are length-prefixed.
    let mut muxer = pair.handle_a.muxer("demux-send");
    let track = muxer
        .add_stream("video", video)
        .expect("a container's H.264 is a stream WebRTC carries");
    let sink = open_toward(muxer, &pair.handle_a, &pair.offers_a, &pair.handle_b)
        .take(track)
        .unwrap();
    // What is sent before the answer applies is let go of.
    thread::sleep(Duration::from_millis(100));

    // Unpaced deliberately: this sends as fast as the file reads and is
    // stopped as soon as the far side has decoded something, so the test
    // does not run for the length of whatever fixture it was given.
    let (send, ()) = Pipeline::new("demux-send", demuxer, move |source, ctx| {
        let branch = ctx.branch().queue("packets", 32).to(sink)?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("send pipeline should wire");
    send.run().expect("send pipeline should start");

    let (source, received) = pair
        .handle_b
        .next_source("demux-recv")
        .expect("the peer should observe H.264 parameter sets crossing RTP");
    assert_eq!(received[0].parameters().id(), ffmpeg::codec::Id::H264);

    let decoder = SwDecoder::new("recv-decode", received[0].parameters().clone())
        .expect("H.264 decoder should open");
    let (counter, decoded) = FrameCounter::new("decoded");
    let (receive, ()) = Pipeline::new("demux-recv", source, |source, ctx| {
        let branch = ctx.branch().pipe(decoder).to(counter)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("receive pipeline should wire");
    receive.run().expect("receive pipeline should start");
    wait_for_frames(&decoded);

    send.stop();
    pair.driver_a.stop();
    pair.driver_b.stop();
    receive.stop();

    let send_events: Vec<_> = send.bus().iter().collect();
    assert!(
        !send_events
            .iter()
            .any(|event| matches!(event, BusEvent::Error { .. })),
        "unexpected error event(s) while sending a demuxer's packets: {send_events:?}"
    );
    let receive_events: Vec<_> = receive.bus().iter().collect();
    // Not "no errors at all", because this receiver joins a stream that is
    // already in flight and cannot do otherwise. The decoder is built from
    // parameter sets *observed crossing RTP*, which is the point of the
    // test — so it cannot exist until packets have been flowing, and the
    // sender is unpaced by design. The first packets it then sees are inter
    // frames whose references went out while nothing was listening, and
    // FFmpeg answers those with `Invalid data found` until the next IDR.
    //
    // That the decoder recovers is already asserted, above, by
    // `wait_for_frames`. What is asserted here is that nothing *else* went
    // wrong. Everything on the RTP path — reading the track, turning what
    // arrives back into packets, pushing them on — is `WebRtcPeer`'s own
    // work and reports under that, so a fault there is still in this list
    // and still fails.
    let unexpected: Vec<_> = receive_events
        .iter()
        .filter(|event| {
            matches!(
                event,
                BusEvent::Error { element_type, .. } if *element_type != ElementType::SwDecoder
            )
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "unexpected error event(s) while receiving them: {unexpected:?}"
    );
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

impl RawSink for CountingSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if matches!(buf, MediaBuffer::Packet(_)) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
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

/// Two peers running over a connected pair, the renegotiation offers each
/// makes caught for the test to carry to the other, as an application's
/// signalling would.
struct Running {
    driver_a: Arc<DriverRunner>,
    handle_a: WebRtcHandle,
    offers_a: Receiver<SdpOffer>,
    driver_b: Arc<DriverRunner>,
    handle_b: WebRtcHandle,
    offers_b: Receiver<SdpOffer>,
}

fn running_pair() -> Running {
    let (rtc_a, socket_a, rtc_b, socket_b) = connected_pair();
    let (offer_a, offers_a) = unbounded();
    let (offer_b, offers_b) = unbounded();
    let (peer_a, handle_a) = WebRtcPeer::new(
        "peer-a",
        rtc_a,
        socket_a,
        move |offer| {
            let _ = offer_a.send(offer);
        },
        |_track| {},
    );
    let (peer_b, handle_b) = WebRtcPeer::new(
        "peer-b",
        rtc_b,
        socket_b,
        move |offer| {
            let _ = offer_b.send(offer);
        },
        |_track| {},
    );
    let driver_a = DriverRunner::new(peer_a);
    let driver_b = DriverRunner::new(peer_b);
    driver_a.run().unwrap();
    driver_b.run().unwrap();
    // Let ICE/DTLS actually finish over the real loopback sockets before
    // renegotiating.
    thread::sleep(Duration::from_millis(200));
    Running {
        driver_a,
        handle_a,
        offers_a,
        driver_b,
        handle_b,
        offers_b,
    }
}

/// Opens `muxer` over `from`'s peer, and carries the offer that makes to
/// `to`'s peer and its answer back.
fn open_toward(
    muxer: WebRtcMuxer,
    from: &WebRtcHandle,
    offers: &Receiver<SdpOffer>,
    to: &WebRtcHandle,
) -> MuxerSinks {
    let sinks = muxer.open().expect("a running peer opens a muxer's tracks");
    let offer = offers
        .recv_timeout(Duration::from_secs(2))
        .expect("opening the muxer makes an offer");
    let answer = to
        .accept_remote_offer(offer)
        .expect("the other peer accepts the offer");
    from.set_answer(answer);
    sinks
}

/// The sink of one H.264 track `from` sends `to`.
fn h264_toward(from: &WebRtcHandle, offers: &Receiver<SdpOffer>, to: &WebRtcHandle) -> BoxSink {
    let mut muxer = from.muxer("send");
    let track = muxer.add_stream("video", h264_stream()).unwrap();
    open_toward(muxer, from, offers, to).take(track).unwrap()
}

/// `to`'s source of what `sink` sends, taken while `push_packets` keeps
/// sending on it: the source is not made until its track has said what it
/// carries, which an H.264 track says with a keyframe's parameter sets.
/// Sent round after round, as a stream sends its keyframes — sent once,
/// after a fixed wait for the answer to apply, a keyframe was lost on a
/// loaded machine and the peer never saw the parameter sets.
fn source_while_sending(
    to: &WebRtcHandle,
    name: &str,
    sink: &mut dyn RawSink,
) -> (WebRtcSource, Vec<StreamInfo>) {
    let taking = {
        let to = to.clone();
        let name = name.to_owned();
        thread::spawn(move || to.next_source(name))
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut round = 0;
    while !taking.is_finished() {
        assert!(Instant::now() < deadline, "the remote track never came");
        push_packets(sink, round);
        round += 1;
        thread::sleep(Duration::from_millis(100));
    }
    taking
        .join()
        .unwrap()
        .expect("the remote track should say what it carries")
}

/// Five packets, the first a keyframe with its parameter sets, stamped on
/// from where the `round` before ended — sent again round after round, as a
/// stream sends its keyframes, until the peer has seen one.
fn push_packets(sink: &mut dyn RawSink, round: i64) {
    for i in round * 5..round * 5 + 5 {
        // `write_track` needs a real time base to build str0m's own
        // `MediaTime` from — a bare `Packet::copy` defaults to 0/0,
        // which is silently unusable (dropped, not an error).
        let payload: &[u8] = if i % 5 == 0 {
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
        sink.consume(MediaBuffer::Packet(Arc::new(packet).into()))
            .expect("push");
    }
}

fn wait_for_frames(frames: &CounterHandle) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if frames.get() > 0 {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("expected at least one decoded frame");
}

/// Wires `source`'s first output into its own `Pipeline`, forwarding every
/// packet into a `CountingSink` that increments `count` — standing in for
/// the real, independent downstream pipeline a source of a connection's
/// tracks is meant to be driven from.
fn wire_counting<M>(source: impl IntoSource<M>, count: Arc<AtomicUsize>) -> Arc<Pipeline> {
    let sink = CountingSink {
        count,
        pp_log: element_pp_log(ElementType::Other, "counter", None),
    };
    let (pipeline, ()) = Pipeline::new("test", source, |source, ctx| {
        let branch = ctx.branch().to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");
    pipeline
}

/// Each peer sends its own H.264 track through a muxer of its own and
/// receives the other's as a source: two renegotiations, one each way,
/// relayed by hand (standing in for a real signaling transport — see
/// `on_offer`'s docs), over real ICE/DTLS-SRTP on loopback UDP, each source
/// driven by its own independent `Pipeline`. What peer-a sends is pushed by
/// hand and counted; what peer-b sends comes from a real OpenH264 encoder
/// and is decoded, which a mislabeled payload would fail.
#[test]
fn each_peer_sends_its_own_track_and_receives_the_others() {
    let pair = running_pair();
    let mut sink_a = h264_toward(&pair.handle_a, &pair.offers_a, &pair.handle_b);

    let video_options = TestVideoOptions {
        width: 160,
        height: 120,
        frame_rate: ffmpeg::Rational::new(15, 1),
    };
    let video_source = TestVideoSource::new("peer-b-video", video_options);
    let encoder = SwEncoder::new(
        "peer-b-h264",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: video_options.width,
            height: video_options.height,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: video_options.frame_rate,
            bit_rate: 250_000,
            gop_size: 15,
            max_b_frames: None,
        },
    )
    .expect("OpenH264 encoder should open");
    // The encoder's stream is the whole declaration of peer-b's track: the
    // payload type, and the SPS/PPS the encoder keeps in `parameters()`
    // rather than in the bitstream. Without the latter peer-a never sees
    // them and its `next_source` below never returns.
    let mut muxer_b = pair.handle_b.muxer("peer-b-send");
    let track_b = muxer_b
        .add_stream(
            "video",
            StreamInfo::new(encoder.parameters(), encoder.time_base()),
        )
        .unwrap();
    let sink_b = open_toward(muxer_b, &pair.handle_b, &pair.offers_b, &pair.handle_a)
        .take(track_b)
        .unwrap();
    let (send_b, ()) = Pipeline::new("peer-b-h264-send", video_source, |source, ctx| {
        let branch = ctx.branch().pipe(encoder).to(sink_b)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("peer-b send pipeline should wire");
    send_b.run().expect("peer-b send pipeline should start");

    // What each source says it carries comes from the payload str0m
    // actually received, not from either side's declaration; the receive
    // pipelines below also prove those first packets stay queued.
    let (source_a, from_b) = pair
        .handle_a
        .next_source("from-b")
        .expect("peer-a should observe peer-b's H.264 parameter sets");
    let (source_b, from_a) = source_while_sending(&pair.handle_b, "from-a", &mut *sink_a);
    assert_eq!(from_b.len(), 1);
    assert_eq!(from_a.len(), 1);
    assert_eq!(from_b[0].parameters().id(), ffmpeg::codec::Id::H264);
    assert_eq!(from_a[0].parameters().id(), ffmpeg::codec::Id::H264);

    let received_by_b = Arc::new(AtomicUsize::new(0));
    let decoder = SwDecoder::new("peer-a-h264-decode", from_b[0].parameters().clone())
        .expect("peer-a H.264 decoder should open");
    let (counter, decoded_by_a) = FrameCounter::new("decoded-by-a");
    let (track_pipeline_a, ()) = Pipeline::new("peer-a-h264-recv", source_a, |source, ctx| {
        let branch = ctx.branch().pipe(decoder).to(counter)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("peer-a receive pipeline should wire");
    let track_pipeline_b = wire_counting(source_b, received_by_b.clone());
    track_pipeline_a.run().unwrap();
    track_pipeline_b.run().unwrap();
    wait_for_frames(&decoded_by_a);
    thread::sleep(Duration::from_millis(100));
    // Every round that got through, whole: nothing pushed was lost on the
    // way to the pipeline wired after it.
    let received = received_by_b.load(Ordering::SeqCst);
    assert!(
        received >= 5 && received.is_multiple_of(5),
        "peer-b should receive every round peer-a pushed, got {received} packets"
    );

    // Trivial now — `DriverRunner::stop` just flips a flag, no
    // rendezvous ack to race (see `StopReceiver`'s own docs).
    send_b.stop();
    pair.driver_a.stop();
    pair.driver_b.stop();
    // Deliberately also stopped even though each source's `Pipeline` may
    // already be ending on its own by this point (the drivers' stop above
    // closes its inboxes, which ends it with a natural `Eos`): this is a
    // regression guard for a real `Pipeline::stop` bug this exact race used
    // to hit, now fixed at its source (see `Pipeline`'s own `control_rx`
    // field docs).
    track_pipeline_a.stop();
    track_pipeline_b.stop();

    for (what, events) in [
        ("peer-a", pair.driver_a.bus().iter().collect::<Vec<_>>()),
        ("peer-b", pair.driver_b.bus().iter().collect()),
        ("peer-b's outbound encoder", send_b.bus().iter().collect()),
        (
            "peer-a's inbound track",
            track_pipeline_a.bus().iter().collect(),
        ),
        (
            "peer-b's inbound track",
            track_pipeline_b.bus().iter().collect(),
        ),
    ] {
        assert!(
            !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
            "unexpected error event(s) on {what}: {events:?}"
        );
    }
}

/// A muxer's tracks are offered in the order its streams were added, so the
/// remote peer's source of them has its outputs in that order. The peer
/// once listed them in a hash map's order, and a video-then-audio muxer
/// came out of `next_source` as audio then video, as often as not.
#[test]
fn a_muxers_tracks_are_offered_in_the_order_it_was_given_them() {
    let pair = running_pair();
    let kinds = [
        ffmpeg::media::Type::Video,
        ffmpeg::media::Type::Audio,
        ffmpeg::media::Type::Video,
        ffmpeg::media::Type::Audio,
        ffmpeg::media::Type::Video,
        ffmpeg::media::Type::Audio,
    ];
    let mut muxer = pair.handle_a.muxer("send");
    for (index, kind) in kinds.into_iter().enumerate() {
        let id = match kind {
            ffmpeg::media::Type::Video => ffmpeg::codec::Id::VP8,
            _ => ffmpeg::codec::Id::OPUS,
        };
        let _track = muxer
            .add_stream(format!("track-{index}"), stream_of(id, kind))
            .unwrap();
    }
    let _sinks = muxer.open().expect("a running peer opens a muxer's tracks");
    let offer = pair
        .offers_a
        .recv_timeout(Duration::from_secs(2))
        .expect("opening the muxer makes an offer");

    let offered: Vec<_> = offer
        .to_sdp_string()
        .lines()
        .filter_map(|line| line.strip_prefix("m="))
        .filter_map(|media| media.split(' ').next())
        .filter(|media| *media != "application")
        .map(str::to_owned)
        .collect();
    assert_eq!(
        offered,
        ["video", "audio", "video", "audio", "video", "audio"]
    );

    pair.driver_a.stop();
    pair.driver_b.stop();
}

/// Regression test for the claim in `Driver::run`'s own docs: stopping
/// one side must clear its `tracks_in` immediately, so a source of its
/// remote tracks already handed out sees its inboxes close and ends with a
/// final `Eos` on its own — not left hanging until some other, unrelated
/// thing (e.g. a caller's own `Pipeline::stop`) intervenes. There's no
/// "reconnect" concept here to test instead: `WebRtcPeer::run` only ever
/// reacts to its `Rtc` going not-alive or an explicit `Stop`, treating both
/// the same way (see that `if` right before `tracks_in.clear()`); `Stop` is
/// the deterministic one of the two to trigger from a test. Deliberately
/// omits the `track_pipeline.stop()` safety net
/// `each_peer_sends_its_own_track_and_receives_the_others` uses — the whole
/// point here is to prove the `Pipeline` ends on its own.
#[test]
fn stopping_a_peer_ends_its_inbound_track_source_with_a_clean_eos() {
    let pair = running_pair();
    let mut sink_a = h264_toward(&pair.handle_a, &pair.offers_a, &pair.handle_b);
    let (source_b, _) = source_while_sending(&pair.handle_b, "from-a", &mut *sink_a);

    let received = Arc::new(AtomicUsize::new(0));
    let track_pipeline_b = wire_counting(source_b, received);
    track_pipeline_b.run().unwrap();

    pair.driver_b.stop();

    // No `track_pipeline_b.stop()` — draining the bus to completion is
    // itself the proof: it only returns once every `Bus` sender,
    // including the one the source's own `Pipeline::run` thread holds, has
    // actually dropped.
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

    pair.driver_a.stop();
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
/// A `WebRtcPeer` with nothing running on it, for exercising the
/// attach/event handling directly. Driving a *real* direction change
/// end to end would mean hand-rolling one side's whole ICE/DTLS poll
/// loop just to reach `SdpApi::set_direction`; what actually needs
/// covering is this element's own reaction to the event, so the event is
/// what the tests below supply.
fn idle_peer(name: &str) -> (WebRtcPeer, WebRtcHandle) {
    let (rtc, socket, _rtc_b, _socket_b) = connected_pair();
    WebRtcPeer::new(name, rtc, socket, |_offer| {}, |_track| {})
}

/// The remote peer adding `mid`, as str0m reports it — from this side, so
/// a section it sends on is one this side receives on.
fn media_added(mid: &str, kind: MediaKind, direction: Direction) -> Event {
    Event::MediaAdded(MediaAdded {
        mid: mid.into(),
        kind,
        direction,
        simulcast: None,
    })
}

/// What one remote offer adds comes out together, a source's worth — and
/// only what the remote peer sends on: a section it adds asking to be sent
/// to, or one inactive, has nothing here to send on it, so nothing is
/// handed out for it.
#[test]
fn the_tracks_one_remote_offer_adds_come_out_together_and_only_those_it_sends() {
    let (mut peer, handle) = idle_peer("peer");
    let (bus, _bus_rx) = Bus::new();

    peer.handle_event(
        media_added("0", MediaKind::Video, Direction::RecvOnly),
        &bus,
    );
    peer.handle_event(
        media_added("1", MediaKind::Video, Direction::SendOnly),
        &bus,
    );
    peer.handle_event(
        media_added("2", MediaKind::Audio, Direction::Inactive),
        &bus,
    );
    peer.handle_event(
        media_added("3", MediaKind::Audio, Direction::SendRecv),
        &bus,
    );
    peer.hand_out_arrivals();
    peer.handle_event(
        media_added("4", MediaKind::Audio, Direction::RecvOnly),
        &bus,
    );
    peer.hand_out_arrivals();

    let first = handle
        .sources_rx
        .try_recv()
        .expect("the first offer's tracks are handed out");
    assert_eq!(first.len(), 2, "the two the remote peer sends on");
    let second = handle
        .sources_rx
        .try_recv()
        .expect("a later offer's tracks are a batch of their own");
    assert_eq!(second.len(), 1);
    assert!(handle.sources_rx.try_recv().is_err());
}

/// `Event::Closed` is the deterministic signal produced by str0m for a
/// received DTLS `close_notify`. Unlike the end-to-end loopback test below,
/// this does not rely on a later UDP send eliciting an ICMP error: the event
/// itself must end the driver and the source of its remote tracks.
#[test]
fn a_remote_closed_event_ends_the_peer_and_its_inbound_source() {
    let (mut peer, handle) = idle_peer("peer");
    let (bus, _bus_rx) = Bus::new();
    peer.handle_event(
        media_added("0", MediaKind::Video, Direction::RecvOnly),
        &bus,
    );
    peer.hand_out_arrivals();
    let mut batch = handle
        .sources_rx
        .try_recv()
        .expect("the inbound track should be handed out");
    // What its first packets would have said, which nothing sends here.
    let (described_tx, described) = bounded(1);
    described_tx.send(vp8_info()).unwrap();
    let inbox = batch.remove(0).inbox;
    let (source, _) = WebRtcSource::arriving("in".into(), vec![Arriving { inbox, described }])
        .expect("the track said what it carries");
    let received = Arc::new(AtomicUsize::new(0));
    let track_pipeline = wire_counting(source, received);
    track_pipeline.run().unwrap();

    peer.handle_event(Event::Closed, &bus);
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
    let mut muxer = handle.muxer("late");
    let _track = muxer.add_stream("video", h264_stream()).unwrap();
    assert!(matches!(
        muxer.open(),
        Err(crate::Error::WebRtcError(WebRtcError::Closed))
    ));
    assert!(matches!(
        handle.next_source("late"),
        Err(crate::Error::WebRtcError(WebRtcError::Closed))
    ));
}

/// A track goes the one way it attached with — here a source's output, the
/// caller's already — so a remote peer renegotiating a different one leaves
/// the caller holding something that no longer describes the connection.
/// Nothing can be re-issued at that point, which is exactly why it has to
/// be reported rather than silently tolerated.
#[test]
fn a_direction_change_after_attaching_is_reported_on_the_bus() {
    let (mut peer, _handle) = idle_peer("peer");
    let (bus, bus_rx) = Bus::new();
    let mid: Mid = "0".into();
    peer.handle_event(
        media_added("0", MediaKind::Video, Direction::RecvOnly),
        &bus,
    );

    peer.handle_event(
        Event::MediaChanged(MediaChanged {
            mid,
            direction: Direction::Inactive,
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
    assert_eq!(from, Direction::RecvOnly);
    assert_eq!(to, Direction::Inactive);
}

/// `MediaChanged` also fires when a renegotiation restates the direction
/// already in force. Reporting that would turn every unrelated
/// renegotiation into a bus error, so only an actual change counts.
#[test]
fn restating_the_direction_a_track_already_has_is_not_reported() {
    let (mut peer, _handle) = idle_peer("peer");
    let (bus, bus_rx) = Bus::new();
    peer.handle_event(
        media_added("0", MediaKind::Video, Direction::RecvOnly),
        &bus,
    );

    peer.handle_event(
        Event::MediaChanged(MediaChanged {
            mid: "0".into(),
            direction: Direction::RecvOnly,
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
    let (bus, bus_rx) = Bus::new();

    peer.handle_event(
        Event::MediaChanged(MediaChanged {
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

/// The other half of `stopping_a_peer_ends_its_inbound_track_source_with_a_clean_eos`:
/// what the peer that *survives* observes. Both what it sends through and
/// what it receives from have to learn that the connection is over — the
/// sink because there is nowhere left to send, the source because its
/// stream has ended — and each says so in its own terms rather than going
/// quiet.
///
/// This remains the end-to-end counterpart to
/// `a_remote_closed_event_ends_the_peer_and_its_inbound_source`. Loopback
/// can also fail a later UDP receive through ICMP, so the direct event test
/// above is what proves the close-notify mechanism independently of that
/// network side effect.
#[test]
fn a_peer_that_dies_notifies_the_surviving_peer_on_both_endpoints() {
    let pair = running_pair();
    let mut sink_a = h264_toward(&pair.handle_a, &pair.offers_a, &pair.handle_b);
    let mut sink_b = h264_toward(&pair.handle_b, &pair.offers_b, &pair.handle_a);
    let (source_a, _) = source_while_sending(&pair.handle_a, "from-b", &mut *sink_b);

    // peer-a's source runs in its own `Pipeline`, the way a real caller
    // drives it, so an `Eos` shows up on that pipeline's bus.
    let received = Arc::new(AtomicUsize::new(0));
    let track_pipeline_a = wire_counting(source_a, received);
    track_pipeline_a.run().unwrap();

    pair.driver_b.stop();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut sink_errored = false;
    let mut source_ended = false;
    while Instant::now() < deadline && !(sink_errored && source_ended) {
        if !sink_errored {
            sink_errored = sink_a.consume(video_packet(&[1, 2, 3, 4], false)).is_err();
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
    pair.driver_a.stop();
}

/// Driven, a source of a connection's tracks waits on their inboxes: what
/// the connection puts in wakes the poll that found nothing, and the
/// connection letting go of the track ends the pipeline.
#[test]
fn a_source_of_tracks_driven_is_woken_by_what_the_connection_receives() {
    use std::task::{Context, Wake, Waker};

    use crate::elements::PacketCounter;
    use crate::pipeline::{Driven, DrivenPipeline};

    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let inbox = Inbox::new(8);
    let (source, _) = WebRtcSource::described(
        "tracks-in".into(),
        vec![(inbox.clone(), Some(vp8_info()))],
        Duration::from_secs(1),
        None,
        element_pp_log(ElementType::WebRtcSource, "tracks-in", None),
    )
    .unwrap();
    let (counter, count) = PacketCounter::new("counter");
    let (mut pipeline, ()) = DrivenPipeline::new("publisher", source, |source, ctx| {
        ctx.attach(source, 0, ctx.branch().to(counter)?)?;
        Ok(())
    })
    .unwrap();
    let wakes = Arc::new(Count(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wakes));
    let mut cx = Context::from_waker(&waker);

    assert_eq!(pipeline.poll(&mut cx), Driven::Pending { at: None });
    inbox.push(video_packet(&[0, 0, 0, 1, 0x65], true));
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1, "a packet wakes it");
    assert_eq!(pipeline.poll(&mut cx), Driven::Pending { at: None });
    assert_eq!(count.get(), 1);
    inbox.close();
    assert_eq!(wakes.0.load(Ordering::SeqCst), 2, "the end wakes it");
    assert_eq!(pipeline.poll(&mut cx), Driven::Finished);
}
