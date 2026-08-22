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
    format::Codec,
    media::{Direction, MediaKind, Mid},
};

use super::command::Command;
use super::peer::packet_rtp_time;
use super::{
    AttachedTrack, TrackEndpoints, TrackId, WebRtcError, WebRtcHandle, WebRtcPeer, WebRtcTrackSink,
    WebRtcTrackSource,
};
use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    control::ControlMsg,
    driver::DriverRunner,
    element::{Element, ElementType, Sink, element_pp_log},
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
        let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
        packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
        packet.set_pts(Some(i * 3_000));
        sink.consume(MediaBuffer::Packet(Arc::new(packet)))
            .expect("push");
    }
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

/// One `Direction::SendRecv` track, opened by `WebRtcHandle::add_track`
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
fn one_sendrecv_track_carries_data_both_ways() {
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
        .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
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

    let received_by_a = Arc::new(AtomicUsize::new(0));
    let received_by_b = Arc::new(AtomicUsize::new(0));
    let track_pipeline_a = wire_counting(source_a, received_by_a.clone());
    let track_pipeline_b = wire_counting(source_b, received_by_b.clone());
    track_pipeline_a.run().unwrap();
    track_pipeline_b.run().unwrap();

    // Let the answer actually apply before pushing media through it.
    thread::sleep(Duration::from_millis(100));

    push_packets(&mut sink_a);
    thread::sleep(Duration::from_millis(300));
    assert_eq!(
        received_by_b.load(Ordering::SeqCst),
        5,
        "peer-b should receive everything peer-a pushed"
    );

    push_packets(&mut sink_b);
    thread::sleep(Duration::from_millis(300));
    assert_eq!(
        received_by_a.load(Ordering::SeqCst),
        5,
        "peer-a should receive everything peer-b pushed back, on the same track"
    );

    // Trivial now — `DriverRunner::stop` just flips a flag, no
    // rendezvous ack to race (see `StopReceiver`'s own docs).
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
/// `one_sendrecv_track_carries_data_both_ways` uses — the whole point
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
