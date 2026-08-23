//! Connects two `WebRtcPeer`s over real loopback UDP with no browser or
//! signaling server. One `Direction::SendRecv` track carries packets both
//! ways, and each peer drives its inbound `WebRtcTrackSource -> CountingSink`
//! pipeline independently.
//!
//! The initial ICE/DTLS connection uses a bootstrap data channel directly
//! through str0m. A real application would carry the same offer and answer
//! over its own HTTP, WebSocket, or other signaling transport.
//!
//!     cargo run -p webrtc_loopback

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{
        net::UdpSocket,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    use media_pp::ffmpeg;
    use media_pp::pp_log::PpLog;
    use media_pp::{
        Result,
        buffer::MediaBuffer,
        control::ControlMsg,
        driver::DriverRunner,
        element::{Element, ElementType, Sink, element_pp_log},
        elements::{AttachedTrack, TrackEndpoints, WebRtcPeer, WebRtcTrackSink, WebRtcTrackSource},
        pipeline::Pipeline,
    };
    use str0m::{
        Candidate, Rtc,
        change::SdpOffer,
        format::Codec,
        media::{Direction, MediaKind},
    };

    pub(super) fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

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
        println!("initial connection negotiated (bootstrap data channel only, no media yet)");

        let (offer_tx, offer_rx) = std::sync::mpsc::channel::<SdpOffer>();

        let (peer_a, handle_a) = WebRtcPeer::new(
            "peer-a",
            rtc_a,
            socket_a,
            move |offer| {
                println!("peer-a: generated a renegotiation offer for its new track");
                let _ = offer_tx.send(offer);
            },
            |_id| {},
        );
        let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});

        // `WebRtcPeer` is a `Driver`, not a `Pipeline` source — no wiring
        // closure needed; inbound tracks attach dynamically via `next_track()`
        // instead, each into its own `Pipeline`.
        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run()?;
        driver_b.run()?;

        thread::sleep(Duration::from_millis(200));
        println!("ICE/DTLS-SRTP established over loopback UDP");

        let _track_id = handle_a
            .add_track(MediaKind::Video, Direction::SendRecv, Codec::Vp8)
            .expect("running peer should accept AddTrack");
        // `next_track()` returns for peer-a's own track the moment `add_track`'s
        // negotiation mints a Mid — before the offer even leaves this process.
        let attached_a = handle_a
            .next_track()
            .expect("peer-a's own track should attach");
        let kind = attached_a.kind;
        let (mut sink_a, source_a) = send_recv(attached_a);
        println!("peer-a: track attached ({kind:?}) — got a WebRtcTrackSink to reply on");

        let offer = offer_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("peer-a should generate a renegotiation offer");
        let answer = handle_b
            .accept_remote_offer(offer)
            .expect("peer-b should accept the offer");
        println!("peer-b: accepted the offer, answer relayed back to peer-a");
        handle_a.set_answer(answer);
        // peer-b's track attaches as part of accepting the offer, independent
        // of the answer round-trip.
        let attached_b = handle_b
            .next_track()
            .expect("peer-b's remote track should attach");
        let kind = attached_b.kind;
        let (mut sink_b, source_b) = send_recv(attached_b);
        println!(
            "peer-b: inbound codecs={:?}, outbound codecs={:?}",
            source_b.negotiated_codecs(),
            sink_b.negotiated_codecs()
        );
        // This endpoint belongs to a track peer-a added, so peer-b selects
        // its outbound codec from the negotiated capability list.
        sink_b
            .set_codec(Codec::Vp8)
            .expect("VP8 should be negotiated for peer-b's outbound half");
        println!("peer-b: track attached ({kind:?}) — got a WebRtcTrackSink to reply on");

        let received_by_a = Arc::new(AtomicUsize::new(0));
        let received_by_b = Arc::new(AtomicUsize::new(0));
        let track_pipeline_a = wire_counting(source_a, received_by_a.clone());
        let track_pipeline_b = wire_counting(source_b, received_by_b.clone());
        track_pipeline_a.run()?;
        track_pipeline_b.run()?;

        thread::sleep(Duration::from_millis(100)); // let the answer actually apply

        println!("\n-- pushing peer-a -> peer-b on the one shared track --");
        push_packets(&mut sink_a);
        thread::sleep(Duration::from_millis(300));
        println!(
            "peer-b: received {}/5 packets",
            received_by_b.load(Ordering::SeqCst)
        );

        println!("\n-- pushing peer-b -> peer-a on that *same* track, no renegotiation --");
        push_packets(&mut sink_b);
        thread::sleep(Duration::from_millis(300));
        println!(
            "peer-a: received {}/5 packets",
            received_by_a.load(Ordering::SeqCst)
        );

        driver_a.stop();
        driver_b.stop();
        track_pipeline_a.stop();
        track_pipeline_b.stop();

        let count_b = received_by_b.load(Ordering::SeqCst);
        let count_a = received_by_a.load(Ordering::SeqCst);
        assert_eq!(
            count_b, 5,
            "expected every packet peer-a sent to reach peer-b"
        );
        assert_eq!(
            count_a, 5,
            "expected every packet peer-b sent to reach peer-a"
        );
        println!("\nok");
        Ok(())
    }

    /// Wires `source` into its own `Pipeline`, forwarding every packet it
    /// produces into a `CountingSink` that increments `count` — a
    /// `WebRtcTrackSource` is a plain `SourceElement`, so it plugs into
    /// `Pipeline`/`ChainBuilder` exactly like any other source.
    /// Both halves of the one `Direction::SendRecv` track this example
    /// opens. `TrackEndpoints` carries only what the negotiated direction
    /// permits, so this is where "we asked for SendRecv" turns back into a
    /// sink and a source — and where anything else would fail loudly
    /// instead of leaving a half that silently does nothing.
    fn send_recv(track: AttachedTrack) -> (WebRtcTrackSink, WebRtcTrackSource) {
        let TrackEndpoints::SendRecv(sink, source) = track.endpoints else {
            panic!("this example opens one SendRecv track and expects both halves back");
        };
        (sink, source)
    }

    fn wire_counting(source: WebRtcTrackSource, count: Arc<AtomicUsize>) -> Arc<Pipeline> {
        let sink = CountingSink {
            count,
            pp_log: element_pp_log(ElementType::Other, "counter", None),
        };
        Pipeline::new("webrtc-loopback", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("WebRTC track pipeline wiring must succeed")
    }

    fn push_packets(sink: &mut WebRtcTrackSink) {
        for i in 0..5 {
            let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
            packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
            packet.set_pts(Some(i * 3_000));
            sink.consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("push");
        }
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
}
