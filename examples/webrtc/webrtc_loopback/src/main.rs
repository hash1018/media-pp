//! Connects two `WebRtcPeer`s over real loopback UDP with no browser or
//! signaling server. Each peer sends one VP8 track through a `WebRtcMuxer`
//! of its own and takes the other's as a `WebRtcSource`, driving its own
//! `WebRtcSource -> CountingSink` pipeline.
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
            mpsc::{self, Receiver},
        },
        thread,
        time::{Duration, Instant},
    };

    use media_pp::ffmpeg;
    use media_pp::pp_log::PpLog;
    use media_pp::{
        Result,
        buffer::MediaBuffer,
        driver::DriverRunner,
        element::{BoxSink, Element, ElementType, RawSink, element_pp_log},
        elements::{StreamInfo, WebRtcHandle, WebRtcPeer, WebRtcSource},
        pipeline::Pipeline,
    };
    use str0m::{Candidate, Rtc, change::SdpOffer};

    pub(super) fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let socket_a = UdpSocket::bind("127.0.0.1:0")?;
        let socket_b = UdpSocket::bind("127.0.0.1:0")?;
        let addr_a = socket_a.local_addr()?;
        let addr_b = socket_b.local_addr()?;

        let mut rtc_a = Rtc::builder().build(Instant::now());
        rtc_a
            .add_local_candidate(Candidate::host(addr_a, "udp")?)
            .expect("add candidate a");
        let mut rtc_b = Rtc::builder().build(Instant::now());
        rtc_b
            .add_local_candidate(Candidate::host(addr_b, "udp")?)
            .expect("add candidate b");

        let mut changes = rtc_a.sdp_api();
        changes.add_channel("bootstrap".to_string());
        let (offer, pending) = changes.apply().expect("adding a channel always offers");
        let answer = rtc_b.sdp_api().accept_offer(offer)?;
        rtc_a.sdp_api().accept_answer(pending, answer)?;
        println!("initial connection negotiated (bootstrap data channel only, no media yet)");

        // Each peer's renegotiation offers, for this program to carry to the
        // other as its signalling would.
        let (offer_a, offers_a) = mpsc::channel::<SdpOffer>();
        let (offer_b, offers_b) = mpsc::channel::<SdpOffer>();
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

        // `WebRtcPeer` is a `Driver`, not a `Pipeline` source: it runs the
        // connection, and what goes through it goes through the muxers and
        // sources its handle makes, each in a pipeline of its own.
        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run()?;
        driver_b.run()?;

        thread::sleep(Duration::from_millis(200));
        println!("ICE/DTLS-SRTP established over loopback UDP");

        let mut sink_a = open_vp8(&handle_a, &offers_a, &handle_b)?;
        println!("peer-a: its track offered, peer-b answered");
        let mut sink_b = open_vp8(&handle_b, &offers_b, &handle_a)?;
        println!("peer-b: its track offered, peer-a answered");
        thread::sleep(Duration::from_millis(100)); // let the answers actually apply

        // Sent before either side has a source of it: what arrives waits in
        // the track's inbox, and is what the source says the track carries
        // from.
        println!("\n-- each peer pushes 5 packets on its own track --");
        push_packets(&mut sink_a);
        push_packets(&mut sink_b);

        let (source_b, streams_b) = handle_b.next_source("from-peer-a")?;
        let (source_a, streams_a) = handle_a.next_source("from-peer-b")?;
        println!(
            "peer-b receives {:?}, peer-a receives {:?}",
            streams_b[0].parameters().id(),
            streams_a[0].parameters().id()
        );

        let received_by_a = Arc::new(AtomicUsize::new(0));
        let received_by_b = Arc::new(AtomicUsize::new(0));
        let pipeline_a = wire_counting(source_a, received_by_a.clone());
        let pipeline_b = wire_counting(source_b, received_by_b.clone());
        pipeline_a.run()?;
        pipeline_b.run()?;
        thread::sleep(Duration::from_millis(300));

        let count_b = received_by_b.load(Ordering::SeqCst);
        let count_a = received_by_a.load(Ordering::SeqCst);
        println!("peer-b: received {count_b}/5 packets");
        println!("peer-a: received {count_a}/5 packets");

        driver_a.stop();
        driver_b.stop();
        pipeline_a.stop();
        pipeline_b.stop();

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

    /// The sink of one VP8 track `from` sends `to`: its muxer opened over
    /// `from`'s connection, the offer that makes carried to `to`, and the
    /// answer back.
    fn open_vp8(
        from: &WebRtcHandle,
        offers: &Receiver<SdpOffer>,
        to: &WebRtcHandle,
    ) -> std::result::Result<BoxSink, Box<dyn std::error::Error>> {
        let mut muxer = from.muxer("send");
        let track = muxer.add_stream(
            "video",
            StreamInfo::video(
                ffmpeg::codec::Id::VP8,
                &[],
                (320, 240),
                ffmpeg::Rational::new(1, 90_000),
            )?,
        )?;
        let mut sinks = muxer.open()?;
        let answer = to.accept_remote_offer(offers.recv_timeout(Duration::from_secs(2))?)?;
        from.set_answer(answer);
        Ok(sinks.take(track)?)
    }

    /// Wires `source` into its own `Pipeline`, forwarding every packet it
    /// produces into a `CountingSink` that increments `count` — a
    /// `WebRtcSource` is a source like any other, so it plugs into
    /// `Pipeline`/`ChainBuilder` the same way.
    fn wire_counting(source: WebRtcSource, count: Arc<AtomicUsize>) -> Arc<Pipeline> {
        let sink = CountingSink {
            count,
            pp_log: element_pp_log(ElementType::Other, "counter", None),
        };
        let (pipeline, ()) = Pipeline::new("webrtc-loopback", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("WebRTC track pipeline wiring must succeed");
        pipeline
    }

    fn push_packets(sink: &mut BoxSink) {
        for i in 0..5 {
            let mut packet = ffmpeg::Packet::copy(&[1, 2, 3, 4]);
            packet.set_time_base(ffmpeg::Rational::new(1, 90_000));
            packet.set_pts(Some(i * 3_000));
            sink.consume(MediaBuffer::Packet(Arc::new(packet).into()))
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

    impl RawSink for CountingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if matches!(buf, MediaBuffer::Packet(_)) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }
}
