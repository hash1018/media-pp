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

    use ffmpeg_next as ffmpeg;
    use media_pp::pp_log::PpLog;
    use media_pp::{
        buffer::MediaBuffer,
        control::ControlMsg,
        driver::DriverRunner,
        element::{Element, ElementType, Sink, element_pp_log},
        elements::{
            AudioCodec, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder, SwEncoderOptions,
            TestAudioOptions, TestAudioSource, TestVideoOptions, TestVideoSource, VideoCodec,
            WebRtcPeer,
        },
        pipeline::PipelineBuilder,
    };
    use str0m::{
        Candidate, Rtc,
        change::SdpOffer,
        format::Codec,
        media::{Direction, MediaKind},
    };

    /// Answers the question that came up while reviewing `WebRtcTrackSink`'s
    /// docs: can *one* `WebRtcPeer` connection carry more than one track at
    /// once? Two `WebRtcPeer`s over loopback UDP (same setup as
    /// `webrtc_loopback`), but peer-a adds *two* tracks — one video, one
    /// audio — onto the *same* connection (two `WebRtcHandle::add_track`
    /// calls, two sequential renegotiations, no second `Rtc`/socket/peer). A
    /// real `TestVideoSource`/`TestAudioSource` are encoded (`SwEncoder`/
    /// `SwAudioEncoder`) and pushed independently onto each track via one
    /// `PipelineBuilder`-built `Pipeline` (two sources — the same shape
    /// `screen_audio_record` uses for two *capture* sources); peer-b receives
    /// both tracks and counts packets on each independently, proving they
    /// don't cross-contaminate.
    ///
    ///     cargo run -p webrtc_av_loopback
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
                let _ = offer_tx.send(offer);
            },
            |_id| {},
        );
        let (peer_b, handle_b) = WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_id| {});

        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run()?;
        driver_b.run()?;

        thread::sleep(Duration::from_millis(200));
        println!("ICE/DTLS-SRTP established over loopback UDP");

        // -- Track 1: video. One full renegotiation round before starting the
        // second track — str0m only allows one SDP exchange in flight at a
        // time.
        handle_a
            .add_track(MediaKind::Video, Direction::SendOnly, Codec::H264)
            .expect("running peer should accept AddTrack");
        let (_id, _mid, kind, video_sink_a, _unused) = handle_a
            .next_track()
            .expect("peer-a's video track should attach");
        println!("peer-a: track attached ({kind:?})");
        let offer = offer_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("peer-a should generate a renegotiation offer for the video track");
        let answer = handle_b
            .accept_remote_offer(offer)
            .expect("peer-b should accept the video offer");
        handle_a.set_answer(answer);
        let (_id, _mid, kind, _unused, video_source_b) = handle_b
            .next_track()
            .expect("peer-b's video track should attach");
        println!("peer-b: track attached ({kind:?})");
        thread::sleep(Duration::from_millis(100)); // let the answer actually apply

        // -- Track 2: audio, on the *same* connection.
        handle_a
            .add_track(MediaKind::Audio, Direction::SendOnly, Codec::Opus)
            .expect("running peer should accept a second AddTrack");
        let (_id, _mid, kind, audio_sink_a, _unused) = handle_a
            .next_track()
            .expect("peer-a's audio track should attach");
        println!("peer-a: track attached ({kind:?})");
        let offer = offer_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("peer-a should generate a renegotiation offer for the audio track");
        let answer = handle_b
            .accept_remote_offer(offer)
            .expect("peer-b should accept the audio offer");
        handle_a.set_answer(answer);
        let (_id, _mid, kind, _unused, audio_source_b) = handle_b
            .next_track()
            .expect("peer-b's audio track should attach");
        println!("peer-b: track attached ({kind:?})");
        thread::sleep(Duration::from_millis(100));

        println!("both tracks negotiated on the same connection — now pushing real encoded media");

        // -- Send side: TestVideoSource/TestAudioSource, each through its own
        // real encoder, straight onto the track `add_track` gave us for it.
        // Two independent sources feeding one `Pipeline` — same shape
        // `screen_audio_record` uses for two *capture* sources.
        let video_options = TestVideoOptions {
            width: 320,
            height: 240,
            framerate: ffmpeg::Rational::new(15, 1),
        };
        let video_source = TestVideoSource::new("video", video_options);
        let video_time_base = video_source.time_base();
        let video_encoder = SwEncoder::new(
            "video-encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: video_options.width,
                height: video_options.height,
                time_base: video_time_base,
                frame_rate: video_options.framerate,
                bit_rate: 500_000,
                gop_size: 30, // ~2s @ 15fps
            },
        )
        .expect("failed to open video encoder");

        let audio_options = TestAudioOptions::default();
        let audio_source = TestAudioSource::new("audio", audio_options);
        let audio_time_base = audio_source.time_base();
        let audio_encoder = SwAudioEncoder::new(
            "audio-encoder",
            SwAudioEncoderOptions {
                codec: AudioCodec::Aac,
                sample_rate: audio_options.sample_rate,
                channels: audio_options.channels,
                time_base: audio_time_base,
                bit_rate: 64_000,
            },
        )
        .expect("failed to open audio encoder");

        let send_pipeline = PipelineBuilder::new("peer-a-send")
            .add_source(video_source, |source, ctx| {
                let branch = ctx
                    .branch()
                    .queue("encode-video", 4)
                    .pipe(video_encoder)
                    .to(Box::new(video_sink_a))?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("video send pipeline wiring must succeed")
            .add_source(audio_source, |source, ctx| {
                let branch = ctx
                    .branch()
                    .queue("encode-audio", 4)
                    .pipe(audio_encoder)
                    .to(Box::new(audio_sink_a))?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("audio send pipeline wiring must succeed")
            .build();

        // -- Receive side: count packets on each track independently.
        let video_count = Arc::new(AtomicUsize::new(0));
        let audio_count = Arc::new(AtomicUsize::new(0));
        let recv_pipeline = PipelineBuilder::new("peer-b-recv")
            .add_source(video_source_b, {
                let count = video_count.clone();
                move |source, ctx| {
                    let branch = ctx.branch().to(Box::new(CountingSink {
                        name: "video-counter".into(),
                        count,
                        pp_log: element_pp_log(ElementType::Other, "video-counter", None),
                    }))?;
                    ctx.attach(source, 0, branch)?;
                    Ok(())
                }
            })
            .expect("video receive pipeline wiring must succeed")
            .add_source(audio_source_b, {
                let count = audio_count.clone();
                move |source, ctx| {
                    let branch = ctx.branch().to(Box::new(CountingSink {
                        name: "audio-counter".into(),
                        count,
                        pp_log: element_pp_log(ElementType::Other, "audio-counter", None),
                    }))?;
                    ctx.attach(source, 0, branch)?;
                    Ok(())
                }
            })
            .expect("audio receive pipeline wiring must succeed")
            .build();

        send_pipeline.run()?;
        recv_pipeline.run()?;

        thread::sleep(Duration::from_secs(2));

        send_pipeline.stop();
        recv_pipeline.stop();
        driver_a.stop();
        driver_b.stop();

        let video_received = video_count.load(Ordering::SeqCst);
        let audio_received = audio_count.load(Ordering::SeqCst);
        println!(
            "\npeer-b received: {video_received} video packets, {audio_received} audio packets"
        );

        assert!(
            video_received > 0,
            "expected at least one video packet on peer-b's video track"
        );
        assert!(
            audio_received > 0,
            "expected at least one audio packet on peer-b's audio track"
        );
        println!("ok — two tracks, one connection, no cross-contamination");
        Ok(())
    }

    struct CountingSink {
        pp_log: PpLog,
        name: Arc<str>,
        count: Arc<AtomicUsize>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            self.name.clone()
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
        fn consume(&mut self, buf: MediaBuffer) -> media_pp::Result<()> {
            if matches!(buf, MediaBuffer::Packet(_)) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> media_pp::Result<()> {
            Ok(())
        }
    }
}
