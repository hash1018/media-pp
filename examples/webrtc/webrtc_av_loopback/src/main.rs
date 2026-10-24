//! Proves that one `WebRtcPeer` connection can carry video and audio tracks at
//! the same time. Peer-a sends both through one `WebRtcMuxer` — one
//! renegotiation for the two — encoding `TestVideoSource` and
//! `TestAudioSource` in one multi-source pipeline, and peer-b takes them as
//! one `WebRtcSource` with an output each, counting the two received packet
//! streams independently.
//!
//! Send:
//! `TestVideoSource -> Queue -> SwEncoder -> WebRtcMuxer` and
//! `TestAudioSource -> Queue -> SwAudioEncoder -> WebRtcMuxer`.
//!
//! Receive: one pipeline of a `WebRtcSource` whose video output goes to one
//! `CountingSink` and audio output to another.
//!
//!     cargo run -p webrtc_av_loopback

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
    use media_pp::{
        buffer::MediaBuffer,
        driver::DriverRunner,
        element::Sink,
        elements::{
            AudioCodec, StreamInfo, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder,
            SwEncoderOptions, TestAudioOptions, TestAudioSource, TestVideoOptions, TestVideoSource,
            VideoCodec, WebRtcPeer,
        },
        pipeline::{Pipeline, PipelineBuilder},
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

        let (offer_tx, offer_rx) = std::sync::mpsc::channel::<SdpOffer>();

        let (peer_a, handle_a) = WebRtcPeer::new(
            "peer-a",
            rtc_a,
            socket_a,
            move |offer| {
                let _ = offer_tx.send(offer);
            },
            |_track| {},
        );
        let (peer_b, handle_b) =
            WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_track| {});

        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run()?;
        driver_b.run()?;

        thread::sleep(Duration::from_millis(200));
        println!("ICE/DTLS-SRTP established over loopback UDP");

        // -- The encoders first: what they make is what the muxer's tracks
        // declare.
        let video_options = TestVideoOptions {
            width: 320,
            height: 240,
            frame_rate: ffmpeg::Rational::new(15, 1),
        };
        let video_source = TestVideoSource::new("video", video_options);
        let video_encoder = SwEncoder::new(
            "video-encoder",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: video_options.width,
                height: video_options.height,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate: video_options.frame_rate,
                bit_rate: 500_000,
                gop_size: 30, // ~2s @ 15fps
                max_b_frames: None,
            },
        )?;
        let audio_options = TestAudioOptions::default();
        let audio_source = TestAudioSource::new("audio", audio_options);
        let audio_encoder = SwAudioEncoder::new(
            "audio-encoder",
            SwAudioEncoderOptions {
                // Opus is the only audio codec WebRTC negotiates at all.
                codec: AudioCodec::Opus,
                sample_rate: audio_options.sample_rate,
                channels: audio_options.channels,
                bit_rate: 64_000,
            },
        )?;

        // -- Both tracks in one muxer: one renegotiation offers the two. Each
        // stream is the whole declaration of its track — the payload type,
        // and for H.264 the SPS/PPS that live in `parameters()` rather than
        // in the bitstream, which the sender puts in front of every keyframe
        // itself. Opus needs nothing of the kind.
        let mut muxer = handle_a.muxer("peer-a-send");
        let video_track = muxer.add_stream(
            "video",
            StreamInfo::new(video_encoder.parameters(), video_encoder.time_base()),
        )?;
        let audio_track = muxer.add_stream(
            "audio",
            StreamInfo::new(audio_encoder.parameters(), audio_encoder.time_base()),
        )?;
        let mut sinks = muxer.open()?;
        let offer = offer_rx.recv_timeout(Duration::from_secs(2))?;
        let answer = handle_b.accept_remote_offer(offer)?;
        handle_a.set_answer(answer);
        println!("peer-a: video and audio offered in one renegotiation, peer-b answered");
        thread::sleep(Duration::from_millis(100)); // let the answer actually apply

        // -- Send side: TestVideoSource/TestAudioSource, each through its own
        // real encoder, onto its track's sink. Two independent sources
        // feeding one `Pipeline` — same shape `screen_record_av` uses for two
        // *capture* sources.
        let video_sink = sinks.take(video_track)?;
        let audio_sink = sinks.take(audio_track)?;
        let builder = PipelineBuilder::new("peer-a-send");
        let (builder, ()) = builder.add_source(video_source, |source, ctx| {
            let branch = ctx
                .branch()
                .queue("encode-video", 4)
                .pipe(video_encoder)
                .to(video_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
        let (builder, ()) = builder.add_source(audio_source, |source, ctx| {
            let branch = ctx
                .branch()
                .queue("encode-audio", 4)
                .pipe(audio_encoder)
                .to(audio_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
        let send_pipeline = builder.build();
        send_pipeline.run()?;

        // -- Receive side: the tracks one offer added come out as one
        // source, an output each, once each has said what it carries — the
        // video track by its first keyframe's parameter sets.
        let (source_b, streams) = handle_b.next_source("peer-b-recv")?;
        let output_of = |kind| {
            streams
                .iter()
                .find(|stream| stream.kind() == kind)
                .map(StreamInfo::index)
                .ok_or("peer-b is missing a track peer-a sent")
        };
        let video_output = output_of(ffmpeg::media::Type::Video)?;
        let audio_output = output_of(ffmpeg::media::Type::Audio)?;
        println!(
            "peer-b: one source of {} tracks (video output {video_output}, audio output {audio_output})",
            streams.len()
        );

        let video_count = Arc::new(AtomicUsize::new(0));
        let audio_count = Arc::new(AtomicUsize::new(0));
        let (recv_pipeline, ()) = Pipeline::new("peer-b-recv", source_b, {
            let video_count = video_count.clone();
            let audio_count = audio_count.clone();
            move |source, ctx| {
                let video = ctx
                    .branch()
                    .to(CountingSink { count: video_count }.named("video-counter"))?;
                ctx.attach(source, video_output, video)?;
                let audio = ctx
                    .branch()
                    .to(CountingSink { count: audio_count }.named("audio-counter"))?;
                ctx.attach(source, audio_output, audio)?;
                Ok(())
            }
        })?;
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
        count: Arc<AtomicUsize>,
    }

    impl Sink for CountingSink {
        fn render(&mut self, buf: MediaBuffer) -> media_pp::Result<()> {
            if matches!(buf, MediaBuffer::Packet(_)) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }
}
