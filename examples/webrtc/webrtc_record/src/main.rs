//! Sends one file's first video and audio streams as H.264/Opus over two
//! WebRTC tracks, then records both received tracks into one MP4 without
//! decoding or re-encoding on the receiving side.
//!
//! Sender:
//!
//! ```text
//! FileDemuxer(video) -> SwDecoder -> Queue -> Pacer -> SwScaler
//!                    -> SwEncoder(H.264) -> WebRtcTrackSink
//! FileDemuxer(audio) -> SwDecoder -> Queue -> Pacer
//!                    -> SwAudioEncoder(Opus) -> WebRtcTrackSink
//! ```
//!
//! Receiver:
//!
//! ```text
//! WebRtcTrackSource(H.264) -\
//!                            -> Mp4Muxer
//! WebRtcTrackSource(Opus)  --/
//! ```
//!
//!     cargo run -p webrtc_record -- input.mp4 output.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{
        collections::HashSet,
        net::UdpSocket,
        sync::{Arc, mpsc::Receiver},
        thread,
        time::{Duration, Instant},
    };

    use media_pp::ffmpeg;
    use media_pp::{
        bus::{BusEvent, BusReceiver},
        clock::Clock,
        driver::DriverRunner,
        element::Element,
        elements::{
            AudioCodec, FileDemuxer, Mp4Muxer, Pacer, SwAudioEncoder, SwAudioEncoderOptions,
            SwDecoder, SwEncoder, SwEncoderOptions, SwScaler, TrackEndpoints, VideoCodec,
            WebRtcHandle, WebRtcPeer, WebRtcTrackSink, WebRtcTrackSource,
        },
        pipeline::{Pipeline, PipelineBuilder},
    };
    use str0m::{
        Candidate, Rtc,
        change::SdpOffer,
        format::Codec,
        media::{Direction, MediaKind},
    };

    const STREAM_INFO_TIMEOUT: Duration = Duration::from_secs(5);
    const AUDIO_RATE: u32 = 48_000;
    const AUDIO_CHANNELS: u16 = 2;

    pub(super) fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let mut args = std::env::args().skip(1);
        let (Some(input_path), Some(output_path)) = (args.next(), args.next()) else {
            eprintln!("usage: webrtc_record <input-media> <output.mp4>");
            std::process::exit(1);
        };
        let input = FileInput::open(&input_path)?;
        let (handle_a, handle_b, driver_a, driver_b, offer_rx) = connect_peers()?;

        let (video_sink, video_source) = add_send_track(
            &handle_a,
            &handle_b,
            &offer_rx,
            MediaKind::Video,
            Codec::H264,
        )?;
        let (audio_sink, audio_source) = add_send_track(
            &handle_a,
            &handle_b,
            &offer_rx,
            MediaKind::Audio,
            Codec::Opus,
        )?;
        println!("one WebRTC connection negotiated video (H.264) and audio (Opus) tracks");

        let video_sink_name = video_sink.name();
        let audio_sink_name = audio_sink.name();
        let send = send_pipeline(input, video_sink, audio_sink)?;
        send.run()?;

        // H.264 does not return until received SPS/PPS have supplied dimensions
        // and codec configuration. All packets remain queued in each source.
        let video_info = video_source.wait_stream_info(STREAM_INFO_TIMEOUT)?;
        let audio_info = audio_source.wait_stream_info(STREAM_INFO_TIMEOUT)?;
        if video_info.codec() != Codec::H264 || audio_info.codec() != Codec::Opus {
            return Err(media_pp::Error::Other(format!(
                "unexpected received codecs: video={:?}, audio={:?}",
                video_info.codec(),
                audio_info.codec()
            ))
            .into());
        }
        println!("received video stream: {video_info:?}");
        println!("received audio stream: {audio_info:?}");

        let recv = receive_pipeline(
            &output_path,
            video_source,
            video_info,
            audio_source,
            audio_info,
        )?;
        recv.run()?;
        println!("recording received tracks to {output_path}");

        wait_for_eos("sender", send.bus(), [&*video_sink_name, &*audio_sink_name])?;
        // Closing the sender peer emits DTLS close_notify. The receiver peer then
        // closes both track channels; each source forwards EOS and the shared MP4
        // muxer writes its trailer only after both tracks are done.
        driver_a.stop();
        drain_driver("peer-a", &driver_a)?;
        wait_for_eos(recv.id(), recv.bus(), ["received-video", "received-audio"])?;

        send.stop();
        recv.stop();
        driver_b.stop();
        drain_driver("peer-b", &driver_b)?;
        println!("wrote {output_path}");
        Ok(())
    }

    struct FileInput {
        source: FileDemuxer,
        video_index: usize,
        video_parameters: ffmpeg::codec::Parameters,
        video_time_base: ffmpeg::Rational,
        width: u32,
        height: u32,
        audio_index: usize,
        audio_parameters: ffmpeg::codec::Parameters,
        audio_time_base: ffmpeg::Rational,
    }

    impl FileInput {
        fn open(path: &str) -> media_pp::Result<Self> {
            let (source, streams) = FileDemuxer::open("input", path).map_err(|error| {
                media_pp::Error::Other(format!("cannot read `{path}` as media: {error}"))
            })?;
            let video_index = streams
                .iter()
                .find(|stream| stream.kind == ffmpeg::media::Type::Video)
                .map(|stream| stream.index)
                .ok_or_else(|| media_pp::Error::Other(format!("`{path}` has no video stream")))?;
            let audio_index = streams
                .iter()
                .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
                .map(|stream| stream.index)
                .ok_or_else(|| media_pp::Error::Other(format!("`{path}` has no audio stream")))?;
            let video_parameters = source
                .stream_parameters(video_index)
                .expect("selected video stream still exists");
            let audio_parameters = source
                .stream_parameters(audio_index)
                .expect("selected audio stream still exists");
            // SAFETY: read-only access to parameters owned by this function.
            let (width, height) = unsafe {
                (
                    (*video_parameters.as_ptr()).width,
                    (*video_parameters.as_ptr()).height,
                )
            };
            if width <= 0 || height <= 0 {
                return Err(media_pp::Error::Other(format!(
                    "`{path}` reports invalid video dimensions {width}x{height}"
                )));
            }
            // H.264 4:2:0 needs even dimensions. Cropping one odd edge is more
            // useful than rejecting an otherwise valid source file.
            let width = (width as u32) & !1;
            let height = (height as u32) & !1;
            Ok(Self {
                video_time_base: source
                    .stream_time_base(video_index)
                    .expect("selected video stream still exists"),
                audio_time_base: source
                    .stream_time_base(audio_index)
                    .expect("selected audio stream still exists"),
                source,
                video_index,
                video_parameters,
                width,
                height,
                audio_index,
                audio_parameters,
            })
        }
    }

    fn send_pipeline(
        input: FileInput,
        video_sink: WebRtcTrackSink,
        audio_sink: WebRtcTrackSink,
    ) -> media_pp::Result<Arc<Pipeline>> {
        let clock = Arc::new(Clock::new());
        let video_decoder = SwDecoder::new("decode-video", input.video_parameters)?;
        let video_pacer = Pacer::new("pace-video", input.video_time_base, clock.clone())?;
        let scaler = SwScaler::new(
            "to-yuv420p",
            ffmpeg::format::Pixel::YUV420P,
            input.width,
            input.height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let video_encoder = SwEncoder::new(
            "encode-h264",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width: input.width,
                height: input.height,
                time_base: input.video_time_base,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 2_000_000,
                gop_size: 60,
            },
        )?;
        let audio_decoder = SwDecoder::new("decode-audio", input.audio_parameters)?;
        let audio_pacer = Pacer::new("pace-audio", input.audio_time_base, clock)?;
        let audio_encoder = SwAudioEncoder::new(
            "encode-opus",
            SwAudioEncoderOptions {
                codec: AudioCodec::Opus,
                sample_rate: AUDIO_RATE,
                channels: AUDIO_CHANNELS,
                time_base: ffmpeg::Rational::new(1, AUDIO_RATE as i32),
                bit_rate: 96_000,
            },
        )?;

        Pipeline::new("webrtc-send", input.source, move |source, ctx| {
            let video = ctx
                .branch()
                .pipe(video_decoder)
                .queue("video-frames", 8)
                .pipe(video_pacer)
                .pipe(scaler)
                .pipe(video_encoder)
                .to(Box::new(video_sink))?;
            ctx.attach(source, input.video_index, video)?;

            let audio = ctx
                .branch()
                .pipe(audio_decoder)
                .queue("audio-frames", 16)
                .pipe(audio_pacer)
                .pipe(audio_encoder)
                .to(Box::new(audio_sink))?;
            ctx.attach(source, input.audio_index, audio)?;
            Ok(())
        })
    }

    fn receive_pipeline(
        output: &str,
        video_source: WebRtcTrackSource,
        video_info: media_pp::elements::WebRtcStreamInfo,
        audio_source: WebRtcTrackSource,
        audio_info: media_pp::elements::WebRtcStreamInfo,
    ) -> media_pp::Result<Arc<Pipeline>> {
        let mut muxer = Mp4Muxer::create(output)?;
        muxer.add_stream(
            "received-video",
            video_info.codec_parameters()?,
            video_info.time_base()?,
        )?;
        muxer.add_stream(
            "received-audio",
            audio_info.codec_parameters()?,
            audio_info.time_base()?,
        )?;
        let mut sinks = muxer.open()?;
        let audio_sink = sinks.pop().expect("two streams were registered");
        let video_sink = sinks.pop().expect("two streams were registered");

        Ok(PipelineBuilder::new("webrtc-receive-record")
            .add_source(video_source, |source, ctx| {
                let branch = ctx.branch().to(video_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?
            .add_source(audio_source, |source, ctx| {
                let branch = ctx.branch().to(audio_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })?
            .build())
    }

    // Naming this tuple would only move the same five types somewhere else:
    // they are the whole result of bringing two peers up, and nothing else in
    // this example refers to the shape. `webrtc_video_call` does the same.
    #[allow(clippy::type_complexity)]
    fn connect_peers() -> std::result::Result<
        (
            WebRtcHandle,
            WebRtcHandle,
            Arc<DriverRunner>,
            Arc<DriverRunner>,
            Receiver<SdpOffer>,
        ),
        Box<dyn std::error::Error>,
    > {
        let socket_a = UdpSocket::bind("127.0.0.1:0")?;
        let socket_b = UdpSocket::bind("127.0.0.1:0")?;
        let addr_a = socket_a.local_addr()?;
        let addr_b = socket_b.local_addr()?;

        let mut rtc_a = Rtc::builder().build(Instant::now());
        rtc_a
            .add_local_candidate(Candidate::host(addr_a, "udp").expect("valid UDP candidate"))
            .expect("add candidate a");
        let mut rtc_b = Rtc::builder().build(Instant::now());
        rtc_b
            .add_local_candidate(Candidate::host(addr_b, "udp").expect("valid UDP candidate"))
            .expect("add candidate b");

        let mut changes = rtc_a.sdp_api();
        changes.add_channel("bootstrap".into());
        let (offer, pending) = changes.apply().expect("adding a channel creates an offer");
        let answer = rtc_b.sdp_api().accept_offer(offer)?;
        rtc_a.sdp_api().accept_answer(pending, answer)?;

        let (offer_tx, offer_rx) = std::sync::mpsc::channel();
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
        Ok((handle_a, handle_b, driver_a, driver_b, offer_rx))
    }

    fn add_send_track(
        sender: &WebRtcHandle,
        receiver: &WebRtcHandle,
        offer_rx: &Receiver<SdpOffer>,
        kind: MediaKind,
        codec: Codec,
    ) -> media_pp::Result<(WebRtcTrackSink, WebRtcTrackSource)> {
        let id = sender.add_track(kind, Direction::SendOnly, codec)?;
        let attached_sender = sender.next_track()?;
        if attached_sender.id != id || attached_sender.kind != kind {
            return Err(media_pp::Error::Other(format!(
                "sender attached unexpected track: id={:?}, kind={:?}",
                attached_sender.id, attached_sender.kind
            )));
        }
        let TrackEndpoints::Send(sink) = attached_sender.endpoints else {
            return Err(media_pp::Error::Other(
                "sender track is not SendOnly".into(),
            ));
        };

        let offer = offer_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| {
                media_pp::Error::Other(format!("timed out waiting for {kind:?} SDP offer: {error}"))
            })?;
        let answer = receiver.accept_remote_offer(offer)?;
        sender.set_answer(answer);
        let attached_receiver = receiver.next_track()?;
        if attached_receiver.kind != kind {
            return Err(media_pp::Error::Other(format!(
                "receiver attached {:?} while waiting for {kind:?}",
                attached_receiver.kind
            )));
        }
        let TrackEndpoints::Recv(source) = attached_receiver.endpoints else {
            return Err(media_pp::Error::Other(
                "receiver track is not RecvOnly".into(),
            ));
        };
        thread::sleep(Duration::from_millis(100));
        Ok((sink, source))
    }

    fn wait_for_eos<'a>(
        owner: &str,
        bus: &BusReceiver,
        names: impl IntoIterator<Item = &'a str>,
    ) -> media_pp::Result<()> {
        let mut pending: HashSet<String> = names.into_iter().map(str::to_owned).collect();
        for event in bus.iter() {
            match &event {
                BusEvent::Eos { name, .. } => {
                    if pending.remove(name.as_ref()) {
                        println!("[{owner}/{name}] eos");
                    }
                }
                BusEvent::Error { name, error, .. } => {
                    return Err(media_pp::Error::Other(format!(
                        "[{owner}/{name}] pipeline error: {error}"
                    )));
                }
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{owner}/{name}] dropped a buffer (queue full)");
                }
                _ => {}
            }
            if pending.is_empty() {
                return Ok(());
            }
        }
        Err(media_pp::Error::Other(format!(
            "{owner} ended before EOS from {pending:?}"
        )))
    }

    fn drain_driver(name: &str, driver: &DriverRunner) -> media_pp::Result<()> {
        for event in driver.bus().iter() {
            if let BusEvent::Error { error, .. } = event {
                return Err(media_pp::Error::Other(format!(
                    "{name} driver error: {error}"
                )));
            }
        }
        Ok(())
    }
}
