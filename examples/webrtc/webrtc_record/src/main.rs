//! Sends one file's best video and audio streams as H.264/Opus over two
//! WebRTC tracks of one `WebRtcMuxer`, then records both received tracks —
//! one `WebRtcSource` with an output each — into one MP4 without decoding or
//! re-encoding on the receiving side.
//!
//! Sender:
//!
//! ```text
//! FileDemuxer(video) -> SwDecoder -> Queue -> Pacer -> SwScaler
//!                    -> SwEncoder(H.264) -> WebRtcMuxer
//! FileDemuxer(audio) -> SwDecoder -> Queue -> Pacer
//!                    -> SwAudioEncoder(Opus) -> WebRtcMuxer
//! ```
//!
//! Receiver:
//!
//! ```text
//! WebRtcSource(H.264) -\
//!                       -> FileMuxer
//! WebRtcSource(Opus)  --/
//! ```
//!
//!     cargo run -p webrtc_record -- input.mp4 output.mp4

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use std::{
        net::UdpSocket,
        sync::{Arc, mpsc::Receiver},
        thread,
        time::{Duration, Instant},
    };

    use media_pp::ffmpeg;
    use media_pp::{
        bus::{BusEvent, BusReceiver},
        driver::DriverRunner,
        elements::{
            AudioCodec, FileDemuxer, FileMuxer, Mp4Layout, Pacer, StreamInfo, SwAudioEncoder,
            SwAudioEncoderOptions, SwDecoder, SwEncoder, SwEncoderOptions, SwScaler, VideoCodec,
            WebRtcHandle, WebRtcPeer, WebRtcSource,
        },
        pipeline::Pipeline,
    };
    use str0m::{Candidate, Rtc, change::SdpOffer};

    const AUDIO_RATE: u32 = 48_000;
    const AUDIO_CHANNELS: u16 = 2;

    pub(super) fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
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

        let send = send_pipeline(input, &handle_a, &handle_b, &offer_rx)?;
        println!("one WebRTC connection negotiated video (H.264) and audio (Opus) tracks");
        send.run()?;

        // The tracks the offer added come out as one source once each has
        // said what it carries — H.264 not until received SPS/PPS have
        // supplied dimensions and codec configuration. All packets remain
        // queued meanwhile.
        let (source, streams) = handle_b.next_source("received")?;
        let output_of = |id| {
            streams
                .iter()
                .find(|stream| stream.parameters().id() == id)
                .cloned()
                .ok_or_else(|| {
                    media_pp::Error::Other(format!(
                        "no {id:?} track was received: {:?}",
                        streams
                            .iter()
                            .map(|stream| stream.parameters().id())
                            .collect::<Vec<_>>()
                    ))
                })
        };
        let video = output_of(ffmpeg::codec::Id::H264)?;
        let audio = output_of(ffmpeg::codec::Id::OPUS)?;
        println!("received video stream: {:?}", video.size());
        println!("received audio stream: output {}", audio.index());

        let recv = receive_pipeline(&output_path, source, video, audio)?;
        recv.run()?;
        println!("recording received tracks to {output_path}");

        wait_for_finished("sender", send.bus())?;
        // Closing the sender peer emits DTLS close_notify. The receiver peer then
        // closes both track channels; each source forwards EOS and the shared MP4
        // muxer writes its trailer only after both tracks are done.
        driver_a.stop();
        drain_driver("peer-a", &driver_a)?;
        wait_for_finished(recv.id(), recv.bus())?;

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
        width: u32,
        height: u32,
        audio_index: usize,
        audio_parameters: ffmpeg::codec::Parameters,
    }

    impl FileInput {
        fn open(path: &str) -> media_pp::Result<Self> {
            let (source, _) = FileDemuxer::open("input", path).map_err(|error| {
                media_pp::Error::Other(format!("cannot read `{path}` as media: {error}"))
            })?;
            let video = source.best(ffmpeg::media::Type::Video)?;
            let audio = source.best(ffmpeg::media::Type::Audio)?;
            let (video_index, audio_index) = (video.index(), audio.index());
            let video_parameters = video.parameters().clone();
            let audio_parameters = audio.parameters().clone();
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

    /// The sender: the file decoded, paced and encoded again, onto a muxer
    /// of `sender`'s connection, whose offer it carries to `receiver` and
    /// whose answer back.
    fn send_pipeline(
        input: FileInput,
        sender: &WebRtcHandle,
        receiver: &WebRtcHandle,
        offer_rx: &Receiver<SdpOffer>,
    ) -> media_pp::Result<Arc<Pipeline>> {
        let video_decoder = SwDecoder::new("decode-video", input.video_parameters)?;
        let video_pacer = Pacer::new("pace-video");
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
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 2_000_000,
                gop_size: 60,
                max_b_frames: None,
            },
        )?;
        let audio_decoder = SwDecoder::new("decode-audio", input.audio_parameters)?;
        let audio_pacer = Pacer::new("pace-audio");
        let audio_encoder = SwAudioEncoder::new(
            "encode-opus",
            SwAudioEncoderOptions {
                codec: AudioCodec::Opus,
                sample_rate: AUDIO_RATE,
                channels: AUDIO_CHANNELS,
                bit_rate: 96_000,
            },
        )?;

        // Each encoder's stream settles its track's payload type and codec
        // headers together. H.264's SPS/PPS are in `parameters()` rather than
        // in the bitstream, and RTP has no container to carry them, so they go
        // out in front of every keyframe — which is what lets the receiver
        // build its decoder parameters at all. Opus has nothing of the kind.
        let mut muxer = sender.muxer("webrtc-send");
        let video_track = muxer.add_stream(
            "video",
            StreamInfo::new(video_encoder.parameters(), video_encoder.time_base()),
        )?;
        let audio_track = muxer.add_stream(
            "audio",
            StreamInfo::new(audio_encoder.parameters(), audio_encoder.time_base()),
        )?;
        let mut sinks = muxer.open()?;
        let offer = offer_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| {
                media_pp::Error::Other(format!("timed out waiting for the SDP offer: {error}"))
            })?;
        let answer = receiver.accept_remote_offer(offer)?;
        sender.set_answer(answer);
        let video_sink = sinks.take(video_track)?;
        let audio_sink = sinks.take(audio_track)?;
        // What is sent before the answer applies is let go of.
        thread::sleep(Duration::from_millis(100));

        let (pipeline, ()) = Pipeline::new("webrtc-send", input.source, move |source, ctx| {
            let video = ctx
                .branch()
                .pipe(video_decoder)
                .queue("video-frames", 8)
                .pipe(video_pacer)
                .pipe(scaler)
                .pipe(video_encoder)
                .to(video_sink)?;
            ctx.attach(source, input.video_index, video)?;

            let audio = ctx
                .branch()
                .pipe(audio_decoder)
                .queue("audio-frames", 16)
                .pipe(audio_pacer)
                .pipe(audio_encoder)
                .to(audio_sink)?;
            ctx.attach(source, input.audio_index, audio)?;
            Ok(())
        })?;
        Ok(pipeline)
    }

    fn receive_pipeline(
        output: &str,
        source: WebRtcSource,
        video: StreamInfo,
        audio: StreamInfo,
    ) -> media_pp::Result<Arc<Pipeline>> {
        let mut muxer = FileMuxer::create(output, Mp4Layout::Standard)?;
        let video_track = muxer.add_stream("received-video", video.clone())?;
        let audio_track = muxer.add_stream("received-audio", audio.clone())?;
        let mut sinks = muxer.open()?;
        let video_sink = sinks.take(video_track)?;
        let audio_sink = sinks.take(audio_track)?;

        let (pipeline, ()) = Pipeline::new("webrtc-receive-record", source, move |source, ctx| {
            ctx.attach(source, video.index(), ctx.branch().to(video_sink)?)?;
            ctx.attach(source, audio.index(), ctx.branch().to(audio_sink)?)?;
            Ok(())
        })?;
        Ok(pipeline)
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
            .add_local_candidate(Candidate::host(addr_a, "udp")?)
            .expect("add candidate a");
        let mut rtc_b = Rtc::builder().build(Instant::now());
        rtc_b
            .add_local_candidate(Candidate::host(addr_b, "udp")?)
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
            |_track| {},
        );
        let (peer_b, handle_b) =
            WebRtcPeer::new("peer-b", rtc_b, socket_b, |_offer| {}, |_track| {});
        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run()?;
        driver_b.run()?;
        thread::sleep(Duration::from_millis(200));
        Ok((handle_a, handle_b, driver_a, driver_b, offer_rx))
    }

    /// Waits until every terminal of `owner`'s pipeline has ended.
    fn wait_for_finished(owner: &str, bus: &BusReceiver) -> media_pp::Result<()> {
        for event in bus.iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{owner}/{name}] eos"),
                BusEvent::Finished => return Ok(()),
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
        }
        Err(media_pp::Error::Other(format!(
            "{owner} ended before every branch did"
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
