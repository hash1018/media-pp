//! A two-way video call between two `WebRtcPeer`s in one process, each
//! presenting what the *other* sent into its own window.
//!
//! One `Direction::SendRecv` track carries both directions on a single
//! connection (see `webrtc_loopback` for the minimal version of that), so
//! `next_track` hands each side a `TrackEndpoints::SendRecv` — a sink to
//! encode into and a source to decode from.
//!
//! The two callers deliberately differ in where their video comes from:
//! peer-a generates it, peer-b transcodes a real file at playback speed.
//!
//!     cargo run -p webrtc_video_call -- path/to/video.mp4

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} example supports Windows and Linux only",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() {
    linux_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::sync::Arc;

    use media_pp::{
        elements::{D3d12Upload, SwDecoder, SwScaler, WebRtcStreamInfo, WebRtcTrackSource},
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{D3d12GpuContext, WindowTarget};
    use str0m::format::Codec;
    use winit::raw_window_handle::RawWindowHandle;

    use super::common::{HEIGHT, WIDTH};

    pub(super) fn run() {
        super::common::run();
    }

    pub(super) struct RenderContext {
        gpu: D3d12GpuContext,
    }

    impl RenderContext {
        pub(super) fn new(_target: &WindowTarget) -> media_pp::Result<Self> {
            let gpu = D3d12GpuContext::new().map_err(|error| {
                media_pp::Error::Other(format!("failed to create the D3D12 context: {error:?}"))
            })?;
            Ok(Self { gpu })
        }
    }

    /// `WebRtcTrackSource -> Queue -> SwDecoder -> SwScaler(NV12) ->
    /// D3d12Upload -> D3d12Renderer`.
    pub(super) fn receive_pipeline(
        name: &str,
        source: WebRtcTrackSource,
        stream_info: WebRtcStreamInfo,
        render: &RenderContext,
        target: WindowTarget,
    ) -> media_pp::Result<Arc<Pipeline>> {
        validate_h264(name, &stream_info)?;
        let decoder = SwDecoder::new(format!("{name}-decode"), stream_info.codec_parameters()?)?;
        // `D3d12Renderer` draws from a device resource only, so decoded frames
        // have to be converted to the layout it samples and uploaded first —
        // the same pair the Linux branch below uses for CUDA.
        let scaler = SwScaler::new(
            format!("{name}-nv12"),
            ffmpeg::format::Pixel::NV12,
            WIDTH,
            HEIGHT,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let upload = D3d12Upload::new(format!("{name}-upload"), render.gpu.device(), WIDTH, HEIGHT)
            .map_err(|error| media_pp::Error::Other(error.to_string()))?;
        let RawWindowHandle::Win32(handle) = target.window else {
            panic!("webrtc_video_call Windows branch received a non-Win32 window");
        };
        let renderer = render_common::d3d12_window_renderer(
            format!("{name}-render"),
            &render.gpu,
            handle.hwnd.get(),
            WIDTH,
            HEIGHT,
        )
        .map_err(|error| media_pp::Error::Other(format!("failed to open a renderer: {error:?}")))?;

        Pipeline::new(name, source, move |source, ctx| {
            let branch = ctx
                .branch()
                .queue("packets", 16)
                .pipe(decoder)
                .pipe(scaler)
                .pipe(upload)
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
    }

    fn validate_h264(name: &str, stream_info: &WebRtcStreamInfo) -> media_pp::Result<()> {
        if stream_info.codec() != Codec::H264 {
            return Err(media_pp::Error::Other(format!(
                "{name} cannot decode the received {:?} stream; this example expects H.264",
                stream_info.codec()
            )));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux_example {
    use std::sync::Arc;

    use media_pp::{
        elements::{
            CudaDevice, CudaFrameFormat, CudaUpload, SwDecoder, SwScaler, WebRtcStreamInfo,
            WebRtcTrackSource,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{VulkanGpuContext, WindowTarget};
    use str0m::format::Codec;

    use super::common::{HEIGHT, WIDTH};

    pub(super) fn run() {
        super::common::run();
    }

    pub(super) struct RenderContext {
        cuda: CudaDevice,
        gpu: VulkanGpuContext,
    }

    impl RenderContext {
        pub(super) fn new(target: &WindowTarget) -> media_pp::Result<Self> {
            let cuda =
                CudaDevice::new().map_err(|error| media_pp::Error::Other(error.to_string()))?;
            let gpu = VulkanGpuContext::new(target.display).map_err(media_pp::Error::Other)?;
            Ok(Self { cuda, gpu })
        }
    }

    /// `WebRtcTrackSource -> Queue -> SwDecoder -> SwScaler(NV12) ->
    /// CudaUpload -> CudaRenderer(Vulkan)`.
    pub(super) fn receive_pipeline(
        name: &str,
        source: WebRtcTrackSource,
        stream_info: WebRtcStreamInfo,
        render: &RenderContext,
        target: WindowTarget,
    ) -> media_pp::Result<Arc<Pipeline>> {
        validate_h264(name, &stream_info)?;
        let decoder = SwDecoder::new(format!("{name}-decode"), stream_info.codec_parameters()?)?;
        let scaler = SwScaler::new(
            format!("{name}-nv12"),
            ffmpeg::format::Pixel::NV12,
            WIDTH,
            HEIGHT,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let upload = CudaUpload::new(
            format!("{name}-upload"),
            &render.cuda,
            CudaFrameFormat::Nv12,
            WIDTH,
            HEIGHT,
        )
        .map_err(|error| media_pp::Error::Other(error.to_string()))?;
        let renderer = render_common::cuda_window_renderer(
            format!("{name}-render"),
            &render.gpu,
            &render.cuda,
            target.display,
            target.window,
            target.width,
            target.height,
        )
        .map_err(media_pp::Error::Other)?;

        Pipeline::new(name, source, move |source, ctx| {
            let branch = ctx
                .branch()
                .queue("packets", 16)
                .pipe(decoder)
                .pipe(scaler)
                .pipe(upload)
                .to(Box::new(renderer))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
    }

    fn validate_h264(name: &str, stream_info: &WebRtcStreamInfo) -> media_pp::Result<()> {
        if stream_info.codec() != Codec::H264 {
            return Err(media_pp::Error::Other(format!(
                "{name} cannot decode the received {:?} stream; this example expects H.264",
                stream_info.codec()
            )));
        }
        Ok(())
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common {
    use std::{
        net::UdpSocket,
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };

    #[cfg(target_os = "linux")]
    use super::linux_example as platform;
    #[cfg(target_os = "windows")]
    use super::windows_example as platform;
    use media_pp::{
        bus::BusEvent,
        clock::Clock,
        element::Element,
        elements::{
            AttachedTrack, FileDemuxer, Pacer, SwDecoder, SwEncoder, SwEncoderOptions, SwScaler,
            TestVideoOptions, TestVideoSource, TrackEndpoints, VideoCodec, WebRtcPeer,
            WebRtcTrackSink, WebRtcTrackSource,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::{Shutdown, WindowTarget};
    use str0m::{
        Candidate, Rtc,
        change::SdpOffer,
        format::Codec,
        media::{Direction, MediaKind},
    };

    /// What both directions of the call are encoded at. Fixed rather than
    /// derived from either source: the file side is scaled to it, and both
    /// window renderers are wired up once at this size.
    pub(super) const WIDTH: u32 = 640;
    pub(super) const HEIGHT: u32 = 480;
    const FPS: i32 = 30;
    const STREAM_INFO_TIMEOUT: Duration = Duration::from_secs(2);

    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: webrtc_video_call <video.mp4>");
            std::process::exit(1);
        };

        render_common::run_windows(
            &[
                "media-pp webrtc_video_call — peer-a (showing peer-b's file)",
                "media-pp webrtc_video_call — peer-b (showing peer-a's test pattern)",
            ],
            WIDTH,
            HEIGHT,
            move |targets, shutdown| {
                let [target_a, target_b] = <[WindowTarget; 2]>::try_from(targets)
                    .unwrap_or_else(|_| panic!("run_windows opened the two windows asked for"));
                play(&path, target_a, target_b, shutdown)
            },
        );
    }

    fn play(
        path: &str,
        target_a: WindowTarget,
        target_b: WindowTarget,
        shutdown: Arc<Shutdown>,
    ) -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        // Before anything else: bringing up two peers, negotiating a track,
        // and opening a GPU context are all slow and all pointless if the
        // file cannot be read. Failing here costs nothing and says why.
        let file = FileSource::open(path)?;

        let (handle_a, handle_b, driver_a, driver_b, offer_rx) = connect_peers();

        // One SendRecv track, so each side gets both halves back and the
        // call needs no second `add_track` for the return direction.
        handle_a
            .add_track(MediaKind::Video, Direction::SendRecv, Codec::H264)
            .expect("running peer should accept AddTrack");
        let (sink_a, source_a) = send_recv(
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
        let (mut sink_b, source_b) = send_recv(
            handle_b
                .next_track()
                .expect("peer-b's remote track should attach"),
        );
        println!(
            "peer-b negotiated inbound codecs: {:?}",
            source_b.negotiated_codecs()
        );
        println!(
            "peer-b negotiated outbound codecs: {:?}",
            sink_b.negotiated_codecs()
        );
        // The negotiated list is known when the endpoints attach, but only
        // this pipeline knows which encoder feeds the outbound half.
        sink_b
            .set_codec(Codec::H264)
            .expect("H.264 should be negotiated for peer-b's outbound half");
        thread::sleep(Duration::from_millis(100)); // let the answer apply
        println!("call established — one SendRecv track carrying both directions");

        // One backend context for both windows. On Linux this also includes
        // the CUDA device into which decoded system-memory frames are
        // uploaded before the Vulkan presentation bridge consumes them.
        let render = platform::RenderContext::new(&target_a)?;

        let send_a = generated_send_pipeline(sink_a)?;
        // Kept before the sink moves into the pipeline: it is the name the
        // bus reports EOS under, and the one event below actually waits for.
        let track_sink_name = sink_b.name();
        let send_b = file_send_pipeline(file, sink_b)?;

        // Publish before starting, so a window close can reach both senders
        // while we wait for the first actual RTP payload on each source.
        let send_pipelines = [send_a.clone(), send_b.clone()];
        if shutdown.publish(&send_pipelines) {
            return Ok(());
        }
        for pipeline in &send_pipelines {
            pipeline.run()?;
        }

        // SDP only says which codecs may arrive. The first payload confirms
        // what each remote sender actually chose; its packet stays queued in
        // the source while the matching decoder pipeline is constructed.
        let info_a = source_a.wait_stream_info(STREAM_INFO_TIMEOUT)?;
        let info_b = source_b.wait_stream_info(STREAM_INFO_TIMEOUT)?;
        println!("peer-a receiving actual codec: {info_a:?}");
        println!("peer-b receiving actual codec: {info_b:?}");
        let recv_a =
            platform::receive_pipeline("peer-a-recv", source_a, info_a, &render, target_a)?;
        let recv_b =
            platform::receive_pipeline("peer-b-recv", source_b, info_b, &render, target_b)?;

        let pipelines = [
            send_a.clone(),
            send_b.clone(),
            recv_a.clone(),
            recv_b.clone(),
        ];
        if shutdown.publish(&pipelines) {
            return Ok(());
        }
        for pipeline in [&recv_a, &recv_b] {
            pipeline.run()?;
        }
        println!("both windows are live — close either one to end the call");

        // The generated side runs until a window closes; the file side ends
        // on its own. Watching the file pipeline's bus is what keeps the call
        // up for exactly as long as there is still something to send.
        //
        // Specifically the `WebRtcTrackSink`'s own EOS, not the first one to
        // appear: every `Queue` along the way reports its own as EOS passes
        // through it, and the earliest of those means only that a queue has
        // drained — the encoder still has delayed frames to flush behind it.
        // The terminal sink's EOS is the one that means everything this side
        // had to send has actually been handed to the peer.
        for event in send_b.bus().iter() {
            match &event {
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
            let sent_everything =
                matches!(&event, BusEvent::Eos { name, .. } if *name == track_sink_name);
            if sent_everything || matches!(event, BusEvent::Error { .. }) {
                break;
            }
        }

        for pipeline in &pipelines {
            pipeline.stop();
        }
        driver_a.stop();
        driver_b.stop();
        Ok(())
    }

    /// Both halves of the one `Direction::SendRecv` track this example opens.
    fn send_recv(track: AttachedTrack) -> (WebRtcTrackSink, WebRtcTrackSource) {
        let TrackEndpoints::SendRecv(sink, source) = track.endpoints else {
            panic!("a SendRecv track should carry both halves");
        };
        (sink, source)
    }

    fn encoder_options(time_base: ffmpeg::Rational) -> SwEncoderOptions {
        SwEncoderOptions {
            // Cisco's BSD-licensed H.264 encoder, so this needs no
            // `--enable-gpl` ffmpeg build — and H.264 is what the track
            // above negotiates.
            codec: VideoCodec::OpenH264,
            width: WIDTH,
            height: HEIGHT,
            time_base,
            frame_rate: ffmpeg::Rational::new(FPS, 1),
            bit_rate: 1_500_000,
            gop_size: 30,
        }
    }

    /// peer-a's outbound half: `TestVideoSource -> Queue -> SwEncoder ->
    /// WebRtcTrackSink`. The source paces itself, so nothing else has to.
    fn generated_send_pipeline(sink: WebRtcTrackSink) -> media_pp::Result<Arc<Pipeline>> {
        let options = TestVideoOptions {
            width: WIDTH,
            height: HEIGHT,
            framerate: ffmpeg::Rational::new(FPS, 1),
        };
        let source = TestVideoSource::new("test-video", options);
        let time_base = source.time_base();
        let encoder = SwEncoder::new("encode-a", encoder_options(time_base))?;

        let pipeline = Pipeline::new("peer-a-send", source, move |source, ctx| {
            let branch = ctx
                .branch()
                .queue("to-encode", 8)
                .pipe(encoder)
                .to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;
        Ok(pipeline)
    }

    /// An opened video file, and everything about it the call needs — kept
    /// separate from building the pipeline so the one step that depends on
    /// user input can happen before any of the setup that does not.
    struct FileSource {
        demuxer: FileDemuxer,
        index: usize,
        params: ffmpeg::codec::Parameters,
        time_base: ffmpeg::Rational,
    }

    impl FileSource {
        /// Both failure modes name the path: ffmpeg reports an unreadable or
        /// non-media file as a bare "Invalid data found when processing
        /// input", which says nothing about which argument was wrong.
        fn open(path: &str) -> media_pp::Result<Self> {
            let (demuxer, streams) = FileDemuxer::open("demux", path).map_err(|error| {
                media_pp::Error::Other(format!("cannot read `{path}` as a media file: {error}"))
            })?;
            let video = streams
                .iter()
                .find(|s| s.kind == ffmpeg::media::Type::Video)
                .ok_or_else(|| {
                    media_pp::Error::Other(format!("`{path}` has no video stream to send"))
                })?;
            let index = video.index;
            let params = demuxer
                .stream_parameters(index)
                .expect("the stream just found still exists");
            let time_base = demuxer
                .stream_time_base(index)
                .expect("the stream just found still exists");
            Ok(Self {
                demuxer,
                index,
                params,
                time_base,
            })
        }
    }

    /// peer-b's outbound half: `FileDemuxer -> SwDecoder -> Queue -> Pacer ->
    /// SwScaler -> Queue -> SwEncoder -> WebRtcTrackSink`.
    ///
    /// The transcode is what makes the file usable as a call source at all:
    /// its frames are whatever size the file holds, and its packets arrive as
    /// fast as the disk can serve them. `SwScaler` fixes the first, `Pacer`
    /// the second — without it the whole file would be encoded and sent in a
    /// couple of seconds.
    fn file_send_pipeline(
        file: FileSource,
        sink: WebRtcTrackSink,
    ) -> media_pp::Result<Arc<Pipeline>> {
        let FileSource {
            demuxer: source,
            index,
            params,
            time_base,
        } = file;

        let decoder = SwDecoder::new("decode-file", params)?;
        let pacer = Pacer::new("pace-file", time_base, Arc::new(Clock::new()))?;
        let scaler = SwScaler::new(
            "scale-file",
            ffmpeg::format::Pixel::YUV420P,
            WIDTH,
            HEIGHT,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );
        let encoder = SwEncoder::new("encode-b", encoder_options(time_base))?;

        let pipeline = Pipeline::new("peer-b-send", source, move |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", 8) // the pacer sleeps; let demux/decode run ahead
                .pipe(pacer)
                .pipe(scaler)
                .queue("to-encode", 8)
                .pipe(encoder)
                .to(Box::new(sink))?;
            ctx.attach(source, index, branch)?;
            Ok(())
        })?;
        Ok(pipeline)
    }

    /// Two `Rtc`s brought up over loopback UDP with a throwaway data channel,
    /// then handed to `WebRtcPeer`s — the same signaling stand-in
    /// `webrtc_loopback` uses, since this example is about the media, not
    /// about transporting SDP.
    #[allow(clippy::type_complexity)]
    fn connect_peers() -> (
        media_pp::elements::WebRtcHandle,
        media_pp::elements::WebRtcHandle,
        Arc<media_pp::driver::DriverRunner>,
        Arc<media_pp::driver::DriverRunner>,
        std::sync::mpsc::Receiver<SdpOffer>,
    ) {
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

        let driver_a = media_pp::driver::DriverRunner::new(peer_a);
        let driver_b = media_pp::driver::DriverRunner::new(peer_b);
        driver_a.run().expect("peer-a should start");
        driver_b.run().expect("peer-b should start");
        thread::sleep(Duration::from_millis(200));
        println!("ICE/DTLS-SRTP established over loopback UDP");

        (handle_a, handle_b, driver_a, driver_b, offer_rx)
    }
}
