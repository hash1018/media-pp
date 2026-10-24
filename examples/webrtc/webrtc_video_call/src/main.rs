//! A two-way video call between two `WebRtcPeer`s in one process, each
//! presenting what the *other* sent into its own window.
//!
//! Each side sends its video through a `WebRtcMuxer` of its own on the one
//! connection, and shows the other's, which comes out of its handle as a
//! `WebRtcSource` (see `webrtc_loopback` for the minimal version of that).
//!
//! The two callers deliberately differ in where their video comes from:
//! peer-a generates it, peer-b transcodes a real file at playback speed.
//!
//!     cargo run -p webrtc_video_call -- path/to/video.mp4

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn main() {
    eprintln!(
        "{} example supports Windows, Linux and macOS only",
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

/// A window is the main thread's on macOS: the main thread runs AppKit's
/// event loop, and the call runs beside it.
#[cfg(target_os = "macos")]
fn main() {
    media_pp::elements::run_with_windows(macos_example::run)
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::sync::Arc;

    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            D3d12Gpu, D3d12WindowRenderer, StreamInfo, SwDecoder, WebRtcSource, WindowOptions,
        },
        pipeline::Pipeline,
    };
    use render_common::Shutdown;

    use super::common::{HEIGHT, WIDTH};

    /// Where one side of the call is shown: the renderer that opened the
    /// window, already drawing into it.
    pub(super) type Target = D3d12WindowRenderer;

    pub(super) fn run() {
        super::common::run();
    }

    /// Opens both windows, and the `Shutdown` that closing either sets off.
    ///
    /// On one D3D12 device: what arrives is H.264 decoded to YUV420P in
    /// system memory, which the renderer draws as it comes and uploads
    /// itself.
    pub(super) fn open_windows(
        titles: [&str; 2],
    ) -> media_pp::Result<(Target, Target, Arc<Shutdown>)> {
        let gpu = D3d12Gpu::new()?;
        let open = |name: &str, title: &str| {
            D3d12WindowRenderer::open(
                name,
                &gpu,
                WindowOptions {
                    title: title.into(),
                    width: WIDTH,
                    height: HEIGHT,
                },
            )
        };
        let (screen_a, window_a) = open("peer-a-render", titles[0])?;
        let (screen_b, window_b) = open("peer-b-render", titles[1])?;
        let shutdown = render_common::stop_on_close([window_a, window_b]);
        Ok((screen_a, screen_b, shutdown))
    }

    /// Nothing to share between the two sides: the renderers already have
    /// their device.
    pub(super) struct RenderContext;

    impl RenderContext {
        pub(super) fn new(_target: &Target) -> media_pp::Result<Self> {
            Ok(Self)
        }
    }

    /// `WebRtcSource -> Queue -> SwDecoder -> D3d12WindowRenderer`.
    pub(super) fn receive_pipeline(
        name: &str,
        source: WebRtcSource,
        stream: StreamInfo,
        _render: &RenderContext,
        screen: Target,
    ) -> media_pp::Result<Arc<Pipeline>> {
        validate_h264(name, &stream)?;
        let decoder = SwDecoder::new(format!("{name}-decode"), stream.parameters().clone())?;

        let (pipeline, ()) = Pipeline::new(name, source, move |source, ctx| {
            let branch = ctx.branch().queue("packets", 16).pipe(decoder).to(screen)?;
            ctx.attach(source, stream.index(), branch)?;
            Ok(())
        })?;
        Ok(pipeline)
    }

    fn validate_h264(name: &str, stream: &StreamInfo) -> media_pp::Result<()> {
        let codec = stream.parameters().id();
        if codec != ffmpeg::codec::Id::H264 {
            return Err(media_pp::Error::Other(format!(
                "{name} cannot decode the received {codec:?} stream; this example expects H.264"
            )));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux_example {
    use std::sync::Arc;

    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            StreamInfo, SwDecoder, VulkanGpu, VulkanWindowRenderer, WebRtcSource, WindowOptions,
        },
        pipeline::Pipeline,
    };
    use render_common::Shutdown;

    use super::common::{HEIGHT, WIDTH};

    /// Where one side of the call is shown: the renderer that opened the
    /// window, already drawing into it.
    pub(super) type Target = VulkanWindowRenderer;

    pub(super) fn run() {
        super::common::run();
    }

    /// Opens both windows, and the `Shutdown` that closing either sets off.
    ///
    /// On one Vulkan device and no CUDA at all: what arrives is H.264
    /// decoded to YUV420P in system memory, which the renderer draws as it
    /// comes and uploads itself.
    pub(super) fn open_windows(
        titles: [&str; 2],
    ) -> media_pp::Result<(Target, Target, Arc<Shutdown>)> {
        let gpu = VulkanGpu::new()?;
        let open = |name: &str, title: &str| {
            VulkanWindowRenderer::open(
                name,
                &gpu,
                WindowOptions {
                    title: title.into(),
                    width: WIDTH,
                    height: HEIGHT,
                },
            )
        };
        let (screen_a, window_a) = open("peer-a-render", titles[0])?;
        let (screen_b, window_b) = open("peer-b-render", titles[1])?;
        let shutdown = render_common::stop_on_close([window_a, window_b]);
        Ok((screen_a, screen_b, shutdown))
    }

    /// Nothing to share between the two sides: the renderers already have
    /// their device.
    pub(super) struct RenderContext;

    impl RenderContext {
        pub(super) fn new(_target: &Target) -> media_pp::Result<Self> {
            Ok(Self)
        }
    }

    /// `WebRtcSource -> Queue -> SwDecoder -> VulkanWindowRenderer`.
    pub(super) fn receive_pipeline(
        name: &str,
        source: WebRtcSource,
        stream: StreamInfo,
        _render: &RenderContext,
        screen: Target,
    ) -> media_pp::Result<Arc<Pipeline>> {
        validate_h264(name, &stream)?;
        let decoder = SwDecoder::new(format!("{name}-decode"), stream.parameters().clone())?;

        let (pipeline, ()) = Pipeline::new(name, source, move |source, ctx| {
            let branch = ctx.branch().queue("packets", 16).pipe(decoder).to(screen)?;
            ctx.attach(source, stream.index(), branch)?;
            Ok(())
        })?;
        Ok(pipeline)
    }

    fn validate_h264(name: &str, stream: &StreamInfo) -> media_pp::Result<()> {
        let codec = stream.parameters().id();
        if codec != ffmpeg::codec::Id::H264 {
            return Err(media_pp::Error::Other(format!(
                "{name} cannot decode the received {codec:?} stream; this example expects H.264"
            )));
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod macos_example {
    use std::sync::Arc;

    use media_pp::ffmpeg;
    use media_pp::{
        elements::{MetalWindowRenderer, StreamInfo, SwDecoder, WebRtcSource, WindowOptions},
        pipeline::Pipeline,
    };
    use render_common::Shutdown;

    use super::common::{HEIGHT, WIDTH};

    /// Where one side of the call is shown: the renderer that opened the
    /// window, already drawing into it.
    pub(super) type Target = MetalWindowRenderer;

    pub(super) fn run() {
        super::common::run();
    }

    /// Opens both windows, and the `Shutdown` that closing either sets off.
    ///
    /// What arrives is H.264 decoded to YUV420P in system memory, which the
    /// renderer draws as it comes and uploads itself; each window's GPU is
    /// its own.
    pub(super) fn open_windows(
        titles: [&str; 2],
    ) -> media_pp::Result<(Target, Target, Arc<Shutdown>)> {
        let open = |name: &str, title: &str| {
            MetalWindowRenderer::open(
                name,
                WindowOptions {
                    title: title.into(),
                    width: WIDTH,
                    height: HEIGHT,
                },
            )
        };
        let (screen_a, window_a) = open("peer-a-render", titles[0])?;
        let (screen_b, window_b) = open("peer-b-render", titles[1])?;
        let shutdown = render_common::stop_on_close([window_a, window_b]);
        Ok((screen_a, screen_b, shutdown))
    }

    /// Nothing to share between the two sides: the renderers already have
    /// their device.
    pub(super) struct RenderContext;

    impl RenderContext {
        pub(super) fn new(_target: &Target) -> media_pp::Result<Self> {
            Ok(Self)
        }
    }

    /// `WebRtcSource -> Queue -> SwDecoder -> MetalWindowRenderer`.
    pub(super) fn receive_pipeline(
        name: &str,
        source: WebRtcSource,
        stream: StreamInfo,
        _render: &RenderContext,
        screen: Target,
    ) -> media_pp::Result<Arc<Pipeline>> {
        validate_h264(name, &stream)?;
        let decoder = SwDecoder::new(format!("{name}-decode"), stream.parameters().clone())?;

        let (pipeline, ()) = Pipeline::new(name, source, move |source, ctx| {
            let branch = ctx.branch().queue("packets", 16).pipe(decoder).to(screen)?;
            ctx.attach(source, stream.index(), branch)?;
            Ok(())
        })?;
        Ok(pipeline)
    }

    fn validate_h264(name: &str, stream: &StreamInfo) -> media_pp::Result<()> {
        let codec = stream.parameters().id();
        if codec != ffmpeg::codec::Id::H264 {
            return Err(media_pp::Error::Other(format!(
                "{name} cannot decode the received {codec:?} stream; this example expects H.264"
            )));
        }
        Ok(())
    }
}

#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
mod common {
    use std::{
        net::UdpSocket,
        sync::{
            Arc,
            mpsc::{self, Receiver},
        },
        thread,
        time::{Duration, Instant},
    };

    #[cfg(target_os = "linux")]
    use super::linux_example as platform;
    #[cfg(target_os = "macos")]
    use super::macos_example as platform;
    #[cfg(target_os = "windows")]
    use super::windows_example as platform;
    use media_pp::{
        bus::BusEvent,
        driver::DriverRunner,
        element::BoxSink,
        elements::{
            FileDemuxer, Pacer, StreamInfo, SwDecoder, SwEncoder, SwEncoderOptions, SwScaler,
            TestVideoOptions, TestVideoSource, VideoCodec, WebRtcHandle, WebRtcPeer,
        },
        ffmpeg,
        pipeline::Pipeline,
    };
    use render_common::Shutdown;
    use str0m::{Candidate, Rtc, change::SdpOffer};

    /// What both directions of the call are encoded at. Fixed rather than
    /// derived from either source: the file side is scaled to it, and both
    /// window renderers are wired up once at this size.
    pub(super) const WIDTH: u32 = 640;
    pub(super) const HEIGHT: u32 = 480;
    const FPS: i32 = 30;

    const TITLES: [&str; 2] = [
        "media-pp webrtc_video_call — peer-a (showing peer-b's file)",
        "media-pp webrtc_video_call — peer-b (showing peer-a's test pattern)",
    ];

    pub(super) fn run() {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: webrtc_video_call <video.mp4>");
            std::process::exit(1);
        };

        // The renderers open their own windows here, and `play` runs on this
        // thread: there is no event loop of this program's to keep turning.
        if let Err(error) = platform::open_windows(TITLES)
            .and_then(|(screen_a, screen_b, shutdown)| play(&path, screen_a, screen_b, shutdown))
        {
            eprintln!("the call failed: {error}");
        }
    }

    fn play(
        path: &str,
        target_a: platform::Target,
        target_b: platform::Target,
        shutdown: Arc<Shutdown>,
    ) -> media_pp::Result<()> {
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

        let peers = connect_peers();

        // A track each way, each side's through a muxer of its own: one
        // renegotiation each, which this program carries between the two as
        // their signalling would.
        let encoder_a = SwEncoder::new("encode-a", encoder_options())?;
        let encoder_b = SwEncoder::new("encode-b", encoder_options())?;
        let sink_a = open_send(
            &peers.handle_a,
            &peers.offers_a,
            &peers.handle_b,
            "peer-a-video",
            &encoder_a,
        )?;
        let sink_b = open_send(
            &peers.handle_b,
            &peers.offers_b,
            &peers.handle_a,
            "peer-b-video",
            &encoder_b,
        )?;
        thread::sleep(Duration::from_millis(100)); // let the answers apply
        println!("call established — a track each way on one connection");

        // What the two receive pipelines share: nothing on either platform,
        // since each renderer already has its device.
        let render = platform::RenderContext::new(&target_a)?;

        let send_a = generated_send_pipeline(encoder_a, sink_a)?;
        // Kept before the sink moves into the pipeline: it is the name the
        // bus reports EOS under, and the one event below actually waits for.
        let track_sink_name = sink_b.name();
        let send_b = file_send_pipeline(file, encoder_b, sink_b)?;

        // Publish before starting, so a window close can reach both senders
        // while we wait for the first actual RTP payload on each source.
        let send_pipelines = [send_a.clone(), send_b.clone()];
        if shutdown.publish(&send_pipelines) {
            return Ok(());
        }
        for pipeline in &send_pipelines {
            pipeline.run()?;
        }

        // SDP only says which codecs may arrive. Each side's source of the
        // other's track comes once its first payload has confirmed what the
        // remote sender actually chose — for H.264, once SPS and PPS have
        // arrived; those packets stay queued while the matching decoder
        // pipeline is constructed.
        let (source_a, received_a) = peers.handle_a.next_source("from-peer-b")?;
        let (source_b, received_b) = peers.handle_b.next_source("from-peer-a")?;
        let [stream_a] = <[StreamInfo; 1]>::try_from(received_a)
            .map_err(|_| media_pp::Error::Other("peer-a expected one track".into()))?;
        let [stream_b] = <[StreamInfo; 1]>::try_from(received_b)
            .map_err(|_| media_pp::Error::Other("peer-b expected one track".into()))?;
        println!(
            "peer-a receiving {:?} {:?}",
            stream_a.parameters().id(),
            stream_a.size()
        );
        println!(
            "peer-b receiving {:?} {:?}",
            stream_b.parameters().id(),
            stream_b.size()
        );
        let recv_a =
            platform::receive_pipeline("peer-a-recv", source_a, stream_a, &render, target_a)?;
        let recv_b =
            platform::receive_pipeline("peer-b-recv", source_b, stream_b, &render, target_b)?;

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
        // Specifically the muxer track's own EOS, not the first one to
        // appear: every `Queue` along the way reports its own as EOS passes
        // through it, and the earliest of those means only that a queue has
        // drained — the encoder still has delayed frames to flush behind it.
        // The terminal sink's EOS is the one that means everything this side
        // had to send has actually been handed to the peer.
        for event in send_b.bus().iter() {
            println!("{event}");
            let sent_everything =
                matches!(&event, BusEvent::Eos { name, .. } if *name == track_sink_name);
            if sent_everything || matches!(event, BusEvent::Error { .. }) {
                break;
            }
        }

        for pipeline in &pipelines {
            pipeline.stop();
        }
        peers.driver_a.stop();
        peers.driver_b.stop();
        Ok(())
    }

    /// The sink of the one track `from` sends `to`, named `name` and fed by
    /// `encoder`: a muxer of `from`'s connection, whose offer goes to `to`
    /// and whose answer comes back.
    ///
    /// The encoder's stream declares the whole track from one value: the
    /// payload type, and the SPS/PPS the encoder keeps in `parameters()`
    /// rather than in the bitstream. RTP has no container to carry those,
    /// so the sink puts them in front of every keyframe — without them the
    /// far side's source would wait for parameter sets that never arrive.
    fn open_send(
        from: &WebRtcHandle,
        offers: &Receiver<SdpOffer>,
        to: &WebRtcHandle,
        name: &str,
        encoder: &SwEncoder,
    ) -> media_pp::Result<BoxSink> {
        let mut muxer = from.muxer(format!("{name}-send"));
        let track = muxer.add_stream(
            name,
            StreamInfo::new(encoder.parameters(), encoder.time_base()),
        )?;
        let mut sinks = muxer.open()?;
        let offer = offers
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| {
                media_pp::Error::Other(format!("{name}: no renegotiation offer: {error}"))
            })?;
        from.set_answer(to.accept_remote_offer(offer)?);
        sinks.take(track)
    }

    fn encoder_options() -> SwEncoderOptions {
        SwEncoderOptions {
            // Cisco's BSD-licensed H.264 encoder, so this needs no
            // `--enable-gpl` ffmpeg build — and H.264 is what every WebRTC
            // peer takes.
            codec: VideoCodec::OpenH264,
            width: WIDTH,
            height: HEIGHT,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(FPS, 1),
            bit_rate: 1_500_000,
            gop_size: 30,
            max_b_frames: None,
        }
    }

    /// peer-a's outbound half: `TestVideoSource -> Queue -> SwEncoder ->
    /// WebRtcMuxer`. The source paces itself, so nothing else has to.
    fn generated_send_pipeline(
        encoder: SwEncoder,
        sink: BoxSink,
    ) -> media_pp::Result<Arc<Pipeline>> {
        let options = TestVideoOptions {
            width: WIDTH,
            height: HEIGHT,
            frame_rate: ffmpeg::Rational::new(FPS, 1),
        };
        let source = TestVideoSource::new("test-video", options);

        let (pipeline, ()) = Pipeline::new("peer-a-send", source, move |source, ctx| {
            let branch = ctx.branch().queue("to-encode", 8).pipe(encoder).to(sink)?;
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
    }

    impl FileSource {
        /// Both failure modes name the path: ffmpeg reports an unreadable or
        /// non-media file as a bare "Invalid data found when processing
        /// input", which says nothing about which argument was wrong.
        fn open(path: &str) -> media_pp::Result<Self> {
            let (demuxer, _) = FileDemuxer::open("demux", path).map_err(|error| {
                media_pp::Error::Other(format!("cannot read `{path}` as a media file: {error}"))
            })?;
            let video = demuxer
                .best(ffmpeg::media::Type::Video)
                .map_err(|error| media_pp::Error::Other(format!("`{path}`: {error}")))?;
            let index = video.index();
            let params = video.parameters().clone();
            Ok(Self {
                demuxer,
                index,
                params,
            })
        }
    }

    /// peer-b's outbound half: `FileDemuxer -> SwDecoder -> Queue -> Pacer ->
    /// SwScaler -> Queue -> SwEncoder -> WebRtcMuxer`.
    ///
    /// The transcode is what makes the file usable as a call source at all:
    /// its frames are whatever size the file holds, and its packets arrive as
    /// fast as the disk can serve them. `SwScaler` fixes the first, `Pacer`
    /// the second — without it the whole file would be encoded and sent in a
    /// couple of seconds.
    fn file_send_pipeline(
        file: FileSource,
        encoder: SwEncoder,
        sink: BoxSink,
    ) -> media_pp::Result<Arc<Pipeline>> {
        let FileSource {
            demuxer: source,
            index,
            params,
        } = file;

        let decoder = SwDecoder::new("decode-file", params)?;
        let pacer = Pacer::new("pace-file");
        let scaler = SwScaler::new(
            "scale-file",
            ffmpeg::format::Pixel::YUV420P,
            WIDTH,
            HEIGHT,
            ffmpeg::software::scaling::Flags::BILINEAR,
        );

        let (pipeline, ()) = Pipeline::new("peer-b-send", source, move |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", 8) // the pacer sleeps; let demux/decode run ahead
                .pipe(pacer)
                .pipe(scaler)
                .queue("to-encode", 8)
                .pipe(encoder)
                .to(sink)?;
            ctx.attach(source, index, branch)?;
            Ok(())
        })?;
        Ok(pipeline)
    }

    /// Both peers of the call, and the renegotiation offers each makes.
    struct Peers {
        handle_a: WebRtcHandle,
        offers_a: Receiver<SdpOffer>,
        driver_a: Arc<DriverRunner>,
        handle_b: WebRtcHandle,
        offers_b: Receiver<SdpOffer>,
        driver_b: Arc<DriverRunner>,
    }

    /// Two `Rtc`s brought up over loopback UDP with a throwaway data channel,
    /// then handed to `WebRtcPeer`s — the same signaling stand-in
    /// `webrtc_loopback` uses, since this example is about the media, not
    /// about transporting SDP.
    fn connect_peers() -> Peers {
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

        let driver_a = DriverRunner::new(peer_a);
        let driver_b = DriverRunner::new(peer_b);
        driver_a.run().expect("peer-a should start");
        driver_b.run().expect("peer-b should start");
        thread::sleep(Duration::from_millis(200));
        println!("ICE/DTLS-SRTP established over loopback UDP");

        Peers {
            handle_a,
            offers_a,
            driver_a,
            handle_b,
            offers_b,
            driver_b,
        }
    }
}
