//! A stream carried across every pair of protocols the crate serves: in
//! through WHIP or RTSP, out through WHEP or RTSP, or into a file — on the
//! loopback, between this crate's own clients and servers.
//!
//! What goes in is a live picture of OpenH264 — no B-frames, which WebRTC
//! needs — and sound: Opus where the FFmpeg build has libopus, AAC where
//! the way in is RTSP and the point is that WebRTC leaves it out. What
//! comes out is judged by pictures that decode, or a file that plays.
#![cfg(all(feature = "webrtc", feature = "rtsp-server"))]

use std::sync::Arc;
use std::time::{Duration, Instant};

use media_pp::bus::BusEvent;
use media_pp::driver::DriverRunner;
use media_pp::element::BoxSink;
use media_pp::elements::{
    AudioCodec, Broadcast, BroadcastOptions, BroadcastSource, CounterHandle, FileDemuxer,
    FileMuxer, FrameCounter, HlsMode, HlsMuxer, HlsOptions, Mp4Layout, PacketCounter, RtspMuxer,
    RtspOptions, RtspPathMode, RtspServer, RtspServerHandle, RtspServerOptions, RtspSource,
    StreamInfo, SwAudioEncoder, SwAudioEncoderOptions, SwDecoder, SwEncoder, SwEncoderOptions,
    TestAudioOptions, TestAudioSource, TestVideoOptions, TestVideoSource, TrackFormat, VideoCodec,
    WebRtcError, WebRtcHttpEndpoint, WebRtcPathMode, WebRtcServer, WebRtcServerHandle,
    WebRtcServerOptions, WhepOptions, WhepSource, WhipOptions, WhipPublisher,
};
use media_pp::ffmpeg::{self, media};
use media_pp::pipeline::{Pipeline, PipelineBuilder};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const RATE: i32 = 15;
/// Pictures a player is to decode before a crossing is taken to work: two
/// seconds' worth, from a keyframe.
const PICTURES: usize = 30;
const WITHIN: Duration = Duration::from_secs(20);
/// How long a recording runs before it is finished and read: some fifty
/// pictures, from a keyframe.
const RECORD_FOR: Duration = Duration::from_millis(3_500);

/// The two encoders a live input is made with.
struct Encoders {
    video: SwEncoder,
    audio: Option<SwAudioEncoder>,
}

/// A picture encoder, and a sound encoder of `audio` where the build has
/// it — Opus needs libopus, which a stripped FFmpeg leaves out.
fn encoders(audio: AudioCodec) -> Encoders {
    let video = SwEncoder::new(
        "video-encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: WIDTH,
            height: HEIGHT,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(RATE, 1),
            bit_rate: 300_000,
            gop_size: RATE as u32,
            max_b_frames: None,
        },
    )
    .expect("OpenH264 is always there");
    let options = TestAudioOptions::default();
    let audio = SwAudioEncoder::new(
        "audio-encoder",
        SwAudioEncoderOptions {
            codec: audio,
            sample_rate: options.sample_rate,
            channels: options.channels,
            bit_rate: 64_000,
        },
    )
    .ok();
    Encoders { video, audio }
}

/// A live input — a picture, and sound where there is an audio sink — into
/// `video` and `audio`, running.
fn live(encoders: Encoders, video: BoxSink, audio: Option<BoxSink>) -> Arc<Pipeline> {
    let pictures = TestVideoSource::new(
        "pictures",
        TestVideoOptions {
            width: WIDTH,
            height: HEIGHT,
            frame_rate: ffmpeg::Rational::new(RATE, 1),
        },
    );
    let encoder = encoders.video;
    let (builder, ()) = PipelineBuilder::new("input")
        .add_source(pictures, |source, ctx| {
            let branch = ctx
                .branch()
                .queue("encode-video", 4)
                .pipe(encoder)
                .to(video)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .unwrap();
    let builder = match (encoders.audio, audio) {
        (Some(encoder), Some(sink)) => {
            let sound = TestAudioSource::new("sound", TestAudioOptions::default());
            builder
                .add_source(sound, |source, ctx| {
                    let branch = ctx
                        .branch()
                        .queue("encode-audio", 4)
                        .pipe(encoder)
                        .to(sink)?;
                    ctx.attach(source, 0, branch)?;
                    Ok(())
                })
                .unwrap()
                .0
        }
        _ => builder,
    };
    let pipeline = builder.build();
    pipeline.run().unwrap();
    pipeline
}

/// Both servers on the loopback, serving `broadcast` under `path` — the
/// WebRTC one taking a WHIP publisher where `webrtc` is `Publish`, the RTSP
/// one an RTSP publisher where `rtsp` is `Record`.
struct Servers {
    webrtc: Arc<DriverRunner>,
    _webrtc_handle: WebRtcServerHandle,
    http: WebRtcHttpEndpoint,
    rtsp: Arc<DriverRunner>,
    rtsp_handle: RtspServerHandle,
}

impl Servers {
    fn serving(
        path: &str,
        broadcast: &Broadcast,
        webrtc: WebRtcPathMode,
        rtsp: RtspPathMode,
    ) -> Self {
        let (server, webrtc_handle) =
            WebRtcServer::bind("webrtc", "127.0.0.1:0", WebRtcServerOptions::default()).unwrap();
        webrtc_handle.add_path(path, broadcast, webrtc).unwrap();
        let webrtc_runner = DriverRunner::new(server);
        webrtc_runner.run().unwrap();
        let http = webrtc_handle.serve_http("127.0.0.1:0").unwrap();
        let (server, rtsp_handle) =
            RtspServer::bind("rtsp", "127.0.0.1:0", RtspServerOptions::default()).unwrap();
        rtsp_handle.add_path(path, broadcast, rtsp).unwrap();
        let rtsp_runner = DriverRunner::new(server);
        rtsp_runner.run().unwrap();
        Self {
            webrtc: webrtc_runner,
            _webrtc_handle: webrtc_handle,
            http,
            rtsp: rtsp_runner,
            rtsp_handle,
        }
    }

    fn stop(self) {
        drop(self.http);
        self.webrtc.stop();
        self.rtsp.stop();
    }
}

/// What a player of a stream got: pictures decoded, sound packets.
struct Played {
    pipeline: Arc<Pipeline>,
    pictures: CounterHandle,
    sound: Option<CounterHandle>,
}

/// Plays `streams` from `source`: its picture decoded and counted, its
/// sound, if it has any, counted as packets.
fn play<M>(source: impl media_pp::element::IntoSource<M>, streams: &[StreamInfo]) -> Played {
    let video = streams
        .iter()
        .find(|stream| stream.kind == media::Type::Video)
        .expect("a picture");
    let audio = streams
        .iter()
        .find(|stream| stream.kind == media::Type::Audio);
    let decoder = SwDecoder::new("decoder", video.parameters.clone()).unwrap();
    let (frames, pictures) = FrameCounter::new("pictures");
    let (packets, sound) = PacketCounter::new("sound");
    let (pipeline, ()) = Pipeline::new("player", source, |source, ctx| {
        ctx.attach(source, video.index, ctx.branch().pipe(decoder).to(frames)?)?;
        if let Some(audio) = audio {
            ctx.attach(source, audio.index, ctx.branch().to(packets)?)?;
        }
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    Played {
        pipeline,
        pictures,
        sound: audio.map(|_| sound),
    }
}

fn wait_for(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + WITHIN;
    while !done() {
        assert!(Instant::now() < deadline, "{what} within {WITHIN:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Waits for `played` to decode `PICTURES` pictures and, where both ends
/// carry sound, for some sound.
fn plays(played: &Played, sound_expected: bool) {
    wait_for("pictures decoded", || played.pictures.get() >= PICTURES);
    if sound_expected {
        let sound = played.sound.as_ref().expect("a sound track");
        wait_for("sound received", || sound.get() > 0);
    }
}

/// A WHIP publisher of `encoders` to `url`, and the input feeding it.
fn publish_whip(url: &str, encoders: Encoders) -> Arc<Pipeline> {
    let mut publisher = WhipPublisher::create("publisher", url, WhipOptions::default()).unwrap();
    let video = publisher.add_track(&encoders.video.parameters()).unwrap();
    let audio = encoders
        .audio
        .as_ref()
        .map(|encoder| publisher.add_track(&encoder.parameters()).unwrap());
    let mut sinks = publisher.open().expect("the server takes the publisher");
    let video = BoxSink::new(sinks.take(video).unwrap());
    let audio = audio.map(|track| BoxSink::new(sinks.take(track).unwrap()));
    live(encoders, video, audio)
}

/// An RTSP publisher of `encoders` to `url`, and the input feeding it.
fn publish_rtsp(url: &str, encoders: Encoders) -> Arc<Pipeline> {
    let mut muxer = RtspMuxer::create(url, RtspOptions::default()).unwrap();
    let video = muxer
        .add_stream("video", TrackFormat::from(&encoders.video))
        .unwrap();
    let audio = encoders.audio.as_ref().map(|encoder| {
        muxer
            .add_stream("audio", TrackFormat::from(encoder))
            .unwrap()
    });
    let mut sinks = muxer.open().expect("the server takes the publisher");
    let video = sinks.take(video).unwrap();
    let audio = audio.map(|track| sinks.take(track).unwrap());
    live(encoders, video, audio)
}

/// Waits for `broadcast` to be published into.
fn until_live(broadcast: &Broadcast) {
    wait_for("the publisher's stream reached its broadcast", || {
        broadcast.is_live()
    });
}

/// Records `streams` of `source` into an MP4 at `path`, running.
fn record<M>(
    source: impl media_pp::element::IntoSource<M>,
    streams: &[StreamInfo],
    path: &std::path::Path,
) -> Arc<Pipeline> {
    let mut muxer = FileMuxer::create(path, Mp4Layout::default()).unwrap();
    let tracks: Vec<_> = streams
        .iter()
        .map(|stream| {
            (
                stream.index,
                muxer
                    .add_stream(format!("track-{}", stream.index), TrackFormat::from(stream))
                    .unwrap(),
            )
        })
        .collect();
    let mut sinks = muxer.open().unwrap();
    let (pipeline, ()) = Pipeline::new("recorder", source, |source, ctx| {
        for (index, track) in tracks {
            ctx.attach(source, index, ctx.branch().to(sinks.take(track)?)?)?;
        }
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    pipeline
}

/// Records `streams` of `source` as an HLS event playlist under `dir` —
/// fMP4 segments of a second, as a server keeps what it serves for playing
/// back later: playable as it grows, and whole once finished — running.
fn record_hls<M>(
    source: impl media_pp::element::IntoSource<M>,
    streams: &[StreamInfo],
    dir: &std::path::Path,
) -> Arc<Pipeline> {
    std::fs::create_dir_all(dir).unwrap();
    let mut options = HlsOptions::new(dir.join("index.m3u8"), dir.join("segment_%05d.m4s"));
    options.mode = HlsMode::Event;
    options.segment_duration = Duration::from_secs(1);
    let mut muxer = HlsMuxer::create(options).unwrap();
    let tracks: Vec<_> = streams
        .iter()
        .map(|stream| {
            (
                stream.index,
                muxer
                    .add_stream(format!("track-{}", stream.index), TrackFormat::from(stream))
                    .unwrap(),
            )
        })
        .collect();
    let mut sinks = muxer.open().unwrap();
    let (pipeline, ()) = Pipeline::new("hls-recorder", source, |source, ctx| {
        for (index, track) in tracks {
            ctx.attach(source, index, ctx.branch().to(sinks.take(track)?)?)?;
        }
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    pipeline
}

/// How many pictures the recording at `path` — an MP4, or an HLS
/// playlist — decodes to, and whether it has sound.
fn in_file(path: &std::path::Path) -> (usize, bool) {
    let (source, streams) = FileDemuxer::open("check", path).unwrap();
    let video = streams
        .iter()
        .find(|stream| stream.kind == media::Type::Video)
        .expect("the file has a picture")
        .clone();
    let sound = streams
        .iter()
        .any(|stream| stream.kind == media::Type::Audio);
    let decoder = SwDecoder::new("decoder", video.parameters.clone()).unwrap();
    let (frames, pictures) = FrameCounter::new("pictures");
    let (pipeline, ()) = Pipeline::new("check", source, |source, ctx| {
        ctx.attach(source, video.index, ctx.branch().pipe(decoder).to(frames)?)?;
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    let deadline = Instant::now() + WITHIN;
    while Instant::now() < deadline {
        if let Ok(BusEvent::Finished) = pipeline.bus().recv_timeout(Duration::from_millis(100)) {
            break;
        }
    }
    pipeline.stop();
    (pictures.get(), sound)
}

/// A path of its own in the system's temporary directory, removed with it.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("media_pp_streaming_{}_{name}", std::process::id())))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Records for long enough to have `PICTURES`, then finishes the recording.
fn recorded_for_a_while(recorder: &Pipeline, input: &Pipeline) {
    std::thread::sleep(RECORD_FOR);
    recorder.finish();
    input.stop();
}

#[test]
fn whip_in_whep_out() {
    let encoders = encoders(AudioCodec::Opus);
    let sound = encoders.audio.is_some();
    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );
    let input = publish_whip(&servers.http.whip_url("live"), encoders);
    until_live(&live_stream);

    let (source, streams) = WhepSource::open(
        "player",
        &servers.http.whep_url("live"),
        WhepOptions::default(),
    )
    .unwrap();
    assert_eq!(
        streams.len(),
        if sound { 2 } else { 1 },
        "a picture, and sound where there is"
    );
    let played = play(source, &streams);
    plays(&played, sound);

    played.pipeline.stop();
    input.stop();
    servers.stop();
}

#[test]
fn whip_in_rtsp_out() {
    let encoders = encoders(AudioCodec::Opus);
    let sound = encoders.audio.is_some();
    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );
    let input = publish_whip(&servers.http.whip_url("live"), encoders);
    until_live(&live_stream);

    let (source, streams) = RtspSource::open(
        "player",
        servers.rtsp_handle.url("live"),
        RtspOptions::default(),
    )
    .unwrap();
    let played = play(source, &streams);
    plays(&played, sound);

    played.pipeline.stop();
    input.stop();
    servers.stop();
}

/// RTSP in, WebRTC out: the picture goes; AAC, which WebRTC does not carry,
/// is left out — and Opus, which it does, goes.
#[test]
fn rtsp_in_whep_out() {
    for audio in [AudioCodec::Aac, AudioCodec::Opus] {
        let encoders = encoders(audio);
        let Some(_) = &encoders.audio else {
            continue; // no libopus here
        };
        let camera = Broadcast::new("camera", BroadcastOptions::default());
        let servers = Servers::serving("cam", &camera, WebRtcPathMode::Play, RtspPathMode::Record);
        let input = publish_rtsp(&servers.rtsp_handle.url("cam"), encoders);
        until_live(&camera);

        let (source, streams) = WhepSource::open(
            "player",
            &servers.http.whep_url("cam"),
            WhepOptions::default(),
        )
        .unwrap();
        let carried = audio == AudioCodec::Opus;
        assert_eq!(
            streams
                .iter()
                .any(|stream| stream.kind == media::Type::Audio),
            carried,
            "{audio:?} sound {} WebRTC",
            if carried {
                "goes through"
            } else {
                "is left out of"
            }
        );
        let played = play(source, &streams);
        plays(&played, carried);

        played.pipeline.stop();
        input.stop();
        servers.stop();
    }
}

#[test]
fn whip_in_recorded() {
    let encoders = encoders(AudioCodec::Opus);
    let sound = encoders.audio.is_some();
    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );
    let input = publish_whip(&servers.http.whip_url("live"), encoders);
    until_live(&live_stream);

    let scratch = Scratch::new("whip.mp4");
    let file = scratch.0.clone();
    let (source, streams) = BroadcastSource::new("recorder", &live_stream).unwrap();
    let recorder = record(source, &streams, &file);
    recorded_for_a_while(&recorder, &input);

    let (pictures, has_sound) = in_file(&file);
    assert!(pictures >= PICTURES, "{pictures} pictures recorded");
    assert_eq!(has_sound, sound);
    servers.stop();
}

#[test]
fn rtsp_in_recorded() {
    let encoders = encoders(AudioCodec::Aac);
    let camera = Broadcast::new("camera", BroadcastOptions::default());
    let servers = Servers::serving("cam", &camera, WebRtcPathMode::Play, RtspPathMode::Record);
    let input = publish_rtsp(&servers.rtsp_handle.url("cam"), encoders);
    until_live(&camera);

    let scratch = Scratch::new("rtsp.mp4");
    let file = scratch.0.clone();
    let (source, streams) = BroadcastSource::new("recorder", &camera).unwrap();
    let recorder = record(source, &streams, &file);
    recorded_for_a_while(&recorder, &input);

    let (pictures, has_sound) = in_file(&file);
    assert!(pictures >= PICTURES, "{pictures} pictures recorded");
    assert!(has_sound, "AAC goes into an MP4 as it came");
    servers.stop();
}

/// A remote WebRTC stream recorded straight from a `WhepSource`.
#[test]
fn whep_out_recorded() {
    let encoders = encoders(AudioCodec::Opus);
    let sound = encoders.audio.is_some();
    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );
    let input = publish_whip(&servers.http.whip_url("live"), encoders);
    until_live(&live_stream);

    let scratch = Scratch::new("whep.mp4");
    let file = scratch.0.clone();
    let (source, streams) = WhepSource::open(
        "recorder",
        &servers.http.whep_url("live"),
        WhepOptions::default(),
    )
    .unwrap();
    let recorder = record(source, &streams, &file);
    recorded_for_a_while(&recorder, &input);

    let (pictures, has_sound) = in_file(&file);
    assert!(pictures >= PICTURES, "{pictures} pictures recorded");
    assert_eq!(has_sound, sound);
    servers.stop();
}

/// A publisher that leaves and comes back: a player playing throughout
/// goes on into the second run.
#[test]
fn a_player_goes_on_across_a_publisher_coming_back() {
    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );
    let input = publish_whip(&servers.http.whip_url("live"), encoders(AudioCodec::Opus));
    until_live(&live_stream);
    let (source, streams) = WhepSource::open(
        "player",
        &servers.http.whep_url("live"),
        WhepOptions::default(),
    )
    .unwrap();
    let played = play(source, &streams);
    wait_for("pictures decoded", || played.pictures.get() >= PICTURES);

    // The publisher leaves: its sinks go with its input, which ends its
    // session, and the broadcast's run with it.
    input.stop();
    drop(input);
    wait_for("the run ended", || !live_stream.is_live());
    let before = played.pictures.get();

    let input = publish_whip(&servers.http.whip_url("live"), encoders(AudioCodec::Opus));
    until_live(&live_stream);
    wait_for("pictures of the second run", || {
        played.pictures.get() >= before + PICTURES
    });

    played.pipeline.stop();
    input.stop();
    servers.stop();
}

#[test]
fn what_a_server_refuses_is_an_error_at_open() {
    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );

    let refused = |error: media_pp::Error| match error {
        media_pp::Error::WebRtcError(WebRtcError::Refused { status, .. }) => status,
        other => panic!("refused with {other}"),
    };
    // A path the server does not serve.
    let error = WhepSource::open(
        "player",
        &servers.http.whep_url("nowhere"),
        WhepOptions::default(),
    )
    .err()
    .unwrap();
    assert_eq!(refused(error), 404);

    // A second publisher while the first publishes.
    let input = publish_whip(&servers.http.whip_url("live"), encoders(AudioCodec::Opus));
    until_live(&live_stream);
    let mut second = WhipPublisher::create(
        "second",
        &servers.http.whip_url("live"),
        WhipOptions::default(),
    )
    .unwrap();
    second
        .add_track(&encoders(AudioCodec::Opus).video.parameters())
        .unwrap();
    assert_eq!(refused(second.open().err().unwrap()), 409);

    // A codec WebRTC does not carry, refused before anything is sent.
    let mut aac = WhipPublisher::create(
        "aac",
        &servers.http.whip_url("live"),
        WhipOptions::default(),
    )
    .unwrap();
    let encoder = encoders(AudioCodec::Aac)
        .audio
        .expect("AAC is always there");
    assert!(matches!(
        aac.add_track(&encoder.parameters()),
        Err(media_pp::Error::WebRtcError(
            WebRtcError::SourceCodecUnsupported(_)
        ))
    ));

    input.stop();
    servers.stop();

    // Nothing there at all.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let error = WhepSource::open(
        "player",
        &format!("http://127.0.0.1:{port}/live/whep"),
        WhepOptions::default(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(
            error,
            media_pp::Error::WebRtcError(WebRtcError::Request { .. })
        ),
        "{error}"
    );
}

/// The same round trip through a WHIP and WHEP server of another make —
/// MediaMTX, say — named by `MEDIA_PP_TEST_WHIP_SERVER` as its base URL
/// (`http://127.0.0.1:8889`), under which `<path>/whip` takes a publisher
/// and `<path>/whep` plays it. Skipped, saying so, where none is named.
#[test]
fn against_a_whip_server_of_another_make() {
    let Ok(base) = std::env::var("MEDIA_PP_TEST_WHIP_SERVER") else {
        eprintln!("skipped: MEDIA_PP_TEST_WHIP_SERVER names no WHIP/WHEP server to try");
        return;
    };
    let base = base.trim_end_matches('/');
    let path = format!("media-pp-{}", std::process::id());
    let encoders = encoders(AudioCodec::Opus);
    let sound = encoders.audio.is_some();
    let input = publish_whip(&format!("{base}/{path}/whip"), encoders);

    let (source, streams) = WhepSource::open(
        "player",
        &format!("{base}/{path}/whep"),
        WhepOptions::default(),
    )
    .unwrap();
    let played = play(source, &streams);
    plays(&played, sound);

    played.pipeline.stop();
    input.stop();
}

/// A `WhepSource` played in a driven pipeline — a relay's, polled on a
/// runner's thread — as on a thread of its own: what its connection receives
/// wakes the poll.
#[test]
fn whep_out_driven() {
    use media_pp::pipeline::{DrivenPipeline, DrivenRunner};

    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );
    let input = publish_whip(&servers.http.whip_url("live"), encoders(AudioCodec::Opus));
    until_live(&live_stream);

    let (source, streams) = WhepSource::open(
        "player",
        &servers.http.whep_url("live"),
        WhepOptions::default(),
    )
    .unwrap();
    let video = streams
        .iter()
        .find(|stream| stream.kind == media::Type::Video)
        .unwrap();
    let decoder = SwDecoder::new("decoder", video.parameters.clone()).unwrap();
    let (frames, pictures) = FrameCounter::new("pictures");
    let (pipeline, ()) = DrivenPipeline::new("player", source, |source, ctx| {
        ctx.attach(source, video.index, ctx.branch().pipe(decoder).to(frames)?)?;
        Ok(())
    })
    .unwrap();
    let handle = pipeline.handle();
    let runner = DrivenRunner::new(1).unwrap();
    runner.spawn(pipeline);
    wait_for("pictures decoded, driven", || pictures.get() >= PICTURES);

    handle.stop();
    input.stop();
    servers.stop();
}

/// One input recorded while it is played both ways at once: a WHIP
/// publisher's stream into an HLS recording, to a WHEP player and to an
/// RTSP player, all from the one broadcast — none of them waiting on
/// another. The recording is read back through its playlist.
#[test]
fn recorded_while_played_both_ways() {
    let encoders = encoders(AudioCodec::Opus);
    let sound = encoders.audio.is_some();
    let live_stream = Broadcast::new("live", BroadcastOptions::default());
    let servers = Servers::serving(
        "live",
        &live_stream,
        WebRtcPathMode::Publish,
        RtspPathMode::Play,
    );
    let input = publish_whip(&servers.http.whip_url("live"), encoders);
    until_live(&live_stream);

    let scratch = Scratch::new("both-ways-hls");
    let (source, streams) = BroadcastSource::new("recorder", &live_stream).unwrap();
    let recorder = record_hls(source, &streams, &scratch.0);
    let (source, streams) = WhepSource::open(
        "webrtc-player",
        &servers.http.whep_url("live"),
        WhepOptions::default(),
    )
    .unwrap();
    let webrtc_player = play(source, &streams);
    let (source, streams) = RtspSource::open(
        "rtsp-player",
        servers.rtsp_handle.url("live"),
        RtspOptions::default(),
    )
    .unwrap();
    let rtsp_player = play(source, &streams);

    plays(&webrtc_player, sound);
    plays(&rtsp_player, sound);
    // A player joins with what the broadcast holds and catches up at
    // once; the recording goes at the stream's own pace, so it is given
    // its own time rather than the players'.
    std::thread::sleep(RECORD_FOR);
    recorder.finish();
    let (pictures, has_sound) = in_file(&scratch.0.join("index.m3u8"));
    assert!(pictures >= PICTURES, "{pictures} pictures recorded");
    assert_eq!(has_sound, sound);
    // The players went on as the recording finished.
    let webrtc_before = webrtc_player.pictures.get();
    let rtsp_before = rtsp_player.pictures.get();
    wait_for("both players go on", || {
        webrtc_player.pictures.get() > webrtc_before && rtsp_player.pictures.get() > rtsp_before
    });

    webrtc_player.pipeline.stop();
    rtsp_player.pipeline.stop();
    input.stop();
    servers.stop();
}
