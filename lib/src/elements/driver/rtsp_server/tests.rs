//! The server against FFmpeg's own RTSP client — `RtspSource::open` — and
//! against requests written by hand, on this machine's loopback alone.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;

use super::*;
use crate::buffer::MediaBuffer;
use crate::driver::DriverRunner;
use crate::elements::{
    BroadcastOptions, FileDemuxer, Pacer, RtspOptions, RtspSource, RtspTransport, StreamInfo,
    TrackFormat,
};
use crate::pipeline::Pipeline;
use crate::test_support::CapturingSink;

/// A server on a port of the system's choice on the loopback, serving one
/// broadcast at `cam`.
fn server(udp: bool) -> (Arc<DriverRunner>, RtspServerHandle, Broadcast) {
    let (server, handle) = RtspServer::bind(
        "server",
        "127.0.0.1:0",
        RtspServerOptions {
            udp,
            publisher_timeout: Duration::from_secs(2),
            ..RtspServerOptions::default()
        },
    )
    .expect("a loopback port is free");
    let broadcast = Broadcast::new("cam", BroadcastOptions::default());
    handle
        .add_path("cam", &broadcast, RtspPathMode::Play)
        .unwrap();
    let runner = DriverRunner::new(server);
    runner.run().unwrap();
    (runner, handle, broadcast)
}

/// Plays the fixture into `broadcast` at its own speed, over and over, as a
/// camera would.
fn feed(broadcast: &Broadcast, path: &str) -> Arc<Pipeline> {
    let (demuxer, streams) = FileDemuxer::open("camera", path).unwrap();
    demuxer.looping_handle().set_looping(true);
    let tracks: Vec<TrackFormat> = streams.iter().map(TrackFormat::from).collect();
    let sinks = broadcast.publish(&tracks).unwrap();
    let (pipeline, ()) = Pipeline::new("camera", demuxer, |source, ctx| {
        for (stream, sink) in streams.iter().zip(sinks) {
            let branch = ctx
                .branch()
                .queue(format!("packets-{}", stream.index), 64)
                .pipe(Pacer::new(format!("pacer-{}", stream.index)))
                .to(sink)?;
            ctx.attach(source, stream.index, branch)?;
        }
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    pipeline
}

type Captured = Arc<Mutex<Vec<MediaBuffer>>>;

/// A player of `url`: what it was described, and what each of its streams
/// has received while it plays.
struct Player {
    streams: Vec<StreamInfo>,
    received: Vec<Captured>,
    pipeline: Arc<Pipeline>,
}

fn play(url: &str, transport: RtspTransport) -> Player {
    let (source, streams) = RtspSource::open(
        "player",
        url,
        RtspOptions {
            transport,
            timeout: Duration::from_secs(5),
        },
    )
    .unwrap_or_else(|error| panic!("{url} opens: {error}"));
    let received: Vec<Captured> = streams.iter().map(|_| Captured::default()).collect();
    let sinks: Vec<_> = received
        .iter()
        .map(|received| CapturingSink {
            pp_log: crate::element::element_pp_log(
                crate::element::ElementType::Other,
                "capture",
                None,
            ),
            received: Arc::clone(received),
        })
        .collect();
    let (pipeline, ()) = Pipeline::new("player", source, |source, ctx| {
        for (stream, sink) in streams.iter().zip(sinks) {
            let branch = ctx
                .branch()
                .queue(format!("received-{}", stream.index), 256)
                .to(sink)?;
            ctx.attach(source, stream.index, branch)?;
        }
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    Player {
        streams,
        received,
        pipeline,
    }
}

impl Player {
    /// Waits until every stream has received at least `count` packets.
    fn wait_for(&self, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while self
            .received
            .iter()
            .any(|r| r.lock().unwrap().len() < count)
        {
            assert!(
                Instant::now() < deadline,
                "only {:?} packets arrived",
                self.received
                    .iter()
                    .map(|r| r.lock().unwrap().len())
                    .collect::<Vec<_>>()
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn index_of(&self, medium: ffmpeg::media::Type) -> usize {
        self.streams.iter().position(|s| s.kind == medium).unwrap()
    }

    fn packets(&self, stream: usize) -> Vec<(Option<i64>, bool)> {
        self.received[stream]
            .lock()
            .unwrap()
            .iter()
            .map(|buf| match buf {
                MediaBuffer::Packet(packet) => (packet.pts(), packet.is_key()),
                _ => panic!("an RTSP player is handed packets"),
            })
            .collect()
    }
}

/// What every played stream must show: the fixture's tracks, the picture
/// beginning at a keyframe, and the two tracks starting together.
fn check_played(player: &Player) {
    assert_eq!(
        player.streams.len(),
        2,
        "both of the fixture's tracks are described"
    );
    let video = player.index_of(ffmpeg::media::Type::Video);
    let audio = player.index_of(ffmpeg::media::Type::Audio);
    assert_eq!(
        player.streams[video].parameters.id(),
        ffmpeg::codec::Id::H264
    );
    assert_eq!(
        player.streams[audio].parameters.id(),
        ffmpeg::codec::Id::AAC
    );
    player.wait_for(30);
    let pictures = player.packets(video);
    assert!(pictures[0].1, "the first picture played is a keyframe");
    let pts: Vec<i64> = pictures.iter().filter_map(|(pts, _)| *pts).collect();
    assert!(
        pts.windows(2).all(|w| w[0] < w[1]),
        "pictures in order: {pts:?}"
    );
    let seconds =
        |stream: usize, pts: i64| pts as f64 * f64::from(player.streams[stream].time_base);
    let first_picture = seconds(video, pts[0]);
    let first_sound = seconds(audio, player.packets(audio)[0].0.unwrap());
    assert!(
        (first_picture - first_sound).abs() < 0.5,
        "the tracks start together: picture at {first_picture}s, sound at {first_sound}s"
    );
}

#[test]
fn a_player_over_tcp_gets_every_track_from_a_keyframe() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(false);
    let camera = feed(&broadcast, &path);
    let player = play(&handle.url("cam"), RtspTransport::Tcp);
    check_played(&player);
    assert_eq!(handle.players(), 1);
    player.pipeline.stop();
    camera.stop();
    runner.stop();
}

#[test]
fn a_player_over_udp_gets_every_track_from_a_keyframe() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(true);
    let camera = feed(&broadcast, &path);
    let player = play(&handle.url("cam"), RtspTransport::Udp);
    check_played(&player);
    player.pipeline.stop();
    camera.stop();
    runner.stop();
}

#[test]
fn many_players_play_at_once() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(true);
    let camera = feed(&broadcast, &path);
    let players: Vec<Player> = (0..8)
        .map(|n| {
            let transport = if n % 2 == 0 {
                RtspTransport::Tcp
            } else {
                RtspTransport::Udp
            };
            play(&handle.url("cam"), transport)
        })
        .collect();
    for player in &players {
        check_played(player);
    }
    assert_eq!(handle.players(), 8);
    for player in &players {
        player.pipeline.stop();
    }
    drop(players);
    let deadline = Instant::now() + Duration::from_secs(5);
    while handle.players() > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        handle.players(),
        0,
        "players that leave are no longer counted"
    );
    camera.stop();
    runner.stop();
}

/// An input that drops and comes back publishes again; its players go on,
/// on one timeline.
#[test]
fn players_go_on_when_the_input_is_published_again() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(false);
    let camera = feed(&broadcast, &path);
    let player = play(&handle.url("cam"), RtspTransport::Tcp);
    player.wait_for(20);
    camera.stop();
    thread::sleep(Duration::from_millis(300));
    let video = player.index_of(ffmpeg::media::Type::Video);
    let before = player.packets(video).len();
    let camera = feed(&broadcast, &path);
    let deadline = Instant::now() + Duration::from_secs(20);
    while player.packets(video).len() < before + 30 {
        assert!(
            Instant::now() < deadline,
            "nothing arrived after the input came back"
        );
        thread::sleep(Duration::from_millis(20));
    }
    // FFmpeg's RTSP client hands its first packet on unstamped.
    let packets: Vec<(i64, bool)> = player
        .packets(video)
        .into_iter()
        .filter_map(|(pts, key)| Some((pts?, key)))
        .collect();
    assert!(
        packets.windows(2).all(|w| w[0].0 < w[1].0),
        "one timeline across the drop"
    );
    // Where the input was gone: a gap of about the 300 ms it was.
    let unit = f64::from(player.streams[video].time_base);
    let gap = packets
        .windows(2)
        .position(|w| (w[1].0 - w[0].0) as f64 * unit > 0.2)
        .expect("the drop shows as a gap")
        + 1;
    assert!(packets[gap].1, "the second run begins at a keyframe");
    player.pipeline.stop();
    camera.stop();
    runner.stop();
}

#[test]
fn stopping_the_server_ends_its_players() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(false);
    let camera = feed(&broadcast, &path);
    let player = play(&handle.url("cam"), RtspTransport::Tcp);
    player.wait_for(5);
    runner.stop();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match player
            .pipeline
            .bus()
            .recv_timeout(Duration::from_millis(100))
        {
            Ok(crate::bus::BusEvent::Finished | crate::bus::BusEvent::Error { .. }) => break,
            _ => assert!(
                Instant::now() < deadline,
                "the player did not see the server go"
            ),
        }
    }
    player.pipeline.stop();
    camera.stop();
}

#[test]
fn paths_are_checked_as_they_are_added() {
    let (_runner, handle, broadcast) = server(false);
    assert!(matches!(
        handle.add_path("cam", &broadcast, RtspPathMode::Play),
        Err(RtspServerError::PathTaken(_))
    ));
    for bad in [
        "",
        "/cam",
        "cam/",
        "a//b",
        "cam/streamid=0",
        "c am",
        "cam?x",
    ] {
        assert!(
            matches!(
                handle.add_path(bad, &broadcast, RtspPathMode::Play),
                Err(RtspServerError::InvalidPath(_))
            ),
            "{bad:?}"
        );
    }
    handle
        .add_path("site/door-1", &broadcast, RtspPathMode::Play)
        .unwrap();
    assert!(handle.remove_path("site/door-1"));
    assert!(!handle.remove_path("site/door-1"));
}

#[test]
fn binding_a_taken_port_fails_at_once() {
    let (_runner, handle, _broadcast) = server(false);
    let taken = handle.local_addr();
    assert!(matches!(
        RtspServer::bind("again", taken, RtspServerOptions::default()),
        Err(RtspServerError::Bind { .. })
    ));
}

/// One request written by hand, and the response's status line and
/// headers.
fn exchange(stream: &mut TcpStream, request: &str) -> String {
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    let mut byte = [0u8; 1];
    while !response.ends_with(b"\r\n\r\n") {
        assert_eq!(stream.read(&mut byte).unwrap(), 1, "the server answers");
        response.push(byte[0]);
    }
    let text = String::from_utf8(response).unwrap();
    if let Some(length) = text
        .lines()
        .find_map(|line| line.strip_prefix("Content-Length: "))
    {
        let mut body = vec![0u8; length.trim().parse().unwrap()];
        stream.read_exact(&mut body).unwrap();
        return text + &String::from_utf8(body).unwrap();
    }
    text
}

#[test]
fn requests_are_answered_as_rtsp_says() {
    let (_runner, handle, broadcast) = server(false);
    let mut stream = TcpStream::connect(handle.local_addr()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let url = handle.url("cam");

    let options = exchange(
        &mut stream,
        &format!("OPTIONS {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n"),
    );
    assert!(
        options.starts_with("RTSP/1.0 200 OK\r\nCSeq: 1\r\n"),
        "{options}"
    );
    assert!(options.contains("DESCRIBE"));

    let missing = exchange(
        &mut stream,
        &format!(
            "DESCRIBE {} RTSP/1.0\r\nCSeq: 2\r\n\r\n",
            handle.url("nowhere")
        ),
    );
    assert!(missing.starts_with("RTSP/1.0 404"), "{missing}");
    let unpublished = exchange(
        &mut stream,
        &format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 3\r\n\r\n"),
    );
    assert!(
        unpublished.starts_with("RTSP/1.0 404"),
        "nothing published yet: {unpublished}"
    );

    let video = TrackFormat::new(
        {
            let mut parameters = ffmpeg::codec::Parameters::new();
            // SAFETY: parameters this test owns alone; plain fields.
            unsafe {
                (*parameters.as_mut_ptr()).codec_type = ffmpeg::media::Type::Video.into();
                (*parameters.as_mut_ptr()).codec_id = ffmpeg::codec::Id::H264.into();
            }
            parameters
        },
        ffmpeg::Rational(1, 90_000),
    );
    let _sinks = broadcast.publish(&[video]).unwrap();
    let described = exchange(
        &mut stream,
        &format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 4\r\n\r\n"),
    );
    assert!(described.starts_with("RTSP/1.0 200"), "{described}");
    assert!(described.contains(&format!("Content-Base: {url}/")));
    assert!(described.contains("m=video 0 RTP/AVP 96"));

    let play = exchange(
        &mut stream,
        &format!("PLAY {url} RTSP/1.0\r\nCSeq: 5\r\nSession: nope\r\n\r\n"),
    );
    assert!(play.starts_with("RTSP/1.0 454"), "{play}");
    let no_transport = exchange(
        &mut stream,
        &format!("SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 6\r\n\r\n"),
    );
    assert!(no_transport.starts_with("RTSP/1.0 461"), "{no_transport}");
    let udp = exchange(
        &mut stream,
        &format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 7\r\nTransport: RTP/AVP;unicast;client_port=5000-5001\r\n\r\n"
        ),
    );
    assert!(
        udp.starts_with("RTSP/1.0 461"),
        "this server has no UDP: {udp}"
    );
    let setup = exchange(
        &mut stream,
        &format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 8\r\nTransport: RTP/AVP/TCP;unicast\r\n\r\n"
        ),
    );
    assert!(setup.starts_with("RTSP/1.0 200"), "{setup}");
    assert!(
        setup.contains("Transport: RTP/AVP/TCP;unicast;interleaved=0-1"),
        "{setup}"
    );
    let session = setup
        .lines()
        .find_map(|line| line.strip_prefix("Session: "))
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_owned();
    let play = exchange(
        &mut stream,
        &format!("PLAY {url} RTSP/1.0\r\nCSeq: 9\r\nSession: {session}\r\n\r\n"),
    );
    assert!(play.starts_with("RTSP/1.0 200"), "{play}");
    assert_eq!(handle.players(), 1);
    let teardown = exchange(
        &mut stream,
        &format!("TEARDOWN {url} RTSP/1.0\r\nCSeq: 10\r\nSession: {session}\r\n\r\n"),
    );
    assert!(teardown.starts_with("RTSP/1.0 200"), "{teardown}");
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty(), "the connection closes after a teardown");
    assert_eq!(handle.players(), 0);
}

#[test]
fn what_is_not_rtsp_closes_the_connection() {
    let (_runner, handle, _broadcast) = server(false);
    let mut stream = TcpStream::connect(handle.local_addr()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let answer = exchange(&mut stream, "GET / HTTP/1.1\r\nHost: x\r\n\r\n");
    assert!(answer.starts_with("RTSP/1.0 400"), "{answer}");
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty());
}

/// A server serving `live` for a publisher to push into, besides `cam`.
fn server_for_publishers(udp: bool) -> (Arc<DriverRunner>, RtspServerHandle, Broadcast) {
    let (runner, handle, _cam) = server(udp);
    let live = Broadcast::new("live", BroadcastOptions::default());
    handle
        .add_path("live", &live, RtspPathMode::Record)
        .unwrap();
    (runner, handle, live)
}

/// Pushes the fixture to `url` at its own speed, over and over, as an
/// encoder would — through FFmpeg's own RTSP muxer.
fn publish(url: &str, path: &str, transport: RtspTransport) -> crate::Result<Arc<Pipeline>> {
    use crate::elements::RtspMuxer;
    let (demuxer, streams) = FileDemuxer::open("encoder", path)?;
    demuxer.looping_handle().set_looping(true);
    let mut muxer = RtspMuxer::create(
        url,
        RtspOptions {
            transport,
            timeout: Duration::from_secs(5),
        },
    )?;
    let tracks = streams
        .iter()
        .map(|stream| muxer.add_stream(format!("track-{}", stream.index), stream))
        .collect::<crate::Result<Vec<_>>>()?;
    let mut sinks = muxer.open()?;
    let sinks = tracks
        .into_iter()
        .map(|track| sinks.take(track))
        .collect::<crate::Result<Vec<_>>>()?;
    let (pipeline, ()) = Pipeline::new("publisher", demuxer, |source, ctx| {
        for (stream, sink) in streams.iter().zip(sinks) {
            let branch = ctx
                .branch()
                .queue(format!("published-{}", stream.index), 64)
                .pipe(Pacer::new(format!("pacer-{}", stream.index)))
                .to(sink)?;
            ctx.attach(source, stream.index, branch)?;
        }
        Ok(())
    })?;
    pipeline.run()?;
    Ok(pipeline)
}

/// Waits until `broadcast` has been published to.
fn wait_published(broadcast: &Broadcast) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !broadcast.is_live() {
        assert!(
            Instant::now() < deadline,
            "the publisher's stream never reached the broadcast"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn a_publisher_is_played_back(transport: RtspTransport) {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, live) = server_for_publishers(true);
    let publisher =
        publish(&handle.url("live"), &path, transport).expect("the server takes a publisher");
    wait_published(&live);

    // What the description left out is read from the first keyframe: the
    // picture's size, which a player's description and a file both need.
    let (_, fixture) = FileDemuxer::open("probe", &path).unwrap();
    let fixture_video = fixture
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Video)
        .unwrap();
    let tracks = live.tracks().unwrap();
    let video = tracks
        .iter()
        .find(|s| s.kind == ffmpeg::media::Type::Video)
        .unwrap();
    // SAFETY: reading plain fields of parameters held here.
    let (width, height, want) = unsafe {
        let got = &*video.parameters.as_ptr();
        let want = &*fixture_video.parameters.as_ptr();
        (got.width, got.height, (want.width, want.height))
    };
    assert_eq!((width, height), want);

    let player = play(&handle.url("live"), transport);
    check_played(&player);
    player.pipeline.stop();

    publisher.stop();
    drop(publisher);
    let deadline = Instant::now() + Duration::from_secs(10);
    while live.is_live() {
        assert!(
            Instant::now() < deadline,
            "the publisher left and its run did not end"
        );
        thread::sleep(Duration::from_millis(20));
    }
    runner.stop();
}

#[test]
fn a_publisher_over_tcp_is_played_back() {
    a_publisher_is_played_back(RtspTransport::Tcp);
}

#[test]
fn a_publisher_over_udp_is_played_back() {
    a_publisher_is_played_back(RtspTransport::Udp);
}

#[test]
fn a_path_holds_one_publisher_and_a_path_for_players_none() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, live) = server_for_publishers(false);
    let first = publish(&handle.url("live"), &path, RtspTransport::Tcp).unwrap();
    wait_published(&live);
    assert!(
        publish(&handle.url("live"), &path, RtspTransport::Tcp).is_err(),
        "a second publisher is refused"
    );
    assert!(
        publish(&handle.url("cam"), &path, RtspTransport::Tcp).is_err(),
        "a path for players alone takes no publisher"
    );
    assert!(
        publish(&handle.url("nowhere"), &path, RtspTransport::Tcp).is_err(),
        "nor does a path not served"
    );
    // The first goes on regardless.
    let player = play(&handle.url("live"), RtspTransport::Tcp);
    player.wait_for(10);
    player.pipeline.stop();
    first.stop();
    runner.stop();
}

/// What a UDP player joins with — everything since the newest keyframe — is
/// sent at twice the stream's speed, not at once: UDP has no way to say slow
/// down, and the burst overflowed players' sockets, losing half of what they
/// were sent (docs/benchmarks/streaming/findings.md).
#[test]
fn a_udp_player_catches_up_at_twice_the_streams_speed() {
    use std::net::UdpSocket;

    let (_runner, handle, broadcast) = server(true);
    let video = TrackFormat::new(
        {
            let mut parameters = ffmpeg::codec::Parameters::new();
            // SAFETY: parameters this test owns alone; plain fields.
            unsafe {
                (*parameters.as_mut_ptr()).codec_type = ffmpeg::media::Type::Video.into();
                (*parameters.as_mut_ptr()).codec_id = ffmpeg::codec::Id::H264.into();
                (*parameters.as_mut_ptr()).width = 320;
                (*parameters.as_mut_ptr()).height = 240;
            }
            parameters
        },
        ffmpeg::Rational(1, 90_000),
    );
    let mut sinks = broadcast.publish(&[video]).unwrap();
    // Four seconds of pictures from one keyframe, written at once: all of it
    // is what a player joining now catches up on.
    let pictures = 100;
    for n in 0..pictures {
        let mut data = vec![0, 0, 0, 1, if n == 0 { 0x65 } else { 0x41 }];
        data.resize(2_000, 0xAB);
        let mut packet = ffmpeg::Packet::copy(&data);
        packet.set_pts(Some(n * 3_600));
        packet.set_dts(Some(n * 3_600));
        packet.set_time_base(ffmpeg::Rational(1, 90_000));
        if n == 0 {
            packet.set_flags(ffmpeg::packet::Flags::KEY);
        }
        crate::element::RawSink::consume(&mut sinks[0], MediaBuffer::packet(packet)).unwrap();
    }

    let rtp = UdpSocket::bind("127.0.0.1:0").unwrap();
    rtp.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let port = rtp.local_addr().unwrap().port();
    let mut stream = TcpStream::connect(handle.local_addr()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let url = handle.url("cam");
    let setup = exchange(
        &mut stream,
        &format!(
            "SETUP {url}/streamid=0 RTSP/1.0\r\nCSeq: 1\r\nTransport: RTP/AVP;unicast;client_port={port}-{}\r\n\r\n",
            port + 1
        ),
    );
    assert!(setup.starts_with("RTSP/1.0 200"), "{setup}");
    let session = setup
        .lines()
        .find_map(|line| line.strip_prefix("Session: "))
        .unwrap()
        .split(';')
        .next()
        .unwrap()
        .to_owned();
    let play = exchange(
        &mut stream,
        &format!("PLAY {url} RTSP/1.0\r\nCSeq: 2\r\nSession: {session}\r\n\r\n"),
    );
    assert!(play.starts_with("RTSP/1.0 200"), "{play}");

    let mut buffer = [0u8; 2048];
    let mut arrivals = Vec::new();
    let mut last_picture = None;
    while arrivals.len() < pictures as usize {
        let length = rtp.recv(&mut buffer).expect("every picture arrives");
        // The last packet of each picture carries the marker.
        if buffer[1] & 0x80 != 0 {
            arrivals.push(Instant::now());
            last_picture = Some(u32::from_be_bytes([
                buffer[4], buffer[5], buffer[6], buffer[7],
            ]));
        }
        assert!(length > 12);
    }
    assert!(last_picture.is_some());
    let took = arrivals[arrivals.len() - 1] - arrivals[0];
    // Four seconds of pictures at twice their speed: two seconds, less the
    // last picture's own 40 ms.
    assert!(
        took >= Duration::from_millis(1_700),
        "four seconds of pictures arrived in {took:?}, at once rather than paced"
    );
    assert!(
        took < Duration::from_millis(3_000),
        "paced slower than twice: {took:?}"
    );
}
