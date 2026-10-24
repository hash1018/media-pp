//! The server against this crate's own RTSP client — `RtspSource::open` —
//! FFmpeg's RTSP muxer, and requests written by hand, on this machine's
//! loopback alone.

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
    AccessRequest, Authorize, FileDemuxer, Pacer, RtspOptions, RtspSource, RtspTransport,
    StreamInfo,
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
    let broadcast = serve(&handle, "cam");
    let runner = DriverRunner::new(server);
    runner.run().unwrap();
    (runner, handle, broadcast)
}

/// Serves `name` on `handle`'s server for the rest of the test, and
/// answers the broadcast its players read — what a path's muxers write
/// into, and what a test writes into or looks at directly.
fn serve(handle: &RtspServerHandle, name: &str) -> Broadcast {
    let path = handle.path(name).unwrap();
    let broadcast = path.broadcast().clone();
    // Dropped, the path would end its players: kept as long as the test.
    std::mem::forget(path);
    broadcast
}

/// Plays the fixture into `broadcast` at its own speed, over and over, as a
/// camera would.
fn feed(broadcast: &Broadcast, path: &str) -> Arc<Pipeline> {
    let (demuxer, streams) = FileDemuxer::open("camera", path).unwrap();
    demuxer.looping_handle().set_looping(true);
    let tracks: Vec<StreamInfo> = streams.iter().map(StreamInfo::from).collect();
    let sinks = broadcast.publish(&tracks).unwrap();
    let (pipeline, ()) = Pipeline::new("camera", demuxer, |source, ctx| {
        for (stream, sink) in streams.iter().zip(sinks) {
            let branch = ctx
                .branch()
                .queue(format!("packets-{}", stream.index()), 64)
                .pipe(Pacer::new(format!("pacer-{}", stream.index())))
                .to(sink)?;
            ctx.attach(source, stream.index(), branch)?;
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
                .queue(format!("received-{}", stream.index()), 256)
                .to(sink)?;
            ctx.attach(source, stream.index(), branch)?;
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
        self.streams
            .iter()
            .position(|s| s.kind() == medium)
            .unwrap()
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
        player.streams[video].parameters().id(),
        ffmpeg::codec::Id::H264
    );
    assert_eq!(
        player.streams[audio].parameters().id(),
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
        |stream: usize, pts: i64| pts as f64 * f64::from(player.streams[stream].time_base());
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

/// Each player is a session of its own, told apart by its path: how many
/// play one path is read off them, with how each is sent and what it has
/// been sent. One that leaves is no longer among them.
#[test]
fn each_session_says_its_path_and_what_it_was_sent() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(true);
    let door = serve(&handle, "door");
    let camera = feed(&broadcast, &path);
    let _door_camera = feed(&door, &path);
    let players = [
        play(&handle.url("cam"), RtspTransport::Tcp),
        play(&handle.url("cam"), RtspTransport::Tcp),
        play(&handle.url("door"), RtspTransport::Udp),
    ];
    for player in &players {
        check_played(player);
    }

    let stats = handle.session_stats();
    let on = |path: &str| stats.iter().filter(|session| session.path == path).count();
    assert_eq!((on("cam"), on("door")), (2, 1));
    for session in &stats {
        assert_eq!(session.role, Access::Play);
        assert_eq!(session.udp, session.path == "door", "{session:?}");
        assert!(session.remote.ip().is_loopback());
        assert!(session.bytes_sent > 0, "{session:?}");
        assert_eq!(session.bytes_received, 0);
    }
    assert_eq!(handle.players(), 3);

    let [first, ..] = &players;
    first.pipeline.stop();
    let deadline = Instant::now() + Duration::from_secs(5);
    while handle.session_stats().len() > 2 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(handle.session_stats().len(), 2, "one that leaves is let go");
    for player in &players {
        player.pipeline.stop();
    }
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
    // A packet with no time is passed over, wherever a client gives one.
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
    let unit = f64::from(player.streams[video].time_base());
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
fn paths_are_checked_as_they_are_taken() {
    let (_runner, handle, _broadcast) = server(false);
    assert!(matches!(
        handle.path("cam"),
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
            matches!(handle.path(bad), Err(RtspServerError::InvalidPath(_))),
            "{bad:?}"
        );
    }
    let door = handle.path("site/door-1").unwrap();
    assert_eq!(door.path(), "site/door-1");
    drop(door);
    handle
        .path("site/door-1")
        .expect("served again once let go of");
}

/// What a path's muxer writes is what its players play, a run each muxer
/// opened, which they go on across; the path let go of ends them, and
/// nobody new plays it.
#[test]
fn a_path_plays_what_its_muxers_write_and_ends_its_players_as_it_goes() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, _cam) = server(false);
    let door = handle.path("door").unwrap();
    let write = |round: &str| {
        let (demuxer, streams) = FileDemuxer::open(round, &path).unwrap();
        demuxer.looping_handle().set_looping(true);
        let mut muxer = door.muxer();
        let tracks: Vec<_> = streams
            .iter()
            .map(|stream| {
                muxer
                    .add_stream(format!("track-{}", stream.index()), stream)
                    .unwrap()
            })
            .collect();
        let mut sinks = muxer.open().unwrap();
        let (pipeline, ()) = Pipeline::new(round, demuxer, |source, ctx| {
            for (stream, track) in streams.iter().zip(tracks) {
                let branch = ctx
                    .branch()
                    .queue(format!("packets-{}", stream.index()), 64)
                    .pipe(Pacer::new(format!("pacer-{}", stream.index())))
                    .to(sinks.take(track)?)?;
                ctx.attach(source, stream.index(), branch)?;
            }
            Ok(())
        })
        .unwrap();
        pipeline.run().unwrap();
        pipeline
    };
    let first = write("first");
    assert_eq!(door.players(), 0);
    let player = play(&handle.url("door"), RtspTransport::Tcp);
    player.wait_for(10);
    assert_eq!(door.players(), 1, "the player reads the path");
    // The input drops and comes back: another run, which the player goes on
    // across.
    first.stop();
    drop(first);
    let second = write("second");
    let had = player.packets(0).len();
    let deadline = Instant::now() + Duration::from_secs(10);
    while player.packets(0).len() < had + 10 {
        assert!(
            Instant::now() < deadline,
            "the player went on across the runs"
        );
        thread::sleep(Duration::from_millis(20));
    }

    drop(door);
    let deadline = Instant::now() + Duration::from_secs(10);
    while handle
        .session_stats()
        .iter()
        .any(|session| session.role == Access::Play)
    {
        assert!(
            Instant::now() < deadline,
            "the path let go of ends its players"
        );
        thread::sleep(Duration::from_millis(20));
    }
    assert!(
        RtspSource::open(
            "late",
            handle.url("door"),
            RtspOptions {
                transport: RtspTransport::Tcp,
                timeout: Duration::from_secs(2),
            }
        )
        .is_err(),
        "nobody new plays it"
    );
    player.pipeline.stop();
    second.stop();
    runner.stop();
}

/// A path whose run has ended still describes its last one: a player
/// plays it, is counted — which what waits for its players hears — and
/// goes on with the next run its muxer opens.
#[test]
fn a_path_between_runs_is_played_and_says_its_players_change() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, _cam) = server(false);
    let door = Arc::new(handle.path("door").unwrap());
    let write = |round: &str| {
        let (demuxer, streams) = FileDemuxer::open(round, &path).unwrap();
        demuxer.looping_handle().set_looping(true);
        let mut muxer = door.muxer();
        let tracks: Vec<_> = streams
            .iter()
            .map(|stream| {
                muxer
                    .add_stream(format!("track-{}", stream.index()), stream)
                    .unwrap()
            })
            .collect();
        let mut sinks = muxer.open().unwrap();
        let (pipeline, ()) = Pipeline::new(round, demuxer, |source, ctx| {
            for (stream, track) in streams.iter().zip(tracks) {
                let branch = ctx
                    .branch()
                    .queue(format!("packets-{}", stream.index()), 64)
                    .pipe(Pacer::new(format!("pacer-{}", stream.index())))
                    .to(sinks.take(track)?)?;
                ctx.attach(source, stream.index(), branch)?;
            }
            Ok(())
        })
        .unwrap();
        pipeline.run().unwrap();
        pipeline
    };
    // A run described the path, and ended.
    let first = write("first");
    first.stop();
    drop(first);

    let waiting = {
        let door = Arc::clone(&door);
        thread::spawn(move || crate::test_support::block_on(door.players_change(0)))
    };
    let player = play(&handle.url("door"), RtspTransport::Tcp);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !waiting.is_finished() {
        assert!(
            Instant::now() < deadline,
            "what waits for the players hears of the one that came"
        );
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(waiting.join().unwrap(), 1);

    let second = write("second");
    player.wait_for(10);
    assert_eq!(door.players(), 1);
    player.pipeline.stop();
    let deadline = Instant::now() + Duration::from_secs(10);
    while door.players() > 0 {
        assert!(
            Instant::now() < deadline,
            "the player that went is not counted"
        );
        thread::sleep(Duration::from_millis(20));
    }
    second.stop();
    runner.stop();
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

    let video = StreamInfo::new(
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

/// What a relay of a path's publishers did: the streams each was handed
/// out with, as the server stopped.
type Relayed = thread::JoinHandle<Vec<Vec<StreamInfo>>>;

/// A server taking publishers on `live`, each relayed into the broadcast
/// its players play under the same name, besides `cam`.
fn server_for_publishers(udp: bool) -> (Arc<DriverRunner>, RtspServerHandle, Broadcast, Relayed) {
    let (runner, handle, _cam) = server(udp);
    let live = serve(&handle, "live");
    let relayed = crate::test_support::relay(handle.take_publishers("live").unwrap(), live.clone());
    (runner, handle, live, relayed)
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
        .map(|stream| muxer.add_stream(format!("track-{}", stream.index()), stream))
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
                .queue(format!("published-{}", stream.index()), 64)
                .pipe(Pacer::new(format!("pacer-{}", stream.index())))
                .to(sink)?;
            ctx.attach(source, stream.index(), branch)?;
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
    let (runner, handle, live, relayed) = server_for_publishers(true);
    let publisher =
        publish(&handle.url("live"), &path, transport).expect("the server takes a publisher");
    wait_published(&live);
    let publishing: Vec<_> = handle
        .session_stats()
        .into_iter()
        .filter(|session| session.role == Access::Publish)
        .collect();
    assert_eq!(publishing.len(), 1, "{publishing:?}");
    assert_eq!(publishing[0].path, "live");
    assert_eq!(publishing[0].udp, transport == RtspTransport::Udp);
    assert!(publishing[0].bytes_received > 0);

    // What the description left out is read from the first keyframe: the
    // picture's size, which a player's description and a file both need.
    let (_, fixture) = FileDemuxer::open("probe", &path).unwrap();
    let fixture_video = fixture
        .iter()
        .find(|s| s.kind() == ffmpeg::media::Type::Video)
        .unwrap();
    let tracks = live.tracks().unwrap();
    let video = tracks
        .iter()
        .find(|s| s.kind() == ffmpeg::media::Type::Video)
        .unwrap();
    // SAFETY: reading plain fields of parameters held here.
    let (width, height, want) = unsafe {
        let got = &*video.parameters().as_ptr();
        let want = &*fixture_video.parameters().as_ptr();
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
    let runs = relayed.join().unwrap();
    assert_eq!(runs.len(), 1);
    let indices: Vec<usize> = runs[0].iter().map(StreamInfo::index).collect();
    assert_eq!(
        indices,
        (0..runs[0].len()).collect::<Vec<_>>(),
        "an output each, in order"
    );
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
    let (runner, handle, live, _relayed) = server_for_publishers(false);
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

/// A player of `cam` over UDP, set up and playing by hand: the socket its
/// RTP arrives on, and its RTSP connection with the socket its RTCP does,
/// which it holds as long as its session lasts.
///
/// Both ports are held: with the RTCP one left free, another test running
/// beside this one could bind it and be sent this session's reports — and
/// a test of this one's could be sent another's.
fn play_over_udp(
    handle: &RtspServerHandle,
) -> (std::net::UdpSocket, (TcpStream, std::net::UdpSocket)) {
    use std::net::UdpSocket;

    let (rtp, rtcp) = loop {
        let rtp = UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = rtp.local_addr().unwrap().port();
        if let Some(Ok(rtcp)) = port
            .checked_add(1)
            .map(|next| UdpSocket::bind(("127.0.0.1", next)))
        {
            break (rtp, rtcp);
        }
    };
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
    (rtp, (stream, rtcp))
}

/// What a UDP player joins with — everything since the newest keyframe — is
/// sent at twice the stream's speed, not at once: UDP has no way to say slow
/// down, and the burst overflowed players' sockets, losing half of what they
/// were sent (docs/benchmarks/streaming/findings.md).
#[test]
fn a_udp_player_catches_up_at_twice_the_streams_speed() {
    let (_runner, handle, broadcast) = server(true);
    let video = StreamInfo::new(
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
    // Another reader, for which what is written is kept.
    let _watching = broadcast.subscribe();
    // Just under a second of pictures from one keyframe, written at once —
    // as much as a UDP player joins at a keyframe with (further behind, it
    // waits for the next): all of it is what a player joining now catches
    // up on.
    let pictures = 24;
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

    let (rtp, _connection) = play_over_udp(&handle);

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
    // 0.92 s of pictures at twice their speed: 0.46 s, less the last
    // picture's own 20 ms.
    assert!(
        took >= Duration::from_millis(350),
        "{pictures} pictures arrived in {took:?}, at once rather than paced"
    );
    assert!(
        took < Duration::from_millis(900),
        "paced slower than twice: {took:?}"
    );
}

/// What an `Authorize` was asked, in order.
type Asked = Arc<Mutex<Vec<AccessRequest>>>;

/// A server whose `Authorize` lets `viewer:pw` play `cam` and `camera:pw`
/// publish into `live`, and nobody anything else; what it was asked, kept.
fn server_asking() -> (
    Arc<DriverRunner>,
    RtspServerHandle,
    Broadcast,
    Broadcast,
    Asked,
) {
    use crate::elements::{Access, Credentials};
    let asked = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&asked);
    let authorize = Authorize::new(move |request: &AccessRequest| {
        kept.lock().unwrap().push(request.clone());
        let user = match &request.credentials {
            Credentials::Basic { user, password } if password == "pw" => user.as_str(),
            Credentials::Digest(digest) if digest.verifies("pw") => digest.user(),
            Credentials::Bearer(token) if *token == long_token() => "viewer",
            _ => return false,
        };
        matches!(
            (request.access, request.path.as_str(), user),
            (Access::Play, "cam", "viewer") | (Access::Publish, "live", "camera")
        )
    });
    let (server, handle) = RtspServer::bind(
        "server",
        "127.0.0.1:0",
        RtspServerOptions {
            udp: false,
            authorize: Some(authorize),
            ..RtspServerOptions::default()
        },
    )
    .expect("a loopback port is free");
    let cam = serve(&handle, "cam");
    let live = serve(&handle, "live");
    // Left to run until the server stops.
    drop(crate::test_support::relay(
        handle.take_publishers("live").unwrap(),
        live.clone(),
    ));
    let runner = DriverRunner::new(server);
    runner.run().unwrap();
    (runner, handle, cam, live, asked)
}

/// A token as long as a signed one is: longer than some players keep of a
/// URL's user name.
fn long_token() -> String {
    format!("{}.signed", "a".repeat(200))
}

fn with_user(url: &str, user: &str) -> String {
    url.replacen("rtsp://", &format!("rtsp://{user}:pw@"), 1)
}

/// Refused, a request is answered 401 asking for a user and password —
/// before the server says whether the path is there at all — and the
/// credentials a request comes with are what decide.
#[test]
fn a_server_that_authorizes_asks_for_credentials_first() {
    let (runner, handle, _cam, _live, asked) = server_asking();
    let mut stream = TcpStream::connect(handle.local_addr()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    for (cseq, path) in [(1, "cam"), (2, "nowhere")] {
        let refused = exchange(
            &mut stream,
            &format!(
                "DESCRIBE {} RTSP/1.0\r\nCSeq: {cseq}\r\n\r\n",
                handle.url(path)
            ),
        );
        assert!(
            refused.starts_with("RTSP/1.0 401 Unauthorized\r\n"),
            "{refused}"
        );
        assert!(
            refused.contains("WWW-Authenticate: Basic realm=\"media-pp\""),
            "{refused}"
        );
    }
    // viewer:pw, which this server lets play `cam`; nothing is published
    // there yet, so it is past the check and not found.
    let allowed = exchange(
        &mut stream,
        &format!(
            "DESCRIBE {} RTSP/1.0\r\nCSeq: 3\r\nAuthorization: Basic dmlld2VyOnB3\r\n\r\n",
            handle.url("cam")
        ),
    );
    assert!(
        allowed.starts_with("RTSP/1.0 404 Not Found\r\n"),
        "{allowed}"
    );
    let asked = asked.lock().unwrap();
    assert_eq!(asked.len(), 3);
    assert_eq!(asked[2].path, "cam");
    assert_eq!(asked[2].peer, stream.local_addr().unwrap());
    drop(asked);
    runner.stop();
}

/// This crate's RTSP client and FFmpeg's muxer, given
/// `rtsp://user:password@…`, answer
/// the 401 with them, and play and publish as anyone would; without, or
/// as the wrong user, they are refused.
#[test]
fn players_and_publishers_get_in_with_their_credentials() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam, live, asked) = server_asking();
    let _camera = feed(&cam, &path);
    wait_published(&cam);

    let options = || RtspOptions {
        transport: RtspTransport::Tcp,
        timeout: Duration::from_secs(5),
    };
    assert!(RtspSource::open("player", handle.url("cam"), options()).is_err());
    assert!(
        RtspSource::open("player", with_user(&handle.url("cam"), "camera"), options()).is_err(),
        "camera:pw may publish, not play"
    );
    let player = play(&with_user(&handle.url("cam"), "viewer"), RtspTransport::Tcp);
    player.wait_for(10);
    check_played(&player);
    player.pipeline.stop();

    assert!(publish(&handle.url("live"), &path, RtspTransport::Tcp).is_err());
    let publisher = publish(
        &with_user(&handle.url("live"), "camera"),
        &path,
        RtspTransport::Tcp,
    )
    .unwrap();
    wait_published(&live);
    publisher.stop();
    runner.stop();
    // Offered Digest beside Basic, FFmpeg hashed its password every time.
    let asked = asked.lock().unwrap();
    let presented: Vec<_> = asked
        .iter()
        .filter(|request| !matches!(request.credentials, crate::elements::Credentials::None))
        .collect();
    assert!(!presented.is_empty());
    assert!(
        presented
            .iter()
            .all(|request| matches!(request.credentials, crate::elements::Credentials::Digest(_))),
        "{presented:?}"
    );
}

/// A Digest response counts only against the challenge this connection
/// was given: made right, it is let in; made for another connection's
/// challenge, it presents nothing and is asked again.
#[test]
fn a_digest_response_answers_this_connections_challenge_alone() {
    let (runner, handle, _cam, _live, _asked) = server_asking();
    let connect = || {
        let stream = TcpStream::connect(handle.local_addr()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
    };
    let url = handle.url("cam");
    let challenge = |stream: &mut TcpStream| {
        let refused = exchange(
            stream,
            &format!("DESCRIBE {url} RTSP/1.0\r\nCSeq: 1\r\n\r\n"),
        );
        refused
            .split("nonce=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .unwrap_or_else(|| panic!("a Digest challenge: {refused}"))
            .to_owned()
    };
    let answer = |nonce: &str| {
        let ha1 = crate::elements::rtsp::digest::md5_hex(
            format!("viewer:{}:pw", super::digest::REALM).as_bytes(),
        );
        let ha2 = crate::elements::rtsp::digest::md5_hex(format!("DESCRIBE:{url}").as_bytes());
        let response =
            crate::elements::rtsp::digest::md5_hex(format!("{ha1}:{nonce}:{ha2}").as_bytes());
        format!(
            "DESCRIBE {url} RTSP/1.0\r\nCSeq: 2\r\nAuthorization: Digest username=\"viewer\", realm=\"{}\", nonce=\"{nonce}\", uri=\"{url}\", response=\"{response}\"\r\n\r\n",
            super::digest::REALM
        )
    };

    let mut other = connect();
    let others = challenge(&mut other);
    let mut stream = connect();
    let own = challenge(&mut stream);
    assert_ne!(own, others);
    let refused = exchange(&mut stream, &answer(&others));
    assert!(refused.starts_with("RTSP/1.0 401"), "{refused}");
    // Nothing is published at `cam`: past the check, it is not found.
    let allowed = exchange(&mut stream, &answer(&own));
    assert!(allowed.starts_with("RTSP/1.0 404"), "{allowed}");
    runner.stop();
}

/// A UDP player joins at the newest keyframe only where it is at most a
/// second behind the newest picture; further, it waits for the next one
/// rather than be sent the whole group at twice the stream's speed, which
/// it has no way to ask for again what it loses of.
#[test]
fn a_udp_player_far_behind_the_newest_keyframe_starts_at_the_next() {
    let (_runner, handle, broadcast) = server(true);
    let video = StreamInfo::new(
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
    let mut write = |n: i64, key: bool| {
        let mut data = vec![0, 0, 0, 1, if key { 0x65 } else { 0x41 }];
        data.resize(2_000, 0xAB);
        let mut packet = ffmpeg::Packet::copy(&data);
        packet.set_pts(Some(n * 3_600));
        packet.set_dts(Some(n * 3_600));
        packet.set_time_base(ffmpeg::Rational(1, 90_000));
        if key {
            packet.set_flags(ffmpeg::packet::Flags::KEY);
        }
        crate::element::RawSink::consume(&mut sinks[0], MediaBuffer::packet(packet)).unwrap();
    };
    // Two seconds from one keyframe: the newest picture is 1.96 s past it.
    for n in 0..50 {
        write(n, n == 0);
    }
    let (rtp, _connection) = play_over_udp(&handle);
    let mut buffer = [0u8; 2048];
    rtp.set_read_timeout(Some(Duration::from_millis(500)))
        .unwrap();
    assert!(
        rtp.recv(&mut buffer).is_err(),
        "nothing of the old group of pictures is sent"
    );

    // The next group begins: the player starts at its keyframe.
    for n in 50..56 {
        write(n, n == 50);
    }
    rtp.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut pictures = 0;
    while pictures < 6 {
        rtp.recv(&mut buffer).expect("the next group arrives");
        if buffer[1] & 0x80 != 0 {
            pictures += 1;
        }
    }
    rtp.set_read_timeout(Some(Duration::from_millis(300)))
        .unwrap();
    assert!(
        rtp.recv(&mut buffer).is_err(),
        "the six pictures of the next group, and nothing before them"
    );
}

/// A camera that changes its picture size partway: the server carries it on
/// as it comes, and a software decoder after the player's `RtspSource` goes
/// on with the new size.
#[test]
fn a_camera_that_changes_its_picture_size_decodes_on_through_the_server() {
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::elements::{AppSink, SwDecoder};

    let (parameters, small) = crate::test_support::h264_configured((320, 240));
    let (_, large) = crate::test_support::h264_configured((640, 360));
    let (runner, handle, broadcast) = server(false);
    let sink = broadcast
        .publish(&[StreamInfo::new(parameters, ffmpeg::Rational::new(1, 30))])
        .unwrap()
        .remove(0);
    let (switch, stop) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let camera = crate::test_support::changing_camera(
        sink,
        small,
        large,
        Arc::clone(&switch),
        Arc::clone(&stop),
    );

    let url = format!("rtsp://{}/cam", handle.local_addr());
    let (source, streams) = RtspSource::open(
        "player",
        &url,
        RtspOptions {
            transport: RtspTransport::Tcp,
            timeout: Duration::from_secs(5),
        },
    )
    .unwrap();
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&sizes);
    let decoder = SwDecoder::new("decoder", streams[0].parameters().clone()).unwrap();
    let (pipeline, ()) = Pipeline::new("player", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(decoder)
            .to(AppSink::new("screen", move |buf| {
                if let MediaBuffer::Video(frame) = &buf {
                    seen.lock().unwrap().push((frame.width(), frame.height()));
                }
                Ok(())
            }))?;
        ctx.attach(source, streams[0].index(), branch)?;
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();

    let wait = |size: (u32, u32)| {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !sizes.lock().unwrap().contains(&size) {
            assert!(Instant::now() < deadline, "no {size:?} picture decoded");
            thread::sleep(Duration::from_millis(20));
        }
    };
    wait((320, 240));
    switch.store(true, Ordering::Release);
    wait((640, 360));
    pipeline.stop();
    stop.store(true, Ordering::Release);
    camera.join().unwrap();
    runner.stop();

    let sizes = sizes.lock().unwrap();
    let first_large = sizes.iter().position(|&size| size == (640, 360)).unwrap();
    assert!(
        sizes[first_large..].iter().all(|&size| size == (640, 360)),
        "the decoder goes on at the new size: {sizes:?}"
    );
}

/// What FFmpeg's client probes of a stream with B-frames says so: what a
/// relay serving it to WebRTC players, which cannot take them, goes by.
/// A stream with B-frames, played: probed as one that reorders, and every
/// picture handed on with a decoding time no later than its showing and
/// later than the last — what RTP does not carry, read from the stream.
/// For each encoder of B-frames this FFmpeg has: x264, kvazaar, x265.
#[test]
fn a_player_is_told_a_stream_reorders_its_pictures() {
    use crate::elements::VideoCodec;
    let mut played = 0;
    for codec in [VideoCodec::H264, VideoCodec::Kvazaar, VideoCodec::H265] {
        played += usize::from(plays_reordered(codec));
    }
    if played == 0 {
        eprintln!("skipping: no encoder of B-frames in this FFmpeg");
    }
}

/// Whether `codec`'s encoder opened, and its stream played as it should.
fn plays_reordered(codec: crate::elements::VideoCodec) -> bool {
    use crate::elements::{SwEncoder, SwEncoderOptions, TestVideoOptions, TestVideoSource};
    let Ok(encoder) = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec,
            width: 320,
            height: 240,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(25, 1),
            bit_rate: 300_000,
            gop_size: 25,
            max_b_frames: Some(2),
        },
    ) else {
        return false;
    };
    let (runner, handle, broadcast) = server(false);
    let sink = broadcast
        .publish(&[StreamInfo::from(&encoder)])
        .unwrap()
        .remove(0);
    let pictures = TestVideoSource::new(
        "pictures",
        TestVideoOptions {
            width: 320,
            height: 240,
            frame_rate: ffmpeg::Rational::new(25, 1),
        },
    );
    let (camera, ()) = Pipeline::new("camera", pictures, |source, ctx| {
        let branch = ctx.branch().queue("encode", 4).pipe(encoder).to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .unwrap();
    camera.run().unwrap();

    let player = play(&handle.url("cam"), RtspTransport::Tcp);
    assert!(
        player.streams[0].reorders(),
        "{codec:?}: probed as reordering"
    );
    player.wait_for(60);
    player.pipeline.stop();
    camera.stop();
    runner.stop();

    let times: Vec<(i64, i64)> = player.received[0]
        .lock()
        .unwrap()
        .iter()
        .map(|buf| match buf {
            MediaBuffer::Packet(packet) => (packet.pts().unwrap(), packet.dts().unwrap()),
            _ => panic!("an RTSP player is handed packets"),
        })
        .collect();
    assert!(
        times.windows(2).any(|w| w[1].0 < w[0].0),
        "{codec:?}: pictures come out of the order they are shown: {times:?}"
    );
    for (n, window) in times.windows(2).enumerate() {
        assert!(
            window[1].1 > window[0].1,
            "{codec:?}: decoded in order at {n}: {times:?}"
        );
    }
    assert!(
        times.iter().all(|(pts, dts)| dts <= pts),
        "{codec:?}: none decoded after it is shown: {times:?}"
    );
    // Evenly spaced, as the pictures are, once the first has shown how far
    // apart they are.
    let frame = 90_000 / 25;
    assert!(
        times[2..].windows(2).all(|w| w[1].1 - w[0].1 == frame),
        "{codec:?}: decoded a picture apart: {times:?}"
    );
    true
}

/// VP8 and VP9 — what a WebRTC publisher may bring — played over RTSP: each
/// frame whole, its keyframes flagged, its size read from its first, and
/// every frame decoded.
#[test]
fn a_player_takes_vp8_and_vp9() {
    use crate::elements::{
        SwEncoder, SwEncoderOptions, TestVideoOptions, TestVideoSource, VideoCodec,
    };
    let mut played = 0;
    for codec in [VideoCodec::Vp8, VideoCodec::Vp9] {
        let Ok(encoder) = SwEncoder::new(
            "encoder",
            SwEncoderOptions {
                codec,
                width: 320,
                height: 240,
                pixel_format: ffmpeg::format::Pixel::YUV420P,
                frame_rate: ffmpeg::Rational::new(25, 1),
                bit_rate: 300_000,
                gop_size: 25,
                max_b_frames: None,
            },
        ) else {
            continue;
        };
        played += 1;
        let (runner, handle, broadcast) = server(false);
        let sink = broadcast
            .publish(&[StreamInfo::from(&encoder)])
            .unwrap()
            .remove(0);
        let pictures = TestVideoSource::new(
            "pictures",
            TestVideoOptions {
                width: 320,
                height: 240,
                frame_rate: ffmpeg::Rational::new(25, 1),
            },
        );
        let (camera, ()) = Pipeline::new("camera", pictures, |source, ctx| {
            let branch = ctx.branch().queue("encode", 4).pipe(encoder).to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .unwrap();
        camera.run().unwrap();

        let player = play(&handle.url("cam"), RtspTransport::Tcp);
        // SAFETY: plain fields of parameters the stream owns.
        let size = unsafe {
            let raw = player.streams[0].parameters().as_ptr();
            ((*raw).width, (*raw).height)
        };
        assert_eq!(size, (320, 240), "{codec:?}: the size, from a keyframe");
        player.wait_for(40);
        player.pipeline.stop();
        camera.stop();
        runner.stop();

        let packets = player.received[0].lock().unwrap();
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(
            player.streams[0].parameters().clone(),
        )
        .unwrap()
        .decoder()
        .video()
        .unwrap();
        let mut decoded = 0;
        let mut keys = 0;
        for buf in packets.iter() {
            let MediaBuffer::Packet(packet) = buf else {
                panic!("an RTSP player is handed packets");
            };
            keys += usize::from(packet.is_key());
            decoder.send_packet(&***packet).unwrap();
            let mut frame = ffmpeg::frame::Video::empty();
            while decoder.receive_frame(&mut frame).is_ok() {
                decoded += 1;
            }
        }
        assert!(packets.len() >= 40, "{codec:?}");
        assert!(
            keys >= 1 && keys < packets.len(),
            "{codec:?}: {keys} keyframes"
        );
        assert!(
            decoded + 2 >= packets.len(),
            "{codec:?}: {decoded} of {} decoded",
            packets.len()
        );
    }
    if played == 0 {
        eprintln!("skipping: no libvpx in this FFmpeg");
    }
}

/// What a player is sent is what RTP carries here: a camera's codecs, and
/// not AV1 or AC-3, which a WebRTC or a file input may bring.
#[test]
fn what_players_are_sent_is_what_rtp_carries() {
    let unit = ffmpeg::Rational::new(1, 90_000);
    let video = |codec| StreamInfo::video(codec, &[], (320, 240), unit).unwrap();
    let audio = |codec| StreamInfo::audio(codec, &[], 48_000, 2, unit).unwrap();
    use ffmpeg::codec::Id;
    for carried in [
        video(Id::H264),
        video(Id::HEVC),
        audio(Id::AAC),
        audio(Id::OPUS),
    ] {
        assert!(
            RtspServer::carries(&carried),
            "{:?}",
            carried.parameters().id()
        );
    }
    for left_out in [video(Id::AV1), audio(Id::AC3)] {
        assert!(
            !RtspServer::carries(&left_out),
            "{:?}",
            left_out.parameters().id()
        );
    }
}

/// A token in the URL's query lets a player in, where it sends no
/// `Authorization`; the description's base leaves the query out, so each
/// track it sets up is still named.
#[test]
fn a_player_presents_a_token_in_its_urls_query() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam, _live, asked) = server_asking();
    let _camera = feed(&cam, &path);
    wait_published(&cam);
    let url = handle.url("cam");
    let options = || RtspOptions {
        transport: RtspTransport::Tcp,
        timeout: Duration::from_secs(5),
    };
    assert!(RtspSource::open("player", format!("{url}?token=wrong"), options()).is_err());
    let player = play(&format!("{url}?token={}", long_token()), RtspTransport::Tcp);
    player.wait_for(10);
    check_played(&player);
    player.pipeline.stop();
    runner.stop();
    assert!(
        asked.lock().unwrap().iter().any(
            |request| request.credentials == crate::elements::Credentials::Bearer(long_token())
        ),
        "presented as a Bearer token"
    );
}

/// A publisher of a path is handed out as a source, while the path's
/// players read what the path was added with: the one URL pushed to and
/// played, apart. Its `Publishers` gone, the path takes none; the source
/// gone, the publisher's connection ends.
#[test]
fn a_publisher_is_handed_out_apart_from_what_players_read() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam) = server(false);
    let camera = feed(&cam, &path);
    wait_published(&cam);
    let publishers = handle.take_publishers("cam").unwrap();
    assert_eq!(publishers.path(), "cam");
    assert!(
        matches!(
            handle.take_publishers("cam"),
            Err(RtspServerError::PathTaken(_))
        ),
        "one taker a path"
    );

    let publisher =
        publish(&handle.url("cam"), &path, RtspTransport::Tcp).expect("the server takes it");
    let (source, streams) = publishers.next().expect("the publisher is handed out");
    let (_, fixture) = FileDemuxer::open("probe", &path).unwrap();
    assert_eq!(streams.len(), fixture.len(), "every track it set up");
    assert_eq!(
        cam.stats().publications,
        1,
        "what players read is untouched"
    );
    let player = play(&handle.url("cam"), RtspTransport::Tcp);
    check_played(&player);
    player.pipeline.stop();

    // Its source let go of, nothing takes the publisher.
    drop(source);
    let deadline = Instant::now() + Duration::from_secs(10);
    while handle
        .session_stats()
        .iter()
        .any(|session| session.role == Access::Publish)
    {
        assert!(Instant::now() < deadline, "the publisher was let go of");
        thread::sleep(Duration::from_millis(20));
    }
    publisher.stop();

    drop(publishers);
    assert!(
        publish(&handle.url("cam"), &path, RtspTransport::Tcp).is_err(),
        "a path players alone use takes no publisher"
    );
    let publishers = handle
        .take_publishers("cam")
        .expect("taken again once let go of");
    camera.stop();
    runner.stop();
    assert!(publishers.next().is_none(), "the server stopped");
}
