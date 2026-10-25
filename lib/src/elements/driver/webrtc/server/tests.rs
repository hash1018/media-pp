//! The server against `str0m` clients of the test's own — a player as a
//! browser's WHEP client makes one, a publisher as a WHIP encoder does — on
//! this machine's loopback alone.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, UdpSocket};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ffmpeg_next::{self as ffmpeg, Rescale};
use str0m::change::{SdpAnswer, SdpPendingOffer};
use str0m::channel::ChannelId;
use str0m::format::Codec;
use str0m::media::{Direction, MediaKind, MediaTime, Mid};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, Input, Output, Rtc};

use super::*;
use crate::driver::DriverRunner;
use crate::elements::{FileDemuxer, Pacer, StreamInfo};
use crate::pipeline::Pipeline;

/// A server on the loopback serving `cam` for players, and taking a
/// publisher on `live`, relayed into the broadcast `live`'s players play.
fn server() -> (Arc<DriverRunner>, WebRtcServerHandle, Broadcast, Broadcast) {
    let (server, handle) =
        WebRtcServer::bind("server", "127.0.0.1:0", WebRtcServerOptions::default())
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
    (runner, handle, cam, live)
}

/// Serves `name` on `handle`'s server for the rest of the test, and
/// answers the broadcast its players read — what a path's muxers write
/// into, and what a test writes into or looks at directly.
fn serve(handle: &WebRtcServerHandle, name: &str) -> Broadcast {
    let path = handle.path(name).unwrap();
    let broadcast = path.broadcast().clone();
    // Dropped, the path would end its players: kept as long as the test.
    std::mem::forget(path);
    broadcast
}

/// Plays the fixture into `broadcast` at its own speed, over and over.
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

/// A WebRTC client of the test's own, as a browser would be one.
struct Client {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    events: Vec<Event>,
    /// What arrived of each medium, whether each was a keyframe: kept
    /// apart from `events` as it comes, so that looking at it after every
    /// datagram does not go through thousands of them each time — which,
    /// on a loaded machine, kept the client from reading its socket in
    /// time and lost packets.
    media: std::collections::HashMap<Mid, Vec<bool>>,
    connected: bool,
}

impl Client {
    fn new() -> Self {
        // A receive buffer as large as a browser's: a picture larger than
        // the system's default arrives at once.
        let socket = {
            let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
            let socket = socket2::Socket::new(
                socket2::Domain::IPV4,
                socket2::Type::DGRAM,
                Some(socket2::Protocol::UDP),
            )
            .unwrap();
            let _ = socket.set_recv_buffer_size(8 * 1024 * 1024);
            socket.bind(&addr.into()).unwrap();
            UdpSocket::from(socket)
        };
        let local = socket.local_addr().unwrap();
        let mut rtc = Rtc::builder().build(Instant::now());
        rtc.add_local_candidate(Candidate::host(local, "udp").unwrap());
        Self {
            rtc,
            socket,
            local,
            events: Vec::new(),
            media: std::collections::HashMap::new(),
            connected: false,
        }
    }

    /// An offer of `media`, each in `direction`, and a data channel where
    /// `channel` names one.
    fn offer(
        &mut self,
        media: &[MediaKind],
        direction: Direction,
        channel: Option<&str>,
    ) -> (String, SdpPendingOffer, Vec<Mid>, Option<ChannelId>) {
        let mut api = self.rtc.sdp_api();
        let mids = media
            .iter()
            .map(|kind| api.add_media(*kind, direction, None, None, None))
            .collect();
        let channel = channel.map(|label| api.add_channel(label.to_owned()));
        let (offer, pending) = api.apply().expect("an offer");
        (offer.to_sdp_string(), pending, mids, channel)
    }

    fn accept(&mut self, pending: SdpPendingOffer, answer: &str) {
        let answer = SdpAnswer::from_sdp_string(answer).expect("the server's answer is SDP");
        self.rtc
            .sdp_api()
            .accept_answer(pending, answer)
            .expect("the answer is accepted");
    }

    /// Runs the client until `done` says so of what it has seen, or
    /// `within` has gone by.
    fn run_until(&mut self, within: Duration, mut done: impl FnMut(&mut Self) -> bool) -> bool {
        let deadline = Instant::now() + within;
        let mut buffer = vec![0u8; 65_536];
        while Instant::now() < deadline {
            if done(self) {
                return true;
            }
            let mut wait = Duration::from_millis(5);
            loop {
                match self.rtc.poll_output().expect("the client runs") {
                    Output::Timeout(at) => {
                        wait = wait.min(at.saturating_duration_since(Instant::now()));
                        break;
                    }
                    Output::Transmit(transmit) => {
                        let _ = self
                            .socket
                            .send_to(&transmit.contents, transmit.destination);
                    }
                    Output::Event(Event::MediaData(data)) => self
                        .media
                        .entry(data.mid)
                        .or_default()
                        .push(data.is_keyframe()),
                    Output::Event(event) => {
                        self.connected |= matches!(event, Event::Connected);
                        self.events.push(event);
                    }
                }
            }
            match crate::elements::udp::recv_within(
                &self.socket,
                &mut buffer,
                wait.max(Duration::from_millis(1)),
            ) {
                Ok((length, source)) => {
                    if let Ok(contents) = buffer[..length].try_into() {
                        let _ = self.rtc.handle_input(Input::Receive(
                            Instant::now(),
                            Receive {
                                proto: Protocol::Udp,
                                source,
                                destination: self.local,
                                contents,
                            },
                        ));
                    }
                }
                Err(_) => {
                    let _ = self.rtc.handle_input(Input::Timeout(Instant::now()));
                }
            }
        }
        done(self)
    }

    fn connected(&self) -> bool {
        self.connected
    }

    /// What arrived of `mid`: whether each was a keyframe.
    fn received(&self, mid: Mid) -> &[bool] {
        self.media.get(&mid).map_or(&[], Vec::as_slice)
    }
}

#[test]
fn a_player_is_sent_the_picture_from_a_keyframe_and_no_track_webrtc_cannot_carry() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam, _live) = server();
    let camera = feed(&cam, &path);

    let mut client = Client::new();
    let (offer, pending, mids, _) = client.offer(
        &[MediaKind::Video, MediaKind::Audio],
        Direction::RecvOnly,
        None,
    );
    let (session, answer) = handle.play("cam", &offer).expect("the offer is answered");
    client.accept(pending, &answer);
    let enough = client.run_until(Duration::from_secs(20), |client| {
        client.received(mids[0]).len() >= 30
    });
    assert!(
        enough,
        "pictures arrived: {}",
        client.received(mids[0]).len()
    );
    assert!(client.connected());
    assert!(
        client.received(mids[0])[0],
        "the first picture is a keyframe"
    );
    assert!(
        client.received(mids[1]).is_empty(),
        "the fixture's AAC is not a codec WebRTC carries"
    );
    assert_eq!(handle.sessions(), 1);

    let mut connected = false;
    while let Some(event) = handle.next_event(Duration::from_millis(100)) {
        if matches!(&event, WebRtcServerEvent::Connected { session: s, path, role: SessionRole::Play } if *s == session && path == "cam")
        {
            connected = true;
        }
    }
    assert!(connected, "the server says the player connected");

    assert!(handle.close(session));
    assert_eq!(handle.sessions(), 0);
    assert!(!handle.close(session), "a session closes once");
    camera.stop();
    runner.stop();
}

#[test]
fn offers_for_what_is_not_served_are_refused() {
    let (runner, handle, _cam, _live) = server();
    let mut client = Client::new();
    let (offer, ..) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    assert!(matches!(
        handle.play("nowhere", &offer),
        Err(WebRtcServerError::NoSuchPath(_))
    ));
    assert!(
        matches!(
            handle.play("cam", &offer),
            Err(WebRtcServerError::NothingToPlay(_))
        ),
        "nothing has been published to cam yet"
    );
    assert!(matches!(
        handle.publish("cam", &offer),
        Err(WebRtcServerError::NotPublishable(_))
    ));
    assert!(matches!(
        handle.play("live", "not sdp"),
        Err(WebRtcServerError::InvalidOffer(_)) | Err(WebRtcServerError::NothingToPlay(_))
    ));
    assert!(matches!(
        handle.publish("live", "not sdp"),
        Err(WebRtcServerError::InvalidOffer(_))
    ));
    assert_eq!(
        handle.sessions(),
        0,
        "a refused offer leaves no session behind"
    );
    runner.stop();
}

/// A publisher is taken where a path's `Publishers` lives, apart from what
/// its players play: refused where nothing is served, refused for a path
/// players alone use, and taken again once let go of.
#[test]
fn publishers_are_taken_apart_from_what_players_play() {
    let (runner, handle, _cam, _live) = server();
    let mut client = Client::new();
    let (offer, ..) = client.offer(&[MediaKind::Video], Direction::SendOnly, None);
    assert!(matches!(
        handle.publish("nowhere", &offer),
        Err(WebRtcServerError::NoSuchPath(_))
    ));
    assert!(matches!(
        handle.take_publishers("live"),
        Err(WebRtcServerError::PathTaken(_))
    ));
    assert!(matches!(
        handle.take_publishers("bad path"),
        Err(WebRtcServerError::InvalidPath(_))
    ));

    // A path that only takes publishers, with nothing to play.
    let studio = handle.take_publishers("studio").unwrap();
    assert_eq!(studio.path(), "studio");
    let (session, _answer) = handle
        .publish("studio", &offer)
        .expect("the path takes a publisher");
    assert!(handle.close(session));
    drop(studio);
    assert!(matches!(
        handle.publish("studio", &offer),
        Err(WebRtcServerError::NoSuchPath(_))
    ));
    let studio = handle
        .take_publishers("studio")
        .expect("taken again once let go of");
    runner.stop();
    assert!(studio.next().is_none(), "the server stopped");
}

/// What a path's muxer writes is what its players are sent; a codec WebRTC
/// does not carry is refused at the muxer; and the path let go of ends its
/// players, and nobody new plays it.
#[test]
fn a_path_plays_what_its_muxer_writes_and_ends_its_players_as_it_goes() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, _cam, _live) = server();
    let door = handle.path("door").unwrap();
    assert_eq!(door.path(), "door");
    assert!(matches!(
        handle.path("door"),
        Err(WebRtcServerError::PathTaken(_))
    ));
    let (demuxer, streams) = FileDemuxer::open("door-camera", &path).unwrap();
    demuxer.looping_handle().set_looping(true);
    let video = streams
        .iter()
        .find(|stream| stream.kind() == ffmpeg::media::Type::Video)
        .unwrap()
        .clone();
    let audio = streams
        .iter()
        .find(|stream| stream.kind() == ffmpeg::media::Type::Audio)
        .unwrap();
    let mut muxer = door.muxer();
    assert!(
        muxer.add_stream("sound", audio).is_err(),
        "the fixture's AAC is not what WebRTC carries"
    );
    let track = muxer.add_stream("pictures", &video).unwrap();
    let mut sinks = muxer.open().unwrap();
    let sink = sinks.take(track).unwrap();
    let (camera, ()) = Pipeline::new("door-camera", demuxer, |source, ctx| {
        let branch = ctx
            .branch()
            .queue("packets", 64)
            .pipe(Pacer::new("pacer"))
            .to(sink)?;
        ctx.attach(source, video.index(), branch)?;
        Ok(())
    })
    .unwrap();
    camera.run().unwrap();

    let mut client = Client::new();
    let (offer, pending, mids, _) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    let deadline = Instant::now() + Duration::from_secs(10);
    let answer = loop {
        match handle.play("door", &offer) {
            Ok((_, answer)) => break answer,
            Err(WebRtcServerError::NothingToPlay(_)) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => panic!("door plays: {error}"),
        }
    };
    client.accept(pending, &answer);
    assert!(
        client.run_until(Duration::from_secs(10), |client| client
            .received(mids[0])
            .len()
            >= 5),
        "a player of the path is sent what its muxer writes"
    );
    assert_eq!(door.players(), 1, "the player reads the path");

    drop(door);
    let deadline = Instant::now() + Duration::from_secs(10);
    while handle.sessions() > 0 {
        assert!(
            Instant::now() < deadline,
            "the path let go of ends its players"
        );
        client.run_until(Duration::from_millis(20), |_| false);
    }
    assert!(matches!(
        handle.play("door", &offer),
        Err(WebRtcServerError::NoSuchPath(_))
    ));
    camera.stop();
    runner.stop();
}

#[test]
fn a_stopped_server_answers_nothing() {
    let (server, handle) =
        WebRtcServer::bind("server", "127.0.0.1:0", WebRtcServerOptions::default()).unwrap();
    drop(server);
    let mut client = Client::new();
    let (offer, ..) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    assert!(matches!(
        handle.play("cam", &offer),
        Err(WebRtcServerError::NotRunning)
    ));
}

#[test]
fn binding_every_interface_needs_the_addresses_players_reach() {
    assert!(matches!(
        WebRtcServer::bind("server", "0.0.0.0:0", WebRtcServerOptions::default()),
        Err(WebRtcServerError::NoCandidate(_))
    ));
}

/// A publisher's H.264, as an encoder for WebRTC sends it, is published
/// into its path once its parameter sets have come, and played back to a
/// player of that path.
#[test]
fn a_publisher_is_played_back() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, _cam, live) = server();

    // The fixture's pictures, as Annex B access units with their time.
    let (_, streams) = FileDemuxer::open("probe", &path).unwrap();
    let video = streams
        .iter()
        .find(|s| s.kind() == ffmpeg::media::Type::Video)
        .unwrap();
    // Made Annex B, with the parameter sets ahead of each keyframe, as a
    // track sink makes what it sends.
    let outbox = super::super::outbox::Outbox::new(1 << 16);
    let mut sink = super::super::track::WebRtcTrackSink::new(
        super::super::command::TrackId(0),
        MediaKind::Video,
        None,
        Arc::new(std::sync::Mutex::new(vec![Codec::H264])),
        outbox.clone(),
    );
    sink.set_source_parameters(video.parameters()).unwrap();
    let mut input = ffmpeg::format::input(&path).unwrap();
    let pictures: Vec<(Vec<u8>, i64)> = input
        .packets()
        .filter(|(stream, _)| stream.index() == video.index())
        .map(|(_, mut packet)| {
            let ts = packet
                .pts()
                .unwrap()
                .rescale(video.time_base(), ffmpeg::Rational(1, 90_000));
            packet.set_time_base(video.time_base());
            crate::element::RawSink::consume(&mut sink, crate::buffer::MediaBuffer::packet(packet))
                .unwrap();
            let sent = outbox.take_all().pop_front().unwrap();
            (sent.payload(), ts)
        })
        .collect();

    let mut publisher = Client::new();
    let (offer, pending, mids, _) = publisher.offer(&[MediaKind::Video], Direction::SendOnly, None);
    let (_session, answer) = handle
        .publish("live", &offer)
        .expect("the path takes a publisher");
    publisher.accept(pending, &answer);
    assert!(publisher.run_until(Duration::from_secs(10), |client| client.connected()));
    assert!(
        matches!(
            handle.publish("live", &offer),
            Err(WebRtcServerError::NotPublishable(_))
        ),
        "one publisher at a time"
    );

    let began = Instant::now();
    let mut next = 0;
    let done = publisher.run_until(Duration::from_secs(30), |client| {
        // Each picture when it is due.
        while next < pictures.len()
            && began.elapsed() >= Duration::from_millis((pictures[next].1 / 90) as u64)
        {
            let (data, ts) = &pictures[next];
            let writer = client.rtc.writer(mids[0]).unwrap();
            let pt = writer
                .payload_params()
                .find(|params| params.spec().codec == Codec::H264)
                .map(|params| params.pt())
                .unwrap();
            writer
                .write(
                    pt,
                    Instant::now(),
                    MediaTime::from_90khz(*ts as u64),
                    data.clone(),
                )
                .unwrap();
            next += 1;
        }
        live.is_live() && next >= pictures.len() / 2
    });
    assert!(done, "the publisher's stream never reached its broadcast");
    let tracks = live.tracks().unwrap();
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].parameters().id(), ffmpeg::codec::Id::H264);
    // SAFETY: plain fields of parameters held here.
    let (width, height) = unsafe {
        (
            (*tracks[0].parameters().as_ptr()).width,
            (*tracks[0].parameters().as_ptr()).height,
        )
    };
    assert!(
        width > 0 && height > 0,
        "the size is read from its parameter sets"
    );

    // A player of `live`, while the publisher goes on.
    let mut client = Client::new();
    let (offer, pending, player_mids, _) =
        client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    let (_, answer) = handle.play("live", &offer).unwrap();
    client.accept(pending, &answer);
    let mid = player_mids[0];
    let mut got = false;
    for _ in 0..200 {
        // Both ends run, a little at a time.
        publisher.run_until(Duration::from_millis(20), |client| {
            while next < pictures.len()
                && began.elapsed() >= Duration::from_millis((pictures[next].1 / 90) as u64)
            {
                let (data, ts) = &pictures[next];
                let writer = client.rtc.writer(mids[0]).unwrap();
                let pt = writer
                    .payload_params()
                    .find(|params| params.spec().codec == Codec::H264)
                    .map(|params| params.pt())
                    .unwrap();
                let _ = writer.write(
                    pt,
                    Instant::now(),
                    MediaTime::from_90khz(*ts as u64),
                    data.clone(),
                );
                next += 1;
            }
            false
        });
        if client.run_until(Duration::from_millis(20), |client| {
            client.received(mid).len() >= 5
        }) {
            got = true;
            break;
        }
    }
    assert!(got, "a player of the publisher's path is sent its pictures");
    assert!(client.received(mid)[0], "from a keyframe");

    // The publisher leaving ends the run.
    drop(publisher);
    runner.stop();
    let deadline = Instant::now() + Duration::from_secs(10);
    while live.is_live() {
        assert!(Instant::now() < deadline, "the publisher's run did not end");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn data_channels_carry_both_ways() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam, _live) = server();
    let camera = feed(&cam, &path);
    let mut client = Client::new();
    let (offer, pending, _, channel) =
        client.offer(&[MediaKind::Video], Direction::RecvOnly, Some("chat"));
    let (session, answer) = handle.play("cam", &offer).unwrap();
    client.accept(pending, &answer);
    assert!(client.run_until(Duration::from_secs(10), |client| {
        client
            .events
            .iter()
            .any(|event| matches!(event, Event::ChannelOpen(..)))
    }));

    // The server hears the channel open, then answers on it.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assert!(
            Instant::now() < deadline,
            "the server never saw the channel open"
        );
        if let Some(WebRtcServerEvent::ChannelOpen {
            session: s,
            channel,
        }) = handle.next_event(Duration::from_millis(50))
            && s == session
        {
            assert_eq!(channel, "chat");
            break;
        }
        client.run_until(Duration::from_millis(10), |_| false);
    }
    handle.send(session, "chat", false, b"hello").unwrap();
    assert!(matches!(
        handle.send(session, "nope", false, b"x"),
        Err(WebRtcServerError::NoSuchChannel(..))
    ));
    assert!(client.run_until(Duration::from_secs(5), |client| {
        client
            .events
            .iter()
            .any(|event| matches!(event, Event::ChannelData(data) if data.data == b"hello"))
    }));

    let id = channel.unwrap();
    client.rtc.channel(id).unwrap().write(false, b"hi").unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        client.run_until(Duration::from_millis(10), |_| false);
        assert!(
            Instant::now() < deadline,
            "the server never heard the client"
        );
        if let Some(WebRtcServerEvent::Data {
            session: s,
            channel,
            binary,
            data,
        }) = handle.next_event(Duration::from_millis(10))
            && s == session
        {
            assert_eq!(
                (channel.as_str(), binary, data.as_slice()),
                ("chat", false, &b"hi"[..])
            );
            break;
        }
    }
    camera.stop();
    runner.stop();
}

/// One HTTP request, and its status line, headers and body.
fn http(addr: SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}

#[test]
fn whep_over_http_answers_offers_and_ends_sessions() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam, _live) = server();
    let camera = feed(&cam, &path);
    let endpoint = handle.serve_http("127.0.0.1:0").unwrap();
    let addr = endpoint.local_addr();

    let preflight = http(
        addr,
        "OPTIONS /cam/whep HTTP/1.1\r\nHost: x\r\nOrigin: http://example\r\n\r\n",
    );
    assert!(preflight.starts_with("HTTP/1.1 204"), "{preflight}");
    assert!(preflight.contains("Access-Control-Allow-Origin: *"));
    assert!(preflight.contains("Access-Control-Expose-Headers: Location"));

    let mut client = Client::new();
    let (offer, pending, mids, _) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    let response = http(
        addr,
        &format!(
            "POST /cam/whep HTTP/1.1\r\nHost: x\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{offer}",
            offer.len()
        ),
    );
    assert!(response.starts_with("HTTP/1.1 201"), "{response}");
    let location = response
        .lines()
        .find_map(|line| line.strip_prefix("Location: "))
        .unwrap()
        .to_owned();
    let answer = response.split("\r\n\r\n").nth(1).unwrap();
    client.accept(pending, answer);
    assert!(client.run_until(
        Duration::from_secs(20),
        |client| client.received(mids[0]).len() >= 10
    ));

    let missing = http(
        addr,
        "POST /nowhere/whep HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
    );
    assert!(missing.starts_with("HTTP/1.1 404"), "{missing}");

    let deleted = http(
        addr,
        &format!("DELETE {location} HTTP/1.1\r\nHost: x\r\n\r\n"),
    );
    assert!(deleted.starts_with("HTTP/1.1 200"), "{deleted}");
    assert_eq!(handle.sessions(), 0);
    let again = http(
        addr,
        &format!("DELETE {location} HTTP/1.1\r\nHost: x\r\n\r\n"),
    );
    assert!(again.starts_with("HTTP/1.1 404"), "{again}");
    drop(endpoint);
    camera.stop();
    runner.stop();
}

/// The candidate port an answer gives.
fn answered_port(answer: &str) -> u16 {
    answer
        .lines()
        .find_map(|line| line.strip_prefix("a=candidate:"))
        .and_then(|candidate| candidate.split_whitespace().nth(5))
        .and_then(|port| port.parse().ok())
        .expect("the answer gives a candidate")
}

/// A new session goes to the thread with the fewest, each thread on a
/// socket of its own; both play.
#[test]
fn sessions_are_spread_over_the_threads() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam, _live) = server();
    let camera = feed(&cam, &path);
    let mut players: Vec<(Client, Mid, u16)> = (0..2)
        .map(|_| {
            let mut client = Client::new();
            let (offer, pending, mids, _) =
                client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
            let (_, answer) = handle.play("cam", &offer).unwrap();
            let port = answered_port(&answer);
            client.accept(pending, &answer);
            (client, mids[0], port)
        })
        .collect();
    assert_ne!(players[0].2, players[1].2, "two threads, two sockets");
    for (client, mid, _) in &mut players {
        let mid = *mid;
        assert!(
            client.run_until(Duration::from_secs(20), |client| client.received(mid).len()
                >= 10)
        );
    }
    camera.stop();
    runner.stop();
}

/// A player catching up is sent its backlog at twice the stream's speed,
/// every packet of it: str0m lets one packet out and asks for the time
/// again, and the server once sent a packet a turn of its loop, falling
/// further behind the larger the pictures.
#[test]
fn a_player_catches_up_on_large_pictures_in_time() {
    if crate::test_support::skipped_in_ci("a_player_catches_up_on_large_pictures_in_time") {
        return;
    }
    let (runner, handle, cam, _live) = server();
    let video = StreamInfo::new(
        {
            let mut parameters = ffmpeg::codec::Parameters::new();
            // SAFETY: parameters this test owns alone; plain fields.
            unsafe {
                (*parameters.as_mut_ptr()).codec_type = ffmpeg::media::Type::Video.into();
                (*parameters.as_mut_ptr()).codec_id = ffmpeg::codec::Id::H264.into();
                (*parameters.as_mut_ptr()).width = 1920;
                (*parameters.as_mut_ptr()).height = 1080;
            }
            parameters
        },
        ffmpeg::Rational(1, 90_000),
    );
    let mut sinks = cam.publish(&[video]).unwrap();
    // Another reader, for which what is written is kept.
    let _watching = cam.subscribe();
    // Just under a second of 100 kB pictures from one keyframe, in Annex B,
    // written at once — as much as a player joins at a keyframe with (see
    // `PLAYER_JOIN_WITHIN`): some two thousand RTP packets for it to catch
    // up on in one. Nothing decodes them; the player only puts them back
    // together. The keyframe carries its parameter sets, as one a player
    // can start at does — a track sink refuses one that has none.
    let pictures = 24usize;
    for n in 0..pictures as i64 {
        let mut data = match n {
            0 => vec![
                0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1f, 0, 0, 0, 1, 0x68, 0xce, 0, 0, 0, 1, 0x65,
            ],
            _ => vec![0, 0, 0, 1, 0x41],
        };
        data.resize(100_000, 0xAB);
        let mut packet = ffmpeg::Packet::copy(&data);
        packet.set_pts(Some(n * 3_600));
        packet.set_dts(Some(n * 3_600));
        packet.set_time_base(ffmpeg::Rational(1, 90_000));
        if n == 0 {
            packet.set_flags(ffmpeg::packet::Flags::KEY);
        }
        crate::element::RawSink::consume(&mut sinks[0], crate::buffer::MediaBuffer::packet(packet))
            .unwrap();
    }

    let mut client = Client::new();
    let (offer, pending, mids, _) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    let (_, answer) = handle.play("cam", &offer).unwrap();
    client.accept(pending, &answer);
    assert!(client.run_until(Duration::from_secs(10), |client| client.connected()));
    let connected = Instant::now();
    let all = client.run_until(Duration::from_secs(10), |client| {
        client.received(mids[0]).len() >= pictures
    });
    let took = connected.elapsed();
    assert!(
        all,
        "only {} of {pictures} pictures arrived",
        client.received(mids[0]).len()
    );
    // Under a second of pictures at twice their speed: half a second at
    // most, with room.
    assert!(
        took < Duration::from_millis(1_400),
        "{pictures} pictures took {took:?} to catch up on"
    );
    runner.stop();
}

/// With an `Authorize`, an offer over HTTP is put to it before anything
/// else — one for a path not served too — and answered 401 where it is
/// refused; the application's own `play` is not asked. A session's URL,
/// all it takes to end it, says nothing of another session's.
#[test]
fn offers_over_http_are_authorized_and_session_urls_cannot_be_guessed() {
    use crate::elements::{Access, AccessRequest, Authorize, Credentials};
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let authorize = Authorize::new(|request: &AccessRequest| {
        request.credentials
            == Credentials::Bearer(match request.access {
                Access::Play => "watch".into(),
                Access::Publish => "send".into(),
            })
    });
    let (server, handle) = WebRtcServer::bind(
        "server",
        "127.0.0.1:0",
        WebRtcServerOptions {
            authorize: Some(authorize),
            ..WebRtcServerOptions::default()
        },
    )
    .expect("a loopback port is free");
    let cam = serve(&handle, "cam");
    let runner = DriverRunner::new(server);
    runner.run().unwrap();
    let camera = feed(&cam, &path);
    let endpoint = handle.serve_http("127.0.0.1:0").unwrap();
    let addr = endpoint.local_addr();

    let preflight = http(addr, "OPTIONS /cam/whep HTTP/1.1\r\nHost: x\r\n\r\n");
    assert!(
        preflight.contains("Access-Control-Allow-Headers: Content-Type, Authorization"),
        "a page on another origin may send a token: {preflight}"
    );
    let post = |target: &str, authorization: &str| {
        let mut client = Client::new();
        let (offer, _, _, _) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
        http(
            addr,
            &format!(
                "POST {target} HTTP/1.1\r\nHost: x\r\n{authorization}Content-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{offer}",
                offer.len()
            ),
        )
    };
    for (target, authorization) in [
        ("/cam/whep", ""),
        ("/cam/whep", "Authorization: Bearer send\r\n"),
        ("/nowhere/whep", ""),
    ] {
        let refused = post(target, authorization);
        assert!(
            refused.starts_with("HTTP/1.1 401"),
            "{target} {authorization}: {refused}"
        );
        assert!(refused.contains("WWW-Authenticate: Bearer"), "{refused}");
    }
    assert_eq!(handle.sessions(), 0);

    let locations: Vec<String> = (0..2)
        .map(|_| {
            let answered = post("/cam/whep", "Authorization: Bearer watch\r\n");
            assert!(answered.starts_with("HTTP/1.1 201"), "{answered}");
            answered
                .lines()
                .find_map(|line| line.strip_prefix("Location: /session/"))
                .unwrap()
                .to_owned()
        })
        .collect();
    let ids: Vec<SessionId> = locations.iter().map(|id| id.parse().unwrap()).collect();
    for (location, id) in locations.iter().zip(&ids) {
        assert_eq!(location.len(), 32, "{location}");
        assert_eq!(&id.to_string(), location);
    }
    assert_ne!(
        locations[0][..16],
        locations[1][..16],
        "half of an id is drawn from the server's key, not counted"
    );

    // The application's own call is its own decision.
    let mut client = Client::new();
    let (offer, _, _, _) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    assert!(handle.play("cam", &offer).is_ok());
    drop(endpoint);
    camera.stop();
    runner.stop();
}

/// A playing session's figures: opened before it connects, then where the
/// client is, what it was sent and the round trip, about once a second —
/// and gone with the session.
#[test]
fn a_session_says_how_it_is_doing() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, cam, _live) = server();
    let camera = feed(&cam, &path);

    let mut client = Client::new();
    let (offer, pending, mids, _) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    let (session, answer) = handle.play("cam", &offer).unwrap();
    let opened = handle.session_stats();
    assert_eq!(opened.len(), 1);
    assert_eq!(opened[0].session, session);
    assert_eq!(opened[0].path, "cam");
    assert_eq!(opened[0].role, SessionRole::Play);
    assert!(!opened[0].connected && opened[0].updated.is_none());

    client.accept(pending, &answer);
    let measured = |handle: &WebRtcServerHandle| {
        handle
            .session_stats()
            .into_iter()
            .find(|stats| stats.session == session && stats.rtt.is_some() && stats.bytes_sent > 0)
    };
    let mut stats = None;
    assert!(
        client.run_until(Duration::from_secs(20), |client| {
            stats = measured(&handle);
            stats.is_some() && client.received(mids[0]).len() >= 30
        }),
        "measured: {:?}",
        handle.session_stats()
    );
    let stats = stats.unwrap();
    assert!(stats.connected);
    assert_eq!(stats.remote, Some(client.local), "where the client is");
    assert!(
        stats.rtt.unwrap() < Duration::from_secs(1),
        "a loopback's round trip: {:?}",
        stats.rtt
    );
    assert!(stats.loss.is_none_or(|loss| (0.0..=1.0).contains(&loss)));
    assert_eq!(stats.fell_behind, 0);
    // Measured again a second on, it has been sent more.
    let mut later = None;
    assert!(
        client.run_until(Duration::from_secs(10), |_| {
            later = measured(&handle).filter(|later| later.updated > stats.updated);
            later.is_some()
        }),
        "measured again"
    );
    let later = later.unwrap();
    assert!(
        later.bytes_sent > stats.bytes_sent,
        "{stats:?} then {later:?}"
    );

    assert!(handle.close(session));
    assert!(handle.session_stats().is_empty(), "gone with its session");
    camera.stop();
    runner.stop();
}

/// A client's DTLS ClientHello follows its nominating STUN at once, and
/// often arrives in the same batch the server reads. It was refused there —
/// the session had not been polled since the nomination — so every
/// connection waited a second for the client to send it again, and under
/// load, as retransmissions were lost too, longer than a client waits.
/// On the loopback a connection takes milliseconds.
///
/// That fault made every connection slow. A datagram the system loses
/// makes one slow now and then — a STUN request gone on a loaded machine,
/// sent again a quarter and a half of a second later — so all but one of
/// four must be quick.
#[test]
fn a_client_connects_without_waiting_to_send_again() {
    let (runner, handle, _cam, _live) = server();
    let mut took = Vec::new();
    for _ in 0..4 {
        let mut client = Client::new();
        let (offer, pending, _, _) = client.offer(&[MediaKind::Video], Direction::SendOnly, None);
        let (session, answer) = handle.publish("live", &offer).unwrap();
        let began = Instant::now();
        client.accept(pending, &answer);
        assert!(client.run_until(Duration::from_secs(5), |client| client.connected()));
        took.push(began.elapsed());
        assert!(handle.close(session), "the path takes the next publisher");
    }
    let quick = took
        .iter()
        .filter(|took| **took < Duration::from_millis(500))
        .count();
    assert!(
        quick >= 3,
        "connected in {took:?}: the DTLS handshake waited to be sent again"
    );
    runner.stop();
}

/// STUN and TURN servers are told as WHIP and WHEP tell them — `Link`
/// headers on the answer to an offer, and on an `OPTIONS` whose credentials
/// pass — and given to the page's browser; a TURN server's credentials are
/// told no one else, and nothing in them breaks a header or the page.
#[test]
fn ice_servers_are_told_to_clients_let_in_and_no_one_else() {
    use crate::elements::{AccessRequest, Authorize, Credentials, IceServer};
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (server, handle) = WebRtcServer::bind(
        "server",
        "127.0.0.1:0",
        WebRtcServerOptions {
            authorize: Some(Authorize::new(|request: &AccessRequest| {
                request.credentials == Credentials::Bearer("watch".into())
            })),
            ice_servers: vec![
                IceServer {
                    urls: vec!["stun:stun.example.org:3478".into()],
                    username: None,
                    credential: None,
                },
                IceServer {
                    urls: vec!["turn:turn.example.org:3478?transport=udp".into()],
                    username: Some("user".into()),
                    credential: Some("pa\"ss</script>".into()),
                },
            ],
            ..WebRtcServerOptions::default()
        },
    )
    .unwrap();
    let cam = serve(&handle, "cam");
    let runner = DriverRunner::new(server);
    runner.run().unwrap();
    let camera = feed(&cam, &path);
    let endpoint = handle.serve_http("127.0.0.1:0").unwrap();
    let addr = endpoint.local_addr();
    let options = |authorization: &str| {
        http(
            addr,
            &format!("OPTIONS /cam/whep HTTP/1.1\r\nHost: x\r\n{authorization}\r\n"),
        )
    };
    let links = |response: &str| {
        response
            .lines()
            .filter(|line| line.starts_with("Link: "))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };

    let preflight = options("");
    assert!(preflight.starts_with("HTTP/1.1 204"), "{preflight}");
    assert!(preflight.contains("Access-Control-Expose-Headers: Location, Link"));
    assert!(
        links(&preflight).is_empty(),
        "nothing for a preflight: {preflight}"
    );
    assert!(links(&options("Authorization: Bearer wrong\r\n")).is_empty());
    let told = links(&options("Authorization: Bearer watch\r\n"));
    assert_eq!(
        told,
        [
            "Link: <stun:stun.example.org:3478>; rel=\"ice-server\"",
            "Link: <turn:turn.example.org:3478?transport=udp>; rel=\"ice-server\"; username=\"user\"; credential=\"pa\\\"ss</script>\"; credential-type=\"password\"",
        ]
    );

    let mut client = Client::new();
    let (offer, ..) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    let answered = http(
        addr,
        &format!(
            "POST /cam/whep HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer watch\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{offer}",
            offer.len()
        ),
    );
    assert!(answered.starts_with("HTTP/1.1 201"), "{answered}");
    assert_eq!(links(&answered), told);

    let page = http(addr, "GET /cam HTTP/1.1\r\nHost: x\r\n\r\n");
    assert!(
        page.contains(
            &format!(r#"iceServers: [{{"urls":["stun:stun.example.org:3478"]}},{{"urls":["turn:turn.example.org:3478?transport=udp"],"username":"user","credential":"pa{b}"ss{b}u003c/script{b}u003e"}}]"#, b = char::from(92))
        ),
        "{page}"
    );
    assert_eq!(page.matches("</script>").count(), 1, "the page's own alone");
    drop(endpoint);
    camera.stop();
    runner.stop();
}

/// A player joins at the newest keyframe only where it is at most a second
/// behind the newest picture; further, it waits for the next keyframe
/// rather than be sent the whole group of pictures at twice its speed.
#[test]
fn a_player_far_behind_the_newest_keyframe_starts_at_the_next() {
    let (runner, handle, cam, _live) = server();
    let video = StreamInfo::new(
        {
            let mut parameters = ffmpeg::codec::Parameters::new();
            // SAFETY: parameters this test owns alone; plain fields.
            unsafe {
                (*parameters.as_mut_ptr()).codec_type = ffmpeg::media::Type::Video.into();
                (*parameters.as_mut_ptr()).codec_id = ffmpeg::codec::Id::H264.into();
                (*parameters.as_mut_ptr()).width = 640;
                (*parameters.as_mut_ptr()).height = 360;
            }
            parameters
        },
        ffmpeg::Rational(1, 90_000),
    );
    let mut sinks = cam.publish(&[video]).unwrap();
    // Picture `n` at 25 a second; a keyframe where `key` says, with its
    // parameter sets.
    let mut write = |n: i64, key: bool| {
        let mut data = if key {
            vec![
                0, 0, 0, 1, 0x67, 0x42, 0xc0, 0x1f, 0, 0, 0, 1, 0x68, 0xce, 0, 0, 0, 1, 0x65,
            ]
        } else {
            vec![0, 0, 0, 1, 0x41]
        };
        data.resize(2_000, 0xAB);
        let mut packet = ffmpeg::Packet::copy(&data);
        packet.set_pts(Some(n * 3_600));
        packet.set_dts(Some(n * 3_600));
        packet.set_time_base(ffmpeg::Rational(1, 90_000));
        if key {
            packet.set_flags(ffmpeg::packet::Flags::KEY);
        }
        crate::element::RawSink::consume(&mut sinks[0], crate::buffer::MediaBuffer::packet(packet))
            .unwrap();
    };
    // Two seconds from one keyframe: the newest picture is 1.96 s past it.
    for n in 0..50 {
        write(n, n == 0);
    }

    let mut client = Client::new();
    let (offer, pending, mids, _) = client.offer(&[MediaKind::Video], Direction::RecvOnly, None);
    let (_, answer) = handle.play("cam", &offer).unwrap();
    client.accept(pending, &answer);
    assert!(client.run_until(Duration::from_secs(10), |client| client.connected()));
    client.run_until(Duration::from_millis(500), |_| false);
    assert!(
        client.received(mids[0]).is_empty(),
        "nothing of the old group of pictures: {} arrived",
        client.received(mids[0]).len()
    );

    // The next group begins: the player starts at its keyframe.
    for n in 50..56 {
        write(n, n == 50);
    }
    assert!(
        client.run_until(Duration::from_secs(10), |client| {
            client.received(mids[0]).len() >= 5
        }),
        "the next group arrives"
    );
    let received = client.received(mids[0]);
    assert!(received[0], "it starts at the next keyframe");
    assert!(
        received.len() <= 6,
        "and at nothing before it: {received:?}"
    );
    runner.stop();
}

/// A camera that changes its picture size partway, played over WHEP: the
/// server carries it on in the same track, and a software decoder after
/// the player's `WebRtcSource` goes on with the new size.
#[test]
fn a_camera_that_changes_its_picture_size_decodes_on_through_the_server() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::buffer::MediaBuffer;
    use crate::elements::{AppSink, SwDecoder, WebRtcOptions, WebRtcSource};

    let (parameters, small) = crate::test_support::h264_configured((320, 240));
    let (_, large) = crate::test_support::h264_configured((640, 360));
    let (runner, handle, cam, _live) = server();
    let http = handle.serve_http("127.0.0.1:0").unwrap();
    let sink = cam
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

    let (source, streams) =
        WebRtcSource::open("player", &http.whep_url("cam"), WebRtcOptions::default()).unwrap();
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
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    wait((320, 240));
    switch.store(true, Ordering::Release);
    wait((640, 360));
    pipeline.stop();
    stop.store(true, Ordering::Release);
    camera.join().unwrap();
    drop(http);
    runner.stop();

    let sizes = sizes.lock().unwrap();
    let first_large = sizes.iter().position(|&size| size == (640, 360)).unwrap();
    assert!(
        sizes[first_large..].iter().all(|&size| size == (640, 360)),
        "the decoder goes on at the new size: {sizes:?}"
    );
}

/// The same through a WHIP publisher of the server's own, as an encoder
/// sends: in through the publish session, out through WHEP, in one track
/// the whole way, decoded on at the new size.
#[test]
fn a_whip_publisher_that_changes_its_picture_size_plays_on_through_the_server() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::buffer::MediaBuffer;
    use crate::elements::{
        AppSink, StreamInfo, SwDecoder, WebRtcMuxer, WebRtcOptions, WebRtcSource,
    };

    let (parameters, small) = crate::test_support::h264_configured((320, 240));
    let (_, large) = crate::test_support::h264_configured((640, 360));
    let (runner, handle, _cam, live) = server();
    let http = handle.serve_http("127.0.0.1:0").unwrap();
    let mut publisher = WebRtcMuxer::create(
        "publisher",
        &http.whip_url("live"),
        WebRtcOptions::default(),
    )
    .unwrap();
    let track = publisher
        .add_stream(
            "video",
            StreamInfo::new(parameters, ffmpeg::Rational::new(1, 90_000)),
        )
        .unwrap();
    let sink = publisher
        .open()
        .expect("the server takes the publisher")
        .take(track)
        .unwrap();
    let (switch, stop) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let camera = crate::test_support::changing_camera(
        sink.into_raw(),
        small,
        large,
        Arc::clone(&switch),
        Arc::clone(&stop),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while !live.is_live() {
        assert!(Instant::now() < deadline, "the publisher is taken in");
        std::thread::sleep(Duration::from_millis(20));
    }

    let (source, streams) =
        WebRtcSource::open("player", &http.whep_url("live"), WebRtcOptions::default()).unwrap();
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
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    wait((320, 240));
    switch.store(true, Ordering::Release);
    wait((640, 360));
    pipeline.stop();
    stop.store(true, Ordering::Release);
    camera.join().unwrap();
    drop(http);
    runner.stop();

    let sizes = sizes.lock().unwrap();
    let first_large = sizes.iter().position(|&size| size == (640, 360)).unwrap();
    assert!(
        sizes[first_large..].iter().all(|&size| size == (640, 360)),
        "the decoder goes on at the new size: {sizes:?}"
    );
}

/// What a player is sent is what WebRTC carries: not a camera's AAC.
#[test]
fn what_players_are_sent_is_what_webrtc_carries() {
    use crate::elements::StreamInfo;
    use ffmpeg::codec::Id;
    let unit = ffmpeg::Rational::new(1, 90_000);
    let video = |codec| StreamInfo::video(codec, &[], (320, 240), unit).unwrap();
    let audio = |codec| StreamInfo::audio(codec, &[], 48_000, 2, unit).unwrap();
    for carried in [
        video(Id::H264),
        video(Id::AV1),
        audio(Id::OPUS),
        audio(Id::PCM_MULAW),
    ] {
        assert!(
            WebRtcServer::carries(&carried),
            "{:?}",
            carried.parameters().id()
        );
    }
    for left_out in [audio(Id::AAC), video(Id::MJPEG)] {
        assert!(
            !WebRtcServer::carries(&left_out),
            "{:?}",
            left_out.parameters().id()
        );
    }
}
