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

use super::annex_b::AnnexB;
use super::*;
use crate::driver::DriverRunner;
use crate::elements::{BroadcastOptions, FileDemuxer, Pacer, TrackFormat};
use crate::pipeline::Pipeline;

/// A server on the loopback serving `cam` for players and `live` for a
/// publisher.
fn server() -> (Arc<DriverRunner>, WebRtcServerHandle, Broadcast, Broadcast) {
    let (server, handle) =
        WebRtcServer::bind("server", "127.0.0.1:0", WebRtcServerOptions::default())
            .expect("a loopback port is free");
    let cam = Broadcast::new("cam", BroadcastOptions::default());
    let live = Broadcast::new("live", BroadcastOptions::default());
    handle.add_path("cam", &cam, WebRtcPathMode::Play).unwrap();
    handle
        .add_path("live", &live, WebRtcPathMode::Publish)
        .unwrap();
    let runner = DriverRunner::new(server);
    runner.run().unwrap();
    (runner, handle, cam, live)
}

/// Plays the fixture into `broadcast` at its own speed, over and over.
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

/// A WebRTC client of the test's own, as a browser would be one.
struct Client {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    events: Vec<Event>,
}

impl Client {
    fn new() -> Self {
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let local = socket.local_addr().unwrap();
        let mut rtc = Rtc::builder().build(Instant::now());
        rtc.add_local_candidate(Candidate::host(local, "udp").unwrap());
        Self {
            rtc,
            socket,
            local,
            events: Vec::new(),
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
                    Output::Event(event) => self.events.push(event),
                }
            }
            self.socket
                .set_read_timeout(Some(wait.max(Duration::from_millis(1))))
                .unwrap();
            match self.socket.recv_from(&mut buffer) {
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
        self.events
            .iter()
            .any(|event| matches!(event, Event::Connected))
    }

    /// What arrived of `mid`: whether each was a keyframe.
    fn received(&self, mid: Mid) -> Vec<bool> {
        self.events
            .iter()
            .filter_map(|event| match event {
                Event::MediaData(data) if data.mid == mid => Some(data.is_keyframe()),
                _ => None,
            })
            .collect()
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
        .find(|s| s.kind == ffmpeg::media::Type::Video)
        .unwrap();
    let annex_b = AnnexB::for_track(super::session::extradata(&video.parameters), false);
    let mut input = ffmpeg::format::input(&path).unwrap();
    let pictures: Vec<(Vec<u8>, i64)> = input
        .packets()
        .filter(|(stream, _)| stream.index() == video.index)
        .map(|(_, packet)| {
            let ts = packet
                .pts()
                .unwrap()
                .rescale(video.time_base, ffmpeg::Rational(1, 90_000));
            (
                annex_b
                    .convert(packet.data().unwrap(), packet.is_key())
                    .unwrap(),
                ts,
            )
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
    assert_eq!(tracks[0].parameters.id(), ffmpeg::codec::Id::H264);
    // SAFETY: plain fields of parameters held here.
    let (width, height) = unsafe {
        (
            (*tracks[0].parameters.as_ptr()).width,
            (*tracks[0].parameters.as_ptr()).height,
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
