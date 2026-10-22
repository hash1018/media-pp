//! The server against FFmpeg's own RTMP client — `RtmpMuxer` — on this
//! machine's loopback alone.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;

use super::*;
use crate::driver::DriverRunner;
use crate::elements::{
    AccessRequest, BroadcastItem, BroadcastOptions, Credentials, FileDemuxer, Pacer, RtmpMuxer,
};
use crate::pipeline::Pipeline;

/// A server on a port of the system's choice on the loopback, taking
/// publishers into one broadcast at `live/cam`.
fn server(options: RtmpServerOptions) -> (Arc<DriverRunner>, RtmpServerHandle, Broadcast) {
    let (server, handle) =
        RtmpServer::bind("server", "127.0.0.1:0", options).expect("a loopback port is free");
    let broadcast = Broadcast::new("cam", BroadcastOptions::default());
    handle.add_path("live/cam", &broadcast).unwrap();
    let runner = DriverRunner::new(server);
    runner.run().unwrap();
    (runner, handle, broadcast)
}

/// Pushes the fixture to `url` at its own speed, over and over, as an
/// encoder would — through FFmpeg's own RTMP client.
fn publish(url: &str, path: &str) -> crate::Result<Arc<Pipeline>> {
    let (demuxer, streams) = FileDemuxer::open("encoder", path)?;
    demuxer.looping_handle().set_looping(true);
    let mut muxer = RtmpMuxer::create(url)?;
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

/// Waits until `broadcast` has been published to `times` times.
fn wait_published(broadcast: &Broadcast, times: u64) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !(broadcast.is_live() && broadcast.stats().publications >= times) {
        assert!(
            Instant::now() < deadline,
            "never published: {:?}",
            broadcast.stats()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// What a reader of `broadcast` reads in `within`, from its newest keyframe:
/// each packet's track and whether it is a keyframe.
fn read(broadcast: &Broadcast, within: Duration) -> Vec<(usize, bool, i64)> {
    let mut reader = broadcast.subscribe();
    let deadline = Instant::now() + within;
    let mut read = Vec::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        if let Some(BroadcastItem::Packet { track, packet, .. }) = reader.next_timeout(left) {
            read.push((track, packet.is_key(), packet.pts().unwrap_or(0)));
        }
    }
    read
}

#[test]
fn a_publisher_is_taken_into_its_broadcast_as_it_sends() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(RtmpServerOptions::default());
    let publisher = publish(&handle.url("live/cam"), &path).unwrap();
    wait_published(&broadcast, 1);
    assert_eq!(handle.publishers(), 1);

    let tracks = broadcast.tracks().unwrap();
    assert_eq!(tracks.len(), 2, "the fixture's picture and sound");
    let video = &tracks[0];
    assert_eq!(video.kind(), ffmpeg::media::Type::Video);
    assert_eq!(video.parameters().id(), ffmpeg::codec::Id::H264);
    // SAFETY: plain fields of parameters this test holds a copy of.
    let (width, height, rate) = unsafe {
        let video = video.parameters().as_ptr();
        let audio = tracks[1].parameters().as_ptr();
        ((*video).width, (*video).height, (*audio).sample_rate)
    };
    assert!(
        width > 0 && height > 0,
        "a picture's size: {width}x{height}"
    );
    assert_eq!(tracks[1].parameters().id(), ffmpeg::codec::Id::AAC);
    assert!(rate > 0, "the sound's rate");

    let read = read(&broadcast, Duration::from_secs(2));
    let pictures: Vec<_> = read.iter().filter(|(track, ..)| *track == 0).collect();
    assert!(
        pictures.len() > 10,
        "pictures go on arriving: {}",
        pictures.len()
    );
    assert!(pictures[0].1, "a reader starts at a keyframe");
    assert!(read.iter().any(|(track, ..)| *track == 1), "and sound");

    // A second publisher of the path is refused; the first goes on.
    assert!(publish(&handle.url("live/cam"), &path).is_err());
    assert!(
        publish(&handle.url("live/nowhere"), &path).is_err(),
        "nor a path not served"
    );
    assert_eq!(handle.publishers(), 1);

    // It drops and comes back: published again into the same broadcast.
    publisher.stop();
    drop(publisher);
    let deadline = Instant::now() + Duration::from_secs(10);
    while handle.publishers() > 0 {
        assert!(Instant::now() < deadline, "the publisher left");
        std::thread::sleep(Duration::from_millis(20));
    }
    let again = publish(&handle.url("live/cam"), &path).unwrap();
    wait_published(&broadcast, 2);
    again.stop();
    runner.stop();
}

#[test]
fn a_publisher_is_authorized_by_what_its_stream_name_asks() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let authorize = Authorize::new(|request: &AccessRequest| {
        request.path == "live/cam"
            && request.credentials
                == Credentials::Basic {
                    user: "camera".into(),
                    password: "s3cret".into(),
                }
    });
    let (runner, handle, broadcast) = server(RtmpServerOptions {
        authorize: Some(authorize),
        ..RtmpServerOptions::default()
    });
    let url = handle.url("live/cam");
    assert!(publish(&url, &path).is_err(), "nothing presented");
    assert!(publish(&format!("{url}?user=camera&pass=wrong"), &path).is_err());
    assert_eq!(broadcast.stats().publications, 0);
    let publisher = publish(&format!("{url}?user=camera&pass=s3cret"), &path).unwrap();
    wait_published(&broadcast, 1);
    publisher.stop();
    runner.stop();
}

/// Stopping the server closes its publishers' connections and returns —
/// none of their threads is left reading.
#[test]
fn stopping_the_server_ends_its_publishers() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let (runner, handle, broadcast) = server(RtmpServerOptions::default());
    let publisher = publish(&handle.url("live/cam"), &path).unwrap();
    wait_published(&broadcast, 1);
    let began = Instant::now();
    runner.stop();
    while runner.bus().recv_timeout(Duration::from_secs(5)).is_ok() {}
    assert!(began.elapsed() < Duration::from_secs(5), "stopped at once");
    assert_eq!(handle.publishers(), 0);
    assert!(!broadcast.is_live(), "its run ended with it");
    publisher.stop();
}

#[test]
fn paths_are_checked_as_they_are_added() {
    let (_server, handle) =
        RtmpServer::bind("server", "127.0.0.1:0", RtmpServerOptions::default()).unwrap();
    let broadcast = Broadcast::new("b", BroadcastOptions::default());
    for path in ["cam", "live/", "/cam", "live//cam", "live/c am"] {
        assert!(
            matches!(
                handle.add_path(path, &broadcast),
                Err(RtmpServerError::InvalidPath(_))
            ),
            "{path:?}"
        );
    }
    handle.add_path("live/cam", &broadcast).unwrap();
    assert!(matches!(
        handle.add_path("live/cam", &broadcast),
        Err(RtmpServerError::PathTaken(_))
    ));
    assert!(handle.remove_path("live/cam"));
    assert!(!handle.remove_path("live/cam"));
}
