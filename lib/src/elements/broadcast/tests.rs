use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;

use super::*;
use crate::buffer::MediaBuffer;
use crate::bus::BusEvent;
use crate::control::ControlMsg;
use crate::element::{RawSink, RawSinkExt};
use crate::elements::{FileDemuxer, FileMuxer, Mp4Layout, PacketCounter};
use crate::error::Error;
use crate::pipeline::Pipeline;
use crate::stream::StreamEvent;

const VIDEO_UNIT: ffmpeg::Rational = ffmpeg::Rational(1, 90_000);
const AUDIO_UNIT: ffmpeg::Rational = ffmpeg::Rational(1, 48_000);
/// A picture at 25 a second, in [`VIDEO_UNIT`].
const PICTURE: i64 = 3_600;
/// A sound packet of 1024 samples, in [`AUDIO_UNIT`].
const SOUND: i64 = 1_024;

fn parameters(medium: ffmpeg::media::Type, codec: ffmpeg::codec::Id) -> ffmpeg::codec::Parameters {
    let mut parameters = ffmpeg::codec::Parameters::new();
    // SAFETY: parameters this test just made and owns alone; both are plain
    // fields of `AVCodecParameters`.
    unsafe {
        (*parameters.as_mut_ptr()).codec_type = medium.into();
        (*parameters.as_mut_ptr()).codec_id = codec.into();
    }
    parameters
}

/// An H.264 picture track and an AAC sound track, in that order.
fn picture_and_sound() -> Vec<StreamInfo> {
    vec![
        StreamInfo::new(
            parameters(ffmpeg::media::Type::Video, ffmpeg::codec::Id::H264),
            VIDEO_UNIT,
        ),
        StreamInfo::new(
            parameters(ffmpeg::media::Type::Audio, ffmpeg::codec::Id::AAC),
            AUDIO_UNIT,
        ),
    ]
}

fn packet(ts: i64, duration: i64, unit: ffmpeg::Rational, key: bool, size: usize) -> MediaBuffer {
    let mut packet = ffmpeg::Packet::new(size);
    packet.set_pts(Some(ts));
    packet.set_dts(Some(ts));
    packet.set_duration(duration);
    packet.set_time_base(unit);
    if key {
        packet.set_flags(ffmpeg::packet::Flags::KEY);
    }
    MediaBuffer::packet(packet)
}

/// Picture `n`, a keyframe every `gop`.
fn picture(n: i64, gop: i64) -> MediaBuffer {
    packet(n * PICTURE, PICTURE, VIDEO_UNIT, n % gop == 0, 100)
}

fn sound(n: i64) -> MediaBuffer {
    packet(n * SOUND, SOUND, AUDIO_UNIT, true, 10)
}

/// Writes `seconds` of pictures from picture `from`, a keyframe a second,
/// with sound alongside.
fn write_seconds(sinks: &mut [BroadcastSink], from: i64, seconds: i64) {
    let pictures = 25 * seconds;
    for n in from..from + pictures {
        sinks[0]
            .consume(picture(n, 25))
            .expect("a picture is taken");
        // About as much sound as picture, interleaved.
        let sound_from = n * PICTURE * 48_000 / 90_000 / SOUND;
        let sound_to = (n + 1) * PICTURE * 48_000 / 90_000 / SOUND;
        for s in sound_from..sound_to {
            sinks[1].consume(sound(s)).expect("a sound is taken");
        }
    }
}

fn options(retain: Duration) -> BroadcastOptions {
    BroadcastOptions { retain }
}

/// What a reader reads until it is up to date.
fn read_all(reader: &mut BroadcastReader) -> Vec<BroadcastItem> {
    std::iter::from_fn(|| reader.try_next()).collect()
}

fn pictures_read(items: &[BroadcastItem]) -> Vec<(i64, bool)> {
    items
        .iter()
        .filter_map(|item| match item {
            BroadcastItem::Packet {
                track: 0, packet, ..
            } => Some((packet.pts().unwrap(), packet.is_key())),
            _ => None,
        })
        .collect()
}

fn is_broadcast_error(error: &Error, want: fn(&BroadcastError) -> bool) -> bool {
    matches!(error, Error::BroadcastError(inner) if want(inner))
}

#[test]
fn the_tracks_are_what_was_published_each_at_its_place() {
    let broadcast = Broadcast::new("tracks", options(Duration::from_secs(10)));
    let rate = ffmpeg::Rational::new(25, 1);
    let mut published = picture_and_sound();
    published[0] = published[0].clone().with_frame_rate(rate);
    // Given out of place: the broadcast numbers its own tracks.
    published[1] = published[1].clone().with_index(7);
    broadcast.publish(&published).unwrap();

    let tracks = broadcast.tracks().expect("published");
    assert_eq!(
        tracks[0].frame_rate(),
        Some(rate),
        "the rate goes on, once dropped"
    );
    assert_eq!(tracks[1].frame_rate(), None);
    assert_eq!(
        tracks.iter().map(StreamInfo::index).collect::<Vec<_>>(),
        [0, 1],
        "each at its place among them"
    );
    assert_eq!(
        tracks.iter().map(StreamInfo::kind).collect::<Vec<_>>(),
        [ffmpeg::media::Type::Video, ffmpeg::media::Type::Audio]
    );
}

#[test]
fn a_reader_joining_starts_at_the_newest_keyframe() {
    let broadcast = Broadcast::new("join", options(Duration::from_secs(10)));
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    write_seconds(&mut sinks, 0, 3);
    // Part of the fourth second, past its keyframe.
    for n in 75..80 {
        sinks[0].consume(picture(n, 25)).unwrap();
    }

    let mut reader = broadcast.subscribe();
    let items = read_all(&mut reader);
    assert!(matches!(items.first(), Some(BroadcastItem::Started(tracks)) if tracks.len() == 2));
    let pictures = pictures_read(&items);
    assert_eq!(
        pictures.first(),
        Some(&(75 * PICTURE, true)),
        "it starts at the newest keyframe"
    );
    assert_eq!(pictures.len(), 5, "and reads everything after it");
    assert!(
        matches!(items[1], BroadcastItem::Packet { track: 0, .. }),
        "nothing of another track comes before the keyframe it starts at"
    );
}

/// What is kept is the newest `retain` and nothing more, packet by packet:
/// no group of pictures is held whole for a reader to start at.
#[test]
fn what_is_kept_is_the_newest_retain_packet_by_packet() {
    for retain in [
        Duration::from_millis(500),
        Duration::from_secs(1),
        Duration::from_millis(2_500),
    ] {
        let broadcast = Broadcast::new("retain", options(retain));
        let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
        write_seconds(&mut sinks, 0, 10);

        let state = broadcast.shared.state.lock().unwrap();
        let newest = state.newest.unwrap();
        let span = newest - state.entries.front().unwrap().at;
        assert!(
            span <= retain.as_secs_f64(),
            "{retain:?}: what is kept spans {span}s, more than asked"
        );
        // A picture's time less than asked: no more could have been kept.
        assert!(
            span > retain.as_secs_f64() - 0.05,
            "{retain:?}: what is kept spans {span}s, less than asked"
        );
    }
}

/// With no keyframe in what is kept, a reader joining waits for the next
/// one written, and starts there.
#[test]
fn a_reader_joining_with_no_keyframe_held_starts_at_the_next() {
    // A keyframe a second, and half a second kept.
    let broadcast = Broadcast::new("join-next", options(Duration::from_millis(500)));
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    write_seconds(&mut sinks, 0, 1);
    for n in 25..45 {
        sinks[0].consume(picture(n, 25)).unwrap();
    }
    {
        let state = broadcast.shared.state.lock().unwrap();
        assert!(state.starts.is_empty(), "the keyframe at 25 is let go of");
    }

    let mut reader = broadcast.subscribe();
    let items = read_all(&mut reader);
    assert!(matches!(items.first(), Some(BroadcastItem::Started(_))));
    assert!(
        pictures_read(&items).is_empty(),
        "nothing until a keyframe is written"
    );
    for n in 45..55 {
        sinks[0].consume(picture(n, 25)).unwrap();
    }
    let pictures = pictures_read(&read_all(&mut reader));
    assert_eq!(
        pictures.first(),
        Some(&(50 * PICTURE, true)),
        "{pictures:?}"
    );
    assert_eq!(pictures.len(), 5, "and everything after it");
}

#[test]
fn a_reader_that_falls_behind_is_moved_on_to_a_keyframe_and_the_writer_never_waits() {
    let broadcast = Broadcast::new("behind", options(Duration::from_secs(2)));
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    write_seconds(&mut sinks, 0, 1);
    assert!(matches!(reader.try_next(), Some(BroadcastItem::Started(_))));
    assert!(matches!(
        reader.try_next(),
        Some(BroadcastItem::Packet { track: 0, .. })
    ));

    // Far more than it keeps, while the reader reads nothing: the writer
    // goes on regardless.
    let began = Instant::now();
    write_seconds(&mut sinks, 25, 30);
    assert!(
        began.elapsed() < Duration::from_secs(5),
        "the writer waited for its reader"
    );

    assert!(matches!(reader.try_next(), Some(BroadcastItem::Skipped)));
    match reader.try_next() {
        Some(BroadcastItem::Packet {
            track: 0, packet, ..
        }) => {
            assert!(packet.is_key(), "it goes on from a keyframe");
            assert_eq!(packet.pts(), Some(30 * 25 * PICTURE), "the newest one");
        }
        other => panic!("expected the keyframe after skipping, got {other:?}"),
    }
}

#[test]
fn sound_before_the_first_keyframe_is_passed_over() {
    let broadcast = Broadcast::new("sound-first", BroadcastOptions::default());
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    sinks[1].consume(sound(0)).unwrap();
    sinks[1].consume(sound(1)).unwrap();
    // A picture that is not a keyframe: a stream joined part way.
    sinks[0].consume(picture(1, 25)).unwrap();
    sinks[0].consume(picture(25, 25)).unwrap();
    sinks[1].consume(sound(2)).unwrap();

    let items = read_all(&mut reader);
    assert!(matches!(items[0], BroadcastItem::Started(_)));
    match &items[1] {
        BroadcastItem::Packet {
            track: 0, packet, ..
        } => assert!(packet.is_key()),
        other => panic!("expected the keyframe first, got {other:?}"),
    }
    assert!(matches!(items[2], BroadcastItem::Packet { track: 1, .. }));
    assert_eq!(items.len(), 3);
}

/// Tracks written from queues of their own reach the broadcast out of
/// step: what another track has of the keyframe's moment on is handed on
/// even where it was written before the keyframe.
#[test]
fn another_tracks_packets_written_ahead_of_the_keyframe_are_kept() {
    let broadcast = Broadcast::new("ahead", options(Duration::from_secs(60)));
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    for joining in [true, false] {
        let mut reader = joining.then(|| broadcast.subscribe());
        let base = if joining { 0 } else { 1_000 };
        // A second of sound ahead of the pictures, from half a second
        // before the keyframe.
        let key_at = (base + 25) * PICTURE;
        let first_sound = (key_at - PICTURE * 12) * 48_000 / 90_000 / SOUND;
        for n in first_sound..first_sound + 70 {
            sinks[1].consume(sound(n)).unwrap();
        }
        for n in base + 20..base + 30 {
            sinks[0].consume(picture(n, 25)).unwrap();
        }
        let mut reader = reader.take().unwrap_or_else(|| broadcast.subscribe());
        let items = read_all(&mut reader);
        let packets: Vec<(usize, i64)> = items
            .iter()
            .filter_map(|item| match item {
                BroadcastItem::Packet { track, packet, .. } => {
                    Some((*track, packet.pts().unwrap()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            packets[0],
            (0, key_at),
            "joining={joining}: the keyframe comes first"
        );
        let key_seconds = key_at as f64 / 90_000.0;
        let sounds: Vec<f64> = packets
            .iter()
            .filter(|(track, _)| *track == 1)
            .map(|(_, pts)| *pts as f64 / 48_000.0)
            .collect();
        assert!(
            !sounds.is_empty(),
            "joining={joining}: the sound written ahead is kept"
        );
        assert!(
            sounds.iter().all(|at| *at >= key_seconds),
            "joining={joining}: nothing from before the keyframe"
        );
        let expected = (first_sound..first_sound + 70)
            .filter(|n| (n * SOUND) as f64 / 48_000.0 >= key_seconds)
            .count();
        assert_eq!(sounds.len(), expected, "joining={joining}: all of it");
        assert!(sounds.windows(2).all(|pair| pair[0] < pair[1]), "in order");
    }
}

#[test]
fn publishing_again_ends_the_run_and_refuses_the_old_sinks() {
    let broadcast = Broadcast::new("again", BroadcastOptions::default());
    let mut old = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    write_seconds(&mut old, 0, 1);
    read_all(&mut reader);

    let mut new = broadcast.publish(&picture_and_sound()[..1]).unwrap();
    let error = old[0].consume(picture(25, 25)).unwrap_err();
    assert!(is_broadcast_error(&error, |e| matches!(
        e,
        BroadcastError::Superseded(_)
    )));
    // Dropping a superseded sink does not end the new run.
    drop(old);
    assert!(broadcast.is_live());

    new[0].consume(picture(0, 25)).unwrap();
    let items = read_all(&mut reader);
    assert!(matches!(items[0], BroadcastItem::Ended), "{items:?}");
    assert!(matches!(&items[1], BroadcastItem::Started(tracks) if tracks.len() == 1));
    assert!(matches!(items[2], BroadcastItem::Packet { track: 0, .. }));
    assert_eq!(broadcast.tracks().unwrap().len(), 1);
}

#[test]
fn the_end_of_every_track_ends_the_run() {
    let broadcast = Broadcast::new("end", BroadcastOptions::default());
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    write_seconds(&mut sinks, 0, 1);

    sinks[0].stream_event(&StreamEvent::Eos).unwrap();
    assert!(broadcast.is_live(), "one track is still going");
    sinks[1].stream_event(&StreamEvent::Eos).unwrap();
    assert!(!broadcast.is_live());

    let items = read_all(&mut reader);
    assert!(
        matches!(items.last(), Some(BroadcastItem::Ended)),
        "{items:?}"
    );
    assert!(
        pictures_read(&items).len() == 25,
        "everything written is read before the end"
    );

    // A stop and a drop end a run alike.
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    sinks[0].control(&ControlMsg::Stop).unwrap();
    assert!(broadcast.is_live());
    drop(sinks);
    assert!(!broadcast.is_live());
}

#[test]
fn closing_tells_readers_after_what_it_holds_and_refuses_sinks() {
    let broadcast = Broadcast::new("close", BroadcastOptions::default());
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    write_seconds(&mut sinks, 0, 1);

    let handle = broadcast.clone();
    drop(broadcast);
    assert!(!handle.is_closed(), "a clone keeps it open");
    drop(handle);

    let items = read_all(&mut reader);
    assert_eq!(
        pictures_read(&items).len(),
        25,
        "what it held is read first"
    );
    assert!(matches!(items.last(), Some(BroadcastItem::Closed)));
    assert!(reader.try_next().is_none());
    assert!(reader.next_timeout(Duration::from_millis(10)).is_none());

    let error = sinks[0].consume(picture(25, 25)).unwrap_err();
    assert!(is_broadcast_error(&error, |e| matches!(
        e,
        BroadcastError::Closed(_)
    )));
}

#[test]
fn a_flush_begins_a_run_of_the_same_tracks() {
    let broadcast = Broadcast::new("flush", BroadcastOptions::default());
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    write_seconds(&mut sinks, 0, 1);
    read_all(&mut reader);

    sinks[0].control(&ControlMsg::Flush).unwrap();
    // The sound track has not seen the seek yet: what it writes now is from
    // before it.
    sinks[1].consume(sound(500)).unwrap();
    sinks[0].consume(picture(100, 25)).unwrap();
    sinks[1].control(&ControlMsg::Flush).unwrap();
    sinks[1].consume(sound(10)).unwrap();

    let items = read_all(&mut reader);
    assert!(matches!(items[0], BroadcastItem::Ended), "{items:?}");
    assert!(matches!(items[1], BroadcastItem::Started(_)));
    match &items[2..] {
        [
            BroadcastItem::Packet { track: 0, .. },
            BroadcastItem::Packet {
                track: 1, packet, ..
            },
        ] => assert_eq!(
            packet.pts(),
            Some(10 * SOUND),
            "what was left from before the seek is dropped"
        ),
        other => panic!("unexpected {other:?}"),
    }

    // Sought back from the end: the tracks go on.
    for sink in &mut sinks {
        sink.stream_event(&StreamEvent::Eos).unwrap();
    }
    assert!(!broadcast.is_live());
    for sink in &mut sinks {
        sink.control(&ControlMsg::Flush).unwrap();
    }
    assert!(broadcast.is_live());
    sinks[0].consume(picture(0, 25)).unwrap();
    let items = read_all(&mut reader);
    assert!(
        matches!(
            items[..],
            [
                BroadcastItem::Ended,
                BroadcastItem::Started(_),
                BroadcastItem::Packet { .. }
            ]
        ),
        "{items:?}"
    );
}

#[test]
fn the_bound_lets_go_of_the_oldest_without_keyframes() {
    let broadcast = Broadcast::new("bound", options(Duration::from_secs(3600)));
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    sinks[0]
        .consume(packet(0, PICTURE, VIDEO_UNIT, true, 64 * 1024))
        .unwrap();
    for n in 1..160 {
        sinks[0]
            .consume(packet(n * PICTURE, PICTURE, VIDEO_UNIT, false, 64 * 1024))
            .unwrap();
    }
    {
        let state = broadcast.shared.state.lock().unwrap();
        assert!(
            state.bytes <= state::MAX_BYTES,
            "it holds {} bytes",
            state.bytes
        );
        assert!(state.starts.is_empty(), "the keyframe went with the oldest");
    }
    // A reader joining now waits for the next keyframe.
    let mut reader = broadcast.subscribe();
    assert!(matches!(reader.try_next(), Some(BroadcastItem::Started(_))));
    assert!(reader.try_next().is_none());
    sinks[0]
        .consume(packet(160 * PICTURE, PICTURE, VIDEO_UNIT, false, 10))
        .unwrap();
    assert!(reader.try_next().is_none());
    sinks[0]
        .consume(packet(161 * PICTURE, PICTURE, VIDEO_UNIT, true, 10))
        .unwrap();
    assert!(
        matches!(reader.try_next(), Some(BroadcastItem::Packet { track: 0, packet, .. }) if packet.is_key())
    );
}

#[test]
fn a_sink_refuses_what_is_not_a_packet() {
    let broadcast = Broadcast::new("decoded", BroadcastOptions::default());
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let frame = ffmpeg::frame::Audio::new(
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
        16,
        ffmpeg::ChannelLayout::MONO,
    );
    let error = sinks[1].consume(MediaBuffer::audio(frame)).unwrap_err();
    assert!(is_broadcast_error(&error, |e| matches!(
        e,
        BroadcastError::UnsupportedBuffer("Audio")
    )));
    // It went nowhere, and the sink still works.
    sinks[1].consume(sound(0)).unwrap();
    assert_eq!(broadcast.shared.state.lock().unwrap().entries.len(), 1);
}

#[test]
fn publishing_refuses_what_cannot_be_carried_and_changes_nothing() {
    let broadcast = Broadcast::new("refused", BroadcastOptions::default());
    assert!(matches!(
        broadcast.publish(&[]),
        Err(BroadcastError::NoTracks)
    ));
    let mut tracks = picture_and_sound();
    tracks.push(StreamInfo::new(
        parameters(ffmpeg::media::Type::Data, ffmpeg::codec::Id::None),
        VIDEO_UNIT,
    ));
    assert!(matches!(
        broadcast.publish(&tracks),
        Err(BroadcastError::UnsupportedTrack { index: 2 })
    ));
    assert!(broadcast.tracks().is_none());
    assert!(!broadcast.is_live());
    broadcast.close();
    assert!(matches!(
        broadcast.publish(&picture_and_sound()),
        Err(BroadcastError::Closed(_))
    ));
}

#[test]
fn readers_are_counted() {
    let broadcast = Broadcast::new("count", BroadcastOptions::default());
    let one = broadcast.subscribe();
    let two = broadcast.subscribe();
    assert_eq!(broadcast.subscribers(), 2);
    drop(one);
    assert_eq!(broadcast.subscribers(), 1);
    drop(two);
    assert_eq!(broadcast.subscribers(), 0);
}

struct CountWakes(AtomicUsize);

impl Wake for CountWakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn a_waiting_poll_is_woken_by_a_write() {
    let broadcast = Broadcast::new("poll", BroadcastOptions::default());
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    let wakes = Arc::new(CountWakes(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wakes));
    let mut cx = Context::from_waker(&waker);

    assert!(matches!(
        reader.poll_next(&mut cx),
        Poll::Ready(Some(BroadcastItem::Started(_)))
    ));
    assert!(reader.poll_next(&mut cx).is_pending());
    assert!(reader.poll_next(&mut cx).is_pending());
    sinks[0].consume(picture(0, 25)).unwrap();
    assert_eq!(
        wakes.0.load(Ordering::SeqCst),
        1,
        "woken once, however often it was polled"
    );
    assert!(matches!(
        reader.poll_next(&mut cx),
        Poll::Ready(Some(BroadcastItem::Packet { .. }))
    ));

    assert!(reader.poll_next(&mut cx).is_pending());
    broadcast.close();
    assert_eq!(wakes.0.load(Ordering::SeqCst), 2, "closing wakes it too");
    assert!(matches!(
        reader.poll_next(&mut cx),
        Poll::Ready(Some(BroadcastItem::Closed))
    ));
    assert!(matches!(reader.poll_next(&mut cx), Poll::Ready(None)));
}

#[test]
fn next_timeout_waits_for_a_write_from_another_thread() {
    let broadcast = Broadcast::new("timeout", BroadcastOptions::default());
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    assert!(matches!(reader.try_next(), Some(BroadcastItem::Started(_))));
    let began = Instant::now();
    assert!(reader.next_timeout(Duration::from_millis(50)).is_none());
    assert!(began.elapsed() >= Duration::from_millis(50));

    let writer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        sinks[0].consume(picture(0, 25)).unwrap();
        sinks
    });
    assert!(matches!(
        reader.next_timeout(Duration::from_secs(10)),
        Some(BroadcastItem::Packet { track: 0, .. })
    ));
    drop(writer.join());
}

#[test]
fn a_broadcast_source_needs_tracks() {
    let broadcast = Broadcast::new("unpublished", BroadcastOptions::default());
    assert!(matches!(
        BroadcastSource::new("source", &broadcast),
        Err(BroadcastError::NotPublished(_))
    ));
    broadcast.close();
    assert!(matches!(
        BroadcastSource::new("source", &broadcast),
        Err(BroadcastError::Closed(_))
    ));
    assert_eq!(
        broadcast.subscribers(),
        0,
        "a source refused leaves no reader behind"
    );
}

/// Plays the fixture into `broadcast` as fast as it reads, and waits for
/// the end of it.
fn play_into(broadcast: &Broadcast, path: &str) {
    let (demuxer, streams) = FileDemuxer::open("demuxer", path).expect("the fixture opens");
    let tracks: Vec<StreamInfo> = streams.iter().map(StreamInfo::from).collect();
    let sinks = broadcast
        .publish(&tracks)
        .expect("the fixture's tracks are carried");
    let (input, ()) = Pipeline::new("input", demuxer, |source, ctx| {
        for (stream, sink) in streams.iter().zip(sinks) {
            let branch = ctx.branch().queue("packets", 64).to(sink)?;
            ctx.attach(source, stream.index(), branch)?;
        }
        Ok(())
    })
    .expect("the input is wired");
    input.run().expect("the input runs");
    wait_for_finished(&input);
    input.stop();
}

fn wait_for_finished(pipeline: &Pipeline) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match pipeline.bus().recv_timeout(left) {
            Ok(BusEvent::Finished) => return,
            Ok(BusEvent::Error { error, .. }) => panic!("the pipeline failed: {error}"),
            Ok(_) => {}
            Err(_) => panic!("the pipeline did not finish in time"),
        }
    }
}

/// Every packet of a file, by stream, with each stream's unit.
fn read_file(
    path: &std::path::Path,
) -> Vec<(ffmpeg::media::Type, ffmpeg::Rational, Vec<ffmpeg::Packet>)> {
    let mut input = ffmpeg::format::input(&path).expect("the recording opens");
    let mut streams: Vec<_> = input
        .streams()
        .map(|stream| (stream.parameters().medium(), stream.time_base(), Vec::new()))
        .collect();
    for (stream, packet) in input.packets() {
        streams[stream.index()].2.push(packet);
    }
    streams
}

/// The contract a recording relies on: a `BroadcastSource` starts at a
/// keyframe, at zero, with every track, and puts a second run after the
/// first on the one timeline rather than starting it over.
#[test]
fn a_recording_through_a_broadcast_source_starts_at_a_keyframe_and_goes_on_across_runs() {
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    // Long enough that the unpaced input never leaves the recording behind.
    let broadcast = Broadcast::new("camera", options(Duration::from_secs(60)));
    let (_, streams) = FileDemuxer::open("probe", &path).unwrap();
    let tracks: Vec<StreamInfo> = streams.iter().map(StreamInfo::from).collect();
    drop(broadcast.publish(&tracks).unwrap());

    let (source, streams) = BroadcastSource::new("recorder", &broadcast).unwrap();
    let out = std::env::temp_dir().join(format!("broadcast_recording_{}.mp4", std::process::id()));
    let mut muxer = FileMuxer::create(&out, Mp4Layout::Standard).unwrap();
    let ids: Vec<_> = streams
        .iter()
        .map(|stream| {
            muxer
                .add_stream(format!("track-{}", stream.index()), stream)
                .unwrap()
        })
        .collect();
    let mut files = muxer.open().unwrap();
    let ends: Vec<_> = ids.into_iter().map(|id| files.take(id).unwrap()).collect();
    let (recording, ()) = Pipeline::new("recording", source, |source, ctx| {
        for (index, end) in ends.into_iter().enumerate() {
            let branch = ctx.branch().queue("recorded", 64).to(end)?;
            ctx.attach(source, index, branch)?;
        }
        Ok(())
    })
    .unwrap();
    recording.run().unwrap();

    play_into(&broadcast, &path);
    let gap = Duration::from_millis(300);
    std::thread::sleep(gap);
    play_into(&broadcast, &path);
    broadcast.close();
    wait_for_finished(&recording);
    recording.stop();

    let fixture = read_file(std::path::Path::new(&path));
    let recorded = read_file(&out);
    let _ = std::fs::remove_file(&out);
    assert_eq!(recorded.len(), fixture.len(), "every track is recorded");
    let seconds = |unit: ffmpeg::Rational, ts: i64| ts as f64 * f64::from(unit);
    let fixture_length = fixture
        .iter()
        .find(|(medium, ..)| *medium == ffmpeg::media::Type::Video)
        .map(|(_, unit, packets)| seconds(*unit, packets.last().unwrap().dts().unwrap()))
        .unwrap();
    for ((medium, unit, packets), (_, _, played)) in recorded.iter().zip(&fixture) {
        assert!(!packets.is_empty(), "{medium:?} has packets");
        let first = seconds(*unit, packets[0].dts().unwrap());
        assert!(
            (-0.001..0.1).contains(&first),
            "{medium:?} begins at {first}s, not at zero"
        );
        let mut last = i64::MIN;
        for packet in packets {
            let dts = packet.dts().unwrap();
            assert!(dts > last, "{medium:?} goes back from {last} to {dts}");
            last = dts;
        }
        if *medium == ffmpeg::media::Type::Video {
            assert!(packets[0].is_key(), "the recording begins at a keyframe");
            assert_eq!(
                packets.len(),
                2 * played.len(),
                "both runs, every picture of each"
            );
            // The second run goes on after the first, at least as far
            // after it as the gap between them.
            let second = &packets[played.len()];
            assert!(second.is_key());
            let at = seconds(*unit, second.dts().unwrap());
            assert!(
                at >= fixture_length + gap.as_secs_f64() * 0.9,
                "the second run begins at {at}s, inside the first ({fixture_length}s) and its gap"
            );
        }
    }
}

/// A source waiting for packets is waiting on its pipeline too: stopping it
/// takes no longer than any other source.
#[test]
fn a_broadcast_source_waiting_for_packets_stops_at_once() {
    let broadcast = Broadcast::new("quiet", BroadcastOptions::default());
    let _sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let (source, _) = BroadcastSource::new("waiting", &broadcast).unwrap();
    let (counter, count) = PacketCounter::new("counter");
    let (pipeline, ()) = Pipeline::new("quiet", source, |source, ctx| {
        let branch = ctx.branch().to(counter)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let began = Instant::now();
    pipeline.stop();
    assert!(
        began.elapsed() < Duration::from_secs(2),
        "stopping took {:?}",
        began.elapsed()
    );
    assert_eq!(count.get(), 0);
    drop(pipeline);
    assert_eq!(
        broadcast.subscribers(),
        0,
        "the source's reader went with it"
    );
}

/// What a source hands on goes on through the pipeline's own end: `finish`
/// ends a recording as closing the broadcast does.
#[test]
fn a_broadcast_source_hands_on_what_is_written_while_it_runs() {
    let broadcast = Broadcast::new("live", options(Duration::from_secs(5)));
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let (source, _) = BroadcastSource::new("relay", &broadcast).unwrap();
    let (pictures, pictures_count) = PacketCounter::new("pictures");
    let (sounds, sounds_count) = PacketCounter::new("sounds");
    let (pipeline, ()) = Pipeline::new("live", source, |source, ctx| {
        let branch = ctx.branch().to(pictures)?;
        ctx.attach(source, 0, branch)?;
        let branch = ctx.branch().to(sounds)?;
        ctx.attach(source, 1, branch)?;
        Ok(())
    })
    .unwrap();
    pipeline.run().unwrap();
    write_seconds(&mut sinks, 0, 2);
    let deadline = Instant::now() + Duration::from_secs(10);
    while pictures_count.get() < 50 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    pipeline.finish();
    assert_eq!(pictures_count.get(), 50);
    assert!(sounds_count.get() > 0);
}

/// Driven, a broadcast source waits on its reader alone: it asks for no
/// time of its own, a write wakes the poll it handed the waker to, and
/// closing the broadcast ends it.
#[test]
fn a_driven_broadcast_source_is_woken_by_writes_and_ends_with_the_broadcast() {
    use crate::pipeline::{Driven, DrivenPipeline};

    let broadcast = Broadcast::new("driven", options(Duration::from_secs(5)));
    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let (source, _) = BroadcastSource::new("relay", &broadcast).unwrap();
    let (pictures, pictures_count) = PacketCounter::new("pictures");
    let (sounds, sounds_count) = PacketCounter::new("sounds");
    let (mut pipeline, ()) = DrivenPipeline::new("driven", source, |source, ctx| {
        ctx.attach(source, 0, ctx.branch().to(pictures)?)?;
        ctx.attach(source, 1, ctx.branch().to(sounds)?)?;
        Ok(())
    })
    .unwrap();
    let wakes = Arc::new(CountWakes(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wakes));
    let mut cx = Context::from_waker(&waker);

    assert_eq!(pipeline.poll(&mut cx), Driven::Pending { at: None });
    write_seconds(&mut sinks, 0, 2);
    assert!(wakes.0.load(Ordering::SeqCst) >= 1, "a write wakes it");
    let mut polls = 0;
    loop {
        polls += 1;
        assert!(polls < 100, "it kept asking for the thread");
        match pipeline.poll(&mut cx) {
            Driven::Pending { at: None } => break,
            Driven::Pending { at: Some(at) } => assert!(at <= Instant::now()),
            Driven::Finished => panic!("finished with the broadcast open"),
        }
    }
    assert_eq!(pictures_count.get(), 50);
    assert!(sounds_count.get() > 0);

    let woken = wakes.0.load(Ordering::SeqCst);
    broadcast.close();
    assert!(wakes.0.load(Ordering::SeqCst) > woken, "closing wakes it");
    assert_eq!(pipeline.poll(&mut cx), Driven::Finished);
    drop(pipeline);
    assert_eq!(broadcast.subscribers(), 0);
}

#[test]
fn stats_say_what_it_was_written_who_reads_it_and_who_fell_behind() {
    let broadcast = Broadcast::new("stats", options(Duration::from_secs(2)));
    let before = broadcast.stats();
    assert!(!before.live);
    assert_eq!(
        (
            before.publications,
            before.bytes,
            before.packets,
            before.readers
        ),
        (0, 0, 0, 0)
    );
    assert_eq!((before.live_since, before.last_written), (None, None));

    let mut sinks = broadcast.publish(&picture_and_sound()).unwrap();
    let mut reader = broadcast.subscribe();
    assert!(matches!(reader.try_next(), Some(BroadcastItem::Started(_))));
    // A second: 25 pictures of 100 bytes, and the 46 sounds of 10 bytes
    // written beside them.
    write_seconds(&mut sinks, 0, 1);
    let live = broadcast.stats();
    assert!(live.live && live.live_since.is_some() && live.last_written.is_some());
    assert_eq!((live.publications, live.readers), (1, 1));
    assert_eq!((live.packets, live.bytes), (25 + 46, 25 * 100 + 46 * 10));
    assert_eq!(live.fell_behind, 0);

    // Far more than it keeps while the reader reads nothing.
    write_seconds(&mut sinks, 25, 30);
    assert!(matches!(reader.try_next(), Some(BroadcastItem::Skipped)));
    let behind = broadcast.stats();
    assert_eq!(behind.fell_behind, 1);
    // 750 pictures, and the sounds from the 46th to the 1453rd.
    assert_eq!(behind.packets, live.packets + 750 + (1453 - 46));

    // An input coming back is published once more; the counts go on.
    let mut again = broadcast.publish(&picture_and_sound()).unwrap();
    write_seconds(&mut again, 0, 1);
    let after = broadcast.stats();
    assert_eq!(after.publications, 2);
    assert_eq!(after.packets, behind.packets + 25 + 46);
    drop(again);
    drop(sinks);
    let ended = broadcast.stats();
    assert!(!ended.live);
    assert_eq!(ended.live_since, None, "nothing is live");
    assert!(ended.last_written.is_some(), "what was written stays known");
}
