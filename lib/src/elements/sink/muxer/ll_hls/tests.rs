use std::sync::Arc;
use std::time::Duration;

use ffmpeg_next as ffmpeg;

use super::*;
use crate::element::BoxSink;
use crate::test_support::block_on;

/// A file's streams, and its packets in file order with the stream each is of.
type Demuxed = (Vec<StreamInfo>, Vec<(usize, ffmpeg::Packet)>);

/// The fixture's packets in file order, with the streams they belong to.
fn fixture() -> Option<Demuxed> {
    let path = crate::test_support::try_test_video()?;
    let mut input = ffmpeg::format::input(&path).expect("the fixture opens");
    let streams = input
        .streams()
        .map(|stream| StreamInfo::new(stream.parameters(), stream.time_base()))
        .collect();
    let packets = input
        .packets()
        .map(|(stream, packet)| (stream.index(), packet))
        .collect();
    Some((streams, packets))
}

/// A run of `stream` opened with `streams`: a sink each, in order.
fn open(stream: &LlHlsStream, streams: &[StreamInfo]) -> Vec<BoxSink> {
    let mut muxer = stream.muxer();
    let tracks: Vec<_> = streams
        .iter()
        .enumerate()
        .map(|(index, info)| muxer.add_stream(format!("t{index}"), info.clone()).unwrap())
        .collect();
    let mut sinks = muxer.open().unwrap();
    tracks
        .into_iter()
        .map(|track| sinks.take(track).unwrap())
        .collect()
}

fn feed(sinks: &mut [BoxSink], packets: &[(usize, ffmpeg::Packet)]) {
    for (index, packet) in packets {
        sinks[*index]
            .consume(MediaBuffer::Packet(Arc::new(packet.clone()).into()))
            .unwrap();
    }
}

fn end(sinks: &mut [BoxSink]) {
    for sink in sinks {
        sink.stream_event(&StreamEvent::Eos).unwrap();
    }
}

fn options(segments: usize) -> LlHlsOptions {
    LlHlsOptions {
        segments,
        ..LlHlsOptions::default()
    }
}

/// The value of `key` in a tag's attribute list.
fn attribute<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split(&[':', ','][..])
        .find_map(|field| field.strip_prefix(key)?.strip_prefix('='))
}

/// A stream written whole lists its parts no longer than the target and
/// its segments no shorter than theirs but at its end; its init section and
/// segments, read back in order, are every picture written, in order.
#[test]
fn a_stream_written_whole_reads_back_as_it_came() {
    let Some((streams, packets)) = fixture() else {
        return;
    };
    let stream = LlHlsStream::new("whole", options(1000)).unwrap();
    let mut sinks = open(&stream, &streams);
    feed(&mut sinks, &packets);
    end(&mut sinks);

    let playlist = block_on(stream.playlist(None)).unwrap();
    assert!(
        !playlist.contains("PRELOAD-HINT"),
        "the run ended: {playlist}"
    );
    let parts: Vec<f64> = playlist
        .lines()
        .filter(|line| line.starts_with("#EXT-X-PART:"))
        .map(|line| attribute(line, "DURATION").unwrap().parse().unwrap())
        .collect();
    assert!(parts.len() > 20, "{playlist}");
    assert!(
        parts.iter().all(|duration| *duration <= 0.2 + 1e-6),
        "a part longer than its target: {parts:?}"
    );
    let segments: Vec<f64> = playlist
        .lines()
        .filter_map(|line| line.strip_prefix("#EXTINF:"))
        .map(|line| line.trim_end_matches(',').parse().unwrap())
        .collect();
    assert!(segments.len() >= 4, "{playlist}");
    assert!(
        segments[..segments.len() - 1]
            .iter()
            .all(|duration| *duration >= 1.0 - 1e-3),
        "a segment ended before its keyframe: {segments:?}"
    );

    let mut whole = block_on(stream.file("init-0.mp4"))
        .expect("the init section")
        .to_vec();
    for msn in 0..segments.len() {
        whole.extend_from_slice(&block_on(stream.file(&format!("seg-{msn}.mp4"))).unwrap());
    }
    let path = std::env::temp_dir().join(format!("ll_hls_whole_{}.mp4", std::process::id()));
    std::fs::write(&path, &whole).unwrap();
    let mut input = ffmpeg::format::input(&path).expect("what was served reads back");
    let mut read = vec![0usize; streams.len()];
    let mut last = vec![i64::MIN; streams.len()];
    for (s, packet) in input.packets() {
        let index = s.index();
        read[index] += 1;
        let dts = packet.dts().unwrap();
        assert!(dts > last[index], "track {index} goes back at {dts}");
        last[index] = dts;
    }
    std::fs::remove_file(&path).ok();
    let video = streams
        .iter()
        .position(|info| info.kind() == ffmpeg::media::Type::Video)
        .unwrap();
    let written = packets.iter().filter(|(index, _)| *index == video).count();
    assert_eq!(read[video], written, "every picture");
    for (index, count) in read.iter().enumerate() {
        let fed = packets.iter().filter(|(i, _)| *i == index).count();
        assert!(*count + 3 >= fed, "track {index}: {count} of {fed}");
    }
}

/// A player asking for a part not yet made is answered once it is in, and
/// the part it is told is coming is handed out once it is; one asking too
/// far ahead is refused at once, and closing the stream ends every wait.
#[test]
fn a_player_waiting_for_a_part_is_answered_once_it_is_in() {
    let Some((streams, packets)) = fixture() else {
        return;
    };
    let stream = LlHlsStream::new("waits", options(7)).unwrap();
    assert_eq!(
        block_on(stream.playlist(Some(LlHlsPosition { msn: 3, part: None }))),
        Err(LlHlsPlaylistError::TooFarAhead)
    );
    let waiting = {
        let stream = stream.clone();
        std::thread::spawn(move || {
            block_on(stream.playlist(Some(LlHlsPosition {
                msn: 0,
                part: Some(2),
            })))
        })
    };
    assert!(
        block_on(stream.file("part-0-0.mp4")).is_none(),
        "nothing is coming before a run"
    );
    std::thread::sleep(Duration::from_millis(50));
    assert!(!waiting.is_finished(), "nothing is written yet");
    let mut sinks = open(&stream, &streams);
    let quarter = packets.len() / 4;
    feed(&mut sinks, &packets[..quarter]);
    let playlist = waiting.join().unwrap().unwrap();
    assert!(playlist.contains("part-0-2.mp4"), "{playlist}");
    // The first part begins at the run's first keyframe and keeps to its
    // target, though the muxer's first fragment writes the track list alone.
    let first = playlist
        .lines()
        .find(|line| line.contains("URI=\"part-0-0.mp4\""))
        .expect("the first part");
    assert!(first.ends_with(",INDEPENDENT=YES"), "{first}");
    let duration: f64 = attribute(first, "DURATION").unwrap().parse().unwrap();
    assert!(duration <= 0.2 + 1e-6, "{first}");

    // The part the playlist says is coming is handed out once it is in.
    let now = block_on(stream.playlist(None)).unwrap();
    let hint = now
        .lines()
        .find_map(|line| line.strip_prefix("#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\""))
        .and_then(|rest| rest.strip_suffix('"'))
        .expect("a part is coming")
        .to_owned();
    let part = {
        let stream = stream.clone();
        let hint = hint.clone();
        std::thread::spawn(move || block_on(stream.file(&hint)))
    };
    std::thread::sleep(Duration::from_millis(50));
    assert!(!part.is_finished(), "{hint} is not in yet");
    feed(&mut sinks, &packets[quarter..quarter * 2]);
    assert!(part.join().unwrap().is_some_and(|data| !data.is_empty()));
    assert!(
        block_on(stream.file("part-9-9.mp4")).is_none(),
        "not coming"
    );

    let closed = {
        let stream = stream.clone();
        std::thread::spawn(move || {
            block_on(stream.playlist(Some(LlHlsPosition {
                msn: 1_000,
                part: None,
            })))
        })
    };
    std::thread::sleep(Duration::from_millis(50));
    stream.close();
    assert!(matches!(
        closed.join().unwrap(),
        Err(LlHlsPlaylistError::TooFarAhead | LlHlsPlaylistError::Closed)
    ));
    assert_eq!(
        block_on(stream.playlist(None)),
        Err(LlHlsPlaylistError::Closed)
    );
    drop(sinks);
}

/// The same camera connected again goes on as the same stream: its segments
/// numbered on from the last, under the same init section, with no
/// discontinuity, its pictures a step after the last run's — however late
/// that run was let go of.
#[test]
fn a_run_of_the_same_tracks_goes_on_as_the_same_stream() {
    let Some((streams, packets)) = fixture() else {
        return;
    };
    let stream = LlHlsStream::new("runs", options(100)).unwrap();
    let half = &packets[..packets.len() / 2];
    let mut first = open(&stream, &streams);
    feed(&mut first, half);
    std::thread::sleep(Duration::from_millis(300));
    end(&mut first);
    let after_first = block_on(stream.playlist(None)).unwrap();
    let last_first = after_first
        .lines()
        .filter_map(|line| line.strip_prefix("seg-"))
        .count();
    drop(first);
    let mut second = open(&stream, &streams);
    feed(&mut second, half);
    end(&mut second);
    let playlist = block_on(stream.playlist(None)).unwrap();
    assert!(!playlist.contains("#EXT-X-DISCONTINUITY\n"), "{playlist}");
    assert!(!playlist.contains("init-1.mp4"), "{playlist}");
    assert!(
        playlist.contains(&format!("seg-{last_first}.mp4")),
        "numbered on: {playlist}"
    );

    // Read back as one file, the second run's pictures follow the first's.
    let mut whole = block_on(stream.file("init-0.mp4")).unwrap().to_vec();
    let segments = playlist
        .lines()
        .filter(|line| line.starts_with("seg-"))
        .count();
    for msn in 0..segments {
        whole.extend_from_slice(&block_on(stream.file(&format!("seg-{msn}.mp4"))).unwrap());
    }
    let path = std::env::temp_dir().join(format!("ll_hls_runs_{}.mp4", std::process::id()));
    std::fs::write(&path, &whole).unwrap();
    let mut input = ffmpeg::format::input(&path).unwrap();
    let unit = f64::from(input.stream(0).unwrap().time_base());
    let times: Vec<f64> = input
        .packets()
        .filter(|(s, _)| s.index() == 0)
        .filter_map(|(_, packet)| packet.dts())
        .map(|dts| dts as f64 * unit)
        .collect();
    std::fs::remove_file(&path).ok();
    let steps: Vec<f64> = times.windows(2).map(|pair| pair[1] - pair[0]).collect();
    assert!(
        steps.iter().all(|step| *step > 0.0 && *step < 0.1),
        "one timeline, a picture's step at a time: {steps:?}"
    );
}

/// A run of other tracks follows behind a discontinuity, with an init
/// section of its own.
#[test]
fn a_run_of_other_tracks_goes_on_behind_a_discontinuity() {
    let Some((streams, packets)) = fixture() else {
        return;
    };
    let stream = LlHlsStream::new("others", options(100)).unwrap();
    let mut first = open(&stream, &streams);
    feed(&mut first, &packets[..packets.len() / 2]);
    end(&mut first);
    drop(first);
    let video = streams
        .iter()
        .position(|info| info.kind() == ffmpeg::media::Type::Video)
        .unwrap();
    let alone = [streams[video].clone()];
    let pictures: Vec<_> = packets
        .iter()
        .filter(|(index, _)| *index == video)
        .map(|(_, packet)| (0, packet.clone()))
        .collect();
    let mut second = open(&stream, &alone);
    feed(&mut second, &pictures[..pictures.len() / 2]);
    end(&mut second);
    let playlist = block_on(stream.playlist(None)).unwrap();
    assert!(playlist.contains("#EXT-X-DISCONTINUITY\n"), "{playlist}");
    assert!(playlist.contains("init-1.mp4"), "{playlist}");
    assert!(block_on(stream.file("init-1.mp4")).is_some());
}

#[test]
fn options_that_cannot_be_kept_are_refused() {
    let part_too_long = LlHlsOptions {
        part_duration: Duration::from_secs(2),
        ..LlHlsOptions::default()
    };
    assert!(LlHlsStream::new("bad", part_too_long).is_err());
    assert!(LlHlsStream::new("bad", options(1)).is_err());
    let stream = LlHlsStream::new("ok", LlHlsOptions::default()).unwrap();
    assert!(stream.muxer().open().is_err(), "no stream added");
}

/// A part a player is told is coming is always in a segment already begun:
/// one told of the next segment's first part takes that segment as on this
/// run's timeline, which the next run, should this one end first, would
/// have contradicted with a discontinuity.
#[test]
fn a_part_is_told_coming_only_in_a_segment_begun() {
    let Some((streams, packets)) = fixture() else {
        return;
    };
    let stream = LlHlsStream::new("hints", options(100)).unwrap();
    let mut sinks = open(&stream, &streams);
    let mut told = 0;
    for (index, packet) in &packets {
        sinks[*index]
            .consume(MediaBuffer::Packet(Arc::new(packet.clone()).into()))
            .unwrap();
        let playlist = stream.shared.lock().playlist();
        let Some(hint) = playlist
            .lines()
            .find_map(|line| line.strip_prefix("#EXT-X-PRELOAD-HINT:TYPE=PART,URI=\"part-"))
        else {
            continue;
        };
        told += 1;
        let msn = hint.split('-').next().unwrap();
        let begun = format!("URI=\"part-{msn}-0.mp4\"");
        assert!(
            playlist
                .lines()
                .any(|line| line.starts_with("#EXT-X-PART:") && line.contains(&begun)),
            "a part of segment {msn} told coming before it began:\n{playlist}"
        );
    }
    assert!(told > 100, "parts were told coming: {told}");
}
