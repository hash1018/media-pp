//! Playing at a rate: the picture and the sound paced that much faster or
//! slower, from where playback is, and the rates a pipeline refuses.

use super::step::Taker;
use super::*;

/// A file's picture and sound off one demuxer, each decoded, queued and
/// paced — with what each end took written down.
/// What a terminal took, where in its media each sample was.
type Taken = Arc<Mutex<Vec<Duration>>>;

fn paced_player() -> Option<(Arc<Pipeline>, Taken, Taken)> {
    let path = try_test_video()?;
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
    let stream = |kind| {
        streams
            .iter()
            .find(|stream| stream.kind == kind)
            .expect("the fixture has picture and sound")
            .clone()
    };
    let (video, audio) = (
        stream(ffmpeg::media::Type::Video),
        stream(ffmpeg::media::Type::Audio),
    );
    let (screen, pictures) = Taker::new("screen");
    let (speakers, sound) = Taker::new("speakers");
    let (pipeline, ()) = Pipeline::new("rate", source, |source, ctx| {
        let picture = ctx
            .branch()
            .queue("video-packets", 64)
            .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
            .queue("video-frames", 8)
            .pipe(Pacer::new("video-pacer"))
            .to(screen)?;
        ctx.attach(source, video.index, picture)?;
        let sound = ctx
            .branch()
            .queue("audio-packets", 64)
            .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
            .queue("audio-frames", 8)
            .pipe(Pacer::new("audio-pacer"))
            .to(speakers)?;
        ctx.attach(source, audio.index, sound)?;
        Ok(())
    })
    .expect("wire the player");
    Some((pipeline, pictures, sound))
}

/// How much media `taken` covered from `since` on — the last sample taken
/// by then to the last taken now.
fn covered(taken: &Mutex<Vec<Duration>>, since: usize) -> Duration {
    let taken = taken.lock().unwrap();
    match (taken.get(since.saturating_sub(1)), taken.last()) {
        (Some(from), Some(to)) => to.saturating_sub(*from),
        _ => Duration::ZERO,
    }
}

/// At twice the rate the picture and the sound each cover two seconds of
/// the file in a second, and so does the position; at half, half of one.
/// Nothing is sought for it: the pictures go on from where they were, none
/// skipped.
#[test]
fn a_rate_paces_picture_and_sound_that_much_faster_or_slower() {
    let Some((pipeline, pictures, sound)) = paced_player() else {
        return;
    };
    pipeline.run().expect("run");
    let deadline = Instant::now() + Duration::from_secs(5);
    while pictures.lock().unwrap().len() < 10 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    for (rate, window) in [
        (2.0, Duration::from_millis(1_500)),
        (0.5, Duration::from_millis(1_000)),
    ] {
        pipeline.set_rate(rate).expect("a rate");
        assert_eq!(pipeline.rate(), rate);
        let (picture_from, sound_from) =
            (pictures.lock().unwrap().len(), sound.lock().unwrap().len());
        let (position_from, started) = (pipeline.position().unwrap(), Instant::now());
        thread::sleep(window);
        let took = started.elapsed().as_secs_f64();
        let speed = |covered: Duration| covered.as_secs_f64() / took;
        let position = pipeline.position().unwrap().saturating_sub(position_from);
        for (what, covered) in [
            ("the picture", covered(&pictures, picture_from)),
            ("the sound", covered(&sound, sound_from)),
            ("the position", position),
        ] {
            assert!(
                (rate * 0.8..rate * 1.2).contains(&speed(covered)),
                "at {rate}, {what} covered {covered:?} in {took:.2} s"
            );
        }
        let after = pictures.lock().unwrap()[picture_from.saturating_sub(1)..].to_vec();
        assert!(
            after
                .windows(2)
                .all(|pair| pair[1] > pair[0] && pair[1] - pair[0] < Duration::from_millis(50)),
            "on from where it was, none skipped: {after:?}"
        );
    }
    pipeline.stop();
}

/// A rate out of range, or no number at all, is refused and changes
/// nothing; one a seek would be refused for is refused the same way; one
/// set before running is where playback starts.
#[test]
fn a_rate_is_refused_where_it_cannot_be_played() {
    let Some((pipeline, _, _)) = paced_player() else {
        return;
    };
    for rate in [0.1, 8.0, 0.0, -0.0, -0.1, -8.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                pipeline.set_rate(rate),
                Err(crate::Error::PipelineError(PipelineError::UnsupportedRate))
            ),
            "{rate} taken"
        );
    }
    assert_eq!(pipeline.rate(), 1.0, "nothing changed");
    pipeline
        .set_rate(Pipeline::MAX_RATE)
        .expect("before running");
    assert_eq!(pipeline.rate(), Pipeline::MAX_RATE);
    pipeline.stop();

    let camera = TestVideoSource::new(
        "camera",
        TestVideoOptions {
            width: 64,
            height: 48,
            frame_rate: ffmpeg::Rational::new(30, 1),
        },
    );
    let (screen, _) = Taker::new("screen");
    let (pipeline, ()) = Pipeline::new("rate-live", camera, |source, ctx| {
        let branch = ctx.branch().queue("frames", 4).to(screen)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire");
    assert!(matches!(
        pipeline.set_rate(2.0),
        Err(crate::Error::SeekError(_))
    ));
    assert_eq!(
        pipeline.rate(),
        1.0,
        "a live source plays at the rate it arrives"
    );
}
