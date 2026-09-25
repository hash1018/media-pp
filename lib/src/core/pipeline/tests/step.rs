//! Stepping the picture: forward and back by pictures, paused, with the
//! sound held back meanwhile and put back in line when playback goes on.

use super::*;

/// One picture of the fixture, which runs at 30 a second.
const FRAME: Duration = Duration::from_nanos(33_333_333);

/// Which picture of the fixture is at `at`.
fn picture(at: Duration) -> u64 {
    (at.as_nanos() as f64 / FRAME.as_nanos() as f64).round() as u64
}

/// Notes where in its media every frame it takes is, pictures and sound
/// alike.
struct Taker {
    name: Arc<str>,
    pp_log: PpLog,
    taken: Arc<Mutex<Vec<Duration>>>,
}

impl Taker {
    fn new(name: &str) -> (Self, Arc<Mutex<Vec<Duration>>>) {
        let taken = Arc::new(Mutex::new(Vec::new()));
        let taker = Self {
            name: name.into(),
            pp_log: element_pp_log(ElementType::Other, name, None),
            taken: Arc::clone(&taken),
        };
        (taker, taken)
    }
}

impl Element for Taker {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Other
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for Taker {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let at = match &buf {
            MediaBuffer::Video(frame) => position(frame.pts(), crate::buffer::time_base(frame)),
            MediaBuffer::Audio(frame) => position(frame.pts(), crate::buffer::time_base(frame)),
            _ => None,
        };
        if let Some(at) = at {
            self.taken.lock().unwrap().push(at);
        }
        Ok(())
    }
}

fn position(pts: Option<i64>, time_base: Option<ffmpeg::Rational>) -> Option<Duration> {
    let ns = pts?.rescale(time_base?, ffmpeg::Rational::new(1, 1_000_000_000));
    u64::try_from(ns).ok().map(Duration::from_nanos)
}

/// A file's picture and sound off one demuxer, each decoded, queued and
/// paced — a player's pipeline, with what each end took written down.
struct Rig {
    pipeline: Arc<Pipeline>,
    pictures: Arc<Mutex<Vec<Duration>>>,
    sound: Arc<Mutex<Vec<Duration>>>,
}

impl Rig {
    fn new() -> Option<Self> {
        let path = try_test_video()?;
        let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
        let video = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Video)
            .expect("the fixture has a picture")
            .clone();
        let audio = streams
            .iter()
            .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
            .expect("the fixture has sound")
            .clone();
        let (screen, pictures) = Taker::new("screen");
        let (speakers, sound) = Taker::new("speakers");
        let (pipeline, ()) = Pipeline::new("step", source, |source, ctx| {
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
        Some(Self {
            pipeline,
            pictures,
            sound,
        })
    }

    /// Opened paused on its first picture, as a player opens a file it is
    /// not yet playing.
    fn paused_on_the_first_picture() -> Option<Self> {
        let rig = Self::new()?;
        rig.pipeline.pause();
        rig.pipeline.run().expect("run");
        rig.pipeline
            .seek(Duration::ZERO, SeekMode::Accurate)
            .expect("the first picture");
        Some(rig)
    }

    fn pictures(&self) -> Vec<u64> {
        self.pictures
            .lock()
            .unwrap()
            .iter()
            .map(|&at| picture(at))
            .collect()
    }

    fn sound_taken(&self) -> usize {
        self.sound.lock().unwrap().len()
    }
}

/// A step forward shows the pictures after the one shown, one by one and
/// none skipped, and holds the last; the sound is not played meanwhile.
#[test]
fn a_step_forward_shows_the_next_pictures_and_holds_the_last() {
    let Some(rig) = Rig::paused_on_the_first_picture() else {
        return;
    };
    assert_eq!(rig.pictures(), [0], "opened on the first picture");
    let sound = rig.sound_taken();

    let at = rig.pipeline.step(1).expect("a step");
    assert_eq!(picture(at), 1);
    assert_eq!(rig.pictures(), [0, 1], "exactly the next picture");

    let at = rig.pipeline.step(5).expect("five steps at once");
    assert_eq!(picture(at), 6);
    assert_eq!(
        rig.pictures(),
        [0, 1, 2, 3, 4, 5, 6],
        "none skipped, none twice"
    );

    thread::sleep(Duration::from_millis(300));
    assert_eq!(rig.pictures().len(), 7, "held on the last");
    assert_eq!(
        rig.sound_taken(),
        sound,
        "the sound is not played while stepping"
    );
    rig.pipeline.stop();
}

/// A step back shows the picture before the one shown, and further back
/// by the spacing of the pictures; it stops at the first.
#[test]
fn a_step_back_shows_the_pictures_before() {
    let Some(rig) = Rig::paused_on_the_first_picture() else {
        return;
    };
    rig.pipeline.step(6).expect("forward first");

    let at = rig.pipeline.step(-1).expect("a step back");
    assert_eq!(picture(at), 5);
    assert_eq!(rig.pictures().last(), Some(&5), "the picture before");

    let at = rig.pipeline.step(-3).expect("three back");
    assert_eq!(picture(at), 2);

    let at = rig.pipeline.step(1).expect("and forward again from there");
    assert_eq!(picture(at), 3);

    let at = rig.pipeline.step(-10).expect("back past the start");
    assert_eq!(picture(at), 0, "stops at the first picture");
    rig.pipeline.stop();
}

/// Playing on after a step starts from the picture shown, with the sound
/// back in line with it — not where the sound was left before stepping,
/// and not where the clock was.
#[test]
fn playing_on_after_a_step_puts_the_sound_back_in_line() {
    let Some(rig) = Rig::paused_on_the_first_picture() else {
        return;
    };
    let at = rig.pipeline.step(20).expect("twenty steps");
    assert_eq!(picture(at), 20);
    let pictures_before = rig.pictures().len();
    let sound_before = rig.sound_taken();

    rig.pipeline.resume();
    let deadline = Instant::now() + Duration::from_secs(5);
    while (rig.sound_taken() < sound_before + 10 || rig.pictures().len() < pictures_before + 10)
        && Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(5));
    }
    let sound = rig.sound.lock().unwrap()[sound_before];
    let lag = sound.abs_diff(at);
    assert!(
        lag < Duration::from_millis(100),
        "the sound went on at {sound:?}, the picture was at {at:?}"
    );
    let after: Vec<_> = rig.pictures()[pictures_before..].to_vec();
    assert_eq!(after.first(), Some(&20), "played on from the picture shown");
    assert!(
        after.windows(2).all(|pair| pair[1] == pair[0] + 1),
        "and on from there, picture by picture: {after:?}"
    );
    rig.pipeline.stop();
}

/// A step while playing pauses, and the pipeline stays paused.
#[test]
fn stepping_while_playing_pauses() {
    let Some(rig) = Rig::new() else {
        return;
    };
    rig.pipeline.run().expect("run");
    let deadline = Instant::now() + Duration::from_secs(5);
    while rig.pictures().len() < 5 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let at = rig.pipeline.step(1).expect("a step");
    let shown = rig.pictures().len();
    assert_eq!(rig.pictures().last(), Some(&picture(at)));
    thread::sleep(Duration::from_millis(300));
    assert_eq!(
        rig.pictures().len(),
        shown,
        "paused on the picture stepped to"
    );
    rig.pipeline.stop();
}

/// Past the end there is no picture left to step to: the step answers at
/// once, on the last one, rather than waiting out its preroll.
#[test]
fn a_step_past_the_end_stays_on_the_last_picture() {
    let Some(rig) = Rig::paused_on_the_first_picture() else {
        return;
    };
    let duration = FileDemuxer::open("measure", try_test_video().expect("fixture"))
        .expect("open")
        .0
        .duration()
        .expect("a duration");
    rig.pipeline
        .seek(duration.saturating_sub(FRAME * 3), SeekMode::Accurate)
        .expect("near the end");

    let started = Instant::now();
    let at = rig.pipeline.step(100).expect("past the end");
    let last = picture(at);
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "took {:?}",
        started.elapsed()
    );
    let started = Instant::now();
    let again = rig.pipeline.step(1).expect("at the end");
    assert_eq!(picture(again), last, "still the last picture");
    assert!(started.elapsed() < Duration::from_secs(1));
    rig.pipeline.stop();
}

/// Fanned out to two screens, a step moves both, back as well as forward,
/// and the sound beside them stays silent. A `Tee` is traced as the end of
/// the chain in front of it, and a step back that silenced every such end
/// but the screens silenced the `Tee` too: neither screen was handed a
/// picture, and the step waited out its preroll.
#[test]
fn a_step_moves_every_screen_a_picture_is_fanned_out_to() {
    let Some(path) = try_test_video() else {
        return;
    };
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
    let (first, first_taken) = Taker::new("screen");
    let (second, second_taken) = Taker::new("second-screen");
    let (speakers, sound) = Taker::new("speakers");
    let (pipeline, ()) = Pipeline::new("step-tee", source, |source, ctx| {
        let mut tee = ctx.tee("tee");
        for (name, screen) in [("screen", first), ("second-screen", second)] {
            tee = tee.branch(
                ctx.branch()
                    .queue(format!("{name}-frames"), 2)
                    .pipe(Pacer::new(format!("{name}-pacer")))
                    .to(screen)?,
            );
        }
        let picture = ctx
            .branch()
            .queue("video-packets", 64)
            .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
            .queue("video-frames", 8)
            .to_branch(tee.build()?)?;
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
    .expect("wire two screens and the sound");
    pipeline.pause();
    pipeline.run().expect("run");
    pipeline
        .seek(Duration::ZERO, SeekMode::Accurate)
        .expect("the first picture");
    let last =
        |taken: &Arc<Mutex<Vec<Duration>>>| taken.lock().unwrap().last().map(|&at| picture(at));
    let sound_before = sound.lock().unwrap().len();

    let at = pipeline.step(6).expect("forward");
    assert_eq!(picture(at), 6);
    assert_eq!(last(&first_taken), Some(6));
    assert_eq!(last(&second_taken), Some(6));

    let at = pipeline.step(-2).expect("back");
    assert_eq!(picture(at), 4);
    assert_eq!(last(&first_taken), Some(4), "the first screen stepped back");
    assert_eq!(last(&second_taken), Some(4), "and the second with it");
    assert_eq!(
        sound.lock().unwrap().len(),
        sound_before,
        "the sound is not played while stepping"
    );
    pipeline.stop();
}

/// A step back from a keyframe shows the picture before it. With B-frames
/// that picture is decoded after the keyframe, or in the group before it,
/// and a seek lands by when a keyframe is decoded: landed on the keyframe,
/// nothing covered the instant before it, and the step stayed where it was.
#[test]
fn a_step_back_from_a_keyframe_shows_the_picture_before_it() {
    let fixture = crate::test_support::synthesize_reordered("step-reordered", 3.0);
    // Where the file's pictures and keyframes are, as it says.
    let mut input = ffmpeg::format::input(&fixture.path).expect("open the fixture");
    let (index, base) = {
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("a picture");
        (stream.index(), stream.time_base())
    };
    let (mut pictures, mut keyframes) = (Vec::new(), Vec::new());
    for (stream, packet) in input.packets() {
        if stream.index() != index {
            continue;
        }
        let at = position(packet.pts(), Some(base)).expect("a timed picture");
        pictures.push(at);
        if packet.is_key() {
            keyframes.push(at);
        }
    }
    pictures.sort();
    let keyframe = *keyframes
        .iter()
        .find(|&&at| at > Duration::ZERO)
        .expect("a keyframe after the first");
    let before = *pictures
        .iter()
        .rfind(|&&at| at < keyframe)
        .expect("a picture before it");

    let (source, streams) = FileDemuxer::open("demux", &fixture.path).expect("open");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("a picture")
        .clone();
    let (screen, shown) = Taker::new("screen");
    let (pipeline, ()) = Pipeline::new("step-reordered", source, |source, ctx| {
        let picture = ctx
            .branch()
            .queue("video-packets", 64)
            .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
            .queue("video-frames", 8)
            .pipe(Pacer::new("video-pacer"))
            .to(screen)?;
        ctx.attach(source, video.index, picture)?;
        Ok(())
    })
    .expect("wire");
    pipeline.pause();
    pipeline.run().expect("run");
    pipeline
        .seek(keyframe, SeekMode::Accurate)
        .expect("onto the keyframe");
    assert_eq!(shown.lock().unwrap().last(), Some(&keyframe));

    let at = pipeline.step(-1).expect("a step back");
    assert_eq!(
        at, before,
        "the picture before the keyframe at {keyframe:?}"
    );
    assert_eq!(shown.lock().unwrap().last(), Some(&before));
    pipeline.stop();
}

/// A step needs a picture to move, and is refused where a seek is.
#[test]
fn a_step_needs_a_picture_and_a_source_that_can_be_sought() {
    let Some(rig) = Rig::new() else {
        return;
    };
    rig.pipeline.pause();
    rig.pipeline.run().expect("run, paused before a picture");
    assert!(matches!(
        rig.pipeline.step(1),
        Err(crate::Error::PipelineError(PipelineError::NoPicture))
    ));
    rig.pipeline.stop();

    let camera = TestVideoSource::new(
        "camera",
        TestVideoOptions {
            width: 64,
            height: 48,
            frame_rate: ffmpeg::Rational::new(30, 1),
        },
    );
    let (screen, _) = Taker::new("screen");
    let (pipeline, ()) = Pipeline::new("step-live", camera, |source, ctx| {
        let branch = ctx.branch().queue("frames", 4).to(screen)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire");
    pipeline.run().expect("run");
    assert!(matches!(pipeline.step(1), Err(crate::Error::SeekError(_))));
    pipeline.stop();
}
