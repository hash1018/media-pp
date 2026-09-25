//! Playing backwards: from the picture shown, every picture before it in
//! turn, last first, down to the start of the file; and forwards again.

use super::step::Taker;
use super::*;
use crate::element::ReversibleDecoder;

/// What a terminal took, where in its media each sample was.
type Taken = Arc<Mutex<Vec<Duration>>>;

/// A file's picture and sound off one demuxer, each decoded, queued and
/// paced — with what each end took written down.
fn player(path: &std::path::Path) -> (Arc<Pipeline>, Taken, Taken) {
    player_queueing(path, 64)
}

/// [`player`], `packets` of each stream queued before its decoder.
fn player_queueing(path: &std::path::Path, packets: usize) -> (Arc<Pipeline>, Taken, Taken) {
    let (source, streams) = FileDemuxer::open("demux", path).expect("open the fixture");
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
    let (pipeline, ()) = Pipeline::new("reverse", source, |source, ctx| {
        let picture = ctx
            .branch()
            .queue("video-packets", packets)
            .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
            .queue("video-frames", 8)
            .pipe(Pacer::new("video-pacer"))
            .to(screen)?;
        ctx.attach(source, video.index, picture)?;
        let sound = ctx
            .branch()
            .queue("audio-packets", packets)
            .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
            .queue("audio-frames", 8)
            .pipe(Pacer::new("audio-pacer"))
            .to(speakers)?;
        ctx.attach(source, audio.index, sound)?;
        Ok(())
    })
    .expect("wire the player");
    (pipeline, pictures, sound)
}

/// Every picture the file has, where it is, in order — as it says.
fn pictures_of(path: &std::path::Path) -> Vec<Duration> {
    let mut input = ffmpeg::format::input(path).expect("open");
    let (index, base) = {
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("a picture");
        (stream.index(), stream.time_base())
    };
    let mut pictures: Vec<_> = input
        .packets()
        .filter(|(stream, _)| stream.index() == index)
        .filter_map(|(_, packet)| packet.pts())
        .map(|pts| {
            Duration::from_nanos(
                pts.rescale(base, ffmpeg::Rational::new(1, 1_000_000_000))
                    .max(0) as u64,
            )
        })
        .collect();
    pictures.sort();
    pictures
}

/// Waits until `done` or five seconds.
fn wait_until(done: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done() {
        if Instant::now() > deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(10));
    }
    true
}

/// Turned around at a picture, playback shows it and then every picture
/// before it, one by one, none twice and none skipped, down to the file's
/// first, and ends there; the sound is not played meanwhile. With B-frames
/// or without, and across stretches read backwards, groups of pictures
/// and the ends of both.
#[test]
fn backwards_shows_every_picture_before_the_one_shown_down_to_the_first() {
    let fixtures = [
        try_test_video().map(std::path::PathBuf::from),
        Some(crate::test_support::synthesize_reordered("reverse-reordered", 4.0).path),
    ];
    for path in fixtures.into_iter().flatten() {
        let every = pictures_of(&path);
        let from = every[every.len() * 2 / 5];
        let (pipeline, pictures, sound) = player(&path);
        pipeline.pause();
        pipeline.run().expect("run");
        pipeline
            .seek(from, SeekMode::Accurate)
            .expect("to the picture to turn at");
        assert_eq!(pictures.lock().unwrap().last(), Some(&from));

        let (pictures_before, sound_before) =
            (pictures.lock().unwrap().len(), sound.lock().unwrap().len());
        pipeline
            .set_rate(Pipeline::REVERSE_RATE)
            .expect("backwards");
        assert_eq!(pipeline.rate(), Pipeline::REVERSE_RATE);
        assert_eq!(
            pictures.lock().unwrap().last(),
            Some(&from),
            "turned on the picture shown"
        );
        pipeline.resume();
        let expected: Vec<_> = every
            .iter()
            .copied()
            .filter(|&at| at <= from)
            .rev()
            .collect();
        let finished = wait_until(|| {
            std::iter::from_fn(|| pipeline.bus().try_recv())
                .any(|event| matches!(event, crate::bus::BusEvent::Finished))
        });
        let shown = pictures.lock().unwrap()[pictures_before..].to_vec();
        assert_eq!(
            shown,
            expected,
            "{}: from {from:?} down to the first, one by one",
            path.display()
        );
        assert!(finished, "{}: ends at the start", path.display());
        assert_eq!(
            sound.lock().unwrap().len(),
            sound_before,
            "no sound backwards"
        );
        pipeline.stop();
    }
}

/// Finished backwards, what was read of a stretch is read to its end
/// first: a stretch is played from its end, and one cut short where reading
/// stopped would skip its later pictures. A short queue leaves the source
/// stopped inside one.
#[test]
fn finished_backwards_skips_nothing() {
    let Some(path) = try_test_video() else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let every = pictures_of(&path);
    let from = every[every.len() * 3 / 5];
    let (pipeline, pictures, _) = player_queueing(&path, 4);
    pipeline.pause();
    pipeline.run().expect("run");
    pipeline
        .seek(from, SeekMode::Accurate)
        .expect("into the file");
    let before = pictures.lock().unwrap().len();
    pipeline.set_rate(-1.0).expect("backwards");
    pipeline.finish();
    let shown = pictures.lock().unwrap()[before..].to_vec();
    let expected: Vec<_> = every
        .iter()
        .copied()
        .filter(|&at| at <= from)
        .rev()
        .take(shown.len())
        .collect();
    assert!(shown.len() > 1, "played on to the end: {shown:?}");
    assert_eq!(shown, expected, "one by one down from {from:?}");
}

/// Forwards again from backwards: on from the picture shown, with the
/// sound back.
#[test]
fn forwards_again_goes_on_from_the_picture_shown() {
    let Some(path) = try_test_video() else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let every = pictures_of(&path);
    let (pipeline, pictures, sound) = player(&path);
    pipeline.pause();
    pipeline.run().expect("run");
    pipeline
        .seek(every[90], SeekMode::Accurate)
        .expect("into the file");
    pipeline.set_rate(-1.0).expect("backwards");
    pipeline.resume();
    assert!(wait_until(|| pictures
        .lock()
        .unwrap()
        .last()
        .is_some_and(|&at| at <= every[70])));

    pipeline.set_rate(1.0).expect("forwards again");
    let turned = *pictures.lock().unwrap().last().unwrap();
    let (pictures_before, sound_before) =
        (pictures.lock().unwrap().len(), sound.lock().unwrap().len());
    assert!(wait_until(|| pictures.lock().unwrap().len()
        > pictures_before + 10
        && sound.lock().unwrap().len() > sound_before + 10));
    let after = pictures.lock().unwrap()[pictures_before..].to_vec();
    assert!(
        after.windows(2).all(|pair| pair[1] > pair[0]),
        "forwards again: {after:?}"
    );
    assert!(
        after
            .first()
            .is_some_and(|&first| first <= turned + Duration::from_millis(40)),
        "on from the picture shown ({turned:?}): {after:?}"
    );
    let heard = sound.lock().unwrap()[sound_before];
    assert!(
        heard.abs_diff(turned) < Duration::from_millis(150),
        "the sound with it: {heard:?} against {turned:?}"
    );
    pipeline.stop();
}

/// Refused where it cannot be played: before running, and on a live
/// source.
#[test]
fn backwards_is_refused_where_it_cannot_be_played() {
    let Some(path) = try_test_video() else {
        return;
    };
    let (pipeline, _, _) = player(std::path::Path::new(&path));
    assert!(pipeline.check_reverse().is_ok(), "a file can be");
    assert!(matches!(
        pipeline.set_rate(-1.0),
        Err(crate::Error::PipelineError(PipelineError::NotRunning))
    ));
    for rate in [-0.5, -2.0] {
        assert!(matches!(
            pipeline.set_rate(rate),
            Err(crate::Error::PipelineError(PipelineError::UnsupportedRate))
        ));
    }

    let camera = TestVideoSource::new(
        "camera",
        TestVideoOptions {
            width: 64,
            height: 48,
            frame_rate: ffmpeg::Rational::new(30, 1),
        },
    );
    let (screen, _) = Taker::new("screen");
    let (pipeline, ()) = Pipeline::new("reverse-live", camera, |source, ctx| {
        let branch = ctx.branch().queue("frames", 4).to(screen)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire");
    assert!(pipeline.check_reverse().is_err());
    pipeline.run().expect("run");
    assert!(matches!(
        pipeline.set_rate(-1.0),
        Err(crate::Error::SeekError(_))
    ));
    pipeline.stop();
}

/// On the GPU the decoder's surfaces are a fixed pool, fewer than a stretch
/// has pictures, so what it holds to play a stretch backwards are copies.
/// Each is the picture it copies: read back, a picture played backwards is
/// the same as that picture played forwards.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn backwards_on_a_d3d11_decoder_shows_the_same_pictures_as_forwards() {
    use crate::elements::{D3d11Download, DecodePath, DecodeTarget, VideoDecodeBin};
    use std::collections::HashMap;

    /// Each picture's place and what its luma adds up to.
    struct Sums {
        pp_log: PpLog,
        seen: Arc<Mutex<Vec<(i64, u64)>>>,
    }
    impl Element for Sums {
        fn name(&self) -> Arc<str> {
            "sums".into()
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
    impl Sink for Sums {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Video(frame) = &buf {
                let (width, height, stride) = (
                    frame.width() as usize,
                    frame.height() as usize,
                    frame.stride(0),
                );
                let luma = frame.data(0);
                let sum = (0..height)
                    .map(|row| {
                        luma[row * stride..row * stride + width]
                            .iter()
                            .map(|&value| u64::from(value))
                            .sum::<u64>()
                    })
                    .sum();
                self.seen
                    .lock()
                    .unwrap()
                    .push((frame.pts().unwrap_or(-1), sum));
            }
            Ok(())
        }
    }

    let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
        return;
    };
    let Some(path) = try_test_video() else {
        return;
    };
    let every = pictures_of(std::path::Path::new(&path));
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("a picture")
        .clone();
    let bin = VideoDecodeBin::open(
        "video",
        video.parameters.clone(),
        DecodeTarget::D3d11 {
            gpu: gpu.clone(),
            downstream_hw_frames: 8 + 3,
        },
        None,
    )
    .expect("open the decode bin");
    if bin.path() != DecodePath::Hardware {
        eprintln!(
            "skipping: this GPU does not decode the fixture ({:?})",
            bin.path()
        );
        return;
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sums = Sums {
        pp_log: element_pp_log(ElementType::Other, "sums", None),
        seen: Arc::clone(&seen),
    };
    let download = D3d11Download::new("download", &gpu).expect("a download");
    let (pipeline, ()) = Pipeline::new("reverse-d3d11", source, |source, ctx| {
        let branch = ctx
            .branch()
            .queue("video-packets", 64)
            .pipe(bin)
            .queue("video-frames", 8)
            .pipe(download)
            .to(sums)?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })
    .expect("wire");
    assert!(pipeline.check_reverse().is_ok(), "D3D11 decoding can");
    // Forwards, unpaced, to the end: every picture's sum.
    pipeline.run().expect("run");
    assert!(wait_until(|| std::iter::from_fn(|| pipeline
        .bus()
        .try_recv())
    .any(|event| matches!(event, crate::bus::BusEvent::Finished))));
    let forwards: HashMap<i64, u64> = seen.lock().unwrap().drain(..).collect();
    assert_eq!(forwards.len(), every.len(), "every picture forwards");

    // Backwards from two thirds in.
    pipeline.pause();
    let from = every[every.len() * 2 / 3];
    pipeline
        .seek(from, SeekMode::Accurate)
        .expect("to the picture to turn at");
    seen.lock().unwrap().clear();
    pipeline
        .set_rate(Pipeline::REVERSE_RATE)
        .expect("backwards");
    pipeline.resume();
    assert!(wait_until(|| std::iter::from_fn(|| pipeline
        .bus()
        .try_recv())
    .any(|event| matches!(event, crate::bus::BusEvent::Finished))));
    let backwards = seen.lock().unwrap().clone();
    let shown: Vec<i64> = backwards.iter().map(|&(pts, _)| pts).collect();
    let mut expected = shown.clone();
    expected.sort_unstable_by(|a, b| b.cmp(a));
    expected.dedup();
    assert_eq!(shown, expected, "last first, none twice");
    assert_eq!(
        shown.len(),
        every.iter().filter(|&&at| at <= from).count(),
        "every picture down to the first"
    );
    for (pts, sum) in backwards {
        assert_eq!(
            forwards.get(&pts),
            Some(&sum),
            "the picture at {pts} is the same backwards"
        );
    }
    pipeline.stop();
}

/// What [`Watched`] was told and handed, in order.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Told {
    Begin,
    End,
    /// A packet, by its decode time.
    Packet(Option<i64>),
    Eos,
    Flush,
}

/// A picture's [`SwDecoder`] with what it is told and handed written down
/// — and, where `reversible` is false, not a [`ReversibleDecoder`]: an
/// element a user wrote that decodes pictures and cannot play them
/// backwards.
struct Watched {
    inner: SwDecoder,
    reversible: bool,
    told: Arc<Mutex<Vec<Told>>>,
}

impl Element for Watched {
    fn name(&self) -> Arc<str> {
        self.inner.name()
    }
    fn element_type(&self) -> ElementType {
        ElementType::Other
    }
    fn pp_log(&self) -> &PpLog {
        self.inner.pp_log()
    }
    fn pp_log_mut(&mut self) -> &mut PpLog {
        self.inner.pp_log_mut()
    }
    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        self.inner.attach_context(context);
    }
}

impl Source for Watched {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        self.inner.src_pads()
    }
}

impl Sink for Watched {
    fn ready_consume(&mut self) -> bool {
        self.inner.ready_consume()
    }
    fn input_contract(&self) -> InputContract {
        self.inner.input_contract()
    }
    fn as_reversible(&mut self) -> Option<&mut dyn ReversibleDecoder> {
        if self.reversible { Some(self) } else { None }
    }
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let told = match &buf {
            MediaBuffer::Packet(packet) => Some(Told::Packet(packet.dts())),
            MediaBuffer::Eos => Some(Told::Eos),
            _ => None,
        };
        self.told.lock().unwrap().extend(told);
        self.inner.consume(buf)
    }
    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        if *msg == ControlMsg::Flush {
            self.told.lock().unwrap().push(Told::Flush);
        }
        self.inner.control(msg)
    }
}

impl ReversibleDecoder for Watched {
    fn begin_stretch(&mut self) -> Result<()> {
        self.told.lock().unwrap().push(Told::Begin);
        self.inner.begin_stretch()
    }
    fn end_stretch(&mut self) -> Result<()> {
        self.told.lock().unwrap().push(Told::End);
        self.inner.end_stretch()
    }
}

/// A file's picture through a [`Watched`] decoder to a paced end.
fn watched(path: &std::path::Path, reversible: bool) -> (Arc<Pipeline>, Arc<Mutex<Vec<Told>>>) {
    let (source, streams) = FileDemuxer::open("demux", path).expect("open the fixture");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("a picture")
        .clone();
    let told = Arc::new(Mutex::new(Vec::new()));
    let decoder = Watched {
        inner: SwDecoder::new("video-decoder", video.parameters.clone()).expect("a decoder"),
        reversible,
        told: Arc::clone(&told),
    };
    let (screen, _) = Taker::new("screen");
    let (pipeline, ()) = Pipeline::new("reverse-watched", source, |source, ctx| {
        let picture = ctx
            .branch()
            .queue("video-packets", 64)
            .pipe(decoder)
            .queue("video-frames", 8)
            .pipe(Pacer::new("video-pacer"))
            .to(screen)?;
        ctx.attach(source, video.index, picture)?;
        Ok(())
    })
    .expect("wire");
    (pipeline, told)
}

/// Backwards, the pipeline tells a decoder where each stretch begins and
/// ends: every packet is inside one, each in decode order, one ends just
/// before a packet whose decode time goes back and the last before the end
/// of the stream — and there are several.
#[test]
fn a_decoder_is_told_where_each_stretch_begins_and_ends() {
    let Some(path) = try_test_video() else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let every = pictures_of(&path);
    let (pipeline, told) = watched(&path, true);
    pipeline.pause();
    pipeline.run().expect("run");
    pipeline
        .seek(every[every.len() / 2], SeekMode::Accurate)
        .expect("into the file");
    let from = told.lock().unwrap().len();
    pipeline.set_rate(-1.0).expect("backwards");
    pipeline.resume();
    assert!(wait_until(|| {
        std::iter::from_fn(|| pipeline.bus().try_recv())
            .any(|event| matches!(event, crate::bus::BusEvent::Finished))
    }));
    pipeline.stop();

    let told = told.lock().unwrap()[from..].to_vec();
    let (mut open, mut last, mut stretches) = (false, None, 0);
    for (at, &event) in told.iter().enumerate() {
        match event {
            Told::Begin => {
                assert!(!open, "{at}: begun twice: {told:?}");
                open = true;
                last = None;
            }
            Told::End => {
                assert!(open, "{at}: ended unbegun: {told:?}");
                open = false;
                stretches += 1;
                let next = told[at + 1..]
                    .iter()
                    .find(|event| !matches!(event, Told::Begin));
                assert!(
                    match (next, last) {
                        (Some(Told::Packet(Some(now))), Some(last)) => *now < last,
                        (Some(Told::Eos) | None, _) => true,
                        _ => false,
                    },
                    "{at}: ended where decode time does not go back: {told:?}"
                );
            }
            Told::Packet(decoded) => {
                assert!(open, "{at}: a packet outside a stretch: {told:?}");
                if let (Some(last), Some(now)) = (last, decoded) {
                    assert!(now > last, "{at}: out of decode order: {told:?}");
                }
                last = decoded.or(last);
            }
            Told::Eos => assert!(!open, "{at}: the end inside a stretch: {told:?}"),
            Told::Flush => {
                open = false;
                last = None;
            }
        }
    }
    assert!(stretches > 1, "several stretches: {told:?}");
}

/// An element that turns a picture's packets into pictures and is not a
/// [`ReversibleDecoder`] cannot play backwards, and says so as it is wired —
/// while the same element that is one can.
#[test]
fn a_decoder_that_cannot_hand_a_stretch_on_backwards_refuses() {
    let Some(path) = try_test_video() else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let (pipeline, _) = watched(&path, false);
    let refused = pipeline.check_reverse().expect_err("refused");
    assert_eq!(refused.rejections().len(), 1, "{refused:?}");
    assert_eq!(
        refused.rejections()[0].reason,
        crate::control::SeekRejectReason::DecoderNotReversible
    );
    assert_eq!(&*refused.rejections()[0].name, "video-decoder");
    assert!(pipeline.check_seek().is_ok(), "seeking is another matter");

    let (pipeline, _) = watched(&path, true);
    assert!(pipeline.check_reverse().is_ok());
}
