//! Seeking: the check before it, the pause and preroll around it, and
//! where playback lands.

use super::*;

#[test]
fn seek_check_rejects_a_live_source_before_flushing() {
    let source = TestVideoSource::new("live", TestVideoOptions::default());
    let pipeline = Pipeline::new("seek-check", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(NoOpSink {
            name: "noop".into(),
            pp_log: element_pp_log(ElementType::Other, "noop", None),
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    let error = pipeline
        .seek(Duration::from_secs(1), SeekMode::Accurate)
        .expect_err("live source must reject seek");
    pipeline.stop();

    let crate::Error::SeekError(error) = error else {
        panic!("expected SeekError, got {error:?}");
    };
    assert_eq!(error.rejections().len(), 1);
    assert_eq!(
        error.rejections()[0].reason,
        crate::control::SeekRejectReason::LiveSource
    );
}

struct SeekLoopSource {
    pp_log: PpLog,
    pad: SrcPad,
    seeks: Arc<AtomicUsize>,
}

impl Element for SeekLoopSource {
    fn name(&self) -> Arc<str> {
        "seek-loop".into()
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

impl Source for SeekLoopSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for SeekLoopSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            if drain_control(control, self, bus)?.stopped {
                return Ok(());
            }
            if self.pad.ready_consume() {
                self.pad
                    .push(MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty())))?;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.seeks.fetch_add(1, Ordering::SeqCst);
        Ok(target)
    }
}

struct ControlRecordingSink {
    pp_log: PpLog,
    count: Arc<AtomicUsize>,
    controls: Arc<Mutex<Vec<&'static str>>>,
    preroll_targets: Arc<Mutex<Vec<Option<Duration>>>>,
}

impl Element for ControlRecordingSink {
    fn name(&self) -> Arc<str> {
        "control-recorder".into()
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

impl Sink for ControlRecordingSink {
    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        let label = match msg {
            ControlMsg::Pause => "pause",
            ControlMsg::Resume => "resume",
            ControlMsg::Stop => "stop",
            ControlMsg::Flush => "flush",
            ControlMsg::CheckSeek(_) => "check-seek",
            ControlMsg::Preroll(context) => {
                self.preroll_targets.lock().unwrap().push(context.target());
                "preroll"
            }
            ControlMsg::Seek(_) => "seek",
        };
        self.controls.lock().unwrap().push(label);
        Ok(())
    }
}

#[test]
fn paused_seek_prerolls_one_timeline_and_restores_pause() {
    let seeks = Arc::new(AtomicUsize::new(0));
    let count = Arc::new(AtomicUsize::new(0));
    let controls = Arc::new(Mutex::new(Vec::new()));
    let preroll_targets = Arc::new(Mutex::new(Vec::new()));
    let source = SeekLoopSource {
        pp_log: element_pp_log(ElementType::Other, "seek-loop", None),
        pad: SrcPad::new("src"),
        seeks: Arc::clone(&seeks),
    };
    let pipeline = Pipeline::new("paused-seek-preroll", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(ControlRecordingSink {
            pp_log: element_pp_log(ElementType::Other, "control-recorder", None),
            count: Arc::clone(&count),
            controls: Arc::clone(&controls),
            preroll_targets: Arc::clone(&preroll_targets),
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    while count.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }
    pipeline.pause();
    controls.lock().unwrap().clear();

    pipeline
        .seek(Duration::from_secs(2), SeekMode::Accurate)
        .expect("paused seek");
    assert_eq!(seeks.load(Ordering::SeqCst), 1);
    let after_preroll = count.load(Ordering::SeqCst);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(count.load(Ordering::SeqCst), after_preroll);
    assert_eq!(
        controls.lock().unwrap().as_slice(),
        ["check-seek", "flush", "seek", "preroll", "pause"]
    );
    assert_eq!(
        preroll_targets.lock().unwrap().as_slice(),
        [Some(Duration::from_secs(2))]
    );

    controls.lock().unwrap().clear();
    preroll_targets.lock().unwrap().clear();
    pipeline
        .seek(Duration::from_secs(3), SeekMode::Keyframe)
        .expect("paused keyframe seek");
    let after_keyframe_preroll = count.load(Ordering::SeqCst);
    assert_eq!(after_keyframe_preroll, after_preroll + 1);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(count.load(Ordering::SeqCst), after_keyframe_preroll);
    assert_eq!(
        controls.lock().unwrap().as_slice(),
        ["check-seek", "flush", "seek", "preroll", "pause"]
    );
    assert_eq!(preroll_targets.lock().unwrap().as_slice(), [None]);

    pipeline.resume();
    while count.load(Ordering::SeqCst) == after_keyframe_preroll {
        thread::yield_now();
    }
    pipeline.stop();
}

#[test]
fn playing_seek_uses_an_internal_pause_then_resumes() {
    let seeks = Arc::new(AtomicUsize::new(0));
    let count = Arc::new(AtomicUsize::new(0));
    let controls = Arc::new(Mutex::new(Vec::new()));
    let preroll_targets = Arc::new(Mutex::new(Vec::new()));
    let source = SeekLoopSource {
        pp_log: element_pp_log(ElementType::Other, "seek-loop", None),
        pad: SrcPad::new("src"),
        seeks: Arc::clone(&seeks),
    };
    let pipeline = Pipeline::new("playing-seek-preroll", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(ControlRecordingSink {
            pp_log: element_pp_log(ElementType::Other, "control-recorder", None),
            count: Arc::clone(&count),
            controls: Arc::clone(&controls),
            preroll_targets,
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    while count.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }
    controls.lock().unwrap().clear();

    pipeline
        .seek(Duration::from_secs(2), SeekMode::Accurate)
        .expect("playing seek");
    let after_preroll = count.load(Ordering::SeqCst);
    while count.load(Ordering::SeqCst) == after_preroll {
        thread::yield_now();
    }
    assert_eq!(seeks.load(Ordering::SeqCst), 1);
    assert_eq!(
        controls.lock().unwrap()[..6],
        ["check-seek", "pause", "flush", "seek", "preroll", "resume"]
    );
    pipeline.stop();
}

/// `seek()` mid-playback should reposition the source (no error from
/// `Input::seek`), reset/flush everything downstream without
/// deadlocking, and let packets keep flowing afterward.
#[test]
fn seek_repositions_and_playback_continues() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;
    let time_base = source.stream_time_base(index).expect("stream disappeared");

    let count = Arc::new(AtomicUsize::new(0));
    let sink = CountingSink {
        name: "counting-sink".into(),
        count: count.clone(),
        pp_log: element_pp_log(ElementType::Other, "counting-sink", None),
    };

    // A `Pacer` here isn't incidental: without it, this whole 10s/
    // 300-packet file races through in well under the 50ms sleep
    // below (no decode, no throttling), so `seek()` would land on an
    // already-finished pipeline and silently no-op — exactly the kind
    // of thing a weak `count > 0` assertion wouldn't have caught (see
    // `seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe`
    // for how this was found).
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let pacer = Pacer::new("pacer", time_base)?;
        let branch = ctx.branch().queue("q", 4).pipe(pacer).to(Box::new(sink))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(50));
    pipeline
        .seek(Duration::from_secs(1), SeekMode::Accurate)
        .expect("seek");
    // Let packets flow again post-seek before tearing down.
    thread::sleep(Duration::from_millis(100));
    pipeline.stop();

    let events: Vec<_> = pipeline.bus().iter().collect();
    assert!(
        !events.iter().any(|e| matches!(e, BusEvent::Error { .. })),
        "unexpected error event(s): {events:?}"
    );
    assert!(
        count.load(Ordering::SeqCst) > 0,
        "expected at least one packet to arrive after the seek"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            BusEvent::Seeked { requested, .. } if *requested == Duration::from_secs(1)
        )),
        "expected a Seeked event reporting the request; got {events:?}"
    );
}

/// Regression test for the bug found manually testing `rtsp_serve_seek`:
/// a container seek can only land on a keyframe at or before `target`
/// (see `FileDemuxer::seek`'s docs), so a `target` inside a GOP lands
/// back at that GOP's keyframe — potentially nowhere near what was
/// requested. Without `BusEvent::Seeked` reporting that gap, this looked
/// indistinguishable from `seek` silently doing nothing.
///
/// The assertions below hold for any fixture: how far back the seek
/// actually lands depends on the file's keyframe spacing, but it must
/// never land *past* the request, and the gap must be reported.
#[test]
fn seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|s| s.kind == ffmpeg_next::media::Type::Video)
        .expect("test video has a video stream");
    let index = video.index;
    let time_base = source.stream_time_base(index).expect("stream disappeared");

    // Paced for the same reason as `seek_repositions_and_playback_continues`
    // — otherwise the file finishes before `seek()` is even called.
    let pipeline = Pipeline::new("test", source, |source, ctx| {
        let pacer = Pacer::new("pacer", time_base)?;
        let branch = ctx
            .branch()
            .queue("q", 4)
            .pipe(pacer)
            .to(Box::new(NoOpSink {
                name: "noop".into(),
                pp_log: element_pp_log(ElementType::Other, "noop", None),
            }))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(50));
    pipeline
        .seek(Duration::from_secs(3), SeekMode::Accurate)
        .expect("seek");
    thread::sleep(Duration::from_millis(100));
    pipeline.stop();

    let events: Vec<_> = pipeline.bus().iter().collect();
    let seeked = events
        .iter()
        .find_map(|e| match e {
            BusEvent::Seeked {
                requested, landed, ..
            } => Some((*requested, *landed)),
            _ => None,
        })
        .expect("expected a Seeked event");
    assert_eq!(seeked.0, Duration::from_secs(3));
    assert!(
        seeked.1 <= seeked.0,
        "a container seek must land at or before the request, got {:?} for {:?}",
        seeked.1,
        seeked.0
    );
}

/// Records where each branch's decoded media actually landed, so a preroll can
/// be checked against the position it was asked for rather than just against
/// "something arrived".
struct PrerollProbe {
    label: &'static str,
    time_base: ffmpeg::Rational,
    samples: Arc<Mutex<Vec<i64>>>,
    pp_log: PpLog,
}

impl Element for PrerollProbe {
    fn name(&self) -> Arc<str> {
        self.label.into()
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

impl Sink for PrerollProbe {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        let pts = match &buf {
            MediaBuffer::Video(frame) => frame.pts(),
            MediaBuffer::Audio(frame) => frame.pts(),
            _ => None,
        };
        if let Some(pts) = pts {
            let ns = pts.rescale(self.time_base, ffmpeg::Rational(1, 1_000_000_000));
            self.samples.lock().unwrap().push(ns);
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// A paused seek has to leave *every* decoded branch holding one sample at the
/// requested position — not just the branch that happens to carry a pacing
/// element, and not a burst of them.
///
/// Two separate defects showed up here, and the assertions below fail on
/// either. Measured against the real fixture at a 3s target:
///
/// - With the suppression gate living in `Pacer`/`VideoSynchronizer`, the
///   audio branch had neither, so it delivered 86 samples spanning 0.000s to
///   2.007s while video correctly delivered one at 3.003s — the two streams
///   ended a second apart, which on resume is a second of frozen picture.
/// - With terminals staying open until the *whole* preroll completed, the
///   branch that reached the target first kept consuming while the other
///   caught up: video ran 31 samples from 3.003s to 4.004s.
#[test]
fn a_paused_seek_leaves_every_branch_holding_one_sample_at_the_target() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let Some(audio) = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
    else {
        eprintln!("skipping: fixture has no audio stream");
        return;
    };
    let video_params = source.stream_parameters(video.index).expect("video params");
    let audio_params = source.stream_parameters(audio.index).expect("audio params");
    let video_tb = source.stream_time_base(video.index).expect("video tb");
    let audio_tb = source.stream_time_base(audio.index).expect("audio tb");

    let target = Duration::from_secs(3);
    let video_samples = Arc::new(Mutex::new(Vec::new()));
    let audio_samples = Arc::new(Mutex::new(Vec::new()));

    let pipeline = Pipeline::new("paused-av-seek", source, |source, ctx| {
        // Only the video branch is paced, which is the ordinary shape: an
        // audio renderer schedules itself against its own device clock.
        let video_branch = ctx
            .branch()
            .pipe(SwDecoder::new("video-decoder", video_params)?)
            .pipe(Pacer::new("video-pacer", video_tb)?)
            .queue("video-frames", 8)
            .to(Box::new(PrerollProbe {
                label: "video",
                time_base: video_tb,
                samples: Arc::clone(&video_samples),
                pp_log: element_pp_log(ElementType::Other, "video", None),
            }))?;
        ctx.attach(source, video.index, video_branch)?;
        let audio_branch = ctx
            .branch()
            .queue("audio-packets", 8)
            .pipe(SwDecoder::new("audio-decoder", audio_params)?)
            .to(Box::new(PrerollProbe {
                label: "audio",
                time_base: audio_tb,
                samples: Arc::clone(&audio_samples),
                pp_log: element_pp_log(ElementType::Other, "audio", None),
            }))?;
        ctx.attach(source, audio.index, audio_branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(100));
    pipeline.pause();
    video_samples.lock().unwrap().clear();
    audio_samples.lock().unwrap().clear();

    pipeline
        .seek(target, SeekMode::Accurate)
        .expect("a decoded A/V graph accepts a seek");
    let taken = |samples: &Arc<Mutex<Vec<i64>>>| samples.lock().unwrap().clone();
    let video = taken(&video_samples);
    let audio = taken(&audio_samples);
    pipeline.stop();

    let seconds = |ns: &[i64]| {
        ns.iter()
            .map(|ns| format!("{:.3}s", *ns as f64 / 1e9))
            .collect::<Vec<_>>()
    };
    for (label, samples) in [("video", &video), ("audio", &audio)] {
        assert_eq!(
            samples.len(),
            1,
            "{label} must hold exactly one preview sample, got {:?}",
            seconds(samples)
        );
    }
}

/// A packet terminal can finish on the landed keyframe while a sibling
/// decoder still needs many more packets to reach the accurate target. The
/// completed branch must close independently without closing the whole Tee.
#[test]
fn a_completed_tee_branch_does_not_starve_a_sibling_preroll() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = source.stream_parameters(video.index).expect("video params");
    let time_base = source.stream_time_base(video.index).expect("video tb");
    let packets = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(Mutex::new(Vec::new()));

    let pipeline = Pipeline::new("tee-preroll", source, |source, ctx| {
        let packet_branch = ctx.branch().to(Box::new(CountingSink {
            name: "packet-terminal".into(),
            count: Arc::clone(&packets),
            pp_log: element_pp_log(ElementType::Other, "packet-terminal", None),
        }))?;
        let decoded_branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", params)?)
            .to(Box::new(PrerollProbe {
                label: "video-terminal",
                time_base,
                samples: Arc::clone(&frames),
                pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
            }))?;
        let tee = TeeBuilder::new("tee", ctx.clone())
            .branch(packet_branch)
            .branch(decoded_branch)
            .build()?;
        ctx.attach(source, video.index, tee)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    // Paused once it is demonstrably running, rather than after a fixed wait.
    // A fixed one has to be long enough to have started and short enough not
    // to have *finished*, and how long the second of those is depends on the
    // fixture: a full-length recording is nowhere near its end after 100ms,
    // while the synthesized one is consumed inside that and the seek below
    // then has a finished source to preroll — which it reports as success,
    // having delivered nothing.
    while packets.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }
    pipeline.pause();
    packets.store(0, Ordering::SeqCst);
    frames.lock().unwrap().clear();

    pipeline
        .seek(Duration::from_secs(3), SeekMode::Accurate)
        .expect("both Tee branches preroll");
    assert_eq!(packets.load(Ordering::SeqCst), 1);
    assert_eq!(frames.lock().unwrap().len(), 1);
    pipeline.stop();
}

/// Container duration can extend beyond the last video PTS (for example when
/// audio is longer). Seeking there still has a well-defined paused-preview
/// result: the last decoded video frame, not a silent EOS success that leaves
/// the old picture on screen.
#[test]
fn accurate_seek_at_known_eof_selects_the_last_presentable_frame() {
    let Some(path) = try_test_video() else { return };
    let input = ffmpeg::format::input(&path).expect("open fixture duration");
    let duration = input.duration();
    if duration <= 0 {
        eprintln!("skipping: fixture has no known container duration");
        return;
    }
    drop(input);
    let target = Duration::from_micros(duration as u64);

    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = source.stream_parameters(video.index).expect("video params");
    let time_base = source.stream_time_base(video.index).expect("video tb");
    let samples = Arc::new(Mutex::new(Vec::new()));

    let pipeline = Pipeline::new("eof-preview", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", params)?)
            .pipe(Pacer::new("pacer", time_base)?)
            .to(Box::new(PrerollProbe {
                label: "video-terminal",
                time_base,
                samples: Arc::clone(&samples),
                pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
            }))?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    thread::sleep(Duration::from_millis(50));
    pipeline.pause();
    samples.lock().unwrap().clear();
    pipeline
        .seek(target, SeekMode::Accurate)
        .expect("known EOF seek prerolls");
    assert_eq!(
        samples.lock().unwrap().len(),
        1,
        "EOF seek must replace the paused picture with the last frame"
    );
    pipeline.stop();
}

/// Seekable, and deliberately silent once it has repositioned — the shape of
/// a source whose preroll can never complete, so the pipeline's wait runs to
/// its full timeout.
struct MuteAfterSeekSource {
    pp_log: PpLog,
    pad: SrcPad,
    sought: Arc<AtomicBool>,
}

impl Element for MuteAfterSeekSource {
    fn name(&self) -> Arc<str> {
        "mute-after-seek".into()
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

impl Source for MuteAfterSeekSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for MuteAfterSeekSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            if drain_control(control, self, bus)?.stopped {
                return Ok(());
            }
            if !self.sought.load(Ordering::Acquire) && self.pad.ready_consume() {
                self.pad
                    .push(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))?;
            }
            thread::yield_now();
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.sought.store(true, Ordering::Release);
        Ok(target)
    }
}

/// `stop` promises to abandon immediately. A seek holds the operation lock
/// for the whole of its preroll wait, so without a way to end that wait from
/// outside the lock, stopping during a seek that cannot preroll waited out the
/// full timeout — and the cancellation terminals already forward on `Stop`
/// could not arrive either, since sending it needs the very same lock.
#[test]
fn stopping_during_a_seek_does_not_wait_out_the_preroll_timeout() {
    let sought = Arc::new(AtomicBool::new(false));
    let seen = Arc::new(AtomicUsize::new(0));
    let source = MuteAfterSeekSource {
        pp_log: element_pp_log(ElementType::Other, "mute-after-seek", None),
        pad: SrcPad::new("src"),
        sought: Arc::clone(&sought),
    };
    let pipeline = Pipeline::new("stop-during-seek", source, |source, ctx| {
        let branch = ctx.branch().to(Box::new(CountingSink {
            name: "sink".into(),
            count: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "sink", None),
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    while seen.load(Ordering::SeqCst) == 0 {
        thread::yield_now();
    }

    let seeking = Arc::clone(&pipeline);
    let seek = thread::spawn(move || seeking.seek(Duration::from_secs(2), SeekMode::Accurate));
    // The seek has to be inside its preroll wait for this to prove anything.
    while !sought.load(Ordering::Acquire) {
        thread::yield_now();
    }

    let started = Instant::now();
    pipeline.stop();
    let elapsed = started.elapsed();

    let outcome = seek.join().expect("seek thread");
    assert!(
        outcome.is_err(),
        "a cancelled preroll must report failure, not success"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "stop waited {elapsed:?} for a preroll it was abandoning"
    );
}

/// Never accepts a buffer, so a `Queue` in front of it parks and the terminal
/// behind it never reports a preroll sample — a branch that cannot preroll.
struct NeverReadySink {
    pp_log: PpLog,
}

impl Element for NeverReadySink {
    fn name(&self) -> Arc<str> {
        "never-ready".into()
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

impl Sink for NeverReadySink {
    fn ready_consume(&mut self) -> bool {
        false
    }

    fn consume(&mut self, _buf: MediaBuffer) -> Result<()> {
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// Preroll expects the terminals the graph had when the seek started. A branch
/// detached while that seek is still waiting takes its terminal out of the
/// graph but not out of the expected set, so the seek waited for a sample
/// nobody was left to produce.
#[test]
fn detaching_a_branch_mid_seek_does_not_strand_its_preroll() {
    let seen = Arc::new(AtomicUsize::new(0));
    let source = SeekLoopSource {
        pp_log: element_pp_log(ElementType::Other, "seek-loop", None),
        pad: SrcPad::new("src"),
        seeks: Arc::new(AtomicUsize::new(0)),
    };

    let handle = Arc::new(Mutex::new(None));
    let stash = Arc::clone(&handle);
    let pipeline = Pipeline::new("detach-mid-seek", source, move |source, ctx| {
        let live = ctx.branch().to(Box::new(CountingSink {
            name: "live".into(),
            count: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "live", None),
        }))?;
        let (tee, tee_handle) = TeeBuilder::new("tee", ctx.clone())
            .branch(live)
            .build_dynamic()?;
        ctx.attach(source, 0, tee)?;
        *stash.lock().unwrap() = Some(tee_handle);
        Ok(())
    })
    .expect("pipeline wiring");
    let tee = handle.lock().unwrap().take().expect("tee handle");

    // A branch that can never take a preroll sample: the queue in front of it
    // parks because its terminal is never ready.
    let stuck = tee
        .branch()
        .expect("dynamic tee")
        .queue("stuck-queue", 4)
        .to(Box::new(NeverReadySink {
            pp_log: element_pp_log(ElementType::Other, "never-ready", None),
        }))
        .expect("stuck branch");
    let stuck_id = tee.attach(stuck).expect("attach stuck branch");

    pipeline.run().expect("run");
    pipeline.pause();

    let seeking = Arc::clone(&pipeline);
    let seek = thread::spawn(move || seeking.seek(Duration::from_secs(1), SeekMode::Accurate));
    // Give the seek time to reach its preroll wait before the topology moves.
    thread::sleep(Duration::from_millis(200));
    tee.detach(stuck_id).expect("detach mid-seek");

    let started = Instant::now();
    let outcome = seek.join().expect("seek thread");
    let elapsed = started.elapsed();
    pipeline.stop();

    assert!(
        outcome.is_ok(),
        "the seek should complete once the branch it was waiting on is gone, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "the seek waited {elapsed:?} on a terminal that had left the graph"
    );
}
