//! Seeking: the check before it, the pause and preroll around it, and
//! where playback lands.

use super::*;

#[test]
fn seek_check_rejects_a_live_source_before_flushing() {
    let source = TestVideoSource::new("live", TestVideoOptions::default());
    let (pipeline, ()) = Pipeline::new("seek-check", source, |source, ctx| {
        let branch = ctx.branch().to(NoOpSink {
            name: "noop".into(),
            pp_log: element_pp_log(ElementType::Other, "noop", None),
        })?;
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

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
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
    let (pipeline, ()) = Pipeline::new("paused-seek-preroll", source, |source, ctx| {
        let branch = ctx.branch().to(ControlRecordingSink {
            pp_log: element_pp_log(ElementType::Other, "control-recorder", None),
            count: Arc::clone(&count),
            controls: Arc::clone(&controls),
            preroll_targets: Arc::clone(&preroll_targets),
        })?;
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
    let (pipeline, ()) = Pipeline::new("playing-seek-preroll", source, |source, ctx| {
        let branch = ctx.branch().to(ControlRecordingSink {
            pp_log: element_pp_log(ElementType::Other, "control-recorder", None),
            count: Arc::clone(&count),
            controls: Arc::clone(&controls),
            preroll_targets,
        })?;
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
    let (pipeline, ()) = Pipeline::new("test", source, |source, ctx| {
        let pacer = Pacer::new("pacer");
        let branch = ctx.branch().queue("q", 4).pipe(pacer).to(sink)?;
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

    // Paced for the same reason as `seek_repositions_and_playback_continues`
    // — otherwise the file finishes before `seek()` is even called.
    let (pipeline, ()) = Pipeline::new("test", source, |source, ctx| {
        let pacer = Pacer::new("pacer");
        let branch = ctx.branch().queue("q", 4).pipe(pacer).to(NoOpSink {
            name: "noop".into(),
            pp_log: element_pp_log(ElementType::Other, "noop", None),
        })?;
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
    let video_params = video.parameters.clone();
    let audio_params = audio.parameters.clone();
    let video_tb = video.time_base;
    let audio_tb = audio.time_base;

    let target = Duration::from_secs(3);
    let video_samples = Arc::new(Mutex::new(Vec::new()));
    let audio_samples = Arc::new(Mutex::new(Vec::new()));

    let (pipeline, ()) = Pipeline::new("paused-av-seek", source, |source, ctx| {
        // Only the video branch is paced, which is the ordinary shape: an
        // audio renderer schedules itself against its own device clock.
        let video_branch = ctx
            .branch()
            .pipe(SwDecoder::new("video-decoder", video_params)?)
            .pipe(Pacer::new("video-pacer"))
            .queue("video-frames", 8)
            .to(PrerollProbe {
                label: "video",
                time_base: video_tb,
                samples: Arc::clone(&video_samples),
                pp_log: element_pp_log(ElementType::Other, "video", None),
            })?;
        ctx.attach(source, video.index, video_branch)?;
        let audio_branch = ctx
            .branch()
            .queue("audio-packets", 8)
            .pipe(SwDecoder::new("audio-decoder", audio_params)?)
            .to(PrerollProbe {
                label: "audio",
                time_base: audio_tb,
                samples: Arc::clone(&audio_samples),
                pp_log: element_pp_log(ElementType::Other, "audio", None),
            })?;
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
    let params = video.parameters.clone();
    let time_base = video.time_base;
    let packets = Arc::new(AtomicUsize::new(0));
    let frames = Arc::new(Mutex::new(Vec::new()));

    let (pipeline, ()) = Pipeline::new("tee-preroll", source, |source, ctx| {
        let packet_branch = ctx.branch().to(CountingSink {
            name: "packet-terminal".into(),
            count: Arc::clone(&packets),
            pp_log: element_pp_log(ElementType::Other, "packet-terminal", None),
        })?;
        let decoded_branch =
            ctx.branch()
                .pipe(SwDecoder::new("decoder", params)?)
                .to(PrerollProbe {
                    label: "video-terminal",
                    time_base,
                    samples: Arc::clone(&frames),
                    pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
                })?;
        let tee = ctx
            .tee("tee")
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
    let params = video.parameters.clone();
    let time_base = video.time_base;
    let samples = Arc::new(Mutex::new(Vec::new()));

    let (pipeline, ()) = Pipeline::new("eof-preview", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", params)?)
            .pipe(Pacer::new("pacer"))
            .to(PrerollProbe {
                label: "video-terminal",
                time_base,
                samples: Arc::clone(&samples),
                pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
            })?;
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
    let (pipeline, ()) = Pipeline::new("stop-during-seek", source, |source, ctx| {
        let branch = ctx.branch().to(CountingSink {
            name: "sink".into(),
            count: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "sink", None),
        })?;
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

    let (pipeline, tee) = Pipeline::new("detach-mid-seek", source, move |source, ctx| {
        let live = ctx.branch().to(CountingSink {
            name: "live".into(),
            count: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "live", None),
        })?;
        let (tee, tee_handle) = ctx.tee("tee").branch(live).build_dynamic()?;
        ctx.attach(source, 0, tee)?;
        Ok(tee_handle)
    })
    .expect("pipeline wiring");

    // A branch that can never take a preroll sample: the queue in front of it
    // parks because its terminal is never ready.
    let stuck = tee
        .branch()
        .expect("dynamic tee")
        .queue("stuck-queue", 4)
        .to(NeverReadySink {
            pp_log: element_pp_log(ElementType::Other, "never-ready", None),
        })
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

/// Pushes `buffers` packets and then `Eos`, and does both again after every
/// seek — a file played to its end and parked there, then played to it
/// again from wherever it was sent back to.
struct EndingSource {
    pp_log: PpLog,
    pad: SrcPad,
    buffers: usize,
    /// Whether the stream from the last start or seek still has to be sent.
    owed: bool,
}

impl Element for EndingSource {
    fn name(&self) -> Arc<str> {
        "ending".into()
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

impl Source for EndingSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for EndingSource {
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
            if self.owed {
                self.owed = false;
                for _ in 0..self.buffers {
                    self.pad
                        .push(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))?;
                }
                self.pad.push(MediaBuffer::Eos)?;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.owed = true;
        Ok(target)
    }
}

/// Reads the bus until `name` posts `Eos`, failing rather than hanging if it
/// never does.
fn wait_for_eos_from(pipeline: &Pipeline, name: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match pipeline.bus().try_recv() {
            Some(BusEvent::Eos { name: from, .. }) if &*from == name => return,
            Some(_) => {}
            None => thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("{name} never reached the end of its stream");
}

/// A pipeline played to its end can be sought back into, through a `Queue`.
///
/// A queue's worker used to end with the `Eos` it forwarded. The source,
/// parked at its end, took the seek and sent its stream again — which then
/// stopped at the queue, so the terminal behind it never reported a preroll
/// sample and the seek failed with `PrerollError::TimedOut` five seconds
/// later. The same pipeline without the queue always worked.
#[test]
fn a_queued_branch_can_be_sought_after_it_has_played_to_its_end() {
    let seen = Arc::new(AtomicUsize::new(0));
    let source = EndingSource {
        pp_log: element_pp_log(ElementType::Other, "ending", None),
        pad: SrcPad::new("ending_src"),
        buffers: 3,
        owed: true,
    };
    let (pipeline, ()) = Pipeline::new("seek-after-end", source, |source, ctx| {
        let branch = ctx.branch().queue("queue", 8).to(CountingSink {
            name: "sink".into(),
            count: Arc::clone(&seen),
            pp_log: element_pp_log(ElementType::Other, "sink", None),
        })?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.run().expect("run");
    wait_for_eos_from(&pipeline, "sink");
    assert_eq!(seen.load(Ordering::SeqCst), 3);

    let started = Instant::now();
    let outcome = pipeline.seek(Duration::ZERO, SeekMode::Keyframe);
    let elapsed = started.elapsed();
    assert!(
        outcome.is_ok(),
        "seeking back into an ended stream failed after {elapsed:?}: {outcome:?}"
    );
    wait_for_eos_from(&pipeline, "sink");
    pipeline.stop();
    assert_eq!(
        seen.load(Ordering::SeqCst),
        6,
        "the whole second stream crossed the queue"
    );
}

/// Paused before it runs, a pipeline starts with every source stopped ahead
/// of its first buffer: nothing reaches the terminal until it is resumed.
/// A `pause` before `run` used to be ignored, and the run started playing.
#[test]
fn a_pipeline_paused_before_it_runs_produces_nothing_until_resumed() {
    let count = Arc::new(AtomicUsize::new(0));
    let (pipeline, ()) = Pipeline::new(
        "start-paused",
        TestVideoSource::new("gen", TestVideoOptions::default()),
        |source, ctx| {
            let branch = ctx.branch().queue("queue", 4).to(CountingSink {
                pp_log: element_pp_log(ElementType::Other, "sink", None),
                name: "sink".into(),
                count: Arc::clone(&count),
            })?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        },
    )
    .expect("pipeline wiring");

    pipeline.pause();
    pipeline.run().expect("run");
    assert!(pipeline.is_running(), "it runs, paused");
    thread::sleep(Duration::from_millis(150));
    assert_eq!(
        count.load(Ordering::SeqCst),
        0,
        "nothing is produced while paused"
    );

    pipeline.resume();
    thread::sleep(Duration::from_millis(150));
    assert!(count.load(Ordering::SeqCst) > 0, "resuming starts it");
    pipeline.stop();
}

/// Paused before it runs and then sought, a pipeline puts exactly one frame
/// through its terminal — the one covering the position asked for — and
/// nothing from before it: a single frame from half way into a file without
/// decoding the first half, which used to need polling and discarding.
#[test]
fn a_seek_from_a_paused_start_delivers_just_the_frame_asked_for() {
    let Some(path) = try_test_video() else { return };
    let input = ffmpeg::format::input(&path).expect("open fixture duration");
    let duration = input.duration();
    let rate = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .map(|stream| stream.avg_frame_rate())
        .filter(|rate| rate.numerator() > 0 && rate.denominator() > 0);
    drop(input);
    let (true, Some(rate)) = (duration > 0, rate) else {
        eprintln!("skipping: fixture has no known duration or frame rate");
        return;
    };
    let target = Duration::from_micros(duration as u64 / 2);
    let frame_ns = 1_000_000_000i64 * i64::from(rate.denominator()) / i64::from(rate.numerator());

    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = video.parameters.clone();
    let time_base = video.time_base;
    let samples = Arc::new(Mutex::new(Vec::new()));

    let (pipeline, ()) = Pipeline::new("paused-seek", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", params)?)
            .to(PrerollProbe {
                label: "video-terminal",
                time_base,
                samples: Arc::clone(&samples),
                pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
            })?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    pipeline.pause();
    pipeline.run().expect("run");
    assert!(
        samples.lock().unwrap().is_empty(),
        "nothing before the seek"
    );

    pipeline
        .seek(target, SeekMode::Accurate)
        .expect("seek from a paused start");
    let got = samples.lock().unwrap().clone();
    let target_ns = target.as_nanos() as i64;
    assert_eq!(
        got.len(),
        1,
        "exactly one frame reached the terminal: {got:?}"
    );
    assert!(
        got[0] <= target_ns && target_ns < got[0] + frame_ns,
        "the frame at {} ns does not cover {target_ns} ns",
        got[0]
    );

    thread::sleep(Duration::from_millis(100));
    assert_eq!(
        samples.lock().unwrap().len(),
        1,
        "and it stays paused after"
    );
    pipeline.stop();
}

/// A seek while the video is waiting on an audio clock that has stopped
/// moving — the moment a player seeks just after resuming, with the picture
/// demuxed a second ahead of the sound.
///
/// `seek` asked every branch whether it could seek before it interrupted the
/// clock. The `VideoSynchronizer` behind the queue was waiting for the audio
/// to reach its frame; the audio could not, because the demuxer had stopped
/// to put that question to the queue; and the queue could not take the
/// question until the synchroniser returned. Nothing timed out, so `seek`
/// never returned. Here the audio clock is held still outright, which is
/// that deadlock every time rather than now and then.
#[test]
fn a_seek_while_video_waits_on_a_stalled_audio_clock_returns() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = video.parameters.clone();
    let time_base = video.time_base;
    let samples = Arc::new(Mutex::new(Vec::new()));

    let (pipeline, ()) = Pipeline::new("stalled-audio-seek", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", params)?)
            .queue("video-frames", 4)
            .pipe(VideoSynchronizer::new("video-sync"))
            .to(PrerollProbe {
                label: "video-terminal",
                time_base,
                samples: Arc::clone(&samples),
                pp_log: element_pp_log(ElementType::Other, "video-terminal", None),
            })?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    // An audio renderer that has played nothing and is not playing: the
    // video can never come due.
    let audio = pipeline
        .playback_clock()
        .register_audio_master()
        .expect("the one audio master");
    audio
        .publish(0, 0, false)
        .expect("publish a stalled position");

    pipeline.run().expect("run");
    // Long enough for the queue to fill and the demuxer to block on it.
    thread::sleep(Duration::from_millis(300));

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let seeking = Arc::clone(&pipeline);
    thread::spawn(move || {
        let _ = done_tx.send(seeking.seek(Duration::from_secs(1), SeekMode::Accurate));
    });
    let outcome = done_rx
        .recv_timeout(Duration::from_secs(15))
        .expect("seek never returned: the branch is deadlocked on the audio clock");
    outcome.expect("the seek completes");
    pipeline.stop();
}

/// Two sources, each its own paced branch — one demuxer for the picture and
/// one for the sound of the same file, which is how a newcomer's player
/// ended up wired — sought twice, half a second apart.
///
/// Control went to one source at a time, after the clock had been
/// interrupted. A source waiting its turn behind the other's cascade was
/// interrupted with nothing to take, so it read on, its interrupted `Pacer`
/// handing each buffer straight back, and reached the end of the file in
/// milliseconds. Its thread had ended by the time its own `Seek` was sent,
/// and the seek failed with its terminal's preroll timed out — on three
/// runs in four here. The sound now stays where the seek put it.
#[test]
fn two_paced_sources_on_one_file_survive_consecutive_seeks() {
    let Some(path) = try_test_video() else { return };
    let (video_source, streams) = FileDemuxer::open("video-demux", &path).expect("open");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("a video stream")
        .clone();
    let (audio_source, streams) = FileDemuxer::open("audio-demux", &path).expect("open");
    let audio = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Audio)
        .expect("an audio stream")
        .clone();
    let video_samples = Arc::new(Mutex::new(Vec::new()));
    let audio_samples = Arc::new(Mutex::new(Vec::new()));

    let (builder, ()) = PipelineBuilder::new("two-demuxers")
        .add_source(video_source, |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
                .pipe(Pacer::new("video-pacer"))
                .to(PrerollProbe {
                    label: "video",
                    time_base: video.time_base,
                    samples: Arc::clone(&video_samples),
                    pp_log: element_pp_log(ElementType::Other, "video", None),
                })?;
            ctx.attach(source, video.index, branch)?;
            Ok(())
        })
        .expect("wire the picture");
    let (builder, ()) = builder
        .add_source(audio_source, |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
                .pipe(Pacer::new("audio-pacer"))
                .to(PrerollProbe {
                    label: "audio",
                    time_base: audio.time_base,
                    samples: Arc::clone(&audio_samples),
                    pp_log: element_pp_log(ElementType::Other, "audio", None),
                })?;
            ctx.attach(source, audio.index, branch)?;
            Ok(())
        })
        .expect("wire the sound");
    let pipeline = builder.build();

    pipeline.run().expect("run");
    thread::sleep(Duration::from_millis(300));
    pipeline
        .seek(Duration::from_secs(1), SeekMode::Accurate)
        .expect("the first seek");
    thread::sleep(Duration::from_millis(500));
    let last_audio_ns = *audio_samples.lock().unwrap().last().expect("audio played");
    assert!(
        last_audio_ns < 2_500_000_000,
        "half a second after seeking to 1 s, the sound is at {last_audio_ns} ns: \
         it ran ahead unpaced"
    );
    pipeline
        .seek(Duration::from_millis(500), SeekMode::Accurate)
        .expect("the second seek");
    pipeline.stop();
}

/// `position` is where playback is, read from the playback clock: nothing
/// before the run, about as far as the wall clock has gone while it plays,
/// still while paused, and on from where a seek put it — which a player
/// used to have to rebuild from the timestamps of the frames going past.
#[test]
fn position_follows_playback_through_pause_and_seek() {
    let Some(path) = try_test_video() else { return };
    let input = ffmpeg::format::input(&path).expect("open fixture duration");
    let duration = input.duration();
    drop(input);
    if duration <= 0 {
        eprintln!("skipping: fixture has no known container duration");
        return;
    }
    let target = Duration::from_micros(duration as u64 / 2);

    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("a video stream")
        .clone();
    let count = Arc::new(AtomicUsize::new(0));
    let (pipeline, ()) = Pipeline::new("position", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("decoder", video.parameters.clone())?)
            .pipe(Pacer::new("pacer"))
            .to(CountingSink {
                pp_log: element_pp_log(ElementType::Other, "screen", None),
                name: "screen".into(),
                count: Arc::clone(&count),
            })?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");

    assert_eq!(
        pipeline.position(),
        None,
        "nothing has played before the run"
    );
    pipeline.run().expect("run");
    thread::sleep(Duration::from_millis(500));
    let playing = pipeline
        .position()
        .expect("a paced pipeline has a position");
    assert!(
        playing > Duration::from_millis(250) && playing < Duration::from_millis(900),
        "half a second in, the position is {playing:?}"
    );

    pipeline.pause();
    let paused = pipeline.position().expect("paused, still somewhere");
    thread::sleep(Duration::from_millis(300));
    let later = pipeline.position().expect("still somewhere");
    assert!(
        later.saturating_sub(paused) < Duration::from_millis(40),
        "paused at {paused:?}, then {later:?}"
    );

    pipeline
        .seek(target, SeekMode::Accurate)
        .expect("seek while paused");
    pipeline.resume();
    thread::sleep(Duration::from_millis(300));
    let after = pipeline.position().expect("playing again");
    assert!(
        after >= target.saturating_sub(Duration::from_millis(40))
            && after < target + Duration::from_millis(900),
        "sought to {target:?}, then at {after:?}"
    );
    pipeline.stop();
}

/// A paused seek close enough to the end that the file runs out before the
/// preroll has its picture — a decoder holding several pictures in flight
/// needs packets past the target to hand the target on, and past the end
/// there are none, so only the end of stream drains it. The source reaches
/// its end while the pipeline is paused, with the queue downstream holding
/// the pictures and the `Eos` behind them until it is resumed.
///
/// The demuxer used to end its thread there, dropping the queue with them in
/// it, so the resume never reached them and the file never finished.
#[test]
fn a_paused_seek_that_runs_into_the_end_still_finishes_once_resumed() {
    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let duration = source.duration().expect("the fixture says how long it is");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = video.parameters.clone();
    let index = video.index;
    let decoder = crate::elements::SwDecoder::with_threading(
        "video-decoder",
        params,
        crate::elements::DecodeThreading {
            threads: None,
            kind: crate::elements::DecodeThreadKind::Frame,
        },
    )
    .expect("the fixture's decoder opens");

    let (pipeline, ()) = Pipeline::new("paused-seek-to-end", source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(decoder)
            .queue("video-frames", 32)
            .pipe(Pacer::new("video-pacer"))
            .to(NoOpSink {
                name: "screen".into(),
                pp_log: element_pp_log(ElementType::Other, "screen", None),
            })?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(100));
    pipeline.pause();
    pipeline
        .seek(
            duration.saturating_sub(Duration::from_millis(50)),
            SeekMode::Accurate,
        )
        .expect("a file accepts a seek");
    pipeline.resume();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let finished = loop {
        match pipeline.bus().recv_timeout(Duration::from_millis(100)) {
            Ok(BusEvent::Finished) => break true,
            Ok(_) => {}
            Err(_) if std::time::Instant::now() > deadline => break false,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break false,
            Err(_) => {}
        }
    };
    pipeline.stop();
    assert!(finished, "the file never finished after resuming");
}

/// Reads the bus until `Finished`, or gives up at `timeout`.
fn wait_for_finished(pipeline: &Pipeline, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        match pipeline.bus().recv_timeout(Duration::from_millis(50)) {
            Ok(BusEvent::Finished) => return true,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return false,
            _ => {}
        }
    }
    false
}

/// A file decoded into a queue and paced out of it, counting what reaches
/// the end — the shape of any player.
fn paced_file(name: &str) -> Option<(Arc<Pipeline>, Duration, Arc<AtomicUsize>)> {
    let path = try_test_video()?;
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open test video");
    let duration = source.duration().expect("the fixture says how long it is");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("test video has a video stream");
    let params = video.parameters.clone();
    let index = video.index;
    let shown = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&shown);
    let (pipeline, ()) = Pipeline::new(name, source, |source, ctx| {
        let branch = ctx
            .branch()
            .pipe(SwDecoder::new("video-decoder", params)?)
            .queue("video-frames", 32)
            .pipe(Pacer::new("video-pacer"))
            .to(CountingSink {
                pp_log: element_pp_log(ElementType::Other, "screen", None),
                name: "screen".into(),
                count,
            })?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");
    Some((pipeline, duration, shown))
}

/// The demuxer reaches the end of the file a queue's depth before the
/// picture does. It used to end its thread there, taking the queue with it,
/// so a pause in that last second reached nothing and the picture played on
/// to the end.
#[test]
fn a_pause_in_the_last_second_still_holds_the_picture() {
    let Some((pipeline, duration, shown)) = paced_file("pause-at-the-end") else {
        return;
    };
    pipeline.run().unwrap();
    // Far enough from the end that the queue fills before the file runs
    // out, near enough that it runs out while the queue still plays.
    pipeline
        .seek(
            duration.saturating_sub(Duration::from_millis(1_500)),
            SeekMode::Accurate,
        )
        .expect("a file accepts a seek");
    thread::sleep(Duration::from_millis(700));

    pipeline.pause();
    let held = shown.load(Ordering::SeqCst);
    thread::sleep(Duration::from_millis(400));
    let after = shown.load(Ordering::SeqCst);
    pipeline.resume();
    let finished = wait_for_finished(&pipeline, Duration::from_secs(5));
    pipeline.stop();

    assert!(
        after <= held + 1,
        "{} pictures went by while paused",
        after - held
    );
    assert!(finished, "the file finished once resumed");
}

/// Once a file has played to its end, a seek plays it again from where it
/// lands, and the pipeline finishes a second time.
#[test]
fn a_file_can_be_sought_back_after_it_finished() {
    let Some((pipeline, duration, shown)) = paced_file("seek-after-the-end") else {
        return;
    };
    pipeline.run().unwrap();
    pipeline
        .seek(
            duration.saturating_sub(Duration::from_millis(300)),
            SeekMode::Accurate,
        )
        .expect("a file accepts a seek");
    assert!(
        wait_for_finished(&pipeline, Duration::from_secs(5)),
        "the file finished"
    );

    let before = shown.load(Ordering::SeqCst);
    pipeline
        .seek(
            duration.saturating_sub(Duration::from_millis(600)),
            SeekMode::Accurate,
        )
        .expect("a file that has finished can still be sought");
    let finished_again = wait_for_finished(&pipeline, Duration::from_secs(5));
    pipeline.stop();
    assert!(finished_again, "and finished again");
    assert!(
        shown.load(Ordering::SeqCst) > before,
        "pictures were shown from where it landed"
    );
}

/// Takes every control request and passes it on, but never waits out a
/// `Pause`: it goes on reading into a paused queue — the state
/// `FileDemuxer` was in at the end of its file before 208af56.
struct UnpausingSource {
    pp_log: PpLog,
    pad: SrcPad,
}

impl Element for UnpausingSource {
    fn name(&self) -> Arc<str> {
        "unpausing".into()
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

impl Source for UnpausingSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for UnpausingSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            while let Some((request, ack)) = control.try_recv() {
                let crate::control::RequestKind::Control(msg) = request else {
                    let _ = ack.send(());
                    return Ok(());
                };
                if crate::control::apply_one(self, bus, &msg, &ack)? {
                    return Ok(());
                }
            }
            self.pad
                .push(MediaBuffer::Packet(Arc::new(ffmpeg_next::Packet::empty())))?;
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        Ok(target)
    }
}

/// Every control request lets go of a thread blocked handing data on, not
/// only the ones that were expected to meet one.
///
/// A source that fails to pause fills the paused queue behind it and blocks
/// handing it the next packet. Only some requests used to raise the
/// interrupt that makes a queue take such a packet as held over; the seek's
/// `Seek` and `Preroll` did not, since the graph is paused when they are
/// sent — so the seek waited for good on a source that could not read them.
#[test]
fn a_seek_reaches_a_source_that_did_not_pause() {
    let source = UnpausingSource {
        pp_log: element_pp_log(ElementType::Other, "unpausing", None),
        pad: SrcPad::new("src"),
    };
    let (pipeline, ()) = Pipeline::new("unpausing", source, |source, ctx| {
        let branch = ctx.branch().queue("q", 1).to(NoOpSink {
            name: "noop".into(),
            pp_log: element_pp_log(ElementType::Other, "noop", None),
        })?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");
    pipeline.run().expect("run");
    pipeline.pause();

    let seeking = Arc::clone(&pipeline);
    let (done, finished) = mpsc::channel();
    thread::spawn(move || {
        let _ = done.send(seeking.seek(Duration::from_secs(1), SeekMode::Keyframe));
    });
    let outcome = finished
        .recv_timeout(Duration::from_secs(10))
        .expect("the seek never returned");
    assert!(outcome.is_ok(), "{outcome:?}");
    pipeline.stop();
}

/// The branch whose preroll finishes first keeps its stream for when
/// playback goes on, however long the other branch takes.
///
/// A decoder that had handed its branch the preroll's sample threw away
/// whatever it decoded after, until the preroll ended — and went on taking
/// packets all the while, since nothing said it was full. With the picture
/// slow to preroll, the sound's decoder was fed and emptied the whole rest
/// of the file in that time: a player resumed after a seek with no sound at
/// all. Found by the conformance sequences on two cores.
#[test]
fn a_seek_keeps_the_sound_that_prerolled_before_the_picture() {
    use crate::elements::AppSink;

    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
    let find = |kind| {
        streams
            .iter()
            .find(|stream| stream.kind == kind)
            .expect("the fixture has picture and sound")
            .clone()
    };
    let (video, audio) = (
        find(ffmpeg::media::Type::Video),
        find(ffmpeg::media::Type::Audio),
    );
    let heard: Arc<Mutex<Vec<Option<i64>>>> = Arc::new(Mutex::new(Vec::new()));
    let speakers = {
        let heard = Arc::clone(&heard);
        AppSink::new("speakers", move |buffer| {
            if let MediaBuffer::Audio(frame) = buffer {
                heard.lock().unwrap().push(frame.pts());
            }
            Ok(())
        })
    };
    // Slow to take a picture, so the sound's preroll is long done before the
    // picture's is.
    let screen = AppSink::new("screen", |_| {
        thread::sleep(Duration::from_millis(400));
        Ok(())
    });
    let (pipeline, ()) = Pipeline::new("sound-before-picture", source, |source, ctx| {
        let picture = ctx
            .branch()
            .queue("video-packets", 8)
            .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
            .queue("video-frames", 2)
            .to(screen)?;
        ctx.attach(source, video.index, picture)?;
        let sound = ctx
            .branch()
            .queue("audio-packets", 8)
            .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
            .queue("audio-frames", 4)
            .to(speakers)?;
        ctx.attach(source, audio.index, sound)?;
        Ok(())
    })
    .expect("wire the pipeline");
    pipeline.run().unwrap();
    pipeline.pause();
    pipeline
        .seek(Duration::from_secs(1), SeekMode::Accurate)
        .expect("a file accepts a seek");
    let prerolled = heard.lock().unwrap().len();
    pipeline.resume();
    let waited = Instant::now();
    while heard.lock().unwrap().len() < prerolled + 3 {
        assert!(
            waited.elapsed() < Duration::from_secs(5),
            "no sound after the seek's own: {:?}",
            heard.lock().unwrap()
        );
        thread::sleep(Duration::from_millis(10));
    }
    pipeline.stop();

    let heard = heard.lock().unwrap();
    let base = audio.time_base;
    let at = |pts: Option<i64>| {
        pts.map(|pts| pts as f64 * f64::from(base.numerator()) / f64::from(base.denominator()))
    };
    let (first, next) = (at(heard[prerolled - 1]), at(heard[prerolled]));
    // The sound goes on from where the seek left it: one packet later, not
    // wherever the file had been read to meanwhile.
    assert!(
        matches!((first, next), (Some(first), Some(next)) if next - first < 0.1),
        "the sound jumped from {first:?} to {next:?} across the resume"
    );
}

/// A branch of a `Tee` that prerolls first loses nothing while its sibling
/// catches up: after the seek it goes on from the sample it prerolled on,
/// not from wherever the source had got to meanwhile.
#[test]
fn a_tee_branch_that_prerolls_first_misses_nothing() {
    use crate::elements::AppSink;

    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("the fixture has a picture")
        .clone();
    let shown: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let screen = {
        let shown = Arc::clone(&shown);
        AppSink::new("screen", move |buffer| {
            if let MediaBuffer::Video(frame) = buffer
                && let Some(pts) = frame.pts()
            {
                shown.lock().unwrap().push(pts);
            }
            Ok(())
        })
    };
    // Slow to take a picture, so its preroll finishes well after the other
    // branch's.
    let slow = AppSink::new("slow-screen", |_| {
        thread::sleep(Duration::from_millis(150));
        Ok(())
    });
    let (pipeline, ()) = Pipeline::new("tee-preroll-gap", source, |source, ctx| {
        let fast = ctx.branch().queue("fast", 2).to(screen)?;
        let slow = ctx.branch().queue("slow", 2).to(slow)?;
        let tee = ctx.tee("tee").branch(fast).branch(slow).build()?;
        let picture = ctx
            .branch()
            .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
            .queue("video-frames", 4)
            .to_branch(tee)?;
        ctx.attach(source, video.index, picture)?;
        Ok(())
    })
    .expect("wire the pipeline");
    pipeline.run().unwrap();
    pipeline.pause();
    pipeline
        .seek(Duration::from_secs(2), SeekMode::Accurate)
        .expect("a file accepts a seek");
    let from = shown.lock().unwrap().len() - 1;
    pipeline.resume();
    let waited = Instant::now();
    while shown.lock().unwrap().len() < from + 10 {
        assert!(
            waited.elapsed() < Duration::from_secs(10),
            "playback stalled"
        );
        thread::sleep(Duration::from_millis(10));
    }
    pipeline.stop();

    let shown = shown.lock().unwrap();
    let after: Vec<i64> = shown[from..].to_vec();
    let step = after
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .min()
        .unwrap_or(1)
        .max(1);
    assert!(
        after.windows(2).all(|pair| pair[1] - pair[0] <= step),
        "pictures went missing after the seek: {after:?}"
    );
}

/// The same with the `Tee` in front of the decoders, fanning out packets —
/// where a branch that lost some would lose the pictures that depend on
/// them too.
#[test]
fn a_tee_of_packets_whose_branch_prerolls_first_misses_nothing() {
    use crate::elements::AppSink;

    let Some(path) = try_test_video() else { return };
    let (source, streams) = FileDemuxer::open("demux", &path).expect("open the fixture");
    let video = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("the fixture has a picture")
        .clone();
    let shown: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
    let screen = {
        let shown = Arc::clone(&shown);
        AppSink::new("screen", move |buffer| {
            if let MediaBuffer::Video(frame) = buffer
                && let Some(pts) = frame.pts()
            {
                shown.lock().unwrap().push(pts);
            }
            Ok(())
        })
    };
    let slow = AppSink::new("slow-screen", |_| {
        thread::sleep(Duration::from_millis(150));
        Ok(())
    });
    let (pipeline, ()) = Pipeline::new("tee-packets-preroll-gap", source, |source, ctx| {
        let fast = ctx
            .branch()
            .queue("fast-packets", 8)
            .pipe(SwDecoder::new("fast-decoder", video.parameters.clone())?)
            .queue("fast", 2)
            .to(screen)?;
        let slow = ctx
            .branch()
            .queue("slow-packets", 8)
            .pipe(SwDecoder::new("slow-decoder", video.parameters.clone())?)
            .queue("slow", 2)
            .to(slow)?;
        let tee = ctx.tee("tee").branch(fast).branch(slow).build()?;
        ctx.attach(source, video.index, tee)?;
        Ok(())
    })
    .expect("wire the pipeline");
    pipeline.run().unwrap();
    pipeline.pause();
    pipeline
        .seek(Duration::from_secs(2), SeekMode::Accurate)
        .expect("a file accepts a seek");
    let from = shown.lock().unwrap().len() - 1;
    pipeline.resume();
    let waited = Instant::now();
    while shown.lock().unwrap().len() < from + 10 {
        assert!(
            waited.elapsed() < Duration::from_secs(10),
            "playback stalled"
        );
        thread::sleep(Duration::from_millis(10));
    }
    pipeline.stop();

    let shown = shown.lock().unwrap();
    let after: Vec<i64> = shown[from..].to_vec();
    let step = after
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .min()
        .unwrap_or(1)
        .max(1);
    assert!(
        after.windows(2).all(|pair| pair[1] - pair[0] <= step),
        "pictures went missing after the seek: {after:?}"
    );
}

/// Reads on regardless, like `UnpausingSource`, and stamps what it reads
/// with where it is: a millisecond a packet, from wherever the last seek put
/// it.
struct ReadingOnSource {
    pp_log: PpLog,
    pad: SrcPad,
    at_ms: i64,
}

impl Element for ReadingOnSource {
    fn name(&self) -> Arc<str> {
        "reading-on".into()
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

impl Source for ReadingOnSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for ReadingOnSource {
    fn is_live(&self) -> bool {
        false
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        loop {
            while let Some((request, ack)) = control.try_recv() {
                let crate::control::RequestKind::Control(msg) = request else {
                    let _ = ack.send(());
                    return Ok(());
                };
                if crate::control::apply_one(self, bus, &msg, &ack)? {
                    return Ok(());
                }
            }
            let mut packet = ffmpeg_next::Packet::empty();
            packet.set_pts(Some(self.at_ms));
            packet.set_time_base(ffmpeg_next::Rational::new(1, 1000));
            self.at_ms += 1;
            self.pad.push(MediaBuffer::Packet(Arc::new(packet)))?;
        }
    }

    fn seek(&mut self, target: Duration) -> Result<Duration> {
        self.at_ms = target.as_millis() as i64;
        Ok(target)
    }
}

/// What a source read before it applied a seek does not reach the
/// terminal after it, even where it slipped past the seek's `Flush`.
///
/// A source that reads on while the seek is cascading puts old packets into
/// the queue behind it after the `Flush` has emptied that queue and before
/// the source has repositioned. Nothing in the cascade can tell them from
/// the new position's; the queue delivered them first, and the seek's own
/// preview was a picture from before it. Each packet now carries the number
/// of the timeline it was read on, and the queue drops what is behind.
#[test]
fn what_was_read_before_a_seek_does_not_arrive_after_it() {
    let source = ReadingOnSource {
        pp_log: element_pp_log(ElementType::Other, "reading-on", None),
        pad: SrcPad::new("src"),
        at_ms: 0,
    };
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = crate::elements::AppSink::with_control(
        "recorder",
        {
            let seen = Arc::clone(&seen);
            move |buffer| {
                if let MediaBuffer::Packet(packet) = buffer {
                    seen.lock()
                        .unwrap()
                        .push(format!("{}", packet.pts().unwrap_or(-1)));
                }
                Ok(())
            }
        },
        {
            let seen = Arc::clone(&seen);
            move |msg| {
                if matches!(msg, ControlMsg::Seek(_)) {
                    seen.lock().unwrap().push("seek".into());
                }
                Ok(())
            }
        },
    );
    let (pipeline, ()) = Pipeline::new("reading-on", source, |source, ctx| {
        let branch = ctx.branch().queue("q", 64).to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("pipeline wiring");
    pipeline.run().expect("run");
    thread::sleep(Duration::from_millis(20));
    let target = Duration::from_secs(1_000);
    pipeline
        .seek(target, SeekMode::Keyframe)
        .expect("the seek completes");
    thread::sleep(Duration::from_millis(20));
    pipeline.stop();

    let seen = seen.lock().unwrap();
    let after: Vec<i64> = seen
        .iter()
        .skip_while(|entry| entry.as_str() != "seek")
        .skip(1)
        .map(|entry| entry.parse().expect("a packet's pts"))
        .collect();
    assert!(!after.is_empty(), "nothing arrived after the seek");
    let stale: Vec<i64> = after
        .iter()
        .copied()
        .filter(|&pts| pts < target.as_millis() as i64)
        .collect();
    assert!(
        stale.is_empty(),
        "{} packet(s) from before the seek arrived after it, the first {:?}",
        stale.len(),
        stale.first()
    );
}
