//! Long-running stress and leak scenarios.
//!
//! Every test here is `#[ignore]`d: they run for tens of seconds each, which
//! does not belong in `cargo test`'s normal path. Run them explicitly:
//!
//! ```text
//! cargo test -p media-pp --features d3d11,d3d12,cuda --test soak -- --ignored --nocapture
//! ```
//!
//! On Linux, `pipewire-screen-capture` replaces `d3d11`.
//!
//! `--nocapture` matters — every scenario prints the series it measured, so
//! a passing run still shows how much headroom it had.
//!
//! Knobs (all optional):
//!
//! - `MEDIA_PP_SOAK_ITERS` — measured iterations per cycle-driven scenario.
//! - `MEDIA_PP_SOAK_SECS` — how long each duration-driven scenario runs.
//! - `MEDIA_PP_TEST_VIDEO` — a real video file. The seek and hardware-decode
//!   scenarios skip without one: no synthetic source in this crate can seek,
//!   and no hardware decoder has anything to decode.
//! - `MEDIA_PP_SOAK_RESTORE_TOKEN` — an xdg-desktop-portal restore token.
//!   The Linux screen-capture scenarios skip without one, since the portal
//!   would otherwise show its picker and block; `cargo run -p screen_record_software
//!   -- out.mp4 2 monitor` prints a token to reuse.
//!
//! What the numbers mean, and how the thresholds were picked, is in
//! `common/mod.rs`; the short version is that each scenario fits a line
//! through its samples and fails on sustained growth, not on a single
//! endpoint delta. The thresholds below are set from measured runs on the
//! development machine with roughly an order of magnitude of headroom over
//! observed noise, so a real leak — a decoder context, a frame pool, a
//! texture set per cycle — trips them immediately while allocator jitter
//! does not.
//!
//! **What a pass here does and does not claim.** A pass is not "there is no
//! leak"; it is "no growth above what this window could resolve", and every
//! scenario prints that figure next to its result (`resolves growth of
//! X/iter and above`). Resolution is set by the noise and the sample count
//! together — the fitted slope's standard error falls off as `n^1.5`, so at
//! the small default counts a scenario resolves a fraction of a MiB per
//! cycle, and `MEDIA_PP_SOAK_ITERS=100` tightens that by roughly 30x.
//! Anything below the printed figure is invisible no matter how often the
//! suite runs at that count, and a window too noisy for its own threshold
//! fails outright rather than passing quietly.

mod common;

use std::{
    path::Path,
    sync::atomic::Ordering,
    thread,
    time::{Duration, Instant},
};

use common::{MIB, TempDir, Trend, iterations, settle, soak_duration, try_test_video};
use ffmpeg_next as ffmpeg;
use media_pp::{
    bus::BusEvent,
    color::Color,
    elements::{
        FileDemuxer, FrameCounter, Mp4Muxer, SegmentPolicy, SegmentedMp4Muxer, SwDecoder,
        SwEncoder, SwEncoderOptions, SwVideoCompositor, TeeBuilder, TestVideoOptions,
        TestVideoSource, VideoCodec, VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
    },
    pipeline::{Pipeline, SeekMode},
};

/// Hands the scenario its own process, and returns from the parent once
/// that child has run (see `common::spawn_isolated`). Every scenario starts
/// with this.
///
/// The name the child is filtered by comes from the enclosing function
/// itself, through the type name of a local item, so it cannot drift out of
/// sync with the test the way a hand-written string would.
macro_rules! isolate {
    () => {{
        fn probe() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let full = type_name_of(probe);
        let path = full.strip_suffix("::probe").unwrap_or(full);
        // `soak::d3d11::scenario` -> `d3d11::scenario`, which is how the
        // test harness names it.
        let name = path.split_once("::").map_or(path, |(_crate, rest)| rest);
        if crate::common::spawn_isolated(name) {
            return;
        }
    }};
}

/// Cycle-driven scenarios discard this many cycles before measuring, so the
/// lazily-initialized FFmpeg tables and the first heap growth are not part
/// of the trend.
const WARMUP: usize = 3;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn frame_rate() -> ffmpeg::Rational {
    ffmpeg::Rational::new(30, 1)
}

fn test_source(name: &str) -> TestVideoSource {
    TestVideoSource::new(
        name,
        TestVideoOptions {
            width: WIDTH,
            height: HEIGHT,
            framerate: frame_rate(),
        },
    )
}

/// How a cycle ends. Both paths tear a running pipeline down, but through
/// different code: `finish` drains EOS through every stage and joins the
/// workers, `stop` abandons buffered work. A leak in either one is a leak,
/// so the scenarios alternate rather than picking a favorite.
#[derive(Clone, Copy, PartialEq)]
enum Teardown {
    Finish,
    Stop,
}

impl Teardown {
    fn for_cycle(cycle: usize) -> Self {
        if cycle.is_multiple_of(2) {
            Self::Finish
        } else {
            Self::Stop
        }
    }

    /// Returns only once the pipeline's threads have actually finished:
    /// `finish` joins them itself, while `stop` merely asks, so draining
    /// the bus to exhaustion — which ends when the last `Bus` sender drops
    /// — is what makes the two comparable. Draining is also how a cycle
    /// notices a failure that a frame count alone would hide, such as a
    /// decode surface pool that ran out partway through.
    fn apply(self, pipeline: &Pipeline) {
        match self {
            Self::Finish => pipeline.finish(),
            Self::Stop => pipeline.stop(),
        }
        let events: Vec<_> = pipeline.bus().iter().collect();
        assert_no_errors(&events);
    }
}

fn assert_no_errors(events: &[BusEvent]) {
    let errors: Vec<_> = events
        .iter()
        .filter(|event| matches!(event, BusEvent::Error { .. }))
        .collect();
    assert!(errors.is_empty(), "unexpected error event(s): {errors:?}");
}

fn encoder(name: &str, time_base: ffmpeg::Rational, gop_size: u32) -> SwEncoder {
    SwEncoder::new(
        name,
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: WIDTH,
            height: HEIGHT,
            time_base,
            frame_rate: frame_rate(),
            bit_rate: 1_000_000,
            gop_size,
        },
    )
    .expect("libopenh264 encoder")
}

/// The fixture's demuxer, the index of its video stream, and that stream's
/// codec parameters — what every scenario that decodes a real file needs,
/// and nothing a particular file has to provide beyond one video track.
fn open_fixture(path: &str) -> (FileDemuxer, usize, ffmpeg::codec::Parameters) {
    let (source, streams) = FileDemuxer::open("demux", path).expect("open the fixture");
    let index = streams
        .iter()
        .find(|stream| stream.kind == ffmpeg::media::Type::Video)
        .expect("the fixture has a video stream")
        .index;
    let parameters = source
        .stream_parameters(index)
        .expect("the stream the demuxer just listed");
    (source, index, parameters)
}

/// One complete recording: build the whole graph, run it, tear it down, and
/// drop every piece — the cycle a caller repeats whenever it records
/// several clips in one process.
fn record_once(path: &Path, teardown: Teardown) {
    let source = test_source("video");
    let time_base = source.time_base();
    let encoder = encoder("encoder", time_base, 30);
    let mut muxer = Mp4Muxer::create(path).expect("create the recording");
    muxer
        .add_stream("video", encoder.parameters(), time_base)
        .expect("add the video track");
    let sink = muxer
        .open()
        .expect("open the recording")
        .pop()
        .expect("one track");

    let pipeline = Pipeline::new("soak-record", source, |source, ctx| {
        let branch = ctx
            .branch()
            .queue("encode-frames", 8)
            .pipe(encoder)
            .to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the recording pipeline");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(250));
    teardown.apply(&pipeline);
}

/// The most common shape of a leak in a library like this one: something
/// that survives a whole build/run/teardown cycle. Encoder, muxer, queue,
/// frame pool, and source thread are all created and destroyed per cycle
/// here, so anything not released shows up as a slope.
#[test]
#[ignore = "soak test; run with --ignored"]
fn repeated_record_cycles_do_not_grow_process_memory() {
    isolate!();
    let _exclusive = common::exclusive();
    media_pp::init().expect("ffmpeg init");
    let dir = TempDir::new("record-cycles");
    let iterations = iterations(20);
    let mut memory = Trend::private_bytes("record cycle private bytes");

    for cycle in 0..(WARMUP + iterations) {
        let path = dir.join("cycle.mp4");
        record_once(&path, Teardown::for_cycle(cycle));
        assert!(path.is_file(), "cycle {cycle} produced no recording");
        std::fs::remove_file(&path)
            .expect("the muxer must have closed the file before the cycle ended");
        if cycle + 1 == WARMUP {
            settle();
        }
        if cycle >= WARMUP {
            memory.sample();
        }
    }

    // Measured noise on the development machine is well under 0.1 MiB per
    // cycle; a leaked encoder context or frame pool would be several.
    memory.assert_flat(0.5 * MIB);
}

/// Pause/resume hammering on one long-lived pipeline. Each round-trip runs
/// the whole synchronous control cascade — source `drain_control`, every
/// `Queue` worker, every `Sink::control` — which is where a per-message
/// allocation or a retained buffer would accumulate.
#[test]
#[ignore = "soak test; run with --ignored"]
fn pause_resume_storm_does_not_grow_process_memory() {
    isolate!();
    let _exclusive = common::exclusive();
    media_pp::init().expect("ffmpeg init");
    let (counter, frames) = FrameCounter::new("counter");
    let pipeline = Pipeline::new("soak-control", test_source("video"), |source, ctx| {
        let branch = ctx.branch().queue("frames", 8).to(Box::new(counter))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the control-storm pipeline");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(200));

    let iterations = iterations(60);
    let mut memory = Trend::private_bytes("control storm private bytes");
    for round in 0..(WARMUP + iterations) {
        pipeline.pause();
        thread::sleep(Duration::from_millis(20));
        pipeline.resume();
        thread::sleep(Duration::from_millis(30));
        if round + 1 == WARMUP {
            settle();
        }
        if round >= WARMUP {
            memory.sample();
        }
    }

    let before_teardown = frames.load(Ordering::Relaxed);
    pipeline.finish();
    assert!(
        before_teardown > 0,
        "the storm paused the pipeline so hard it never delivered a frame"
    );
    // A single control round-trip allocates almost nothing, so this
    // threshold is deliberately tight compared to the cycle scenarios.
    memory.assert_flat(0.25 * MIB);
}

/// Seeking repeatedly through a real file: the one control path that makes
/// a decoder flush, a `Queue` drop its backlog, and a `Pacer` re-anchor,
/// all while frames are in flight. No synthetic source in this crate can
/// seek, so this scenario needs a fixture and skips without one.
#[test]
#[ignore = "soak test; run with --ignored"]
fn seek_storm_does_not_grow_process_memory() {
    isolate!();
    let _exclusive = common::exclusive();
    media_pp::init().expect("ffmpeg init");
    let Some(path) = try_test_video() else { return };
    let (source, index, parameters) = open_fixture(&path);

    let (counter, frames) = FrameCounter::new("counter");
    let pipeline = Pipeline::new("soak-seek", source, |source, ctx| {
        let decoder = SwDecoder::new("decoder", parameters)?;
        let branch = ctx
            .branch()
            .pipe(decoder)
            .queue("frames", 8)
            .to(Box::new(counter))?;
        ctx.attach(source, index, branch)?;
        Ok(())
    })
    .expect("wire the seek-storm pipeline");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(200));

    let iterations = iterations(30);
    let mut memory = Trend::private_bytes("seek storm private bytes");
    for round in 0..(WARMUP + iterations) {
        // Two positions, alternating, so every seek is a real reposition
        // in one direction or the other rather than a no-op.
        let target = if round % 2 == 0 {
            Duration::from_millis(200)
        } else {
            Duration::from_millis(1200)
        };
        pipeline.seek(target, SeekMode::Accurate).expect("seek");
        thread::sleep(Duration::from_millis(60));
        if round + 1 == WARMUP {
            settle();
        }
        if round >= WARMUP {
            memory.sample();
        }
    }

    assert!(
        frames.load(Ordering::Relaxed) > 0,
        "the storm seeked so hard nothing ever decoded"
    );
    pipeline.stop();
    let events: Vec<_> = pipeline.bus().iter().collect();
    assert_no_errors(&events);
    // A seek flushes a decoder and drops a queue backlog, so its per-round
    // footprint is larger than a pause round-trip's but still bounded.
    memory.assert_flat(0.5 * MIB);
}

/// Attaching and detaching a `Tee` branch while frames flow through it —
/// the dynamic-topology path, where a stale branch, sink, or handle from a
/// previous attach is exactly the kind of thing that accumulates.
#[test]
#[ignore = "soak test; run with --ignored"]
fn tee_branch_churn_does_not_grow_process_memory() {
    isolate!();
    let _exclusive = common::exclusive();
    media_pp::init().expect("ffmpeg init");
    let (fixed_counter, fixed_frames) = FrameCounter::new("fixed-counter");
    let mut tee_handle = None;
    let pipeline = Pipeline::new("soak-tee", test_source("video"), |source, ctx| {
        let fixed = ctx.branch().to(Box::new(fixed_counter))?;
        let (tee_branch, handle) = TeeBuilder::new("tee", ctx.clone())
            .branch(fixed)
            .build_dynamic()?;
        ctx.attach(source, 0, tee_branch)?;
        tee_handle = Some(handle);
        Ok(())
    })
    .expect("wire the tee-churn pipeline");
    let tee_handle = tee_handle.expect("the wire closure provides the handle");

    pipeline.run().unwrap();
    thread::sleep(Duration::from_millis(200));

    let iterations = iterations(40);
    let mut memory = Trend::private_bytes("tee churn private bytes");
    for round in 0..(WARMUP + iterations) {
        let (counter, _frames) = FrameCounter::new("churn-counter");
        let branch = tee_handle
            .branch()
            .expect("the tee is alive while its pipeline runs")
            .to(Box::new(counter))
            .expect("build the runtime branch");
        let id = tee_handle
            .attach(branch)
            .expect("attach the runtime branch");
        thread::sleep(Duration::from_millis(40));
        tee_handle.detach(id).expect("detach the runtime branch");
        if round + 1 == WARMUP {
            settle();
        }
        if round >= WARMUP {
            memory.sample();
        }
    }

    assert_eq!(
        tee_handle.sink_count(),
        1,
        "every churned branch must be gone, leaving only the fixed one"
    );
    assert!(
        fixed_frames.load(Ordering::Relaxed) > 0,
        "the fixed branch must keep receiving frames throughout the churn"
    );
    pipeline.finish();
    memory.assert_flat(0.5 * MIB);
}

/// Adding and removing compositor inputs while the compositor keeps
/// emitting. Each round builds a whole input pipeline of its own, so this
/// churns the compositor's input registry, its layer handles, and a
/// short-lived source pipeline at the same time.
#[test]
#[ignore = "soak test; run with --ignored"]
fn compositor_input_churn_does_not_grow_process_memory() {
    isolate!();
    let _exclusive = common::exclusive();
    media_pp::init().expect("ffmpeg init");
    let (compositor, handle) = SwVideoCompositor::new(
        "compositor",
        VideoCompositorOptions {
            width: WIDTH,
            height: HEIGHT,
            frame_rate: frame_rate(),
            background: Color::new(16, 16, 16),
        },
    )
    .expect("create the compositor");

    let (counter, frames) = FrameCounter::new("counter");
    let output = Pipeline::new("soak-compositor", compositor, |source, ctx| {
        let branch = ctx.branch().queue("composited", 4).to(Box::new(counter))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the compositor output pipeline");
    output.run().unwrap();

    let iterations = iterations(20);
    let mut memory = Trend::private_bytes("compositor churn private bytes");
    for round in 0..(WARMUP + iterations) {
        let mut layer = VideoLayer::new(VideoRect::new(0, 0, WIDTH, HEIGHT));
        layer.fit = VideoFit::Cover;
        let input = handle
            .add_source("churned", layer)
            .expect("add a compositor input")
            .expect("the compositor is alive while its pipeline runs");

        let feeder = Pipeline::new("soak-compositor-input", test_source("input"), {
            let sink = input.sink;
            move |source, ctx| {
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            }
        })
        .expect("wire the compositor input pipeline");
        feeder.run().unwrap();
        thread::sleep(Duration::from_millis(150));

        // The input's own pipeline is torn down first, the way a real
        // caller stops feeding a layer before removing it.
        feeder.finish();
        handle.remove_source("churned");
        assert_eq!(
            handle.source_count(),
            0,
            "round {round} left a compositor input behind"
        );
        if round + 1 == WARMUP {
            settle();
        }
        if round >= WARMUP {
            memory.sample();
        }
    }

    assert!(
        frames.load(Ordering::Relaxed) > 0,
        "the compositor must keep emitting while its inputs churn"
    );
    output.finish();
    // Each round builds and destroys a whole extra pipeline, so this one
    // is allowed more room than the pure control scenarios.
    memory.assert_flat(1.0 * MIB);
}

/// A compositor left running, rather than built and torn down: what the
/// scenarios above cannot see.
///
/// Answering a repeated picture with the one already composed is retention
/// per *frame*, not per cycle. The compositor holds the picture it may offer
/// again, and holds a replaced one until nothing references it any more —
/// `buffer::picture_is_referenced`, checked on every composite. If that
/// release ever fails to happen, this is where it shows: an output frame is
/// a whole picture, so a retained one per tick at this rate is megabytes a
/// second, and a cycle-driven scenario that runs for 150 ms would not notice.
///
/// The input changes on every frame, which is deliberately the case the
/// repeat path cannot help with: every tick composes, every tick replaces
/// what was held, and the queue behind it keeps some of those frames alive
/// while it does. The still case retains strictly less.
#[test]
#[ignore = "soak test; run with --ignored"]
fn a_running_compositor_does_not_grow_while_it_answers_frames() {
    isolate!();
    let _exclusive = common::exclusive();
    media_pp::init().expect("ffmpeg init");
    let (compositor, handle) = SwVideoCompositor::new(
        "compositor",
        VideoCompositorOptions {
            width: WIDTH,
            height: HEIGHT,
            frame_rate: frame_rate(),
            background: Color::new(16, 16, 16),
        },
    )
    .expect("create the compositor");

    let mut layer = VideoLayer::new(VideoRect::new(0, 0, WIDTH, HEIGHT));
    layer.fit = VideoFit::Cover;
    let input = handle
        .add_source("moving", layer)
        .expect("add a compositor input")
        .expect("the compositor is alive");

    let (counter, frames) = FrameCounter::new("counter");
    let output = Pipeline::new("soak-running-compositor", compositor, |source, ctx| {
        // A queue, so composited frames are still referenced when the next
        // composite replaces them — with a synchronous sink nothing is ever
        // held and the release has nothing to prove.
        let branch = ctx.branch().queue("composited", 4).to(Box::new(counter))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the compositor output pipeline");
    let feeder = Pipeline::new("soak-running-input", test_source("input"), {
        let sink = input.sink;
        move |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        }
    })
    .expect("wire the compositor input pipeline");
    output.run().unwrap();
    feeder.run().unwrap();

    let duration = soak_duration(20);
    let sample_interval = Duration::from_secs(1);
    let deadline = Instant::now() + duration;
    let mut memory = Trend::private_bytes("running compositor private bytes");
    // The first interval is the pools filling; every later one is steady
    // state, which is the only part a trend can be read from.
    thread::sleep(sample_interval);
    settle();
    let started = frames.load(Ordering::Relaxed);
    while Instant::now() < deadline {
        thread::sleep(sample_interval);
        memory.sample();
    }
    let composited = frames.load(Ordering::Relaxed) - started;
    feeder.finish();
    output.finish();

    assert!(
        composited > 0,
        "the compositor stopped emitting, so a flat trend proves nothing"
    );
    eprintln!("composited {composited} frames over {duration:?}");
    // One retained 1280x720 BGRA frame is 3.5 MiB, so a per-tick leak is
    // hundreds of times this slope; the budget is the allocator's own
    // jitter, not a fraction of what would be leaked.
    memory.assert_flat(0.5 * MIB);
}

/// A long segmented recording: one pipeline, many files. Rotation reopens a
/// muxer per segment, which is the per-file allocation most likely to
/// accumulate over a recording that runs for hours.
#[test]
#[ignore = "soak test; run with --ignored"]
fn segment_rotation_does_not_grow_process_memory_or_hold_files() {
    isolate!();
    let _exclusive = common::exclusive();
    media_pp::init().expect("ffmpeg init");
    let dir = TempDir::new("segments");
    let source = test_source("video");
    let time_base = source.time_base();
    // A keyframe every half second, so a one-second policy actually cuts
    // rather than waiting on a sparse GOP.
    let encoder = encoder("encoder", time_base, 15);

    let segment_dir = dir.path().to_path_buf();
    let mut muxer = SegmentedMp4Muxer::create(
        SegmentPolicy::Duration(Duration::from_secs(1)),
        move |index| segment_dir.join(format!("segment_{index:04}.mp4")),
    );
    muxer.add_stream("video", encoder.parameters(), time_base);
    let sink = muxer
        .open()
        .expect("open the first segment")
        .pop()
        .expect("one track");

    let pipeline = Pipeline::new("soak-segments", source, |source, ctx| {
        let branch = ctx
            .branch()
            .queue("encode-frames", 8)
            .pipe(encoder)
            .to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the segmented pipeline");
    pipeline.run().unwrap();

    let duration = soak_duration(20);
    let sample_interval = Duration::from_secs(1);
    let deadline = Instant::now() + duration;
    let mut memory = Trend::private_bytes("segment rotation private bytes");
    // The first interval covers the warm-up of the first segment; every
    // later one is a steady-state sample.
    thread::sleep(sample_interval);
    settle();
    while Instant::now() < deadline {
        thread::sleep(sample_interval);
        memory.sample();
    }
    pipeline.finish();

    let segments: Vec<_> = std::fs::read_dir(dir.path())
        .expect("read the segment directory")
        .map(|entry| entry.expect("read a segment entry").path())
        .collect();
    assert!(
        segments.len() > 1,
        "the policy never actually rotated: {} file(s) in {duration:?}",
        segments.len()
    );
    for segment in &segments {
        assert!(
            std::fs::metadata(segment).expect("stat a segment").len() > 0,
            "{} was left empty",
            segment.display()
        );
        // Removing each one proves no rotation left a file handle open;
        // `TempDir`'s own teardown then proves the same for the last.
        std::fs::remove_file(segment).expect("a finalized segment must be closed");
    }

    memory.assert_flat(0.5 * MIB);
}

/// The D3D11 zero-copy path. Textures are both the resource most likely to
/// leak here and the one no CPU-side gauge can see at all, so this scenario
/// watches adapter video memory — and, when the SDK debug layer is
/// installed, the device's own live-object count — alongside private bytes.
#[cfg(all(windows, feature = "d3d11"))]
mod d3d11 {
    use std::{
        sync::{Arc, Mutex, atomic::Ordering},
        thread,
        time::{Duration, Instant},
    };

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        color::Color,
        elements::{
            ChromaKeyMethod, ChromaKeyOptions, D3d11ChromaKey, D3d11Decoder, D3d11NvencCodec,
            D3d11NvencEncoder, D3d11NvencEncoderOptions, D3d11NvencInputFormat, D3d11Scaler,
            D3d11ScalerFormat, D3d11Upload, D3d11VideoCompositor, FrameCounter, PacketCounter,
            SwScaler, VideoCompositorOptions, VideoLayer, VideoRect,
        },
        pipeline::Pipeline,
    };
    use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11DeviceContext};

    use crate::common::{
        MIB, Trend, Unit, exclusive,
        gpu::{D3d11LiveObjects, try_d3d11_device, vram_bytes},
        iterations, settle, soak_duration, try_test_video,
    };
    use crate::{HEIGHT, Teardown, WARMUP, WIDTH, frame_rate, test_source};

    /// How long the running-compositor scenario feeds the graph before its
    /// measurement window opens — see that scenario's own comment on the
    /// driver allocation cache this waits out.
    const WARMUP_SECS: Duration = Duration::from_secs(20);

    /// Deliberately not a multiple of the input size, so the scaler really
    /// scales, and even in both axes, which NV12 requires.
    const SCALED_WIDTH: u32 = 176;
    const SCALED_HEIGHT: u32 = 144;

    /// Upload every frame to a texture, scale it on the video processor,
    /// and hold the results in a queue — the shape of a real GPU pipeline,
    /// minus the renderer a headless test cannot present to.
    fn cycle(
        device: &ID3D11Device,
        context: &Arc<Mutex<ID3D11DeviceContext>>,
        teardown: Teardown,
    ) -> usize {
        let (counter, frames) = FrameCounter::new("counter");
        let device = device.clone();
        let context = context.clone();
        let pipeline = Pipeline::new("soak-d3d11", test_source("video"), move |source, ctx| {
            let to_nv12 = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                WIDTH,
                HEIGHT,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = D3d11Upload::new("upload", &device, WIDTH, HEIGHT);
            let scaler = D3d11Scaler::new(
                "scaler",
                &device,
                context.clone(),
                D3d11ScalerFormat::Preserve,
                SCALED_WIDTH,
                SCALED_HEIGHT,
            )?;
            let branch = ctx
                .branch()
                .pipe(to_nv12)
                .pipe(upload)
                .pipe(scaler)
                .queue("gpu-frames", 4)
                .to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the D3D11 pipeline");

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// Key a compositor's own BGRA output on the GPU. What this adds over
    /// the upload/scale scenario is the per-frame render-target and
    /// shader-resource views `D3d11ChromaKey` creates around each draw:
    /// those are COM objects the shared context also holds a reference to
    /// until the element clears its bindings, so a missed release shows up
    /// on the debug layer's live-object count long before it is visible in
    /// adapter memory.
    ///
    /// The compositor is here because it is the only headless producer of
    /// GPU-resident BGRA in this crate — `D3d11Upload` is NV12-only, and
    /// desktop capture needs a desktop. It is also the real topology: this
    /// element exists to key a layer without leaving video memory.
    fn chroma_key_cycle(
        device: &ID3D11Device,
        context: &Arc<Mutex<ID3D11DeviceContext>>,
        teardown: Teardown,
    ) -> usize {
        let (compositor, compositor_handle) = D3d11VideoCompositor::new(
            "compositor",
            device,
            context.clone(),
            VideoCompositorOptions {
                width: WIDTH,
                height: HEIGHT,
                frame_rate: frame_rate(),
                background: Color::new(0, 255, 0),
            },
        )
        .expect("build the compositor");
        let layer_sink = compositor_handle
            .add_source(
                "layer",
                VideoLayer::new(VideoRect::new(0, 0, WIDTH, HEIGHT)),
            )
            .expect("register the compositor input")
            .expect("the compositor is alive")
            .sink;

        let input_device = device.clone();
        let input_pipeline = Pipeline::new("soak-d3d11-key-input", test_source("video"), {
            move |source, ctx| {
                let to_nv12 = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    WIDTH,
                    HEIGHT,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload = D3d11Upload::new("upload", &input_device, WIDTH, HEIGHT);
                let branch = ctx.branch().pipe(to_nv12).pipe(upload).to(layer_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            }
        })
        .expect("wire the compositor input pipeline");

        let (counter, frames) = FrameCounter::new("counter");
        let key_device = device.clone();
        let key_context = context.clone();
        let output_pipeline = Pipeline::new("soak-d3d11-key", compositor, move |source, ctx| {
            let key = D3d11ChromaKey::new(
                "key",
                &key_device,
                key_context,
                ChromaKeyOptions {
                    method: ChromaKeyMethod::Green,
                    threshold: 0.15,
                    smoothing: 0.1,
                },
            )?;
            let branch = ctx
                .branch()
                .pipe(key)
                .queue("keyed-frames", 4)
                .to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the chroma-key pipeline");

        output_pipeline.run().unwrap();
        input_pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        // The input goes first: the compositor keeps drawing from whatever
        // it last received, so tearing it down first would leave the input
        // pushing into a sink that is already gone.
        teardown.apply(&input_pipeline);
        teardown.apply(&output_pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// How deep the decode scenario's queue is, and therefore how many
    /// extra decode surfaces its decoder has to be given: D3D11VA's pool is
    /// a fixed-size texture array sized once at init, so every frame still
    /// sitting in that queue holds a slot (see `D3d11Decoder::new`).
    const DECODE_QUEUE_DEPTH: usize = 4;

    /// Decode a real file straight onto the GPU. What this adds over the
    /// upload scenario is the decoder's own fixed-size surface pool: a
    /// decoder that outlived its cycle would keep that whole texture array
    /// while the next cycle allocates another one.
    fn decode_cycle(device: &ID3D11Device, path: &str, teardown: Teardown) -> usize {
        let (source, index, parameters) = crate::open_fixture(path);
        let (counter, frames) = FrameCounter::new("counter");
        let device = device.clone();
        let pipeline = Pipeline::new("soak-d3d11-decode", source, move |source, ctx| {
            let decoder =
                D3d11Decoder::new("decoder", parameters, &device, DECODE_QUEUE_DEPTH as i32)?;
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("decoded-frames", DECODE_QUEUE_DEPTH)
                .to(Box::new(counter))?;
            ctx.attach(source, index, branch)?;
            Ok(())
        })
        .expect(
            "wire the D3D11 decode pipeline — a failure here after the first cycle means \
             an earlier decoder never released its fixed-size surface pool",
        );

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// What every NVENC cycle below encodes with. Kept in one place because
    /// the support probe has to open an encoder with exactly the options a
    /// cycle will use — a codec or input format the driver refuses is a
    /// skip, not a leak.
    fn nvenc_options(time_base: ffmpeg::Rational) -> D3d11NvencEncoderOptions {
        D3d11NvencEncoderOptions {
            codec: D3d11NvencCodec::H264,
            input_format: D3d11NvencInputFormat::Nv12,
            width: WIDTH,
            height: HEIGHT,
            time_base,
            frame_rate: frame_rate(),
            bit_rate: 1_000_000,
            gop_size: 30,
        }
    }

    /// Encode on the GPU's NVENC block. Each cycle opens and closes a
    /// hardware encoding session, and consumer drivers cap how many can be
    /// open at once — so a session that outlived its cycle stops a later
    /// cycle from opening at all, rather than merely costing memory.
    fn encode_cycle(
        device: &ID3D11Device,
        context: &Arc<Mutex<ID3D11DeviceContext>>,
        teardown: Teardown,
    ) -> usize {
        let (counter, packets) = PacketCounter::new("counter");
        let source = test_source("video");
        let time_base = source.time_base();
        let device = device.clone();
        let context = context.clone();
        let pipeline = Pipeline::new("soak-d3d11-nvenc", source, move |source, ctx| {
            let to_nv12 = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                WIDTH,
                HEIGHT,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = D3d11Upload::new("upload", &device, WIDTH, HEIGHT);
            let encoder = D3d11NvencEncoder::new(
                "encoder",
                &device,
                context.clone(),
                nvenc_options(time_base),
            )?;
            let branch = ctx
                .branch()
                .pipe(to_nv12)
                .pipe(upload)
                .queue("gpu-frames", 4)
                .pipe(encoder)
                .to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect(
            "wire the D3D11 NVENC pipeline — a failure here after the first cycle means an \
             earlier encoder never closed its NVENC session",
        );

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        teardown.apply(&pipeline);
        packets.load(Ordering::Relaxed)
    }

    /// Whether this GPU and FFmpeg build have NVENC at all.
    fn nvenc_supported(device: &ID3D11Device, context: &Arc<Mutex<ID3D11DeviceContext>>) -> bool {
        let time_base = test_source("probe").time_base();
        match D3d11NvencEncoder::new("probe", device, context.clone(), nvenc_options(time_base)) {
            Ok(_) => true,
            Err(error) => {
                eprintln!("skipping: no D3D11 NVENC encoder on this machine ({error})");
                false
            }
        }
    }

    /// Whether this adapter has a D3D11VA decoder for the fixture's codec.
    /// One that does not is not a failure of anything this scenario
    /// measures, so it skips with a reason the same way a missing device
    /// does.
    fn decode_supported(device: &ID3D11Device, path: &str) -> bool {
        let (_source, _index, parameters) = crate::open_fixture(path);
        match D3d11Decoder::new("probe", parameters, device, 0) {
            Ok(_) => true,
            Err(error) => {
                eprintln!("skipping: no D3D11VA decoder for this fixture ({error})");
                false
            }
        }
    }

    /// How long a scenario warms up for, how long it measures, and how much
    /// growth each gauge is allowed per cycle.
    ///
    /// `memory` is optional because on this side of the crate it often
    /// cannot resolve anything worth asserting: a GPU cycle's private bytes
    /// move by megabytes in both directions as the graphics driver grows and
    /// trims its own bookkeeping, so the residual spread swamps any
    /// threshold small enough to catch a leak. Where that is the case the
    /// scenario records the trend and says so instead of pretending to
    /// judge it — adapter memory and the live-object count are the gauges
    /// with real sensitivity here, and they carry the assertion.
    struct Budget {
        warmup: usize,
        iterations: usize,
        memory: Option<f64>,
        vram: f64,
    }

    /// Runs `cycle` and judges every GPU gauge. The scenarios below differ
    /// only in the graph each cycle builds and in their `Budget`, not in how
    /// one is measured.
    fn measure_cycles(
        label: &str,
        device: &ID3D11Device,
        live: Option<Arc<D3d11LiveObjects>>,
        budget: Budget,
        mut cycle: impl FnMut(Teardown) -> usize,
    ) {
        let Budget {
            warmup,
            iterations,
            memory: max_memory_slope,
            vram: max_vram_slope,
        } = budget;
        let mut memory = Trend::private_bytes(format!("{label} private bytes"));
        let mut vram = Trend::new(format!("{label} adapter memory"), Unit::Bytes, {
            let device = device.clone();
            move || vram_bytes(&device)
        });
        let mut objects = live.clone().map(|live| {
            Trend::new(format!("{label} live objects"), Unit::Objects, move || {
                live.count()
            })
        });

        for index in 0..(warmup + iterations) {
            let teardown = Teardown::for_cycle(index);
            let frames = cycle(teardown);
            // Only a `finish` cycle has to have produced something. `stop`
            // means abandon, so a stateful stage — NVENC holds several
            // frames before its first packet — legitimately emits nothing
            // when a cycle ends that way.
            if teardown == Teardown::Finish {
                assert!(
                    frames > 0,
                    "{label} {index} pushed nothing through the GPU path before draining"
                );
            }
            if index + 1 == warmup {
                // The warm-up cycles are also what makes the previous
                // scenario's release land before this window opens.
                settle();
            }
            if index >= warmup {
                memory.sample();
                vram.sample();
                if let Some(objects) = objects.as_mut() {
                    objects.sample();
                }
            }
        }

        match max_memory_slope {
            Some(max) => memory.assert_flat(max),
            None => {
                memory.print();
                eprintln!(
                    "  recorded, not asserted: the graphics driver's own allocation cache \
                     dominates this gauge here and saturates at a different cycle every run \
                     (see Budget); adapter memory carries this scenario's assertion"
                );
            }
        }
        vram.assert_flat(max_vram_slope);

        let Some(objects) = objects else {
            return;
        };
        if objects.slope() > 0.0 {
            let live = live.expect("the trend exists only with the debug layer");
            eprintln!("live D3D11 objects after the last cycle:");
            for line in live.describe() {
                eprintln!("  {line}");
            }
        }
        // The device and its context are live throughout, so the count is
        // never zero — but an identical cycle must not add to it.
        objects.assert_flat(0.0);
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn upload_and_scale_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some((device, context, live)) = try_d3d11_device() else {
            return;
        };
        // Every cycle allocates its own upload and scaler textures and
        // releases them at teardown; the driver frees lazily enough for a
        // cycle's worth of jitter, but not for a cycle's worth of growth.
        measure_cycles(
            "d3d11 cycle",
            &device,
            live.map(Arc::new),
            Budget {
                warmup: WARMUP,
                iterations: iterations(15),
                memory: Some(0.5 * MIB),
                vram: 1.0 * MIB,
            },
            |teardown| cycle(&device, &context, teardown),
        );
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn chroma_key_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some((device, context, live)) = try_d3d11_device() else {
            return;
        };
        // The live-object count is the gauge that matters here, and the
        // reason this scenario exists. Every frame builds an output
        // texture, its render-target view, and the input.s shader-resource
        // view in both the compositor and the key, and a D3D11 object whose
        // last reference is dropped is only queued for destruction — it
        // survives until the context is flushed. Before `D3d11ChromaKey`
        // flushed, this cycle grew the device.s object count by 65 per
        // cycle and adapter memory by 5 MiB per cycle, both dead straight
        // over 40 cycles; all three gauges are flat with the flush in
        // place, which is where these thresholds come from.
        measure_cycles(
            "d3d11 chroma key cycle",
            &device,
            live.map(Arc::new),
            Budget {
                warmup: WARMUP,
                iterations: iterations(15),
                memory: Some(0.5 * MIB),
                vram: 1.0 * MIB,
            },
            |teardown| chroma_key_cycle(&device, &context, teardown),
        );
    }

    /// The GPU half of `a_running_compositor_does_not_grow_while_it_answers_frames`.
    ///
    /// What is retained here is a texture rather than a heap allocation, so
    /// the gauges that matter are adapter memory and the device's live-object
    /// count. A frame the compositor holds past its release — the picture it
    /// may offer again, or one replaced while a queue still refers to it — is
    /// a whole render target, and at this rate a per-tick retention is tens of
    /// MiB a second. `D3d11Upload` answers repeats the same way, so its
    /// retention is inside this window too.
    ///
    /// The input changes on every frame, deliberately: that is the case where
    /// every tick composes, replaces what was held, and asks the release to
    /// happen again.
    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn a_running_d3d11_compositor_does_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some((device, context, live)) = try_d3d11_device() else {
            return;
        };

        let (compositor, handle) = D3d11VideoCompositor::new(
            "compositor",
            &device,
            context.clone(),
            VideoCompositorOptions {
                width: WIDTH,
                height: HEIGHT,
                frame_rate: frame_rate(),
                background: Color::new(16, 16, 16),
            },
        )
        .expect("build the compositor");
        let layer_sink = handle
            .add_source(
                "moving",
                VideoLayer::new(VideoRect::new(0, 0, WIDTH, HEIGHT)),
            )
            .expect("register the compositor input")
            .expect("the compositor is alive")
            .sink;

        let input_device = device.clone();
        let feeder = Pipeline::new("soak-running-d3d11-input", test_source("video"), {
            move |source, ctx| {
                let to_nv12 = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    WIDTH,
                    HEIGHT,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload = D3d11Upload::new("upload", &input_device, WIDTH, HEIGHT);
                let branch = ctx.branch().pipe(to_nv12).pipe(upload).to(layer_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            }
        })
        .expect("wire the compositor input pipeline");

        let (counter, frames) = FrameCounter::new("counter");
        let output = Pipeline::new("soak-running-d3d11", compositor, |source, ctx| {
            // As in the software scenario: without a queue nothing is ever
            // still referenced when the next composite replaces it.
            let branch = ctx.branch().queue("composited", 4).to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the compositor output pipeline");
        output.run().unwrap();
        feeder.run().unwrap();

        let duration = soak_duration(20);
        let sample_interval = Duration::from_secs(1);
        let mut memory = Trend::private_bytes("running d3d11 compositor private bytes");
        let mut vram = Trend::new("running d3d11 compositor adapter memory", Unit::Bytes, {
            let device = device.clone();
            move || vram_bytes(&device)
        });
        let mut objects = live.map(Arc::new).map(|live| {
            Trend::new(
                "running d3d11 compositor live objects",
                Unit::Objects,
                move || live.count(),
            )
        });
        // Far more warm-up than the software scenario, and measured rather
        // than guessed: `D3d11Upload` allocates a texture per frame, and the
        // user-mode driver keeps freed ones in its own allocation cache
        // instead of returning the pages. Private bytes therefore climb
        // about 4 MiB a second from a cold device and stop dead once that
        // cache is saturated — a run that opened its window at one second
        // measured +2.1 MiB/iter over the rise and flat over the rest of the
        // same run. This waits for the plateau, which is what the scenario
        // is about; `CAPTURE_WARMUP` documents the same driver behavior for
        // the capture scenarios.
        thread::sleep(WARMUP_SECS);
        settle();
        let started = frames.load(Ordering::Relaxed);
        let deadline = Instant::now() + duration;
        while Instant::now() < deadline {
            thread::sleep(sample_interval);
            memory.sample();
            vram.sample();
            if let Some(objects) = objects.as_mut() {
                objects.sample();
            }
        }
        let composited = frames.load(Ordering::Relaxed) - started;
        feeder.finish();
        output.finish();

        assert!(
            composited > 0,
            "the compositor stopped emitting, so a flat trend proves nothing"
        );
        eprintln!("composited {composited} frames over {duration:?}");
        memory.assert_flat(0.5 * MIB);
        // One retained 1280x720 BGRA render target is 3.5 MiB.
        vram.assert_flat(1.0 * MIB);
        if let Some(objects) = objects {
            objects.print();
            eprintln!(
                "  recorded, not asserted: a running graph creates D3D11 objects per frame — an \
                 upload texture and the shader-resource views a draw binds — and dropping the \
                 last reference only queues one for destruction, so this gauge climbs and then \
                 falls back wholesale when the driver flushes. It measures that queue rather \
                 than what this graph owns; the cycle scenarios above are where it is an \
                 ownership gauge, and adapter memory carries the assertion here"
            );
        }
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn decode_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(path) = try_test_video() else { return };
        let Some((device, _context, live)) = try_d3d11_device() else {
            return;
        };
        if !decode_supported(&device, &path) {
            return;
        }
        // A decode surface pool is much larger than the upload scenario's
        // textures — one leaked pool is tens of MiB, so this threshold is
        // still far below a single missed release.
        measure_cycles(
            "d3d11 decode cycle",
            &device,
            live.map(Arc::new),
            Budget {
                warmup: WARMUP,
                iterations: iterations(10),
                memory: Some(0.5 * MIB),
                vram: 2.0 * MIB,
            },
            |teardown| decode_cycle(&device, &path, teardown),
        );
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn nvenc_encode_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some((device, context, live)) = try_d3d11_device() else {
            return;
        };
        if !nvenc_supported(&device, &context) {
            return;
        }
        // NVENC keeps its own reference-frame surfaces per session, so a
        // leaked session costs more than the upload scenario's textures.
        measure_cycles(
            "d3d11 nvenc cycle",
            &device,
            live.map(Arc::new),
            Budget {
                warmup: WARMUP,
                iterations: iterations(10),
                memory: Some(0.5 * MIB),
                vram: 2.0 * MIB,
            },
            |teardown| encode_cycle(&device, &context, teardown),
        );
    }

    /// Far more warm-up than any other scenario, because of a driver
    /// behavior this scenario had to be measured against rather than
    /// assumed away.
    ///
    /// Opening a capture session allocates screen-sized textures on a
    /// device the session owns and destroys with itself. D3D11 releases
    /// resources lazily, and the user-mode driver keeps the committed pages
    /// in an allocation cache instead of returning them, so the first
    /// sessions a process opens each add roughly one screen's worth of
    /// private bytes — ~8 MiB at 1080p. It is a *cache*, not a leak: on
    /// this machine it saturates after six or so sessions and then stays
    /// put (measured flat across the following 24). Two things pinned that
    /// down: the same growth reproduces with plain `CreateTexture2D` calls
    /// and no media-pp code at all, and `CaptureMode::Cpu`, which allocates
    /// no per-frame textures, stays flat from the first cycle.
    ///
    /// So the measurement window has to start after saturation. What the
    /// scenario still catches is the thing that matters: growth that keeps
    /// going, which no cache explains.
    #[cfg(any(feature = "dxgi-capture", feature = "wgc-capture"))]
    const CAPTURE_WARMUP: usize = 10;

    /// One capture session: open it, let it emit for a while, tear it all
    /// down. Unlike every other scenario here the source brings its own
    /// `ID3D11Device` (see `DxgiCaptureSource::open`, which returns one), so
    /// each cycle creates and destroys a whole device, duplication, and
    /// texture set.
    #[cfg(feature = "dxgi-capture")]
    fn capture_cycle(mode: media_pp::elements::CaptureMode, teardown: Teardown) -> usize {
        use media_pp::elements::{CaptureArea, DxgiCaptureOptions, DxgiCaptureSource};

        let (counter, frames) = FrameCounter::new("counter");
        let (source, _format, _device) = DxgiCaptureSource::open(
            "capture",
            DxgiCaptureOptions {
                area: CaptureArea::Output { output_index: 0 },
                capture_mode: mode,
                ..Default::default()
            },
        )
        .expect(
            "open desktop duplication — a failure here after the first cycle means an earlier \
             capture source never released its output duplication",
        );

        let pipeline = Pipeline::new("soak-dxgi-capture", source, |source, ctx| {
            let branch = ctx.branch().queue("captured", 4).to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the capture pipeline");

        pipeline.run().unwrap();
        // Longer than the other cycles: this source emits on its own fixed
        // schedule, so a cycle has to span several of its ticks to prove
        // frames really flowed.
        thread::sleep(Duration::from_millis(400));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// Whether desktop duplication is available at all — it is not, for
    /// instance, in a session without an attached output.
    #[cfg(feature = "dxgi-capture")]
    fn capture_supported(mode: media_pp::elements::CaptureMode) -> bool {
        use media_pp::elements::{CaptureArea, DxgiCaptureOptions, DxgiCaptureSource};

        match DxgiCaptureSource::open(
            "probe",
            DxgiCaptureOptions {
                area: CaptureArea::Output { output_index: 0 },
                capture_mode: mode,
                ..Default::default()
            },
        ) {
            Ok(_) => true,
            Err(error) => {
                eprintln!("skipping: no desktop duplication available here ({error})");
                false
            }
        }
    }

    /// The default capture mode: staging texture, `Map`, row copies into
    /// pooled CPU frames. No per-frame GPU allocation at all, which is what
    /// lets this scenario hold private bytes to the same tight threshold as
    /// the CPU-only scenarios — the sensitive half of the capture coverage.
    ///
    /// It still opens one staging texture per session, so it sees a small
    /// version of the same driver cache `CAPTURE_WARMUP` describes: a run of
    /// 250 cycles climbs about 4 MiB and then stops, its slope falling from
    /// +0.021 MiB/cycle over the first half to +0.004 over the second. At
    /// the default cycle count that step is under the measurement's own
    /// noise; at a hundred cycles or more it is visible and still bounded,
    /// which is what it should look like — not a reason to go hunting.
    #[test]
    #[ignore = "soak test; run with --ignored"]
    #[cfg(feature = "dxgi-capture")]
    fn cpu_desktop_capture_cycles_do_not_grow_process_memory() {
        use media_pp::elements::CaptureMode;

        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        // `CaptureMode` is not `Copy`, so each use builds its own value.
        let cpu_mode = || CaptureMode::Cpu {
            include_cursor: false,
        };
        if !capture_supported(cpu_mode()) {
            return;
        }
        let Some((device, _context, _live)) = try_d3d11_device() else {
            return;
        };
        measure_cycles(
            "cpu capture cycle",
            &device,
            None,
            Budget {
                warmup: WARMUP,
                iterations: iterations(10),
                memory: Some(0.5 * MIB),
                vram: 1.0 * MIB,
            },
            |teardown| capture_cycle(cpu_mode(), teardown),
        );
    }

    /// The zero-copy mode: every emitted frame is its own GPU texture. Its
    /// private-bytes budget is much looser than any other scenario's, for
    /// the driver-cache reason `CAPTURE_WARMUP` documents — 3 MiB still
    /// catches a screen-sized allocation retained per cycle, which is what a
    /// leak here would look like, and adapter memory stays strict.
    #[test]
    #[ignore = "soak test; run with --ignored"]
    #[cfg(feature = "dxgi-capture")]
    fn gpu_desktop_capture_cycles_do_not_grow_gpu_memory() {
        use media_pp::elements::CaptureMode;

        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        if !capture_supported(CaptureMode::Gpu) {
            return;
        }
        let Some((device, _context, _live)) = try_d3d11_device() else {
            return;
        };
        // No live-object trend: this scenario's D3D11 objects belong to the
        // capture source's own device, which the gauge device cannot see.
        // Adapter memory is per process, so it still covers them.
        measure_cycles(
            "gpu capture cycle",
            &device,
            None,
            Budget {
                warmup: CAPTURE_WARMUP,
                iterations: iterations(10),
                memory: None,
                vram: 1.0 * MIB,
            },
            |teardown| capture_cycle(CaptureMode::Gpu, teardown),
        );
    }

    #[cfg(feature = "wgc-capture")]
    struct WgcTestWindow {
        hwnd: usize,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    #[cfg(feature = "wgc-capture")]
    impl WgcTestWindow {
        fn create() -> windows::core::Result<Self> {
            use windows::{
                Win32::UI::WindowsAndMessaging::{
                    CreateWindowExW, DestroyWindow, DispatchMessageW, GetMessageW, IsWindow, MSG,
                    TranslateMessage, WINDOW_EX_STYLE, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
                },
                core::w,
            };

            let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
            let thread = thread::spawn(move || {
                // SAFETY: `STATIC` is a preregistered window class and every
                // other argument is a plain value or an absent
                // parent/menu/instance, so nothing here has to outlive the
                // call. This thread owns the window for its whole lifetime,
                // which is what `GetMessageW`/`DestroyWindow` below require.
                let created = unsafe {
                    CreateWindowExW(
                        WINDOW_EX_STYLE::default(),
                        w!("STATIC"),
                        w!("media-pp WGC soak window"),
                        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                        0,
                        0,
                        320,
                        240,
                        None,
                        None,
                        None,
                        None,
                    )
                };
                let hwnd = match created {
                    Ok(hwnd) => hwnd,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                if ready_tx.send(Ok(hwnd.0 as usize)).is_err() {
                    // SAFETY: this thread created `hwnd` and no one else has
                    // seen it, so destroying it here happens exactly once on
                    // the owning thread.
                    let _ = unsafe { DestroyWindow(hwnd) };
                    return;
                }

                let mut message = MSG::default();
                // SAFETY: `message` is a live out-parameter and a `None` window
                // filter asks for this thread's own queue, which is the queue
                // the window above belongs to.
                while unsafe { GetMessageW(&mut message, None, 0, 0) }.0 > 0 {
                    // SAFETY: both calls only read the message `GetMessageW`
                    // just filled in.
                    unsafe {
                        let _ = TranslateMessage(&message);
                        DispatchMessageW(&message);
                    }
                    // SAFETY: reads only whether `hwnd` still identifies a
                    // window; it retains no caller storage.
                    if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
                        break;
                    }
                }
                // SAFETY: same as the check above — a stale handle simply
                // reports false.
                if unsafe { IsWindow(Some(hwnd)) }.as_bool() {
                    // SAFETY: the loop exited with the window still alive, so
                    // this destroys it once, on the thread that created it.
                    let _ = unsafe { DestroyWindow(hwnd) };
                }
            });
            let hwnd = ready_rx
                .recv()
                .expect("WGC window thread stopped before reporting readiness")?;
            Ok(Self {
                hwnd,
                thread: Some(thread),
            })
        }

        fn hwnd(&self) -> windows::Win32::Foundation::HWND {
            windows::Win32::Foundation::HWND(self.hwnd as *mut _)
        }
    }

    #[cfg(feature = "wgc-capture")]
    impl Drop for WgcTestWindow {
        fn drop(&mut self) {
            use windows::Win32::{
                Foundation::{LPARAM, WPARAM},
                UI::WindowsAndMessaging::{PostMessageW, WM_CLOSE},
            };

            // SAFETY: posting is thread-agnostic by design — it queues
            // `WM_CLOSE` for the owning thread's loop to dispatch rather than
            // touching the window here. A stale handle only fails the call.
            let _ = unsafe {
                PostMessageW(
                    Some(self.hwnd()),
                    WM_CLOSE,
                    WPARAM::default(),
                    LPARAM::default(),
                )
            };
            if let Some(thread) = self.thread.take() {
                thread.join().expect("WGC window thread panicked");
            }
        }
    }

    /// Opens and tears down the complete WGC graph. WGC owns a fresh device,
    /// frame pool, session, WinRT callbacks, destroy-hook registration, owner
    /// process handle, and changing textures in every cycle — and with one
    /// source alive at a time the shared watcher thread and its hook are
    /// created and torn down every cycle too — so the process-wide adapter
    /// gauge is used just like the DXGI source above.
    #[cfg(feature = "wgc-capture")]
    fn wgc_capture_cycle(hwnd: windows::Win32::Foundation::HWND, teardown: Teardown) -> usize {
        use media_pp::elements::{WgcCaptureOptions, WgcCaptureSource};
        use windows::Win32::UI::WindowsAndMessaging::{
            SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, SetWindowPos,
        };

        static CYCLE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

        let (counter, frames) = FrameCounter::new("counter");
        let (source, _device) = WgcCaptureSource::open(
            "window-capture",
            hwnd,
            WgcCaptureOptions {
                fps: 30,
                include_cursor: false,
            },
        )
        .expect("open WGC source");
        let pipeline = Pipeline::new("soak-wgc-capture", source, |source, ctx| {
            let branch = ctx.branch().queue("captured", 4).to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire WGC pipeline");

        pipeline.run().expect("start WGC pipeline");
        // WGC is change-driven. Repeatedly capturing an otherwise motionless
        // test window is allowed to produce no new notification, so resize by
        // one pixel after every session starts. This also exercises the frame-
        // pool recreation path instead of relying on a cosmetic repaint.
        let cycle = CYCLE.fetch_add(1, Ordering::Relaxed);
        let width = if cycle.is_multiple_of(2) { 320 } else { 321 };
        // SAFETY: `hwnd` is the live window owned by `WgcTestWindow`, which
        // outlives this call, and `SWP_NOMOVE`/`SWP_NOZORDER` make the ignored
        // position and z-order arguments inert.
        unsafe {
            SetWindowPos(
                hwnd,
                None,
                0,
                0,
                width,
                240,
                SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
            )
        }
        .expect("resize WGC target window");
        thread::sleep(Duration::from_millis(750));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    #[test]
    #[ignore = "soak test; requires an interactive Windows Graphics Capture session"]
    #[cfg(feature = "wgc-capture")]
    fn window_capture_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let window = match WgcTestWindow::create() {
            Ok(window) => window,
            Err(error) => {
                eprintln!("skipping: cannot create WGC target window ({error})");
                return;
            }
        };
        let Some((gauge_device, _context, _live)) = try_d3d11_device() else {
            return;
        };

        // WGC and the user-mode driver retain bounded initialization caches,
        // so discard the same longer warm-up used for GPU DXGI sessions. The
        // source devices differ per cycle; adapter memory is process-wide,
        // while a live-object gauge on `gauge_device` could not see them.
        measure_cycles(
            "WGC window capture cycle",
            &gauge_device,
            None,
            Budget {
                warmup: CAPTURE_WARMUP,
                iterations: iterations(10),
                memory: None,
                vram: 1.0 * MIB,
            },
            |teardown| wgc_capture_cycle(window.hwnd(), teardown),
        );
    }
}

/// The D3D12VA transfer and video-process path. Each cycle creates and
/// destroys the upload/scaler FFmpeg contexts, both fixed GPU surface pools,
/// the video processor's queue/list/fences, and the CPU frame pool
/// `D3d12Download` fills. The device itself stays alive so the trend measures
/// element ownership rather than adapter initialization.
#[cfg(all(windows, feature = "d3d12"))]
mod d3d12 {
    use std::{sync::atomic::Ordering, thread, time::Duration};

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        elements::{D3d12Download, D3d12Scaler, D3d12Upload, FrameCounter, SwScaler},
        pipeline::Pipeline,
    };
    use windows::Win32::Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0,
        Direct3D12::{D3D12CreateDevice, ID3D12Device},
    };

    use crate::common::{Trend, Unit, exclusive, gpu::d3d12_vram_bytes, iterations, settle};
    use crate::{HEIGHT, Teardown, WARMUP, WIDTH, test_source};

    const SCALED_WIDTH: u32 = 176;
    const SCALED_HEIGHT: u32 = 144;

    fn try_device() -> Option<ID3D12Device> {
        let mut device = None;
        // SAFETY: null adapter requests the default hardware adapter and
        // `device` is the correctly typed live out-parameter for this feature
        // level.
        let result = unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device) };
        if let Err(error) = result {
            eprintln!("skipping: D3D12CreateDevice failed on this machine: {error}");
            return None;
        }
        device
    }

    fn cycle(device: &ID3D12Device, teardown: Teardown) -> usize {
        let (counter, frames) = FrameCounter::new("counter");
        let device = device.clone();
        let pipeline = Pipeline::new("soak-d3d12", test_source("video"), move |source, ctx| {
            let to_nv12 = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                WIDTH,
                HEIGHT,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = D3d12Upload::new("upload", &device, WIDTH, HEIGHT)?;
            let scaler = D3d12Scaler::new("scaler", &device, SCALED_WIDTH, SCALED_HEIGHT)?;
            let download = D3d12Download::new("download", SCALED_WIDTH, SCALED_HEIGHT);
            let branch = ctx
                .branch()
                .pipe(to_nv12)
                .pipe(upload)
                .pipe(scaler)
                .queue("gpu-frames", 4)
                .pipe(download)
                .to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect(
            "wire the D3D12 upload/scale/download pipeline — a failure after the first cycle may mean \
             an earlier FFmpeg D3D12VA frames context retained its surface pool",
        );

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn upload_scale_and_download_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(device) = try_device() else { return };

        let mut memory = Trend::private_bytes("d3d12 scale cycle private bytes");
        let mut vram = Trend::new("d3d12 scale cycle adapter memory", Unit::Bytes, {
            let device = device.clone();
            move || d3d12_vram_bytes(&device)
        });
        for index in 0..(WARMUP + iterations(15)) {
            let teardown = Teardown::for_cycle(index);
            let frames = cycle(&device, teardown);
            if teardown == Teardown::Finish {
                assert!(frames > 0, "D3D12 finish cycle {index} produced no frames");
            }
            if index + 1 == WARMUP {
                settle();
            }
            if index >= WARMUP {
                memory.sample();
                vram.sample();
            }
        }

        // Measured with upload + video-process scale + download over the
        // default 15-cycle window: private bytes grew +0.032 MiB/cycle
        // (0.018 MiB/cycle standard error), while adapter memory was flat.
        // The thresholds remain below one retained output pool at this
        // resolution and comfortably above the observed fitted slopes.
        memory.assert_flat(0.25 * crate::common::MIB);
        vram.assert_flat(0.5 * crate::common::MIB);
    }
}

/// The CUDA-resident path, same shape as the D3D11 one. CUDA surfaces are
/// invisible to both the process heap and DXGI, so this scenario asks the
/// driver directly when it can.
#[cfg(feature = "cuda")]
mod cuda {
    use std::{sync::atomic::Ordering, thread, time::Duration};

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        color::Color,
        elements::{
            CudaCodec, CudaDecoder, CudaDevice, CudaDownload, CudaEncoder, CudaEncoderOptions,
            CudaFrameFormat, CudaScaler, CudaScalerInterp, CudaUpload, CudaVideoCompositor,
            FrameCounter, PacketCounter, SwScaler, VideoCompositorOptions, VideoLayer, VideoRect,
        },
        pipeline::Pipeline,
    };

    use crate::common::{
        MIB, Trend, Unit, exclusive, gpu::nvidia_process_bytes, iterations, settle, soak_duration,
        try_test_video,
    };
    use crate::{HEIGHT, Teardown, WARMUP, WIDTH, frame_rate, test_source};

    const SCALED_WIDTH: u32 = 176;
    const SCALED_HEIGHT: u32 = 144;

    /// One device for the whole scenario, on purpose: creating and
    /// destroying a `CudaDevice` retains and releases the process-wide
    /// primary context, which is exactly what `test_support`'s own CUDA
    /// helper documents as unsafe to churn next to in-flight NVDEC/NVENC
    /// work. The elements each take their own reference to it per cycle,
    /// which is the ownership this scenario is actually measuring.
    fn cycle(device: &CudaDevice, teardown: Teardown) -> usize {
        let (counter, frames) = FrameCounter::new("counter");
        let pipeline = Pipeline::new("soak-cuda", test_source("video"), move |source, ctx| {
            let to_nv12 = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                WIDTH,
                HEIGHT,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new("upload", device, CudaFrameFormat::Nv12, WIDTH, HEIGHT)?;
            let scaler = CudaScaler::new(
                "scaler",
                device,
                SCALED_WIDTH,
                SCALED_HEIGHT,
                CudaScalerInterp::Bilinear,
            );
            let download = CudaDownload::new(
                "download",
                device,
                CudaFrameFormat::Nv12,
                SCALED_WIDTH,
                SCALED_HEIGHT,
            );
            let branch = ctx
                .branch()
                .pipe(to_nv12)
                .pipe(upload)
                .pipe(scaler)
                .queue("gpu-frames", 4)
                .pipe(download)
                .to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the CUDA pipeline");

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// NVDEC's surface pool is fixed-size *and* capped at 32 surfaces
    /// including the codec's own reference frames (see `CudaDecoder::new`),
    /// so a decode scenario's queue depth is part of that budget rather
    /// than a free choice.
    const DECODE_QUEUE_DEPTH: usize = 4;

    /// NVDEC's process-side driver cache takes substantially longer to
    /// settle than the lazily initialized tables covered by [`WARMUP`]. On
    /// this machine the first 15 cycles climbed by 5-7 MiB each, but a
    /// 250-cycle A/B run both before and after the `AvBufferRef` RAII change
    /// repeatedly returned tens of MiB and flattened to +0.083 MiB/cycle.
    /// Discard enough cycles to measure after that cache has saturated;
    /// otherwise this scenario deterministically reports driver warm-up as
    /// a decoder leak.
    const DECODE_CACHE_WARMUP: usize = 50;

    /// Decode a real file straight into CUDA memory. The decoder's pool is
    /// allocated per cycle and, unlike everything else measured here, is
    /// capped — a cycle that failed to release one would run the next
    /// cycle's `cuvidCreateDecoder` into that cap rather than merely using
    /// more memory.
    fn decode_cycle(device: &CudaDevice, path: &str, teardown: Teardown) -> usize {
        let (source, index, parameters) = crate::open_fixture(path);
        let (counter, frames) = FrameCounter::new("counter");
        let pipeline = Pipeline::new("soak-cuda-decode", source, move |source, ctx| {
            let decoder =
                CudaDecoder::new("decoder", parameters, device, DECODE_QUEUE_DEPTH as i32)?;
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("decoded-frames", DECODE_QUEUE_DEPTH)
                .to(Box::new(counter))?;
            ctx.attach(source, index, branch)?;
            Ok(())
        })
        .expect(
            "wire the CUDA decode pipeline — a failure here after the first cycle means \
             an earlier decoder never released its NVDEC surfaces, which are capped at 32",
        );

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// What every NVENC cycle below encodes with — same reasoning as the
    /// D3D11 module's own options helper.
    fn nvenc_options(time_base: ffmpeg::Rational) -> CudaEncoderOptions {
        CudaEncoderOptions {
            codec: CudaCodec::H264,
            input_format: CudaFrameFormat::Nv12,
            width: WIDTH,
            height: HEIGHT,
            time_base,
            frame_rate: frame_rate(),
            bit_rate: 1_000_000,
            gop_size: 30,
        }
    }

    /// Encode CUDA-resident surfaces on NVENC. A session that outlived its
    /// cycle would run a later cycle into the driver's concurrent-session
    /// cap, so opening successfully every cycle is itself the assertion.
    fn encode_cycle(device: &CudaDevice, teardown: Teardown) -> usize {
        let (counter, packets) = PacketCounter::new("counter");
        let source = test_source("video");
        let time_base = source.time_base();
        let pipeline = Pipeline::new("soak-cuda-nvenc", source, move |source, ctx| {
            let to_nv12 = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                WIDTH,
                HEIGHT,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new("upload", device, CudaFrameFormat::Nv12, WIDTH, HEIGHT)?;
            let encoder = CudaEncoder::new("encoder", device, nvenc_options(time_base))?;
            let branch = ctx
                .branch()
                .pipe(to_nv12)
                .pipe(upload)
                .queue("gpu-frames", 4)
                .pipe(encoder)
                .to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect(
            "wire the CUDA NVENC pipeline — a failure here after the first cycle means an \
             earlier encoder never closed its NVENC session",
        );

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(250));
        teardown.apply(&pipeline);
        packets.load(Ordering::Relaxed)
    }

    /// Whether this GPU and FFmpeg build have NVENC at all.
    fn nvenc_supported(device: &CudaDevice) -> bool {
        let time_base = test_source("probe").time_base();
        match CudaEncoder::new("probe", device, nvenc_options(time_base)) {
            Ok(_) => true,
            Err(error) => {
                eprintln!("skipping: no CUDA NVENC encoder on this machine ({error})");
                false
            }
        }
    }

    /// Whether NVDEC has a decoder for the fixture's codec on this machine.
    fn decode_supported(device: &CudaDevice, path: &str) -> bool {
        let (_source, _index, parameters) = crate::open_fixture(path);
        match CudaDecoder::new("probe", parameters, device, 0) {
            Ok(_) => true,
            Err(error) => {
                eprintln!("skipping: no NVDEC decoder for this fixture ({error})");
                false
            }
        }
    }

    /// A CUDA device, or `None` after printing why. Kept for the whole
    /// scenario on purpose: creating and destroying a `CudaDevice` retains
    /// and releases the process-wide primary context, which
    /// `test_support`'s own CUDA helper documents as unsafe to churn next
    /// to in-flight NVDEC/NVENC work.
    ///
    /// `pub(crate)` for the `pipewire` module's GPU capture scenario, whose
    /// frames land in CUDA memory through this same device.
    pub(crate) fn try_device() -> Option<CudaDevice> {
        match CudaDevice::new() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: no usable CUDA device on this machine ({error})");
                None
            }
        }
    }

    /// Runs `cycle` and judges what this platform lets us see. Both
    /// scenarios below differ only in the graph each cycle builds.
    fn measure_cycles(
        label: &str,
        post_settle_warmup: usize,
        iterations: usize,
        max_memory_slope: f64,
        mut cycle: impl FnMut(Teardown) -> usize,
    ) {
        let mut memory = Trend::private_bytes(format!("{label} private bytes"));
        let mut gpu = match nvidia_process_bytes() {
            Some(_) => Some(Trend::new(
                format!("{label} driver-reported GPU memory"),
                Unit::Bytes,
                || nvidia_process_bytes().unwrap_or_default(),
            )),
            None => {
                eprintln!(
                    "note: this driver does not report per-process GPU memory; measuring private \
                     bytes only"
                );
                None
            }
        };

        let measurement_start = WARMUP + post_settle_warmup;
        for index in 0..(measurement_start + iterations) {
            let teardown = Teardown::for_cycle(index);
            let frames = cycle(teardown);
            // See the D3D11 module's own note: only a `finish` cycle owes
            // us output, since `stop` abandons whatever a stateful stage
            // was still holding.
            if teardown == Teardown::Finish {
                assert!(
                    frames > 0,
                    "{label} {index} pushed nothing through the CUDA path before draining"
                );
            }
            if index + 1 == WARMUP {
                settle();
            }
            if index >= measurement_start {
                memory.sample();
                if let Some(gpu) = gpu.as_mut() {
                    gpu.sample();
                }
            }
        }

        memory.assert_flat(max_memory_slope);
        if let Some(gpu) = gpu {
            // Reported in whole MiB, so a threshold below that would fail
            // on quantization alone.
            gpu.assert_flat(1.0 * MIB);
        }
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn upload_scale_and_download_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(device) = try_device() else { return };

        // More cycles than the D3D11 scenario on purpose: the CUDA
        // driver's own host-side allocations make private bytes jitter by
        // roughly +-4 MiB peak to peak, several times what the CPU-only
        // scenarios show, so a longer window and a looser threshold are
        // what keep that jitter from dominating the fitted slope.
        //
        // At the default count this gauge also shows a climb of about
        // +0.1 MiB per cycle, which is the same host-side allocation
        // settling rather than anything retained: measured over 250
        // cycles it is a bounded ~14 MiB rise that stops outright, its
        // slope falling +0.075 over the first half, +0.032 over the
        // second, and exactly 0.000 over the last 62 cycles. A leak would
        // hold its slope as the window grows instead of flattening.
        measure_cycles("cuda cycle", 0, iterations(25), 1.0 * MIB, |teardown| {
            cycle(&device, teardown)
        });
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn decode_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(path) = try_test_video() else { return };
        let Some(device) = try_device() else { return };
        if !decode_supported(&device, &path) {
            return;
        }

        measure_cycles(
            "cuda decode cycle",
            DECODE_CACHE_WARMUP,
            iterations(50),
            1.0 * MIB,
            |teardown| decode_cycle(&device, &path, teardown),
        );
    }

    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn nvenc_encode_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(device) = try_device() else { return };
        if !nvenc_supported(&device) {
            return;
        }

        measure_cycles(
            "cuda nvenc cycle",
            0,
            iterations(15),
            1.0 * MIB,
            |teardown| encode_cycle(&device, teardown),
        );
    }

    /// The CUDA half of `a_running_compositor_does_not_grow_while_it_answers_frames`.
    ///
    /// The cycle scenarios above build and tear a graph down, which is the
    /// wrong shape to see retention per *frame*: `CudaVideoCompositor` holds
    /// the surface it may offer again when nothing changed, and `CudaUpload`
    /// and `CudaConverter` hold their own inputs and outputs for the same
    /// reason. What is retained is a surface out of an FFmpeg hardware frames
    /// pool, so the gauge that matters is the driver's per-process GPU
    /// memory; at this rate a per-tick retention is tens of MiB a second, and
    /// a 250 ms cycle would not notice.
    ///
    /// The input changes on every frame, deliberately: that is the case the
    /// repeat path cannot help with, where every tick composes and replaces
    /// what was held. A queue behind the compositor keeps some of those
    /// frames alive while it does, so the release has something to prove.
    /// The still case retains strictly less.
    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn a_running_cuda_compositor_does_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(device) = try_device() else { return };

        let (compositor, handle) = CudaVideoCompositor::new(
            "compositor",
            &device,
            VideoCompositorOptions {
                width: WIDTH,
                height: HEIGHT,
                frame_rate: frame_rate(),
                background: Color::new(16, 16, 16),
            },
        )
        .expect("build the compositor");
        let layer_sink = handle
            .add_source(
                "moving",
                VideoLayer::new(VideoRect::new(0, 0, WIDTH, HEIGHT)),
            )
            .expect("register the compositor input")
            .sink;

        // Borrowed into the builder, the way every cycle above does it:
        // `CudaDevice` is not cloneable, and retaining the primary context a
        // second time is what its own docs warn against next to in-flight work.
        let input_device = &device;
        let feeder = Pipeline::new("soak-running-cuda-input", test_source("video"), {
            move |source, ctx| {
                let to_nv12 = SwScaler::new(
                    "to-nv12",
                    ffmpeg::format::Pixel::NV12,
                    WIDTH,
                    HEIGHT,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                );
                let upload =
                    CudaUpload::new("upload", input_device, CudaFrameFormat::Nv12, WIDTH, HEIGHT)?;
                let branch = ctx.branch().pipe(to_nv12).pipe(upload).to(layer_sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            }
        })
        .expect("wire the compositor input pipeline");

        let (counter, frames) = FrameCounter::new("counter");
        let output = Pipeline::new("soak-running-cuda", compositor, |source, ctx| {
            // As in the software and D3D11 scenarios: without a queue nothing
            // is ever still referenced when the next composite replaces it.
            let branch = ctx.branch().queue("composited", 4).to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the compositor output pipeline");
        output.run().unwrap();
        feeder.run().unwrap();

        let duration = soak_duration(20);
        let sample_interval = Duration::from_secs(1);
        let mut memory = Trend::private_bytes("running cuda compositor private bytes");
        let mut gpu = match nvidia_process_bytes() {
            Some(_) => Some(Trend::new(
                "running cuda compositor driver-reported GPU memory",
                Unit::Bytes,
                || nvidia_process_bytes().unwrap_or_default(),
            )),
            None => {
                eprintln!(
                    "note: this driver does not report per-process GPU memory; measuring private \
                     bytes only"
                );
                None
            }
        };
        // The hardware frames pools fill over the first interval — the
        // compositor's own, `CudaUpload`'s, and the surface the compositor
        // holds to offer again — and every later one is steady state, which
        // is the only part a trend can be read from.
        thread::sleep(sample_interval);
        settle();
        let started = frames.load(Ordering::Relaxed);
        let deadline = std::time::Instant::now() + duration;
        while std::time::Instant::now() < deadline {
            thread::sleep(sample_interval);
            memory.sample();
            if let Some(gpu) = gpu.as_mut() {
                gpu.sample();
            }
        }
        let composited = frames.load(Ordering::Relaxed) - started;
        feeder.finish();
        output.finish();

        assert!(
            composited > 0,
            "the compositor stopped emitting, so a flat trend proves nothing"
        );
        eprintln!("composited {composited} frames over {duration:?}");
        // Private bytes carry the CUDA driver's own host-side jitter, which
        // the cycle scenarios above measure at several MiB peak to peak, so
        // this gauge is deliberately the looser of the two.
        memory.assert_flat(1.0 * MIB);
        if let Some(gpu) = gpu {
            // One retained NV12 surface at this size is a fraction of a MiB
            // and the driver reports in whole MiB, so the threshold is
            // quantization rather than a fraction of what a per-tick leak
            // would be: that is tens of MiB a second.
            gpu.assert_flat(1.0 * MIB);
        }
    }
}

/// The Linux desktop-capture path, the counterpart of the D3D11 module's
/// two DXGI capture scenarios. Each cycle runs a whole portal handshake and
/// PipeWire stream — session, node, negotiated format, and in GPU mode a
/// set of DMA-BUF imports — so what these measure is whether a session
/// releases all of that when its source drops.
///
/// Both need `MEDIA_PP_SOAK_RESTORE_TOKEN`; see `common::try_restore_token`
/// for why that cannot be defaulted or detected.
#[cfg(all(target_os = "linux", feature = "pipewire-screen-capture"))]
mod pipewire {
    use std::{sync::atomic::Ordering, thread, time::Duration};

    use media_pp::{
        elements::{
            CaptureSourceKind, FrameCounter, PipeWireScreenCaptureOptions,
            PipeWireScreenCaptureSource,
        },
        pipeline::Pipeline,
    };

    use crate::common::{MIB, Trend, exclusive, iterations, settle, try_restore_token};
    use crate::{Teardown, WARMUP};

    /// Every ScreenCast session **this process** holds open on the portal, by
    /// object path.
    ///
    /// Read from the portal itself rather than from anything this crate
    /// tracks, because what is being tested is the portal's own view: a
    /// session it still holds is one still shown as "screen is being shared",
    /// whatever the process believes.
    ///
    /// Filtered by the owning connection's pid, which is the only thing that
    /// makes this scenario sound. Every gauge in this file is isolated by
    /// `isolate!` giving the scenario its own process — but the portal's
    /// session tree is system-wide, and `exclusive()` is an in-process mutex
    /// that the parent never reaches, so the capture scenarios above really do
    /// run alongside this one, opening and dropping sessions of their own. The
    /// portal files each session under the unique name of the connection that
    /// opened it, and the bus will say which pid owns that name.
    ///
    /// This also settles what `ashpd` will not: it opens its connection lazily
    /// into a private `OnceLock` and never exposes it, so the unique name its
    /// sessions live under cannot be asked for directly.
    fn portal_sessions() -> Vec<String> {
        use ashpd::zbus;

        const ROOT: &str = "/org/freedesktop/portal/desktop/session";

        async fn children(connection: &zbus::Connection, path: &str) -> Vec<String> {
            let Ok(proxy) = zbus::fdo::IntrospectableProxy::builder(connection)
                .destination("org.freedesktop.portal.Desktop")
                .expect("a valid destination")
                .path(path.to_owned())
                .expect("a valid path")
                .build()
                .await
            else {
                return Vec::new();
            };
            // A path with no node is not a failure — it is zero children.
            let Ok(xml) = proxy.introspect().await else {
                return Vec::new();
            };
            xml.split("<node name=\"")
                .skip(1)
                .filter_map(|rest| rest.split('"').next())
                .map(|name| format!("{path}/{name}"))
                .collect()
        }

        pollster::block_on(async {
            let connection = zbus::Connection::session()
                .await
                .expect("a session bus to read the portal from");
            let bus = zbus::fdo::DBusProxy::new(&connection)
                .await
                .expect("the bus's own interface");
            let me = std::process::id();

            let mut sessions = Vec::new();
            for owner in children(&connection, ROOT).await {
                // The node name is the owning connection's unique name with
                // its dots turned into underscores: `1_290` is `:1.290`.
                let unique = format!(
                    ":{}",
                    owner.rsplit('/').next().expect("a non-empty node name").replace('_', ".")
                );
                let Ok(name) = zbus::names::BusName::try_from(unique) else {
                    continue;
                };
                // A connection that has gone since the tree was read is not
                // ours, and asking about it fails rather than answering.
                if bus.get_connection_unix_process_id(name).await != Ok(me) {
                    continue;
                }
                sessions.extend(children(&connection, &owner).await);
            }
            sessions.sort();
            sessions
        })
    }

    /// `ashpd::desktop::Session` has no `Drop` of its own, and every proxy in
    /// the process shares one cached `zbus` connection — so nothing about
    /// dropping a capture closes its portal session, and without an explicit
    /// `Close` the portal keeps it, and its "screen is being shared"
    /// indicator, until the process exits.
    ///
    /// A cycle scenario would not catch this. The sessions are on the portal's
    /// side of the bus, so they cost this process almost nothing and no memory
    /// gauge here moves; the only thing that notices is the user, and the
    /// compositor.
    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn dropping_a_capture_closes_its_portal_session() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(restore_token) = try_restore_token() else {
            return;
        };
        if !capture_supported(&restore_token) {
            return;
        }

        let before = portal_sessions();
        let opened: Vec<_> = (0..3)
            .map(|index| {
                PipeWireScreenCaptureSource::open(format!("probe-{index}"), options(&restore_token))
                    .expect("open a capture with the restore token")
                    .0
            })
            .collect();

        let during = portal_sessions();
        assert_eq!(
            during.len(),
            before.len() + 3,
            "three captures did not open three portal sessions; before {before:?}, during \
             {during:?}"
        );

        drop(opened);
        let after = portal_sessions();
        assert_eq!(
            after, before,
            "a dropped capture left its portal session open — the portal still shows the screen \
             as being shared. before {before:?}, after {after:?}"
        );
    }

    /// What every cycle below opens with. A monitor rather than a window:
    /// the token restores whichever source was picked once, and a monitor
    /// is the one kind that is certain to still exist on a later run.
    fn options(restore_token: &str) -> PipeWireScreenCaptureOptions {
        PipeWireScreenCaptureOptions {
            fps: 30,
            source_kind: CaptureSourceKind::Monitor,
            include_cursor: false,
            restore_token: Some(restore_token.to_owned()),
        }
    }

    /// How long a cycle lets the capture run. Longer than the CPU-only
    /// scenarios' cycles for the same reason the DXGI one is: this source
    /// emits on its own fixed schedule, so a cycle has to span several of
    /// its ticks to prove frames really flowed.
    const CAPTURE_MILLIS: u64 = 400;

    /// One CPU capture session: portal handshake, stream, frames into
    /// pooled BGRA `AVFrame`s, teardown.
    fn cpu_cycle(restore_token: &str, teardown: Teardown) -> usize {
        let (counter, frames) = FrameCounter::new("counter");
        let (source, _format, _token) =
            PipeWireScreenCaptureSource::open("capture", options(restore_token)).expect(
                "open the portal capture — a failure here after the first cycle means an \
                 earlier source never closed its portal session or PipeWire stream",
            );

        let pipeline = Pipeline::new("soak-pipewire-capture", source, |source, ctx| {
            let branch = ctx.branch().queue("captured", 4).to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the capture pipeline");

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(CAPTURE_MILLIS));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// The same session in GPU mode: `open_gpu` negotiates DMA-BUF only and
    /// imports each buffer into CUDA memory, so a cycle also creates and
    /// destroys the EGL display, the cached `EGLImage`s, and a CUDA frames
    /// context. None of that is visible to the process heap, which is why
    /// this scenario asks the driver for its own number.
    #[cfg(feature = "cuda")]
    fn gpu_cycle(
        device: &media_pp::elements::CudaDevice,
        restore_token: &str,
        teardown: Teardown,
    ) -> usize {
        let (counter, frames) = FrameCounter::new("counter");
        let (source, _format, _token) =
            PipeWireScreenCaptureSource::open_gpu("capture", options(restore_token), device)
                .expect(
                    "open the portal capture in GPU mode — a failure here after the first \
                     cycle means an earlier source never released its EGL images or CUDA \
                     surfaces",
                );

        let pipeline = Pipeline::new("soak-pipewire-capture-gpu", source, |source, ctx| {
            let branch = ctx.branch().queue("captured", 4).to(Box::new(counter))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wire the GPU capture pipeline");

        pipeline.run().unwrap();
        thread::sleep(Duration::from_millis(CAPTURE_MILLIS));
        teardown.apply(&pipeline);
        frames.load(Ordering::Relaxed)
    }

    /// Whether this desktop will hand out a capture at all with the token
    /// it was given, so that a stale token or a compositor without the
    /// portal skips with a reason instead of failing the suite.
    ///
    /// The probe is the honest place to find out: an `open` whose token no
    /// longer restores falls back to the picker, and the picker being
    /// dismissed is `Cancelled` — a routine outcome, not a malfunction.
    fn capture_supported(restore_token: &str) -> bool {
        match PipeWireScreenCaptureSource::open("probe", options(restore_token)) {
            Ok(_) => true,
            Err(error) => {
                eprintln!("skipping: no portal screen capture available here ({error})");
                false
            }
        }
    }

    /// Runs `cycle` and judges what this platform lets us see — the same
    /// shape as the `cuda` module's own helper, since the GPU gauge is the
    /// same driver query.
    ///
    /// `watch_gpu` is what the cycle's own mode decides: the CPU path never
    /// creates a CUDA context, so querying the driver for its share would
    /// only add a flat zero to the report and assert nothing.
    fn measure_cycles(
        label: &str,
        watch_gpu: bool,
        iterations: usize,
        max_memory_slope: f64,
        mut cycle: impl FnMut(Teardown) -> usize,
    ) {
        use crate::common::{Unit, gpu::nvidia_process_bytes};

        let mut memory = Trend::private_bytes(format!("{label} private bytes"));
        let mut gpu = match nvidia_process_bytes() {
            Some(_) if watch_gpu => Some(Trend::new(
                format!("{label} driver-reported GPU memory"),
                Unit::Bytes,
                || nvidia_process_bytes().unwrap_or_default(),
            )),
            None if watch_gpu => {
                eprintln!(
                    "note: this driver does not report per-process GPU memory; measuring private \
                     bytes only"
                );
                None
            }
            _ => None,
        };

        for index in 0..(WARMUP + iterations) {
            let teardown = Teardown::for_cycle(index);
            let frames = cycle(teardown);
            // Only a `finish` cycle owes us output, same as everywhere
            // else here — and an idle desktop still produces frames,
            // because this source re-emits its latest image on its own
            // schedule rather than only when the screen changes.
            if teardown == Teardown::Finish {
                assert!(
                    frames > 0,
                    "{label} {index} captured nothing before draining"
                );
            }
            if index + 1 == WARMUP {
                settle();
            }
            if index >= WARMUP {
                memory.sample();
                if let Some(gpu) = gpu.as_mut() {
                    gpu.sample();
                }
            }
        }

        memory.assert_flat(max_memory_slope);
        if let Some(gpu) = gpu {
            // Reported in whole MiB, so a threshold below that would fail
            // on quantization alone.
            gpu.assert_flat(1.0 * MIB);
        }
    }

    /// The CPU path: every captured image is copied into a pooled frame in
    /// system memory, so a session that kept its buffers shows up in
    /// private bytes directly — a screen's worth of BGRA is ~8 MiB at
    /// 1080p, sixteen times this threshold.
    ///
    /// What it does show is a bounded step, not a ramp: over 40 cycles on
    /// the development machine private bytes climbed 74.4 -> 75.1 MiB and
    /// then stayed there, all of it inside the first twenty cycles. Same
    /// shape as the DXGI CPU scenario's own note — an allocator and driver
    /// settling into a working set, which no per-cycle threshold above it
    /// mistakes for growth.
    #[test]
    #[ignore = "soak test; run with --ignored"]
    fn cpu_desktop_capture_cycles_do_not_grow_process_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(token) = try_restore_token() else {
            return;
        };
        if !capture_supported(&token) {
            return;
        }

        measure_cycles(
            "pipewire capture cycle",
            false,
            iterations(10),
            0.5 * MIB,
            |teardown| cpu_cycle(&token, teardown),
        );
    }

    /// The zero-copy path: nothing captured reaches system memory, so the
    /// driver's per-process figure carries this scenario and private bytes
    /// only covers the host side of the EGL and CUDA bookkeeping.
    ///
    /// That driver figure is the sharp instrument here: 40 cycles measured
    /// 82.0 MiB on every single one, so a retained frames context or a
    /// leaked DMA-BUF import has nowhere to hide. Private bytes gets the
    /// looser threshold instead, because the CUDA and EGL host allocations
    /// arrive as occasional sub-MiB steps rather than a flat line.
    #[test]
    #[ignore = "soak test; run with --ignored"]
    #[cfg(feature = "cuda")]
    fn gpu_desktop_capture_cycles_do_not_grow_gpu_memory() {
        isolate!();
        let _exclusive = exclusive();
        media_pp::init().expect("ffmpeg init");
        let Some(token) = try_restore_token() else {
            return;
        };
        let Some(device) = crate::cuda::try_device() else {
            return;
        };
        if !capture_supported(&token) {
            return;
        }

        measure_cycles(
            "pipewire gpu capture cycle",
            true,
            iterations(10),
            1.0 * MIB,
            |teardown| gpu_cycle(&device, &token, teardown),
        );
    }
}
