//! Shared fixtures for this crate's own tests.

use crate::elements::VideoCodec;

/// Path to a video file for tests that need one — synthesized here, every
/// time, on every machine.
///
/// No media is checked into this repository, and until this built its own
/// there was nothing to open: the tests below this line skipped unless
/// someone had pointed an environment variable at a recording of their own,
/// which on CI nobody ever had. A skipped test reports as passing, so what
/// those tests covered was, everywhere it mattered most, not covered.
///
/// A file of this crate's own making is also the same file everywhere. What
/// a variable pointed at was whatever the machine happened to have — a clip
/// too short for the seek tests, a container with no sound, a file that had
/// been swapped since — and a failure then said nothing about the library
/// until someone had worked out which of the two was wrong.
///
/// # What it costs
///
/// Coverage of what only a real recording has: B-frames, an edit list, a
/// start time that is not zero, frame intervals that vary. Nothing built
/// here has any of those — see [`synthesize`] — so this crate's demuxing,
/// seeking and decoding are exercised against its own encoder's output and
/// no other. `lib/tests/soak.rs` still reads `MEDIA_PP_TEST_VIDEO` and is
/// where a real recording gets to say something.
///
/// Returns `None` — after printing why — only when the fixture cannot be
/// built at all, which needs an FFmpeg without the two encoders this project
/// treats as always present. Tests using this must still assert real
/// behavior when it does return a path: skipping is for the machine that
/// cannot build one, not a way to make a failing assertion optional.
pub(crate) fn try_test_video() -> Option<String> {
    /// Long enough for the seek tests, which reposition to three seconds and
    /// to the middle of the file.
    const SECONDS: f64 = 8.0;
    /// Not the rate a mix runs at, deliberately — see [`synthesize`].
    const AUDIO_RATE: u32 = 44_100;

    static SYNTHESIZED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    SYNTHESIZED
        .get_or_init(|| {
            match build_fixture(
                "test-video",
                SECONDS,
                AUDIO_RATE,
                VideoCodec::OpenH264,
                None,
            ) {
                Ok(fixture) => {
                    eprintln!("using a synthesized fixture: {}", fixture.path.display());
                    Some(fixture.path.to_string_lossy().into_owned())
                }
                Err(error) => {
                    eprintln!("skipping: could not synthesize a test video: {error}");
                    None
                }
            }
        })
        .clone()
}

/// A hardware D3D11 device and its shared immediate context for unit tests.
/// Prints the platform error and returns `None` when the machine cannot create
/// one, so callers can use the repository's normal hardware-test skip path.
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub(crate) fn try_d3d11_device() -> Option<(
    windows::Win32::Graphics::Direct3D11::ID3D11Device,
    std::sync::Arc<std::sync::Mutex<windows::Win32::Graphics::Direct3D11::ID3D11DeviceContext>>,
)> {
    use windows::Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{D3D11_SDK_VERSION, D3D11CreateDevice},
    };

    let mut device = None;
    let mut context = None;
    // SAFETY: null adapter/software pointers select the hardware driver path,
    // feature levels use D3D defaults, and `device`/`context` are live,
    // correctly typed out-parameters.
    let result = unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            Default::default(),
            Default::default(),
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    };
    if let Err(error) = result {
        eprintln!("skipping: D3D11CreateDevice failed on this machine: {error}");
        return None;
    }
    Some((
        device.expect("D3D11CreateDevice succeeded without producing a device"),
        std::sync::Arc::new(std::sync::Mutex::new(
            context.expect("D3D11CreateDevice succeeded without producing a context"),
        )),
    ))
}

/// A hardware D3D11 device created with `D3D11_CREATE_DEVICE_SINGLETHREADED`.
///
/// Every entry point here that accepts a caller-owned device has to refuse one
/// of these, because the flag promises the runtime that the device is used from
/// a single thread and a pipeline cannot keep that promise. Skips the same way
/// as [`try_d3d11_device`].
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub(crate) fn try_single_threaded_d3d11_device()
-> Option<windows::Win32::Graphics::Direct3D11::ID3D11Device> {
    use windows::Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_FLAG,
            D3D11_CREATE_DEVICE_SINGLETHREADED, D3D11_SDK_VERSION, D3D11CreateDevice,
        },
    };

    let mut device = None;
    let mut context = None;
    // SAFETY: null adapter/software pointers select the hardware driver path,
    // feature levels use D3D defaults, and `device`/`context` are live,
    // correctly typed out-parameters.
    let result = unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            Default::default(),
            D3D11_CREATE_DEVICE_FLAG(
                D3D11_CREATE_DEVICE_BGRA_SUPPORT.0 | D3D11_CREATE_DEVICE_SINGLETHREADED.0,
            ),
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
    };
    if let Err(error) = result {
        eprintln!("skipping: D3D11CreateDevice failed on this machine: {error}");
        return None;
    }
    device
}

/// A hardware D3D12 device for unit tests, with the same graceful skip
/// behavior as the D3D11 test-device helper.
#[cfg(all(target_os = "windows", feature = "d3d12"))]
pub(crate) fn try_d3d12_device() -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Device> {
    use windows::Win32::Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0, Direct3D12::D3D12CreateDevice,
    };

    let mut device = None;
    // SAFETY: a null adapter requests the default hardware adapter and
    // `device` is the correctly typed live out-parameter for the requested
    // minimum feature level.
    if let Err(error) = unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device) } {
        eprintln!("skipping: D3D12CreateDevice failed on this machine: {error}");
        return None;
    }
    device
}

/// A CUDA device for a hardware test, together with the lock that keeps
/// CUDA tests from overlapping. `None` — after printing why — on a machine
/// without a usable device, the same way [`try_test_video`] skips without a
/// fixture.
///
/// The two are returned together because they are not separable in practice.
/// Creating or destroying a `CudaDevice` retains/releases the *process-wide*
/// CUDA primary context (see [`crate::elements::CudaDevice`]'s own docs on
/// why it uses that context), and doing so on one thread while another
/// thread has NVDEC or NVENC work in flight segfaults inside `libnvcuvid` —
/// on a thread the driver itself owns, so nothing in this crate can catch or
/// recover it.
///
/// Running the whole suite hid that, because cheap tests were interleaved
/// between the CUDA ones often enough to keep them from overlapping; a run
/// filtered down to CUDA tests alone (`cargo test --features cuda cuda_`)
/// crashed reliably. Bind the guard for the body of the test:
///
/// ```ignore
/// let Some((device, _cuda_lock)) = try_cuda_device() else {
///     return;
/// };
/// ```
///
/// A test needing a *second* device — to check that a frame from a foreign
/// context is rejected — calls `CudaDevice::new()` directly rather than this
/// again: the lock is already held, and it does not nest.
#[cfg(feature = "cuda")]
pub(crate) fn try_cuda_device() -> Option<(
    crate::elements::CudaDevice,
    std::sync::MutexGuard<'static, ()>,
)> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A test that panics while holding this must not turn every later CUDA
    // test into a poison error instead of its own real result.
    let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match crate::elements::CudaDevice::new() {
        Ok(device) => Some((device, guard)),
        Err(error) => {
            eprintln!("skipping: no usable CUDA device on this machine ({error})");
            None
        }
    }
}

/// A media file this crate builds for itself, rather than one a test has to
/// find.
///
/// Every fixture these tests run against, and the same one on every machine:
/// two streams that have to stay together, a sample rate that differs from a
/// mix's, a duration to measure an output against. Those a synthetic file has
/// just as truly as a recording, and unlike a recording it is always there and
/// always the same.
///
/// What it does not have is what only a camera or a screen recorder produces
/// — B-frames, variable frame timing, an edit list, a start time that is not
/// zero. `libopenh264` will not emit B-frames whatever `max_b_frames` asks
/// for (measured, in `build_fixture`), and the encoders that would are not
/// ones this project can rely on being installed. `lib/tests/soak.rs` reads
/// `MEDIA_PP_TEST_VIDEO` for a real recording; nothing here does.
///
/// # Why nothing here shells out to `ffmpeg`
///
/// There is no `ffmpeg` to shell out to. CI installs FFmpeg's *development
/// libraries* from a pinned vcpkg port and no command-line tool at all, and
/// the hosted runner images no longer carry one. The codecs are chosen on the
/// same footing: OpenH264 is the one H.264 encoder this project treats as
/// always present — `.github/actions/setup-ffmpeg` adds it to the port for
/// exactly that reason, x264 being GPL and therefore absent — and FFmpeg's
/// native AAC encoder is built into every configuration there is.
///
/// # It is left behind on purpose
///
/// In the temporary directory, under a fixed name, so the next run overwrites
/// it rather than accumulating another. Not deleted afterwards: what these
/// fixtures are for is measuring produced media, and a test that fails wants
/// its file still there to be opened. A `Drop` that cleaned up would delete
/// exactly the failing case, since unwinding runs it too.
pub(crate) fn synthesize(name: &str, seconds: f64, audio_rate: u32) -> Fixture {
    build_fixture(name, seconds, audio_rate, VideoCodec::OpenH264, None)
        .expect("synthesize a fixture")
}

/// The same, but with the packets in decode order rather than presentation
/// order.
///
/// A B-frame is coded from frames on both sides of it, so the encoder hands
/// its packets over out of order and their `dts` stops equalling their `pts`.
/// Reordering is a path a muxer, a payloader, a pacer and a seek all have to
/// get right, and the ordinary fixture never exercises it: `libopenh264`
/// emits no B-frames whatever it is asked for, and the H.264 encoders that
/// would are not ones this project can rely on being installed.
///
/// So this one is MPEG-4 Part 2, which is FFmpeg's own encoder and therefore
/// always there — see [`VideoCodec::Mpeg4`]. Not for tests about anything
/// else: a hardware decoder may well have no profile for it, and there is no
/// RTP payloader here that takes it.
pub(crate) fn synthesize_reordered(name: &str, seconds: f64) -> Fixture {
    build_fixture(name, seconds, 44_100, VideoCodec::Mpeg4, Some(2))
        .expect("synthesize a reordered fixture")
}

/// A synthesized file, and the facts about it a test may hold it to.
pub(crate) struct Fixture {
    pub(crate) path: std::path::PathBuf,
    pub(crate) audio_rate: u32,
    pub(crate) channels: u16,
}

/// Small on purpose: what these fixtures are asked about is timing and rates,
/// and a bigger picture only makes the encode slower.
const FIXTURE_WIDTH: u32 = 320;
const FIXTURE_HEIGHT: u32 = 240;
const FIXTURE_FPS: i32 = 30;

/// Builds one, or reports why it could not.
///
/// Takes `seconds` of real time: both synthetic sources pace themselves to the
/// rate they claim, which is what makes the file's own timing worth measuring.
///
/// `Stop` rather than an end of stream, because neither source has an end —
/// they generate until told otherwise. A muxer finalizes on either, so the
/// file that lands is complete.
fn build_fixture(
    name: &str,
    seconds: f64,
    audio_rate: u32,
    codec: VideoCodec,
    max_b_frames: Option<u32>,
) -> std::result::Result<Fixture, Box<dyn std::error::Error>> {
    use crate::elements::{
        AudioCodec, FileMuxer, SwAudioEncoder, SwAudioEncoderOptions, SwEncoder, SwEncoderOptions,
        SwScaler, TestAudioOptions, TestAudioSource, TestVideoOptions, TestVideoSource,
    };
    use crate::pipeline::PipelineBuilder;
    use ffmpeg_next as ffmpeg;

    crate::init()?;
    let directory = std::env::temp_dir().join("media-pp-fixtures");
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{name}.mp4"));
    let _ = std::fs::remove_file(&path);

    let channels = 2;
    let video = TestVideoSource::new(
        format!("{name}-video"),
        TestVideoOptions {
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            framerate: ffmpeg::Rational::new(FIXTURE_FPS, 1),
        },
    );
    let audio = TestAudioSource::new(
        format!("{name}-audio"),
        TestAudioOptions {
            sample_rate: audio_rate,
            channels,
            frequency: 440.0,
        },
    );
    let audio_time_base = audio.time_base();
    let video_time_base = ffmpeg::Rational::new(1, FIXTURE_FPS);

    let video_encoder = SwEncoder::new(
        format!("{name}-video-encoder"),
        SwEncoderOptions {
            codec,
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            time_base: video_time_base,
            frame_rate: ffmpeg::Rational::new(FIXTURE_FPS, 1),
            bit_rate: 1_000_000,
            // Deliberately not a whole number of seconds. Seek tests pick
            // round targets, and a keyframe every second would put one on
            // every target they pick: an accurate seek would land exactly,
            // the decoder would need no packets to reach it, and the tests
            // that exist to cover landing *between* keyframes would pass
            // without ever doing so. At 30fps this is a keyframe every
            // 1.333s, which no round target shares.
            gop_size: 40,
            max_b_frames,
        },
    )?;
    let audio_encoder = SwAudioEncoder::new(
        format!("{name}-audio-encoder"),
        SwAudioEncoderOptions {
            codec: AudioCodec::Aac,
            sample_rate: audio_rate,
            channels,
            time_base: audio_time_base,
            bit_rate: 128_000,
        },
    )?;

    let mut muxer = FileMuxer::create(&path)?;
    muxer.add_stream("video", video_encoder.parameters(), video_time_base)?;
    muxer.add_stream(
        "audio",
        audio_encoder.parameters(),
        audio_encoder.time_base(),
    )?;
    let mut sinks = muxer.open()?;
    let audio_sink = sinks.pop().expect("the audio stream was added second");
    let video_sink = sinks.pop().expect("the video stream was added first");

    // The encoder takes YUV420P and the synthetic source does not produce it,
    // the same conversion `tee_recording` puts in front of its own encoder.
    let scaler = SwScaler::new(
        format!("{name}-to-yuv"),
        ffmpeg::format::Pixel::YUV420P,
        FIXTURE_WIDTH,
        FIXTURE_HEIGHT,
        ffmpeg::software::scaling::Flags::BILINEAR,
    );

    let pipeline = PipelineBuilder::new(format!("{name}-fixture"))
        .add_source(video, move |source, context| {
            let branch = context
                .branch()
                .pipe(scaler)
                .pipe(video_encoder)
                .to(video_sink)?;
            context.attach(source, 0, branch)?;
            Ok(())
        })?
        .add_source(audio, move |source, context| {
            let branch = context.branch().pipe(audio_encoder).to(audio_sink)?;
            context.attach(source, 0, branch)?;
            Ok(())
        })?
        .build();

    pipeline.run()?;
    std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
    pipeline.stop();

    Ok(Fixture {
        path,
        audio_rate,
        channels,
    })
}
