//! One `ID3D11Device` driven by several elements on several threads.
//!
//! Each `Queue` in a pipeline is a thread boundary, and every D3D11 element
//! past one issues its GPU commands on the immediate context the device it was
//! handed owns. That context is not free-threaded, and the runtime's
//! `ID3D11Multithread` protection that makes it safe is off unless something
//! turns it on — so the property under test is not "does a frame arrive" but
//! "does a whole chain of unrelated D3D11 elements, each on its own thread,
//! share one device without corrupting it".
//!
//! Hardware test: skips with a reason when the machine has no usable D3D11
//! device or video processor.
#![cfg(all(target_os = "windows", feature = "d3d11"))]

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use media_pp::{
    buffer::MediaBuffer,
    bus::BusEvent,
    elements::{
        AppSink, ChromaKeyMethod, ChromaKeyOptions, D3d11ChromaKey, D3d11Download, D3d11Scaler,
        D3d11ScalerFormat, D3d11Upload, SwScaler, TestVideoOptions, TestVideoSource,
    },
    pipeline::Pipeline,
};
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11DeviceContext};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const SCALED_WIDTH: u32 = 160;
const SCALED_HEIGHT: u32 = 120;

/// A hardware device and the one shared immediate context every element below
/// is given, exactly as an application would build them.
fn try_shared_device() -> Option<(ID3D11Device, Arc<Mutex<ID3D11DeviceContext>>)> {
    use windows::Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice},
    };

    let mut device = None;
    let mut context = None;
    // SAFETY: a null adapter selects the default hardware driver, feature
    // levels use the D3D defaults, and both slots are live out-parameters.
    let result = unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            Default::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
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
        Arc::new(Mutex::new(context.expect(
            "D3D11CreateDevice succeeded without producing a context",
        ))),
    ))
}

/// Upload, scale, key, and download the same frames on four different threads
/// through one device.
///
/// The chain is deliberately longer than any single element's own test: a
/// device that is not protected produces interleaved bind/draw sequences on
/// the shared context, which shows up as corrupted output or a device removal
/// rather than as a returned error — so this asserts on frames arriving intact
/// and on an empty bus, not merely on construction succeeding.
#[test]
fn four_d3d11_elements_share_one_device_across_queue_boundaries() {
    media_pp::init().expect("initialize FFmpeg");
    let Some((device, context)) = try_shared_device() else {
        return;
    };

    let scaler = match D3d11Scaler::new(
        "d3d11-scale",
        &device,
        context.clone(),
        D3d11ScalerFormat::Preserve,
        SCALED_WIDTH,
        SCALED_HEIGHT,
    ) {
        Ok(scaler) => scaler,
        Err(error) => {
            eprintln!("skipping: this adapter has no usable video processor ({error})");
            return;
        }
    };
    let chroma_key = D3d11ChromaKey::new(
        "d3d11-key",
        &device,
        context.clone(),
        ChromaKeyOptions {
            method: ChromaKeyMethod::Green,
            threshold: 0.2,
            smoothing: 0.05,
        },
    )
    .expect("create the chroma key");
    let download = D3d11Download::new(
        "d3d11-download",
        &device,
        context.clone(),
        SCALED_WIDTH,
        SCALED_HEIGHT,
    )
    .expect("create the download");

    // The device arrived here unprotected — nothing above asked for it — so
    // this is the elements' doing, and it is what the rest of the test relies
    // on. A race is not something a passing run can prove the absence of; this
    // assertion is the deterministic half.
    {
        use windows::Win32::Graphics::Direct3D11::ID3D11Multithread;
        use windows::core::Interface;

        let context = context.lock().expect("shared context");
        let multithread: ID3D11Multithread =
            context.cast().expect("the immediate context exposes it");
        // SAFETY: reads one boolean property from the live context interface.
        let protected = unsafe { multithread.GetMultithreadProtected() };
        assert!(
            protected.as_bool(),
            "constructing D3D11 elements must have protected the shared context"
        );
    }

    let frames = Arc::new(AtomicUsize::new(0));
    let counted = frames.clone();
    let sink = AppSink::new("sink", move |buffer| {
        if let MediaBuffer::Video(frame) = buffer {
            assert_eq!(frame.format(), ffmpeg_next::format::Pixel::BGRA);
            assert_eq!(frame.width(), SCALED_WIDTH);
            assert_eq!(frame.height(), SCALED_HEIGHT);
            // A frame that survived four threads' worth of shared-context work
            // still has to carry its own pixels: an unprotected context tends
            // to hand back a blank or half-written surface rather than fail.
            assert!(
                frame.data(0).iter().any(|byte| *byte != 0),
                "a downloaded frame must not be blank"
            );
            counted.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    });

    let source = TestVideoSource::new(
        "test-video",
        TestVideoOptions {
            width: WIDTH,
            height: HEIGHT,
            framerate: ffmpeg_next::Rational::new(60, 1),
        },
    );
    let pipeline = Pipeline::new("d3d11-shared-device", source, |source, ctx| {
        let branch = ctx
            .branch()
            // Every `queue` here puts the element after it on its own thread.
            .queue("to-upload", 4)
            .pipe(SwScaler::new(
                "to-bgra",
                ffmpeg_next::format::Pixel::BGRA,
                WIDTH,
                HEIGHT,
                ffmpeg_next::software::scaling::Flags::BILINEAR,
            ))
            .pipe(D3d11Upload::new("d3d11-upload", &device, WIDTH, HEIGHT))
            .queue("to-scale", 4)
            .pipe(scaler)
            .queue("to-key", 4)
            .pipe(chroma_key)
            .queue("to-download", 4)
            .pipe(download)
            .to(Box::new(sink))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the shared-device pipeline");

    pipeline.run().expect("start the shared-device pipeline");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while frames.load(Ordering::Relaxed) < 30 && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    pipeline.stop();

    let errors: Vec<_> = pipeline
        .bus()
        .iter()
        .filter(|event| matches!(event, BusEvent::Error { .. }))
        .collect();
    assert!(errors.is_empty(), "unexpected pipeline errors: {errors:?}");
    assert!(
        frames.load(Ordering::Relaxed) >= 30,
        "only {} frames crossed four D3D11 elements on four threads",
        frames.load(Ordering::Relaxed)
    );
}

/// A compositor keeps its configured rate while a desktop capture shares its
/// device.
///
/// Sharing one device is not only about correctness — it is also a place one
/// element can quietly take the others' throughput. `AcquireNextFrame` holds
/// the device's own lock for as long as it waits, so a capture that waited
/// out its whole frame interval inside that call stalled everything else on
/// the device for most of every tick: a 60 fps compositor measured about 40,
/// and dropping the capture to 30 fps — a longer wait each tick — took it to
/// about 21. The rate asserted here is the compositor's, deliberately: it is
/// what a recording would be made of, and it must not follow its input's.
///
/// Hardware test: skips when the machine has no desktop duplication.
#[cfg(feature = "dxgi-capture")]
#[test]
fn a_capture_sharing_the_device_does_not_slow_the_compositor() {
    use media_pp::{
        color::Color,
        elements::{
            CaptureArea, CaptureMode, D3d11VideoCompositor, DxgiCaptureOptions, DxgiCaptureSource,
            VideoCompositorOptions, VideoFit, VideoLayer, VideoRect,
        },
    };

    const FPS: u32 = 60;
    /// One second of measurement, less the ramp-up a first tick costs.
    const MEASURED: Duration = Duration::from_millis(1500);
    /// The regression showed as roughly two thirds of the configured rate, so
    /// this sits above that and below what a healthy run reaches. A machine
    /// too slow to composite 1080p60 at all would trip it, which is why the
    /// canvas below is small.
    const MINIMUM: f64 = 0.85;

    media_pp::init().expect("initialize FFmpeg");
    let Some((device, context)) = try_shared_device() else {
        return;
    };

    // Composited at the capture's own resolution, which is the desktop's and
    // not something this test picks. A smaller canvas does not reproduce the
    // regression: the stall only shows once a composite tick is long enough
    // to collide with the capture's, and a tiny canvas finishes inside
    // whatever slice the capture leaves.
    let (capture, format) = match DxgiCaptureSource::open_with_device(
        "capture",
        DxgiCaptureOptions {
            area: CaptureArea::Output { output_index: 0 },
            fps: FPS,
            capture_mode: CaptureMode::Gpu,
        },
        &device,
    ) {
        Ok(opened) => opened,
        Err(error) => {
            eprintln!("skipping: no desktop duplication available here ({error})");
            return;
        }
    };

    let (compositor, handle) = D3d11VideoCompositor::new(
        "compositor",
        &device,
        context.clone(),
        VideoCompositorOptions {
            width: format.width,
            height: format.height,
            frame_rate: ffmpeg_next::Rational::new(FPS as i32, 1),
            background: Color::BLACK,
        },
    )
    .expect("create the compositor");

    let input = handle
        .add_source(
            "desktop",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, format.width, format.height))
            },
        )
        .expect("register the capture's layer")
        .expect("the compositor is still running");

    let composited = Arc::new(AtomicUsize::new(0));
    let counted = composited.clone();
    let sink = AppSink::new("composited", move |buffer| {
        if matches!(buffer, MediaBuffer::Video(_)) {
            counted.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    });

    let capture_pipeline = Pipeline::new("capture", capture, |source, ctx| {
        let branch = ctx.branch().queue("captured", 4).to(input.sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the capture pipeline");
    // Counted synchronously: a queue here would report its own worker's pace
    // rather than the compositor's, which is the number under test.
    let composite_pipeline = Pipeline::new("composite", compositor, |source, ctx| {
        let branch = ctx.branch().to(Box::new(sink))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire the compositing pipeline");

    composite_pipeline.run().expect("start compositing");
    capture_pipeline.run().expect("start capturing");
    // Counted from after the first frames, so neither pipeline's start-up is
    // charged against the rate.
    std::thread::sleep(Duration::from_millis(300));
    let started = std::time::Instant::now();
    let before = composited.load(Ordering::Relaxed);
    std::thread::sleep(MEASURED);
    let measured = composited.load(Ordering::Relaxed) - before;
    let elapsed = started.elapsed().as_secs_f64();
    capture_pipeline.stop();
    composite_pipeline.stop();

    for pipeline in [&capture_pipeline, &composite_pipeline] {
        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. }))
            .collect();
        assert!(errors.is_empty(), "unexpected pipeline errors: {errors:?}");
    }

    let rate = measured as f64 / elapsed;
    assert!(
        rate >= FPS as f64 * MINIMUM,
        "the compositor produced {rate:.1} fps of its configured {FPS} while a capture \
         shared its device"
    );
}
