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
