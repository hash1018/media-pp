//! Shared fixtures for this crate's own tests.

/// Path to a real video file for tests that need one, from
/// `MEDIA_PP_TEST_VIDEO`. Returns `None` — after printing why — when the
/// variable is unset or names something that is not a readable file, the same
/// way a hardware test's `try_device()` skips on a machine without the device.
///
/// No media is checked into this repository, so there is no default to fall
/// back to:
///
/// ```text
/// MEDIA_PP_TEST_VIDEO=/path/to/video.mp4 cargo test -p media-pp
/// ```
///
/// Any container `FileDemuxer` can open works, as long as it holds a video
/// stream and runs for at least a few seconds — the seek tests pace playback
/// and then reposition, so a clip shorter than that finishes before they get
/// to it. Nothing depends on a particular codec, resolution, or keyframe
/// spacing; a test that would need one must assert the contract instead (see
/// `pipeline::tests::seek_reports_where_it_actually_landed_when_target_is_not_a_keyframe`).
///
/// Tests using this must still assert real behavior when it does return a
/// path — skipping is for the machine that has no fixture, not a way to make
/// a failing assertion optional.
pub(crate) fn try_test_video() -> Option<String> {
    let Ok(path) = std::env::var("MEDIA_PP_TEST_VIDEO") else {
        eprintln!(
            "skipping: set MEDIA_PP_TEST_VIDEO to a video file to run this test \
             (no media is checked into this repository)"
        );
        return None;
    };
    if !std::path::Path::new(&path).is_file() {
        eprintln!("skipping: MEDIA_PP_TEST_VIDEO=`{path}` is not a readable file");
        return None;
    }
    eprintln!("using test video: {path}");
    Some(path)
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
