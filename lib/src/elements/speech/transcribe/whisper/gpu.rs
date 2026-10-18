//! Which GPU a transcriber's model runs on.
//!
//! whisper.cpp takes a GPU as a number: the n-th GPU of every backend it
//! was built with, counted in the order they register. Built with both CUDA
//! and Vulkan, one NVIDIA card is two of them — CUDA0 and Vulkan0 — so a
//! number alone cannot say which backend runs. Each element asks for the
//! n-th GPU of its own backend, and this turns that into whisper.cpp's
//! number.

use std::ffi::CStr;

use whisper_rs::whisper_rs_sys as sys;

use super::WhisperTranscriberError;

/// whisper.cpp's number for the `index`-th GPU its `backend` offers, by the
/// name that backend registers under — `CUDA`, `Vulkan`, `Metal`.
///
/// # Errors
///
/// [`WhisperTranscriberError::NoGpu`] when the backend offers fewer GPUs:
/// whisper.cpp itself would run on the CPU instead, without a word, which
/// is the one thing a caller who named a GPU did not ask for.
pub(super) fn device(backend: &'static str, index: usize) -> Result<i32, WhisperTranscriberError> {
    let mut gpus = 0;
    let mut found = 0;
    // SAFETY: the registry is ggml's own, built from the backends compiled
    // in on first use and never torn down; each device it hands out, and
    // the names it answers, live as long as the process.
    unsafe {
        for at in 0..sys::ggml_backend_dev_count() {
            let device = sys::ggml_backend_dev_get(at);
            let kind = sys::ggml_backend_dev_type(device);
            if kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                && kind != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                continue;
            }
            let registry = sys::ggml_backend_dev_backend_reg(device);
            let name = CStr::from_ptr(sys::ggml_backend_reg_name(registry));
            if name.to_bytes().eq_ignore_ascii_case(backend.as_bytes()) {
                if found == index {
                    return Ok(gpus);
                }
                found += 1;
            }
            gpus += 1;
        }
    }
    Err(WhisperTranscriberError::NoGpu {
        backend,
        index,
        found,
    })
}
