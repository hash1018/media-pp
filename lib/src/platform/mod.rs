#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(any(
    feature = "cuda",
    feature = "d3d11",
    feature = "d3d12",
    feature = "vulkan",
    all(target_os = "macos", feature = "videotoolbox")
))]
pub(crate) mod ffmpeg;
#[cfg(target_os = "linux")]
pub(crate) mod linux;
#[cfg(target_os = "macos")]
pub(crate) mod macos;
#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;
#[cfg(target_os = "windows")]
pub(crate) mod windows;
