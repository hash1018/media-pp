#[cfg(all(
    target_os = "windows",
    any(feature = "dxgi-capture", feature = "wasapi-capture")
))]
mod windows;

#[cfg(all(
    target_os = "windows",
    any(feature = "dxgi-capture", feature = "wasapi-capture")
))]
pub use windows::*;
