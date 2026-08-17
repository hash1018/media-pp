#[cfg(all(
    target_os = "linux",
    any(
        feature = "pipewire-screen-capture",
        feature = "pipewire-audio-capture"
    )
))]
mod linux;
#[cfg(all(
    target_os = "windows",
    any(feature = "dxgi-capture", feature = "wasapi-capture")
))]
mod windows;

#[cfg(all(
    target_os = "linux",
    any(
        feature = "pipewire-screen-capture",
        feature = "pipewire-audio-capture"
    )
))]
pub use linux::*;
#[cfg(all(
    target_os = "windows",
    any(feature = "dxgi-capture", feature = "wasapi-capture")
))]
pub use windows::*;
