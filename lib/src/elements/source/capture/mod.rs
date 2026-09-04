#[cfg(all(
    target_os = "linux",
    any(
        feature = "pipewire-screen-capture",
        feature = "pipewire-audio-capture",
        feature = "v4l2-capture"
    )
))]
mod linux;
#[cfg(all(
    target_os = "windows",
    any(
        feature = "dxgi-capture",
        feature = "wgc-capture",
        feature = "wasapi-capture",
        feature = "mf-capture"
    )
))]
mod windows;

#[cfg(all(
    target_os = "linux",
    any(
        feature = "pipewire-screen-capture",
        feature = "pipewire-audio-capture",
        feature = "v4l2-capture"
    )
))]
pub use linux::*;
#[cfg(all(
    target_os = "windows",
    any(
        feature = "dxgi-capture",
        feature = "wgc-capture",
        feature = "wasapi-capture",
        feature = "mf-capture"
    )
))]
pub use windows::*;
