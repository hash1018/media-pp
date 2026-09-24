//! What this project's windowed examples share: turning a close of a
//! renderer's own window into a stop of the pipelines drawing into it.
//!
//! Every example draws through one of the library's window renderers, in a
//! window the renderer opens itself — `D3d11WindowRenderer` or
//! `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux — and all
//! of them report that window through the library's one `WindowEvents`, so
//! [`stop_on_close`] and [`Shutdown`] are the same on both platforms. On Linux
//! there is one thing more, `to_drawable`, which fits a software decode to
//! `VulkanWindowRenderer`'s input; it is named rather than linked, since on
//! any other host the link would be a broken one.

mod shutdown;
mod window;

pub use shutdown::Shutdown;
pub use window::stop_on_close;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::to_drawable;
