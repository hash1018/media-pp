//! What this project's windowed examples share: turning a close of a
//! renderer's own window into a stop of the pipelines drawing into it.
//!
//! Every example draws through one of the library's window renderers, in a
//! window the renderer opens itself — `D3d11WindowRenderer` or
//! `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux — and all
//! of them report that window through the library's one `WindowEvents`, so
//! [`stop_on_close`] and [`Shutdown`] are the same on both platforms, and so
//! is [`to_drawable`], which fits a software decode to whichever renderer's
//! input: all three take system-memory YUV420P, NV12 and BGRA.

mod drawable;
mod shutdown;
mod window;

pub use drawable::to_drawable;
pub use shutdown::Shutdown;
pub use window::stop_on_close;
