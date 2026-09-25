//! What this project's windowed examples share: turning a close of a
//! renderer's own window into a stop of the pipelines drawing into it.
//!
//! Every example draws through one of the library's window renderers, in a
//! window the renderer opens itself — `D3d11WindowRenderer` or
//! `D3d12WindowRenderer` on Windows, `VulkanWindowRenderer` on Linux — and all
//! of them report that window through the library's one `WindowEvents`, so
//! [`stop_on_close`] and [`Shutdown`] are the same on both platforms. What
//! fits a software decode to a renderer's input is the library's own,
//! `SwScaler::if_needed`.

mod shutdown;
mod window;

pub use shutdown::Shutdown;
pub use window::stop_on_close;
