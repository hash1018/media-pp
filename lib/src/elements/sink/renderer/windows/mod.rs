#[cfg(feature = "d3d11")]
mod d3d11_renderer;
#[cfg(feature = "d3d11")]
mod d3d11_window_renderer;
#[cfg(feature = "d3d12")]
mod d3d12_renderer;
#[cfg(feature = "d3d12")]
mod d3d12_window_renderer;
#[cfg(any(feature = "d3d11", feature = "d3d12"))]
mod present_timing;
#[cfg(any(feature = "d3d11", feature = "d3d12"))]
mod system_frame;
#[cfg(feature = "wasapi-renderer")]
mod wasapi_renderer;

#[cfg(feature = "d3d11")]
pub use d3d11_renderer::{D3d11FrameRenderer, D3d11Renderer, D3d11RendererError};
#[cfg(feature = "d3d11")]
pub use d3d11_window_renderer::{D3d11WindowRenderer, D3d11WindowRendererError};
#[cfg(feature = "d3d12")]
pub use d3d12_renderer::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError};
#[cfg(feature = "d3d12")]
pub use d3d12_window_renderer::{D3d12WindowRenderer, D3d12WindowRendererError};
#[cfg(feature = "wasapi-renderer")]
pub use wasapi_renderer::{WasapiRenderer, WasapiRendererError, WasapiRendererOptions};
