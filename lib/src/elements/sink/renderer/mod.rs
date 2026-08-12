//! Every terminal element that submits decoded/uploaded frames to an
//! actual GPU rendering implementation — [`d3d11_renderer`]/
//! [`d3d12_renderer`], grouped here by concept rather than by GPU API,
//! same reasoning as [`crate::elements::filter::decoder`].
//! [`submit_error`]'s `SubmitError` is shared by both (and is
//! `media-pp`'s own type — always available regardless of which, if
//! either, renderer feature is on).

#[cfg(feature = "d3d11-renderer")]
mod d3d11_renderer;
#[cfg(feature = "d3d12-renderer")]
mod d3d12_renderer;
mod submit_error;
#[cfg(feature = "wasapi-renderer")]
mod wasapi_renderer;

#[cfg(feature = "d3d11-renderer")]
pub use d3d11_renderer::{D3d11FrameRenderer, D3d11Renderer, D3d11RendererError};
#[cfg(feature = "d3d12-renderer")]
pub use d3d12_renderer::{D3d12FrameRenderer, D3d12Renderer, D3d12RendererError, RawPlane};
pub use submit_error::SubmitError;
#[cfg(feature = "wasapi-renderer")]
pub use wasapi_renderer::{WasapiRenderer, WasapiRendererError, WasapiRendererOptions};
