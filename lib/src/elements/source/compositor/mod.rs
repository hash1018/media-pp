//! Every element that composites multiple video inputs into one output —
//! [`video_compositor`] (CPU, `libswscale`) and [`d3d11_video_compositor`]
//! (GPU, D3D11), grouped here by concept rather than by backend, same
//! reasoning as [`crate::elements::source::capture`]. [`video_layer`]'s
//! `VideoRect`/`VideoLayer`/`VideoFit`/`layer_geometry` and [`text_layer`]'s
//! `TextLayer` are shared, backend-agnostic pieces the D3D11 compositor
//! builds on (colors use [`crate::color::Color`], shared crate-wide).

#[cfg(feature = "d3d11-renderer")]
mod d3d11_video_compositor;
mod text_layer;
mod video_compositor;
mod video_layer;

#[cfg(feature = "d3d11-renderer")]
pub use d3d11_video_compositor::{
    D3d11TextLayerError, D3d11TextLayerHandle, D3d11VideoCompositor, D3d11VideoCompositorError,
    D3d11VideoCompositorHandle, D3d11VideoCompositorInput, D3d11VideoCompositorInputSink,
    D3d11VideoLayerHandle,
};
pub use text_layer::TextLayer;
pub use video_compositor::{
    VideoCompositor, VideoCompositorError, VideoCompositorHandle, VideoCompositorInput,
    VideoCompositorInputSink, VideoCompositorOptions, VideoLayerHandle,
};
pub use video_layer::{VideoFit, VideoInputId, VideoLayer, VideoRect};
