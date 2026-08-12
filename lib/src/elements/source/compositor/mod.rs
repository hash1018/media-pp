//! Every element that composites multiple video inputs into one output —
//! [`video_compositor`] (CPU, `libswscale`) and [`d3d11_video_compositor`]
//! (GPU, D3D11), grouped here by concept rather than by backend, same
//! reasoning as [`crate::elements::source::capture`]. [`video_layer`]'s
//! `VideoRect`/`VideoLayer`/`VideoFit`/`VideoColor`/`layer_geometry` are
//! shared, backend-agnostic pieces both compositors build on.

#[cfg(feature = "d3d11-renderer")]
mod d3d11_video_compositor;
mod video_compositor;
mod video_layer;

#[cfg(feature = "d3d11-renderer")]
pub use d3d11_video_compositor::{
    D3d11VideoCompositor, D3d11VideoCompositorError, D3d11VideoCompositorHandle,
    D3d11VideoCompositorInput, D3d11VideoCompositorInputSink, D3d11VideoLayerHandle,
};
pub use video_compositor::{
    VideoCompositor, VideoCompositorError, VideoCompositorHandle, VideoCompositorInput,
    VideoCompositorInputSink, VideoCompositorOptions, VideoLayerHandle,
};
pub use video_layer::{VideoColor, VideoFit, VideoInputId, VideoLayer, VideoRect};
