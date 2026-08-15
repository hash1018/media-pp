//! Every element that composites multiple video inputs into one output.
//! [`video_compositor`] is backend-independent CPU code; the D3D11 GPU
//! implementation lives under [`windows`]. [`video_layer`]'s
//! `VideoRect`/`VideoLayer`/`VideoFit`/`layer_geometry` and [`text_layer`]'s
//! `TextLayer` are shared, backend-agnostic pieces the D3D11 compositor
//! builds on (colors use [`crate::color::Color`], shared crate-wide).

mod text_layer;
mod video_compositor;
mod video_layer;
#[cfg(all(target_os = "windows", feature = "d3d11-renderer"))]
mod windows;

pub use text_layer::TextLayer;
pub use video_compositor::{
    VideoCompositor, VideoCompositorError, VideoCompositorHandle, VideoCompositorInput,
    VideoCompositorInputSink, VideoCompositorOptions, VideoLayerHandle,
};
pub use video_layer::{VideoFit, VideoInputId, VideoLayer, VideoRect};
#[cfg(all(target_os = "windows", feature = "d3d11-renderer"))]
pub use windows::*;
