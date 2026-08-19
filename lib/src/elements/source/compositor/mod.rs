//! Every element that composites multiple video inputs into one output.
//! [`video_compositor`] is backend-independent CPU code; the D3D11 GPU
//! implementation lives under [`windows`]. [`video_layer`]'s
//! `VideoRect`/`VideoLayer`/`VideoFit`/`layer_geometry` and [`text_layer`]'s
//! `TextLayer` are shared, backend-agnostic pieces the D3D11 compositor
//! builds on (colors use [`crate::color::Color`], shared crate-wide).

mod sw_video_compositor;
mod text_layer;
mod video_layer;
#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

pub use sw_video_compositor::{
    SwVideoCompositor, SwVideoCompositorError, SwVideoCompositorHandle, SwVideoCompositorInput,
    SwVideoCompositorInputSink, SwVideoLayerHandle, VideoCompositorOptions,
};
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;
pub use text_layer::TextLayer;
pub use video_layer::{VideoFit, VideoInputId, VideoLayer, VideoRect};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;
