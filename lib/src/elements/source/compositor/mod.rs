//! The video compositors, one per backend, and the layer types they share.
//!
//! `sw_video_compositor` is backend-independent CPU code; the GPU
//! implementations live under `windows` (D3D11, D3D12), `cuda`, `vulkan` and
//! `macos` (Metal). `video_layer`'s
//! `VideoRect`/`VideoLayer`/`VideoFit`/`layer_geometry` and `text_layer`'s
//! `TextLayer` are shared, backend-agnostic pieces every compositor
//! builds on (colors use [`crate::color::Color`], shared crate-wide), and
//! `control`'s traits are the one API they all answer to.

mod control;
mod sw_text_layer;
mod sw_video_compositor;
mod text_layer;
mod timed_inputs;
mod video_layer;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
mod windows;

pub use control::{
    CompositorInput, LayerFrame, TextLayerControl, VideoCompositorControl, VideoLayerControl,
};
pub use sw_text_layer::{SwTextLayerError, SwTextLayerHandle};
pub use sw_video_compositor::{
    SwVideoCompositor, SwVideoCompositorError, SwVideoCompositorHandle, SwVideoCompositorInput,
    SwVideoCompositorInputSink, SwVideoLayerHandle, VideoCompositorOptions,
};
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod macos;
#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use macos::*;
pub use text_layer::TextLayer;
// What the detection overlays draw their labels with, which rasterize as a
// compositor's text layers do.
pub(in crate::elements) use text_layer::{
    TextFontError, TextMask, TextRasterError, load_font, rasterize_coverage,
};
pub use video_layer::{VideoFit, VideoInputId, VideoLayer, VideoRect, VideoSourceRect};
#[cfg(feature = "vulkan")]
pub use vulkan::*;
#[cfg(all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")))]
pub use windows::*;
