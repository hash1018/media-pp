#[cfg(feature = "d3d11")]
mod d3d11_video_compositor;

#[cfg(feature = "d3d11")]
pub use d3d11_video_compositor::{
    D3d11TextLayerError, D3d11TextLayerHandle, D3d11VideoCompositor, D3d11VideoCompositorError,
    D3d11VideoCompositorHandle, D3d11VideoCompositorInput, D3d11VideoCompositorInputSink,
    D3d11VideoLayerHandle,
};
