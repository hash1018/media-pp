#[cfg(feature = "coreaudio-renderer")]
mod coreaudio_renderer;
#[cfg(feature = "metal")]
mod metal_renderer;
#[cfg(feature = "metal")]
mod metal_window_renderer;
#[cfg(feature = "coreaudio-renderer")]
mod sample_ring;

#[cfg(feature = "coreaudio-renderer")]
pub use coreaudio_renderer::{CoreAudioRenderer, CoreAudioRendererError, CoreAudioRendererOptions};
#[cfg(feature = "metal")]
pub use metal_renderer::{
    MetalFrame, MetalFramePlanes, MetalFrameRenderer, MetalRenderer, MetalRendererError,
    MetalTexture,
};
#[cfg(feature = "metal")]
pub use metal_window_renderer::{MetalWindowRenderer, MetalWindowRendererError};
