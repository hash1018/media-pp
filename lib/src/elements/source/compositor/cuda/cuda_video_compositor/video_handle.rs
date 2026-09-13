//! [`CudaVideoLayerHandle`] ??thread-safe runtime placement control for
//! one [`super::CudaVideoCompositor`] input.

use std::sync::{Arc, Weak};

use super::super::super::video_layer::{
    self, VideoFit, VideoInputId, VideoLayer, VideoRect, VideoSourceRect,
};
use super::{
    CudaVideoCompositorError, VideoInput, layer_error, validate_layer, validate_opacity,
    validate_rect,
};

/// Runtime placement control for one registered input — the CUDA sibling of
/// [`crate::elements::SwVideoLayerHandle`], with the same API.
#[derive(Clone)]
pub struct CudaVideoLayerHandle {
    pub(super) id: VideoInputId,
    pub(super) name: Arc<str>,
    pub(super) input: Weak<VideoInput>,
}

impl CudaVideoLayerHandle {
    /// Returns the stable identity of this particular input registration.
    pub fn id(&self) -> VideoInputId {
        self.id
    }

    /// Returns the registration name, which may be reused by a newer input.
    pub fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    /// Returns the current settings, or `None` after the input is removed.
    pub fn layer(&self) -> Option<VideoLayer> {
        self.input
            .upgrade()
            .map(|input| *input.layer.lock().unwrap())
    }

    /// Atomically replaces every layer setting.
    pub fn set_layer(
        &self,
        layer: VideoLayer,
    ) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_layer(layer)?;
        self.update(|current| *current = layer)
    }

    /// Replaces the destination rectangle while retaining other settings.
    pub fn set_rect(&self, rect: VideoRect) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_rect(rect)?;
        self.update(|layer| layer.rect = rect)
    }

    /// Replaces opacity after validating the `0.0..=1.0` range.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_opacity(opacity)?;
        self.update(|layer| layer.opacity = opacity)
    }

    /// Changes the stacking order; larger values are drawn later.
    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    /// Shows or hides the input without removing its registration.
    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

    /// Changes how the input aspect ratio maps into its rectangle.
    pub fn set_fit(&self, fit: VideoFit) -> std::result::Result<(), CudaVideoCompositorError> {
        self.update(|layer| layer.fit = fit)
    }

    /// Draws only part of the input, or all of it again with `None`.
    ///
    /// Not checked against the frame, which may not have arrived yet and may
    /// change size later — see [`VideoSourceRect`]. On NV12 surfaces the
    /// region is aligned inwards to even pixels when it is drawn, since a
    /// chroma sample covers two.
    pub fn set_source(
        &self,
        source: Option<VideoSourceRect>,
    ) -> std::result::Result<(), CudaVideoCompositorError> {
        video_layer::validate_source(source).map_err(layer_error)?;
        self.update(|layer| layer.source = source)
    }

    fn update(
        &self,
        change: impl FnOnce(&mut VideoLayer),
    ) -> std::result::Result<(), CudaVideoCompositorError> {
        let Some(input) = self.input.upgrade() else {
            return Err(CudaVideoCompositorError::SourceRemoved);
        };
        change(&mut input.layer.lock().unwrap());
        Ok(())
    }
}
