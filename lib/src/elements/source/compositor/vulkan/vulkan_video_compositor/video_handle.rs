//! [`VulkanVideoLayerHandle`]: thread-safe runtime placement control for
//! one [`super::VulkanVideoCompositor`] input.

use std::sync::{Arc, Weak};

use ffmpeg_next as ffmpeg;

use crate::pool::UnboundObjectPoolRef;

use super::super::super::video_layer::{
    self, VideoFit, VideoInputId, VideoLayer, VideoRect, VideoSourceRect,
};
use super::{
    CompositorShared, VideoInput, VulkanVideoCompositorError, layer_error, validate_input_frame,
    validate_layer, validate_opacity, validate_rect,
};

/// Runtime placement control for one registered input — the Vulkan sibling of
/// [`crate::elements::SwVideoLayerHandle`], with the same API.
#[derive(Clone)]
pub struct VulkanVideoLayerHandle {
    pub(super) id: VideoInputId,
    pub(super) name: Arc<str>,
    pub(super) input: Weak<VideoInput>,
    /// The compositor, for the device a frame set here must come from.
    pub(super) shared: Weak<CompositorShared>,
}

impl VulkanVideoLayerHandle {
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

    /// The `Pixel::VULKAN` frame this input will be drawn from next, or
    /// `None` before its first and once the input is removed — see
    /// [`crate::elements::SwVideoLayerHandle::latest_frame`], whose contract
    /// this shares, pool slot included.
    ///
    /// Its image is whichever of the two layouts this compositor draws —
    /// NV12 or BGRA — as the input was handed it.
    pub fn latest_frame(&self) -> Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>> {
        self.input.upgrade()?.latest_frame.load_full()
    }

    /// Makes `frame` this input's picture, drawn from the next frame composed
    /// until another replaces it — for a layer from
    /// [`super::VulkanVideoCompositorHandle::add_layer`]. A layer fed through a
    /// sink has its picture replaced by the next frame that arrives.
    ///
    /// The frame has to be on the compositor's own device, as one through a
    /// sink does: one in system memory, or from another Vulkan device, is
    /// refused here rather than when it is drawn. Returns
    /// [`VulkanVideoCompositorError::SourceRemoved`] if this handle is stale.
    pub fn set_frame(
        &self,
        frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        let shared = self
            .shared
            .upgrade()
            .ok_or(VulkanVideoCompositorError::Stopped)?;
        validate_input_frame(&frame, shared.device_ctx)?;
        let input = self
            .input
            .upgrade()
            .ok_or(VulkanVideoCompositorError::SourceRemoved)?;
        input.latest_frame.store(Some(frame));
        Ok(())
    }

    /// Atomically replaces every layer setting.
    pub fn set_layer(
        &self,
        layer: VideoLayer,
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        validate_layer(layer)?;
        self.update(|current| *current = layer)
    }

    /// Replaces the destination rectangle while retaining other settings.
    pub fn set_rect(&self, rect: VideoRect) -> std::result::Result<(), VulkanVideoCompositorError> {
        validate_rect(rect)?;
        self.update(|layer| layer.rect = rect)
    }

    /// Replaces opacity after validating the `0.0..=1.0` range.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), VulkanVideoCompositorError> {
        validate_opacity(opacity)?;
        self.update(|layer| layer.opacity = opacity)
    }

    /// Changes the stacking order; larger values are drawn later.
    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), VulkanVideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    /// Shows or hides the input without removing its registration.
    pub fn set_visible(
        &self,
        visible: bool,
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

    /// Changes how the input aspect ratio maps into its rectangle.
    pub fn set_fit(&self, fit: VideoFit) -> std::result::Result<(), VulkanVideoCompositorError> {
        self.update(|layer| layer.fit = fit)
    }

    /// Draws only part of the input, or all of it again with `None`.
    ///
    /// Not checked against the frame, which may not have arrived yet and may
    /// change size later — see [`VideoSourceRect`].
    pub fn set_source(
        &self,
        source: Option<VideoSourceRect>,
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        video_layer::validate_source(source).map_err(layer_error)?;
        self.update(|layer| layer.source = source)
    }

    fn update(
        &self,
        change: impl FnOnce(&mut VideoLayer),
    ) -> std::result::Result<(), VulkanVideoCompositorError> {
        let Some(input) = self.input.upgrade() else {
            return Err(VulkanVideoCompositorError::SourceRemoved);
        };
        change(&mut input.layer.lock().unwrap());
        Ok(())
    }
}
