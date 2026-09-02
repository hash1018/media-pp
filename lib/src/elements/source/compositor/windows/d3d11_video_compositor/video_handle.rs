//! [`D3d11VideoLayerHandle`] — thread-safe runtime placement control for
//! one [`super::D3d11VideoCompositor`] input.

use std::sync::{Arc, Weak};

use ffmpeg_next as ffmpeg;

use super::{D3d11VideoCompositorError, GpuVideoInput, map_layer_error};
use crate::pool::UnboundObjectPoolRef;

use super::super::super::video_layer::{self, VideoLayer};

/// Thread-safe runtime placement control for one [`super::D3d11VideoCompositor`]
/// input — the GPU sibling of [`crate::elements::SwVideoLayerHandle`].
///
/// Fields are `pub(super)` rather than fully private: `D3d11VideoCompositorHandle::register_input`
/// (defined in the parent module) constructs and reads this struct
/// directly rather than through a constructor function — a private
/// struct-literal shortcut that's fine to keep since both live in the same
/// tightly-coupled `d3d11_video_compositor` module tree.
#[derive(Clone)]
pub struct D3d11VideoLayerHandle {
    pub(super) id: video_layer::VideoInputId,
    pub(super) name: Arc<str>,
    pub(super) input: Weak<GpuVideoInput>,
}

impl D3d11VideoLayerHandle {
    /// Returns the stable identity of this particular input registration.
    pub fn id(&self) -> video_layer::VideoInputId {
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
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_layer(layer).map_err(map_layer_error)?;
        self.update(|current| *current = layer)
    }

    /// Replaces the destination rectangle while retaining other settings.
    pub fn set_rect(
        &self,
        rect: video_layer::VideoRect,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_rect(rect).map_err(map_layer_error)?;
        self.update(|layer| layer.rect = rect)
    }

    /// Replaces opacity after validating the `0.0..=1.0` range.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_opacity(opacity).map_err(map_layer_error)?;
        self.update(|layer| layer.opacity = opacity)
    }

    /// Changes the stacking order; larger values are drawn later.
    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    /// Shows or hides the input without removing its registration.
    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

    /// Changes how the input aspect ratio maps into its rectangle.
    pub fn set_fit(
        &self,
        fit: video_layer::VideoFit,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.fit = fit)
    }

    /// Draws only part of the input, or all of it again with `None`.
    ///
    /// Not checked against the frame, which may not have arrived yet and may
    /// change size later — see [`video_layer::VideoSourceRect`].
    pub fn set_source(
        &self,
        source: Option<video_layer::VideoSourceRect>,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_source(source).map_err(map_layer_error)?;
        self.update(|layer| layer.source = source)
    }

    /// Pushes a new `Pixel::D3D11` frame for this input to draw next
    /// composite, without going through a `Sink` at all — for handles
    /// obtained via [`super::D3d11VideoCompositorHandle::add_layer`], which
    /// receive frames by direct call instead of `Pipeline` dataflow. Does
    /// exactly what `D3d11VideoCompositorInputSink::consume`'s
    /// `MediaBuffer::Video` arm does (format check + atomic store).
    pub fn set_frame(
        &self,
        frame: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        if frame.format() != ffmpeg::format::Pixel::D3D11 {
            return Err(D3d11VideoCompositorError::UnsupportedFormat(frame.format()));
        }
        let input = self
            .input
            .upgrade()
            .ok_or(D3d11VideoCompositorError::SourceRemoved)?;
        input.latest_frame.store(Some(frame));
        Ok(())
    }

    fn update(
        &self,
        update: impl FnOnce(&mut VideoLayer),
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        let input = self
            .input
            .upgrade()
            .ok_or(D3d11VideoCompositorError::SourceRemoved)?;
        update(&mut input.layer.lock().unwrap());
        Ok(())
    }
}
