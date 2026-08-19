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
    pub fn id(&self) -> video_layer::VideoInputId {
        self.id
    }

    pub fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    pub fn layer(&self) -> Option<VideoLayer> {
        self.input
            .upgrade()
            .map(|input| *input.layer.lock().unwrap())
    }

    pub fn set_layer(
        &self,
        layer: VideoLayer,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_layer(layer).map_err(map_layer_error)?;
        self.update(|current| *current = layer)
    }

    pub fn set_rect(
        &self,
        rect: video_layer::VideoRect,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_rect(rect).map_err(map_layer_error)?;
        self.update(|layer| layer.rect = rect)
    }

    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), D3d11VideoCompositorError> {
        video_layer::validate_opacity(opacity).map_err(map_layer_error)?;
        self.update(|layer| layer.opacity = opacity)
    }

    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.z_index = z_index)
    }

    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.visible = visible)
    }

    pub fn set_fit(
        &self,
        fit: video_layer::VideoFit,
    ) -> std::result::Result<(), D3d11VideoCompositorError> {
        self.update(|layer| layer.fit = fit)
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
