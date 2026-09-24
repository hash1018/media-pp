//! [`SwTextLayerHandle`] — dynamic text drawn into a
//! [`SwVideoCompositor`](super::SwVideoCompositor) scene, the software
//! sibling of the D3D11 and CUDA compositors' text layers.

use std::sync::Arc;

use ab_glyph::{FontArc, InvalidFont};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

use super::{
    sw_video_compositor::{SwVideoCompositorError, SwVideoCompositorHandle, SwVideoLayerHandle},
    text_layer::{TextLayer, TextRasterError, rasterize_bgra},
    video_layer::{VideoLayer, VideoRect},
};
use crate::{color::Color, pool::UnboundObjectPool};

/// Why a [`SwTextLayerHandle`] could not be made or could not draw.
#[derive(Debug, ThisError)]
pub enum SwTextLayerError {
    /// The supplied bytes are not a supported TrueType or OpenType font.
    #[error("invalid font data: {0}")]
    InvalidFont(#[from] InvalidFont),

    /// The glyph pixel height is non-positive or non-finite.
    #[error("font size must be finite and greater than zero, got {0}")]
    InvalidFontSize(f32),

    /// Rasterizing the text would exceed the dimensions a layer can have.
    #[error("rasterized text is too large: {width}x{height}")]
    TextTooLarge {
        /// Computed raster width in pixels.
        width: u64,
        /// Computed raster height in pixels.
        height: u64,
    },

    /// Memory for the rasterized picture could not be reserved.
    #[error("could not allocate {bytes} bytes for rasterized text")]
    AllocationFailed {
        /// Number of bytes requested.
        bytes: usize,
    },

    /// The compositor, or this layer's registration in it, is gone.
    #[error(transparent)]
    Compositor(#[from] SwVideoCompositorError),
}

impl From<TextRasterError> for SwTextLayerError {
    fn from(error: TextRasterError) -> Self {
        match error {
            TextRasterError::TooLarge { width, height } => Self::TextTooLarge { width, height },
            TextRasterError::AllocationFailed { bytes } => Self::AllocationFailed { bytes },
        }
    }
}

/// Dynamic text drawn into a [`SwVideoCompositor`](super::SwVideoCompositor)
/// scene, from [`SwVideoCompositorHandle::add_text_layer`]. No branch feeds
/// it: [`Self::set_text`] rasterizes the string on the calling thread into a
/// straight-alpha BGRA picture exactly the text's size, and the compositor
/// blends it like any other layer from its next tick on.
///
/// Stacked like a video input — see [`TextLayer`] — and placed by its
/// top-left corner; its size is always the text's own, set by each
/// `set_text`. Not cheap to clone, and not `Clone`: it owns its parsed font.
/// It keeps nothing of the compositor alive; once the compositor or this
/// registration is gone, every call returns
/// [`SwVideoCompositorError::SourceRemoved`] inside
/// [`SwTextLayerError::Compositor`].
pub struct SwTextLayerHandle {
    layer: SwVideoLayerHandle,
    font: FontArc,
    font_size: f32,
    color: Color,
    /// Wrappers for the rasterized pictures, reused as the compositor lets
    /// go of each.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl SwTextLayerHandle {
    /// Rasterizes `text` and makes it this layer's picture, sizing the layer
    /// to it and keeping its top-left corner. Empty, whitespace-only or
    /// control-only text hides the layer rather than drawing nothing.
    pub fn set_text(&self, text: &str) -> Result<(), SwTextLayerError> {
        let current = self
            .layer
            .layer()
            .ok_or(SwVideoCompositorError::SourceRemoved)?;
        let Some((width, height, pixels)) =
            rasterize_bgra(&self.font, self.font_size, text, self.color)?
        else {
            self.layer.set_visible(false)?;
            return Ok(());
        };
        let mut frame = self.pool.get();
        *frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
        let stride = frame.stride(0);
        let row_bytes = width as usize * 4;
        let rows = frame.data_mut(0);
        for (row, source) in pixels.chunks_exact(row_bytes).enumerate() {
            rows[row * stride..row * stride + row_bytes].copy_from_slice(source);
        }
        self.layer.set_frame(Arc::new(frame))?;
        self.layer.set_rect(VideoRect::new(
            current.rect.x,
            current.rect.y,
            width,
            height,
        ))?;
        self.layer.set_visible(true)?;
        Ok(())
    }

    /// Moves the text's top-left corner, keeping its size.
    pub fn set_position(&self, x: i32, y: i32) -> Result<(), SwTextLayerError> {
        let current = self
            .layer
            .layer()
            .ok_or(SwVideoCompositorError::SourceRemoved)?;
        self.layer.set_rect(VideoRect::new(
            x,
            y,
            current.rect.width,
            current.rect.height,
        ))?;
        Ok(())
    }

    /// Changes the text's opacity, `0.0..=1.0`.
    pub fn set_opacity(&self, opacity: f32) -> Result<(), SwTextLayerError> {
        self.layer.set_opacity(opacity)?;
        Ok(())
    }

    /// Changes the stacking order; larger values are drawn later.
    pub fn set_z_index(&self, z_index: i32) -> Result<(), SwTextLayerError> {
        self.layer.set_z_index(z_index)?;
        Ok(())
    }

    /// Shows or hides the text, keeping what it says.
    pub fn set_visible(&self, visible: bool) -> Result<(), SwTextLayerError> {
        self.layer.set_visible(visible)?;
        Ok(())
    }
}

impl SwVideoCompositorHandle {
    /// Registers a text layer and returns the handle that sets what it says
    /// — see [`SwTextLayerHandle`]. It shows nothing until the first
    /// [`SwTextLayerHandle::set_text`].
    ///
    /// The font and size are checked before anything is registered, so a
    /// bad font leaves an existing layer of the same name in place. Reusing
    /// `name` otherwise replaces that registration, as
    /// [`Self::add_source`] does.
    pub fn add_text_layer(
        &self,
        name: impl Into<String>,
        text_layer: TextLayer,
    ) -> Result<SwTextLayerHandle, SwTextLayerError> {
        if !text_layer.font_size.is_finite() || text_layer.font_size <= 0.0 {
            return Err(SwTextLayerError::InvalidFontSize(text_layer.font_size));
        }
        let font = FontArc::try_from_vec(text_layer.font_data)?;
        // No text yet, so no size: a hidden placeholder until the first
        // `set_text` gives it one.
        let mut placeholder = VideoLayer::new(VideoRect::new(text_layer.x, text_layer.y, 1, 1));
        placeholder.visible = false;
        let layer = self.register_input(name, placeholder)?;
        Ok(SwTextLayerHandle {
            layer,
            font,
            font_size: text_layer.font_size,
            color: text_layer.color,
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })
    }
}
