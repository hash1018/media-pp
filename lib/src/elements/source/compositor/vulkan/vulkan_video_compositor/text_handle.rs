//! [`VulkanTextLayerHandle`]: text drawn into a
//! [`super::VulkanVideoCompositor`] scene, rasterized on the CPU into a
//! coverage mask on every [`VulkanTextLayerHandle::set_text`].

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering},
};

use arc_swap::ArcSwapOption;

use super::super::super::text_layer::{
    TextFontError, TextLayer, TextMask, TextRasterError, load_font, rasterize_coverage,
};
use super::{VulkanVideoCompositorError, VulkanVideoCompositorHandle, validate_opacity};
use crate::color::Color;

/// One registered text layer's live state, shared between its handle and the
/// compositor's own thread.
pub(super) struct TextLayerState {
    pub(super) font: ab_glyph::FontArc,
    pub(super) font_size: f32,
    pub(super) color: Color,
    /// Replaced wholesale by `set_text`, which is what makes a text change
    /// atomic from the compositor's point of view: it either draws the whole
    /// previous mask or the whole new one. Kept in host memory: the
    /// compositor copies a mask it has not drawn before onto the GPU in its
    /// own recording, so this handle never submits work of its own.
    pub(super) mask: ArcSwapOption<TextMask>,
    pub(super) x: AtomicI32,
    pub(super) y: AtomicI32,
    /// `f32` bits — the compositor only ever reads it, and a torn read is
    /// impossible for a 32-bit atomic.
    pub(super) opacity: AtomicU32,
    pub(super) visible: AtomicBool,
    /// Where it stacks among the video layers and the other text layers:
    /// drawn by it, a video layer first where the two are equal.
    pub(super) z_index: AtomicI32,
}

/// Runtime control for one text layer — the Vulkan sibling of
/// `CudaTextLayerHandle`.
///
/// Cloning is cheap and every clone controls the same layer. `set_text`
/// rasterizes the string on the CPU, so it is not something to call per
/// frame if the text has not actually changed; the compositor copies the
/// new mask to the GPU once, the next time it draws.
#[derive(Clone)]
pub struct VulkanTextLayerHandle {
    state: Arc<TextLayerState>,
}

impl VulkanTextLayerHandle {
    /// Rasterizes `text`, replacing whatever was drawn before. Text with no
    /// drawable glyphs (empty, whitespace, control characters) clears the
    /// layer.
    pub fn set_text(&self, text: &str) -> std::result::Result<(), VulkanVideoCompositorError> {
        let rasterized = rasterize_coverage(&self.state.font, self.state.font_size, text)
            .map_err(text_raster_error)?;
        self.state.mask.store(rasterized.map(Arc::new));
        Ok(())
    }

    /// Moves the layer's top-left corner, in canvas pixels.
    pub fn set_position(&self, x: i32, y: i32) {
        self.state.x.store(x, Ordering::Relaxed);
        self.state.y.store(y, Ordering::Relaxed);
    }

    /// Blends the text at `opacity`, from 0 to 1.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), VulkanVideoCompositorError> {
        validate_opacity(opacity)?;
        self.state
            .opacity
            .store(opacity.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    /// Shows or hides the text without discarding its mask.
    pub fn set_visible(&self, visible: bool) {
        self.state.visible.store(visible, Ordering::Relaxed);
    }

    /// Where it stacks: among video layers and other text layers by
    /// `z_index`, a video layer first where the two are equal. Zero to
    /// begin with, which is over every video layer at zero or below.
    pub fn set_z_index(&self, z_index: i32) {
        self.state.z_index.store(z_index, Ordering::Relaxed);
    }
}

impl VulkanVideoCompositorHandle {
    /// Registers a text layer and returns its control handle. Reusing `name`
    /// replaces the previous registration.
    ///
    /// The font is parsed here so bad font data fails at registration rather
    /// than at the first `set_text`. Nothing is drawn until `set_text` is
    /// called — a text layer with no text is not an error, just empty.
    pub fn add_text_layer(
        &self,
        name: impl Into<String>,
        text_layer: TextLayer,
    ) -> std::result::Result<VulkanTextLayerHandle, VulkanVideoCompositorError> {
        let font =
            load_font(text_layer.font_data, text_layer.font_size).map_err(|error| match error {
                TextFontError::Size(size) => VulkanVideoCompositorError::InvalidFontSize(size),
                TextFontError::Font(error) => {
                    VulkanVideoCompositorError::InvalidFont(error.to_string())
                }
            })?;
        let Some(shared) = self.shared.upgrade() else {
            return Err(VulkanVideoCompositorError::Stopped);
        };
        let state = Arc::new(TextLayerState {
            font,
            font_size: text_layer.font_size,
            color: text_layer.color,
            mask: ArcSwapOption::empty(),
            x: AtomicI32::new(text_layer.x),
            y: AtomicI32::new(text_layer.y),
            opacity: AtomicU32::new(1.0f32.to_bits()),
            visible: AtomicBool::new(true),
            z_index: AtomicI32::new(0),
        });
        // Reusing a name replaces that registration, the same contract
        // `add_source` has; the replaced handle then controls a layer nothing
        // draws any more.
        let name: Arc<str> = name.into().into();
        let mut layers = shared.text_layers.lock().unwrap();
        match layers.iter_mut().find(|(existing, _)| *existing == name) {
            Some(slot) => slot.1 = state.clone(),
            None => layers.push((name, state.clone())),
        }
        drop(layers);
        Ok(VulkanTextLayerHandle { state })
    }
}

fn text_raster_error(error: TextRasterError) -> VulkanVideoCompositorError {
    match error {
        TextRasterError::TooLarge { width, height } => {
            VulkanVideoCompositorError::TextTooLarge { width, height }
        }
        TextRasterError::AllocationFailed { bytes } => {
            VulkanVideoCompositorError::AllocationFailed { bytes }
        }
    }
}

impl crate::elements::TextLayerControl for VulkanTextLayerHandle {
    fn set_text(&self, text: &str) -> std::result::Result<(), crate::error::Error> {
        Ok(Self::set_text(self, text)?)
    }

    fn set_position(&self, x: i32, y: i32) -> std::result::Result<(), crate::error::Error> {
        Self::set_position(self, x, y);
        Ok(())
    }

    fn set_opacity(&self, opacity: f32) -> std::result::Result<(), crate::error::Error> {
        Ok(Self::set_opacity(self, opacity)?)
    }

    fn set_z_index(&self, z_index: i32) -> std::result::Result<(), crate::error::Error> {
        Self::set_z_index(self, z_index);
        Ok(())
    }

    fn set_visible(&self, visible: bool) -> std::result::Result<(), crate::error::Error> {
        Self::set_visible(self, visible);
        Ok(())
    }
}
