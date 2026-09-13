//! [`CudaTextLayerHandle`] ??text drawn into a [`super::CudaVideoCompositor`]
//! scene, rasterized on the CPU and uploaded as a coverage mask on every
//! [`CudaTextLayerHandle::set_text`].

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering},
};

use arc_swap::ArcSwapOption;

use super::super::super::text_layer::{TextLayer, TextRasterError, rasterize_coverage};
use super::{CudaVideoCompositorError, CudaVideoCompositorHandle, validate_opacity};
use crate::{
    color::Color,
    platform::cuda::driver::{CudaDriver, CudaMask},
};

/// One registered text layer's live state, shared between its handle and the
/// compositor's own thread.
pub(super) struct TextLayerState {
    pub(super) font: ab_glyph::FontArc,
    pub(super) font_size: f32,
    pub(super) color: Color,
    /// Replaced wholesale by `set_text`, which is what makes a text change
    /// atomic from the compositor's point of view: it either draws the whole
    /// previous mask or the whole new one.
    pub(super) mask: ArcSwapOption<CudaMask>,
    pub(super) x: AtomicI32,
    pub(super) y: AtomicI32,
    /// `f32` bits — the compositor only ever reads it, and a torn read is
    /// impossible for a 32-bit atomic.
    pub(super) opacity: AtomicU32,
    pub(super) visible: AtomicBool,
}

/// Runtime control for one text layer — the CUDA sibling of
/// `D3d11TextLayerHandle`.
///
/// Cloning is cheap and every clone controls the same layer. Unlike a
/// [`CudaVideoLayerHandle`](super::CudaVideoLayerHandle), this one does real work on `set_text`: it
/// rasterizes the string on the CPU and uploads the resulting coverage mask
/// to the GPU, so it is not something to call per frame if the text has not
/// actually changed.
#[derive(Clone)]
pub struct CudaTextLayerHandle {
    state: Arc<TextLayerState>,
    driver: Arc<CudaDriver>,
}

impl CudaTextLayerHandle {
    /// Rasterizes `text` and uploads it, replacing whatever was drawn
    /// before. Text with no drawable glyphs (empty, whitespace, control
    /// characters) clears the layer.
    pub fn set_text(&self, text: &str) -> std::result::Result<(), CudaVideoCompositorError> {
        let rasterized = rasterize_coverage(&self.state.font, self.state.font_size, text)
            .map_err(text_raster_error)?;
        let Some(mask) = rasterized else {
            self.state.mask.store(None);
            return Ok(());
        };
        // Uploaded before it is published, so the compositor never sees a
        // half-written mask — the same reason `add_source` validates before
        // it replaces a registration.
        let uploaded = self
            .driver
            .upload_mask(&mask.coverage, mask.width, mask.height)?;
        self.state.mask.store(Some(Arc::new(uploaded)));
        Ok(())
    }

    /// Moves the layer's top-left corner. Coordinates are aligned to even
    /// pixels when drawn, for the chroma reason
    /// [`CudaVideoCompositor`](super::CudaVideoCompositor) documents.
    pub fn set_position(&self, x: i32, y: i32) {
        self.state.x.store(x, Ordering::Relaxed);
        self.state.y.store(y, Ordering::Relaxed);
    }

    /// Unlike a video layer's, this opacity is free: the text is already
    /// drawn through the blend kernel, which takes the layer's own alpha as
    /// one more factor.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), CudaVideoCompositorError> {
        validate_opacity(opacity)?;
        self.state
            .opacity
            .store(opacity.to_bits(), Ordering::Relaxed);
        Ok(())
    }

    /// Shows or hides the text without discarding its uploaded mask.
    pub fn set_visible(&self, visible: bool) {
        self.state.visible.store(visible, Ordering::Relaxed);
    }
}

impl CudaVideoCompositorHandle {
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
    ) -> std::result::Result<CudaTextLayerHandle, CudaVideoCompositorError> {
        if !text_layer.font_size.is_finite() || text_layer.font_size <= 0.0 {
            return Err(CudaVideoCompositorError::InvalidFontSize(
                text_layer.font_size,
            ));
        }
        let Some(shared) = self.shared.upgrade() else {
            return Err(CudaVideoCompositorError::SourceRemoved);
        };
        let font = ab_glyph::FontArc::try_from_vec(text_layer.font_data)
            .map_err(|error| CudaVideoCompositorError::InvalidFont(error.to_string()))?;
        let state = Arc::new(TextLayerState {
            font,
            font_size: text_layer.font_size,
            color: text_layer.color,
            mask: ArcSwapOption::empty(),
            x: AtomicI32::new(text_layer.x),
            y: AtomicI32::new(text_layer.y),
            opacity: AtomicU32::new(1.0f32.to_bits()),
            visible: AtomicBool::new(true),
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
        Ok(CudaTextLayerHandle {
            state,
            driver: shared.driver.clone(),
        })
    }
}

fn text_raster_error(error: TextRasterError) -> CudaVideoCompositorError {
    match error {
        TextRasterError::TooLarge { width, height } => {
            CudaVideoCompositorError::TextTooLarge { width, height }
        }
        TextRasterError::AllocationFailed { bytes } => {
            CudaVideoCompositorError::AllocationFailed { bytes }
        }
    }
}
