//! [`D3d12TextLayerHandle`] — text drawn into a [`super::D3d12VideoCompositor`]
//! scene, rasterized on the CPU and uploaded as a coverage mask on every
//! [`D3d12TextLayerHandle::set_text`].

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering},
};

use arc_swap::ArcSwapOption;
use windows::Win32::Graphics::{Direct3D12::ID3D12Resource, Dxgi::Common::DXGI_FORMAT_R8_UNORM};

use super::super::super::text_layer::{
    TextFontError, TextLayer, TextRasterError, load_font, rasterize_coverage,
};
use super::{D3d12VideoCompositorError, D3d12VideoCompositorHandle, Uploads, validate_opacity};
use crate::{
    color::Color,
    platform::windows::{
        d3d12_compute::{Commands, upload_texture},
        d3d12_gpu::D3d12Gpu,
    },
};

/// A text layer's coverage, on the GPU: one byte a pixel.
pub(crate) struct D3d12Mask {
    pub(super) texture: ID3D12Resource,
    pub(super) width: u32,
    pub(super) height: u32,
}

// SAFETY: a D3D12 resource is free-threaded, and this one is never written
// once uploaded.
unsafe impl Send for D3d12Mask {}
// SAFETY: as above.
unsafe impl Sync for D3d12Mask {}

/// One registered text layer's live state, shared between its handle and
/// the compositor's own thread.
pub(crate) struct TextLayerState {
    pub(super) font: ab_glyph::FontArc,
    pub(super) font_size: f32,
    pub(super) color: Color,
    /// Replaced wholesale by `set_text`: the compositor draws the whole
    /// previous mask or the whole new one.
    pub(super) mask: ArcSwapOption<D3d12Mask>,
    pub(super) x: AtomicI32,
    pub(super) y: AtomicI32,
    /// `f32` bits.
    pub(super) opacity: AtomicU32,
    pub(super) visible: AtomicBool,
    /// Where it stacks among the video layers and the other text layers.
    pub(super) z_index: AtomicI32,
}

/// Writes and places one text layer of a
/// [`D3d12VideoCompositor`](super::D3d12VideoCompositor) while it runs.
///
/// Made by [`D3d12VideoCompositorHandle::add_text_layer`]; the
/// [`D3d12VideoCompositor`](super::D3d12VideoCompositor) example shows one in
/// use. The D3D12 sibling of `CudaTextLayerHandle`.
///
/// Cloning is cheap and every clone controls the same layer. It holds the
/// layer's own state, the device and what masks are uploaded through —
/// not the compositor or its inputs. `set_text` does real
/// work: it rasterizes the string on the CPU and uploads the coverage to
/// the GPU, waiting for the upload, so it is not something to call per
/// frame if the text has not changed.
#[derive(Clone)]
pub struct D3d12TextLayerHandle {
    state: Arc<TextLayerState>,
    gpu: D3d12Gpu,
    uploads: Uploads,
}

impl D3d12TextLayerHandle {
    /// Rasterizes `text` and uploads it, replacing whatever was drawn
    /// before. Text with no drawable glyphs clears the layer.
    pub fn set_text(&self, text: &str) -> std::result::Result<(), D3d12VideoCompositorError> {
        let rasterized = rasterize_coverage(&self.state.font, self.state.font_size, text)
            .map_err(text_raster_error)?;
        let Some(mask) = rasterized else {
            self.state.mask.store(None);
            return Ok(());
        };
        let gpu = &self.gpu;
        // Uploaded before it is published, so the compositor never sees a
        // half-written mask.
        let texture = {
            let mut uploads = self.uploads.lock().unwrap();
            let commands = match uploads.as_mut() {
                Some(commands) => commands,
                None => uploads.insert(Commands::new(gpu.device(), gpu.queue())?),
            };
            let width = mask.width as usize;
            upload_texture(
                gpu.device(),
                commands,
                DXGI_FORMAT_R8_UNORM,
                (mask.width, mask.height),
                (&mask.coverage, width, width),
            )?
        };
        self.state.mask.store(Some(Arc::new(D3d12Mask {
            texture,
            width: mask.width,
            height: mask.height,
        })));
        Ok(())
    }

    /// Moves the layer's top-left corner, aligned to even pixels when drawn.
    pub fn set_position(&self, x: i32, y: i32) {
        self.state.x.store(x, Ordering::Relaxed);
        self.state.y.store(y, Ordering::Relaxed);
    }

    /// Blends the text at `opacity`, from 0 to 1.
    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), D3d12VideoCompositorError> {
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

    /// Where it stacks: among video layers and other text layers by
    /// `z_index`, a video layer first where the two are equal.
    pub fn set_z_index(&self, z_index: i32) {
        self.state.z_index.store(z_index, Ordering::Relaxed);
    }
}

impl D3d12VideoCompositorHandle {
    /// Registers a text layer and returns its control handle. Reusing
    /// `name` replaces the previous registration.
    ///
    /// The font is parsed here, so bad font data fails at registration
    /// rather than at the first `set_text`. Nothing is drawn until
    /// `set_text` is called.
    pub fn add_text_layer(
        &self,
        name: impl Into<String>,
        text_layer: TextLayer,
    ) -> std::result::Result<D3d12TextLayerHandle, D3d12VideoCompositorError> {
        let font =
            load_font(text_layer.font_data, text_layer.font_size).map_err(|error| match error {
                TextFontError::Size(size) => D3d12VideoCompositorError::InvalidFontSize(size),
                TextFontError::Font(error) => {
                    D3d12VideoCompositorError::InvalidFont(error.to_string())
                }
            })?;
        let Some(shared) = self.shared.upgrade() else {
            return Err(D3d12VideoCompositorError::Stopped);
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
        let name: Arc<str> = name.into().into();
        let mut layers = shared.text_layers.lock().unwrap();
        let replaced = match layers.iter_mut().find(|(existing, _)| *existing == name) {
            Some(slot) => Some(std::mem::replace(&mut slot.1, state.clone())),
            None => {
                layers.push((name, state.clone()));
                None
            }
        };
        drop(layers);
        drop(replaced);
        Ok(D3d12TextLayerHandle {
            state,
            gpu: shared.gpu.clone(),
            uploads: shared.uploads.clone(),
        })
    }
}

fn text_raster_error(error: TextRasterError) -> D3d12VideoCompositorError {
    match error {
        TextRasterError::TooLarge { width, height } => {
            D3d12VideoCompositorError::TextTooLarge { width, height }
        }
        TextRasterError::AllocationFailed { bytes } => {
            D3d12VideoCompositorError::AllocationFailed { bytes }
        }
    }
}

impl crate::elements::TextLayerControl for D3d12TextLayerHandle {
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
