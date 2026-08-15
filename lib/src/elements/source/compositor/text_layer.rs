//! [`TextLayer`] — backend-agnostic construction-time settings for a
//! dynamic text layer, the text sibling of [`super::video_layer::VideoLayer`].
//! Mirrors that module's own role: plain data only, no backend-specific
//! implementation. The only backend that currently draws a `TextLayer` is
//! [`super::d3d11_video_compositor`] (`D3d11TextLayerHandle`); this file
//! has no D3D11 dependency itself.

use crate::color::Color;

/// Construction-time settings for one text layer, passed to
/// `D3d11VideoCompositorHandle::add_text_layer` — the
/// text sibling of [`super::video_layer::VideoLayer`], which `add_source`
/// takes the same way. `font_data` (raw TTF/OTF bytes; this crate bundles
/// no font of its own) has no sane default, so — mirroring
/// [`super::video_layer::VideoLayer::new`], which takes the one field a
/// caller must supply (`rect`) and defaults the rest — [`Self::new`] takes
/// only `font_data` and defaults `font_size`/`color`/`x`/`y`, all freely
/// reassignable before the call to `add_text_layer`.
#[derive(Debug, Clone)]
pub struct TextLayer {
    pub font_data: Vec<u8>,
    /// Pixel height of rendered glyphs (not a point size).
    pub font_size: f32,
    pub color: Color,
    /// Initial top-left corner of the layer — like `VideoLayer::rect`'s
    /// `x`/`y`, but with no `width`/`height` counterpart, since a text
    /// layer's size is only known once something has actually rasterized
    /// its content (e.g. `D3d11TextLayerHandle::set_text`).
    pub x: i32,
    pub y: i32,
}

impl TextLayer {
    pub const fn new(font_data: Vec<u8>) -> Self {
        Self {
            font_data,
            font_size: 32.0,
            color: Color::WHITE,
            x: 0,
            y: 0,
        }
    }
}
