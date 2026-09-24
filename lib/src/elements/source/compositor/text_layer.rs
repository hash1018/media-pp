//! [`TextLayer`] — backend-agnostic construction-time settings for a
//! dynamic text layer, the text sibling of [`super::video_layer::VideoLayer`]
//! — plus the glyph rasterization both backends that draw one share.
//!
//! Rasterizing is backend-agnostic by nature: `ab_glyph` turns a font and a
//! string into per-pixel coverage, and what differs is only what each
//! backend does with that coverage. D3D11 and the software compositor
//! expand it into straight-alpha BGRA — [`rasterize_bgra`] — for their
//! blending; CUDA hands it to a blend kernel as a mask with the color as a
//! scalar.

use crate::color::Color;

/// A rasterized string: tightly packed per-pixel coverage, one byte each.
///
/// Coverage, not color: `TextLayer::color` is uniform over the whole layer,
/// so carrying it per pixel would be three redundant bytes each. Each
/// backend combines the two in whatever form it draws with.
pub(crate) struct TextMask {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// `width * height` bytes, row-major, 0 = untouched by any glyph.
    pub(crate) coverage: Vec<u8>,
}

/// Errors from rasterizing, which each backend maps into its own text-layer
/// error type so a caller matching on one sees only that backend's enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextRasterError {
    TooLarge { width: u64, height: u64 },
    AllocationFailed { bytes: usize },
}

/// A rasterized string this crate refuses to allocate for — a guard against
/// a pathological font size turning one `set_text` into gigabytes.
pub(crate) const MAX_TEXT_PIXELS: usize = 16 * 1024 * 1024;

/// Rasterizes `text` at `size_px` (pixel height) into tightly-bounding
/// coverage. `None` for text with no drawable glyphs (empty, all-whitespace,
/// or all-control).
pub(crate) fn rasterize_coverage(
    font: &ab_glyph::FontArc,
    size_px: f32,
    text: &str,
) -> Result<Option<TextMask>, TextRasterError> {
    use ab_glyph::{Font, PxScale, ScaleFont, point};

    use super::video_layer::MAX_DIMENSION;

    let scaled = font.as_scaled(PxScale::from(size_px));
    let mut glyphs = Vec::new();
    let mut caret = point(0.0, scaled.ascent());
    let mut last_id = None;
    for c in text.chars() {
        if c.is_control() {
            continue;
        }
        let mut glyph = scaled.scaled_glyph(c);
        if let Some(last_id) = last_id {
            caret.x += scaled.kern(last_id, glyph.id);
        }
        glyph.position = caret;
        caret.x += scaled.h_advance(glyph.id);
        last_id = Some(glyph.id);
        glyphs.push(glyph);
    }
    if glyphs.is_empty() {
        return Ok(None);
    }

    let outlined: Vec<_> = glyphs
        .into_iter()
        .filter_map(|glyph| font.outline_glyph(glyph))
        .collect();
    if outlined.is_empty() {
        return Ok(None);
    }

    let width_f = caret.x.ceil();
    let height_f = scaled.height().ceil();
    if !width_f.is_finite()
        || !height_f.is_finite()
        || width_f > MAX_DIMENSION as f32
        || height_f > MAX_DIMENSION as f32
    {
        return Err(TextRasterError::TooLarge {
            width: width_f.max(0.0) as u64,
            height: height_f.max(0.0) as u64,
        });
    }
    let width = width_f.max(1.0) as u32;
    let height = height_f.max(1.0) as u32;
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .filter(|&count| count <= MAX_TEXT_PIXELS)
        .ok_or(TextRasterError::TooLarge {
            width: width.into(),
            height: height.into(),
        })?;
    let mut coverage = Vec::new();
    coverage
        .try_reserve_exact(pixel_count)
        .map_err(|_| TextRasterError::AllocationFailed { bytes: pixel_count })?;
    coverage.resize(pixel_count, 0u8);
    for outlined in outlined {
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, value| {
            let px = bounds.min.x as i32 + gx as i32;
            let py = bounds.min.y as i32 + gy as i32;
            if px < 0 || py < 0 || px as u32 >= width || py as u32 >= height {
                return;
            }
            let index = (py as u32 * width + px as u32) as usize;
            let alpha = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            // Glyphs can overlap (kerning, accents); the strongest coverage
            // wins rather than the last one drawn.
            coverage[index] = coverage[index].max(alpha);
        });
    }
    Ok(Some(TextMask {
        width,
        height,
        coverage,
    }))
}

/// [`rasterize_coverage`], expanded into the straight-alpha BGRA the D3D11
/// and software compositors blend: RGB is `color` uniformly, alpha is glyph
/// coverage — tightly packed, `width * height * 4` bytes. `None` for text
/// with no drawable glyphs.
pub(crate) fn rasterize_bgra(
    font: &ab_glyph::FontArc,
    size_px: f32,
    text: &str,
    color: Color,
) -> Result<Option<(u32, u32, Vec<u8>)>, TextRasterError> {
    let Some(mask) = rasterize_coverage(font, size_px, text)? else {
        return Ok(None);
    };
    let byte_count = mask
        .coverage
        .len()
        .checked_mul(4)
        .ok_or(TextRasterError::TooLarge {
            width: mask.width.into(),
            height: mask.height.into(),
        })?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(byte_count)
        .map_err(|_| TextRasterError::AllocationFailed { bytes: byte_count })?;
    for alpha in mask.coverage {
        pixels.extend_from_slice(&[color.blue, color.green, color.red, alpha]);
    }
    Ok(Some((mask.width, mask.height, pixels)))
}

/// Construction-time settings for one text layer, passed to a compositor
/// handle's `add_text_layer` — the software, D3D11 or CUDA one — the
/// text sibling of [`super::video_layer::VideoLayer`], which `add_source`
/// takes the same way. `font_data` (raw TTF/OTF bytes; this crate bundles
/// no font of its own) has no sane default, so — mirroring
/// [`super::video_layer::VideoLayer::new`], which takes the one field a
/// caller must supply (`rect`) and defaults the rest — [`Self::new`] takes
/// only `font_data` and defaults `font_size`/`color`/`x`/`y`, all freely
/// reassignable before the call to `add_text_layer`.
///
/// A text layer is stacked like a video input, at `z_index` 0 until its
/// handle's `set_z_index` says otherwise — and among equal `z_index` values
/// the one added later is drawn over the one added earlier, so a label added
/// after the video it labels is on top of it as it is.
#[derive(Debug, Clone)]
pub struct TextLayer {
    /// Raw TrueType or OpenType font bytes owned by the layer.
    pub font_data: Vec<u8>,
    /// Pixel height of rendered glyphs (not a point size).
    pub font_size: f32,
    /// Initial glyph color, including alpha.
    pub color: Color,
    /// Initial top-left corner of the layer — like `VideoLayer::rect`'s
    /// `x`/`y`, but with no `width`/`height` counterpart, since a text
    /// layer's size is only known once something has actually rasterized
    /// its content (e.g. `D3d11TextLayerHandle::set_text`).
    pub x: i32,
    /// Initial vertical offset of the layer's top edge, in output pixels.
    pub y: i32,
}

impl TextLayer {
    /// Creates a text layer with the supplied font bytes and default visual settings.
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
