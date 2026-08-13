//! [`D3d11TextLayerHandle`] — dynamic text drawn into a
//! [`super::D3d11VideoCompositor`] scene by rasterizing with `ab_glyph` and
//! uploading a fresh GPU texture on every [`D3d11TextLayerHandle::set_text`]
//! call.

use std::{ffi::c_void, sync::Arc};

use ab_glyph::{Font, FontArc, InvalidFont, PxScale, ScaleFont, point};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC,
        D3D11_USAGE_DEFAULT, ID3D11Device, ID3D11Texture2D,
    },
    Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC},
};

use super::{D3d11VideoCompositorError, video_handle::D3d11VideoLayerHandle};
use crate::{
    color::Color, elements::filter::decoder::d3d11va_decoder::wrap_d3d11_texture,
    pool::UnboundObjectPool,
};

use super::super::video_layer::VideoRect;

/// Errors specific to [`D3d11TextLayerHandle`].
#[derive(Debug, ThisError)]
pub enum D3d11TextLayerError {
    #[error("invalid font data: {0}")]
    InvalidFont(#[from] InvalidFont),

    #[error(transparent)]
    Compositor(#[from] D3d11VideoCompositorError),

    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),
}

/// Dynamic text drawn into a [`super::D3d11VideoCompositor`] scene, obtained via
/// [`super::D3d11VideoCompositorHandle::add_text_layer`] — there's no upstream
/// `Pipeline` branch feeding this input, only
/// [`D3d11TextLayerHandle::set_text`] calls. Not an
/// [`crate::element::Element`] — never wired into a `Pipeline` (no `Sink`,
/// no bus/topology identity). Named to match [`D3d11VideoLayerHandle`]'s
/// family, but it isn't as thin as the rest of that family: every other
/// `*Handle` in this crate is a cheap, `Clone`-able `Weak`-backed proxy
/// over state it doesn't own, whereas this one owns a real
/// device/font/frame-pool and does actual GPU work (rasterize + upload) on
/// every `set_text` call.
///
/// Deliberately doesn't expose [`VideoRect`]/`VideoFit` the way
/// [`crate::elements::VideoLayer`] does: unlike video, whose input aspect
/// ratio is arbitrary and must be fit into a caller-chosen box, a
/// rasterized text bitmap's pixel size is always known exactly the moment
/// its content is decided, so `set_text` computes and applies the layer's
/// rect itself from that exact size every time the text changes.
pub struct D3d11TextLayerHandle {
    layer: D3d11VideoLayerHandle,
    device: ID3D11Device,
    font: FontArc,
    font_size: f32,
    color: Color,
    /// Reused across every rasterized frame — see [`crate::elements::D3d11Upload`]'s
    /// own docs on this same pattern. Only the small CPU-side `AVFrame`
    /// wrapper is reused; the GPU texture itself is a fresh allocation
    /// every `set_text` call (text changes are rare compared to video
    /// frame rate, so there's no per-frame pooling pressure here).
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl D3d11TextLayerHandle {
    /// `pub(crate)`, not `pub`: [`super::D3d11VideoCompositorHandle::add_text_layer`]
    /// is the only place outside this crate that can ever produce a
    /// `D3d11TextLayerHandle`, which is what guarantees `device` actually
    /// matches `layer`'s own compositor — a public constructor here would
    /// let a caller reintroduce that mismatch.
    pub(crate) fn new(
        layer: D3d11VideoLayerHandle,
        device: &ID3D11Device,
        font_data: Vec<u8>,
        font_size: f32,
        color: Color,
    ) -> std::result::Result<Self, D3d11TextLayerError> {
        let font = FontArc::try_from_vec(font_data)?;
        Ok(Self {
            layer,
            device: device.clone(),
            font,
            font_size,
            color,
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })
    }

    /// Rasterizes `text` and swaps it in as this layer's frame, resizing
    /// the layer's rect to the new bitmap's exact pixel size (keeping the
    /// top-left position from the last [`Self::set_position`]/creation
    /// call). Empty (or whitespace/control-only) `text` hides the layer
    /// instead of drawing a zero-size frame.
    pub fn set_text(&self, text: &str) -> std::result::Result<(), D3d11TextLayerError> {
        let current = self
            .layer
            .layer()
            .ok_or(D3d11VideoCompositorError::SourceRemoved)?;
        let (x, y) = (current.rect.x, current.rect.y);

        let Some((width, height, pixels)) = rasterize(&self.font, self.font_size, text, self.color)
        else {
            self.layer.set_visible(false)?;
            return Ok(());
        };

        let texture = upload_bgra(&self.device, width, height, &pixels)?;
        let mut frame = self.pool.get();
        // Overwrites the pooled slot's previous contents in place — see
        // `D3d11Upload::consume`'s identical pattern for why this is safe
        // (the old GPU texture is released by `ffmpeg::frame::Video`'s own
        // `Drop` right here, once nothing downstream still holds it).
        *frame = wrap_d3d11_texture(texture, width, height);
        self.layer.set_frame(Arc::new(frame))?;
        self.layer.set_rect(VideoRect::new(x, y, width, height))?;
        self.layer.set_visible(true)?;
        Ok(())
    }

    /// Moves this layer's top-left corner without touching its current
    /// size (unaffected until the next [`Self::set_text`] call).
    pub fn set_position(&self, x: i32, y: i32) -> std::result::Result<(), D3d11TextLayerError> {
        let current = self
            .layer
            .layer()
            .ok_or(D3d11VideoCompositorError::SourceRemoved)?;
        self.layer.set_rect(VideoRect::new(
            x,
            y,
            current.rect.width,
            current.rect.height,
        ))?;
        Ok(())
    }

    pub fn set_opacity(&self, opacity: f32) -> std::result::Result<(), D3d11TextLayerError> {
        self.layer.set_opacity(opacity)?;
        Ok(())
    }

    pub fn set_z_index(&self, z_index: i32) -> std::result::Result<(), D3d11TextLayerError> {
        self.layer.set_z_index(z_index)?;
        Ok(())
    }

    pub fn set_visible(&self, visible: bool) -> std::result::Result<(), D3d11TextLayerError> {
        self.layer.set_visible(visible)?;
        Ok(())
    }
}

/// Rasterizes `text` at `size_px` (pixel height) into a tightly-bounding
/// straight-alpha BGRA buffer: RGB is `color` uniformly, alpha is glyph
/// coverage. Returns `None` for text with no drawable glyphs (empty,
/// all-whitespace, or all-control).
fn rasterize(
    font: &FontArc,
    size_px: f32,
    text: &str,
    color: Color,
) -> Option<(u32, u32, Vec<u8>)> {
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
        return None;
    }

    let width = (caret.x.ceil() as i64).clamp(1, i64::from(u32::MAX)) as u32;
    let height = (scaled.height().ceil() as i64).clamp(1, i64::from(u32::MAX)) as u32;
    let mut pixels = vec![0u8; width as usize * height as usize * 4];
    for glyph in glyphs {
        let Some(outlined) = font.outline_glyph(glyph) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|gx, gy, coverage| {
            let px = bounds.min.x as i32 + gx as i32;
            let py = bounds.min.y as i32 + gy as i32;
            if px < 0 || py < 0 || px as u32 >= width || py as u32 >= height {
                return;
            }
            let index = (py as u32 * width + px as u32) as usize * 4;
            let alpha = (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
            // BGRA, straight (non-premultiplied) alpha — matches
            // `D3d11VideoCompositor`'s BGRA blend state
            // (`SrcBlend = SRC_ALPHA`, `DestBlend = INV_SRC_ALPHA`).
            pixels[index] = color.blue;
            pixels[index + 1] = color.green;
            pixels[index + 2] = color.red;
            pixels[index + 3] = pixels[index + 3].max(alpha);
        });
    }
    Some((width, height, pixels))
}

/// Builds one GPU `ID3D11Texture2D` (`DXGI_FORMAT_B8G8R8A8_UNORM`,
/// `D3D11_USAGE_DEFAULT`, `D3D11_BIND_SHADER_RESOURCE`) with `pixels`
/// (tightly packed `width * height * 4` BGRA bytes) as its initial
/// contents — same construction as [`crate::elements::D3d11Upload::upload`],
/// just BGRA instead of NV12 and driven by a one-off CPU buffer instead of
/// an `ffmpeg::frame::Video`.
fn upload_bgra(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> std::result::Result<ID3D11Texture2D, windows::core::Error> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let initial_data = D3D11_SUBRESOURCE_DATA {
        pSysMem: pixels.as_ptr() as *const c_void,
        SysMemPitch: width * 4,
        SysMemSlicePitch: 0,
    };
    let mut texture: Option<ID3D11Texture2D> = None;
    unsafe {
        device.CreateTexture2D(&desc, Some(&initial_data), Some(&mut texture))?;
    }
    Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
}
