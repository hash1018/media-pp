//! Shared, backend-agnostic layer types/math for anything that composites
//! multiple video inputs into one output — [`crate::elements::SwVideoCompositor`]
//! (CPU, `libswscale`) and [`crate::elements::D3d11VideoCompositor`] (GPU,
//! D3D11) both use these exact same types, so a caller's layer-control code
//! doesn't change shape when switching between them. Only the actual pixel
//! work (scale+blend vs. shader draw) differs per backend.

use thiserror::Error as ThisError;

pub(crate) const MAX_DIMENSION: u32 = 16_384;

/// An opaque, stable identity for one compositor input registration.
/// Replacing an input with the same name creates a different identity, so
/// an old sink or layer handle can never affect its replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VideoInputId(pub(crate) u64);

/// An output-space rectangle. Signed coordinates allow a layer to be
/// moved partially outside the canvas while its size remains positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoRect {
    /// Horizontal offset of the rectangle's left edge, in output pixels.
    pub x: i32,
    /// Vertical offset of the rectangle's top edge, in output pixels.
    pub y: i32,
    /// Rectangle width in output pixels; must be nonzero.
    pub width: u32,
    /// Rectangle height in output pixels; must be nonzero.
    pub height: u32,
}

impl VideoRect {
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// How an input's aspect ratio is mapped into its [`VideoRect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFit {
    /// Distort the input to exactly fill the rectangle.
    Stretch,
    /// Preserve aspect ratio and letterbox/pillarbox inside the rectangle.
    Contain,
    /// Preserve aspect ratio, fill the rectangle, and crop overflow.
    Cover,
}

/// Runtime-adjustable spatial settings for one compositor input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoLayer {
    /// Output-space destination and clipping rectangle.
    pub rect: VideoRect,
    /// Stacking order; larger values are drawn over smaller values.
    pub z_index: i32,
    /// Layer alpha in the inclusive range `0.0..=1.0`.
    pub opacity: f32,
    /// Whether the compositor draws this layer.
    pub visible: bool,
    /// Aspect-ratio policy used to map the input into [`Self::rect`].
    pub fit: VideoFit,
}

impl VideoLayer {
    pub const fn new(rect: VideoRect) -> Self {
        Self {
            rect,
            z_index: 0,
            opacity: 1.0,
            visible: true,
            fit: VideoFit::Contain,
        }
    }
}

/// Validation/geometry failures shared by every compositor backend. Each
/// backend's own `{Backend}CompositorError` maps these into its own
/// variants (see e.g. [`crate::elements::SwVideoCompositorError`]) rather
/// than exposing this type directly, so a caller matching on a specific
/// backend's error type sees only that backend's own enum.
#[derive(Debug, Clone, Copy, PartialEq, ThisError)]
pub(crate) enum VideoLayerError {
    #[error(
        "invalid layer dimensions {width}x{height}; each dimension must be 1..={MAX_DIMENSION}"
    )]
    InvalidDimensions { width: u32, height: u32 },

    #[error("layer opacity must be finite and between 0.0 and 1.0, got {0}")]
    InvalidOpacity(f32),

    #[error("input frame has invalid dimensions {width}x{height}")]
    InvalidInputDimensions { width: u32, height: u32 },

    #[error("scaled layer would exceed {MAX_DIMENSION}px: {width}x{height}")]
    ScaledLayerTooLarge { width: u32, height: u32 },
}

pub(crate) fn validate_layer(layer: VideoLayer) -> Result<(), VideoLayerError> {
    validate_rect(layer.rect)?;
    validate_opacity(layer.opacity)
}

pub(crate) fn validate_rect(rect: VideoRect) -> Result<(), VideoLayerError> {
    if rect.width == 0
        || rect.height == 0
        || rect.width > MAX_DIMENSION
        || rect.height > MAX_DIMENSION
    {
        Err(VideoLayerError::InvalidDimensions {
            width: rect.width,
            height: rect.height,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_opacity(opacity: f32) -> Result<(), VideoLayerError> {
    if opacity.is_finite() && (0.0..=1.0).contains(&opacity) {
        Ok(())
    } else {
        Err(VideoLayerError::InvalidOpacity(opacity))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerGeometry {
    pub(crate) image_x: i64,
    pub(crate) image_y: i64,
    pub(crate) image_width: u32,
    pub(crate) image_height: u32,
    pub(crate) clip: VideoRect,
}

/// Computes where and at what size an input actually gets drawn inside its
/// [`VideoRect`], given [`VideoFit`] — shared, backend-agnostic pixel-space
/// math. A CPU backend blits with this directly; a GPU backend turns it
/// into vertex positions/UVs for the same quad.
pub(crate) fn layer_geometry(
    source_width: u32,
    source_height: u32,
    rect: VideoRect,
    fit: VideoFit,
) -> Result<LayerGeometry, VideoLayerError> {
    if source_width == 0 || source_height == 0 {
        return Err(VideoLayerError::InvalidInputDimensions {
            width: source_width,
            height: source_height,
        });
    }

    let source_is_wider = u128::from(rect.width) * u128::from(source_height)
        <= u128::from(rect.height) * u128::from(source_width);
    let (image_width, image_height) = match fit {
        VideoFit::Stretch => (rect.width, rect.height),
        VideoFit::Contain if source_is_wider => (
            rect.width,
            scaled_dimension(source_height, rect.width, source_width),
        ),
        VideoFit::Contain => (
            scaled_dimension(source_width, rect.height, source_height),
            rect.height,
        ),
        VideoFit::Cover if source_is_wider => (
            scaled_dimension(source_width, rect.height, source_height),
            rect.height,
        ),
        VideoFit::Cover => (
            rect.width,
            scaled_dimension(source_height, rect.width, source_width),
        ),
    };
    if image_width > MAX_DIMENSION || image_height > MAX_DIMENSION {
        return Err(VideoLayerError::ScaledLayerTooLarge {
            width: image_width,
            height: image_height,
        });
    }

    Ok(LayerGeometry {
        image_x: i64::from(rect.x) + (i64::from(rect.width) - i64::from(image_width)) / 2,
        image_y: i64::from(rect.y) + (i64::from(rect.height) - i64::from(image_height)) / 2,
        image_width,
        image_height,
        clip: rect,
    })
}

pub(crate) fn scaled_dimension(source: u32, target: u32, divisor: u32) -> u32 {
    let scaled =
        (u128::from(source) * u128::from(target) + u128::from(divisor) / 2) / u128::from(divisor);
    scaled.max(1).min(u128::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contain_and_cover_preserve_aspect_ratio() {
        let rect = VideoRect::new(10, 20, 100, 100);
        let contain = layer_geometry(160, 90, rect, VideoFit::Contain).unwrap();
        assert_eq!((contain.image_width, contain.image_height), (100, 56));
        assert_eq!((contain.image_x, contain.image_y), (10, 42));

        let cover = layer_geometry(160, 90, rect, VideoFit::Cover).unwrap();
        assert_eq!((cover.image_width, cover.image_height), (178, 100));
        assert_eq!((cover.image_x, cover.image_y), (-29, 20));
    }
}
