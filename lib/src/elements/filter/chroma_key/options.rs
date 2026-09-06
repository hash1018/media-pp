//! What to key and how hard — the settings both chroma-key backends read,
//! so a caller's keying configuration doesn't change shape when moving
//! between [`super::SwChromaKey`] and the GPU-resident sibling.

use crate::color::Color;

/// Which background color a chroma key treats as transparent. `Green`/
/// `Blue` are the two conventional screen colors (mirroring GStreamer's
/// `alpha` element's `method` property); `Custom` covers anything else —
/// a differently colored backdrop, or a solid-color background that isn't
/// a screen at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaKeyMethod {
    /// Key the conventional pure-green screen color.
    Green,
    /// Key the conventional pure-blue screen color.
    Blue,
    /// Key an explicitly supplied RGB color.
    Custom(Color),
}

impl ChromaKeyMethod {
    pub(crate) fn key_color(self) -> Color {
        match self {
            ChromaKeyMethod::Green => Color::new(0, 255, 0),
            ChromaKeyMethod::Blue => Color::new(0, 0, 255),
            ChromaKeyMethod::Custom(color) => color,
        }
    }
}

/// Resolves `threshold`/`smoothing` into the feather band a GPU backend
/// evaluates: `saturate((distance - band_low) * inv_band_width)`.
///
/// A hard key (`smoothing <= 0.0`) is a band of no width, which that
/// expression cannot represent directly — so it is given a `band_low` of
/// exactly `threshold` and an `inv_band_width` large enough that any
/// distance above `threshold`, by however little, saturates to 1.0 while
/// `threshold` itself still lands on 0.0. That is precisely
/// [`SwChromaKey`](super::SwChromaKey)'s own step, and it costs neither a
/// branch nor a division by zero.
///
/// Shared rather than resolved per backend: the D3D11 shader and the CUDA
/// kernel evaluate the same expression, and two copies of this would be two
/// chances for one of them to drift into keying differently from the other.
/// Only the GPU backends resolve the band ahead of time; the software one
/// evaluates `threshold`/`smoothing` directly where it needs them, so a build
/// with neither backend has no caller for this.
#[cfg(any(feature = "cuda", all(target_os = "windows", feature = "d3d11")))]
pub(crate) fn feather_band(threshold: f32, smoothing: f32) -> (f32, f32) {
    let smoothing = smoothing.max(0.0);
    if smoothing > 0.0 {
        (threshold - smoothing / 2.0, 1.0 / smoothing)
    } else {
        (threshold, f32::MAX)
    }
}

/// The settings either chroma-key backend keys by.
///
/// Passed at construction and changed afterwards through
/// [`ChromaKeyHandle`](super::ChromaKeyHandle).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaKeyOptions {
    /// Color selection used as the transparent key.
    pub method: ChromaKeyMethod,
    /// How far (as a fraction of the maximum possible RGB distance, so
    /// `0.0..=1.0` is the meaningful range) a pixel may differ from the key
    /// color before it counts as foreground rather than background.
    pub threshold: f32,
    /// Width of the linear feather band straddling `threshold`, in the
    /// same 0.0..=1.0 units — this is what keeps a key edge from aliasing
    /// into a hard, jagged cutout. `0.0` (or negative) is a hard step with
    /// no feathering at all.
    pub smoothing: f32,
}
