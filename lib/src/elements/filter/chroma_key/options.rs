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

/// Construction-time options for either chroma-key backend.
#[derive(Debug, Clone, Copy)]
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
