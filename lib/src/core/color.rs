//! A plain RGB color value, shared by anything in this crate that needs
//! one — compositor backgrounds, layer/text colors, and so on. Kept in
//! `core` rather than under a specific element's module because nothing
//! about it is pipeline- or backend-specific.

/// An opaque RGB color (no alpha — compositor output is always opaque;
/// per-layer translucency is a separate `opacity` field, not per-pixel
/// alpha here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    /// Red component.
    pub red: u8,
    /// Green component.
    pub green: u8,
    /// Blue component.
    pub blue: u8,
}

impl Color {
    /// Opaque black — the default compositor background.
    pub const BLACK: Self = Self::new(0, 0, 0);
    /// Opaque white.
    pub const WHITE: Self = Self::new(255, 255, 255);

    /// Creates an opaque color from 8-bit red, green, and blue components.
    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }
}
