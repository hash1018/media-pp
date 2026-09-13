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

/// What a video's Y'CbCr numbers mean: the matrix they were made with, their
/// range, and the primaries and transfer function of the RGB they stand for.
///
/// A frame carries these in its own fields, and a scaler reads them — see
/// [`SwScaler`](crate::elements::SwScaler). A *stream* carries them only if
/// its encoder was told them before it opened, which is what
/// [`SwEncoder::with_color`](crate::elements::SwEncoder::with_color) and,
/// with the `cuda` feature, `CudaEncoder::with_color` are for. Without
/// them a player guesses, and the guesses differ: FFmpeg reads an untagged
/// stream as BT.601 whatever its size, which turned a BT.709 recording's
/// (230, 20, 20) into (211, 0, 22).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorDescription {
    /// The matrix between R'G'B' and Y'CbCr.
    pub space: ffmpeg_next::color::Space,
    /// Limited (`MPEG`) or full (`JPEG`) range.
    pub range: ffmpeg_next::color::Range,
    /// The primaries of the RGB.
    pub primaries: ffmpeg_next::color::Primaries,
    /// The transfer function of the RGB.
    pub transfer: ffmpeg_next::color::TransferCharacteristic,
}

impl ColorDescription {
    /// HD video: BT.709 throughout, limited range. What this crate's CUDA
    /// path converts to — the compositor's canvas, the converter's output —
    /// and what it says its frames are.
    pub const BT709_LIMITED: Self = Self {
        space: ffmpeg_next::color::Space::BT709,
        range: ffmpeg_next::color::Range::MPEG,
        primaries: ffmpeg_next::color::Primaries::BT709,
        transfer: ffmpeg_next::color::TransferCharacteristic::BT709,
    };

    /// Says this about `frame` — for a frame made outside this crate whose
    /// colour a scaler or an encoder downstream should read.
    pub fn describe(self, frame: &mut ffmpeg_next::frame::Video) {
        frame.set_color_space(self.space);
        frame.set_color_range(self.range);
        frame.set_color_primaries(self.primaries);
        frame.set_color_transfer_characteristic(self.transfer);
    }

    /// Tells an encoder not yet opened what its stream holds, for it to
    /// write into the stream's own headers.
    ///
    /// # Safety
    ///
    /// `context` must be a live `AVCodecContext` that has not been opened.
    pub(crate) unsafe fn tell(self, context: *mut ffmpeg_next::ffi::AVCodecContext) {
        // SAFETY: the caller's promise — a live context, before `open`, which
        // is when these are read.
        unsafe {
            (*context).colorspace = self.space.into();
            (*context).color_range = self.range.into();
            (*context).color_primaries = self.primaries.into();
            (*context).color_trc = self.transfer.into();
        }
    }
}
