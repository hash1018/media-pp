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
/// with the `cuda` feature, `CudaEncoder::with_color` — and on Windows with
/// `d3d11`, `D3d11VideoEncoder::with_color` — are for. Without
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

/// Builds three affine rows that turn normalized `(Y, Cb, Cr, 1)` samples
/// into RGB. Unspecified color metadata follows the common SD/HD fallback:
/// BT.601 through 576 lines and BT.709 above it; unspecified range is
/// treated as MPEG/limited, matching ordinary decoded NV12 video.
#[cfg(any(
    feature = "cuda",
    all(target_os = "windows", any(feature = "d3d11", feature = "d3d12"))
))]
pub(crate) fn yuv_to_rgb_rows(
    space: ffmpeg_next::color::Space,
    range: ffmpeg_next::color::Range,
    height: u32,
) -> [[f32; 4]; 3] {
    let (kr, kb) = match space {
        ffmpeg_next::color::Space::BT709 => (0.2126f32, 0.0722f32),
        ffmpeg_next::color::Space::BT2020NCL | ffmpeg_next::color::Space::BT2020CL => {
            (0.2627f32, 0.0593f32)
        }
        ffmpeg_next::color::Space::FCC => (0.30f32, 0.11f32),
        ffmpeg_next::color::Space::SMPTE240M => (0.212f32, 0.087f32),
        ffmpeg_next::color::Space::Unspecified if height > 576 => (0.2126f32, 0.0722f32),
        _ => (0.299f32, 0.114f32),
    };
    let kg = 1.0 - kr - kb;
    let (y_offset, y_scale, chroma_scale) = match range {
        ffmpeg_next::color::Range::JPEG => (0.0, 1.0, 1.0),
        ffmpeg_next::color::Range::MPEG | ffmpeg_next::color::Range::Unspecified => {
            (16.0 / 255.0, 255.0 / 219.0, 255.0 / 224.0)
        }
    };
    let chroma_offset = 128.0 / 255.0;
    let red_cr = 2.0 * (1.0 - kr) * chroma_scale;
    let blue_cb = 2.0 * (1.0 - kb) * chroma_scale;
    let green_cb = -2.0 * kb * (1.0 - kb) / kg * chroma_scale;
    let green_cr = -2.0 * kr * (1.0 - kr) / kg * chroma_scale;
    let offset = |cb: f32, cr: f32| -y_scale * y_offset - cb * chroma_offset - cr * chroma_offset;

    [
        [y_scale, 0.0, red_cr, offset(0.0, red_cr)],
        [y_scale, green_cb, green_cr, offset(green_cb, green_cr)],
        [y_scale, blue_cb, 0.0, offset(blue_cb, 0.0)],
    ]
}

/// Whether `space` is a BT.2020 matrix — and so, as this crate reads it,
/// BT.2020 primaries too.
///
/// A frame's primaries tag is often missing where its matrix is not, and a
/// BT.2020 matrix is not paired with anything else in practice; a D3D11
/// video processor, told only `DXGI_COLOR_SPACE_*_P2020`, assumes the same.
pub(crate) fn is_bt2020(space: ffmpeg_next::color::Space) -> bool {
    matches!(
        space,
        ffmpeg_next::color::Space::BT2020NCL | ffmpeg_next::color::Space::BT2020CL
    )
}

/// Linear-light BT.2020 RGB to linear-light BT.709 RGB, from the two sets
/// of primaries and their shared D65 white point (ITU-R BT.2087). A
/// saturated BT.2020 colour has no BT.709 equivalent and comes out of range,
/// to be clipped.
#[cfg(any(feature = "cuda", all(target_os = "windows", feature = "d3d11")))]
pub(crate) const BT2020_TO_BT709: [[f32; 3]; 3] = [
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
];
