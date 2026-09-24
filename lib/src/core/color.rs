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

    /// What `frame` says of its colour, each part as it says it —
    /// `Unspecified` where it says nothing.
    pub fn of(frame: &ffmpeg_next::frame::Video) -> Self {
        Self {
            space: frame.color_space(),
            range: frame.color_range(),
            primaries: frame.color_primaries(),
            transfer: frame.color_transfer_characteristic(),
        }
    }

    /// Three affine rows that turn a Y'CbCr sample of a picture `height`
    /// rows high into R'G'B': normalized `(Y, Cb, Cr, 1)`, each `0.0..=1.0`,
    /// dotted with each row gives red, green and blue. What a shader drawing
    /// a YUV picture multiplies by — the ones this crate's own renderers use.
    ///
    /// The matrix and range are this description's; where it names no
    /// matrix, BT.709 for a picture over 576 rows and BT.601 otherwise, and
    /// where no range, limited — what an untagged stream most often is.
    /// Primaries and transfer are not converted: BT.2020 colour is read with
    /// its own matrix into BT.2020 RGB.
    pub fn yuv_to_rgb_rows(&self, height: u32) -> [[f32; 4]; 3] {
        yuv_to_rgb_rows(self.space, self.range, height)
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg_next::color::{Range, Space};

    /// Red as each matrix makes it: (230, 20, 20) in limited range.
    fn red(rows: [[f32; 4]; 3], yuv: [u8; 3]) -> [u8; 3] {
        let [y, cb, cr] = yuv.map(|value| f32::from(value) / 255.0);
        rows.map(|[from_y, from_cb, from_cr, offset]| {
            ((from_y * y + from_cb * cb + from_cr * cr + offset).clamp(0.0, 1.0) * 255.0 + 0.5)
                as u8
        })
    }

    /// A frame's own description is read as it says it, and the rows follow
    /// its matrix — or, where it names none, its height.
    #[test]
    fn the_rows_follow_the_matrix_or_else_the_height() {
        let mut frame = ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::NV12, 16, 16);
        let untagged = ColorDescription::of(&frame);
        assert_eq!(untagged.space, Space::Unspecified);
        ColorDescription::BT709_LIMITED.describe(&mut frame);
        assert_eq!(
            ColorDescription::of(&frame),
            ColorDescription::BT709_LIMITED
        );

        // BT.709 red, (72, 107, 220), read as BT.709 whatever the height.
        let bt709 = [72, 107, 220];
        let near = |got: [u8; 3], want: [u8; 3]| {
            got.iter()
                .zip(want)
                .all(|(got, want)| got.abs_diff(want) <= 2)
        };
        assert!(near(
            red(ColorDescription::BT709_LIMITED.yuv_to_rgb_rows(240), bt709),
            [230, 20, 20]
        ));
        // Untagged: BT.709 above 576 rows, BT.601 at or below — which reads
        // the same samples as another red.
        assert!(near(
            red(untagged.yuv_to_rgb_rows(1080), bt709),
            [230, 20, 20]
        ));
        assert!(!near(
            red(untagged.yuv_to_rgb_rows(480), bt709),
            [230, 20, 20]
        ));
        let full = ColorDescription {
            range: Range::JPEG,
            ..ColorDescription::BT709_LIMITED
        };
        assert_eq!(
            full.yuv_to_rgb_rows(240)[0][0],
            1.0,
            "full range is not stretched"
        );
    }
}
