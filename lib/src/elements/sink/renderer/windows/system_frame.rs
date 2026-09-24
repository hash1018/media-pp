//! What the D3D11 and D3D12 window renderers need to draw a frame straight
//! from system memory: which layouts they take, the planes each is uploaded
//! into, and the colour conversion a frame asks for.

use ffmpeg_next::{self as ffmpeg, format::Pixel};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM,
};

use crate::color::yuv_to_rgb_rows;

/// A system-memory layout a window renderer draws, each with a pixel shader
/// of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SystemLayout {
    /// A luma plane and an interleaved chroma plane at half size.
    Nv12,
    /// Luma, Cb and Cr planes, the chroma ones at half size.
    Yuv420p,
    /// One plane of packed B, G, R and A bytes.
    Bgra,
}

/// One plane of a frame, as the texture it is uploaded into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PlaneShape {
    pub(super) format: DXGI_FORMAT,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) texel_bytes: u32,
}

impl PlaneShape {
    pub(super) fn row_bytes(&self) -> usize {
        self.width as usize * self.texel_bytes as usize
    }
}

impl SystemLayout {
    pub(super) fn of(format: Pixel) -> Option<Self> {
        match format {
            Pixel::NV12 => Some(Self::Nv12),
            Pixel::YUV420P | Pixel::YUVJ420P => Some(Self::Yuv420p),
            Pixel::BGRA => Some(Self::Bgra),
            _ => None,
        }
    }

    /// The planes of a `width` x `height` frame. Chroma is rounded up, as
    /// FFmpeg rounds it: an odd-sized frame's last column and row have a
    /// chroma sample of their own.
    pub(super) fn planes(self, width: u32, height: u32) -> Vec<PlaneShape> {
        let (half_width, half_height) = (width.div_ceil(2), height.div_ceil(2));
        let plane = |format, width, height, texel_bytes| PlaneShape {
            format,
            width,
            height,
            texel_bytes,
        };
        match self {
            Self::Nv12 => vec![
                plane(DXGI_FORMAT_R8_UNORM, width, height, 1),
                plane(DXGI_FORMAT_R8G8_UNORM, half_width, half_height, 2),
            ],
            Self::Yuv420p => vec![
                plane(DXGI_FORMAT_R8_UNORM, width, height, 1),
                plane(DXGI_FORMAT_R8_UNORM, half_width, half_height, 1),
                plane(DXGI_FORMAT_R8_UNORM, half_width, half_height, 1),
            ],
            Self::Bgra => vec![plane(DXGI_FORMAT_B8G8R8A8_UNORM, width, height, 4)],
        }
    }
}

/// Each of `frame`'s planes as `layout` has them — its bytes and the stride
/// between rows — or `None` if the frame does not hold them: a plane
/// missing, a stride shorter than a row, or fewer rows than the plane has.
/// Checked before anything is uploaded, so a malformed frame is refused
/// rather than read past its end.
pub(super) fn planes_of(
    frame: &ffmpeg::frame::Video,
    layout: SystemLayout,
) -> Option<Vec<(&[u8], usize)>> {
    let shapes = layout.planes(frame.width(), frame.height());
    if frame.width() == 0 || frame.height() == 0 || frame.planes() < shapes.len() {
        return None;
    }
    shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            let (data, stride) = (frame.data(index), frame.stride(index));
            let rows = shape.height as usize;
            let needed = stride
                .checked_mul(rows - 1)?
                .checked_add(shape.row_bytes())?;
            (stride >= shape.row_bytes() && data.len() >= needed).then_some((data, stride))
        })
        .collect()
}

/// The three rows a frame's Y'CbCr is converted to R'G'B' with: its own
/// matrix and range, and full range for a YUVJ420P frame whatever it says —
/// the J is FFmpeg's name for full-range 4:2:0, which not every producer
/// also says in the range field.
pub(super) fn colour_rows(frame: &ffmpeg::frame::Video) -> [[f32; 4]; 3] {
    let range = match frame.format() {
        Pixel::YUVJ420P => ffmpeg::color::Range::JPEG,
        _ => frame.color_range(),
    };
    yuv_to_rgb_rows(frame.color_space(), range, frame.height())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_planes_round_up_for_an_odd_size() {
        let planes = SystemLayout::Yuv420p.planes(5, 3);
        assert_eq!(
            planes
                .iter()
                .map(|plane| (plane.width, plane.height))
                .collect::<Vec<_>>(),
            [(5, 3), (3, 2), (3, 2)]
        );
        assert_eq!(SystemLayout::Nv12.planes(5, 3)[1].row_bytes(), 6);
    }

    #[test]
    fn a_frame_short_of_a_plane_is_refused() {
        let frame = ffmpeg::frame::Video::new(Pixel::NV12, 64, 32);
        assert_eq!(
            planes_of(&frame, SystemLayout::Nv12).map(|planes| planes.len()),
            Some(2)
        );
        // Read as three planes, an NV12 frame has no third.
        assert!(planes_of(&frame, SystemLayout::Yuv420p).is_none());
    }
}
