//! What the uploads share in putting a 4:2:0 frame in system memory up as
//! NV12: a software decode's YUV420P is NV12's samples with its chroma in two
//! planes rather than one, so it goes up with its Cb and Cr interleaved on
//! the way and no scaler in front.

use ffmpeg_next as ffmpeg;

/// A plane of the frame shorter than its rows at its stride — reading it
/// would run past the end of its buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaneTooSmall {
    /// Bytes the plane holds.
    pub(crate) actual: usize,
    /// Its row stride in bytes.
    pub(crate) stride: usize,
    /// The rows it has to hold.
    pub(crate) height: u32,
}

/// Whether `format` goes up as NV12 by way of [`interleave_chroma_row`].
pub(crate) fn is_planar_420(format: ffmpeg::format::Pixel) -> bool {
    matches!(
        format,
        ffmpeg::format::Pixel::YUV420P | ffmpeg::format::Pixel::YUVJ420P
    )
}

/// Checks that every plane of `frame` — NV12, or YUV420P or YUVJ420P —
/// holds its rows at its stride, so the rows can then be sliced without a
/// panic.
pub(crate) fn check_planes(frame: &ffmpeg::frame::Video) -> Result<(), PlaneTooSmall> {
    let width = frame.width() as usize;
    let luma_rows = frame.height() as usize;
    let chroma_rows = frame.height().div_ceil(2) as usize;
    let chroma_width = frame.width().div_ceil(2) as usize;
    let planes: &[(usize, usize, usize)] = if is_planar_420(frame.format()) {
        &[
            (0, luma_rows, width),
            (1, chroma_rows, chroma_width),
            (2, chroma_rows, chroma_width),
        ]
    } else {
        &[(0, luma_rows, width), (1, chroma_rows, width)]
    };
    for &(plane, rows, bytes) in planes {
        let stride = frame.stride(plane);
        let needed = stride * rows.saturating_sub(1) + bytes;
        if stride < bytes || frame.data(plane).len() < needed {
            return Err(PlaneTooSmall {
                actual: frame.data(plane).len(),
                stride,
                height: rows as u32,
            });
        }
    }
    Ok(())
}

/// Writes chroma row `row` of a YUV420P `frame` as NV12 has it — Cb and Cr
/// in turn — into `destination`, as many samples as it has room for. `frame`
/// must have passed [`check_planes`].
pub(crate) fn interleave_chroma_row(
    frame: &ffmpeg::frame::Video,
    row: usize,
    destination: &mut [u8],
) {
    let cb = &frame.data(1)[row * frame.stride(1)..];
    let cr = &frame.data(2)[row * frame.stride(2)..];
    for (column, pair) in destination.chunks_mut(2).enumerate() {
        pair[0] = cb[column];
        if let Some(second) = pair.get_mut(1) {
            *second = cr[column];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_rows_come_out_cb_then_cr() {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, 4, 2);
        frame.data_mut(1)[..2].copy_from_slice(&[1, 2]);
        frame.data_mut(2)[..2].copy_from_slice(&[10, 20]);
        check_planes(&frame).unwrap();
        let mut row = [0u8; 4];
        interleave_chroma_row(&frame, 0, &mut row);
        assert_eq!(row, [1, 10, 2, 20]);
    }
}
