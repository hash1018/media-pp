use ffmpeg_next::{self as ffmpeg, Rescale};

#[cfg(feature = "cuda")]
mod cuda;
mod sw_encoder;

#[cfg(feature = "cuda")]
pub use cuda::{CudaCodec, CudaEncoder, CudaEncoderError, CudaEncoderOptions};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
mod windows;

pub use sw_encoder::{SwEncoder, SwEncoderError, SwEncoderOptions, VideoCodec};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
pub use windows::*;

/// The unit a video encoder here counts its output in, unless its codec
/// cannot: 1/90000, the MPEG systems clock. Fine enough that a variable-rate
/// capture's frames never land on the same tick, and it counts every common
/// frame rate — 24, 25, 30, 50, 60 and their 1000/1001 variants — in whole
/// ticks or within one.
///
/// The encoder's own rather than the frames': a codec is opened with its time
/// base before the first frame arrives, and each frame says what unit it is
/// in — see [`crate::buffer::time_base`] — so the encoder converts into this
/// instead of asking to be told the frames' and trusting it matches.
pub(super) const TIME_BASE: ffmpeg::Rational = ffmpeg::Rational(1, 90_000);

/// A frame had a `pts` and no unit to read it in. Each encoder turns this
/// into a `NoTimeBase` of its own error.
pub(super) struct NoTimeBase;

/// `frame`'s `pts` in `time_base`, read in the unit `frame` carries: `None`
/// for a frame with no `pts`, [`NoTimeBase`] for one that has a `pts` and
/// does not say what it counts.
pub(super) fn pts_in(
    frame: &ffmpeg::Frame,
    time_base: ffmpeg::Rational,
) -> Result<Option<i64>, NoTimeBase> {
    let Some(pts) = frame.pts() else {
        return Ok(None);
    };
    let unit = crate::buffer::time_base(frame).ok_or(NoTimeBase)?;
    Ok(Some(pts.rescale(unit, time_base)))
}

/// A reference to `frame`'s picture stamped with `pts` in `time_base` — the
/// frame handed to an encoder, since the one it arrived as may be shared
/// with a sibling branch that must keep its own timestamp. Nothing is
/// copied: the new frame points at the same buffers.
pub(super) fn restamped(
    frame: &ffmpeg::frame::Video,
    pts: Option<i64>,
    time_base: ffmpeg::Rational,
) -> Result<ffmpeg::frame::Video, ffmpeg::Error> {
    let mut stamped = ffmpeg::frame::Video::empty();
    // SAFETY: `stamped` is a fresh frame this function owns, and `frame` is
    // a live one; `av_frame_ref` copies its properties and takes a new
    // reference to each of its buffers.
    let code = unsafe { ffmpeg::ffi::av_frame_ref(stamped.as_mut_ptr(), frame.as_ptr()) };
    if code < 0 {
        return Err(ffmpeg::Error::from(code));
    }
    stamped.set_pts(pts);
    crate::buffer::set_time_base(&mut stamped, time_base);
    Ok(stamped)
}
