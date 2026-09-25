//! Which of FFmpeg's decoders for a codec can decode on a given kind of
//! device — what every hardware decoder here opens, rather than whichever
//! decoder FFmpeg would pick by default.
//!
//! FFmpeg may register several decoders for one codec, and its default can be
//! a software library with no hardware path at all: with `libdav1d` built in,
//! AV1 goes to that ahead of FFmpeg's own `av1` decoder, which is the one
//! NVDEC, D3D11VA and D3D12VA are reached through. Opened by default, such a
//! decoder takes the device context it is handed, ignores it, and fails only
//! at the first frame — after a pipeline has been wired around it.

use ffmpeg_next::{self as ffmpeg, ffi};

/// The first decoder for `id` that decodes on `device`, walking FFmpeg's
/// decoders in the order `avcodec_find_decoder` does, so among several that
/// qualify the one FFmpeg prefers wins. Experimental decoders are passed
/// over, as that function passes them over whenever anything else is
/// registered.
pub(super) fn capable_decoder(
    id: ffmpeg::codec::Id,
    device: ffi::AVHWDeviceType,
) -> Option<ffmpeg::Codec> {
    let id: ffi::AVCodecID = id.into();
    let mut opaque = std::ptr::null_mut();
    loop {
        // SAFETY: `opaque` is the iteration state `av_codec_iterate` owns and
        // began from null; what it returns is null at the end and otherwise a
        // static descriptor that lives as long as the program.
        let codec = unsafe { ffi::av_codec_iterate(&mut opaque) };
        if codec.is_null() {
            return None;
        }
        // SAFETY: `codec` is non-null and static, as above.
        unsafe {
            let experimental = (*codec).capabilities & ffi::AV_CODEC_CAP_EXPERIMENTAL as i32 != 0;
            if (*codec).id == id
                && ffi::av_codec_is_decoder(codec) != 0
                && !experimental
                && decodes_on(codec, device)
            {
                return Some(ffmpeg::Codec::wrap(codec));
            }
        }
    }
}

/// Whether `codec` can be handed a device context of `device`'s kind to
/// decode with, which is the only way the hardware decoders here set their
/// hardware up.
///
/// # Safety
///
/// `codec` must be a valid, non-null `AVCodec`.
pub(super) unsafe fn decodes_on(codec: *const ffi::AVCodec, device: ffi::AVHWDeviceType) -> bool {
    let mut index = 0;
    loop {
        // SAFETY: the caller's promise; past the last configuration this
        // returns null rather than reading out of bounds.
        let config = unsafe { ffi::avcodec_get_hw_config(codec, index) };
        if config.is_null() {
            return false;
        }
        // SAFETY: non-null, and static like the codec it belongs to.
        let (methods, device_type) = unsafe { ((*config).methods, (*config).device_type) };
        if methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0
            && device_type == device
        {
            return true;
        }
        index += 1;
    }
}

/// Whether a hardware decoder's `get_format` was left with no hardware
/// format it could use — the GPU refusing the stream.
///
/// Nothing else says so. A GPU without the stream's profile fails FFmpeg's
/// hwaccel setup inside format negotiation; FFmpeg then asks `get_format`
/// again without the hardware format, the callback has nothing to answer
/// but `AV_PIX_FMT_NONE`, and what reaches the caller is `EPERM` and then
/// `AVERROR_INVALIDDATA` — measured with VP9 profile 1 against D3D11VA —
/// which is also what a damaged stream says. The callback is the one place
/// that knows, so it leaves its answer here, found through
/// `AVCodecContext::opaque`.
///
/// Boxed so its address survives the decoder that owns it being moved.
pub(super) struct NegotiationRefusal(Box<std::sync::atomic::AtomicBool>);

impl NegotiationRefusal {
    pub(super) fn new() -> Self {
        Self(Box::default())
    }

    /// What to store in `AVCodecContext::opaque` before the codec opens.
    /// Valid for as long as `self` is, which the decoder owning both makes
    /// at least as long as the codec context.
    pub(super) fn opaque(&self) -> *mut std::ffi::c_void {
        std::ptr::from_ref(&*self.0).cast_mut().cast()
    }

    /// Records the refusal from inside `get_format`.
    ///
    /// # Safety
    ///
    /// `ctx` must be a live codec context whose `opaque` is null or was set
    /// from [`Self::opaque`] of a refusal that is still alive.
    pub(super) unsafe fn mark(ctx: *const ffi::AVCodecContext) {
        // SAFETY: the caller's promise about `ctx` and what `opaque` holds.
        unsafe {
            let flag = (*ctx).opaque as *const std::sync::atomic::AtomicBool;
            if !flag.is_null() {
                (*flag).store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Whether `get_format` has refused since the codec opened.
    pub(super) fn happened(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Where a hardware decoder copies the pictures it holds while playing
/// backwards — see [`crate::element::ReversibleDecoder`]: a frames context
/// on the decoder's own device, shaped like the one it decodes into but its
/// own. A stretch is more pictures than a decoder's pool has surfaces, and
/// what is held must not keep them from the rest of the stretch.
///
/// Built on the first copy, and again when the stream's shape changes. The
/// copies come from FFmpeg's own pool, so what they are handed to takes
/// them as it takes the decoder's frames: same device, same kind of frame.
#[cfg(any(feature = "cuda", all(target_os = "windows", feature = "d3d12")))]
#[derive(Default)]
pub(super) struct CopyFrames {
    frames: Option<(crate::platform::ffmpeg::AvBufferRef, CopyShape)>,
}

/// A frames context's hardware format, software format and size.
#[cfg(any(feature = "cuda", all(target_os = "windows", feature = "d3d12")))]
type CopyShape = (ffi::AVPixelFormat, ffi::AVPixelFormat, i32, i32);

#[cfg(any(feature = "cuda", all(target_os = "windows", feature = "d3d12")))]
impl CopyFrames {
    /// A frame of this pool shaped like `frame`, a decoded hardware frame,
    /// at its visible size, its pixels not yet written. `initial` surfaces
    /// are made up front: none where the device's pool grows as asked.
    pub(super) fn frame_like(
        &mut self,
        frame: &ffmpeg::frame::Video,
        initial: i32,
    ) -> Result<ffmpeg::frame::Video, ffmpeg::Error> {
        // SAFETY: a decoded hardware frame's `hw_frames_ctx` is a live
        // `AVHWFramesContext`, and its `device_ref` is live as long as it
        // is; the context allocated here is wrapped before anything can
        // fail, and its fields are the ones `av_hwframe_ctx_init` reads.
        unsafe {
            let source = (*frame.as_ptr()).hw_frames_ctx;
            if source.is_null() {
                return Err(ffmpeg::Error::InvalidData);
            }
            let source = &*((*source).data as *const ffi::AVHWFramesContext);
            let shape = (source.format, source.sw_format, source.width, source.height);
            let frames = match self.frames.take() {
                Some((frames, built)) if built == shape => frames,
                _ => {
                    let frames = crate::platform::ffmpeg::AvBufferRef::from_raw(
                        ffi::av_hwframe_ctx_alloc(source.device_ref),
                    )
                    .ok_or(ffmpeg::Error::Other {
                        errno: ffmpeg::error::ENOMEM,
                    })?;
                    let context = (*frames.as_ptr()).data as *mut ffi::AVHWFramesContext;
                    (*context).format = shape.0;
                    (*context).sw_format = shape.1;
                    (*context).width = shape.2;
                    (*context).height = shape.3;
                    (*context).initial_pool_size = initial;
                    let code = ffi::av_hwframe_ctx_init(frames.as_ptr());
                    if code < 0 {
                        return Err(ffmpeg::Error::from(code));
                    }
                    frames
                }
            };
            let mut copy = ffmpeg::frame::Video::empty();
            let code = ffi::av_hwframe_get_buffer(frames.as_ptr(), copy.as_mut_ptr(), 0);
            self.frames = Some((frames, shape));
            if code < 0 {
                return Err(ffmpeg::Error::from(code));
            }
            (*copy.as_mut_ptr()).width = (*frame.as_ptr()).width;
            (*copy.as_mut_ptr()).height = (*frame.as_ptr()).height;
            Ok(copy)
        }
    }
}
