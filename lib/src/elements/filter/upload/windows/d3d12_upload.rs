use std::sync::Arc;

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        windows::d3d12va::{create_hw_device_ctx, create_hw_frames_ctx},
    },
    pool::UnboundObjectPool,
};

/// How many GPU frames [`D3d12Upload`]'s `hw_frames_ctx` pool starts with.
/// Unlike [`crate::pool::UnboundObjectPool`], a D3D12VA frames context
/// can't necessarily grow past its `initial_pool_size` once
/// `av_hwframe_ctx_init` has run, so this needs to comfortably cover
/// however many uploaded frames can legitimately be in flight at once
/// (this element's own reused `AVFrame` plus whatever's sitting in a
/// downstream `Queue`). Unlike [`crate::elements::SwScaler`]'s growable
/// object pool, this value can be an effective hard limit for the FFmpeg
/// implementation in use; downstream queue capacity and renderer-held
/// frames must be sized with that in mind.
const POOL_SIZE: i32 = 4;

/// Errors specific to `D3d12Upload`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12UploadError {
    /// FFmpeg could not wrap the supplied D3D12 device.
    #[error("failed to create D3D12VA hw device context (code {0})")]
    HwDeviceInit(i32),
    /// FFmpeg could not initialize the D3D12 hardware frame pool.

    #[error("failed to create D3D12VA hw frames context (code {0})")]
    HwFramesInit(i32),
    /// FFmpeg could not acquire a texture from the D3D12 frame pool.

    #[error("failed to allocate a GPU frame (code {0})")]
    GetBuffer(i32),
    /// FFmpeg failed to transfer CPU pixels into the D3D12 texture.

    #[error("failed to upload frame to the GPU (code {0})")]
    TransferData(i32),
    /// FFmpeg could not copy timing and color metadata to the GPU frame.

    #[error("failed to copy uploaded frame metadata (code {0})")]
    CopyProperties(i32),
    /// The CPU frame format is not the required NV12 layout.

    #[error(
        "D3d12Upload only accepts Pixel::NV12 frames (chain a SwScaler in \
         front of it), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// Input dimensions differ from the fixed upload dimensions.

    #[error(
        "frame is {actual_width}x{actual_height}, but D3d12Upload was \
         opened for {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Input frame width in pixels.
        actual_width: u32,
        /// Input frame height in pixels.
        actual_height: u32,
        /// Width configured for this uploader.
        expected_width: u32,
        /// Height configured for this uploader.
        expected_height: u32,
    },
    /// The sink received a buffer other than decoded video or end-of-stream.

    #[error("D3d12Upload only handles Video frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Uploads CPU-resident `Pixel::NV12` video frames (e.g. from
/// [`crate::elements::SwScaler`], fed by a synthetic source, a screen
/// capture, ...) to GPU-resident `Video` frames tagged `Pixel::D3D12` —
/// the mirror image of [`crate::elements::D3d12Decoder`]: that one
/// decodes compressed `Packet`s straight to GPU frames; this one moves
/// frames that start out on the CPU onto the same device a
/// [`crate::elements::D3d12Renderer`] reads from, so the renderer can take
/// its zero-copy path (`Pixel::D3D12`) instead of its CPU-upload one
/// (`Pixel::YUV420P`) — or so any other GPU-side stage downstream can work
/// on the frame without a CPU round trip.
///
/// Only accepts `Pixel::NV12` input — that's the layout
/// [`crate::elements::D3d12Renderer`]'s zero-copy `submit_nv12_texture`
/// path requires of every `Pixel::D3D12` frame, decoder-produced or not,
/// so there's no reason to support uploading anything else. Chain a
/// [`crate::elements::SwScaler`] (`dst_format = Pixel::NV12`) in front of
/// this if the source produces something else (e.g.
/// [`crate::elements::TestVideoSource`]'s `Pixel::YUV420P`).
///
/// `width`/`height` are fixed for this element's lifetime, set once in
/// [`D3d12Upload::new`] — the underlying `AVHWFramesContext`'s allocated
/// dimensions can't be changed after `av_hwframe_ctx_init`, so every frame
/// `consume` receives must match exactly (a resolution change means
/// tearing this element down and building a fresh one).
pub struct D3d12Upload {
    pp_log: PpLog,
    name: Arc<str>,
    _hw_device_ctx: AvBufferRef,
    hw_frames_ctx: AvBufferRef,
    width: u32,
    height: u32,
    pad: SrcPad,
    /// Reused across every uploaded frame — see [`UnboundObjectPool`]'s
    /// docs. Only the small CPU-side `AVFrame` wrapper is actually reused
    /// here; the GPU texture behind it comes fresh from `hw_frames_ctx`'s
    /// own pool every time (`consume` unrefs the previous one first) —
    /// same division of labor [`crate::elements::D3d12Decoder`]'s own
    /// `pool` field docs describe.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx`/`hw_frames_ctx` are heap-allocated FFmpeg
// buffers with no thread affinity; `&mut self` on every method that
// touches them rules out concurrent access — same reasoning as
// `D3d12Decoder`'s own `unsafe impl Send`.
unsafe impl Send for D3d12Upload {}

impl D3d12Upload {
    /// The D3D12VA hardware context owns an independent COM reference to
    /// `device`, so the caller does not need to keep its handle alive. It
    /// must be the same underlying `ID3D12Device` your
    /// [`crate::elements::D3d12Renderer`] was created
    /// with — same requirement [`crate::elements::D3d12Decoder::new`]
    /// documents, for the same reason: frames landing on a different
    /// device than the one the renderer submits to would make the
    /// zero-copy path invalid.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, D3d12UploadError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12Upload, &name, None);

        // SAFETY: `device` is live and the helper transfers a cloned COM
        // reference into the returned FFmpeg device context.
        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d12UploadError::HwDeviceInit)?;

        // SAFETY: `hw_device_ctx` is live and initialized; dimensions were
        // validated above and `POOL_SIZE` is a positive fixed allocation size.
        let hw_frames_ctx =
            unsafe { create_hw_frames_ctx(&hw_device_ctx, width, height, POOL_SIZE) }
                .map_err(D3d12UploadError::HwFramesInit)?;

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: {width}x{height}");
        Ok(Self {
            name,
            pp_log,
            _hw_device_ctx: hw_device_ctx,
            hw_frames_ctx,
            width,
            height,
            pad,
            pool,
        })
    }
}

impl Element for D3d12Upload {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12Upload
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d12Upload {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d12Upload {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if frame.format() != ffmpeg::format::Pixel::NV12 {
                    pp_error!(self, "unsupported pixel format: {:?}", frame.format());
                    return Err(D3d12UploadError::UnsupportedFormat(frame.format()).into());
                }
                if frame.width() != self.width || frame.height() != self.height {
                    let error = D3d12UploadError::DimensionMismatch {
                        actual_width: frame.width(),
                        actual_height: frame.height(),
                        expected_width: self.width,
                        expected_height: self.height,
                    };
                    pp_error!(self, "{error}");
                    return Err(error.into());
                }

                let mut gpu_frame = self.pool.get();
                // SAFETY: the pooled destination is exclusively owned and
                // unreffed before reuse. Its frames context remains live, and
                // the validated CPU source is readable for the transfer.
                unsafe {
                    // `av_hwframe_get_buffer` requires an "empty (freshly
                    // allocated or unreffed)" frame — a reused pool item
                    // may still hold a reference to the GPU texture it was
                    // last uploaded into, so drop that first.
                    ffi::av_frame_unref(gpu_frame.as_mut_ptr());
                    let ret = ffi::av_hwframe_get_buffer(
                        self.hw_frames_ctx.as_ptr(),
                        gpu_frame.as_mut_ptr(),
                        0,
                    );
                    if ret < 0 {
                        pp_error!(self, "av_hwframe_get_buffer failed: {ret}");
                        return Err(D3d12UploadError::GetBuffer(ret).into());
                    }
                    let ret =
                        ffi::av_hwframe_transfer_data(gpu_frame.as_mut_ptr(), frame.as_ptr(), 0);
                    if ret < 0 {
                        pp_error!(self, "av_hwframe_transfer_data failed: {ret}");
                        return Err(D3d12UploadError::TransferData(ret).into());
                    }
                }
                // SAFETY: both frames remain live, the destination is uniquely
                // owned, and `av_frame_copy_props` copies metadata only.
                unsafe {
                    // The transfer copies pixels only. Keep the source
                    // timeline and color description on the GPU frame so a
                    // later D3d12Download can restore the complete contract.
                    let ret = ffi::av_frame_copy_props(gpu_frame.as_mut_ptr(), frame.as_ptr());
                    if ret < 0 {
                        pp_error!(self, "av_frame_copy_props failed: {ret}");
                        return Err(D3d12UploadError::CopyProperties(ret).into());
                    }
                }

                self.pad.push(MediaBuffer::Video(Arc::new(gpu_frame)))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => {
                pp_error!(self, "unsupported buffer: Packet");
                Err(D3d12UploadError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(D3d12UploadError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame CPU->GPU transfer,
        // same reasoning as `SwScaler::control`.
        self.pad.control(msg)
    }
}

impl Drop for D3d12Upload {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_frames_ctx/hw_device_ctx");
    }
}
