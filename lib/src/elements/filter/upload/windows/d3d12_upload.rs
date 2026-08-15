use std::sync::Arc;

use crate::clog::{CLog, cerror, cinfo};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_clog},
    elements::filter::decoder::d3d12va_decoder::{create_hw_device_ctx, free_buffer},
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// How many GPU frames [`D3d12Upload`]'s `hw_frames_ctx` pool starts with.
/// Unlike [`crate::pool::UnboundObjectPool`], a D3D12VA frames context
/// can't necessarily grow past its `initial_pool_size` once
/// `av_hwframe_ctx_init` has run, so this needs to comfortably cover
/// however many uploaded frames can legitimately be in flight at once
/// (this element's own reused `AVFrame` plus whatever's sitting in a
/// downstream `Queue`). Unlike [`crate::elements::Scaler`]'s growable
/// object pool, this value can be an effective hard limit for the FFmpeg
/// implementation in use; downstream queue capacity and renderer-held
/// frames must be sized with that in mind.
const POOL_SIZE: i32 = 4;

/// Errors specific to `D3d12Upload`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12UploadError {
    #[error("failed to create D3D12VA hw device context (code {0})")]
    HwDeviceInit(i32),

    #[error("failed to create D3D12VA hw frames context (code {0})")]
    HwFramesInit(i32),

    #[error("failed to allocate a GPU frame (code {0})")]
    GetBuffer(i32),

    #[error("failed to upload frame to the GPU (code {0})")]
    TransferData(i32),

    #[error(
        "D3d12Upload only accepts Pixel::NV12 frames (chain a Scaler in \
         front of it), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "frame is {actual_width}x{actual_height}, but D3d12Upload was \
         opened for {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    #[error("D3d12Upload only handles Video frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Uploads CPU-resident `Pixel::NV12` video frames (e.g. from
/// [`crate::elements::Scaler`], fed by a synthetic source, a screen
/// capture, ...) to GPU-resident `Video` frames tagged `Pixel::D3D12` —
/// the mirror image of [`crate::elements::D3d12vaDecoder`]: that one
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
/// [`crate::elements::Scaler`] (`dst_format = Pixel::NV12`) in front of
/// this if the source produces something else (e.g.
/// [`crate::elements::TestVideoSource`]'s `Pixel::YUV420P`).
///
/// `width`/`height` are fixed for this element's lifetime, set once in
/// [`D3d12Upload::new`] — the underlying `AVHWFramesContext`'s allocated
/// dimensions can't be changed after `av_hwframe_ctx_init`, so every frame
/// `consume` receives must match exactly (a resolution change means
/// tearing this element down and building a fresh one).
pub struct D3d12Upload {
    clog: CLog,
    name: Arc<str>,
    hw_device_ctx: *mut ffi::AVBufferRef,
    hw_frames_ctx: *mut ffi::AVBufferRef,
    width: u32,
    height: u32,
    pad: SrcPad,
    /// Reused across every uploaded frame — see [`UnboundObjectPool`]'s
    /// docs. Only the small CPU-side `AVFrame` wrapper is actually reused
    /// here; the GPU texture behind it comes fresh from `hw_frames_ctx`'s
    /// own pool every time (`consume` unrefs the previous one first) —
    /// same division of labor [`crate::elements::D3d12vaDecoder`]'s own
    /// `pool` field docs describe.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx`/`hw_frames_ctx` are heap-allocated FFmpeg
// buffers with no thread affinity; `&mut self` on every method that
// touches them rules out concurrent access — same reasoning as
// `D3d12vaDecoder`'s own `unsafe impl Send`.
unsafe impl Send for D3d12Upload {}

impl D3d12Upload {
    /// `device` must outlive this element (and, transitively, every frame
    /// it produces that's still alive downstream) and must be the same
    /// `ID3D12Device` your [`crate::elements::D3d12Renderer`] was created
    /// with — same requirement [`crate::elements::D3d12vaDecoder::new`]
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
        let clog = element_clog(ElementType::D3d12Upload, &name, None);

        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d12UploadError::HwDeviceInit)?;

        let hw_frames_ctx = match unsafe { create_hw_frames_ctx(hw_device_ctx, width, height) } {
            Ok(ctx) => ctx,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                return Err(D3d12UploadError::HwFramesInit(error));
            }
        };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        cinfo!(clog: &clog, "opened: {width}x{height}");
        Ok(Self {
            name,
            clog,
            hw_device_ctx,
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

    fn clog(&self) -> &CLog {
        &self.clog
    }

    fn clog_mut(&mut self) -> &mut CLog {
        &mut self.clog
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
                    cerror!(self, "unsupported pixel format: {:?}", frame.format());
                    return Err(D3d12UploadError::UnsupportedFormat(frame.format()).into());
                }
                if frame.width() != self.width || frame.height() != self.height {
                    let error = D3d12UploadError::DimensionMismatch {
                        actual_width: frame.width(),
                        actual_height: frame.height(),
                        expected_width: self.width,
                        expected_height: self.height,
                    };
                    cerror!(self, "{error}");
                    return Err(error.into());
                }

                let mut gpu_frame = self.pool.get();
                unsafe {
                    // `av_hwframe_get_buffer` requires an "empty (freshly
                    // allocated or unreffed)" frame — a reused pool item
                    // may still hold a reference to the GPU texture it was
                    // last uploaded into, so drop that first.
                    ffi::av_frame_unref(gpu_frame.as_mut_ptr());
                    let ret =
                        ffi::av_hwframe_get_buffer(self.hw_frames_ctx, gpu_frame.as_mut_ptr(), 0);
                    if ret < 0 {
                        cerror!(self, "av_hwframe_get_buffer failed: {ret}");
                        return Err(D3d12UploadError::GetBuffer(ret).into());
                    }
                    let ret =
                        ffi::av_hwframe_transfer_data(gpu_frame.as_mut_ptr(), frame.as_ptr(), 0);
                    if ret < 0 {
                        cerror!(self, "av_hwframe_transfer_data failed: {ret}");
                        return Err(D3d12UploadError::TransferData(ret).into());
                    }
                }
                gpu_frame.set_pts(frame.pts());

                self.pad.push(MediaBuffer::Video(Arc::new(gpu_frame)))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => {
                cerror!(self, "unsupported buffer: Packet");
                Err(D3d12UploadError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                cerror!(self, "unsupported buffer: Audio");
                Err(D3d12UploadError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame CPU->GPU transfer,
        // same reasoning as `Scaler::control`.
        self.pad.control(msg)
    }
}

impl Drop for D3d12Upload {
    fn drop(&mut self) {
        cinfo!(self, "dropped: freeing hw_frames_ctx/hw_device_ctx");
        unsafe {
            free_buffer(self.hw_frames_ctx);
            free_buffer(self.hw_device_ctx);
        }
    }
}

/// Allocates and initializes an `AVHWFramesContext` (D3D12VA) tied to
/// `hw_device_ctx` — `Pixel::NV12` frames at `width`x`height`, the layout
/// [`crate::elements::D3d12Renderer`]'s zero-copy path requires.
unsafe fn create_hw_frames_ctx(
    hw_device_ctx: *mut ffi::AVBufferRef,
    width: u32,
    height: u32,
) -> std::result::Result<*mut ffi::AVBufferRef, i32> {
    unsafe {
        let buf = ffi::av_hwframe_ctx_alloc(hw_device_ctx);
        if buf.is_null() {
            return Err(-1);
        }

        let frames_ctx = (*buf).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D12;
        (*frames_ctx).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_NV12;
        (*frames_ctx).width = width as i32;
        (*frames_ctx).height = height as i32;
        (*frames_ctx).initial_pool_size = POOL_SIZE;

        let result = ffi::av_hwframe_ctx_init(buf);
        if result < 0 {
            free_buffer(buf);
            return Err(result);
        }

        Ok(buf)
    }
}
