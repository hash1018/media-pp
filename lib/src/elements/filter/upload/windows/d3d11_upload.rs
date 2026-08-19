use std::{ffi::c_void, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC,
        D3D11_USAGE_DEFAULT, ID3D11Device, ID3D11Texture2D,
    },
    Dxgi::Common::{DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC},
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::filter::decoder::d3d11va_decoder::wrap_d3d11_texture,
    error::Result,
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// Errors specific to `D3d11Upload`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11UploadError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    #[error(
        "D3d11Upload only accepts Pixel::NV12 frames (chain a SwScaler in \
         front of it), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "frame is {actual_width}x{actual_height}, but D3d11Upload was \
         opened for {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    #[error("D3d11Upload only handles Video frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Uploads CPU-resident `Pixel::NV12` video frames to GPU-resident `Video`
/// frames tagged `Pixel::D3D11` — the D3D11 sibling of
/// [`crate::elements::D3d12Upload`], for a pipeline built entirely on one
/// shared `ID3D11Device` (see [`crate::elements::D3d11Renderer`]'s own
/// docs on why). Only accepts `Pixel::NV12` input — chain a
/// [`crate::elements::SwScaler`] (`dst_format = Pixel::NV12`) in front of
/// this if the source produces something else.
///
/// Unlike `D3d12Upload`, this does **not** go through FFmpeg's
/// `av_hwframe_get_buffer`/`av_hwframe_transfer_data` hwframe-pool
/// machinery at all — `consume` creates a plain `ID3D11Texture2D` directly
/// via ordinary `windows-rs` calls (with the CPU pixel data as its initial
/// contents) every call, then wraps it as a `Pixel::D3D11` frame via
/// `wrap_d3d11_texture`. See that function's own docs for why: driving
/// D3D11VA's real frames-context init from a hand-mirrored
/// `AVD3D11VAFramesContext*` corrupted memory in testing, for a reason not
/// fully root-caused even against FFmpeg's real (version-matched) source.
/// This is plain, well-understood D3D11 API usage instead — no
/// struct-layout guessing, at the cost of a fresh texture allocation per
/// frame rather than a pre-sized pool (not pooled/reused across calls the
/// way `D3d12Upload`'s GPU textures are, since there's no
/// `av_hwframe_ctx`-managed pool here to reuse from).
///
/// `width`/`height` are fixed for this element's lifetime, set once in
/// [`D3d11Upload::new`] — every frame `consume` receives must match
/// exactly.
pub struct D3d11Upload {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D11Device,
    width: u32,
    height: u32,
    pad: SrcPad,
    /// Reused across every uploaded frame — see [`UnboundObjectPool`]'s
    /// docs. Only the small CPU-side `AVFrame` wrapper is actually reused
    /// here (each `consume` call overwrites it in place with a freshly
    /// built [`wrap_d3d11_texture`] result, whose own `Drop` — via
    /// [`ffmpeg::frame::Video`]'s normal `Drop` — releases whatever GPU
    /// texture the pooled slot held before); every GPU texture itself is
    /// still a fresh allocation per frame (see [`D3d11Upload::upload`]'s
    /// own docs on why there's no GPU-side pool here to reuse from).
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl D3d11Upload {
    /// `device` must outlive this element (and, transitively, every frame
    /// it produces that's still alive downstream), and must be the same
    /// `ID3D11Device` every other D3D11 element in this pipeline shares —
    /// see [`crate::elements::D3d11Renderer`]'s own docs on why.
    pub fn new(name: impl Into<String>, device: &ID3D11Device, width: u32, height: u32) -> Self {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Upload, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: {width}x{height}");
        Self {
            name,
            pp_log,
            device: device.clone(),
            width,
            height,
            pad,
            pool,
        }
    }

    /// Builds one GPU `ID3D11Texture2D` (`DXGI_FORMAT_NV12`, `D3D11_USAGE_DEFAULT`,
    /// `D3D11_BIND_SHADER_RESOURCE` — enough for
    /// [`crate::elements::D3d11Renderer`] to build an SRV from, nothing
    /// decode-specific) with `frame`'s pixel data as its initial contents.
    /// NV12 is one GPU resource covering both planes — D3D11 expects the
    /// source buffer laid out as the full-height luma rows immediately
    /// followed by the half-height interleaved-chroma rows, all sharing
    /// one row pitch, which is why this copies both of `frame`'s own
    /// (independently strided) planes into one packed buffer first rather
    /// than uploading them separately.
    fn upload(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<ID3D11Texture2D, windows::core::Error> {
        let row_bytes = self.width as usize;
        let luma_rows = self.height as usize;
        let chroma_rows = self.height.div_ceil(2) as usize;
        let mut packed = vec![0u8; row_bytes * (luma_rows + chroma_rows)];
        let (luma_stride, chroma_stride) = (frame.stride(0), frame.stride(1));
        let (luma_src, chroma_src) = (frame.data(0), frame.data(1));
        for row in 0..luma_rows {
            packed[row * row_bytes..row * row_bytes + row_bytes]
                .copy_from_slice(&luma_src[row * luma_stride..row * luma_stride + row_bytes]);
        }
        let chroma_offset = row_bytes * luma_rows;
        for row in 0..chroma_rows {
            let dst = chroma_offset + row * row_bytes;
            packed[dst..dst + row_bytes]
                .copy_from_slice(&chroma_src[row * chroma_stride..row * chroma_stride + row_bytes]);
        }

        let desc = D3D11_TEXTURE2D_DESC {
            Width: self.width,
            Height: self.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_NV12,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let initial_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: packed.as_ptr() as *const c_void,
            SysMemPitch: row_bytes as u32,
            SysMemSlicePitch: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            self.device
                .CreateTexture2D(&desc, Some(&initial_data), Some(&mut texture))?;
        }
        Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
    }
}

impl Element for D3d11Upload {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11Upload
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11Upload {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11Upload {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                if frame.format() != ffmpeg::format::Pixel::NV12 {
                    pp_error!(self, "unsupported pixel format: {:?}", frame.format());
                    return Err(D3d11UploadError::UnsupportedFormat(frame.format()).into());
                }
                if frame.width() != self.width || frame.height() != self.height {
                    let error = D3d11UploadError::DimensionMismatch {
                        actual_width: frame.width(),
                        actual_height: frame.height(),
                        expected_width: self.width,
                        expected_height: self.height,
                    };
                    pp_error!(self, "{error}");
                    return Err(error.into());
                }

                let texture = self
                    .upload(&frame)
                    .inspect_err(|error| pp_error!(self, "GPU upload failed: {error}"))
                    .map_err(D3d11UploadError::from)?;
                let mut gpu_frame = self.pool.get();
                // Overwrites the pooled slot's previous contents in
                // place — `ffmpeg::frame::Video`'s own `Drop` runs on
                // whatever was there before, releasing that frame's GPU
                // texture (via `release_d3d11_texture`) right here.
                *gpu_frame = wrap_d3d11_texture(texture, self.width, self.height);
                gpu_frame.set_pts(frame.pts());
                gpu_frame.set_color_space(frame.color_space());
                gpu_frame.set_color_range(frame.color_range());

                self.pad.push(MediaBuffer::Video(Arc::new(gpu_frame)))
            }
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            MediaBuffer::Packet(_) => {
                pp_error!(self, "unsupported buffer: Packet");
                Err(D3d11UploadError::UnsupportedBuffer("Packet").into())
            }
            MediaBuffer::Audio(_) => {
                pp_error!(self, "unsupported buffer: Audio");
                Err(D3d11UploadError::UnsupportedBuffer("Audio").into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame CPU->GPU transfer,
        // same reasoning as `D3d12Upload::control`.
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use windows::Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{D3D11_SDK_VERSION, D3D11CreateDevice},
    };

    use super::*;
    use crate::pool::UnboundObjectPool;

    /// Real hardware round trip for the plain-`windows-rs` (not
    /// hand-mirrored-struct) upload path — see [`D3d11Upload`]'s own docs
    /// on why it's built this way.
    #[test]
    fn cpu_nv12_frame_round_trips_to_a_gpu_d3d11_frame() {
        let mut device = None;
        let result = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                Default::default(),
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                None,
            )
        };
        let Ok(()) = result else {
            eprintln!("skipping: D3D11CreateDevice failed on this machine: {result:?}");
            return;
        };
        let device = device.expect("D3D11CreateDevice succeeded without producing a device");

        let (width, height) = (16u32, 16u32);
        let mut upload = D3d11Upload::new("test-upload", &device, width, height);

        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height),
            |_| {},
        );
        let mut frame = pool.get();
        frame.data_mut(0).fill(16); // Y
        frame.data_mut(1).fill(128); // interleaved UV
        frame.set_pts(Some(42));

        upload
            .consume(MediaBuffer::Video(Arc::new(frame)))
            .expect("CPU NV12 -> GPU D3D11 upload should succeed on a working device");
    }
}
