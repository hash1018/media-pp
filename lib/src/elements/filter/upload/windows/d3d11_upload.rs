use std::{ffi::c_void, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC,
        D3D11_USAGE_DEFAULT, ID3D11Device, ID3D11Texture2D,
    },
    Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC},
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::windows::d3d11va::wrap_d3d11_texture,
    pool::UnboundObjectPool,
};

/// Errors specific to `D3d11Upload`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11UploadError {
    /// Creating or updating a D3D11 texture failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),
    /// The CPU pixel format cannot be represented by this uploader.

    #[error(
        "D3d11Upload only accepts Pixel::NV12 and Pixel::BGRA frames (chain a \
         SwScaler in front of it), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),
    /// A CPU frame plane is shorter than its stride and height require.

    #[error(
        "frame's plane holds {actual} bytes, too few for {height} rows of \
         stride {stride}; uploading it would read past the end of the buffer"
    )]
    PlaneTooSmall {
        /// Bytes actually available in the plane.
        actual: usize,
        /// Declared row stride in bytes.
        stride: usize,
        /// Number of rows that must be uploaded.
        height: u32,
    },
    /// Input dimensions differ from the fixed texture dimensions.

    #[error(
        "frame is {actual_width}x{actual_height}, but D3d11Upload was \
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

    #[error("D3d11Upload only handles Video frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Uploads CPU-resident `Pixel::NV12` and `Pixel::BGRA` video frames to
/// GPU-resident `Video` frames tagged `Pixel::D3D11` — the D3D11 sibling of
/// `D3d12Upload`, for a pipeline built entirely on one
/// shared `ID3D11Device` (see [`crate::elements::D3d11Renderer`]'s own
/// docs on why). Chain a [`crate::elements::SwScaler`] in front of this if
/// the source produces anything else.
///
/// # Which of the two to feed it
///
/// The texture's format follows the frame's rather than being configured:
/// there is exactly one right answer per frame, and asking a caller to
/// declare it alongside would only create a second thing to get wrong.
/// `NV12` is the format for a decode/encode path — it is what
/// [`crate::elements::D3d11Decoder`] produces and what a hardware encoder
/// wants. `BGRA` is the format for anything that composites: it carries an
/// alpha channel, so it is what a [`crate::elements::D3d11VideoCompositor`]
/// layer and [`crate::elements::D3d11ChromaKey`] work in, and it skips the
/// color conversion a YUV round trip would cost.
///
/// Which one a frame already is decides this, so the choice really belongs
/// to whatever produced it. To cross between the two once a frame is
/// already on the GPU, see [`crate::elements::D3d11Scaler`] — its
/// [`crate::elements::D3d11ScalerFormat`] converts on the video processor
/// without a CPU round trip.
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

    /// Repacks an NV12 frame into the single buffer D3D11 wants. NV12 is
    /// one GPU resource covering both planes, and D3D11 expects the source
    /// laid out as the full-height luma rows immediately followed by the
    /// half-height interleaved-chroma rows, all sharing one row pitch —
    /// whereas `frame`'s two planes are separately allocated and
    /// independently strided.
    fn pack_nv12(&self, frame: &ffmpeg::frame::Video) -> Vec<u8> {
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
        packed
    }

    /// Builds one GPU `ID3D11Texture2D` (`D3D11_USAGE_DEFAULT`,
    /// `D3D11_BIND_SHADER_RESOURCE` — enough for
    /// [`crate::elements::D3d11Renderer`] to build an SRV from, nothing
    /// decode-specific) with `frame`'s pixel data as its initial contents.
    ///
    /// A BGRA frame is handed over in place, with its own stride as the
    /// row pitch: it is a single plane, so unlike NV12 there is nothing to
    /// repack and no staging copy to pay for.
    fn upload(
        &self,
        frame: &ffmpeg::frame::Video,
        format: DXGI_FORMAT,
    ) -> std::result::Result<ID3D11Texture2D, D3d11UploadError> {
        let packed = (format == DXGI_FORMAT_NV12).then(|| self.pack_nv12(frame));
        let (pixels, pitch) = match &packed {
            Some(packed) => (packed.as_slice(), self.width as usize),
            None => (frame.data(0), frame.stride(0)),
        };
        // D3D11 reads `pitch` bytes for each of the texture's rows straight
        // out of this pointer. The NV12 buffer was just sized here, but a
        // BGRA frame's plane belongs to whoever allocated it, so its extent
        // is checked rather than assumed — a short buffer would be an
        // out-of-bounds read inside the driver, not a Rust panic.
        let rows = match &packed {
            Some(_) => self.height as usize + self.height.div_ceil(2) as usize,
            None => self.height as usize,
        };
        if pixels.len() < pitch * rows {
            return Err(D3d11UploadError::PlaneTooSmall {
                actual: pixels.len(),
                stride: pitch,
                height: rows as u32,
            });
        }

        let desc = D3D11_TEXTURE2D_DESC {
            Width: self.width,
            Height: self.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
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
            pSysMem: pixels.as_ptr() as *const c_void,
            SysMemPitch: pitch as u32,
            SysMemSlicePitch: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        // SAFETY: `initial_data` points into the live source frame for the
        // dimensions and pitch declared in `desc`; `texture` is a live
        // out-parameter and D3D copies the initialization before returning.
        unsafe {
            self.device
                .CreateTexture2D(&desc, Some(&initial_data), Some(&mut texture))?;
        }
        Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
    }
}

/// The DXGI format a CPU pixel format uploads into, or `None` for one this
/// element cannot upload at all.
fn texture_format(format: ffmpeg::format::Pixel) -> Option<DXGI_FORMAT> {
    match format {
        ffmpeg::format::Pixel::NV12 => Some(DXGI_FORMAT_NV12),
        ffmpeg::format::Pixel::BGRA => Some(DXGI_FORMAT_B8G8R8A8_UNORM),
        _ => None,
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
                let Some(format) = texture_format(frame.format()) else {
                    pp_error!(self, "unsupported pixel format: {:?}", frame.format());
                    return Err(D3d11UploadError::UnsupportedFormat(frame.format()).into());
                };
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
                    .upload(&frame, format)
                    .inspect_err(|error| pp_error!(self, "GPU upload failed: {error}"))?;
                let mut gpu_frame = self.pool.get();
                // Overwrites the pooled slot's previous contents in
                // place — `ffmpeg::frame::Video`'s own `Drop` runs on
                // whatever was there before, releasing that frame's GPU
                // texture (via `release_d3d11_texture`) right here.
                *gpu_frame = wrap_d3d11_texture(texture, self.width, self.height)?;
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
    use std::sync::Mutex;

    use windows::{
        Win32::Graphics::Direct3D11::{
            D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_USAGE_STAGING,
            ID3D11DeviceContext,
        },
        core::Interface,
    };

    use super::*;
    use crate::{
        element::{Element, ElementType, Sink, Source, element_pp_log},
        platform::windows::d3d11va::d3d11va_texture,
        pool::UnboundObjectPool,
        test_support::try_d3d11_device,
    };

    struct CapturingSink {
        pp_log: PpLog,
        received: Arc<Mutex<Vec<MediaBuffer>>>,
    }

    impl Element for CapturingSink {
        fn name(&self) -> Arc<str> {
            "capture".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CapturingSink {
        fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
            self.received.lock().unwrap().push(buf);
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let received = Arc::new(Mutex::new(Vec::new()));
        element.src_pads()[0].link(Box::new(CapturingSink {
            received: received.clone(),
            pp_log: element_pp_log(ElementType::Other, "capture", None),
        }));
        received
    }

    /// Reads a BGRA texture back through a staging copy, tightly packed —
    /// what makes "the pixels arrived intact" checkable rather than just
    /// "the API returned S_OK".
    fn read_bgra(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        texture: &ID3D11Texture2D,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
        };
        let mut staging = None;
        // SAFETY: `desc` describes a valid readback texture, no initial data
        // is supplied, and `staging` is a live COM out-parameter.
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut staging))
                .expect("CreateTexture2D(BGRA staging) failed");
        }
        let staging = staging.expect("CreateTexture2D succeeded without producing a texture");

        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        // SAFETY: source and staging have identical device/format/dimensions.
        // Successful `Map` keeps the pointer valid through `Unmap`, and each
        // copied row is bounded by its reported `RowPitch`.
        unsafe {
            context.CopyResource(&staging, texture);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .expect("Map(BGRA staging) failed");
            let stride = mapped.RowPitch as usize;
            let base = mapped.pData as *const u8;
            for row in 0..height as usize {
                let row_bytes =
                    std::slice::from_raw_parts(base.add(row * stride), width as usize * 4);
                pixels.extend_from_slice(row_bytes);
            }
            context.Unmap(&staging, 0);
        }
        pixels
    }

    /// The compositing path's format. A layer, a chroma key, and a renderer
    /// all work in BGRA, so an upload that could only produce NV12 forced a
    /// color round trip on anything headed for one of them.
    #[test]
    fn a_cpu_bgra_frame_uploads_to_a_bgra_texture_with_its_pixels_intact() {
        let Some((device, context)) = try_d3d11_device() else {
            return;
        };
        let (width, height) = (16u32, 16u32);
        let mut upload = D3d11Upload::new("test-upload", &device, width, height);
        let received = capture(&mut upload);

        let color = [10u8, 200, 30, 255]; // BGRA
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
            |_| {},
        );
        let mut frame = pool.get();
        let stride = frame.stride(0);
        {
            let data = frame.data_mut(0);
            for row in 0..height as usize {
                for column in 0..width as usize {
                    let offset = row * stride + column * 4;
                    data[offset..offset + 4].copy_from_slice(&color);
                }
            }
        }
        frame.set_pts(Some(42));

        upload
            .consume(MediaBuffer::Video(Arc::new(frame)))
            .expect("CPU BGRA -> GPU D3D11 upload should succeed on a working device");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(uploaded) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(uploaded.format(), ffmpeg::format::Pixel::D3D11);
        assert_eq!(uploaded.pts(), Some(42), "the upload dropped the pts");

        let (texture_raw, _) =
            d3d11va_texture(uploaded).expect("the uploaded frame carries a texture");
        // SAFETY: the live frame owns `texture_raw`; cloning the borrowed COM
        // wrapper acquires an independent reference.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("the texture pointer must not be null")
                .clone()
        };
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `desc` is a live out-parameter for the live texture.
        unsafe { texture.GetDesc(&mut desc) };
        assert_eq!(
            desc.Format, DXGI_FORMAT_B8G8R8A8_UNORM,
            "a BGRA frame must not be uploaded as anything else"
        );
        assert_eq!((desc.Width, desc.Height), (width, height));
        // The bind flag every shader-based D3D11 element needs to read it.
        assert_ne!(desc.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32, 0);

        let pixels = read_bgra(&device, &context.lock().unwrap(), &texture, width, height);
        for row in 0..height as usize {
            for column in 0..width as usize {
                let offset = (row * width as usize + column) * 4;
                assert_eq!(
                    &pixels[offset..offset + 4],
                    color,
                    "row {row}, column {column} did not survive the upload"
                );
            }
        }
    }

    /// The texture's format is taken from the frame, so the two supported
    /// inputs must land in different DXGI formats through the same element.
    #[test]
    fn an_nv12_frame_uploads_to_an_nv12_texture() {
        let Some((device, _context)) = try_d3d11_device() else {
            return;
        };
        let (width, height) = (16u32, 16u32);
        let mut upload = D3d11Upload::new("test-upload", &device, width, height);
        let received = capture(&mut upload);

        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height),
            |_| {},
        );
        let mut frame = pool.get();
        frame.data_mut(0).fill(16);
        frame.data_mut(1).fill(128);
        upload
            .consume(MediaBuffer::Video(Arc::new(frame)))
            .expect("CPU NV12 -> GPU D3D11 upload should succeed on a working device");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(uploaded) = &received[0] else {
            panic!("expected a Video buffer");
        };
        let (texture_raw, _) =
            d3d11va_texture(uploaded).expect("the uploaded frame carries a texture");
        // SAFETY: the live frame owns `texture_raw`; cloning the borrowed COM
        // wrapper acquires an independent reference.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("the texture pointer must not be null")
                .clone()
        };
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `desc` is a live out-parameter for the live texture.
        unsafe { texture.GetDesc(&mut desc) };
        assert_eq!(desc.Format, DXGI_FORMAT_NV12);
    }

    #[test]
    fn a_format_neither_path_handles_is_a_typed_error_not_a_panic() {
        let Some((device, _context)) = try_d3d11_device() else {
            return;
        };
        let (width, height) = (16u32, 16u32);
        let mut upload = D3d11Upload::new("test-upload", &device, width, height);

        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width, height),
            |_| {},
        );
        let error = upload
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("YUV420P must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11UploadError(D3d11UploadError::UnsupportedFormat(
                    ffmpeg::format::Pixel::YUV420P
                ))
            ),
            "unexpected error: {error}"
        );
    }
}
