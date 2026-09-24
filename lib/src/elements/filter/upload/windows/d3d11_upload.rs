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
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::filter::upload::nv12,
    error::Result,
    pad::SrcPad,
    platform::windows::{d3d11_gpu::D3d11Gpu, d3d11va::wrap_d3d11_texture},
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
};

/// Errors specific to `D3d11Upload`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11UploadError {
    /// Creating or updating a D3D11 texture failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// FFmpeg could not take a second reference to the upload already in
    /// hand, which is how an unchanged CPU picture is answered.
    #[error("failed to reference the previous upload (code {0})")]
    FrameRef(i32),
    /// The CPU pixel format cannot be represented by this uploader.

    #[error(
        "D3d11Upload only accepts Pixel::NV12, Pixel::YUV420P and Pixel::BGRA \
         frames (chain a SwScaler in front of it), got {0:?}"
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
    /// The sink received a buffer other than decoded video or end-of-stream.

    #[error("D3d11Upload only handles Video frames, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Uploads CPU-resident `Pixel::NV12`, `Pixel::YUV420P` (and `YUVJ420P`) and
/// `Pixel::BGRA` video frames to GPU-resident `Video` frames tagged
/// `Pixel::D3D11` — the D3D11 sibling of
/// `D3d12Upload`, for a pipeline built entirely on one
/// shared `ID3D11Device` (see [`crate::elements::D3d11Renderer`]'s own
/// docs on why). A YUV420P frame — what a software decode usually gives —
/// goes up as NV12, its two chroma planes interleaved on the way: the same
/// samples, so no scaler is needed for it. Chain a
/// [`crate::elements::SwScaler`] in front of this if the source produces
/// anything else.
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
/// Each texture is made for the frame it holds, so a source that changes
/// resolution mid-stream simply produces differently sized textures from
/// then on — there is nothing here allocated ahead of a frame to disagree
/// with it.
pub struct D3d11Upload {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D11Device,
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
    /// The last upload and the CPU picture it came from, so a producer that
    /// re-emits an unchanged picture — `DxgiCaptureSource` under
    /// [`CaptureMode::Cpu`](crate::elements::CaptureMode::Cpu) does exactly
    /// that on a still screen — is answered with the texture already on the
    /// GPU, instead of a fresh full-size allocation and a copy across PCIe
    /// per frame. See [`RepeatedOutput`].
    repeated: RepeatedOutput,
}

impl D3d11Upload {
    /// `gpu` must be the [`D3d11Gpu`] every other D3D11 element in this
    /// pipeline shares — see its own docs on why. The element keeps the
    /// device alive for as long as it, and every frame it produced that is
    /// still alive downstream, needs it.
    pub fn new(name: impl Into<String>, gpu: &D3d11Gpu) -> Self {
        let device = gpu.device();
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Upload, &name, None);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::SameLayout(
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11)
                    .with_layouts(crate::contract::PixelLayoutSet::NV12_OR_BGRA),
            ),
        );
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened");
        Self {
            name,
            pp_log,
            device: device.clone(),
            pad,
            pool,
            repeated: RepeatedOutput::new(),
        }
    }

    /// Repacks an NV12 frame into the single buffer D3D11 wants. NV12 is
    /// one GPU resource covering both planes, and D3D11 expects the source
    /// laid out as the full-height luma rows immediately followed by the
    /// half-height interleaved-chroma rows, all sharing one row pitch —
    /// whereas `frame`'s two planes are separately allocated and
    /// independently strided.
    ///
    /// A YUV420P frame is packed the same way, its separate Cb and Cr planes
    /// interleaved into the chroma rows NV12 has.
    fn pack_nv12(frame: &ffmpeg::frame::Video) -> std::result::Result<Vec<u8>, D3d11UploadError> {
        let row_bytes = frame.width() as usize;
        let luma_rows = frame.height() as usize;
        let chroma_rows = frame.height().div_ceil(2) as usize;
        let planar = nv12::is_planar_420(frame.format());
        // Each plane's rows are read at its stride, so every plane is checked
        // to hold them first: a short plane would be a panic here, not an
        // error returned to the caller.
        nv12::check_planes(frame).map_err(
            |nv12::PlaneTooSmall {
                 actual,
                 stride,
                 height,
             }| D3d11UploadError::PlaneTooSmall {
                actual,
                stride,
                height,
            },
        )?;
        let mut packed = vec![0u8; row_bytes * (luma_rows + chroma_rows)];
        let (luma_stride, luma_src) = (frame.stride(0), frame.data(0));
        for row in 0..luma_rows {
            packed[row * row_bytes..row * row_bytes + row_bytes]
                .copy_from_slice(&luma_src[row * luma_stride..row * luma_stride + row_bytes]);
        }
        let chroma_offset = row_bytes * luma_rows;
        if planar {
            for row in 0..chroma_rows {
                let destination = &mut packed[chroma_offset + row * row_bytes..][..row_bytes];
                nv12::interleave_chroma_row(frame, row, destination);
            }
        } else {
            let (chroma_stride, chroma_src) = (frame.stride(1), frame.data(1));
            for row in 0..chroma_rows {
                let dst = chroma_offset + row * row_bytes;
                packed[dst..dst + row_bytes].copy_from_slice(
                    &chroma_src[row * chroma_stride..row * chroma_stride + row_bytes],
                );
            }
        }
        Ok(packed)
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
        let (width, height) = (frame.width(), frame.height());
        let packed = (format == DXGI_FORMAT_NV12)
            .then(|| Self::pack_nv12(frame))
            .transpose()?;
        let (pixels, pitch) = match &packed {
            Some(packed) => (packed.as_slice(), width as usize),
            None => (frame.data(0), frame.stride(0)),
        };
        // D3D11 reads `pitch` bytes for each of the texture's rows straight
        // out of this pointer. The NV12 buffer was just sized here, but a
        // BGRA frame's plane belongs to whoever allocated it, so its extent
        // is checked rather than assumed — a short buffer would be an
        // out-of-bounds read inside the driver, not a Rust panic.
        let rows = match &packed {
            Some(_) => height as usize + height.div_ceil(2) as usize,
            None => height as usize,
        };
        if pixels.len() < pitch * rows {
            return Err(D3d11UploadError::PlaneTooSmall {
                actual: pixels.len(),
                stride: pitch,
                height: rows as u32,
            });
        }

        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
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
        ffmpeg::format::Pixel::NV12
        | ffmpeg::format::Pixel::YUV420P
        | ffmpeg::format::Pixel::YUVJ420P => Some(DXGI_FORMAT_NV12),
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
    /// CPU-readable planes specifically: uploading is what this element does, so a frame already on the device has no work here.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System).with_layouts(
                crate::contract::PixelLayoutSet::from_slice(&[
                    crate::contract::PixelLayout::Nv12,
                    crate::contract::PixelLayout::Yuv420p,
                    crate::contract::PixelLayout::Bgra,
                ]),
            ),
        )
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            // The same CPU buffer as last time is the same pixels as last
            // time, and uploading them again produces the texture already on
            // the GPU — see [`PerFrameTransform`], which is where that is
            // decided.
            MediaBuffer::Video(frame) => {
                let uploaded = self.transform(&frame)?;
                self.pad.push(MediaBuffer::Video(uploaded))
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
        // Nothing local to react to beyond the cached upload — a pure
        // per-frame CPU->GPU transfer, same reasoning as
        // `D3d12Upload::control`.
        if matches!(msg, ControlMsg::Flush | ControlMsg::Stop) {
            self.repeated.clear();
        }
        self.pad.control(msg)
    }
}

impl PerFrameTransform for D3d11Upload {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        D3d11UploadError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        let Some(format) = texture_format(frame.format()) else {
            pp_error!(self, "unsupported pixel format: {:?}", frame.format());
            return Err(D3d11UploadError::UnsupportedFormat(frame.format()).into());
        };
        let texture = self
            .upload(frame, format)
            .inspect_err(|error| pp_error!(self, "GPU upload failed: {error}"))?;
        let mut gpu_frame = self.pool.get();
        // Overwrites the pooled slot's previous contents in place —
        // `ffmpeg::frame::Video`'s own `Drop` runs on whatever was there
        // before, releasing that frame's GPU texture (via
        // `release_d3d11_texture`) right here.
        *gpu_frame = wrap_d3d11_texture(texture, frame.width(), frame.height())?;
        crate::buffer::carry_timing(&mut gpu_frame, frame);
        gpu_frame.set_color_space(frame.color_space());
        // The J of YUVJ420P is its range, which not every producer also says
        // in the range field; on NV12 only the field can say it.
        gpu_frame.set_color_range(match frame.format() {
            ffmpeg::format::Pixel::YUVJ420P => ffmpeg::color::Range::JPEG,
            _ => frame.color_range(),
        });
        gpu_frame.set_color_primaries(frame.color_primaries());
        gpu_frame.set_color_transfer_characteristic(frame.color_transfer_characteristic());
        Ok(gpu_frame)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::capture;

    use windows::{
        Win32::Graphics::Direct3D11::{
            D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_USAGE_STAGING,
            ID3D11DeviceContext,
        },
        core::Interface,
    };

    use super::*;
    use crate::{
        element::Sink, platform::windows::d3d11va::d3d11va_texture, pool::UnboundObjectPool,
        test_support::try_d3d11_gpu,
    };

    /// One pooled CPU frame, as an upstream element hands one over.
    fn cpu_frame(width: u32, height: u32, pts: i64) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
        frame.set_pts(Some(pts));
        frame.data_mut(0).fill(0x40);
        MediaBuffer::video(frame)
    }

    /// Another `AVFrame` over the same picture, with its own timestamp —
    /// what `DxgiCaptureSource` under `CaptureMode::Cpu` hands over on every
    /// tick of a still screen.
    fn repeat_of(buffer: &MediaBuffer, pts: i64) -> MediaBuffer {
        let MediaBuffer::Video(source) = buffer else {
            panic!("expected a Video buffer");
        };
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        // SAFETY: both are live `AVFrame`s and distinct — the slot is the
        // empty one just taken from the pool.
        unsafe {
            assert!(ffmpeg::ffi::av_frame_ref(slot.as_mut_ptr(), source.as_ptr()) >= 0);
        }
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    /// The texture a frame's pixels live in.
    fn texture_of(buffer: &MediaBuffer) -> *mut std::ffi::c_void {
        let MediaBuffer::Video(frame) = buffer else {
            panic!("expected a Video buffer");
        };
        d3d11va_texture(frame).expect("a D3D11 frame").0
    }

    /// A capture of a still screen re-emits the picture it already has, and
    /// uploading it again would produce the texture already on the GPU —
    /// another full-size allocation and another crossing of PCIe. The
    /// repeat carries its own timestamp; only the texture is shared.
    #[test]
    fn a_repeated_input_is_uploaded_once() {
        let Some(gpu) = try_d3d11_gpu() else {
            return;
        };
        let mut upload = D3d11Upload::new("upload", &gpu);
        let received = capture(&mut upload);
        let source = cpu_frame(4, 4, 100);
        let repeat = repeat_of(&source, 200);

        upload.consume(source).expect("upload the first frame");
        upload.consume(repeat).expect("upload the repeat");

        let received = received.lock().unwrap();
        assert_eq!(received.len(), 2, "every tick still produces a frame");
        assert_eq!(
            texture_of(&received[1]),
            texture_of(&received[0]),
            "an unchanged picture was uploaded a second time"
        );
        let MediaBuffer::Video(repeated) = &received[1] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(
            repeated.pts(),
            Some(200),
            "a repeat carries this frame's timestamp, not the one it points at"
        );
    }

    /// Every part of what a frame says of its colour survives the upload:
    /// primaries and transfer are what tell a BT.2020 HDR picture from an
    /// SDR one downstream, and a presenter is handed all four.
    #[test]
    fn an_uploaded_frame_keeps_its_whole_colour_description() {
        use crate::{color::ColorDescription, repeat::PerFrameTransform};

        let Some(gpu) = try_d3d11_gpu() else {
            return;
        };
        let mut picture = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 16, 16);
        let hdr = ColorDescription {
            space: ffmpeg::color::Space::BT2020NCL,
            range: ffmpeg::color::Range::MPEG,
            primaries: ffmpeg::color::Primaries::BT2020,
            transfer: ffmpeg::color::TransferCharacteristic::SMPTE2084,
        };
        hdr.describe(&mut picture);
        let MediaBuffer::Video(picture) = MediaBuffer::video(picture) else {
            unreachable!()
        };
        let uploaded = D3d11Upload::new("upload", &gpu)
            .transform(&picture)
            .expect("the frame uploads");
        assert_eq!(ColorDescription::of(&uploaded), hdr);
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
        let Some(gpu) = try_d3d11_gpu() else {
            return;
        };
        let device = gpu.device().clone();
        let context = gpu.context();
        let (width, height) = (16u32, 16u32);
        let mut upload = D3d11Upload::new("test-upload", &gpu);
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
        let Some(gpu) = try_d3d11_gpu() else {
            return;
        };
        let (width, height) = (16u32, 16u32);
        let mut upload = D3d11Upload::new("test-upload", &gpu);
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

    /// A source that changes resolution mid-stream — an RTSP camera, a
    /// window capture being resized — was a per-frame error here while the
    /// texture size was settled before any frame had arrived.
    #[test]
    fn a_source_that_changes_resolution_is_followed() {
        let Some(gpu) = try_d3d11_gpu() else {
            return;
        };
        let mut upload = D3d11Upload::new("test-upload", &gpu);
        let received = capture(&mut upload);

        upload
            .consume(cpu_frame(16, 16, 1))
            .expect("the first size");
        upload
            .consume(cpu_frame(8, 8, 2))
            .expect("and the next one");

        let received = received.lock().unwrap();
        let sizes: Vec<(u32, u32)> = received
            .iter()
            .map(|buffer| {
                let MediaBuffer::Video(frame) = buffer else {
                    panic!("expected a Video buffer");
                };
                let (texture_raw, _) = d3d11va_texture(frame).expect("a D3D11 frame");
                // SAFETY: the live frame owns `texture_raw`; cloning the
                // borrowed COM wrapper acquires an independent reference,
                // and `desc` is a live out-parameter for it.
                unsafe {
                    let texture = ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                        .expect("the texture pointer must not be null")
                        .clone();
                    let mut desc = D3D11_TEXTURE2D_DESC::default();
                    texture.GetDesc(&mut desc);
                    (desc.Width, desc.Height)
                }
            })
            .collect();
        assert_eq!(
            sizes,
            [(16, 16), (8, 8)],
            "each texture is made for the frame it holds"
        );
    }

    #[test]
    fn a_format_neither_path_handles_is_a_typed_error_not_a_panic() {
        let Some(gpu) = try_d3d11_gpu() else {
            return;
        };
        let (width, height) = (16u32, 16u32);
        let mut upload = D3d11Upload::new("test-upload", &gpu);

        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, width, height),
            |_| {},
        );
        let error = upload
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("RGB24 must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11UploadError(D3d11UploadError::UnsupportedFormat(
                    ffmpeg::format::Pixel::RGB24
                ))
            ),
            "unexpected error: {error}"
        );
    }
}
