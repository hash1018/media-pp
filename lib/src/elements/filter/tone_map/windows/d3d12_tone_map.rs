//! [`D3d12ToneMap`]: HDR D3D12 pictures brought to SDR NV12 by a compute
//! shader.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D12::*,
        Dxgi::Common::{
            DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_R8_UNORM,
            DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM,
        },
    },
    core::{Interface, s},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Attached, ElementType, Filter, Identity, Output},
    error::Result,
    frame_size::ForSize,
    platform::ffmpeg::AvBufferRef,
    platform::windows::{
        d3d12_compute::{
            Commands, Kernel, copy_plane, plane_view, plane_write_view, transition,
            writable_texture,
        },
        d3d12_gpu::D3d12Gpu,
        d3d12va::{
            create_hw_device_ctx, create_hw_frames_ctx_of, d3d12va_texture, set_d3d12va_fence_value,
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
    tone_map::ToneMap,
    transform::{FilterStage, filter_stage},
};

const SHADER: &[u8] = include_bytes!("../../../../shaders/d3d12/tone_map.hlsl");

/// Each run of the shader is 8 by 8 blocks of 2 by 2 pixels.
const GROUP: u32 = 8;

/// Errors specific to [`D3d12ToneMap`].
///
/// Converts into the crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d12ToneMapError {
    /// D3D12 failed: a shader, a resource or a submission.
    #[error(transparent)]
    D3d12(#[from] windows::core::Error),
    /// FFmpeg could not take a second reference to the frame already in
    /// hand, which is how an unchanged input texture is answered.
    #[error("failed to reference the previous frame (code {0})")]
    FrameRef(i32),
    /// The input is not a D3D12 picture, or holds a layout other than P010
    /// or NV12: what it is.
    #[error("D3d12ToneMap takes P010 or NV12 D3D12 pictures, got {0}")]
    UnsupportedPicture(String),
    /// The picture does not say it is PQ or HLG, so there is nothing to say
    /// what its numbers mean in light.
    #[error("D3d12ToneMap brings PQ or HLG video to SDR, and this picture is tagged {0:?}")]
    NotHdr(ffmpeg::color::TransferCharacteristic),
    /// The picture was made on another device than the [`D3d12Gpu`] this
    /// element was given.
    #[error("D3d12ToneMap was handed a picture of another D3D12 device")]
    DeviceMismatch,
    /// The visible size is zero or odd, which 4:2:0 cannot be, or larger
    /// than the texture.
    #[error("a {picture:?} picture in a {texture:?} texture")]
    InvalidSize {
        /// The picture's size.
        picture: (u32, u32),
        /// The texture's.
        texture: (u32, u32),
    },
    /// FFmpeg could not take the device for the frames the SDR pictures are
    /// written to — a device without D3D12 Video, as a software adapter is.
    #[error("failed to create the D3D12VA device context (code {0})")]
    HwDeviceInit(i32),
    /// Making the frames the SDR pictures are written to failed.
    #[error("could not make the frames to write to (code {0})")]
    Frames(i32),
    /// The sink received a buffer other than decoded video.
    #[error("D3d12ToneMap only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// Brings HDR video (PQ or HLG) on D3D12 down to SDR BT.709 NV12, on the GPU.
///
/// Takes P010 or NV12 pictures tagged PQ or HLG — a
/// [`D3d12Decoder`](crate::elements::D3d12Decoder)'s of an HDR stream — and
/// hands on NV12 pictures of the same size, tagged BT.709 limited range,
/// that every D3D12 element of this crate reads, without the picture
/// leaving video memory. What `D3d11ToneMap` draws on D3D11, on any
/// vendor's GPU. Needs the `d3d12` feature.
///
/// # Example
///
/// An HDR file recorded again as SDR, the new stream told what it now holds
/// through
/// [`D3d12VideoEncoder::with_color`](crate::elements::D3d12VideoEncoder::with_color):
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::color::ColorDescription;
/// use media_pp::elements::{
///     D3d12Decoder, D3d12Gpu, D3d12ToneMap, D3d12VideoCodec, D3d12VideoEncoder,
///     D3d12VideoEncoderOptions, FileDemuxer, FileMuxer, Mp4Layout,
/// };
/// use media_pp::{ffmpeg, pipeline::Pipeline};
///
/// let gpu = D3d12Gpu::new()?;
/// let (source, _) = FileDemuxer::open("file", "hdr.mp4")?;
/// let video = source.best(ffmpeg::media::Type::Video)?;
/// let (width, height) = video.size().ok_or(media_pp::Error::Other("no size".into()))?;
/// let decoder = D3d12Decoder::new("decode", video.parameters.clone(), &gpu)?;
/// let tone_map = D3d12ToneMap::new("sdr", &gpu)?;
/// let options = D3d12VideoEncoderOptions {
///     codec: D3d12VideoCodec::H265,
///     width,
///     height,
///     frame_rate: video.frame_rate.unwrap_or(ffmpeg::Rational::new(30, 1)),
///     bit_rate: 8_000_000,
///     gop_size: 60,
///     max_b_frames: None,
/// };
/// let color = ColorDescription::BT709_LIMITED; // what the tone map makes
/// let encoder = D3d12VideoEncoder::with_color("encode", &gpu, options, color)?;
///
/// let mut muxer = FileMuxer::create("sdr.mp4", Mp4Layout::Standard)?;
/// let track = muxer.add_stream("video", &encoder)?;
/// let sink = muxer.open()?.take(track)?;
/// let (pipeline, ()) = Pipeline::new("to-sdr", source, |source, ctx| {
///     let branch = ctx.branch().pipe(decoder).pipe(tone_map).queue("frames", 4);
///     ctx.attach(source, video.index, branch.pipe(encoder).to(sink)?)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// A compute shader.
///
/// What it does to each pixel is `core/tone_map.rs`'s definition: BT.2020's
/// matrix, the picture's transfer into nits, BT.2020's primaries into
/// BT.709's, BT.2390's EETF from 1000 nits down to 203 on the largest
/// channel, and 203 nits as SDR white, encoded with gamma 2.2. Each pixel's
/// colour is then taken as the eight bits an RGB picture of it would hold
/// and made BT.709 limited-range Y', and each 2x2 block's average colour its
/// Cb and Cr, as the CUDA converter makes NV12 of BGRA.
///
/// NV12 rather than the BGRA the D3D11 and CUDA elements make, as every D3D12
/// element of this crate — the renderers, the encoder, the detectors, the
/// overlay — reads NV12. The output is the visible size of the input, from
/// a pool of this element's own, tagged BT.709 limited range, with the
/// input's timing. A picture that says neither PQ nor HLG is refused with
/// [`D3d12ToneMapError::NotHdr`] rather than guessed at. A repeated input is
/// answered with the output already made from it.
pub struct D3d12ToneMap(FilterStage<Mapping>);

filter_stage!(D3d12ToneMap);

/// The SDR picture's planes as textures the shader writes, and the size
/// they were made for.
struct Work {
    luma: ID3D12Resource,
    chroma: ID3D12Resource,
    size: (u32, u32),
}

/// What a [`D3d12ToneMap`] does to each picture: all of its work, which the
/// framework makes the filter.
struct Mapping {
    pp_log: PpLog,
    device: ID3D12Device,
    /// `tone_map.hlsl`'s `hdr_to_nv12`.
    kernel: Kernel,
    commands: Commands,
    /// The picture's planes' views and the SDR planes', shader-visible.
    heap: ID3D12DescriptorHeap,
    heap_step: usize,
    work: Option<Work>,
    /// This element's own reference to the device its outputs are made on.
    hw_device_ctx: AvBufferRef,
    /// The pool the outputs come from, made for the size of the pictures
    /// arriving — see `ForSize`.
    frames: ForSize<AvBufferRef>,
    /// Only the CPU-side `AVFrame` wrapper is reused; each texture comes
    /// from `frames`.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The last output and the picture it was made from — see
    /// [`RepeatedOutput`].
    repeated: RepeatedOutput,
    /// The picture the last pass reads, held until the GPU has: let go of
    /// when it is handed on, its texture would go back to its producer's
    /// pool, to be written again under the read.
    reading: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

// SAFETY: the FFmpeg buffers have no thread affinity, and the D3D12 objects
// are free-threaded and touched only through `&mut self`, by the one thread
// transforming at a time.
unsafe impl Send for Mapping {}

impl D3d12ToneMap {
    /// A tone map on pictures made on `gpu` — the same [`D3d12Gpu`] every
    /// other D3D12 element of the pipeline was made with — its outputs made
    /// there too. Every picture comes out the size it went in.
    ///
    /// # Errors
    ///
    /// D3D12's, where the shader will not compile, and FFmpeg's where it
    /// cannot take the device.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
    ) -> std::result::Result<Self, D3d12ToneMapError> {
        let identity = Identity::new(ElementType::D3d12ToneMap, name);
        let pp_log = identity.pp_log().clone();
        let device = gpu.device().clone();
        // SAFETY: a live device, and a description built here.
        let (kernel, heap, heap_step) = unsafe {
            let kernel = Kernel::new(&device, SHADER, s!("tone_map.hlsl"), s!("hdr_to_nv12"))?;
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 4,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let heap_step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            (kernel, heap, heap_step)
        };
        let commands = Commands::new(&device, gpu.queue())?;
        // SAFETY: a live device, whose context owns its own reference.
        let hw_device_ctx =
            unsafe { create_hw_device_ctx(&device) }.map_err(D3d12ToneMapError::HwDeviceInit)?;
        pp_info!(pp_log: &pp_log, "created");
        Ok(Self(FilterStage::new(
            identity,
            Mapping {
                pp_log,
                device,
                kernel,
                commands,
                heap,
                heap_step,
                work: None,
                hw_device_ctx,
                frames: ForSize::new(),
                pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
                repeated: RepeatedOutput::new(),
                reading: None,
            },
        )))
    }
}

/// `tone_map.hlsl`'s thirty-two constants for `map`, on a picture of
/// `size`, in its `ToneMap` order.
fn constants(map: &ToneMap, size: (u32, u32)) -> [u32; 32] {
    let mut words = [0u32; 32];
    for (word, value) in words[..12].iter_mut().zip(map.rows.iter().flatten()) {
        *word = value.to_bits();
    }
    words[12] = map.transfer;
    for (word, value) in
        words[13..16]
            .iter_mut()
            .zip([map.source_peak_pq, map.target_peak, map.knee])
    {
        *word = value.to_bits();
    }
    for (word, value) in words[16..28].iter_mut().zip(map.gamut.iter().flatten()) {
        *word = value.to_bits();
    }
    words[28] = size.0;
    words[29] = size.1;
    words
}

/// A frame's texture and fence, cloned out of it, where it is a D3D12 one.
fn d3d12_parts(frame: &ffmpeg::frame::Video) -> Option<(ID3D12Resource, ID3D12Fence, u64)> {
    let (texture, fence, value) = d3d12va_texture(frame)
        .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())?;
    // SAFETY: the live frame owns these non-null objects; cloning the
    // borrowed wrappers takes references of the frame's own.
    unsafe {
        Some((
            ID3D12Resource::from_raw_borrowed(&texture)?.clone(),
            ID3D12Fence::from_raw_borrowed(&fence)?.clone(),
            value,
        ))
    }
}

impl Mapping {
    fn draw(
        &mut self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, D3d12ToneMapError> {
        let (texture, fence, fence_value) = d3d12_parts(frame).ok_or_else(|| {
            D3d12ToneMapError::UnsupportedPicture(format!("{:?}", frame.format()))
        })?;
        let map = ToneMap::of_frame(frame).ok_or(D3d12ToneMapError::NotHdr(
            frame.color_transfer_characteristic(),
        ))?;
        let mut made_on: Option<ID3D12Device> = None;
        // SAFETY: a live resource and a correctly typed out-parameter.
        unsafe { texture.GetDevice(&mut made_on) }?;
        if made_on.is_none_or(|made_on| made_on.as_raw() != self.device.as_raw()) {
            return Err(D3d12ToneMapError::DeviceMismatch);
        }
        // SAFETY: a live resource's plain description, returned by value.
        let desc = unsafe { texture.GetDesc() };
        let planes: (DXGI_FORMAT, DXGI_FORMAT) = match desc.Format {
            DXGI_FORMAT_P010 => (DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM),
            DXGI_FORMAT_NV12 => (DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM),
            other => {
                return Err(D3d12ToneMapError::UnsupportedPicture(format!(
                    "a texture of DXGI format {}",
                    other.0
                )));
            }
        };
        let size = (frame.width(), frame.height());
        let surface = (desc.Width as u32, desc.Height);
        if size.0 == 0
            || size.1 == 0
            || !size.0.is_multiple_of(2)
            || !size.1.is_multiple_of(2)
            || size.0 > surface.0
            || size.1 > surface.1
        {
            return Err(D3d12ToneMapError::InvalidSize {
                picture: size,
                texture: surface,
            });
        }
        // What the last pass queued still reads the views and that picture:
        // it runs out before either is touched or let go of.
        self.commands.wait_idle()?;
        self.reading = None;
        if self.work.as_ref().is_none_or(|work| work.size != size) {
            // SAFETY: a live device.
            let (luma, chroma) = unsafe {
                (
                    writable_texture(&self.device, DXGI_FORMAT_R8_UNORM, size)?,
                    writable_texture(
                        &self.device,
                        DXGI_FORMAT_R8G8_UNORM,
                        (size.0 / 2, size.1 / 2),
                    )?,
                )
            };
            self.work = Some(Work { luma, chroma, size });
        }
        let work = self.work.as_ref().expect("made above");
        let (luma, chroma) = (work.luma.clone(), work.chroma.clone());

        let (device, pp_log) = (&self.hw_device_ctx, &self.pp_log);
        let frames_ctx = self
            .frames
            .try_get(size.0, size.1, |width, height| {
                pp_info!(pp_log: pp_log, "writing NV12 {width}x{height}");
                // SAFETY: a live device context, which is what the owned
                // `AvBufferRef` is; the pool grows as frames are taken.
                unsafe {
                    create_hw_frames_ctx_of(
                        device,
                        ffi::AVPixelFormat::AV_PIX_FMT_NV12,
                        width,
                        height,
                        0,
                    )
                }
                .map_err(D3d12ToneMapError::Frames)
            })?
            .as_ptr();
        let mut output = self.pool.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous texture
        // handed back first; the frames context is this element's own.
        unsafe {
            let dst = output.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames_ctx, dst, 0);
            if code < 0 {
                return Err(D3d12ToneMapError::Frames(code));
            }
        }
        let (to, to_fence, to_value) = d3d12_parts(&output).ok_or_else(|| {
            D3d12ToneMapError::UnsupportedPicture("a frame of its own pool".into())
        })?;

        let step = self.heap_step;
        // SAFETY: the GPU is idle on this element's work, waited for above,
        // so the heap's views may be written; each names a live resource.
        unsafe {
            let start = self.heap.GetCPUDescriptorHandleForHeapStart();
            let at = |index: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + index * step,
            };
            self.device
                .CreateShaderResourceView(&texture, Some(&plane_view(planes.0, 0)), at(0));
            self.device
                .CreateShaderResourceView(&texture, Some(&plane_view(planes.1, 1)), at(1));
            self.device.CreateUnorderedAccessView(
                &luma,
                None,
                Some(&plane_write_view(DXGI_FORMAT_R8_UNORM, 0)),
                at(2),
            );
            self.device.CreateUnorderedAccessView(
                &chroma,
                None,
                Some(&plane_write_view(DXGI_FORMAT_R8G8_UNORM, 0)),
                at(3),
            );
        }
        let words = constants(&map, size);
        let list = self.commands.begin(Some(&self.kernel.pipeline))?;
        // SAFETY: the list is open and the GPU idle on this element's work;
        // every view and barrier names a live resource this element, the
        // picture or the output holds until the queue has run what reads it.
        unsafe {
            list.SetComputeRootSignature(&self.kernel.root_signature);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootDescriptorTable(1, self.heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRoot32BitConstants(0, 32, words.as_ptr().cast(), 0);
            transition(
                &list,
                &texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            list.Dispatch(
                (size.0 / 2).div_ceil(GROUP),
                (size.1 / 2).div_ceil(GROUP),
                1,
            );
            transition(
                &list,
                &texture,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            for plane in [&luma, &chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
            }
            transition(
                &list,
                &to,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            copy_plane(&list, (&to, 0), (&luma, 0));
            copy_plane(&list, (&to, 1), (&chroma, 0));
            transition(
                &list,
                &to,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            );
            for plane in [&luma, &chroma] {
                transition(
                    &list,
                    plane,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
        }
        // After whatever wrote the picture, and whatever last wrote the frame
        // written to; the frame's own fence says when it is written.
        let written = to_value + 1;
        self.commands.submit(
            &[(&fence, fence_value), (&to_fence, to_value)],
            &[(&to_fence, written)],
        )?;
        set_d3d12va_fence_value(&mut output, written);
        self.reading = Some(Arc::clone(frame));
        crate::buffer::carry_timing(&mut output, frame);
        output.set_color_space(ffmpeg::color::Space::BT709);
        output.set_color_range(ffmpeg::color::Range::MPEG);
        output.set_color_primaries(ffmpeg::color::Primaries::BT709);
        output.set_color_transfer_characteristic(ffmpeg::color::TransferCharacteristic::BT709);
        Ok(output)
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // What is queued still reads the views and the picture held: it runs
        // out before they go.
        let _ = self.commands.wait_idle();
    }
}

impl PerFrameTransform for Mapping {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        D3d12ToneMapError::FrameRef(code).into()
    }

    fn produce(
        &mut self,
        frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        Ok(self
            .draw(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?)
    }
}

impl Filter for Mapping {
    /// P010 or NV12 D3D12 pictures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::YUV420),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => {
                let drawn = PerFrameTransform::transform(self, &frame)?;
                out.push(MediaBuffer::Video(drawn.into()));
                Ok(())
            }
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(D3d12ToneMapError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn reset(&mut self) {
        self.repeated.clear();
    }

    /// SDR NV12 D3D12 pictures.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::RawSink;
    use crate::test_support::{capture, try_d3d12_gpu};

    /// A P010 picture `width` by `height`, ten-bit luma across the range and
    /// chroma pairs off grey, tagged BT.2020 limited range in `transfer`.
    fn p010(transfer: ffmpeg::color::TransferCharacteristic) -> ffmpeg::frame::Video {
        let (width, height) = (64usize, 32usize);
        let mut frame =
            ffmpeg::frame::Video::new(ffmpeg::format::Pixel::P010LE, width as u32, height as u32);
        let (stride, cstride) = (frame.stride(0), frame.stride(1));
        for y in 0..height {
            for x in 0..width {
                frame.data_mut(0)[y * stride + x * 2..][..2]
                    .copy_from_slice(&(luma(x, y) << 6).to_le_bytes());
            }
        }
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                let (cb, cr) = chroma(x, y);
                let at = y * cstride + x * 4;
                frame.data_mut(1)[at..at + 2].copy_from_slice(&(cb << 6).to_le_bytes());
                frame.data_mut(1)[at + 2..at + 4].copy_from_slice(&(cr << 6).to_le_bytes());
            }
        }
        crate::color::ColorDescription {
            space: ffmpeg::color::Space::BT2020NCL,
            range: ffmpeg::color::Range::MPEG,
            primaries: ffmpeg::color::Primaries::BT2020,
            transfer,
        }
        .describe(&mut frame);
        frame.set_pts(Some(9));
        frame
    }

    fn luma(x: usize, y: usize) -> u16 {
        (64 + (x * 13 + y * 7) % 877) as u16
    }

    fn chroma(x: usize, y: usize) -> (u16, u16) {
        match (x + y) % 3 {
            0 => (512, 512),
            1 => (400, 620),
            _ => (600, 450),
        }
    }

    /// `frame` as a D3D12 picture on `gpu`; `None` where FFmpeg cannot make
    /// one.
    fn upload(gpu: &D3d12Gpu, frame: &ffmpeg::frame::Video) -> Option<MediaBuffer> {
        // SAFETY: a live device; the frames context is wrapped before it is
        // filled in, and the frame is this function's own until returned.
        unsafe {
            let device = create_hw_device_ctx(gpu.device()).ok()?;
            let Ok(frames) = create_hw_frames_ctx_of(
                &device,
                frame.format().into(),
                frame.width(),
                frame.height(),
                0,
            ) else {
                eprintln!("skipping: FFmpeg could not make D3D12 frames");
                return None;
            };
            let mut on_gpu = ffmpeg::frame::Video::empty();
            assert!(ffi::av_hwframe_get_buffer(frames.as_ptr(), on_gpu.as_mut_ptr(), 0) >= 0);
            assert!(ffi::av_hwframe_transfer_data(on_gpu.as_mut_ptr(), frame.as_ptr(), 0) >= 0);
            assert!(ffi::av_frame_copy_props(on_gpu.as_mut_ptr(), frame.as_ptr()) >= 0);
            Some(MediaBuffer::video(on_gpu))
        }
    }

    /// The picture `buf` holds, copied to system memory.
    fn download(buf: &MediaBuffer) -> ffmpeg::frame::Video {
        let MediaBuffer::Video(frame) = buf else {
            panic!("a picture");
        };
        let mut down = ffmpeg::frame::Video::empty();
        // SAFETY: a live D3D12 frame read into a fresh one FFmpeg allocates.
        unsafe {
            assert!(ffi::av_hwframe_transfer_data(down.as_mut_ptr(), frame.as_ptr(), 0) >= 0);
        }
        down
    }

    /// BT.709 limited-range Y', Cb, Cr of R'G'B' 0 to 255, unrounded — the
    /// shader's `bt709_limited`.
    fn bt709_limited([r, g, b]: [f32; 3]) -> [f32; 3] {
        let (r, g, b) = (r / 255.0, g / 255.0, b / 255.0);
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        [
            16.0 + 219.0 * y,
            128.0 + 224.0 * (b - y) / 1.8556,
            128.0 + 224.0 * (r - y) / 1.5748,
        ]
    }

    /// An HLG or PQ picture comes out SDR NV12 as `core/tone_map.rs`'s
    /// definition makes each pixel, made BT.709 limited range — each
    /// pixel's luma, each 2x2 block's average colour its chroma — to within
    /// the rounding of a GPU's powers; tagged so, at its size and with its
    /// timing.
    #[test]
    fn hdr_comes_out_as_the_shared_definition_says() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        for transfer in [
            ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
            ffmpeg::color::TransferCharacteristic::SMPTE2084,
        ] {
            let picture = p010(transfer);
            let map = ToneMap::of_frame(&picture).expect("HDR");
            let Some(input) = upload(&gpu, &picture) else {
                return;
            };
            let mut mapping = D3d12ToneMap::new("tone map", &gpu).expect("opens");
            let kept = capture(&mut mapping);
            mapping.consume(input).expect("brought down");
            let output = kept.lock().unwrap().remove(0);
            let MediaBuffer::Video(frame) = &output else {
                panic!("a picture");
            };
            assert_eq!((frame.width(), frame.height()), (64, 32));
            assert_eq!(frame.pts(), Some(9));
            assert_eq!(frame.color_space(), ffmpeg::color::Space::BT709);
            assert_eq!(frame.color_range(), ffmpeg::color::Range::MPEG);
            assert_eq!(
                frame.color_transfer_characteristic(),
                ffmpeg::color::TransferCharacteristic::BT709
            );
            let down = download(&output);
            assert_eq!(down.format(), ffmpeg::format::Pixel::NV12);
            let unit = |code: u16| f32::from(code << 6) / 65535.0;
            let rgb = |x: usize, y: usize| {
                let (cb, cr) = chroma(x / 2, y / 2);
                map.apply(unit(luma(x, y)), unit(cb), unit(cr))
                    .map(f32::from)
            };
            let mut worst = 0.0f32;
            for y in 0..32 {
                for x in 0..64 {
                    let want = bt709_limited(rgb(x, y))[0];
                    let got = f32::from(down.data(0)[y * down.stride(0) + x]);
                    worst = worst.max((got - want).abs());
                }
            }
            for y in 0..16 {
                for x in 0..32 {
                    let mut sum = [0.0f32; 3];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        for (sum, value) in sum.iter_mut().zip(rgb(2 * x + dx, 2 * y + dy)) {
                            *sum += value;
                        }
                    }
                    let want = bt709_limited(sum.map(|value| value / 4.0));
                    let got = &down.data(1)[y * down.stride(1) + 2 * x..][..2];
                    for (got, want) in got.iter().zip(&want[1..]) {
                        worst = worst.max((f32::from(*got) - want).abs());
                    }
                }
            }
            assert!(worst <= 2.5, "{transfer:?}: off by {worst} of 255");
        }
    }

    /// A ten-bit picture tagged neither HLG nor PQ is refused, saying so, and
    /// the next is brought down.
    #[test]
    fn a_picture_that_is_not_hdr_is_refused() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let mut mapping = match D3d12ToneMap::new("tone map", &gpu) {
            Ok(mapping) => mapping,
            Err(D3d12ToneMapError::HwDeviceInit(code)) => {
                eprintln!("skipping: FFmpeg could not take this device for D3D12 frames ({code})");
                return;
            }
            Err(error) => panic!("did not open: {error}"),
        };
        let kept = capture(&mut mapping);
        let Some(sdr) = upload(&gpu, &p010(ffmpeg::color::TransferCharacteristic::BT709)) else {
            return;
        };
        let refused = mapping.consume(sdr);
        assert!(
            matches!(
                refused,
                Err(crate::error::Error::D3d12ToneMapError(
                    D3d12ToneMapError::NotHdr(_)
                ))
            ),
            "{refused:?}"
        );
        let hlg = upload(
            &gpu,
            &p010(ffmpeg::color::TransferCharacteristic::ARIB_STD_B67),
        )
        .expect("uploaded above");
        mapping.consume(hlg).expect("brought down");
        assert_eq!(kept.lock().unwrap().len(), 1);
    }
}
