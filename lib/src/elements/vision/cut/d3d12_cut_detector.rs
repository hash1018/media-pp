//! [`D3d12CutDetector`]: where one shot ends and the next begins, in D3D12
//! pictures, their thumbnails summed on the GPU.

use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D12::*,
        Dxgi::Common::{
            DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8_UINT,
            DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16_UINT,
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
    element::{Attached, ElementType, Identity},
    elements::vision::batch::PerStream,
    error::Result,
    platform::windows::{
        d3d12_compute::{Commands, Kernel, buffer, plane_view, transition, uav_barrier},
        d3d12_gpu::D3d12Gpu,
        d3d12va::d3d12va_texture,
    },
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::{CELLS, CutDetectorOptions, CutDetectorOptionsError, Judge, Thumbnail};

const SHADER: &[u8] = include_bytes!("../../../shaders/d3d12/cut.hlsl");

/// The floats a thumbnail's sums take at most: a luma cell's one, a chroma
/// cell's two.
const SUMS: usize = (CELLS.0 * CELLS.1 + (CELLS.0 / 2) * (CELLS.1 / 2) * 2) as usize;

/// Errors from a [`D3d12CutDetector`].
#[derive(Debug, ThisError)]
pub enum D3d12CutDetectorError {
    /// The options were refused.
    #[error(transparent)]
    Options(#[from] CutDetectorOptionsError),
    /// It was handed something other than a picture.
    #[error("D3d12CutDetector takes D3D12 pictures, got {0}")]
    UnsupportedBuffer(&'static str),
    /// A picture it cannot read: not a D3D12 one, or holding a layout other
    /// than NV12 or P010.
    #[error("D3d12CutDetector reads NV12 or P010 D3D12 pictures, got {0}")]
    UnsupportedPicture(String),
    /// A D3D12 picture made on another device than the [`D3d12Gpu`] it was
    /// given.
    #[error("D3d12CutDetector was handed a picture of another D3D12 device")]
    DeviceMismatch,
    /// A picture larger than the texture it is said to be in.
    #[error("a {picture:?} picture in a {texture:?} texture")]
    PictureOutsideTexture {
        /// The picture's width and height.
        picture: (u32, u32),
        /// The texture's width and height.
        texture: (u32, u32),
    },
    /// D3D12 failed.
    #[error(transparent)]
    D3d12(#[from] windows::core::Error),
}

/// Finds where one shot of an edited video ends and the next begins, in
/// D3D12 pictures.
///
/// It takes NV12 or P010 D3D12 pictures made on the [`D3d12Gpu`] it is
/// given and hands each picture on — the first of each shot after the
/// stream's first carrying a [`SceneCut`](crate::elements::SceneCut) — as
/// [`SwCutDetector`](crate::elements::SwCutDetector) does in system memory,
/// the two finding the same cuts in the same pictures. Needs the `d3d12`
/// feature, on Windows.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{CutDetectorOptions, D3d12CutDetector, D3d12Decoder, D3d12Gpu};
/// # use media_pp::elements::{AppSink, FileDemuxer};
/// # use media_pp::{ffmpeg::media, pipeline::Pipeline};
/// # let (source, _) = FileDemuxer::open("demux", "input.mp4")?;
/// # let video = source.best(media::Type::Video)?;
/// # let params = video.parameters.clone();
/// # let sink = AppSink::new("shots", |_buf| Ok(()));
///
/// let gpu = D3d12Gpu::new()?;
/// let cuts = D3d12CutDetector::new("cuts", &gpu, CutDetectorOptions::default())?;
/// let (pipeline, ()) = Pipeline::new("cuts", source, |source, ctx| {
///     let decoder = D3d12Decoder::new("decoder", params, &gpu)?;
///     let branch = ctx.branch().pipe(decoder).pipe(cuts).to(sink)?;
///     ctx.attach(source, video.index, branch)?;
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// The cells of each picture's thumbnail are summed on the GPU, by a
/// compute shader on the GPU's queue, where the picture is; only the sums,
/// under 5,000 floats, come down, and the CPU divides each by its count as
/// the software detector does — a P010 picture's samples read for their
/// high bytes, as the CPU reads them. It holds
/// [`CutDetectorOptions::lookahead`] pictures of each stream.
pub struct D3d12CutDetector(FilterStage<Detecting>);

filter_stage!(D3d12CutDetector);

/// What a [`D3d12CutDetector`] does with each picture.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    options: CutDetectorOptions,
    judges: PerStream<Judge>,
    device: ID3D12Device,
    /// `cut.hlsl`'s `luma_sums`, and `chroma_sums` under its root
    /// signature.
    luma: Kernel,
    chroma: ID3D12PipelineState,
    /// The views of a picture's two planes, shader-visible.
    heap: ID3D12DescriptorHeap,
    heap_step: usize,
    commands: Commands,
    /// The sums, which the shader writes, and what they are copied into for
    /// the CPU to read.
    sums: ID3D12Resource,
    readback: ID3D12Resource,
}

// SAFETY: the D3D12 objects are free-threaded; the list, heap and buffers
// are only touched through `&mut self`, by the one thread transforming at a
// time.
unsafe impl Send for Detecting {}

impl D3d12CutDetector {
    /// A cut detector judging as `options` say, on pictures made on `gpu` —
    /// the same [`D3d12Gpu`] every other D3D12 element of the pipeline was
    /// made with.
    ///
    /// # Errors
    ///
    /// Options it cannot judge with, and D3D12's where its shader will not
    /// compile or its buffers cannot be made.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        options: CutDetectorOptions,
    ) -> std::result::Result<Self, D3d12CutDetectorError> {
        options.check()?;
        let identity = Identity::new(ElementType::D3d12CutDetector, name);
        let name = identity.name();
        let pp_log = identity.pp_log().clone();
        let device = gpu.device().clone();
        let bytes = (SUMS * size_of::<f32>()) as u64;
        // SAFETY: every call is on a live device with descriptions built
        // here.
        let (luma, chroma, heap, heap_step, sums, readback) = unsafe {
            let luma = Kernel::new(&device, SHADER, s!("cut.hlsl"), s!("luma_sums"))?;
            let chroma = luma.sibling(&device, SHADER, s!("cut.hlsl"), s!("chroma_sums"))?;
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let heap_step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let sums = buffer(
                &device,
                bytes,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            let readback = buffer(
                &device,
                bytes,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            (luma, chroma, heap, heap_step, sums, readback)
        };
        let commands = Commands::new(&device, gpu.queue())?;
        pp_info!(pp_log: &pp_log, "opened: {options:?}");
        Ok(Self(FilterStage::new(
            identity,
            Detecting {
                name,
                pp_log,
                options,
                judges: PerStream::default(),
                device,
                luma,
                chroma,
                heap,
                heap_step,
                commands,
                sums,
                readback,
            },
        )))
    }
}

impl Detecting {
    /// `frame`'s thumbnail, its cells summed on the GPU.
    fn thumbnail(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Thumbnail, D3d12CutDetectorError> {
        let (texture, fence, fence_value) = d3d12va_texture(frame)
            .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())
            .ok_or_else(|| {
                D3d12CutDetectorError::UnsupportedPicture(format!("{:?}", frame.format()))
            })?;
        // SAFETY: the live frame owns these non-null objects; cloning the
        // borrowed wrappers takes references of the picture's own.
        let (texture, fence) = unsafe {
            (
                ID3D12Resource::from_raw_borrowed(&texture)
                    .expect("not null")
                    .clone(),
                ID3D12Fence::from_raw_borrowed(&fence)
                    .expect("not null")
                    .clone(),
            )
        };
        let mut made_on: Option<ID3D12Device> = None;
        // SAFETY: a live resource and a correctly typed out-parameter.
        unsafe { texture.GetDevice(&mut made_on) }?;
        if made_on.is_none_or(|made_on| made_on.as_raw() != self.device.as_raw()) {
            return Err(D3d12CutDetectorError::DeviceMismatch);
        }
        // SAFETY: a live resource's plain description, returned by value.
        let desc = unsafe { texture.GetDesc() };
        // Each plane read as integers, a P010 sample for its high byte.
        let (luma_format, chroma_format, shift) = match desc.Format {
            DXGI_FORMAT_NV12 => (DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8_UINT, 0),
            DXGI_FORMAT_P010 => (DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16_UINT, 8),
            other => {
                return Err(D3d12CutDetectorError::UnsupportedPicture(format!(
                    "a texture of DXGI format {}",
                    other.0
                )));
            }
        };
        let (width, height) = (frame.width(), frame.height());
        let surface = (desc.Width as u32, desc.Height);
        if width == 0 || height == 0 || width > surface.0 || height > surface.1 {
            return Err(D3d12CutDetectorError::PictureOutsideTexture {
                picture: (width, height),
                texture: surface,
            });
        }
        let chroma_size = (width.div_ceil(2), height.div_ceil(2));
        let (cells, chroma_cells) = Thumbnail::cells_of(width, height);
        let luma_cells = (cells.0 * cells.1) as usize;
        let chroma_count = (chroma_cells.0 * chroma_cells.1) as usize;
        let floats = luma_cells + 2 * chroma_count;
        let list = self.commands.begin(Some(&self.luma.pipeline))?;
        // SAFETY: the list is open and the GPU idle on this element's work
        // (`begin` waited); every view and barrier names a live resource this
        // element or the picture holds until the queue has run what reads it.
        unsafe {
            let cpu_start = self.heap.GetCPUDescriptorHandleForHeapStart();
            for (plane, format) in [luma_format, chroma_format].into_iter().enumerate() {
                self.device.CreateShaderResourceView(
                    &texture,
                    Some(&plane_view(format, plane as u32)),
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: cpu_start.ptr + plane * self.heap_step,
                    },
                );
            }
            transition(
                &list,
                &texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );
            list.SetComputeRootSignature(&self.luma.root_signature);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootDescriptorTable(1, self.heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRootUnorderedAccessView(2, self.sums.GetGPUVirtualAddress());
            for (pipeline, size, count, at) in [
                (&self.luma.pipeline, (width, height), cells, 0),
                (&self.chroma, chroma_size, chroma_cells, luma_cells),
            ] {
                let constants = [size.0, size.1, count.0, count.1, shift, at as u32, 0, 0];
                list.SetPipelineState(pipeline);
                list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                list.Dispatch((count.0 * count.1).div_ceil(64), 1, 1);
            }
            transition(
                &list,
                &texture,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            uav_barrier(&list, &self.sums);
            transition(
                &list,
                &self.sums,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            list.CopyBufferRegion(
                &self.readback,
                0,
                &self.sums,
                0,
                (floats * size_of::<f32>()) as u64,
            );
            transition(
                &list,
                &self.sums,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
        }
        // A GPU-side wait for whatever wrote the picture: sharing a device
        // does not order this queue after the decoder's.
        self.commands.submit(&[(&fence, fence_value)], &[])?;
        self.commands.wait_idle()?;
        // SAFETY: the read-back buffer holds `SUMS` floats, of which the
        // copy just waited for wrote the first `floats`; it is unmapped
        // before this returns.
        let sums = unsafe {
            let mut mapped = std::ptr::null_mut();
            self.readback.Map(0, None, Some(&mut mapped))?;
            let sums = std::slice::from_raw_parts(mapped.cast::<f32>(), floats).to_vec();
            self.readback.Unmap(0, None);
            sums
        };
        // Each sum over its count, as `cell_means` takes it on the CPU.
        let edge = |index: u32, cells: u32, length: u32| {
            (u64::from(index) * u64::from(length) / u64::from(cells)) as usize
        };
        let counts = |cells: (u32, u32), size: (u32, u32)| {
            (0..cells.1).flat_map(move |cy| {
                (0..cells.0).map(move |cx| {
                    let across = edge(cx + 1, cells.0, size.0) - edge(cx, cells.0, size.0);
                    let down = edge(cy + 1, cells.1, size.1) - edge(cy, cells.1, size.1);
                    (across * down) as f32
                })
            })
        };
        let (luma, chroma) = sums.split_at(luma_cells);
        Ok(Thumbnail {
            cells,
            luma: luma
                .iter()
                .zip(counts(cells, (width, height)))
                .map(|(sum, count)| sum * (1.0 / count))
                .collect(),
            chroma_cells,
            chroma: chroma
                .as_chunks::<2>()
                .0
                .iter()
                .zip(counts(chroma_cells, chroma_size))
                .flat_map(|(&[cb, cr], count)| [cb * (1.0 / count), cr * (1.0 / count)])
                .collect(),
        })
    }
}

impl Drop for Detecting {
    fn drop(&mut self) {
        // What is queued still reads the views and writes the sums: it runs
        // out before they go.
        let _ = self.commands.wait_idle();
    }
}

impl Filter for Detecting {
    /// NV12 or P010 D3D12 pictures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::YUV420),
        )
    }

    /// The pictures it was handed.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(D3d12CutDetectorError::UnsupportedBuffer(kind).into());
        };
        let thumbnail = self
            .thumbnail(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        let (name, options) = (&self.name, self.options);
        let (judge, moved) = self
            .judges
            .get(&buf, |_| Judge::new(Arc::clone(name), options));
        if moved {
            // That stream was sought: what it held is from before.
            judge.clear();
        }
        judge.push(buf, thumbnail, out);
        Ok(())
    }

    /// Every picture still held, judged by those after it there are.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        for judge in self.judges.values_mut() {
            judge.finish(out);
        }
        Ok(())
    }

    /// A seek or a flush: what was held is let go of.
    fn reset(&mut self) {
        self.judges.clear();
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

#[cfg(test)]
mod tests {
    use super::super::sw_cut_detector::thumbnail_of;
    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::D3d12Upload;
    use crate::test_support::{capture, try_d3d12_gpu};
    use ffmpeg::format::Pixel;

    /// A picture in which no sample is its neighbour's: NV12 or P010, every
    /// byte of it.
    fn noise_of(format: Pixel, width: u32, height: u32, seed: u32) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        for plane in 0..2 {
            for (index, byte) in frame.data_mut(plane).iter_mut().enumerate() {
                let mut v =
                    (index as u32 ^ seed).wrapping_mul(2_654_435_761) ^ (plane as u32 * 977);
                v ^= v >> 15;
                *byte = (v.wrapping_mul(2_246_822_519) >> 24) as u8;
            }
        }
        frame
    }

    /// `frame` as a D3D12 picture on `gpu`, uploaded as FFmpeg uploads its
    /// layout; `None` where FFmpeg cannot.
    fn on_gpu(gpu: &D3d12Gpu, frame: &ffmpeg::frame::Video) -> Option<ffmpeg::frame::Video> {
        use crate::platform::ffmpeg::AvBufferRef;
        use crate::platform::windows::d3d12va::create_hw_device_ctx;
        use ffmpeg::ffi;
        // SAFETY: a live device; the frames context is wrapped before it is
        // filled in, and the frame is this function's own until returned.
        unsafe {
            let device = create_hw_device_ctx(gpu.device()).ok()?;
            let frames = AvBufferRef::from_raw(ffi::av_hwframe_ctx_alloc(device.as_ptr()))?;
            let context = (*frames.as_ptr()).data as *mut ffi::AVHWFramesContext;
            (*context).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D12;
            (*context).sw_format = frame.format().into();
            (*context).width = frame.width() as i32;
            (*context).height = frame.height() as i32;
            if ffi::av_hwframe_ctx_init(frames.as_ptr()) < 0 {
                eprintln!("skipping: FFmpeg could not make D3D12 frames");
                return None;
            }
            let mut on_gpu = ffmpeg::frame::Video::empty();
            assert!(ffi::av_hwframe_get_buffer(frames.as_ptr(), on_gpu.as_mut_ptr(), 0) >= 0);
            assert!(ffi::av_hwframe_transfer_data(on_gpu.as_mut_ptr(), frame.as_ptr(), 0) >= 0);
            assert!(ffi::av_frame_copy_props(on_gpu.as_mut_ptr(), frame.as_ptr()) >= 0);
            Some(on_gpu)
        }
    }

    /// The GPU's thumbnail of a picture is the CPU's, float for float, at a
    /// size the cells divide evenly and one they do not, NV12 and P010 —
    /// whose high bytes are what is read.
    #[test]
    fn the_gpu_makes_the_cpus_thumbnail() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let mut detecting = D3d12CutDetector::new("cuts", &gpu, CutDetectorOptions::default())
            .unwrap()
            .0;
        for (format, width, height) in [
            (Pixel::NV12, 128, 72),
            (Pixel::NV12, 1918, 1078),
            (Pixel::P010LE, 128, 72),
            (Pixel::P010LE, 1918, 1078),
        ] {
            let picture = noise_of(format, width, height, width);
            let Some(frame) = on_gpu(&gpu, &picture) else {
                return;
            };
            let gpu = detecting.inner_mut().thumbnail(&frame).expect("thumbnail");
            assert_eq!(
                Some(gpu),
                thumbnail_of(&picture),
                "{format:?} {width}x{height}"
            );
        }
    }

    /// Two shots: every picture comes out, in order, the last held back
    /// until the end of the stream, and the first of the second shot carries
    /// the cut — in NV12 and in P010.
    #[test]
    fn the_first_picture_of_a_new_shot_carries_the_cut() {
        use crate::buffer::set_time_base;
        use crate::elements::SceneCut;
        use crate::stream::{StreamEvent, deliver};
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        for format in [Pixel::NV12, Pixel::P010LE] {
            let mut detector = D3d12CutDetector::new(
                "cuts",
                &gpu,
                CutDetectorOptions {
                    min_scene: std::time::Duration::ZERO,
                    ..CutDetectorOptions::default()
                },
            )
            .unwrap();
            let kept = capture(&mut detector);
            for index in 0..20 {
                let (left, right) = if index < 12 { (40, 200) } else { (200, 40) };
                let mut frame = ffmpeg::frame::Video::new(format, 128, 72);
                let (stride, bytes) = (frame.stride(0), if format == Pixel::NV12 { 1 } else { 2 });
                for row in frame.data_mut(0).chunks_mut(stride).take(72) {
                    for (x, sample) in row[..128 * bytes].chunks_mut(bytes).enumerate() {
                        sample.fill(if x < 64 { left } else { right });
                    }
                }
                frame.data_mut(1).fill(128);
                frame.set_pts(Some(index));
                set_time_base(&mut frame, ffmpeg::Rational(1, 30));
                let Some(on_gpu) = on_gpu(&gpu, &frame) else {
                    return;
                };
                detector
                    .consume(MediaBuffer::video(on_gpu))
                    .expect("judged");
            }
            assert_eq!(kept.lock().unwrap().len(), 18, "{format:?}: two held back");
            deliver(&mut detector, &StreamEvent::Eos).expect("drained");
            let kept = kept.lock().unwrap();
            assert_eq!(kept.len(), 20, "{format:?}");
            let cuts: Vec<usize> = kept
                .iter()
                .enumerate()
                .filter(|(_, buf)| {
                    buf.metadata()
                        .is_some_and(|metadata| metadata.get::<SceneCut>().is_some())
                })
                .map(|(index, _)| index)
                .collect();
            assert_eq!(cuts, vec![12], "{format:?}");
        }
    }

    /// A picture in system memory, and one of another device, are refused,
    /// and nothing is handed on; the next picture is read.
    #[test]
    fn what_it_cannot_read_is_refused() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let mut detector = D3d12CutDetector::new(
            "cuts",
            &gpu,
            CutDetectorOptions {
                lookahead: 0,
                ..CutDetectorOptions::default()
            },
        )
        .unwrap();
        let kept = capture(&mut detector);
        let in_memory = noise_of(Pixel::NV12, 64, 36, 1);
        assert!(
            detector.consume(MediaBuffer::video(in_memory)).is_err(),
            "a CPU picture is no D3D12 one"
        );
        // A second device, where the adapter makes one apart from the first.
        if let Some(other) = try_d3d12_gpu()
            && other.device().as_raw() != gpu.device().as_raw()
            && let Ok(mut upload) = D3d12Upload::new("upload", &other)
        {
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(noise_of(Pixel::NV12, 64, 36, 2)))
                .expect("uploads");
            let elsewhere = uploaded.lock().unwrap().remove(0);
            let refused = detector.consume(elsewhere);
            assert!(
                matches!(
                    refused,
                    Err(crate::error::Error::D3d12CutDetectorError(
                        D3d12CutDetectorError::DeviceMismatch
                    ))
                ),
                "{refused:?}"
            );
        }
        assert!(kept.lock().unwrap().is_empty());
        let Some(frame) = on_gpu(&gpu, &noise_of(Pixel::NV12, 64, 36, 3)) else {
            return;
        };
        detector
            .consume(MediaBuffer::video(frame))
            .expect("reads the next");
        assert_eq!(kept.lock().unwrap().len(), 1, "and hands it on");
    }

    /// Behind a `D3d12Decoder`, holding as many of its pictures as a long
    /// lookahead asks, it hands every picture of the fixture on, in order,
    /// and finds the cuts the software detector finds in the same stream
    /// decoded on the CPU.
    #[test]
    fn behind_a_decoder_it_finds_what_the_cpu_finds() {
        use crate::elements::{D3d12Decoder, SceneCut, SwCutDetector, SwDecoder};
        use crate::stream::{StreamEvent, deliver};
        let (Some(gpu), Some(video)) = (try_d3d12_gpu(), crate::test_support::try_test_video())
        else {
            return;
        };
        let mut input = ffmpeg::format::input(&video).expect("opens");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("a picture stream");
        let (index, time_base) = (stream.index(), stream.time_base());
        let parameters = stream.parameters();
        let packets: Vec<ffmpeg::Packet> = input
            .packets()
            .filter(|(stream, _)| stream.index() == index)
            .map(|(_, packet)| packet)
            .collect();
        let options = CutDetectorOptions {
            lookahead: 30,
            ..CutDetectorOptions::default()
        };
        let cuts = |kept: &[MediaBuffer]| -> Vec<Option<i64>> {
            kept.iter()
                .filter(|buf| {
                    buf.metadata()
                        .is_some_and(|metadata| metadata.get::<SceneCut>().is_some())
                })
                .map(|buf| match buf {
                    MediaBuffer::Video(frame) => frame.pts(),
                    _ => None,
                })
                .collect()
        };
        let packet = |packet: &ffmpeg::Packet| {
            let mut packet = packet.clone();
            packet.set_time_base(time_base);
            MediaBuffer::Packet(Arc::new(packet).into())
        };

        let mut decoder = match D3d12Decoder::new("decoder", parameters.clone(), &gpu) {
            Ok(decoder) => decoder,
            Err(crate::elements::D3d12DecoderError::HwDeviceInit(code)) => {
                eprintln!("skipping: this device does not do video decoding (code {code})");
                return;
            }
            Err(error) => panic!("the fixture did not open: {error}"),
        };
        let mut detector = D3d12CutDetector::new("cuts", &gpu, options).unwrap();
        let kept = capture(&mut detector);
        decoder.src_pads()[0].link(Box::new(detector));
        for packet_of in &packets {
            decoder
                .consume(packet(packet_of))
                .expect("decodes and judges");
        }
        deliver(&mut decoder, &StreamEvent::Eos).expect("drains");

        let mut cpu_decoder = SwDecoder::new("cpu", parameters).expect("opens");
        let mut cpu = SwCutDetector::new("cpu cuts", options).unwrap();
        let cpu_kept = capture(&mut cpu);
        cpu_decoder.src_pads()[0].link(Box::new(cpu));
        for packet_of in &packets {
            cpu_decoder.consume(packet(packet_of)).expect("decodes");
        }
        deliver(&mut cpu_decoder, &StreamEvent::Eos).expect("drains");

        let (kept, cpu_kept) = (kept.lock().unwrap(), cpu_kept.lock().unwrap());
        assert_eq!(kept.len(), packets.len(), "every picture comes out");
        let order: Vec<Option<i64>> = kept
            .iter()
            .map(|buf| match buf {
                MediaBuffer::Video(frame) => frame.pts(),
                _ => None,
            })
            .collect();
        assert!(order.is_sorted(), "in order");
        assert_eq!(cuts(&kept), cuts(&cpu_kept));
    }
}
