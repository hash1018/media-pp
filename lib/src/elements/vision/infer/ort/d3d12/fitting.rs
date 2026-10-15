//! What the D3D12 detector runs on the GPU ahead of the model: the compute
//! shader that fits a D3D12 picture — or a rectangle of it — into a model's
//! input, and that input, a buffer on the GPU that DirectML reads where it
//! is.

use std::mem::ManuallyDrop;

use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE},
        Graphics::{
            Direct3D::{
                Fxc::{D3D_BLOB_ROOT_SIGNATURE, D3DGetBlobPart},
                ID3DBlob,
            },
            Direct3D12::*,
            Dxgi::Common::{
                DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM,
                DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
            },
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
    },
    core::{Interface, s},
};

use crate::color::ColorDescription;
use crate::ffmpeg;
use crate::orientation::Orientation;
use crate::platform::windows::{
    d3d12_gpu::D3d12Gpu, d3d12va::d3d12va_texture, hlsl::compile_shader,
};

use super::super::{ChannelOrder, ModelInput, OrtError};

const SHADER: &[u8] = include_bytes!("../../../../../shaders/d3d12/fit.hlsl");

/// Each run of the shader is 16 by 16 pixels of the input.
const GROUP: u32 = 16;

/// A D3D12 picture a [`Fitting`] can read: NV12, on the fitting's device,
/// no smaller than the picture says it is, with what the GPU must wait for
/// before reading it.
pub(super) struct Picture {
    texture: ID3D12Resource,
    /// Signalled at `fence_value` once whatever wrote the picture has.
    fence: ID3D12Fence,
    fence_value: u64,
    /// Its width and height.
    pub(super) size: (u32, u32),
    /// The rows its own colour description makes RGB with.
    rows: [[f32; 4]; 3],
}

// SAFETY: the texture and fence are free-threaded COM objects, only read
// through `&self`, and the picture is handed between threads whole.
unsafe impl Send for Picture {}

impl Picture {
    /// `frame`, where it is a picture a fitting on `device` reads.
    pub(super) fn of(
        frame: &ffmpeg::frame::Video,
        device: &ID3D12Device,
    ) -> Result<Self, OrtError> {
        let (texture, fence, fence_value) = d3d12va_texture(frame)
            .filter(|(texture, fence, _)| !texture.is_null() && !fence.is_null())
            .ok_or_else(|| OrtError::UnsupportedD3d12Picture(format!("{:?}", frame.format())))?;
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
        if made_on.is_none_or(|made_on| made_on.as_raw() != device.as_raw()) {
            return Err(OrtError::D3d12DeviceMismatch);
        }
        // SAFETY: a live resource's plain description, returned by value.
        let desc = unsafe { texture.GetDesc() };
        if desc.Format != DXGI_FORMAT_NV12 {
            return Err(OrtError::UnsupportedD3d12Picture(format_name(desc.Format)));
        }
        let size = (frame.width(), frame.height());
        let surface = (desc.Width as u32, desc.Height);
        if size.0 == 0 || size.1 == 0 || size.0 > surface.0 || size.1 > surface.1 {
            return Err(OrtError::PictureOutsideTexture {
                picture: size,
                texture: surface,
            });
        }
        Ok(Self {
            texture,
            fence,
            fence_value,
            size,
            rows: ColorDescription::of(frame).yuv_to_rgb_rows(frame.height()),
        })
    }
}

/// What a texture format is called, for an error.
fn format_name(format: DXGI_FORMAT) -> String {
    format!("a texture of DXGI format {}", format.0)
}

/// One rectangle of a picture fitted into one input: `crop` — left, top,
/// width, height, inside the picture as stored — turned as `orientation`
/// says, scaled to `scaled` and placed at `offset` in input `slot`, the
/// rest grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Cut {
    pub(super) crop: (u32, u32, u32, u32),
    pub(super) orientation: Orientation,
    pub(super) offset: (u32, u32),
    pub(super) scaled: (u32, u32),
    pub(super) slot: usize,
}

/// The shader, and `capacity` inputs of a model's size in a buffer on the
/// GPU — apart from any session, so a test can check them without a model.
///
/// Its work goes on the [`D3d12Gpu`]'s queue, which DirectML is given too: a
/// model run after a fit is ordered after it by the queue alone.
pub(super) struct Fitting {
    pub(super) model: (u32, u32),
    capacity: usize,
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    root_signature: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
    /// Two views — luma, chroma — for each picture of a fit, shader-visible.
    heap: ID3D12DescriptorHeap,
    heap_step: usize,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// Signalled at `submitted` once the last fit has run.
    fence: ID3D12Fence,
    event: HANDLE,
    submitted: u64,
    /// The inputs, each three planes of `f32`, which the shader writes and
    /// DirectML reads.
    tensor: ID3D12Resource,
}

// SAFETY: the D3D12 objects are free-threaded; the command list, allocator
// and heap are only touched through `&mut self`, by the one thread
// transforming at a time.
unsafe impl Send for Fitting {}

impl Fitting {
    /// The shader on `gpu`'s device, and `capacity` inputs of `model`'s
    /// size.
    pub(super) fn new(
        gpu: &D3d12Gpu,
        model: (u32, u32),
        capacity: usize,
    ) -> Result<Self, OrtError> {
        let device = gpu.device().clone();
        let capacity = capacity.max(1);
        // SAFETY: every call below is on a live device with descriptions
        // built here; the shader's blob outlives the calls that read it.
        unsafe {
            let shader = compile_shader(SHADER, s!("fit.hlsl"), s!("fit_nv12"), s!("cs_5_1"))?;
            let root = D3DGetBlobPart(
                shader.GetBufferPointer(),
                shader.GetBufferSize(),
                D3D_BLOB_ROOT_SIGNATURE,
                0,
            )?;
            let root_signature: ID3D12RootSignature =
                device.CreateRootSignature(0, blob_bytes(&root))?;
            let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: shader.GetBufferPointer(),
                    BytecodeLength: shader.GetBufferSize(),
                },
                ..Default::default()
            };
            let pipeline = device.CreateComputePipelineState(&desc);
            ManuallyDrop::drop(&mut desc.pRootSignature);
            let pipeline: ID3D12PipelineState = pipeline?;
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2 * capacity as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let heap_step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            // Created recording; closed so the first fit's `Reset` finds it
            // as it expects.
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let tensor = match buffer(
                &device,
                (capacity * Self::floats(model) * size_of::<f32>()) as u64,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ) {
                Ok(tensor) => tensor,
                Err(error) => {
                    let _ = CloseHandle(event);
                    return Err(error.into());
                }
            };
            Ok(Self {
                model,
                capacity,
                device,
                queue: gpu.queue().clone(),
                root_signature,
                pipeline,
                heap,
                heap_step,
                allocator,
                list,
                fence,
                event,
                submitted: 0,
                tensor,
            })
        }
    }

    /// `frame` as a picture this fitting reads.
    pub(super) fn picture(&self, frame: &ffmpeg::frame::Video) -> Result<Picture, OrtError> {
        Picture::of(frame, &self.device)
    }

    /// How many `f32`s one input holds.
    fn floats(model: (u32, u32)) -> usize {
        3 * model.0 as usize * model.1 as usize
    }

    /// How many inputs it holds.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// The buffer the inputs are in, for DirectML to read.
    pub(super) fn tensor(&self) -> &ID3D12Resource {
        &self.tensor
    }

    /// Waits for the last fit to have run, so its list, allocator and views
    /// can be used again.
    fn wait_idle(&self) -> Result<(), OrtError> {
        // SAFETY: a live fence and event; the wait is for a value this
        // fitting signals.
        unsafe {
            if self.fence.GetCompletedValue() < self.submitted {
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)?;
                WaitForSingleObject(self.event, INFINITE);
            }
        }
        Ok(())
    }

    /// Fits each picture's cuts into their inputs, into the values a model
    /// wants — its planes in its own order, each then times its scale plus
    /// its bias — in one submission to the queue, after whatever wrote each
    /// picture. It returns once the work is queued: what reads the inputs
    /// next on the same queue reads them fitted.
    ///
    /// # Panics
    ///
    /// If a cut's slot is past the capacity, there are more pictures than
    /// inputs, or a crop is not inside its picture: the callers' own
    /// arithmetic.
    pub(super) fn fit_each(
        &mut self,
        pictures: &[(&Picture, &[Cut])],
        values: ModelInput,
    ) -> Result<(), OrtError> {
        assert!(
            pictures.len() <= self.capacity,
            "{} pictures for {} inputs",
            pictures.len(),
            self.capacity
        );
        let ModelInput { scale, bias, order } = values;
        self.wait_idle()?;
        // SAFETY: the list and allocator are idle (waited for above) and
        // recorded by this thread alone; every view and barrier names a
        // live resource the pictures or this fitting hold until the queue
        // has run what reads them.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, &self.pipeline)?;
            let list = &self.list;
            list.SetComputeRootSignature(&self.root_signature);
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootUnorderedAccessView(2, self.tensor.GetGPUVirtualAddress());
            let cpu_start = self.heap.GetCPUDescriptorHandleForHeapStart();
            let gpu_start = self.heap.GetGPUDescriptorHandleForHeapStart();
            for (index, &(picture, cuts)) in pictures.iter().enumerate() {
                for (plane, format) in [DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM]
                    .into_iter()
                    .enumerate()
                {
                    let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: cpu_start.ptr + (2 * index + plane) * self.heap_step,
                    };
                    self.device.CreateShaderResourceView(
                        &picture.texture,
                        Some(&plane_view(format, plane as u32)),
                        handle,
                    );
                }
                transition(
                    list,
                    &picture.texture,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
                list.SetComputeRootDescriptorTable(
                    1,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: gpu_start.ptr + (2 * index * self.heap_step) as u64,
                    },
                );
                for cut in cuts {
                    let constants = self.constants(picture, cut, (scale, bias), order);
                    list.SetComputeRoot32BitConstants(0, 40, constants.as_ptr().cast(), 0);
                    list.Dispatch(
                        self.model.0.div_ceil(GROUP),
                        self.model.1.div_ceil(GROUP),
                        1,
                    );
                }
                // Back to `COMMON`: the texture is its producer's again once
                // this list has run.
                transition(
                    list,
                    &picture.texture,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
            }
            uav_barrier(list, &self.tensor);
            list.Close()?;
            // A GPU-side wait for whatever wrote each picture: sharing a
            // device does not order this queue after the decoder's.
            for &(picture, _) in pictures {
                self.queue.Wait(&picture.fence, picture.fence_value)?;
            }
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
            self.submitted += 1;
            self.queue.Signal(&self.fence, self.submitted)?;
        }
        Ok(())
    }

    /// The shader's forty constants for `cut` of `picture`, in `fit.hlsl`'s
    /// `Fit` order.
    fn constants(
        &self,
        picture: &Picture,
        cut: &Cut,
        (scale, bias): ([f32; 3], [f32; 3]),
        order: ChannelOrder,
    ) -> [u32; 40] {
        let (left, top, width, height) = cut.crop;
        assert!(
            cut.slot < self.capacity,
            "input {} of {}",
            cut.slot,
            self.capacity
        );
        assert!(
            width > 0
                && height > 0
                && left + width <= picture.size.0
                && top + height <= picture.size.1,
            "{:?} inside a {:?} picture",
            cut.crop,
            picture.size
        );
        let shown = cut.orientation.display_size(width, height);
        let [xx, xy, x0, yx, yy, y0] = cut.orientation.sampling(width, height);
        let mut words = [0u32; 40];
        words[..12].copy_from_slice(&[
            self.model.0,
            self.model.1,
            cut.offset.0,
            cut.offset.1,
            cut.scaled.0,
            cut.scaled.1,
            shown.0,
            shown.1,
            left,
            top,
            cut.slot as u32,
            u32::from(order == ChannelOrder::Bgr),
        ]);
        for (word, value) in words[12..20].iter_mut().zip([xx, xy, x0, 0, yx, yy, y0, 0]) {
            *word = value as u32;
        }
        let affine = [
            [scale[0], scale[1], scale[2], 1.0],
            [bias[0], bias[1], bias[2], 0.0],
        ];
        for (word, value) in words[20..]
            .iter_mut()
            .zip(picture.rows.iter().chain(&affine).flatten())
        {
            *word = value.to_bits();
        }
        words
    }

    /// The first `inputs` inputs as the shader left them, waited for and
    /// read back: what a test checks the shader by.
    #[cfg(test)]
    pub(super) fn read_back(&mut self, inputs: usize) -> Result<Vec<f32>, OrtError> {
        let floats = inputs * Self::floats(self.model);
        let bytes = (floats * size_of::<f32>()) as u64;
        self.wait_idle()?;
        // SAFETY: as in `fit_each`; the read-back buffer is mapped only once
        // the copy into it has run, and unmapped before it is dropped.
        unsafe {
            let readback = buffer(
                &self.device,
                bytes,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.list
                .CopyBufferRegion(&readback, 0, &self.tensor, 0, bytes);
            self.list.Close()?;
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
            self.submitted += 1;
            self.queue.Signal(&self.fence, self.submitted)?;
            self.wait_idle()?;
            let mut mapped = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut mapped))?;
            let values = std::slice::from_raw_parts(mapped.cast::<f32>(), floats).to_vec();
            readback.Unmap(0, None);
            Ok(values)
        }
    }
}

impl Drop for Fitting {
    fn drop(&mut self) {
        // What is queued still reads the views and writes the buffer: it
        // runs out before they go.
        let _ = self.wait_idle();
        // SAFETY: the event was made by `new` and is closed once, here.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

/// A buffer of `bytes` on `heap`.
unsafe fn buffer(
    device: &ID3D12Device,
    bytes: u64,
    heap: D3D12_HEAP_TYPE,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: descriptions built here, and a correctly typed out-parameter.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: bytes.max(1),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
            },
            state,
            None,
            &mut resource,
        )?;
    }
    Ok(resource.expect("CreateCommittedResource succeeded without a resource"))
}

/// One plane of an NV12 texture: the format picks which, the plane slice
/// says where.
fn plane_view(format: DXGI_FORMAT, plane_slice: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: plane_slice,
                ResourceMinLODClamp: 0.0,
            },
        },
    }
}

/// Records `resource` going from `before` to `after`.
unsafe fn transition(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: records one barrier on an open list, then releases the
    // resource reference the barrier description took.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        let transition = &mut *barrier.Anonymous.Transition;
        ManuallyDrop::drop(&mut transition.pResource);
    }
}

/// Records a barrier between the shader's writes to `resource` and what
/// reads it after them.
unsafe fn uav_barrier(list: &ID3D12GraphicsCommandList, resource: &ID3D12Resource) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
            }),
        },
    };
    // SAFETY: records one barrier on an open list, then releases the
    // resource reference the barrier description took.
    unsafe {
        list.ResourceBarrier(std::slice::from_ref(&barrier));
        let uav = &mut *barrier.Anonymous.UAV;
        ManuallyDrop::drop(&mut uav.pResource);
    }
}

/// The bytes a blob holds.
unsafe fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: a blob owns `GetBufferSize()` readable bytes at
    // `GetBufferPointer()` for as long as it lives.
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::buffer::MediaBuffer;
    use crate::element::{ElementType, RawSink, SrcPads, element_pp_log};
    use crate::elements::D3d12Upload;
    use crate::orientation::Rotation;
    use crate::pool::UnboundObjectPool;
    use crate::test_support::{CapturingSink, try_d3d12_gpu};

    /// An NV12 picture whose every pixel differs from its neighbours, as it
    /// was in system memory and as a D3D12 picture on `gpu`; `None` where
    /// FFmpeg cannot make D3D12 frames.
    pub(in super::super) fn upload(
        gpu: &D3d12Gpu,
        (width, height): (u32, u32),
    ) -> Option<(ffmpeg::frame::Video, MediaBuffer)> {
        let Ok(mut upload) = D3d12Upload::new("upload", gpu) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return None;
        };
        let received = Arc::new(Mutex::new(Vec::new()));
        upload.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height),
            |_| {},
        );
        let mut source = pool.get();
        for row in 0..height as usize {
            let stride = source.stride(0);
            let data = source.data_mut(0);
            for column in 0..width as usize {
                data[row * stride + column] = (16 + (row * 7 + column * 13) % 220) as u8;
            }
        }
        for row in 0..height as usize / 2 {
            let stride = source.stride(1);
            let data = source.data_mut(1);
            for column in 0..width as usize / 2 {
                data[row * stride + 2 * column] = (64 + (row * 5 + column * 3) % 128) as u8;
                data[row * stride + 2 * column + 1] = (192 - (row * 3 + column * 11) % 128) as u8;
            }
        }
        ColorDescription::BT709_LIMITED.describe(&mut source);
        let copy = (*source).clone();
        upload
            .consume(MediaBuffer::Video(Arc::new(source).into()))
            .expect("uploads");
        let uploaded = received.lock().unwrap().remove(0);
        Some((copy, uploaded))
    }

    /// What the shader is to write for `cut` of `frame`, an NV12 picture in
    /// system memory: each input pixel the mean of the pixels it covers,
    /// made RGB by the frame's own rows, and the margins grey — three
    /// planes, R, G and B, of one input.
    pub(in super::super) fn expected(
        frame: &ffmpeg::frame::Video,
        cut: &Cut,
        model: (u32, u32),
    ) -> Vec<f32> {
        let rows = ColorDescription::of(frame).yuv_to_rgb_rows(frame.height());
        let (left, top, width, height) = cut.crop;
        let shown = cut.orientation.display_size(width, height);
        let [xx, xy, x0, yx, yy, y0] = cut.orientation.sampling(width, height);
        let plane = (model.0 * model.1) as usize;
        let mut tensor = vec![114.0 / 255.0; 3 * plane];
        let (luma, luma_stride) = (frame.data(0), frame.stride(0));
        let (chroma, chroma_stride) = (frame.data(1), frame.stride(1));
        let span = |inside: u32, source: u32, scaled: u32| {
            let lo = inside * source / scaled;
            let hi = ((inside + 1) * source)
                .div_ceil(scaled)
                .min(source)
                .max(lo + 1);
            lo..hi
        };
        for y in 0..cut.scaled.1 {
            for x in 0..cut.scaled.0 {
                let mut sum = [0.0f32; 3];
                let mut count = 0.0;
                for sy in span(y, shown.1, cut.scaled.1) {
                    for sx in span(x, shown.0, cut.scaled.0) {
                        let (sx, sy) = (sx as i32, sy as i32);
                        let px = (left as i32 + xx * sx + xy * sy + x0) as usize;
                        let py = (top as i32 + yx * sx + yy * sy + y0) as usize;
                        sum[0] += f32::from(luma[py * luma_stride + px]) / 255.0;
                        let c = (py / 2) * chroma_stride + (px / 2) * 2;
                        sum[1] += f32::from(chroma[c]) / 255.0;
                        sum[2] += f32::from(chroma[c + 1]) / 255.0;
                        count += 1.0;
                    }
                }
                let yuv = [sum[0] / count, sum[1] / count, sum[2] / count, 1.0];
                let at = ((cut.offset.1 + y) * model.0 + cut.offset.0 + x) as usize;
                for (channel, row) in rows.iter().enumerate() {
                    let value: f32 = row.iter().zip(yuv).map(|(r, v)| r * v).sum();
                    tensor[channel * plane + at] = value.clamp(0.0, 1.0);
                }
            }
        }
        tensor
    }

    fn largest_difference(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// The shader writes what the CPU works out for the same picture:
    /// shrunk whole into the input with grey around it, and a rectangle of
    /// it turned a quarter, as a phone's portrait recording is shown, into
    /// the second input of the same fit.
    #[test]
    fn the_shader_fits_a_picture_as_the_cpu_works_it_out() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some((frame, uploaded)) = upload(&gpu, (96, 64)) else {
            return;
        };
        let MediaBuffer::Video(on_gpu) = &uploaded else {
            panic!("a picture goes on");
        };
        let model = (40, 40);
        let mut fitting = Fitting::new(&gpu, model, 2).expect("the shader compiles");
        let picture = fitting.picture(on_gpu).expect("reads the upload");
        assert_eq!(picture.size, (96, 64));
        let whole = Cut {
            crop: (0, 0, 96, 64),
            orientation: Orientation::UPRIGHT,
            offset: (0, 7),
            scaled: (40, 26),
            slot: 0,
        };
        let turned = Cut {
            crop: (32, 16, 48, 32),
            orientation: Orientation::rotated(Rotation::Clockwise90),
            offset: (7, 0),
            scaled: (26, 40),
            slot: 1,
        };
        fitting
            .fit_each(&[(&picture, &[whole, turned])], ModelInput::default())
            .expect("fits");
        let tensor = fitting.read_back(2).expect("reads back");
        let floats = Fitting::floats(model);
        for (cut, written) in [(whole, &tensor[..floats]), (turned, &tensor[floats..])] {
            let wanted = expected(&frame, &cut, model);
            let off = largest_difference(written, &wanted);
            assert!(off < 1e-3, "{cut:?}: off by {off}");
        }
    }

    /// A model's own values — its planes B, G, R, each times a scale plus a
    /// bias — are the shader's to write.
    #[test]
    fn the_shader_writes_the_values_a_model_wants() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some((frame, uploaded)) = upload(&gpu, (64, 64)) else {
            return;
        };
        let MediaBuffer::Video(on_gpu) = &uploaded else {
            panic!("a picture goes on");
        };
        let model = (32, 32);
        let mut fitting = Fitting::new(&gpu, model, 1).expect("the shader compiles");
        let picture = fitting.picture(on_gpu).expect("reads the upload");
        let cut = Cut {
            crop: (0, 0, 64, 64),
            orientation: Orientation::UPRIGHT,
            offset: (0, 0),
            scaled: (32, 32),
            slot: 0,
        };
        let values = ModelInput {
            order: ChannelOrder::Bgr,
            scale: [255.0; 3],
            bias: [-104.0, -117.0, -123.0],
        };
        fitting
            .fit_each(&[(&picture, &[cut])], values)
            .expect("fits");
        let written = fitting.read_back(1).expect("reads back");
        let rgb = expected(&frame, &cut, model);
        let plane = (model.0 * model.1) as usize;
        let mut wanted = vec![0.0; 3 * plane];
        for at in 0..plane {
            for (channel, from) in [2, 1, 0].into_iter().enumerate() {
                wanted[channel * plane + at] =
                    rgb[from * plane + at] * values.scale[channel] + values.bias[channel];
            }
        }
        let off = largest_difference(&written, &wanted);
        assert!(off < 0.3, "off by {off} of 255");
    }

    /// A picture of another device is refused as that, before anything is
    /// recorded.
    #[test]
    fn a_picture_of_another_device_is_refused() {
        let (Some(gpu), Some(other)) = (try_d3d12_gpu(), try_d3d12_gpu()) else {
            return;
        };
        if gpu.device().as_raw() == other.device().as_raw() {
            eprintln!("skipping: both devices are one");
            return;
        }
        let Some((_, uploaded)) = upload(&other, (32, 32)) else {
            return;
        };
        let MediaBuffer::Video(on_other) = &uploaded else {
            panic!("a picture goes on");
        };
        let fitting = Fitting::new(&gpu, (32, 32), 1).expect("the shader compiles");
        assert!(matches!(
            fitting.picture(on_other),
            Err(OrtError::D3d12DeviceMismatch)
        ));
    }
}
