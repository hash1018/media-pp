//! What the D3D12 detector runs on the GPU ahead of the model: the compute
//! shader that fits a D3D12 picture — or a rectangle of it — into a model's
//! input, and that input, a buffer on the GPU that DirectML reads where it
//! is. An HDR picture is first brought to SDR by a shader of its own, as the
//! CUDA and Metal detectors look at one.

use windows::{
    Win32::Graphics::{
        Direct3D12::*,
        Dxgi::Common::{
            DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_FORMAT_P010, DXGI_FORMAT_R8_UNORM,
            DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16_UNORM,
            DXGI_FORMAT_R16G16_UNORM,
        },
    },
    core::{Interface, s},
};

use crate::color::ColorDescription;
use crate::ffmpeg;
use crate::orientation::Orientation;
use crate::platform::windows::{
    d3d12_compute::{
        Commands, Kernel, buffer, plane_view, plane_write_view, transition, uav_barrier,
        writable_texture,
    },
    d3d12_gpu::D3d12Gpu,
    d3d12va::d3d12va_texture,
};
use crate::tone_map::ToneMap;

use super::super::{ChannelOrder, ModelInput, OrtError};

const SHADER: &[u8] = include_bytes!("../../../../../shaders/d3d12/fit.hlsl");
const WARP_SHADER: &[u8] = include_bytes!("../../../../../shaders/d3d12/warp.hlsl");
const TONE_MAP_SHADER: &[u8] = include_bytes!("../../../../../shaders/d3d12/tone_map.hlsl");

/// Each run of the shader is 16 by 16 pixels of the input.
const GROUP: u32 = 16;

/// A D3D12 picture a [`Fitting`] can read: NV12, or HDR P010 tagged HLG or
/// PQ, on the fitting's device, no smaller than the picture says it is,
/// with what the GPU must wait for before reading it.
pub(super) struct Picture {
    texture: ID3D12Resource,
    /// Signalled at `fence_value` once whatever wrote the picture has.
    fence: ID3D12Fence,
    fence_value: u64,
    /// Its width and height.
    pub(super) size: (u32, u32),
    /// The rows its own colour description makes RGB with.
    rows: [[f32; 4]; 3],
    /// For an HDR picture, `tone_map.hlsl`'s constants to bring it to SDR
    /// with, which is what is fitted.
    hdr: Option<[u32; 32]>,
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
        let size = (frame.width(), frame.height());
        let hdr = match desc.Format {
            DXGI_FORMAT_NV12 => None,
            // A ten-bit picture tagged neither HLG nor PQ has no tone map.
            DXGI_FORMAT_P010 => Some(tone_map_constants(
                &ToneMap::of_frame(frame).ok_or_else(|| {
                    OrtError::UnsupportedD3d12Picture("a P010 one tagged neither HLG nor PQ".into())
                })?,
                size,
            )),
            other => return Err(OrtError::UnsupportedD3d12Picture(format_name(other))),
        };
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
            hdr,
        })
    }
}

/// `tone_map.hlsl`'s thirty-two constants for `map`, on a picture of
/// `size`, in its `ToneMap` order.
fn tone_map_constants(map: &ToneMap, size: (u32, u32)) -> [u32; 32] {
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
    /// `fit.hlsl`'s, for a detector or a classifier: an NV12 picture's, and
    /// an SDR copy's under the same root signature.
    fit: Kernel,
    fit_rgba: ID3D12PipelineState,
    /// `warp.hlsl`'s, for an embedder, likewise.
    warp: Kernel,
    warp_rgba: ID3D12PipelineState,
    /// `tone_map.hlsl`'s, which makes an HDR picture's SDR copy.
    tone_map: Kernel,
    /// The SDR copy of the HDR picture of each index of a fit, made the
    /// first time one is needed and again for a larger picture.
    copies: Vec<Option<(ID3D12Resource, (u32, u32))>>,
    /// For each picture of a fit, shader-visible: two views — luma and
    /// chroma, or its SDR copy twice — from the start, and three for
    /// making its SDR copy — luma, chroma, the copy — after `2 * capacity`.
    heap: ID3D12DescriptorHeap,
    heap_step: usize,
    commands: Commands,
    /// The inputs, each three planes of `f32`, which the shader writes and
    /// DirectML reads.
    tensor: ID3D12Resource,
    /// What the inputs are copied into for the CPU to read, made the first
    /// time they are.
    readback: Option<ID3D12Resource>,
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
        // built here.
        unsafe {
            let fit = Kernel::new(&device, SHADER, s!("fit.hlsl"), s!("fit_nv12"))?;
            let fit_rgba = fit.sibling(&device, SHADER, s!("fit.hlsl"), s!("fit_rgba"))?;
            let warp = Kernel::new(&device, WARP_SHADER, s!("warp.hlsl"), s!("warp_nv12"))?;
            let warp_rgba = warp.sibling(&device, WARP_SHADER, s!("warp.hlsl"), s!("warp_rgba"))?;
            let tone_map = Kernel::new(
                &device,
                TONE_MAP_SHADER,
                s!("tone_map.hlsl"),
                s!("hdr_to_rgba"),
            )?;
            let heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 5 * capacity as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let heap_step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let commands = Commands::new(&device, gpu.queue())?;
            let tensor = buffer(
                &device,
                (capacity * Self::floats(model) * size_of::<f32>()) as u64,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?;
            Ok(Self {
                model,
                capacity,
                device,
                fit,
                fit_rgba,
                warp,
                warp_rgba,
                tone_map,
                copies: Vec::new(),
                heap,
                heap_step,
                commands,
                tensor,
                readback: None,
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
        let list = self.commands.begin(Some(&self.fit.pipeline))?;
        // SAFETY: the list is open, the GPU idle on this fitting's work
        // (`begin` waited); every view and barrier names a live resource the
        // pictures or this fitting hold until the queue has run what reads
        // them.
        unsafe {
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            let only: Vec<&Picture> = pictures.iter().map(|&(picture, _)| picture).collect();
            let copies = self.record_copies(&list, &only)?;
            list.SetComputeRootSignature(&self.fit.root_signature);
            list.SetComputeRootUnorderedAccessView(2, self.tensor.GetGPUVirtualAddress());
            for (index, (&(picture, cuts), copy)) in pictures.iter().zip(&copies).enumerate() {
                let pipeline = match copy {
                    None => &self.fit.pipeline,
                    Some(_) => &self.fit_rgba,
                };
                self.bind(&list, index, picture, copy.as_ref(), pipeline);
                for cut in cuts {
                    let constants = self.constants(picture, cut, (scale, bias), order);
                    list.SetComputeRoot32BitConstants(0, 40, constants.as_ptr().cast(), 0);
                    list.Dispatch(
                        self.model.0.div_ceil(GROUP),
                        self.model.1.div_ceil(GROUP),
                        1,
                    );
                }
                release(&list, picture, copy.as_ref());
            }
            uav_barrier(&list, &self.tensor);
        }
        let waits: Vec<(&ID3D12Fence, u64)> = pictures
            .iter()
            .map(|&(picture, _)| (&picture.fence, picture.fence_value))
            .collect();
        self.commands.submit(&waits)?;
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

    /// Records making the SDR copy of each HDR picture of `pictures` — the
    /// one at `index` into copy `index`, through the views after `2 *
    /// capacity` — and says what each picture is fitted from: itself, or
    /// its copy, left for a shader to read. Where it makes any, the list is
    /// left with the tone map's root signature, which the caller replaces.
    ///
    /// # Safety
    ///
    /// `list` is this fitting's, open, its descriptor heap set, and the GPU
    /// idle on this fitting's work, so a copy may be replaced.
    unsafe fn record_copies(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        pictures: &[&Picture],
    ) -> Result<Vec<Option<ID3D12Resource>>, OrtError> {
        let mut copies = Vec::with_capacity(pictures.len());
        let mut bound = false;
        // SAFETY: the caller's contract; every view and barrier names a live
        // resource the pictures or this fitting hold until the queue has run
        // what reads them.
        unsafe {
            let cpu_start = self.heap.GetCPUDescriptorHandleForHeapStart();
            let gpu_start = self.heap.GetGPUDescriptorHandleForHeapStart();
            for (index, picture) in pictures.iter().enumerate() {
                let Some(constants) = &picture.hdr else {
                    copies.push(None);
                    continue;
                };
                if self.copies.len() <= index {
                    self.copies.resize_with(index + 1, || None);
                }
                let copy = match &self.copies[index] {
                    Some((copy, size)) if size.0 >= picture.size.0 && size.1 >= picture.size.1 => {
                        copy.clone()
                    }
                    _ => {
                        let made = writable_texture(
                            &self.device,
                            DXGI_FORMAT_R8G8B8A8_UNORM,
                            picture.size,
                        )?;
                        self.copies[index] = Some((made.clone(), picture.size));
                        made
                    }
                };
                if !bound {
                    list.SetComputeRootSignature(&self.tone_map.root_signature);
                    list.SetPipelineState(&self.tone_map.pipeline);
                    bound = true;
                }
                let first = 2 * self.capacity + 3 * index;
                let handle = |at: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_start.ptr + (first + at) * self.heap_step,
                };
                for (plane, format) in [DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_UNORM]
                    .into_iter()
                    .enumerate()
                {
                    self.device.CreateShaderResourceView(
                        &picture.texture,
                        Some(&plane_view(format, plane as u32)),
                        handle(plane),
                    );
                }
                self.device.CreateUnorderedAccessView(
                    &copy,
                    None,
                    Some(&plane_write_view(DXGI_FORMAT_R8G8B8A8_UNORM, 0)),
                    handle(2),
                );
                transition(
                    list,
                    &picture.texture,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
                list.SetComputeRootDescriptorTable(
                    1,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: gpu_start.ptr + (first * self.heap_step) as u64,
                    },
                );
                list.SetComputeRoot32BitConstants(0, 32, constants.as_ptr().cast(), 0);
                list.Dispatch(
                    picture.size.0.div_ceil(GROUP),
                    picture.size.1.div_ceil(GROUP),
                    1,
                );
                transition(
                    list,
                    &picture.texture,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                transition(
                    list,
                    &copy,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );
                copies.push(Some(copy));
            }
        }
        Ok(copies)
    }

    /// Records reading `picture` — or its SDR `copy` — with `pipeline`: its
    /// views at `2 * index`, the table set to them, and the texture made
    /// readable; [`release`] undoes the last.
    ///
    /// # Safety
    ///
    /// As [`Self::record_copies`], the root signature `pipeline`'s.
    unsafe fn bind(
        &self,
        list: &ID3D12GraphicsCommandList,
        index: usize,
        picture: &Picture,
        copy: Option<&ID3D12Resource>,
        pipeline: &ID3D12PipelineState,
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            let cpu_start = self.heap.GetCPUDescriptorHandleForHeapStart();
            let gpu_start = self.heap.GetGPUDescriptorHandleForHeapStart();
            let handle = |plane: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start.ptr + (2 * index + plane) * self.heap_step,
            };
            match copy {
                None => {
                    for (plane, format) in [DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM]
                        .into_iter()
                        .enumerate()
                    {
                        self.device.CreateShaderResourceView(
                            &picture.texture,
                            Some(&plane_view(format, plane as u32)),
                            handle(plane),
                        );
                    }
                    transition(
                        list,
                        &picture.texture,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    );
                }
                // The copy twice: the table has two views, of which the
                // shader reads the first.
                Some(copy) => {
                    for plane in 0..2 {
                        self.device.CreateShaderResourceView(
                            copy,
                            Some(&plane_view(DXGI_FORMAT_R8G8B8A8_UNORM, 0)),
                            handle(plane),
                        );
                    }
                }
            }
            list.SetPipelineState(pipeline);
            list.SetComputeRootDescriptorTable(
                1,
                D3D12_GPU_DESCRIPTOR_HANDLE {
                    ptr: gpu_start.ptr + (2 * index * self.heap_step) as u64,
                },
            );
        }
    }

    /// Reads each of `maps` of `picture` into the input of its index, in the
    /// model's `values`, in one submission to the queue after whatever wrote
    /// the picture: input pixel `(u, v)` from `(m[0] u + m[1] v + m[2], m[3]
    /// u + m[4] v + m[5])` of the picture, between the four pixels around
    /// it, black outside it — what the CUDA and Metal embedders read. It
    /// returns once the work is queued, as [`Self::fit_each`] does.
    ///
    /// # Panics
    ///
    /// If there are more maps than inputs: the caller's own arithmetic.
    pub(super) fn warp(
        &mut self,
        picture: &Picture,
        maps: &[[f32; 6]],
        values: ModelInput,
    ) -> Result<(), OrtError> {
        assert!(
            maps.len() <= self.capacity,
            "{} maps for {} inputs",
            maps.len(),
            self.capacity
        );
        let ModelInput { scale, bias, order } = values;
        let list = self.commands.begin(Some(&self.warp.pipeline))?;
        // SAFETY: as in `fit_each`.
        unsafe {
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            let copy = self.record_copies(&list, &[picture])?.remove(0);
            list.SetComputeRootSignature(&self.warp.root_signature);
            list.SetComputeRootUnorderedAccessView(2, self.tensor.GetGPUVirtualAddress());
            let pipeline = match copy {
                None => &self.warp.pipeline,
                Some(_) => &self.warp_rgba,
            };
            self.bind(&list, 0, picture, copy.as_ref(), pipeline);
            for (slot, map) in maps.iter().enumerate() {
                let mut words = [0u32; 36];
                words[..8].copy_from_slice(&[
                    self.model.0,
                    self.model.1,
                    picture.size.0,
                    picture.size.1,
                    slot as u32,
                    u32::from(order == ChannelOrder::Bgr),
                    0,
                    0,
                ]);
                let floats = [
                    [map[0], map[1], map[2], 0.0],
                    [map[3], map[4], map[5], 0.0],
                    picture.rows[0],
                    picture.rows[1],
                    picture.rows[2],
                    [scale[0], scale[1], scale[2], 1.0],
                    [bias[0], bias[1], bias[2], 0.0],
                ];
                for (word, value) in words[8..].iter_mut().zip(floats.iter().flatten()) {
                    *word = value.to_bits();
                }
                list.SetComputeRoot32BitConstants(0, 36, words.as_ptr().cast(), 0);
                list.Dispatch(
                    self.model.0.div_ceil(GROUP),
                    self.model.1.div_ceil(GROUP),
                    1,
                );
            }
            release(&list, picture, copy.as_ref());
            uav_barrier(&list, &self.tensor);
        }
        self.commands
            .submit(&[(&picture.fence, picture.fence_value)])?;
        Ok(())
    }

    /// The first `inputs` inputs as the last fit or warp left them, waited
    /// for and read back: an embedder's cut-outs, and what a test checks the
    /// shaders by. The buffer they come through is made the first time.
    ///
    /// # Panics
    ///
    /// If `inputs` is past the capacity.
    pub(super) fn read_back(&mut self, inputs: usize) -> Result<Vec<f32>, OrtError> {
        assert!(inputs <= self.capacity, "{inputs} of {}", self.capacity);
        let floats = inputs * Self::floats(self.model);
        let bytes = (floats * size_of::<f32>()) as u64;
        self.commands.wait_idle()?;
        // SAFETY: as in `fit_each`; the read-back buffer is mapped only once
        // the copy into it has run, and unmapped before this returns.
        unsafe {
            let readback = match &self.readback {
                Some(readback) => readback.clone(),
                None => {
                    let made = buffer(
                        &self.device,
                        (self.capacity * Self::floats(self.model) * size_of::<f32>()) as u64,
                        D3D12_HEAP_TYPE_READBACK,
                        D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )?;
                    self.readback.insert(made).clone()
                }
            };
            let list = self.commands.begin(None)?;
            list.CopyBufferRegion(&readback, 0, &self.tensor, 0, bytes);
            self.commands.submit(&[])?;
            self.commands.wait_idle()?;
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
        let _ = self.commands.wait_idle();
    }
}

/// Records `picture` — or its SDR `copy` — no longer read: the picture back
/// to `COMMON`, its producer's again once the list has run, or the copy back
/// to be written.
unsafe fn release(
    list: &ID3D12GraphicsCommandList,
    picture: &Picture,
    copy: Option<&ID3D12Resource>,
) {
    // SAFETY: the caller's open list, and live resources.
    unsafe {
        match copy {
            None => transition(
                list,
                &picture.texture,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            ),
            Some(copy) => transition(
                list,
                copy,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        }
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

    /// The four pixels around `p` in a plane `extent` in pixels, held inside
    /// it, each weighted, naught outside — `warp.hlsl`'s `around`.
    fn around(p: (f32, f32), extent: (usize, usize)) -> ([(usize, usize); 4], [f32; 4], f32) {
        let (cx, cy) = (p.0.floor(), p.1.floor());
        let (tx, ty) = (p.0 - cx, p.1 - cy);
        let weights = [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ];
        let mut at = [(0, 0); 4];
        let mut weight = [0.0; 4];
        let mut inside = 0.0;
        for (i, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let (qx, qy) = (cx as i64 + dx, cy as i64 + dy);
            let within = qx >= 0 && qy >= 0 && qx < extent.0 as i64 && qy < extent.1 as i64;
            at[i] = (
                qx.clamp(0, extent.0 as i64 - 1) as usize,
                qy.clamp(0, extent.1 as i64 - 1) as usize,
            );
            weight[i] = if within { weights[i] } else { 0.0 };
            inside += weight[i];
        }
        (at, weight, inside)
    }

    /// What `warp.hlsl` is to write for `map` of `frame`, RGB 0 to 1.
    fn warped(frame: &ffmpeg::frame::Video, map: [f32; 6], model: (u32, u32)) -> Vec<f32> {
        let rows = ColorDescription::of(frame).yuv_to_rgb_rows(frame.height());
        let size = (frame.width() as usize, frame.height() as usize);
        let (luma, luma_stride) = (frame.data(0), frame.stride(0));
        let (chroma, chroma_stride) = (frame.data(1), frame.stride(1));
        let plane = (model.0 * model.1) as usize;
        let mut tensor = vec![0.0; 3 * plane];
        for v in 0..model.1 {
            for u in 0..model.0 {
                let (fu, fv) = (u as f32, v as f32);
                let p = (
                    map[0] * fu + map[1] * fv + map[2],
                    map[3] * fu + map[4] * fv + map[5],
                );
                let (at, weight, inside) = around(p, size);
                let mut rgb = [0.0f32; 3];
                if inside > 0.0 {
                    let y: f32 = (0..4)
                        .map(|i| {
                            f32::from(luma[at[i].1 * luma_stride + at[i].0]) / 255.0 * weight[i]
                        })
                        .sum::<f32>()
                        / inside;
                    let half = (size.0.div_ceil(2), size.1.div_ceil(2));
                    let (cat, cweight, cinside) =
                        around(((p.0 + 0.5) * 0.5 - 0.5, (p.1 + 0.5) * 0.5 - 0.5), half);
                    let mut c = [0.5f32, 0.5];
                    if cinside > 0.0 {
                        c = [0.0, 0.0];
                        for i in 0..4 {
                            let base = cat[i].1 * chroma_stride + cat[i].0 * 2;
                            c[0] += f32::from(chroma[base]) / 255.0 * cweight[i];
                            c[1] += f32::from(chroma[base + 1]) / 255.0 * cweight[i];
                        }
                        c = [c[0] / cinside, c[1] / cinside];
                    }
                    let yuv = [y, c[0], c[1], 1.0];
                    for (channel, row) in rows.iter().enumerate() {
                        let value: f32 = row.iter().zip(yuv).map(|(r, x)| r * x).sum();
                        rgb[channel] = value.clamp(0.0, 1.0) * inside;
                    }
                }
                let index = (v * model.0 + u) as usize;
                for (channel, value) in rgb.into_iter().enumerate() {
                    tensor[channel * plane + index] = value;
                }
            }
        }
        tensor
    }

    /// The warp reads what the CPU works out for the same maps: one turned
    /// and shrunk, and one hanging off the picture's edge, black past it.
    #[test]
    fn the_warp_reads_an_object_as_the_cpu_works_it_out() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some((frame, uploaded)) = upload(&gpu, (96, 64)) else {
            return;
        };
        let MediaBuffer::Video(on_gpu) = &uploaded else {
            panic!("a picture goes on");
        };
        let model = (24, 24);
        let mut fitting = Fitting::new(&gpu, model, 2).expect("the shaders compile");
        let picture = fitting.picture(on_gpu).expect("reads the upload");
        let (cos, sin) = (0.3f32.cos() * 1.7, 0.3f32.sin() * 1.7);
        let maps = [
            [cos, -sin, 30.5, sin, cos, 12.25],
            [1.25, 0.0, 80.0, 0.0, 1.25, -6.0],
        ];
        fitting
            .warp(&picture, &maps, ModelInput::default())
            .expect("warps");
        let tensor = fitting.read_back(2).expect("reads back");
        let floats = Fitting::floats(model);
        for (slot, map) in maps.into_iter().enumerate() {
            let wanted = warped(&frame, map, model);
            let off = largest_difference(&tensor[slot * floats..(slot + 1) * floats], &wanted);
            assert!(off < 1e-3, "map {slot}: off by {off}");
        }
    }

    /// `frame`, a P010 picture in system memory, as a D3D12 one on `gpu`,
    /// made as a ten-bit decoder's is; `None` where FFmpeg cannot.
    pub(in super::super) fn upload_p010(
        gpu: &D3d12Gpu,
        frame: &ffmpeg::frame::Video,
    ) -> Option<ffmpeg::frame::Video> {
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
            (*context).sw_format = ffi::AVPixelFormat::AV_PIX_FMT_P010LE;
            (*context).width = frame.width() as i32;
            (*context).height = frame.height() as i32;
            if ffi::av_hwframe_ctx_init(frames.as_ptr()) < 0 {
                eprintln!("skipping: FFmpeg could not make P010 D3D12 frames");
                return None;
            }
            let mut on_gpu = ffmpeg::frame::Video::empty();
            assert!(ffi::av_hwframe_get_buffer(frames.as_ptr(), on_gpu.as_mut_ptr(), 0) >= 0);
            assert!(ffi::av_hwframe_transfer_data(on_gpu.as_mut_ptr(), frame.as_ptr(), 0) >= 0);
            assert!(ffi::av_frame_copy_props(on_gpu.as_mut_ptr(), frame.as_ptr()) >= 0);
            Some(on_gpu)
        }
    }

    /// An HDR picture, HLG or PQ, is fitted and warped from an SDR copy
    /// that is `core/tone_map.rs`'s definition, pixel for pixel to within
    /// the rounding of a GPU's powers and of the copy's eight bits, as the
    /// CUDA and Metal detectors look at one; a P010 picture tagged neither
    /// is refused, having no tone map.
    #[test]
    fn an_hdr_picture_is_looked_at_in_sdr() {
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let (width, height) = (64usize, 32usize);
        let model = (width as u32, height as u32);
        let mut fitting = Fitting::new(&gpu, model, 2).expect("the shaders compile");
        // Ten-bit luma across the range, two chroma pairs off grey.
        let luma = |x: usize, y: usize| (64 + (x * 13 + y * 7) % 877) as u16;
        let chroma = |x: usize, y: usize| -> (u16, u16) {
            match (x + y) % 3 {
                0 => (512, 512),
                1 => (400, 620),
                _ => (600, 450),
            }
        };
        for transfer in [
            ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
            ffmpeg::color::TransferCharacteristic::SMPTE2084,
            ffmpeg::color::TransferCharacteristic::BT709,
        ] {
            let mut frame = ffmpeg::frame::Video::new(
                ffmpeg::format::Pixel::P010LE,
                width as u32,
                height as u32,
            );
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
            ColorDescription {
                space: ffmpeg::color::Space::BT2020NCL,
                range: ffmpeg::color::Range::MPEG,
                primaries: ffmpeg::color::Primaries::BT2020,
                transfer,
            }
            .describe(&mut frame);
            let Some(on_gpu) = upload_p010(&gpu, &frame) else {
                return;
            };
            let Some(map) = ToneMap::of_frame(&frame) else {
                assert!(
                    matches!(
                        fitting.picture(&on_gpu),
                        Err(OrtError::UnsupportedD3d12Picture(_))
                    ),
                    "SDR P010 has no tone map"
                );
                continue;
            };
            let picture = fitting.picture(&on_gpu).expect("reads a P010 picture");
            // One input pixel to a picture pixel, fitted into the first
            // input and warped into the second.
            let whole = Cut {
                crop: (0, 0, model.0, model.1),
                orientation: Orientation::UPRIGHT,
                offset: (0, 0),
                scaled: model,
                slot: 0,
            };
            fitting
                .fit_each(&[(&picture, &[whole])], ModelInput::default())
                .expect("fits");
            let fitted = fitting.read_back(1).expect("reads back");
            let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            fitting
                .warp(&picture, &[identity], ModelInput::default())
                .expect("warps");
            let warped = fitting.read_back(1).expect("reads back");
            let plane = width * height;
            for (what, tensor) in [("fitted", &fitted), ("warped", &warped)] {
                let mut worst = 0.0f32;
                for y in 0..height {
                    for x in 0..width {
                        let (cb, cr) = chroma(x / 2, y / 2);
                        let unit = |code: u16| f32::from(code << 6) / 65535.0;
                        let want = map.apply(unit(luma(x, y)), unit(cb), unit(cr));
                        for (channel, want) in want.into_iter().enumerate() {
                            let got = tensor[channel * plane + y * width + x] * 255.0;
                            worst = worst.max((got - f32::from(want)).abs());
                        }
                    }
                }
                assert!(worst <= 2.0, "{transfer:?} {what}: off by {worst} of 255");
            }
        }
    }
}
