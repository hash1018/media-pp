use std::{
    mem::{ManuallyDrop, size_of},
    sync::Arc,
};

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, RECT, WAIT_OBJECT_0},
        Graphics::{
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS, D3D12_COMMAND_QUEUE_DESC,
                D3D12_FENCE_FLAG_NONE, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
                D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ, D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE,
                D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER, ID3D12CommandAllocator,
                ID3D12CommandList, ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12Resource,
            },
            Dxgi::Common::{
                DXGI_COLOR_SPACE_TYPE, DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601,
                DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709,
                DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601,
                DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709, DXGI_FORMAT, DXGI_FORMAT_NV12,
                DXGI_RATIONAL,
            },
        },
        Media::MediaFoundation::{
            D3D12_FEATURE_DATA_VIDEO_PROCESS_SUPPORT, D3D12_FEATURE_VIDEO_PROCESS_SUPPORT,
            D3D12_VIDEO_FIELD_TYPE_NONE, D3D12_VIDEO_FORMAT, D3D12_VIDEO_FRAME_STEREO_FORMAT_NONE,
            D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE_OPAQUE, D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS,
            D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC, D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAG_NONE,
            D3D12_VIDEO_PROCESS_ORIENTATION_DEFAULT, D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS,
            D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC, D3D12_VIDEO_PROCESS_SUPPORT_FLAG_SUPPORTED,
            D3D12_VIDEO_SAMPLE, D3D12_VIDEO_SIZE_RANGE, ID3D12VideoDevice,
            ID3D12VideoProcessCommandList, ID3D12VideoProcessor,
        },
        System::Threading::{CreateEventW, INFINITE, WaitForSingleObject},
    },
    core::Interface,
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::windows::d3d12va::{
        create_hw_device_ctx, create_hw_frames_ctx, d3d12va_texture, free_buffer,
        set_d3d12va_fence_value,
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

const COMMAND_SLOT_COUNT: usize = 4;
const OUTPUT_POOL_SIZE: i32 = 8;

#[derive(Debug, ThisError)]
pub enum D3d12ScalerError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    #[error("D3d12Scaler output dimensions must be non-zero, got {width}x{height}")]
    InvalidOutputDimensions { width: u32, height: u32 },

    #[error("NV12 dimensions must both be even, got {width}x{height}")]
    OddNv12Dimensions { width: u32, height: u32 },

    #[error("failed to create D3D12VA hardware device context (code {0})")]
    HwDeviceInit(i32),

    #[error("failed to create D3D12VA output frames context (code {0})")]
    HwFramesInit(i32),

    #[error("failed to allocate a D3D12 output frame (code {0})")]
    GetBuffer(i32),

    #[error("failed to copy scaled frame metadata (code {0})")]
    CopyProperties(i32),

    #[error("D3d12Scaler only accepts Pixel::D3D12 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error("D3d12Scaler only scales NV12 D3D12 surfaces, got {0:?}")]
    UnsupportedSurfaceFormat(ffmpeg::format::Pixel),

    #[error("D3d12Scaler only scales DXGI_FORMAT_NV12 textures, got {0:?}")]
    UnsupportedTextureFormat(DXGI_FORMAT),

    #[error("D3D12 frame has no valid hardware frames context")]
    MissingFramesContext,

    #[error("D3D12 frame has no valid texture or synchronization fence")]
    InvalidD3d12Frame,

    #[error("D3D12 texture belongs to a different device than this scaler")]
    DeviceMismatch,

    #[error(
        "D3D12 texture is {actual_width}x{actual_height}, smaller than the frame's {expected_width}x{expected_height} visible size"
    )]
    TextureTooSmall {
        actual_width: u64,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    #[error("D3d12Scaler requires single-sample NV12 textures, got sample count {0}")]
    MultisampledTexture(u32),

    #[error(
        "this GPU does not support D3D12 NV12 video-process scaling from {input_width}x{input_height} to {output_width}x{output_height}"
    )]
    UnsupportedByVideoProcessor {
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
    },

    #[error("D3d12Scaler only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),

    #[error("waiting for the D3D12 video-process fence failed")]
    FenceWaitFailed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ProcessorShape {
    width: u32,
    height: u32,
    color_space: DXGI_COLOR_SPACE_TYPE,
}

struct CommandSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12VideoProcessCommandList,
    fence_value: u64,
    input: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

/// Resizes GPU-resident NV12 `Pixel::D3D12` frames with D3D12's video
/// processor. Input and output stay NV12; this element never performs a
/// pixel-format conversion or a CPU transfer.
///
/// The output size is fixed at construction. The input size and color space
/// are learned from the first frame and the video processor is rebuilt after
/// a later input renegotiation. PTS, duration, and color metadata are copied
/// to the scaled frame unchanged.
pub struct D3d12Scaler {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D12Device,
    video_device: ID3D12VideoDevice,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    fence_event: HANDLE,
    next_fence_value: u64,
    slots: Vec<CommandSlot>,
    next_slot: usize,
    processor: Option<(ProcessorShape, ID3D12VideoProcessor)>,
    hw_device_ctx: *mut ffi::AVBufferRef,
    hw_frames_ctx: *mut ffi::AVBufferRef,
    width: u32,
    height: u32,
    pad: SrcPad,
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

unsafe impl Send for D3d12Scaler {}

impl D3d12Scaler {
    pub fn new(
        name: impl Into<String>,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, D3d12ScalerError> {
        validate_dimensions(width, height)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12Scaler, &name, None);

        let video_device: ID3D12VideoDevice = device.cast()?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
            ..Default::default()
        };
        let queue = unsafe { device.CreateCommandQueue(&queue_desc) }?;
        let fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
        let fence_event = unsafe { CreateEventW(None, false, false, None) }?;

        let slots_result = (|| -> std::result::Result<Vec<_>, windows::core::Error> {
            let mut slots = Vec::with_capacity(COMMAND_SLOT_COUNT);
            for _ in 0..COMMAND_SLOT_COUNT {
                let allocator = unsafe {
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS)
                }?;
                let list: ID3D12VideoProcessCommandList = unsafe {
                    device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
                        &allocator,
                        None,
                    )
                }?;
                unsafe { list.Close()? };
                slots.push(CommandSlot {
                    allocator,
                    list,
                    fence_value: 0,
                    input: None,
                });
            }
            Ok(slots)
        })();
        let slots = match slots_result {
            Ok(slots) => slots,
            Err(error) => {
                unsafe { CloseHandle(fence_event).ok() };
                return Err(error.into());
            }
        };

        let hw_device_ctx = match unsafe { create_hw_device_ctx(device) } {
            Ok(ctx) => ctx,
            Err(code) => {
                unsafe { CloseHandle(fence_event).ok() };
                return Err(D3d12ScalerError::HwDeviceInit(code));
            }
        };
        let hw_frames_ctx =
            match unsafe { create_hw_frames_ctx(hw_device_ctx, width, height, OUTPUT_POOL_SIZE) } {
                Ok(ctx) => ctx,
                Err(code) => {
                    unsafe {
                        free_buffer(hw_device_ctx);
                        CloseHandle(fence_event).ok();
                    }
                    return Err(D3d12ScalerError::HwFramesInit(code));
                }
            };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: NV12 -> {width}x{height} NV12");
        Ok(Self {
            pp_log,
            name,
            device: device.clone(),
            video_device,
            queue,
            fence,
            fence_event,
            next_fence_value: 1,
            slots,
            next_slot: 0,
            processor: None,
            hw_device_ctx,
            hw_frames_ctx,
            width,
            height,
            pad,
            pool,
        })
    }

    fn scale(&mut self, source: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>) -> Result<()> {
        let (shape, texture, input_fence, input_fence_value) = self.validate_input(&source)?;
        self.ensure_processor(shape)?;

        let mut destination = self.pool.get();
        unsafe {
            ffi::av_frame_unref(destination.as_mut_ptr());
            let ret = ffi::av_hwframe_get_buffer(self.hw_frames_ctx, destination.as_mut_ptr(), 0);
            if ret < 0 {
                return Err(D3d12ScalerError::GetBuffer(ret).into());
            }
            let ret = ffi::av_frame_copy_props(destination.as_mut_ptr(), source.as_ptr());
            if ret < 0 {
                return Err(D3d12ScalerError::CopyProperties(ret).into());
            }
            (*destination.as_mut_ptr()).width = self.width as i32;
            (*destination.as_mut_ptr()).height = self.height as i32;
        }

        let (output_texture_raw, output_fence_raw, output_fence_value) =
            d3d12va_texture(&destination).ok_or(D3d12ScalerError::InvalidD3d12Frame)?;
        if output_texture_raw.is_null() || output_fence_raw.is_null() {
            return Err(D3d12ScalerError::InvalidD3d12Frame.into());
        }
        let output_texture = unsafe {
            ID3D12Resource::from_raw_borrowed(&output_texture_raw)
                .unwrap()
                .clone()
        };
        let output_fence = unsafe {
            ID3D12Fence::from_raw_borrowed(&output_fence_raw)
                .unwrap()
                .clone()
        };
        let new_output_fence_value = output_fence_value.saturating_add(1);

        unsafe {
            self.queue
                .Wait(&input_fence, input_fence_value)
                .map_err(D3d12ScalerError::from)?
        };
        self.prepare_slot()?;
        let slot_index = self.next_slot;
        let processor = &self.processor.as_ref().unwrap().1;
        let slot = &mut self.slots[slot_index];
        unsafe {
            slot.allocator.Reset().map_err(D3d12ScalerError::from)?;
            slot.list
                .Reset(&slot.allocator)
                .map_err(D3d12ScalerError::from)?;
        }

        let mut before = [
            transition_barrier(
                &texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ,
            ),
            transition_barrier(
                &output_texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE,
            ),
        ];
        unsafe { slot.list.ResourceBarrier(&before) };
        drop_barrier_resources(&mut before);

        let mut input_args = [D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS::default()];
        input_args[0].InputStream[0].pTexture2D = ManuallyDrop::new(Some(texture.clone()));
        input_args[0].Transform.SourceRectangle = rect(shape.width, shape.height);
        input_args[0].Transform.DestinationRectangle = rect(self.width, self.height);
        input_args[0].Transform.Orientation = D3D12_VIDEO_PROCESS_ORIENTATION_DEFAULT;
        input_args[0].Flags = D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAG_NONE;

        let mut output_args = D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS::default();
        output_args.OutputStream[0].pTexture2D = ManuallyDrop::new(Some(output_texture.clone()));
        output_args.TargetRectangle = rect(self.width, self.height);
        unsafe {
            slot.list
                .ProcessFrames(processor, &output_args, &input_args);
            ManuallyDrop::drop(&mut input_args[0].InputStream[0].pTexture2D);
            ManuallyDrop::drop(&mut output_args.OutputStream[0].pTexture2D);
        }

        let mut after = [
            transition_barrier(
                &texture,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ,
                D3D12_RESOURCE_STATE_COMMON,
            ),
            transition_barrier(
                &output_texture,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE,
                D3D12_RESOURCE_STATE_COMMON,
            ),
        ];
        unsafe {
            slot.list.ResourceBarrier(&after);
            slot.list.Close().map_err(D3d12ScalerError::from)?;
        }
        drop_barrier_resources(&mut after);

        let command_list: ID3D12CommandList = slot.list.cast().map_err(D3d12ScalerError::from)?;
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(command_list)]);
            self.queue
                .Signal(&self.fence, self.next_fence_value)
                .map_err(D3d12ScalerError::from)?;
        }
        slot.fence_value = self.next_fence_value;
        slot.input = Some(source);
        if let Err(error) = unsafe { self.queue.Signal(&output_fence, new_output_fence_value) } {
            // The shared fence was queued first, so do not release either
            // frame until the submitted video-process work has completed.
            wait_for_fence(&self.fence, self.next_fence_value, self.fence_event)?;
            slot.input = None;
            return Err(D3d12ScalerError::from(error).into());
        }
        if !set_d3d12va_fence_value(&mut destination, new_output_fence_value) {
            wait_for_fence(&self.fence, self.next_fence_value, self.fence_event)?;
            slot.input = None;
            return Err(D3d12ScalerError::InvalidD3d12Frame.into());
        }
        self.next_fence_value += 1;
        self.next_slot = (slot_index + 1) % self.slots.len();

        self.pad.push(MediaBuffer::Video(Arc::new(destination)))
    }

    fn validate_input(
        &self,
        source: &ffmpeg::frame::Video,
    ) -> std::result::Result<(ProcessorShape, ID3D12Resource, ID3D12Fence, u64), D3d12ScalerError>
    {
        if source.format() != ffmpeg::format::Pixel::D3D12 {
            return Err(D3d12ScalerError::UnsupportedFormat(source.format()));
        }
        validate_dimensions(source.width(), source.height())?;
        unsafe {
            let frames_ref = (*source.as_ptr()).hw_frames_ctx;
            if frames_ref.is_null() || (*frames_ref).data.is_null() {
                return Err(D3d12ScalerError::MissingFramesContext);
            }
            let frames_ctx = (*frames_ref).data as *const ffi::AVHWFramesContext;
            let sw_format = ffmpeg::format::Pixel::from((*frames_ctx).sw_format);
            if sw_format != ffmpeg::format::Pixel::NV12 {
                return Err(D3d12ScalerError::UnsupportedSurfaceFormat(sw_format));
            }
            if (*frames_ctx).device_ctx.is_null() {
                return Err(D3d12ScalerError::MissingFramesContext);
            }
        }

        let (texture_raw, fence_raw, fence_value) =
            d3d12va_texture(source).ok_or(D3d12ScalerError::InvalidD3d12Frame)?;
        if texture_raw.is_null() || fence_raw.is_null() {
            return Err(D3d12ScalerError::InvalidD3d12Frame);
        }
        let texture = unsafe {
            ID3D12Resource::from_raw_borrowed(&texture_raw)
                .unwrap()
                .clone()
        };
        let fence = unsafe { ID3D12Fence::from_raw_borrowed(&fence_raw).unwrap().clone() };
        let mut texture_device: Option<ID3D12Device> = None;
        unsafe { texture.GetDevice(&mut texture_device) }
            .map_err(|_| D3d12ScalerError::DeviceMismatch)?;
        if texture_device
            .is_none_or(|texture_device| texture_device.as_raw() != self.device.as_raw())
        {
            return Err(D3d12ScalerError::DeviceMismatch);
        }
        let desc = unsafe { texture.GetDesc() };
        if desc.Format != DXGI_FORMAT_NV12 {
            return Err(D3d12ScalerError::UnsupportedTextureFormat(desc.Format));
        }
        if desc.Width < source.width() as u64 || desc.Height < source.height() {
            return Err(D3d12ScalerError::TextureTooSmall {
                actual_width: desc.Width,
                actual_height: desc.Height,
                expected_width: source.width(),
                expected_height: source.height(),
            });
        }
        if desc.SampleDesc.Count != 1 {
            return Err(D3d12ScalerError::MultisampledTexture(desc.SampleDesc.Count));
        }

        Ok((
            ProcessorShape {
                width: source.width(),
                height: source.height(),
                color_space: color_space(source),
            },
            texture,
            fence,
            fence_value,
        ))
    }

    fn ensure_processor(
        &mut self,
        shape: ProcessorShape,
    ) -> std::result::Result<(), D3d12ScalerError> {
        if self
            .processor
            .as_ref()
            .is_some_and(|(old, _)| *old == shape)
        {
            return Ok(());
        }
        let video_format = D3D12_VIDEO_FORMAT {
            Format: DXGI_FORMAT_NV12,
            ColorSpace: shape.color_space,
        };
        let rate = DXGI_RATIONAL {
            Numerator: 1,
            Denominator: 1,
        };
        let mut support = D3D12_FEATURE_DATA_VIDEO_PROCESS_SUPPORT {
            NodeIndex: 0,
            InputSample: D3D12_VIDEO_SAMPLE {
                Width: shape.width,
                Height: shape.height,
                Format: video_format,
            },
            InputFieldType: D3D12_VIDEO_FIELD_TYPE_NONE,
            InputStereoFormat: D3D12_VIDEO_FRAME_STEREO_FORMAT_NONE,
            InputFrameRate: rate,
            OutputFormat: video_format,
            OutputStereoFormat: D3D12_VIDEO_FRAME_STEREO_FORMAT_NONE,
            OutputFrameRate: rate,
            ..Default::default()
        };
        unsafe {
            self.video_device.CheckFeatureSupport(
                D3D12_FEATURE_VIDEO_PROCESS_SUPPORT,
                (&mut support as *mut D3D12_FEATURE_DATA_VIDEO_PROCESS_SUPPORT).cast(),
                size_of::<D3D12_FEATURE_DATA_VIDEO_PROCESS_SUPPORT>() as u32,
            )?;
        }
        if !support
            .SupportFlags
            .contains(D3D12_VIDEO_PROCESS_SUPPORT_FLAG_SUPPORTED)
            || self.width < support.ScaleSupport.OutputSizeRange.MinWidth
            || self.height < support.ScaleSupport.OutputSizeRange.MinHeight
            || self.width > support.ScaleSupport.OutputSizeRange.MaxWidth
            || self.height > support.ScaleSupport.OutputSizeRange.MaxHeight
        {
            return Err(D3d12ScalerError::UnsupportedByVideoProcessor {
                input_width: shape.width,
                input_height: shape.height,
                output_width: self.width,
                output_height: self.height,
            });
        }

        let source_range = exact_size(shape.width, shape.height);
        let destination_range = exact_size(self.width, self.height);
        let input_desc = D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC {
            Format: DXGI_FORMAT_NV12,
            ColorSpace: shape.color_space,
            SourceAspectRatio: DXGI_RATIONAL {
                Numerator: shape.width,
                Denominator: shape.height,
            },
            DestinationAspectRatio: DXGI_RATIONAL {
                Numerator: self.width,
                Denominator: self.height,
            },
            FrameRate: rate,
            SourceSizeRange: source_range,
            DestinationSizeRange: destination_range,
            StereoFormat: D3D12_VIDEO_FRAME_STEREO_FORMAT_NONE,
            FieldType: D3D12_VIDEO_FIELD_TYPE_NONE,
            ..Default::default()
        };
        let output_desc = D3D12_VIDEO_PROCESS_OUTPUT_STREAM_DESC {
            Format: DXGI_FORMAT_NV12,
            ColorSpace: shape.color_space,
            AlphaFillMode: D3D12_VIDEO_PROCESS_ALPHA_FILL_MODE_OPAQUE,
            FrameRate: rate,
            ..Default::default()
        };
        let processor = unsafe {
            self.video_device
                .CreateVideoProcessor(0, &output_desc, &[input_desc])?
        };
        self.wait_all_slots()?;
        self.processor = Some((shape, processor));
        Ok(())
    }

    fn prepare_slot(&mut self) -> std::result::Result<(), D3d12ScalerError> {
        let fence_value = self.slots[self.next_slot].fence_value;
        if fence_value != 0 {
            wait_for_fence(&self.fence, fence_value, self.fence_event)?;
            self.slots[self.next_slot].input = None;
        }
        Ok(())
    }

    fn wait_all_slots(&mut self) -> std::result::Result<(), D3d12ScalerError> {
        for slot in &mut self.slots {
            if slot.fence_value != 0 {
                wait_for_fence(&self.fence, slot.fence_value, self.fence_event)?;
                slot.input = None;
            }
        }
        Ok(())
    }
}

impl Element for D3d12Scaler {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }
    fn element_type(&self) -> ElementType {
        ElementType::D3d12Scaler
    }
    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }
    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d12Scaler {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d12Scaler {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.scale(frame),
            MediaBuffer::Eos => {
                self.wait_all_slots()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => {
                let error = D3d12ScalerError::UnsupportedBuffer(other.kind());
                pp_error!(self, "{error}");
                Err(error.into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.pad.control(msg)
    }
}

impl Drop for D3d12Scaler {
    fn drop(&mut self) {
        if let Err(error) = self.wait_all_slots() {
            pp_error!(self, "failed to drain GPU work during drop: {error}");
        }
        pp_info!(
            self,
            "dropped: freeing D3D12 video processor and frame contexts"
        );
        unsafe {
            free_buffer(self.hw_frames_ctx);
            free_buffer(self.hw_device_ctx);
            CloseHandle(self.fence_event).ok();
        }
    }
}

fn validate_dimensions(width: u32, height: u32) -> std::result::Result<(), D3d12ScalerError> {
    if width == 0 || height == 0 {
        return Err(D3d12ScalerError::InvalidOutputDimensions { width, height });
    }
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(D3d12ScalerError::OddNv12Dimensions { width, height });
    }
    Ok(())
}

fn color_space(frame: &ffmpeg::frame::Video) -> DXGI_COLOR_SPACE_TYPE {
    let bt709 = match frame.color_space() {
        ffmpeg::color::Space::BT709 => true,
        ffmpeg::color::Space::BT470BG | ffmpeg::color::Space::SMPTE170M => false,
        _ => frame.height() > 576,
    };
    let full = frame.color_range() == ffmpeg::color::Range::JPEG;
    match (bt709, full) {
        (true, true) => DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709,
        (true, false) => DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
        (false, true) => DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601,
        (false, false) => DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601,
    }
}

fn rect(width: u32, height: u32) -> RECT {
    RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    }
}

fn exact_size(width: u32, height: u32) -> D3D12_VIDEO_SIZE_RANGE {
    D3D12_VIDEO_SIZE_RANGE {
        MaxWidth: width,
        MaxHeight: height,
        MinWidth: width,
        MinHeight: height,
    }
}

fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
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
    }
}

fn drop_barrier_resources(barriers: &mut [D3D12_RESOURCE_BARRIER]) {
    for barrier in barriers {
        unsafe {
            let transition = &mut barrier.Anonymous.Transition;
            ManuallyDrop::drop(&mut transition.pResource);
        }
    }
}

fn wait_for_fence(
    fence: &ID3D12Fence,
    value: u64,
    event: HANDLE,
) -> std::result::Result<(), D3d12ScalerError> {
    if unsafe { fence.GetCompletedValue() } >= value {
        return Ok(());
    }
    unsafe {
        fence.SetEventOnCompletion(value, event)?;
        if WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0 {
            return Err(D3d12ScalerError::FenceWaitFailed);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use windows::Win32::Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0,
        Direct3D12::{D3D12CreateDevice, ID3D12Device},
        Dxgi::{CreateDXGIFactory1, IDXGIFactory1},
    };

    use super::*;
    use crate::elements::{D3d12Download, D3d12Upload};

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

    fn try_device() -> Option<ID3D12Device> {
        let mut device = None;
        if let Err(error) = unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut device) }
        {
            eprintln!("skipping: D3D12CreateDevice failed: {error}");
            return None;
        }
        device
    }

    fn try_distinct_device(first: &ID3D12Device) -> Option<ID3D12Device> {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
        let mut index = 0;
        while let Ok(adapter) = unsafe { factory.EnumAdapters1(index) } {
            let mut device: Option<ID3D12Device> = None;
            if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }.is_ok()
                && let Some(device) = device
                && device.as_raw() != first.as_raw()
            {
                return Some(device);
            }
            index += 1;
        }
        None
    }

    #[test]
    fn scales_nv12_on_gpu_and_preserves_metadata() {
        let Some(device) = try_device() else {
            return;
        };
        let (input_width, input_height) = (128, 128);
        let (output_width, output_height) = (64, 64);
        let Ok(mut upload) = D3d12Upload::new("upload", &device, input_width, input_height) else {
            eprintln!("skipping: FFmpeg could not create D3D12VA frames");
            return;
        };
        let Ok(mut scaler) = D3d12Scaler::new("scaler", &device, output_width, output_height)
        else {
            eprintln!("skipping: D3D12 video processing is unavailable");
            return;
        };
        let mut download = D3d12Download::new("download", output_width, output_height);
        let received = Arc::new(Mutex::new(Vec::new()));
        download.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        scaler.src_pads()[0].link(Box::new(download));
        upload.src_pads()[0].link(Box::new(scaler));

        let pool = UnboundObjectPool::new(
            0,
            move || {
                ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, input_width, input_height)
            },
            |_| {},
        );
        let mut source = pool.get();
        for row in 0..input_height as usize {
            let stride = source.stride(0);
            source.data_mut(0)[row * stride..row * stride + input_width as usize].fill(82);
        }
        for row in 0..input_height as usize / 2 {
            let stride = source.stride(1);
            for column in (0..input_width as usize).step_by(2) {
                source.data_mut(1)[row * stride + column..][..2].copy_from_slice(&[90, 240]);
            }
        }
        source.set_pts(Some(42));
        source.set_color_space(ffmpeg::color::Space::BT709);
        source.set_color_range(ffmpeg::color::Range::MPEG);
        unsafe { (*source.as_mut_ptr()).duration = 3 };

        let result = upload.consume(MediaBuffer::Video(Arc::new(source)));
        if let Err(error) = &result
            && error.to_string().contains("does not support D3D12 NV12")
        {
            eprintln!("skipping: {error}");
            return;
        }
        result.expect("D3D12 upload/scale/download should succeed");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(frame) = &received[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(
            (frame.width(), frame.height()),
            (output_width, output_height)
        );
        assert_eq!(frame.format(), ffmpeg::format::Pixel::NV12);
        assert_eq!(frame.pts(), Some(42));
        assert_eq!(unsafe { (*frame.as_ptr()).duration }, 3);
        assert_eq!(frame.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(frame.color_range(), ffmpeg::color::Range::MPEG);
        for row in 0..output_height as usize {
            let stride = frame.stride(0);
            for &value in &frame.data(0)[row * stride..row * stride + output_width as usize] {
                assert!(value.abs_diff(82) <= 2, "unexpected luma value {value}");
            }
        }
        for row in 0..output_height as usize / 2 {
            let stride = frame.stride(1);
            for column in (0..output_width as usize).step_by(2) {
                let uv = &frame.data(1)[row * stride + column..][..2];
                assert!(uv[0].abs_diff(90) <= 2, "unexpected U value {}", uv[0]);
                assert!(uv[1].abs_diff(240) <= 2, "unexpected V value {}", uv[1]);
            }
        }
    }

    #[test]
    fn validates_nv12_dimensions_without_touching_hardware() {
        assert!(matches!(
            validate_dimensions(0, 16),
            Err(D3d12ScalerError::InvalidOutputDimensions { .. })
        ));
        assert!(matches!(
            validate_dimensions(15, 16),
            Err(D3d12ScalerError::OddNv12Dimensions { .. })
        ));
    }

    #[test]
    fn rejects_cpu_and_packet_buffers_and_forwards_eos() {
        let Some(device) = try_device() else {
            return;
        };
        let Ok(mut scaler) = D3d12Scaler::new("scaler", &device, 64, 64) else {
            eprintln!("skipping: D3D12 video processing is unavailable");
            return;
        };
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 64),
            |_| {},
        );
        let error = scaler
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("a CPU frame must be rejected");
        assert!(error.to_string().contains("only accepts Pixel::D3D12"));

        let error = scaler
            .consume(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::empty())))
            .expect_err("a packet must be rejected");
        assert!(error.to_string().contains("Video and Eos"));

        let received = Arc::new(Mutex::new(Vec::new()));
        scaler.src_pads()[0].link(Box::new(CapturingSink {
            pp_log: element_pp_log(ElementType::Other, "capture", None),
            received: received.clone(),
        }));
        scaler
            .consume(MediaBuffer::Eos)
            .expect("EOS should forward");
        assert!(matches!(
            received.lock().unwrap().as_slice(),
            [MediaBuffer::Eos]
        ));
    }

    #[test]
    fn rejects_a_texture_from_another_device() {
        let Some(first) = try_device() else {
            return;
        };
        let Some(second) = try_distinct_device(&first) else {
            eprintln!("skipping: no second D3D12 adapter is available");
            return;
        };
        if first.as_raw() == second.as_raw() {
            eprintln!("skipping: D3D12CreateDevice returned the same device object twice");
            return;
        }
        let Ok(mut upload) = D3d12Upload::new("upload", &second, 128, 128) else {
            eprintln!("skipping: FFmpeg could not create D3D12VA frames");
            return;
        };
        let Ok(scaler) = D3d12Scaler::new("scaler", &first, 64, 64) else {
            eprintln!("skipping: D3D12 video processing is unavailable");
            return;
        };
        upload.src_pads()[0].link(Box::new(scaler));
        let pool = UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 128, 128),
            |_| {},
        );
        let error = upload
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("a foreign-device texture must be rejected");
        assert!(error.to_string().contains("different device"));
    }
}
