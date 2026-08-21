use std::{
    mem::{ManuallyDrop, size_of},
    sync::Arc,
};

use ffmpeg_next as ffmpeg;
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
            Dxgi::Common::{DXGI_COLOR_SPACE_TYPE, DXGI_FORMAT_NV12, DXGI_RATIONAL},
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

use crate::{platform::windows::d3d12va::set_d3d12va_fence_value, pool::UnboundObjectPoolRef};

use super::d3d12_scaler::D3d12ScalerError;

const COMMAND_SLOT_COUNT: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct ProcessorShape {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) color_space: DXGI_COLOR_SPACE_TYPE,
}

pub(super) struct VideoProcessFrame<'a> {
    pub(super) source: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    pub(super) shape: ProcessorShape,
    pub(super) input_texture: ID3D12Resource,
    pub(super) input_fence: ID3D12Fence,
    pub(super) input_fence_value: u64,
    pub(super) destination: &'a mut ffmpeg::frame::Video,
    pub(super) output_texture: ID3D12Resource,
    pub(super) output_fence: ID3D12Fence,
    pub(super) output_fence_value: u64,
}

struct CommandSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12VideoProcessCommandList,
    fence_value: u64,
    input: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

pub(super) struct D3d12VideoProcessor {
    video_device: ID3D12VideoDevice,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    fence_event: HANDLE,
    next_fence_value: u64,
    slots: Vec<CommandSlot>,
    next_slot: usize,
    processor: Option<(ProcessorShape, ID3D12VideoProcessor)>,
    output_width: u32,
    output_height: u32,
}

// SAFETY: the D3D12 COM interfaces are free-threaded. Mutable command-list,
// slot, and fence bookkeeping is reachable only through `&mut self`, and the
// event handle is used only for signalling/waiting rather than Rust memory.
unsafe impl Send for D3d12VideoProcessor {}

impl D3d12VideoProcessor {
    pub(super) fn new(
        device: &ID3D12Device,
        output_width: u32,
        output_height: u32,
    ) -> std::result::Result<Self, D3d12ScalerError> {
        let video_device: ID3D12VideoDevice = device.cast()?;
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
            ..Default::default()
        };
        // SAFETY: the descriptor is fully initialized for video-process work
        // and `device` is live.
        let queue = unsafe { device.CreateCommandQueue(&queue_desc) }?;
        // SAFETY: `device` is live and the initial fence value and flags are
        // valid D3D12 values.
        let fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;
        // SAFETY: default security, auto-reset, initially nonsignalled, and no
        // name are valid; the returned owned handle is closed on every path.
        let fence_event = unsafe { CreateEventW(None, false, false, None) }?;

        let slots_result = (|| -> std::result::Result<Vec<_>, windows::core::Error> {
            let mut slots = Vec::with_capacity(COMMAND_SLOT_COUNT);
            for _ in 0..COMMAND_SLOT_COUNT {
                // SAFETY: the live device creates an allocator whose type
                // matches the queue and command lists used by this processor.
                let allocator = unsafe {
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS)
                }?;
                // SAFETY: the allocator is live, belongs to `device`, and has
                // the requested list type; video process lists need no initial
                // pipeline-state object.
                let list: ID3D12VideoProcessCommandList = unsafe {
                    device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
                        &allocator,
                        None,
                    )
                }?;
                // SAFETY: a newly created command list is open; closing it
                // establishes the reset-before-recording lifecycle below.
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
                // SAFETY: this is the live handle created above, not yet
                // transferred into `Self`, and it is closed exactly once.
                unsafe { CloseHandle(fence_event).ok() };
                return Err(error.into());
            }
        };

        Ok(Self {
            video_device,
            queue,
            fence,
            fence_event,
            next_fence_value: 1,
            slots,
            next_slot: 0,
            processor: None,
            output_width,
            output_height,
        })
    }

    pub(super) fn process(
        &mut self,
        frame: VideoProcessFrame<'_>,
    ) -> std::result::Result<(), D3d12ScalerError> {
        let VideoProcessFrame {
            source,
            shape,
            input_texture,
            input_fence,
            input_fence_value,
            destination,
            output_texture,
            output_fence,
            output_fence_value,
        } = frame;
        self.ensure_processor(shape)?;
        // SAFETY: queue and producer fence are live objects and the value is
        // the completion value carried with this input texture.
        unsafe { self.queue.Wait(&input_fence, input_fence_value)? };
        self.prepare_slot()?;

        let slot_index = self.next_slot;
        let processor = &self.processor.as_ref().unwrap().1;
        let slot = &mut self.slots[slot_index];
        // SAFETY: `prepare_slot` waited for this slot's previous submission,
        // so neither allocator nor list is in GPU use; their types and device
        // match each other.
        unsafe {
            slot.allocator.Reset()?;
            slot.list.Reset(&slot.allocator)?;
        }

        let mut before = [
            transition_barrier(
                &input_texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ,
            ),
            transition_barrier(
                &output_texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE,
            ),
        ];
        // SAFETY: each barrier owns a live resource and describes its tracked
        // COMMON-to-video-process transition for this open command list.
        unsafe { slot.list.ResourceBarrier(&before) };
        drop_barrier_resources(&mut before);

        let mut input_args = [D3D12_VIDEO_PROCESS_INPUT_STREAM_ARGUMENTS::default()];
        input_args[0].InputStream[0].pTexture2D = ManuallyDrop::new(Some(input_texture.clone()));
        input_args[0].Transform.SourceRectangle = rect(shape.width, shape.height);
        input_args[0].Transform.DestinationRectangle = rect(self.output_width, self.output_height);
        input_args[0].Transform.Orientation = D3D12_VIDEO_PROCESS_ORIENTATION_DEFAULT;
        input_args[0].Flags = D3D12_VIDEO_PROCESS_INPUT_STREAM_FLAG_NONE;

        let mut output_args = D3D12_VIDEO_PROCESS_OUTPUT_STREAM_ARGUMENTS::default();
        output_args.OutputStream[0].pTexture2D = ManuallyDrop::new(Some(output_texture.clone()));
        output_args.TargetRectangle = rect(self.output_width, self.output_height);
        // SAFETY: processor/list and argument resources belong to one device;
        // the argument arrays retain their textures through `ProcessFrames`.
        // Both manually held references are then released exactly once.
        unsafe {
            slot.list
                .ProcessFrames(processor, &output_args, &input_args);
            ManuallyDrop::drop(&mut input_args[0].InputStream[0].pTexture2D);
            ManuallyDrop::drop(&mut output_args.OutputStream[0].pTexture2D);
        }

        let mut after = [
            transition_barrier(
                &input_texture,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ,
                D3D12_RESOURCE_STATE_COMMON,
            ),
            transition_barrier(
                &output_texture,
                D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE,
                D3D12_RESOURCE_STATE_COMMON,
            ),
        ];
        // SAFETY: these live resources are in the video-process states set by
        // `before`; the list is still open and is closed exactly once here.
        unsafe {
            slot.list.ResourceBarrier(&after);
            slot.list.Close()?;
        }
        drop_barrier_resources(&mut after);

        let command_list: ID3D12CommandList = slot.list.cast()?;
        // SAFETY: the command list is closed, belongs to this queue's device,
        // and the internal fence value is monotonically unique.
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(command_list)]);
            self.queue.Signal(&self.fence, self.next_fence_value)?;
        }
        slot.fence_value = self.next_fence_value;
        slot.input = Some(source);
        // SAFETY: `output_fence` is the live fence embedded in `destination`;
        // the same signalled value is stored back into that frame below.
        if let Err(error) = unsafe { self.queue.Signal(&output_fence, output_fence_value) } {
            self.wait_and_release_slot(slot_index)?;
            return Err(error.into());
        }
        if !set_d3d12va_fence_value(destination, output_fence_value) {
            self.wait_and_release_slot(slot_index)?;
            return Err(D3d12ScalerError::InvalidD3d12Frame);
        }

        self.next_fence_value += 1;
        self.next_slot = (slot_index + 1) % self.slots.len();
        Ok(())
    }

    pub(super) fn wait_all(&mut self) -> std::result::Result<(), D3d12ScalerError> {
        for slot in &mut self.slots {
            if slot.fence_value != 0 {
                wait_for_fence(&self.fence, slot.fence_value, self.fence_event)?;
                slot.input = None;
            }
        }
        Ok(())
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
        // SAFETY: `support` is the correctly typed live in/out structure and
        // the byte count exactly matches the selected feature's structure.
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
            || self.output_width < support.ScaleSupport.OutputSizeRange.MinWidth
            || self.output_height < support.ScaleSupport.OutputSizeRange.MinHeight
            || self.output_width > support.ScaleSupport.OutputSizeRange.MaxWidth
            || self.output_height > support.ScaleSupport.OutputSizeRange.MaxHeight
        {
            return Err(D3d12ScalerError::UnsupportedByVideoProcessor {
                input_width: shape.width,
                input_height: shape.height,
                output_width: self.output_width,
                output_height: self.output_height,
            });
        }

        let input_desc = D3D12_VIDEO_PROCESS_INPUT_STREAM_DESC {
            Format: DXGI_FORMAT_NV12,
            ColorSpace: shape.color_space,
            SourceAspectRatio: DXGI_RATIONAL {
                Numerator: shape.width,
                Denominator: shape.height,
            },
            DestinationAspectRatio: DXGI_RATIONAL {
                Numerator: self.output_width,
                Denominator: self.output_height,
            },
            FrameRate: rate,
            SourceSizeRange: exact_size(shape.width, shape.height),
            DestinationSizeRange: exact_size(self.output_width, self.output_height),
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
        // SAFETY: the stream descriptions are fully initialized from the
        // feature-supported shape and the video device remains live.
        let processor = unsafe {
            self.video_device
                .CreateVideoProcessor(0, &output_desc, &[input_desc])?
        };
        self.wait_all()?;
        self.processor = Some((shape, processor));
        Ok(())
    }

    fn prepare_slot(&mut self) -> std::result::Result<(), D3d12ScalerError> {
        self.wait_and_release_slot(self.next_slot)
    }

    fn wait_and_release_slot(
        &mut self,
        slot_index: usize,
    ) -> std::result::Result<(), D3d12ScalerError> {
        let slot = &mut self.slots[slot_index];
        if slot.fence_value != 0 {
            wait_for_fence(&self.fence, slot.fence_value, self.fence_event)?;
            slot.input = None;
        }
        Ok(())
    }
}

impl Drop for D3d12VideoProcessor {
    fn drop(&mut self) {
        let _ = self.wait_all();
        // SAFETY: `fence_event` is the owned live handle created in `new` and
        // is closed exactly once after pending submissions have been waited.
        unsafe { CloseHandle(self.fence_event).ok() };
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
        // SAFETY: `transition_barrier` initialized this union arm with one
        // cloned resource reference; this is its single matching drop site.
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
    // SAFETY: `fence` is live and querying its completed value borrows no
    // external memory.
    if unsafe { fence.GetCompletedValue() } >= value {
        return Ok(());
    }
    // SAFETY: `event` is this processor's live auto-reset event and `fence`
    // remains alive through registration and the wait for `value`.
    unsafe {
        fence.SetEventOnCompletion(value, event)?;
        if WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0 {
            return Err(D3d12ScalerError::FenceWaitFailed);
        }
    }
    Ok(())
}
