use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D12::{ID3D12Device, ID3D12Fence, ID3D12Resource},
        Dxgi::Common::{
            DXGI_COLOR_SPACE_TYPE, DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601,
            DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709, DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601,
            DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709, DXGI_FORMAT, DXGI_FORMAT_NV12,
        },
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
    platform::{
        ffmpeg::AvBufferRef,
        windows::d3d12va::{create_hw_device_ctx, create_hw_frames_ctx, d3d12va_texture},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

use super::d3d12_video_processor::{D3d12VideoProcessor, ProcessorShape, VideoProcessFrame};

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
    processor: D3d12VideoProcessor,
    _hw_device_ctx: AvBufferRef,
    hw_frames_ctx: AvBufferRef,
    width: u32,
    height: u32,
    pad: SrcPad,
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: D3D12 COM interfaces and the FFmpeg buffer owners are free-threaded;
// the video processor documents its own `Send` contract, and all mutable
// scaler/pool state is accessible only through `&mut self`.
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

        let processor = D3d12VideoProcessor::new(device, width, height)?;

        // SAFETY: `device` is live and the helper clones its COM reference
        // into the returned FFmpeg hardware-device context.
        let hw_device_ctx = match unsafe { create_hw_device_ctx(device) } {
            Ok(ctx) => ctx,
            Err(code) => return Err(D3d12ScalerError::HwDeviceInit(code)),
        };
        // SAFETY: the device context is initialized and kept alive by this
        // scaler; dimensions were validated and the pool size is positive.
        let hw_frames_ctx = match unsafe {
            create_hw_frames_ctx(&hw_device_ctx, width, height, OUTPUT_POOL_SIZE)
        } {
            Ok(ctx) => ctx,
            Err(code) => return Err(D3d12ScalerError::HwFramesInit(code)),
        };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: NV12 -> {width}x{height} NV12");
        Ok(Self {
            pp_log,
            name,
            device: device.clone(),
            processor,
            _hw_device_ctx: hw_device_ctx,
            hw_frames_ctx,
            width,
            height,
            pad,
            pool,
        })
    }

    fn scale(&mut self, source: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>) -> Result<()> {
        let (shape, texture, input_fence, input_fence_value) = self.validate_input(&source)?;

        let mut destination = self.pool.get();
        // SAFETY: the destination is exclusively owned, unreffed before pool
        // reuse, and allocated from this scaler's live frames context.
        unsafe {
            ffi::av_frame_unref(destination.as_mut_ptr());
            let ret = ffi::av_hwframe_get_buffer(
                self.hw_frames_ctx.as_ptr(),
                destination.as_mut_ptr(),
                0,
            );
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
        // SAFETY: the validated destination frame owns this non-null resource
        // pointer; cloning the borrowed wrapper acquires an independent COM ref.
        let output_texture = unsafe {
            ID3D12Resource::from_raw_borrowed(&output_texture_raw)
                .unwrap()
                .clone()
        };
        // SAFETY: as above for the non-null fence pointer stored in the same
        // live D3D12VA frame payload.
        let output_fence = unsafe {
            ID3D12Fence::from_raw_borrowed(&output_fence_raw)
                .unwrap()
                .clone()
        };
        let new_output_fence_value = output_fence_value.saturating_add(1);
        self.processor.process(VideoProcessFrame {
            source,
            shape,
            input_texture: texture,
            input_fence,
            input_fence_value,
            destination: &mut destination,
            output_texture,
            output_fence,
            output_fence_value: new_output_fence_value,
        })?;

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
        // SAFETY: `source` is live and confirmed D3D12. Its hardware-frame
        // reference is null-checked before dereference, and only the device
        // identity is read while the source retains both contexts.
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
        // SAFETY: the live validated source owns this non-null resource;
        // cloning the borrowed wrapper takes an independent COM reference.
        let texture = unsafe {
            ID3D12Resource::from_raw_borrowed(&texture_raw)
                .unwrap()
                .clone()
        };
        // SAFETY: the non-null fence belongs to the same live D3D12VA payload;
        // cloning the borrowed wrapper takes an independent COM reference.
        let fence = unsafe { ID3D12Fence::from_raw_borrowed(&fence_raw).unwrap().clone() };
        let mut texture_device: Option<ID3D12Device> = None;
        // SAFETY: `texture` is live and `texture_device` is a correctly typed
        // out-parameter for its creating device.
        unsafe { texture.GetDevice(&mut texture_device) }
            .map_err(|_| D3d12ScalerError::DeviceMismatch)?;
        if texture_device
            .is_none_or(|texture_device| texture_device.as_raw() != self.device.as_raw())
        {
            return Err(D3d12ScalerError::DeviceMismatch);
        }
        // SAFETY: `texture` is a live resource and `GetDesc` returns its plain
        // descriptor by value.
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
                self.processor.wait_all()?;
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
        if let Err(error) = self.processor.wait_all() {
            pp_error!(self, "failed to drain GPU work during drop: {error}");
        }
        pp_info!(
            self,
            "dropped: freeing D3D12 video processor and frame contexts"
        );
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
    use crate::test_support::try_d3d12_device as try_device;

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

    fn try_distinct_device(first: &ID3D12Device) -> Option<ID3D12Device> {
        // SAFETY: creates the documented DXGI factory interface with no raw
        // caller-owned storage.
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.ok()?;
        let mut index = 0;
        // SAFETY: indices are enumerated monotonically until this live factory
        // reports exhaustion.
        while let Ok(adapter) = unsafe { factory.EnumAdapters1(index) } {
            let mut device: Option<ID3D12Device> = None;
            // SAFETY: `adapter` is live and `device` is the correctly typed
            // out-parameter for the requested minimum feature level.
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
        // SAFETY: the test exclusively owns this live frame and writes only
        // its plain duration metadata before publishing it.
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
        // SAFETY: `frame` is live for this read of its plain metadata field.
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
