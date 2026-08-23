use std::sync::{Arc, Mutex};

use ffmpeg_next as ffmpeg;
use ffmpeg_next::Rescale;
use ffmpeg_next::ffi;
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D11::{
            D3D11_BIND_SHADER_RESOURCE, D3D11_BOX, ID3D11Device, ID3D11DeviceContext,
            ID3D11Texture2D,
        },
        Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12},
    },
    core::Interface,
};

use crate::pp_log::{PpLog, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::filter::is_codec_drain_boundary,
    error::Result,
    pad::SrcPad,
    platform::{
        ffmpeg::AvBufferRef,
        windows::d3d11va::{create_hw_device_ctx, d3d11va_texture, or_frames_bind_flags},
    },
};

/// Errors specific to `D3d11NvencEncoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11NvencEncoderError {
    /// The requested NVENC implementation is unavailable in the FFmpeg build.
    #[error(
        "encoder {0:?} not found — this ffmpeg build wasn't compiled with \
         NVENC support (run `ffmpeg -encoders` to check)"
    )]
    CodecNotFound(String),

    /// FFmpeg could not wrap the supplied D3D11 device as a hardware device context.
    #[error("failed to create D3D11VA hw device context (code {0})")]
    HwDeviceInit(i32),

    /// FFmpeg could not allocate the encoder's D3D11 frame-pool description.
    #[error("failed to allocate a D3D11 hw frames context")]
    HwFramesAlloc,

    /// FFmpeg could not initialize the encoder's fixed-size D3D11 frame pool.
    #[error(
        "failed to initialize the D3D11 hw frames context (code {0}) — most \
         often the requested {1}x{2} exceeds what this GPU's encoder accepts"
    )]
    HwFramesInit(i32, u32, u32),

    /// FFmpeg could not acquire a frame from the encoder-owned D3D11 pool.
    #[error("failed to acquire a frame from the encoder's own D3D11 pool (code {0})")]
    HwFrameGet(i32),

    /// The input frame is not backed by a D3D11 texture.
    #[error(
        "D3d11NvencEncoder only accepts Pixel::D3D11 frames (chain a \
         D3d11Upload, or feed it a D3d11VideoCompositor/DxgiCaptureSource \
         GPU-mode output), got {0:?}"
    )]
    UnsupportedFormat(ffmpeg::format::Pixel),

    /// Input dimensions differ from the fixed encoder dimensions.
    #[error(
        "frame is {actual_width}x{actual_height}, but D3d11NvencEncoder was \
         opened for {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        /// Input frame width in pixels.
        actual_width: u32,
        /// Input frame height in pixels.
        actual_height: u32,
        /// Width configured for the encoder.
        expected_width: u32,
        /// Height configured for the encoder.
        expected_height: u32,
    },

    /// The supplied immediate context belongs to another D3D11 device.
    #[error("the shared ID3D11DeviceContext belongs to a different ID3D11Device than the encoder")]
    ContextDeviceMismatch,

    /// The input texture belongs to another D3D11 device.
    #[error("a Pixel::D3D11 frame's texture lives on a different ID3D11Device than this encoder")]
    DeviceMismatch,

    /// The backing texture is smaller than the visible frame to encode.
    #[error(
        "D3D11 texture is {actual_width}x{actual_height}, smaller than the encoder's {expected_width}x{expected_height} visible frame"
    )]
    TextureTooSmall {
        /// Backing texture width in pixels.
        actual_width: u32,
        /// Backing texture height in pixels.
        actual_height: u32,
        /// Minimum width required by the encoder.
        expected_width: u32,
        /// Minimum height required by the encoder.
        expected_height: u32,
    },

    /// The frame selects a texture-array slice outside the resource bounds.
    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex {
        /// Invalid texture-array index from the frame.
        index: isize,
        /// Number of slices in the backing texture array.
        array_size: u32,
    },

    /// The texture format differs from the format fixed at encoder creation.
    #[error(
        "frame carries a {actual:?} texture, but D3d11NvencEncoder was opened \
         for {expected:?} — an encoder's input format is fixed at \
         avcodec_open2 time and cannot change mid-stream"
    )]
    TextureFormatMismatch {
        /// Actual DXGI format numeric value.
        actual: i32,
        /// Expected DXGI format numeric value.
        expected: i32,
    },

    /// A frame tagged as D3D11 contains no valid texture reference.
    #[error("a Pixel::D3D11 frame arrived without a texture in data[0]")]
    MissingTexture,

    /// The sink received a buffer other than decoded video or end-of-stream.
    #[error("D3d11NvencEncoder only accepts Video or Eos buffers, got {0}")]
    UnsupportedBuffer(&'static str),

    /// A Direct3D resource or copy operation failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// FFmpeg rejected encoder, hardware-context, frame, or packet processing.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),
}

/// Which NVENC encoder to open. Both are hardware encoders on the GPU's
/// dedicated encode block, so neither carries the GPL/licensing question
/// [`crate::elements::VideoCodec`]'s software encoders document — but both
/// still have to actually exist in the linked ffmpeg build, and
/// [`D3d11NvencEncoder::new`] fails with
/// [`D3d11NvencEncoderError::CodecNotFound`] (not a panic) if they don't.
///
/// AV1 is deliberately absent: `av1_nvenc` exists in ffmpeg, but NVENC's
/// AV1 encode block only shipped with Ada (RTX 40) and later, so offering
/// it here would mean a variant that fails at `open` on a large share of
/// otherwise-working GPUs. Add it when there is a real requirement and a
/// machine to verify it on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D3d11NvencCodec {
    /// `h264_nvenc` — H.264/AVC.
    H264,
    /// `hevc_nvenc` — H.265/HEVC.
    H265,
}

impl D3d11NvencCodec {
    fn encoder_name(self) -> &'static str {
        match self {
            Self::H264 => "h264_nvenc",
            Self::H265 => "hevc_nvenc",
        }
    }
}

/// Which D3D11 texture format this encoder's input frames carry. This is
/// the `sw_format` of the encoder's own hw frames context, fixed at
/// `avcodec_open2` time like every other codec parameter — it is *not*
/// inferred from the first frame, for the same reason
/// [`crate::elements::SwEncoderOptions`] takes `width`/`height` up front.
///
/// Both variants were verified against `h264_nvenc`/`hevc_nvenc` on real
/// hardware. [`D3d11NvencInputFormat::Bgra`] is what makes a screen
/// recording need no color conversion at all: `D3d11VideoCompositor` and
/// `DxgiCaptureSource`'s GPU mode both produce BGRA textures, and NVENC
/// converts to its own internal YUV as part of encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D3d11NvencInputFormat {
    /// `DXGI_FORMAT_NV12` — what [`crate::elements::D3d11Upload`] and
    /// [`crate::elements::D3d11Decoder`] produce.
    Nv12,
    /// `DXGI_FORMAT_B8G8R8A8_UNORM` — what
    /// [`crate::elements::D3d11VideoCompositor`] and
    /// `DxgiCaptureSource`'s GPU mode produce.
    Bgra,
}

impl D3d11NvencInputFormat {
    fn sw_format(self) -> ffi::AVPixelFormat {
        match self {
            Self::Nv12 => ffi::AVPixelFormat::AV_PIX_FMT_NV12,
            Self::Bgra => ffi::AVPixelFormat::AV_PIX_FMT_BGRA,
        }
    }

    fn dxgi_format(self) -> DXGI_FORMAT {
        match self {
            Self::Nv12 => DXGI_FORMAT_NV12,
            Self::Bgra => DXGI_FORMAT_B8G8R8A8_UNORM,
        }
    }
}

/// Everything `avcodec_open2` needs before this encoder can be opened at
/// all — same convention, and the same reasoning, as
/// [`crate::elements::SwEncoderOptions`], which documents at length why
/// `time_base` and `frame_rate` are separate values rather than one
/// derived from the other.
#[derive(Debug, Clone, Copy)]
pub struct D3d11NvencEncoderOptions {
    /// NVENC bitstream codec to open.
    pub codec: D3d11NvencCodec,
    /// The texture format every input frame must carry. See
    /// [`D3d11NvencInputFormat`].
    pub input_format: D3d11NvencInputFormat,
    /// Encoded frame width in pixels.
    pub width: u32,
    /// Encoded frame height in pixels.
    pub height: u32,
    /// Must match the `pts` unit of whatever frames this receives.
    pub time_base: ffmpeg::Rational,
    /// The nominal rate NVENC uses for rate control and writes into the
    /// bitstream — not required to match the real interval between
    /// `consume` calls. See [`crate::elements::SwEncoderOptions::frame_rate`].
    pub frame_rate: ffmpeg::Rational,
    /// Target encoded bit rate, in bits per second.
    pub bit_rate: usize,
    /// Frames between keyframes (`AVCodecContext.gop_size`). Always set
    /// explicitly, for the join-latency and segmenting reasons
    /// [`crate::elements::SwEncoderOptions::gop_size`] documents.
    pub gop_size: u32,
}

/// Encodes GPU-resident `Pixel::D3D11` `Video` frames into `Packet`s on the
/// GPU's dedicated NVENC block — the hardware counterpart to
/// [`crate::elements::SwEncoder`], for a pipeline built entirely on one
/// shared `ID3D11Device` (see [`crate::elements::D3d11Renderer`]'s own docs
/// on why that means no explicit fence/sync is needed anywhere in this
/// stack). A `Filter`: receives via `Sink`, pushes what it produces into
/// its own single src pad.
///
/// # Why the input texture is copied rather than handed over directly
///
/// NVENC needs `AVCodecContext.hw_frames_ctx` set before `avcodec_open2`,
/// and every frame passed to `send_frame` must come from *that* context's
/// own pool. The `Pixel::D3D11` frames flowing through this crate don't:
/// [`crate::elements::D3d11Upload`],
/// `DxgiCaptureSource`'s GPU mode and
/// [`crate::elements::D3d11VideoCompositor`] all build their textures with
/// plain `windows-rs` calls and wrap them via
/// `platform::windows::d3d11va::wrap_d3d11_texture`, deliberately bypassing FFmpeg's
/// frames-context machinery altogether (that function's own docs explain
/// why: driving it from a hand-mirrored `AVD3D11VAFramesContext*` corrupted
/// memory badly enough to trip `/GS`).
///
/// So `consume` takes the other route: it lets libavutil allocate and own
/// the encoder's input pool through the ordinary, publicly documented
/// `av_hwframe_ctx_alloc`/`av_hwframe_ctx_init`/`av_hwframe_get_buffer`
/// API — touching only bindgen-generated `AVHWFramesContext` fields, never
/// the D3D11VA-specific struct that has no binding — and copies each
/// incoming texture into a pool texture with `CopySubresourceRegion`.
///
/// That copy never leaves the GPU. It replaces the
/// `D3d11Download` → `SwScaler` → `SwEncoder` chain that was previously the
/// only way to record a GPU-resident stream, which read every frame back
/// over PCIe, converted it on the CPU, and encoded it on the CPU.
///
/// # Frame delay
///
/// One frame can turn into zero or one packets per `send_frame` — NVENC's
/// lookahead and B-frame reordering delay some packets until later frames
/// arrive, or until `Eos` flushes what's left. `consume` drains
/// `receive_packet` in a loop after every `send_frame`/`send_eof`, the same
/// shape as [`crate::elements::SwEncoder`]'s own drain loop.
pub struct D3d11NvencEncoder {
    pp_log: PpLog,
    name: Arc<str>,
    encoder: ffmpeg::encoder::Video,
    /// Kept so every incoming texture can be checked before a context copy.
    /// `CopySubresourceRegion` returns no error value for a foreign resource,
    /// so relying on the call itself would silently corrupt encoder input.
    device: ID3D11Device,
    /// The shared immediate context. `CopySubresourceRegion` is a
    /// context-level call, not a device-level one, so this must be the
    /// exact same `Arc<Mutex<ID3D11DeviceContext>>` every other
    /// context-touching D3D11 element in this pipeline holds — see
    /// [`crate::elements::D3d11Download`]'s own docs on why a lone
    /// per-element context handle isn't enough.
    context: Arc<Mutex<ID3D11DeviceContext>>,
    /// Owned for this element's lifetime; the encoder holds its own
    /// `av_buffer_ref` on both of these.
    _hw_device_ctx: AvBufferRef,
    hw_frames_ctx: AvBufferRef,
    width: u32,
    height: u32,
    input_format: D3d11NvencInputFormat,
    /// Nominal frame duration in `time_base` ticks. NVENC leaves
    /// `AVPacket::duration` at zero; muxers such as
    /// [`crate::elements::HlsMuxer`] need it for precise segment durations.
    packet_duration: i64,
    /// Stamped onto every produced packet, since `avcodec_receive_packet`
    /// never sets `AVPacket.time_base` itself — see
    /// [`crate::elements::SwEncoder`]'s own field of the same name.
    time_base: ffmpeg::Rational,
    pad: SrcPad,
}

// SAFETY: `hw_device_ctx`/`hw_frames_ctx` are heap-allocated FFmpeg buffers
// with no thread affinity, and `encoder`'s own `Send` covers the codec
// context. `&mut self` on every method that touches them rules out
// concurrent access — same reasoning as `D3d11Decoder`.
unsafe impl Send for D3d11NvencEncoder {}

fn nominal_packet_duration(time_base: ffmpeg::Rational, frame_rate: ffmpeg::Rational) -> i64 {
    if frame_rate.numerator() <= 0 || time_base.numerator() <= 0 || time_base.denominator() <= 0 {
        return 0;
    }
    1i64.rescale(
        ffmpeg::Rational::new(frame_rate.denominator(), frame_rate.numerator()),
        time_base,
    )
    .max(1)
}

/// Allocates and initializes the encoder's own D3D11 input pool.
///
/// Every field written here is a documented public field of the
/// bindgen-generated `AVHWFramesContext`; `hwctx` (the D3D11VA-specific
/// struct libavutil layers on top, which `ffmpeg-sys-next` generates no
/// binding for) is deliberately left entirely to libavutil. That is the
/// whole difference between this and the approach recorded in
/// `wrap_d3d11_texture`'s docs as having corrupted memory.
unsafe fn create_hw_frames_ctx(
    hw_device_ctx: &AvBufferRef,
    options: &D3d11NvencEncoderOptions,
) -> std::result::Result<AvBufferRef, D3d11NvencEncoderError> {
    // SAFETY: `hw_device_ctx` owns a live initialized D3D11VA device context;
    // the allocation is wrapped immediately, only public
    // `AVHWFramesContext` fields are written before initialization, and the
    // helper updating bind flags has the same pre-init lifetime.
    unsafe {
        let buf = AvBufferRef::from_raw(ffi::av_hwframe_ctx_alloc(hw_device_ctx.as_ptr()))
            .ok_or(D3d11NvencEncoderError::HwFramesAlloc)?;

        let frames_ctx = (*buf.as_ptr()).data as *mut ffi::AVHWFramesContext;
        (*frames_ctx).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D11;
        (*frames_ctx).sw_format = options.input_format.sw_format();
        (*frames_ctx).width = options.width as i32;
        (*frames_ctx).height = options.height as i32;
        // Left at libavutil's dynamic `AVBufferPool` rather than a fixed
        // array texture, which is what a non-zero `initial_pool_size` asks
        // for. A `DXGI_FORMAT_NV12` *array* needs decoder-class binding to
        // be created at all, so pairing it with the shader-resource binding
        // below fails outright (`E_INVALIDARG`), while BGRA arrays succeed —
        // an asymmetry that would make the same options struct work for one
        // input format and not the other. The dynamic pool still recycles
        // textures through `AVBufferPool`; it just allocates them one at a
        // time, exactly as ffmpeg's own `hwupload` filter does on the path
        // this element was verified against.
        (*frames_ctx).initial_pool_size = 0;
        // Required, not an optimization: libavutil copies `BindFlags`
        // straight into its `D3D11_TEXTURE2D_DESC` and defaults it to
        // nothing, and `CreateTexture2D` rejects a `D3D11_USAGE_DEFAULT`
        // texture with no bind flags (`E_INVALIDARG`).
        or_frames_bind_flags(frames_ctx, D3D11_BIND_SHADER_RESOURCE.0 as u32);

        let code = ffi::av_hwframe_ctx_init(buf.as_ptr());
        if code < 0 {
            return Err(D3d11NvencEncoderError::HwFramesInit(
                code,
                options.width,
                options.height,
            ));
        }
        Ok(buf)
    }
}

impl D3d11NvencEncoder {
    /// `device` must be the same `ID3D11Device`, and `context` the same
    /// shared immediate context, every other D3D11 element in this pipeline
    /// uses — see this type's own docs on why.
    ///
    /// Opens the encoder eagerly, so a missing `h264_nvenc`/`hevc_nvenc`, a
    /// driver too old for the linked ffmpeg's NVENC API version, or a
    /// resolution this GPU's encode block rejects all surface here as a
    /// typed error rather than at the first frame.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        options: D3d11NvencEncoderOptions,
    ) -> std::result::Result<Self, D3d11NvencEncoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11NvencEncoder, &name, None);

        let context_device = {
            let context = context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // SAFETY: `context` is a live immediate context and `GetDevice`
            // returns an owned reference to its creating device.
            unsafe { context.GetDevice() }?
        };
        if context_device.as_raw() != device.as_raw() {
            return Err(D3d11NvencEncoderError::ContextDeviceMismatch);
        }

        let encoder_name = options.codec.encoder_name();
        let codec = ffmpeg::encoder::find_by_name(encoder_name)
            .ok_or_else(|| D3d11NvencEncoderError::CodecNotFound(encoder_name.into()))?;

        // SAFETY: `device` is live and the helper transfers a cloned COM
        // reference into the returned FFmpeg device context.
        let hw_device_ctx = unsafe { create_hw_device_ctx(device) }
            .map_err(D3d11NvencEncoderError::HwDeviceInit)?;
        // SAFETY: the device context remains live and `options` has already
        // passed the constructor's dimension/format validation.
        let hw_frames_ctx = unsafe { create_hw_frames_ctx(&hw_device_ctx, &options) }?;

        // From here on every early return has to release both contexts, so
        // the work is done in a closure and the cleanup written once.
        let opened = (|| -> std::result::Result<ffmpeg::encoder::Video, D3d11NvencEncoderError> {
            let mut ctx = ffmpeg::codec::context::Context::new_with_codec(codec);
            ctx.set_time_base(options.time_base);

            let mut video = ctx.encoder().video()?;
            video.set_width(options.width);
            video.set_height(options.height);
            // The *frame* format is the hardware one; `sw_format` on the
            // frames context above is what says what those textures hold.
            video.set_format(ffmpeg::format::Pixel::D3D11);
            video.set_time_base(options.time_base);
            video.set_frame_rate(Some(options.frame_rate));
            video.set_bit_rate(options.bit_rate);
            video.set_gop(options.gop_size);
            // SAFETY: `video` exclusively owns an unopened codec context;
            // ownership of the cloned frames-context reference is transferred
            // into the field before `open_as` can consume it.
            unsafe {
                let ptr = video.as_mut_ptr();
                (*ptr).hw_frames_ctx = hw_frames_ctx
                    .try_clone()
                    .ok_or(D3d11NvencEncoderError::HwFramesAlloc)?
                    .into_raw();
            }
            Ok(video.open_as(codec)?)
        })();

        let encoder = opened?;

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::of(MediaKind::VideoPacket)),
        );
        pp_info!(
            pp_log: &pp_log,
            "opened: codec={encoder_name}, {}x{}, input={:?}, bit_rate={}, gop_size={}",
            options.width,
            options.height,
            options.input_format,
            options.bit_rate,
            options.gop_size
        );
        Ok(Self {
            name,
            pp_log,
            encoder,
            device: device.clone(),
            context,
            _hw_device_ctx: hw_device_ctx,
            hw_frames_ctx,
            width: options.width,
            height: options.height,
            input_format: options.input_format,
            packet_duration: nominal_packet_duration(options.time_base, options.frame_rate),
            time_base: options.time_base,
            pad,
        })
    }

    /// This encoder's own codec parameters — what a
    /// [`crate::elements::Mp4Muxer`] track needs when there's no
    /// container/demuxer in the loop to get them from, same as
    /// [`crate::elements::SwEncoder::parameters`].
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        ffmpeg::codec::Parameters::from(&self.encoder)
    }

    /// Takes one texture from the encoder's own pool and copies `texture`'s
    /// `source_subresource` slice into it, under `context`'s lock since
    /// `CopySubresourceRegion` touches the shared immediate context.
    ///
    /// The returned frame owns its pool reference: dropping it returns the
    /// texture to libavutil's pool.
    fn stage_input(
        &self,
        texture: &ID3D11Texture2D,
        source_subresource: u32,
        source_box: &D3D11_BOX,
    ) -> std::result::Result<ffmpeg::frame::Video, D3d11NvencEncoderError> {
        let mut staged = ffmpeg::frame::Video::empty();
        // SAFETY: `staged` is a fresh writable frame and this encoder retains
        // the initialized frames context used to allocate its texture.
        unsafe {
            let code =
                ffi::av_hwframe_get_buffer(self.hw_frames_ctx.as_ptr(), staged.as_mut_ptr(), 0);
            if code < 0 {
                return Err(D3d11NvencEncoderError::HwFrameGet(code));
            }
        }

        // A pool frame carries its texture exactly the way every other
        // `Pixel::D3D11` frame in this crate does — pointer in `data[0]`,
        // array slice in `data[1]` — so the same reader works here.
        let (destination, destination_slice) =
            d3d11va_texture(&staged).ok_or(D3d11NvencEncoderError::MissingTexture)?;
        if destination.is_null() {
            return Err(D3d11NvencEncoderError::MissingTexture);
        }

        let context = self
            .context
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: `destination` is the non-null borrowed texture stored in the
        // live staged frame; `ManuallyDrop` prevents releasing that borrowed
        // reference. Slice/subresource bounds are validated before the copy,
        // and source/destination belong to the locked context's device.
        unsafe {
            // Borrowed, not owned: `ManuallyDrop` keeps this from releasing
            // a COM reference the pool frame still holds.
            let destination = std::mem::ManuallyDrop::new(ID3D11Texture2D::from_raw(destination));
            let mut destination_desc = Default::default();
            destination.GetDesc(&mut destination_desc);
            if destination_slice < 0
                || destination_slice as u64 >= u64::from(destination_desc.ArraySize)
            {
                return Err(D3d11NvencEncoderError::InvalidArrayIndex {
                    index: destination_slice,
                    array_size: destination_desc.ArraySize,
                });
            }
            let destination_subresource = (destination_slice as u32) * destination_desc.MipLevels;
            context.CopySubresourceRegion(
                &*destination,
                destination_subresource,
                0,
                0,
                0,
                texture,
                source_subresource,
                Some(source_box),
            );
        }
        Ok(staged)
    }

    fn drain(&mut self) -> Result<()> {
        let mut packet = ffmpeg::Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_time_base(self.time_base);
                    if packet.duration() == 0 && self.packet_duration > 0 {
                        packet.set_duration(self.packet_duration);
                    }
                    self.pad.push(MediaBuffer::Packet(Arc::new(packet)))?;
                    packet = ffmpeg::Packet::empty();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(D3d11NvencEncoderError::from(error).into()),
            }
        }
        Ok(())
    }

    fn encode(&mut self, frame: &ffmpeg::frame::Video) -> Result<()> {
        if frame.format() != ffmpeg::format::Pixel::D3D11 {
            pp_error!(self, "unsupported pixel format: {:?}", frame.format());
            return Err(D3d11NvencEncoderError::UnsupportedFormat(frame.format()).into());
        }
        if frame.width() != self.width || frame.height() != self.height {
            let error = D3d11NvencEncoderError::DimensionMismatch {
                actual_width: frame.width(),
                actual_height: frame.height(),
                expected_width: self.width,
                expected_height: self.height,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }

        let (source, source_slice) =
            d3d11va_texture(frame).ok_or(D3d11NvencEncoderError::MissingTexture)?;
        // SAFETY: `source` is borrowed from the still-live frame; null is
        // rejected and cloning the borrowed wrapper acquires an independent ref.
        let source = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&source)
                .ok_or(D3d11NvencEncoderError::MissingTexture)?
                .clone()
        };

        // SAFETY: `source` is a live COM texture; `GetDevice` returns an owned
        // reference to its creating device.
        let texture_device = unsafe { source.GetDevice() }.map_err(D3d11NvencEncoderError::from)?;
        if texture_device.as_raw() != self.device.as_raw() {
            let error = D3d11NvencEncoderError::DeviceMismatch;
            pp_error!(self, "{error}");
            return Err(error.into());
        }

        // The clone above owns one temporary COM reference; the incoming
        // frame keeps its independent ownership throughout the copy.
        let mut description = Default::default();
        // SAFETY: `description` is a live out-parameter for the live texture.
        unsafe { source.GetDesc(&mut description) };
        if description.Format != self.input_format.dxgi_format() {
            let error = D3d11NvencEncoderError::TextureFormatMismatch {
                actual: description.Format.0,
                expected: self.input_format.dxgi_format().0,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }

        if description.Width < self.width || description.Height < self.height {
            let error = D3d11NvencEncoderError::TextureTooSmall {
                actual_width: description.Width,
                actual_height: description.Height,
                expected_width: self.width,
                expected_height: self.height,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }
        if source_slice < 0 || source_slice as u64 >= u64::from(description.ArraySize) {
            let error = D3d11NvencEncoderError::InvalidArrayIndex {
                index: source_slice,
                array_size: description.ArraySize,
            };
            pp_error!(self, "{error}");
            return Err(error.into());
        }
        let source_subresource = (source_slice as u32) * description.MipLevels;
        let source_box = D3D11_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: self.width,
            bottom: self.height,
            back: 1,
        };

        let mut staged = self
            .stage_input(&source, source_subresource, &source_box)
            .inspect_err(|error| pp_error!(self, "staging the input texture failed: {error}"))?;
        // Metadata is part of the buffer contract, not decoration — the
        // staged frame is a different AVFrame than the one that arrived.
        staged.set_pts(frame.pts());
        staged.set_color_space(frame.color_space());
        staged.set_color_range(frame.color_range());

        self.encoder
            .send_frame(&staged)
            .inspect_err(|error| pp_error!(self, "send_frame failed: {error}"))
            .map_err(D3d11NvencEncoderError::from)?;
        self.drain()
    }
}

impl Drop for D3d11NvencEncoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_frames_ctx and hw_device_ctx");
    }
}

impl Element for D3d11NvencEncoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11NvencEncoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11NvencEncoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11NvencEncoder {
    /// NVENC reads the texture directly; a system-memory frame needs a D3d11Upload first.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::of(MediaKind::VideoFrame).in_memory(MemoryDomain::D3d11))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.encode(&frame),
            MediaBuffer::Eos => {
                self.encoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(D3d11NvencEncoderError::from)?;
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => Err(D3d11NvencEncoderError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Same deliberate choice as `SwEncoder::control`: Seek is forwarded
        // without flushing, since NVENC can still emit packets originating
        // before the seek from later `send_frame` calls. A caller needing a
        // hard encoded-stream discontinuity rebuilds the encoder.
        self.pad.control(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::try_d3d11_device as try_device;

    fn options(
        codec: D3d11NvencCodec,
        input_format: D3d11NvencInputFormat,
    ) -> D3d11NvencEncoderOptions {
        D3d11NvencEncoderOptions {
            codec,
            input_format,
            width: 320,
            height: 240,
            time_base: ffmpeg::Rational::new(1, 30),
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 1_000_000,
            gop_size: 30,
        }
    }

    /// Builds a GPU texture the way every real producer in this crate does
    /// — plain `windows-rs`, then `wrap_d3d11_texture` — so this exercises
    /// the actual input shape `consume` receives rather than a frame from
    /// the encoder's own pool.
    fn gpu_frame_with_backing_size(
        device: &ID3D11Device,
        format: DXGI_FORMAT,
        texture_width: u32,
        texture_height: u32,
        frame_width: u32,
        frame_height: u32,
    ) -> ffmpeg::frame::Video {
        use windows::Win32::Graphics::{
            Direct3D11::{D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT},
            Dxgi::Common::DXGI_SAMPLE_DESC,
        };
        let description = D3D11_TEXTURE2D_DESC {
            Width: texture_width,
            Height: texture_height,
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
        let mut texture = None;
        // SAFETY: the texture descriptor is fully initialized, no initial
        // data is supplied, and `texture` is a live out-parameter.
        unsafe { device.CreateTexture2D(&description, None, Some(&mut texture)) }
            .expect("creating a plain D3D11 texture should succeed on a working device");
        crate::platform::windows::d3d11va::wrap_d3d11_texture(
            texture.expect("CreateTexture2D succeeded without producing a texture"),
            frame_width,
            frame_height,
        )
        .unwrap()
    }

    fn gpu_frame(
        device: &ID3D11Device,
        format: DXGI_FORMAT,
        width: u32,
        height: u32,
    ) -> ffmpeg::frame::Video {
        gpu_frame_with_backing_size(device, format, width, height, width, height)
    }

    /// The whole point of this element, end to end: real GPU textures in,
    /// real encoded packets out, for both input formats. Asserts on the
    /// packets rather than on `consume` merely returning `Ok`, since a
    /// silently-empty encoder would pass the latter.
    #[test]
    fn encodes_gpu_textures_into_packets() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let (width, height) = (320u32, 240u32);

        for (codec, input_format) in [
            (D3d11NvencCodec::H264, D3d11NvencInputFormat::Nv12),
            (D3d11NvencCodec::H264, D3d11NvencInputFormat::Bgra),
            (D3d11NvencCodec::H265, D3d11NvencInputFormat::Nv12),
            (D3d11NvencCodec::H265, D3d11NvencInputFormat::Bgra),
        ] {
            let mut encoder = match D3d11NvencEncoder::new(
                format!("test-nvenc-encode-{codec:?}-{input_format:?}"),
                &device,
                context.clone(),
                options(codec, input_format),
            ) {
                Ok(encoder) => encoder,
                Err(error) if is_absent_hardware(&error) => {
                    eprintln!(
                        "skipping {codec:?}/{input_format:?}: NVENC unavailable here: {error}"
                    );
                    continue;
                }
                Err(error) => panic!("{codec:?}/{input_format:?} failed to open: {error}"),
            };

            let packets = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counted = packets.clone();
            let sink = crate::elements::AppSink::new("test-nvenc-sink", move |buf| {
                if matches!(buf, MediaBuffer::Packet(_)) {
                    counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(())
            });
            encoder.src_pads()[0].link(Box::new(sink));

            // The factory runs whenever the pool is empty, so this both
            // exercises fresh textures and — once frames start returning to
            // the pool — the reuse a real source does.
            let pool = crate::pool::UnboundObjectPool::new(
                0,
                {
                    let device = device.clone();
                    let format = input_format.dxgi_format();
                    move || gpu_frame(&device, format, width, height)
                },
                |_| {},
            );
            for index in 0..30i64 {
                let mut frame = pool.get();
                frame.set_pts(Some(index));
                encoder
                    .consume(MediaBuffer::Video(Arc::new(frame)))
                    .unwrap_or_else(|error| {
                        panic!("{codec:?}/{input_format:?} frame {index}: {error}")
                    });
            }
            encoder
                .consume(MediaBuffer::Eos)
                .unwrap_or_else(|error| panic!("{codec:?}/{input_format:?} Eos: {error}"));

            let produced = packets.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                produced > 0,
                "{codec:?}/{input_format:?}: 30 GPU frames produced no packets at all"
            );
        }
    }

    /// Only the two conditions this machine genuinely may not meet — no
    /// NVENC in the linked ffmpeg build, or a driver too old for its NVENC
    /// API version — are grounds for skipping. Anything else is a real
    /// failure and must not be swallowed: an earlier version of this test
    /// treated *every* error as "skip" and reported a green run while
    /// `av_hwframe_ctx_init` was failing with `E_INVALIDARG` on every call.
    fn is_absent_hardware(error: &D3d11NvencEncoderError) -> bool {
        match error {
            D3d11NvencEncoderError::CodecNotFound(_) => true,
            // `AVERROR_EXTERNAL` is what NVENC returns when no capable
            // NVIDIA device is present. An old driver reports `ENOSYS`.
            // Other codec-open failures are real regressions, not skips.
            D3d11NvencEncoderError::Ffmpeg(ffmpeg::Error::External) => true,
            D3d11NvencEncoderError::Ffmpeg(ffmpeg::Error::Other { errno })
                if *errno == ffmpeg::util::error::ENOSYS =>
            {
                true
            }
            _ => false,
        }
    }

    /// Opening is where every environmental precondition is checked at
    /// once: `h264_nvenc` present in the linked ffmpeg, a driver new enough
    /// for its NVENC API version, and a frames context this GPU accepts.
    #[test]
    fn opens_for_both_codecs_and_input_formats_on_real_hardware() {
        let Some((device, context)) = try_device() else {
            return;
        };
        for (codec, input_format) in [
            (D3d11NvencCodec::H264, D3d11NvencInputFormat::Nv12),
            (D3d11NvencCodec::H264, D3d11NvencInputFormat::Bgra),
            (D3d11NvencCodec::H265, D3d11NvencInputFormat::Nv12),
            (D3d11NvencCodec::H265, D3d11NvencInputFormat::Bgra),
        ] {
            match D3d11NvencEncoder::new(
                format!("test-nvenc-{codec:?}-{input_format:?}"),
                &device,
                context.clone(),
                options(codec, input_format),
            ) {
                Ok(encoder) => {
                    assert_eq!(encoder.element_type(), ElementType::D3d11NvencEncoder);
                }
                Err(error) if is_absent_hardware(&error) => {
                    eprintln!(
                        "skipping {codec:?}/{input_format:?}: NVENC unavailable here: {error}"
                    );
                }
                Err(error) => panic!("{codec:?}/{input_format:?} failed to open: {error}"),
            }
        }
    }

    #[test]
    fn rejects_a_context_from_a_different_device() {
        let Some((device, _)) = try_device() else {
            return;
        };
        let Some((_other_device, other_context)) = try_device() else {
            return;
        };

        let error = match D3d11NvencEncoder::new(
            "test-nvenc-context-mismatch",
            &device,
            other_context,
            options(D3d11NvencCodec::H264, D3d11NvencInputFormat::Nv12),
        ) {
            Ok(_) => panic!("a context from another device must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            D3d11NvencEncoderError::ContextDeviceMismatch
        ));
    }

    /// Every external texture property is checked before the void-returning
    /// D3D11 copy call. Each rejection is followed by valid input to prove a
    /// bad frame does not poison the encoder's subsequent state.
    #[test]
    fn rejects_invalid_textures_then_continues_encoding() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let Some((foreign_device, _)) = try_device() else {
            return;
        };
        let mut encoder = match D3d11NvencEncoder::new(
            "test-nvenc-texture-validation",
            &device,
            context,
            options(D3d11NvencCodec::H264, D3d11NvencInputFormat::Nv12),
        ) {
            Ok(encoder) => encoder,
            Err(error) if is_absent_hardware(&error) => {
                eprintln!("skipping: NVENC unavailable here: {error}");
                return;
            }
            Err(error) => panic!("encoder failed to open: {error}"),
        };

        let foreign_pool = crate::pool::UnboundObjectPool::new(
            0,
            move || gpu_frame(&foreign_device, DXGI_FORMAT_NV12, 320, 240),
            |_| {},
        );
        let error = encoder
            .consume(MediaBuffer::Video(Arc::new(foreign_pool.get())))
            .expect_err("a foreign-device texture must be rejected");
        assert!(matches!(
            error,
            crate::error::Error::D3d11NvencEncoderError(D3d11NvencEncoderError::DeviceMismatch)
        ));

        let small_device = device.clone();
        let small_pool = crate::pool::UnboundObjectPool::new(
            0,
            move || {
                gpu_frame_with_backing_size(&small_device, DXGI_FORMAT_NV12, 160, 120, 320, 240)
            },
            |_| {},
        );
        let error = encoder
            .consume(MediaBuffer::Video(Arc::new(small_pool.get())))
            .expect_err("a too-small backing texture must be rejected");
        assert!(matches!(
            error,
            crate::error::Error::D3d11NvencEncoderError(
                D3d11NvencEncoderError::TextureTooSmall { .. }
            )
        ));

        let slice_device = device.clone();
        let invalid_slice_pool = crate::pool::UnboundObjectPool::new(
            0,
            move || gpu_frame(&slice_device, DXGI_FORMAT_NV12, 320, 240),
            |_| {},
        );
        let mut invalid_slice = invalid_slice_pool.get();
        // SAFETY: this test uniquely owns the unpublished frame; address 1 is
        // the D3D11VA integer encoding for slice 1 and is never dereferenced.
        unsafe {
            // D3D11 frames encode the integer array slice in this pointer
            // slot; a dangling address of 1 therefore represents slice 1.
            (*invalid_slice.as_mut_ptr()).data[1] = std::ptr::dangling_mut::<u8>();
        }
        let error = encoder
            .consume(MediaBuffer::Video(Arc::new(invalid_slice)))
            .expect_err("an out-of-range texture-array slice must be rejected");
        assert!(matches!(
            error,
            crate::error::Error::D3d11NvencEncoderError(
                D3d11NvencEncoderError::InvalidArrayIndex { .. }
            )
        ));

        let packets = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = packets.clone();
        encoder.src_pads()[0].link(Box::new(crate::elements::AppSink::new(
            "test-nvenc-validation-sink",
            move |buf| {
                if matches!(buf, MediaBuffer::Packet(_)) {
                    counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(())
            },
        )));

        // A padded backing texture is valid when the AVFrame's visible size
        // matches the encoder. `D3D11_BOX` copies just that visible region.
        let padded_device = device.clone();
        let padded_pool = crate::pool::UnboundObjectPool::new(
            0,
            move || {
                gpu_frame_with_backing_size(&padded_device, DXGI_FORMAT_NV12, 352, 256, 320, 240)
            },
            |_| {},
        );
        let mut padded = padded_pool.get();
        padded.set_pts(Some(0));
        encoder
            .consume(MediaBuffer::Video(Arc::new(padded)))
            .expect("a padded texture containing the visible frame must encode");

        let valid_device = device.clone();
        let valid_pool = crate::pool::UnboundObjectPool::new(
            0,
            move || gpu_frame(&valid_device, DXGI_FORMAT_NV12, 320, 240),
            |_| {},
        );
        for index in 1..30i64 {
            let mut frame = valid_pool.get();
            frame.set_pts(Some(index));
            encoder
                .consume(MediaBuffer::Video(Arc::new(frame)))
                .unwrap_or_else(|error| panic!("valid frame {index}: {error}"));
        }
        encoder.consume(MediaBuffer::Eos).expect("valid Eos");
        assert!(
            packets.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "valid frames after rejected inputs produced no packets"
        );
    }

    /// The input format is fixed at open time, so a texture that does not
    /// match must be rejected with a typed error rather than silently
    /// producing garbage or tripping a Windows API failure later.
    #[test]
    fn rejects_a_non_d3d11_frame() {
        let Some((device, context)) = try_device() else {
            return;
        };
        let mut encoder = match D3d11NvencEncoder::new(
            "test-nvenc-reject",
            &device,
            context,
            options(D3d11NvencCodec::H264, D3d11NvencInputFormat::Nv12),
        ) {
            Ok(encoder) => encoder,
            Err(error) if is_absent_hardware(&error) => {
                eprintln!("skipping: NVENC unavailable here: {error}");
                return;
            }
            Err(error) => panic!("encoder failed to open: {error}"),
        };

        let pool = crate::pool::UnboundObjectPool::new(
            0,
            || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 320, 240),
            |_| {},
        );
        let error = encoder
            .consume(MediaBuffer::Video(Arc::new(pool.get())))
            .expect_err("a CPU NV12 frame is not a Pixel::D3D11 frame");
        assert!(
            error.to_string().contains("Pixel::D3D11"),
            "expected an UnsupportedFormat error, got: {error}"
        );
    }
}
