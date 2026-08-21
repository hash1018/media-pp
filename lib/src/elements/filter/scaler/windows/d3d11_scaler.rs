use std::sync::{Arc, Mutex};

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};
use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::{
        Direct3D11::{
            D3D11_BIND_DECODER, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE,
            D3D11_BIND_UNORDERED_ACCESS, D3D11_BIND_VIDEO_ENCODER, D3D11_TEXTURE2D_DESC,
            D3D11_USAGE_DEFAULT, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
            ID3D11VideoContext, ID3D11VideoDevice,
        },
        Dxgi::Common::{
            DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC,
        },
    },
    core::Interface,
};

use super::video_processor::{BltColorSpaces, InputShape, ScaleProcessor, color_space};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    error::Result,
    pad::SrcPad,
    platform::windows::d3d11va::{d3d11va_texture, wrap_d3d11_texture},
    pool::UnboundObjectPool,
};

/// Errors specific to `D3d11Scaler`. Converts into the crate-wide `Error`
/// via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11ScalerError {
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    #[error("D3d11Scaler only scales Pixel::D3D11 frames, got {0:?}")]
    UnsupportedFormat(ffmpeg::format::Pixel),

    #[error(
        "frame claimed the D3D11 pixel format but carries no texture — must \
         come from D3d11Upload/D3d11Decoder/DxgiCaptureSource's GPU mode/D3d11VideoCompositor"
    )]
    InvalidD3d11Frame,

    #[error(
        "D3d11Scaler only scales DXGI_FORMAT_NV12 or DXGI_FORMAT_B8G8R8A8_UNORM textures, got {0:?}"
    )]
    UnsupportedTextureFormat(DXGI_FORMAT),

    #[error(
        "a Pixel::D3D11 frame's texture lives on a different ID3D11Device \
         than this D3d11Scaler was created with — every D3D11 element in one \
         pipeline must share exactly one device for zero-copy to be valid"
    )]
    DeviceMismatch,

    #[error(
        "the supplied ID3D11DeviceContext belongs to a different ID3D11Device than this \
         D3d11Scaler"
    )]
    ContextDeviceMismatch,

    #[error("D3d11Scaler output dimensions must be non-zero, got {width}x{height}")]
    InvalidOutputDimensions { width: u32, height: u32 },

    #[error(
        "D3D11 texture is {actual_width}x{actual_height}, smaller than the \
         frame's {expected_width}x{expected_height} visible size"
    )]
    TextureTooSmall {
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },

    #[error("D3D11 texture array index {index} is outside ArraySize {array_size}")]
    InvalidArrayIndex { index: isize, array_size: u32 },

    #[error("D3d11Scaler does not accept multisampled textures (SampleDesc.Count={0})")]
    MultisampledTexture(u32),

    #[error(
        "an NV12 surface cannot be {width}x{height}: its chroma plane is half \
         resolution in both directions, so both must be even"
    )]
    OddNv12Output { width: u32, height: u32 },

    #[error("this GPU's video processor does not support {0:?} on the side it is needed for")]
    UnsupportedByVideoProcessor(DXGI_FORMAT),

    #[error("D3d11Scaler only accepts Video and Eos buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
}

/// What surface format [`D3d11Scaler`] writes its output in.
///
/// The two variants name the only formats this crate's D3D11 elements
/// produce and consume, and `Preserve` is the third useful answer: a
/// caller resizing a decoder's output usually should not have to know
/// which of them is arriving, since the input side is learned from the
/// frames themselves rather than declared (see [`D3d11Scaler`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D3d11ScalerFormat {
    /// Write whatever the input carried — a pure resize, and the only
    /// setting that needs no assumption about what is upstream.
    Preserve,
    /// `DXGI_FORMAT_NV12` — what [`crate::elements::D3d11Decoder`] produces
    /// and what a hardware encoder ingests.
    Nv12,
    /// `DXGI_FORMAT_B8G8R8A8_UNORM` — the format everything that
    /// composites works in: [`crate::elements::D3d11VideoCompositor`]
    /// layers, [`crate::elements::D3d11ChromaKey`], and
    /// [`crate::elements::D3d11Download`].
    Bgra,
}

impl D3d11ScalerFormat {
    /// The DXGI format to write, given what the input turned out to hold.
    fn resolve(self, input: DXGI_FORMAT) -> DXGI_FORMAT {
        match self {
            D3d11ScalerFormat::Preserve => input,
            D3d11ScalerFormat::Nv12 => DXGI_FORMAT_NV12,
            D3d11ScalerFormat::Bgra => DXGI_FORMAT_B8G8R8A8_UNORM,
        }
    }
}

/// Resizes and, when asked, converts GPU-resident `Pixel::D3D11` `Video`
/// frames without ever touching the CPU, through D3D11's own video
/// processor (`VideoProcessorBlt`). A `Filter`: receives via `Sink`, pushes
/// the scaled frame into its own single src pad.
///
/// This is what makes a resolution change possible inside a D3D11 pipeline
/// at all — `D3d11Decoder -> D3d11Scaler -> D3d11NvencEncoder` stays on the
/// GPU end to end, where the alternatives were
/// [`crate::elements::D3d11Download`] plus [`crate::elements::SwScaler`]
/// plus [`crate::elements::D3d11Upload`] (two PCIe crossings per frame), or
/// [`crate::elements::D3d11VideoCompositor`], which resizes a layer but is a
/// source: it produces its own fixed-rate timeline rather than preserving the
/// timestamps flowing through it.
///
/// # Resize, and optionally convert
///
/// [`D3d11ScalerFormat::Preserve`] gives back whatever the input held —
/// NV12 stays NV12 and BGRA stays BGRA — by handing both sides of the `Blt`
/// the same format and the same colorimetry, which is what keeps the result
/// a pure resize. Naming `Nv12` or `Bgra` instead asks the same `Blt` for
/// the conversion as well, at no extra pass.
///
/// That conversion is what connects the two halves of this crate's D3D11
/// support. A decoder produces NV12, while everything that composites works
/// in BGRA — [`crate::elements::D3d11ChromaKey`] keys it,
/// [`crate::elements::D3d11VideoCompositor`] takes it as a layer, and
/// [`crate::elements::D3d11Download`] reads it — so
/// `D3d11Decoder -> D3d11Scaler(Bgra) -> D3d11ChromaKey` is what lets a
/// decoded green screen be keyed without ever leaving the GPU. The other
/// direction exists for the same reason in reverse, though it is needed
/// less often: [`crate::elements::D3d11NvencEncoder`] ingests either format
/// and [`crate::elements::D3d11Renderer`] draws either.
///
/// A converted frame is retagged for what it now holds rather than
/// inheriting the source's tags — an encoder reads those to describe its
/// stream. RGB is tagged full-range; Y'CbCr is tagged BT.709 limited, the
/// same definition [`crate::elements::D3d11VideoCompositor`] and
/// [`crate::elements::CudaConverter`] use, so converted frames and
/// composited ones agree.
///
/// Shader-resource-only textures from [`crate::elements::D3d11Upload`] and
/// DXGI GPU capture cannot be used directly as D3D11 video-processor input
/// views. Those inputs receive one GPU-to-GPU compatibility copy first;
/// decoder surfaces and compositor/scaler outputs already have suitable bind
/// flags and go straight into the processor. Neither path maps pixels to the
/// CPU.
///
/// # The processor is built from the first frame
///
/// A video processor's input size and surface format are fixed when its
/// enumerator is created, so there is nothing to build until a frame
/// actually shows what those are. The processor is therefore built lazily
/// from the first frame and rebuilt if a later frame's size or format
/// changes — a live source can renegotiate mid-stream, and a decoder
/// rebuilt after a seek can hand out differently shaped surfaces. The
/// output size is fixed at construction. This is the same lazy-rebuild
/// contract [`crate::elements::CudaScaler`] and
/// [`crate::elements::SwScaler`] have for their own scaling state.
///
/// # Device and context
///
/// `VideoProcessorBlt` and the processor's own setters are context-level
/// calls on `ID3D11VideoContext`, which is the shared immediate context
/// under another interface — not a separate one. So `context` must be the
/// exact same `Arc<Mutex<ID3D11DeviceContext>>` every other context-touching
/// D3D11 element in this pipeline shares (e.g.
/// `render_common::D3d11GpuContext::context()`), and this element holds that
/// lock for a whole configure-and-`Blt` sequence, for the same reason
/// [`crate::elements::D3d11Download`] holds it across its own copy and map.
pub struct D3d11Scaler {
    pp_log: PpLog,
    name: Arc<str>,
    device: ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    /// Queried once at construction rather than per frame: both are the
    /// `device`/`context` above under another interface, so they are exactly
    /// as shareable — and exactly as much in need of `context`'s lock.
    video_device: ID3D11VideoDevice,
    video_context: ID3D11VideoContext,
    width: u32,
    height: u32,
    /// What to write. Resolved against each frame's own format, since
    /// [`D3d11ScalerFormat::Preserve`] has no answer until one arrives.
    format: D3d11ScalerFormat,
    /// Built from the first frame and rebuilt when the input changes — see
    /// this type's own docs.
    processor: Option<ScaleProcessor>,
    pad: SrcPad,
    /// Reused across every scaled frame — see [`UnboundObjectPool`]'s docs.
    /// Only the small CPU-side `AVFrame` wrapper is actually reused; the
    /// output texture itself is a fresh allocation per frame, since
    /// downstream may still hold `Arc` clones of the previous one.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

struct ValidatedInput {
    texture: ID3D11Texture2D,
    array_slice: u32,
    shape: InputShape,
    desc: D3D11_TEXTURE2D_DESC,
}

// SAFETY: every field is either a `windows-rs` COM interface wrapper (the
// device-level ones free-threaded, the context-level ones behind `context`'s
// own `Mutex`) or plain data. `&mut self` on every method that touches
// non-`Arc`/`Mutex` state already rules out concurrent access to those parts
// from multiple threads — same reasoning as `D3d11Download`.
unsafe impl Send for D3d11Scaler {}

/// The colorimetry an output frame should be tagged with, given what came
/// in and what surface format is being written.
///
/// An unchanged format is not a conversion, so the source's own tags cross
/// untouched — including `Unspecified`, which is information the next
/// element is entitled to see rather than something to invent an answer
/// for. A conversion has to replace them: the pixels are no longer what the
/// old tags described, and an encoder builds its stream's description from
/// these. The two answers match what this crate already produces elsewhere
/// — `D3d11VideoCompositor` tags its BGRA output full-range RGB, and
/// `CudaConverter` tags its Y'CbCr output BT.709 limited.
fn converted_colorimetry(
    frame: &ffmpeg::frame::Video,
    output_format: DXGI_FORMAT,
    input_format: DXGI_FORMAT,
) -> (ffmpeg::color::Space, ffmpeg::color::Range) {
    if output_format == input_format {
        return (frame.color_space(), frame.color_range());
    }
    if output_format == DXGI_FORMAT_NV12 {
        (ffmpeg::color::Space::BT709, ffmpeg::color::Range::MPEG)
    } else {
        (ffmpeg::color::Space::RGB, ffmpeg::color::Range::JPEG)
    }
}

fn validate_output_size(
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
) -> std::result::Result<(), D3d11ScalerError> {
    if format == DXGI_FORMAT_NV12 && (!width.is_multiple_of(2) || !height.is_multiple_of(2)) {
        return Err(D3d11ScalerError::OddNv12Output { width, height });
    }
    Ok(())
}

fn video_processor_accepts_input(desc: &D3D11_TEXTURE2D_DESC) -> bool {
    let accepted_bind_flags = (D3D11_BIND_DECODER.0
        | D3D11_BIND_VIDEO_ENCODER.0
        | D3D11_BIND_RENDER_TARGET.0
        | D3D11_BIND_UNORDERED_ACCESS.0) as u32;
    desc.Usage == D3D11_USAGE_DEFAULT
        && (desc.BindFlags == 0 || desc.BindFlags & accepted_bind_flags != 0)
}

fn create_processor_input_texture(
    device: &ID3D11Device,
    source: &D3D11_TEXTURE2D_DESC,
) -> std::result::Result<ID3D11Texture2D, windows::core::Error> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: source.Width,
        Height: source.Height,
        MipLevels: 1,
        ArraySize: 1,
        Format: source.Format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: 0,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe {
        device.CreateTexture2D(&desc, None, Some(&mut texture))?;
    }
    Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
}

impl D3d11Scaler {
    /// `device` must be the same `ID3D11Device`, and `context` the same
    /// shared immediate context, every other D3D11 element in this pipeline
    /// uses — see this type's own docs on why.
    ///
    /// `format`, `width`, and `height` are what every output frame will be;
    /// the input side is learned from whatever frames actually arrive, so
    /// this needs neither input dimensions nor an input format up front
    /// (see this type's own docs). `format` is ordered before the size for
    /// the same reason [`crate::elements::SwScaler`] orders its own
    /// `dst_format` that way.
    pub fn new(
        name: impl Into<String>,
        device: &ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        format: D3d11ScalerFormat,
        width: u32,
        height: u32,
    ) -> std::result::Result<Self, D3d11ScalerError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Scaler, &name, None);
        if width == 0 || height == 0 {
            return Err(D3d11ScalerError::InvalidOutputDimensions { width, height });
        }
        // The caller's own mistake is diagnosed before the adapter's
        // capabilities are: a context belonging to another device is
        // reported as `ContextDeviceMismatch` even on hardware with no
        // video processor, where casting the device first would mask it as
        // a bare `E_NOINTERFACE`.
        let video_context: ID3D11VideoContext = {
            let context = context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let context_device = unsafe { context.GetDevice() }?;
            if context_device.as_raw() != device.as_raw() {
                return Err(D3d11ScalerError::ContextDeviceMismatch);
            }
            context.cast()?
        };
        let video_device: ID3D11VideoDevice = device.cast()?;
        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: dst={width}x{height} {format:?}");
        Ok(Self {
            name,
            pp_log,
            device: device.clone(),
            context,
            video_device,
            video_context,
            width,
            height,
            format,
            processor: None,
            pad,
            pool,
        })
    }

    /// Rejects anything that is not a texture this element can actually
    /// scale, so a bad frame fails here rather than as an opaque D3D11
    /// error code — or, worse, as a `Blt` reading a texture that belongs to
    /// another device. Returns the borrowed texture, its array slice, and
    /// the input shape the processor has to be configured for.
    fn validate(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<ValidatedInput, D3d11ScalerError> {
        if frame.format() != ffmpeg::format::Pixel::D3D11 {
            return Err(D3d11ScalerError::UnsupportedFormat(frame.format()));
        }
        let (texture_raw, index) =
            d3d11va_texture(frame).ok_or(D3d11ScalerError::InvalidD3d11Frame)?;
        if texture_raw.is_null() {
            return Err(D3d11ScalerError::InvalidD3d11Frame);
        }
        // Safety: `texture_raw` is a borrowed raw `ID3D11Texture2D*` — still
        // owned by `frame`'s own buffer reference, not by us. `.clone()`
        // (`AddRef`) gives an independently ref-counted handle, valid for as
        // long as the caller keeps it.
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .ok_or(D3d11ScalerError::InvalidD3d11Frame)?
                .clone()
        };

        let texture_device = unsafe { texture.GetDevice() }?;
        if texture_device.as_raw() != self.device.as_raw() {
            return Err(D3d11ScalerError::DeviceMismatch);
        }

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut desc) };
        if desc.Format != DXGI_FORMAT_NV12 && desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
            return Err(D3d11ScalerError::UnsupportedTextureFormat(desc.Format));
        }
        if desc.Width < frame.width() || desc.Height < frame.height() {
            return Err(D3d11ScalerError::TextureTooSmall {
                actual_width: desc.Width,
                actual_height: desc.Height,
                expected_width: frame.width(),
                expected_height: frame.height(),
            });
        }
        if index < 0 || index as u64 >= u64::from(desc.ArraySize) {
            return Err(D3d11ScalerError::InvalidArrayIndex {
                index,
                array_size: desc.ArraySize,
            });
        }
        if desc.SampleDesc.Count != 1 {
            return Err(D3d11ScalerError::MultisampledTexture(desc.SampleDesc.Count));
        }

        Ok(ValidatedInput {
            texture,
            array_slice: index as u32,
            shape: InputShape {
                // The visible size, not the texture's: a decoder pads its
                // surfaces up to its own alignment, and the padding is not
                // part of the picture.
                width: frame.width(),
                height: frame.height(),
                format: desc.Format,
            },
            desc,
        })
    }

    fn scale(&mut self, frame: &ffmpeg::frame::Video) -> Result<()> {
        let validated = self
            .validate(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        let input = validated.shape;
        let output_format = self.format.resolve(input.format);

        validate_output_size(output_format, self.width, self.height)
            .inspect_err(|error| pp_error!(self, "{error}"))?;

        // `D3d11Upload` and DXGI GPU capture produce shader-resource-only
        // textures. D3D11 does not permit that bind-flag combination for a
        // video-processor input view, so copy those surfaces to a temporary
        // default-usage texture with no bind flags. Decoder surfaces and
        // compositor/scaler outputs already carry an accepted video bind flag
        // and go straight into the processor without this copy.
        let input_copy = if video_processor_accepts_input(&validated.desc) {
            None
        } else {
            Some(
                create_processor_input_texture(&self.device, &validated.desc)
                    .inspect_err(|error| {
                        pp_error!(
                            self,
                            "failed to allocate a compatible input texture: {error}"
                        )
                    })
                    .map_err(D3d11ScalerError::from)?,
            )
        };

        let output = create_output_texture(&self.device, output_format, self.width, self.height)
            .inspect_err(|error| pp_error!(self, "failed to allocate the output texture: {error}"))
            .map_err(D3d11ScalerError::from)?;
        // What each side holds. The two are identical whenever the format
        // is unchanged, which is what keeps a pure resize from becoming a
        // color conversion; when they differ they are also the tags this
        // element is about to stamp on the outgoing frame, so the processor
        // and the frame's own metadata describe the same pixels.
        let (output_space, output_range) =
            converted_colorimetry(frame, output_format, input.format);
        let color_spaces = BltColorSpaces {
            input: color_space(
                frame.color_space(),
                frame.color_range(),
                input.height,
                input.format,
            ),
            output: color_space(output_space, output_range, self.height, output_format),
        };

        {
            let context = self
                .context
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let (processor_input, processor_slice) = match &input_copy {
                Some(copy) => {
                    let source_subresource = validated.array_slice * validated.desc.MipLevels;
                    unsafe {
                        context.CopySubresourceRegion(
                            copy,
                            0,
                            0,
                            0,
                            0,
                            &validated.texture,
                            source_subresource,
                            None,
                        );
                    }
                    (copy, 0)
                }
                None => (&validated.texture, validated.array_slice),
            };
            if !self.processor.as_ref().is_some_and(|processor| {
                processor.matches(input, self.width, self.height, output_format)
            }) {
                pp_debug!(
                    self,
                    "input is {}x{} {:?}, output {:?}, building the video processor",
                    input.width,
                    input.height,
                    input.format,
                    output_format
                );
                // Assigned only once the new processor exists, so a
                // failure here leaves the previous working one in place
                // rather than an empty slot.
                self.processor = Some(
                    ScaleProcessor::new(
                        &self.video_device,
                        &self.video_context,
                        input,
                        self.width,
                        self.height,
                        output_format,
                    )
                    .inspect_err(|error| {
                        pp_error!(self, "failed to build the video processor: {error}")
                    })?,
                );
            }
            self.processor
                .as_mut()
                .expect("the processor was just built or already matched")
                .scale(
                    &self.video_device,
                    &self.video_context,
                    processor_input,
                    processor_slice,
                    &output,
                    color_spaces,
                )
                .inspect_err(|error| pp_error!(self, "scale failed: {error}"))?;
        }

        let mut scaled = self.pool.get();
        // Overwrites the pooled slot's previous contents in place —
        // `ffmpeg::frame::Video`'s own `Drop` runs on whatever was there
        // before, releasing that frame's GPU texture right here.
        *scaled = wrap_d3d11_texture(output, self.width, self.height);
        scaled.set_pts(frame.pts());
        // A pure resize carries the source's tags through unchanged; a
        // conversion replaces them, because the pixels are no longer what
        // they described. `converted_colorimetry` is what decides which of
        // those this was.
        scaled.set_color_space(output_space);
        scaled.set_color_range(output_range);

        self.pad.push(MediaBuffer::Video(Arc::new(scaled)))
    }
}

impl Element for D3d11Scaler {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11Scaler
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11Scaler {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11Scaler {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Video(frame) => self.scale(&frame),
            // Nothing is buffered here — one `Blt` per frame, pushed
            // before `consume` returns — so there is nothing to drain.
            MediaBuffer::Eos => self.pad.push(MediaBuffer::Eos),
            other => {
                let kind = other.kind();
                pp_error!(self, "unsupported buffer: {kind}");
                Err(D3d11ScalerError::UnsupportedBuffer(kind).into())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        // Nothing local to react to — a pure per-frame spatial transform,
        // same reasoning as `CudaScaler::control`.
        self.pad.control(msg)
    }
}

/// One output surface for a single frame, in the same format the input
/// carried. `D3D11_BIND_RENDER_TARGET` is what
/// `CreateVideoProcessorOutputView` requires; `D3D11_BIND_SHADER_RESOURCE`
/// is what a downstream [`crate::elements::D3d11Renderer`] or
/// [`crate::elements::D3d11VideoCompositor`] needs to sample it.
fn create_output_texture(
    device: &ID3D11Device,
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
) -> std::result::Result<ID3D11Texture2D, windows::core::Error> {
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
        BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture: Option<ID3D11Texture2D> = None;
    unsafe {
        device.CreateTexture2D(&desc, None, Some(&mut texture))?;
    }
    Ok(texture.expect("CreateTexture2D succeeded without producing a texture"))
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use windows::Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{
            D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION,
            D3D11_SUBRESOURCE_DATA, D3D11_USAGE_STAGING, D3D11CreateDevice,
        },
        Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
    };

    use super::*;
    use crate::elements::{D3d11Download, D3d11Upload};

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

    /// A D3D11 device and its immediate context, or `None` — after printing
    /// why — on a machine without one, the same way every other hardware
    /// test here skips.
    ///
    /// Enough for the tests that only exercise argument validation, which
    /// `D3d11Scaler::new` rejects before it looks at what the adapter can
    /// do. Anything that actually scales needs `try_video_device`.
    fn try_device() -> Option<(ID3D11Device, Arc<Mutex<ID3D11DeviceContext>>)> {
        let mut device = None;
        let mut context = None;
        let result = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                Default::default(),
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
        };
        if result.is_err() {
            eprintln!("skipping: D3D11CreateDevice failed on this machine: {result:?}");
            return None;
        }
        Some((
            device.expect("D3D11CreateDevice succeeded without producing a device"),
            Arc::new(Mutex::new(context.expect(
                "D3D11CreateDevice succeeded without producing a context",
            ))),
        ))
    }

    /// The same device, but only when it can actually run a video
    /// processor. `D3d11Scaler` scales through
    /// `ID3D11VideoDevice`/`ID3D11VideoContext`, and an adapter without one
    /// — a CI runner's basic render driver, for instance — still hands out a
    /// perfectly usable `ID3D11Device` whose cast to those interfaces fails
    /// with `E_NOINTERFACE`. That belongs in the skip check rather than in
    /// each test's `D3d11Scaler::new(..).expect(..)`.
    fn try_video_device() -> Option<(ID3D11Device, Arc<Mutex<ID3D11DeviceContext>>)> {
        let (device, context) = try_device()?;
        if let Err(error) = device.cast::<ID3D11VideoDevice>() {
            eprintln!("skipping: this device has no ID3D11VideoDevice: {error}");
            return None;
        }
        let supported = {
            let context = context.lock().unwrap();
            context.cast::<ID3D11VideoContext>()
        };
        if let Err(error) = supported {
            eprintln!("skipping: this context has no ID3D11VideoContext: {error}");
            return None;
        }
        Some((device, context))
    }

    /// A flat BGRA texture: after any interpolation every output pixel must
    /// still be that same color, so a scaled result can be checked exactly
    /// rather than "looks plausible". `slices` builds a texture array, whose
    /// last slice holds the color and the rest a contrasting one.
    fn bgra_texture(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        color: [u8; 4],
        slices: u32,
    ) -> ID3D11Texture2D {
        let other = [255u8, 255, 255, 255];
        let planes: Vec<Vec<u8>> = (0..slices)
            .map(|slice| {
                let pixel = if slice == slices - 1 { color } else { other };
                pixel.repeat((width * height) as usize)
            })
            .collect();
        let initial: Vec<D3D11_SUBRESOURCE_DATA> = planes
            .iter()
            .map(|plane| D3D11_SUBRESOURCE_DATA {
                pSysMem: plane.as_ptr().cast::<c_void>(),
                SysMemPitch: width * 4,
                SysMemSlicePitch: 0,
            })
            .collect();
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: slices,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
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
        unsafe {
            device
                .CreateTexture2D(&desc, Some(initial.as_ptr()), Some(&mut texture))
                .expect("CreateTexture2D(BGRA) failed");
        }
        texture.expect("CreateTexture2D succeeded without producing a texture")
    }

    /// Wraps a texture as the pooled `Pixel::D3D11` `MediaBuffer` an
    /// upstream element would push.
    fn frame(texture: ID3D11Texture2D, width: u32, height: u32, pts: i64) -> MediaBuffer {
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = wrap_d3d11_texture(texture, width, height);
        slot.set_pts(Some(pts));
        MediaBuffer::Video(Arc::new(slot))
    }

    /// One flat NV12 GPU frame, built through `D3d11Upload` so the input is
    /// a texture this crate really produces rather than a hand-made one.
    fn nv12_frame(
        device: &ID3D11Device,
        width: u32,
        height: u32,
        luma: u8,
        pts: i64,
    ) -> MediaBuffer {
        let mut upload = D3d11Upload::new("upload", device, width, height);
        let uploaded = capture(&mut upload);
        let mut cpu = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
        cpu.set_pts(Some(pts));
        cpu.data_mut(0).fill(luma);
        cpu.data_mut(1).fill(128);
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut slot = pool.get();
        *slot = cpu;
        upload
            .consume(MediaBuffer::Video(Arc::new(slot)))
            .expect("upload");
        uploaded.lock().unwrap().remove(0)
    }

    /// Reads an NV12 texture back through a staging copy — `D3d11Download`
    /// only reads BGRA, and the point of the NV12 test is that the surface
    /// never became BGRA.
    fn read_nv12(
        device: &ID3D11Device,
        context: &Arc<Mutex<ID3D11DeviceContext>>,
        texture: &ID3D11Texture2D,
        width: u32,
        height: u32,
    ) -> (Vec<u8>, Vec<u8>) {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_NV12,
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
        unsafe {
            device
                .CreateTexture2D(&desc, None, Some(&mut staging))
                .expect("CreateTexture2D(NV12 staging) failed");
        }
        let staging = staging.expect("CreateTexture2D succeeded without producing a texture");

        let context = context.lock().unwrap();
        let (mut luma, mut chroma) = (Vec::new(), Vec::new());
        unsafe {
            context.CopyResource(&staging, texture);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .expect("Map(NV12 staging) failed");
            let stride = mapped.RowPitch as usize;
            let base = mapped.pData as *const u8;
            for row in 0..height as usize {
                let row_bytes = std::slice::from_raw_parts(base.add(row * stride), width as usize);
                luma.extend_from_slice(row_bytes);
            }
            // NV12 keeps the half-height interleaved chroma rows directly
            // after the full-height luma ones, sharing one row pitch.
            for row in 0..(height as usize / 2) {
                let row_bytes = std::slice::from_raw_parts(
                    base.add((height as usize + row) * stride),
                    width as usize,
                );
                chroma.extend_from_slice(row_bytes);
            }
            context.Unmap(&staging, 0);
        }
        (luma, chroma)
    }

    /// The contract: the frame comes out at the configured size, still on
    /// the GPU, still carrying its timestamp — and a resize alone does not
    /// touch the pixels. `D3d11Download` on the tail is what makes the last
    /// part checkable at all.
    #[test]
    fn a_bgra_frame_is_resized_on_the_gpu_and_keeps_its_pts() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let color = [20u8, 40, 60, 255];
        let mut source = frame(bgra_texture(&device, 16, 16, color, 1), 16, 16, 777);
        let MediaBuffer::Video(source_frame) = &mut source else {
            unreachable!("frame always returns a Video buffer");
        };
        let source_frame = Arc::get_mut(source_frame).expect("the frame is not shared yet");
        source_frame.set_color_space(ffmpeg::color::Space::BT709);
        source_frame.set_color_range(ffmpeg::color::Range::MPEG);

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context.clone(),
            D3d11ScalerFormat::Preserve,
            8,
            8,
        )
        .expect("D3d11Scaler::new should succeed");
        let mut download = D3d11Download::new("download", &device, context, 8, 8)
            .expect("D3d11Download::new should succeed");
        let received = capture(&mut download);
        scaler.src_pads()[0].link(Box::new(download));

        scaler.consume(source).expect("scale");
        scaler.consume(MediaBuffer::Eos).expect("eos");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(scaled) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(scaled.width(), 8);
        assert_eq!(scaled.height(), 8);
        assert_eq!(scaled.pts(), Some(777), "the scaler dropped the pts");
        assert_eq!(scaled.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(scaled.color_range(), ffmpeg::color::Range::MPEG);
        let stride = scaled.stride(0);
        for row in 0..8usize {
            for column in 0..8usize {
                let pixel =
                    &scaled.data(0)[row * stride + column * 4..row * stride + column * 4 + 4];
                assert_eq!(
                    pixel, color,
                    "a flat frame must stay flat through the resize (row {row}, column {column})"
                );
            }
        }
        assert!(
            received.last().is_some_and(MediaBuffer::is_eos),
            "Eos was not forwarded"
        );
    }

    /// The decode/encode path's format. The video processor could convert
    /// NV12 to BGRA, and must not: nothing downstream asked for that, and a
    /// silent conversion would cost a color round trip per frame.
    #[test]
    fn an_nv12_frame_is_resized_and_stays_nv12() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let source = nv12_frame(&device, 32, 32, 200, 5);

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context.clone(),
            D3d11ScalerFormat::Preserve,
            16,
            16,
        )
        .expect("D3d11Scaler::new should succeed");
        let received = capture(&mut scaler);
        scaler.consume(source).expect("scale");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(scaled) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(scaled.width(), 16);
        assert_eq!(scaled.height(), 16);
        assert_eq!(scaled.pts(), Some(5));

        let (texture_raw, _) = d3d11va_texture(scaled).expect("the output carries a texture");
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("the output texture must not be null")
                .clone()
        };
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut desc) };
        assert_eq!(
            desc.Format, DXGI_FORMAT_NV12,
            "the scaler changed the surface format"
        );

        let (luma, chroma) = read_nv12(&device, &context, &texture, 16, 16);
        assert!(
            luma.iter().all(|&sample| sample == 200),
            "a flat luma plane must stay flat through the resize"
        );
        assert!(
            chroma.iter().all(|&sample| sample == 128),
            "a flat chroma plane must stay flat through the resize"
        );
    }

    /// A decoder hands out slices of one array texture, so the scaler has
    /// to resize the slice the frame actually names — not slice zero.
    #[test]
    fn scales_only_the_selected_texture_array_slice() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let color = [20u8, 40, 60, 255];
        let mut source = frame(bgra_texture(&device, 16, 16, color, 2), 16, 16, 0);
        let MediaBuffer::Video(video) = &mut source else {
            panic!("expected a Video buffer");
        };
        unsafe {
            // Same encoding `wrap_d3d11_texture` documents: the array
            // slice index goes directly in `data[1]`, so index 1 is a
            // pointer whose address is 1.
            (*Arc::get_mut(video)
                .expect("the frame is not shared yet")
                .as_mut_ptr())
            .data[1] = std::ptr::dangling_mut::<u8>();
        }

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context.clone(),
            D3d11ScalerFormat::Preserve,
            8,
            8,
        )
        .expect("D3d11Scaler::new should succeed");
        let mut download = D3d11Download::new("download", &device, context, 8, 8)
            .expect("D3d11Download::new should succeed");
        let received = capture(&mut download);
        scaler.src_pads()[0].link(Box::new(download));
        scaler.consume(source).expect("scale");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(scaled) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(
            &scaled.data(0)[..4],
            color,
            "the scaler resized the wrong array slice"
        );
    }

    /// A mid-stream input change must rebuild the processor rather than
    /// fail or keep scaling from a stale one — same contract `CudaScaler`
    /// and `SwScaler` have for their own scaling state.
    #[test]
    fn rebuilds_its_processor_when_the_input_changes_mid_stream() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let first = frame(
            bgra_texture(&device, 16, 16, [10, 20, 30, 255], 1),
            16,
            16,
            0,
        );
        let second = nv12_frame(&device, 32, 16, 180, 1);

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context,
            D3d11ScalerFormat::Preserve,
            8,
            8,
        )
        .expect("D3d11Scaler::new should succeed");
        let received = capture(&mut scaler);
        scaler.consume(first).expect("the first frame must scale");
        scaler
            .consume(second)
            .expect("a differently shaped second frame must still scale");

        let received = received.lock().unwrap();
        assert_eq!(received.len(), 2);
        for buf in received.iter() {
            let MediaBuffer::Video(scaled) = buf else {
                panic!("expected a Video buffer, got {}", buf.kind());
            };
            assert_eq!(scaled.width(), 8);
            assert_eq!(scaled.height(), 8);
        }
    }

    /// A CPU frame has no texture at all, and one from another device has a
    /// texture this element's context cannot read — both must be refused
    /// before the video processor ever sees them.
    #[test]
    fn a_cpu_frame_and_a_foreign_device_frame_are_typed_errors() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let Some((other_device, _other_context)) = try_video_device() else {
            return;
        };

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context,
            D3d11ScalerFormat::Preserve,
            8,
            8,
        )
        .expect("D3d11Scaler::new should succeed");
        let _received = capture(&mut scaler);

        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        let mut cpu = pool.get();
        *cpu = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 16, 16);
        let error = scaler
            .consume(MediaBuffer::Video(Arc::new(cpu)))
            .expect_err("a CPU frame must not be scaled");
        assert!(
            error
                .to_string()
                .contains("only scales Pixel::D3D11 frames"),
            "expected UnsupportedFormat, got {error}"
        );

        let foreign = frame(
            bgra_texture(&other_device, 16, 16, [10, 20, 30, 255], 1),
            16,
            16,
            0,
        );
        let error = scaler
            .consume(foreign)
            .expect_err("a frame from a foreign device must not be scaled");
        assert!(
            error.to_string().contains("different ID3D11Device"),
            "expected DeviceMismatch, got {error}"
        );
    }

    /// An odd NV12 output has no representable chroma plane. The caller
    /// picked that size, so the error has to name it rather than surface as
    /// an `E_INVALIDARG` from somewhere inside the D3D11 call sequence.
    #[test]
    fn an_odd_nv12_output_size_is_a_typed_error() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let source = nv12_frame(&device, 32, 32, 200, 0);

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context,
            D3d11ScalerFormat::Preserve,
            15,
            15,
        )
        .expect("D3d11Scaler::new should succeed");
        let _received = capture(&mut scaler);
        let error = scaler
            .consume(source)
            .expect_err("an odd NV12 output must not be scaled");
        assert!(
            error.to_string().contains("cannot be 15x15"),
            "expected OddNv12Output, got {error}"
        );
    }

    /// Both of these reject their arguments before `D3d11Scaler::new` casts
    /// to the video interfaces, so they take a plain `try_device` and hold
    /// on an adapter with no video processor too — which is exactly where a
    /// check placed after the cast would report `E_NOINTERFACE` instead.
    #[test]
    fn rejects_zero_output_dimensions_before_allocating_a_texture() {
        let Some((device, context)) = try_device() else {
            return;
        };
        assert!(matches!(
            D3d11Scaler::new(
                "scaler",
                &device,
                context,
                D3d11ScalerFormat::Preserve,
                0,
                8
            ),
            Err(D3d11ScalerError::InvalidOutputDimensions {
                width: 0,
                height: 8
            })
        ));
    }

    #[test]
    fn rejects_a_context_from_another_device_at_construction() {
        let Some((device, _context)) = try_device() else {
            return;
        };
        let Some((_other_device, other_context)) = try_device() else {
            return;
        };
        assert!(matches!(
            D3d11Scaler::new(
                "scaler",
                &device,
                other_context,
                D3d11ScalerFormat::Preserve,
                8,
                8
            ),
            Err(D3d11ScalerError::ContextDeviceMismatch)
        ));
    }

    /// How far a converted channel may land from the arithmetic answer.
    /// The video processor is fixed-function hardware and each vendor
    /// rounds its own way, so the contract worth asserting is "this is the
    /// color that came out", not a bit-exact match.
    const CHANNEL_TOLERANCE: i32 = 4;

    fn assert_close(actual: u8, expected: u8, what: &str) {
        let difference = i32::from(actual) - i32::from(expected);
        assert!(
            difference.abs() <= CHANNEL_TOLERANCE,
            "{what}: expected about {expected}, got {actual}"
        );
    }

    /// The conversion that connects a decoder to everything that
    /// composites. A flat limited-range luma of 235 with neutral chroma is
    /// white, so the BGRA that comes out has to be white in all three
    /// channels — a suppressed conversion would hand back the raw luma
    /// instead, and a wrong matrix would tint it.
    #[test]
    fn an_nv12_frame_converts_to_bgra_on_the_way_through() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        // 235 / neutral chroma: limited-range white.
        let source = nv12_frame(&device, 32, 32, 235, 9);

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context.clone(),
            D3d11ScalerFormat::Bgra,
            16,
            16,
        )
        .expect("D3d11Scaler::new should succeed");
        let mut download = D3d11Download::new("download", &device, context, 16, 16)
            .expect("D3d11Download::new should succeed");
        let received = capture(&mut download);
        scaler.src_pads()[0].link(Box::new(download));

        scaler.consume(source).expect("convert");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(converted) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        assert_eq!(converted.width(), 16);
        assert_eq!(converted.height(), 16);
        assert_eq!(converted.pts(), Some(9), "the conversion dropped the pts");

        let stride = converted.stride(0);
        for row in 0..16usize {
            for column in 0..16usize {
                let offset = row * stride + column * 4;
                let pixel = &converted.data(0)[offset..offset + 4];
                for (channel, name) in [(0, "blue"), (1, "green"), (2, "red")] {
                    assert_close(
                        pixel[channel],
                        255,
                        &format!("row {row}, column {column}, {name}"),
                    );
                }
            }
        }
    }

    /// The other direction, for a capture or compositor output headed into
    /// something that wants NV12. Full-range white in, limited-range luma
    /// 235 out.
    #[test]
    fn a_bgra_frame_converts_to_nv12_on_the_way_through() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let white = [255u8, 255, 255, 255];
        let source = frame(bgra_texture(&device, 16, 16, white, 1), 16, 16, 3);

        let mut scaler = D3d11Scaler::new(
            "scaler",
            &device,
            context.clone(),
            D3d11ScalerFormat::Nv12,
            8,
            8,
        )
        .expect("D3d11Scaler::new should succeed");
        let received = capture(&mut scaler);
        scaler.consume(source).expect("convert");

        let received = received.lock().unwrap();
        let MediaBuffer::Video(converted) = &received[0] else {
            panic!("expected a Video buffer, got {}", received[0].kind());
        };
        let (texture_raw, _) = d3d11va_texture(converted).expect("the output carries a texture");
        let texture = unsafe {
            ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .expect("the output texture must not be null")
                .clone()
        };
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut desc) };
        assert_eq!(
            desc.Format, DXGI_FORMAT_NV12,
            "asking for Nv12 must produce an NV12 surface"
        );

        let (luma, chroma) = read_nv12(&device, &context, &texture, 8, 8);
        for (index, sample) in luma.iter().enumerate() {
            assert_close(*sample, 235, &format!("luma sample {index}"));
        }
        for (index, sample) in chroma.iter().enumerate() {
            assert_close(*sample, 128, &format!("chroma sample {index}"));
        }
    }

    /// A conversion changes what the pixels are, so the tags that describe
    /// them have to change with them — an encoder builds its stream's
    /// description from exactly these.
    #[test]
    fn a_converted_frame_is_retagged_and_a_resized_one_is_not() {
        let Some((device, context)) = try_video_device() else {
            return;
        };

        // Converted: the source's BT.709/limited tags describe Y'CbCr, and
        // the output is RGB, so they are replaced rather than carried.
        let mut source = nv12_frame(&device, 32, 32, 128, 0);
        let MediaBuffer::Video(source_frame) = &mut source else {
            unreachable!("nv12_frame always returns a Video buffer");
        };
        let source_frame = Arc::get_mut(source_frame).expect("the frame is not shared yet");
        source_frame.set_color_space(ffmpeg::color::Space::BT709);
        source_frame.set_color_range(ffmpeg::color::Range::MPEG);

        let mut converting = D3d11Scaler::new(
            "converting",
            &device,
            context.clone(),
            D3d11ScalerFormat::Bgra,
            16,
            16,
        )
        .expect("D3d11Scaler::new should succeed");
        let converted = capture(&mut converting);
        converting.consume(source).expect("convert");
        {
            let converted = converted.lock().unwrap();
            let MediaBuffer::Video(frame) = &converted[0] else {
                panic!("expected a Video buffer");
            };
            assert_eq!(frame.color_space(), ffmpeg::color::Space::RGB);
            assert_eq!(frame.color_range(), ffmpeg::color::Range::JPEG);
        }

        // Resized only: the same tags cross untouched, because the pixels
        // still are what they said they were.
        let mut source = nv12_frame(&device, 32, 32, 128, 0);
        let MediaBuffer::Video(source_frame) = &mut source else {
            unreachable!("nv12_frame always returns a Video buffer");
        };
        let source_frame = Arc::get_mut(source_frame).expect("the frame is not shared yet");
        source_frame.set_color_space(ffmpeg::color::Space::BT709);
        source_frame.set_color_range(ffmpeg::color::Range::MPEG);

        let mut resizing = D3d11Scaler::new(
            "resizing",
            &device,
            context,
            D3d11ScalerFormat::Preserve,
            16,
            16,
        )
        .expect("D3d11Scaler::new should succeed");
        let resized = capture(&mut resizing);
        resizing.consume(source).expect("resize");
        let resized = resized.lock().unwrap();
        let MediaBuffer::Video(frame) = &resized[0] else {
            panic!("expected a Video buffer");
        };
        assert_eq!(frame.color_space(), ffmpeg::color::Space::BT709);
        assert_eq!(frame.color_range(), ffmpeg::color::Range::MPEG);
    }

    /// The odd-size guard follows the output format, not the input's: a
    /// BGRA input can be any size, but the NV12 it is being converted into
    /// still has no half-sample to write.
    #[test]
    fn an_odd_size_is_rejected_when_the_output_is_nv12_even_from_a_bgra_input() {
        let Some((device, context)) = try_video_device() else {
            return;
        };
        let source = frame(bgra_texture(&device, 16, 16, [0, 0, 0, 255], 1), 16, 16, 0);

        let mut scaler =
            D3d11Scaler::new("scaler", &device, context, D3d11ScalerFormat::Nv12, 15, 15)
                .expect("D3d11Scaler::new should succeed");
        let _received = capture(&mut scaler);
        let error = scaler
            .consume(source)
            .expect_err("an odd NV12 output must be rejected");
        assert!(
            matches!(
                error,
                crate::error::Error::D3d11ScalerError(D3d11ScalerError::OddNv12Output {
                    width: 15,
                    height: 15
                })
            ),
            "unexpected error: {error}"
        );
    }
}
