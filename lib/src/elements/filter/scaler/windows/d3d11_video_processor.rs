//! The D3D11 `ID3D11VideoProcessor` that [`super::D3d11Scaler`] resizes with,
//! kept separate from the element the same way `CudaScaler` keeps its
//! `scale_cuda` graph in `scale_graph.rs`: one object configured for one
//! input shape, rebuilt when that shape changes.

use std::mem::ManuallyDrop;

use ffmpeg_next as ffmpeg;
use windows::{
    Win32::{
        Foundation::RECT,
        Graphics::{
            Direct3D11::{
                D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
                D3D11_VIDEO_PROCESSOR_CONTENT_DESC, D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT,
                D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC,
                D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC,
                D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_STREAM,
                D3D11_VIDEO_USAGE_PLAYBACK_NORMAL, D3D11_VPIV_DIMENSION_TEXTURE2D,
                D3D11_VPOV_DIMENSION_TEXTURE2D, ID3D11Texture2D, ID3D11VideoContext,
                ID3D11VideoContext1, ID3D11VideoDevice, ID3D11VideoProcessor,
                ID3D11VideoProcessorEnumerator,
            },
            Dxgi::Common::{
                DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P709,
                DXGI_COLOR_SPACE_TYPE, DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601,
                DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709,
                DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020,
                DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601,
                DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
                DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020, DXGI_FORMAT, DXGI_FORMAT_NV12,
                DXGI_FORMAT_P010, DXGI_RATIONAL,
            },
        },
    },
    core::Interface,
};

use super::d3d11_scaler::D3d11ScalerError;

/// What a processor is configured for. A frame that disagrees with any of
/// it needs its own processor — the input size and surface format are baked
/// into the enumerator's content description, not per-`Blt` parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct InputShape {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: DXGI_FORMAT,
}

/// One `ID3D11VideoProcessor` plus the enumerator it came from, configured
/// for exactly one [`InputShape`] and output size.
///
/// Everything that stays constant for that pair — progressive frame format,
/// source/destination rectangles, and the driver's own "auto processing"
/// (denoise, edge enhancement, and friends, which would alter pixels this
/// element only promised to resize) — is set once here. Only the input and
/// output views, which name the actual textures, are per frame.
pub(super) struct ScaleProcessor {
    enumerator: ID3D11VideoProcessorEnumerator,
    processor: ID3D11VideoProcessor,
    input: InputShape,
    output_width: u32,
    output_height: u32,
    output_format: DXGI_FORMAT,
    /// Last colorimetry handed to the processor, input side and output
    /// side. Kept so an unchanged stream does not re-issue the setters per
    /// frame, and so a frame that *does* retag mid-stream is followed
    /// without rebuilding anything — unlike size or format, colorimetry is
    /// processor state, not part of the enumerator's content description.
    ///
    color_space: Option<BltColorSpaces>,
}

/// What each side of one `Blt` is tagged with.
///
/// The two are equal for a pure resize and deliberately differ for a
/// conversion: telling the processor that a Y'CbCr input and an RGB output
/// share colorimetry is exactly what would suppress the conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct BltColorSpaces {
    pub(super) input: DXGI_COLOR_SPACE_TYPE,
    pub(super) output: DXGI_COLOR_SPACE_TYPE,
}

impl BltColorSpaces {
    /// `input` to `output`, told to the processor in a way it can carry out.
    ///
    /// A processor reads BT.2020 Y'CbCr but need not write it: the RTX
    /// 3050's reports BT.2020 to BT.2020 unsupported through
    /// `CheckVideoProcessorFormatConversion`, and asked anyway wrote
    /// (0, 89, 0) for red. Where the two sides are the same there is no
    /// colour to change — a resize, or P010 down to NV12 — so both are
    /// described as BT.709 instead, which every processor writes; what
    /// matters is only that they are equal.
    pub(super) fn new(input: DXGI_COLOR_SPACE_TYPE, output: DXGI_COLOR_SPACE_TYPE) -> Self {
        if input != output {
            return Self { input, output };
        }
        let same = match input {
            DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020 => {
                DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709
            }
            DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020 => DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709,
            other => other,
        };
        Self {
            input: same,
            output: same,
        }
    }
}

impl ScaleProcessor {
    /// `video_context` is the shared immediate context this element's
    /// caller already holds the lock for — every call below is a
    /// context-level one (see [`super::D3d11Scaler`]'s own docs).
    pub(super) fn new(
        video_device: &ID3D11VideoDevice,
        video_context: &ID3D11VideoContext,
        input: InputShape,
        output_width: u32,
        output_height: u32,
        output_format: DXGI_FORMAT,
    ) -> Result<Self, D3d11ScalerError> {
        // Chroma is half-resolution in both directions, so an odd NV12
        // output has no representable chroma plane — `CreateTexture2D`
        // would fail with a bare `E_INVALIDARG` later, well away from the
        // size the caller actually chose. This follows the *output*
        // format: converting an odd BGRA input into NV12 is just as
        // impossible as resizing NV12 to an odd size.
        if is_yuv420(output_format)
            && (!output_width.is_multiple_of(2) || !output_height.is_multiple_of(2))
        {
            return Err(D3d11ScalerError::OddNv12Output {
                width: output_width,
                height: output_height,
            });
        }

        let desc = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
            InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            // No rate conversion happens here — one `Blt` per input frame,
            // whatever the stream's real rate is — so these two only have
            // to be non-zero and equal.
            InputFrameRate: DXGI_RATIONAL {
                Numerator: 1,
                Denominator: 1,
            },
            InputWidth: input.width,
            InputHeight: input.height,
            OutputFrameRate: DXGI_RATIONAL {
                Numerator: 1,
                Denominator: 1,
            },
            OutputWidth: output_width,
            OutputHeight: output_height,
            Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
        };
        // SAFETY: `desc` is fully initialized and the live video device owns
        // the returned enumerator interface.
        let enumerator = unsafe { video_device.CreateVideoProcessorEnumerator(&desc) }?;

        // Each side is asked for only the direction it is actually used in:
        // the same format is both input and output for a pure resize, but a
        // conversion may well be supported one way and not the other.
        // Asking first turns "this GPU's video processor does not do this
        // format" into that sentence, rather than an `E_INVALIDARG` from
        // view creation.
        for (format, needed) in [
            (input.format, D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT),
            (output_format, D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT),
        ] {
            // SAFETY: `enumerator` is live and `format` is a DXGI enum value;
            // the method returns the support mask by value.
            let support = unsafe { enumerator.CheckVideoProcessorFormat(format) }?;
            if support & needed.0 as u32 == 0 {
                return Err(D3d11ScalerError::UnsupportedByVideoProcessor(format));
            }
        }

        // Index 0 is the plain rate-conversion capability every driver
        // exposes; the others exist for frame-rate conversion, which this
        // element does not do.
        // SAFETY: `enumerator` is live and capability index 0 is guaranteed by
        // the enumerator created for this content description.
        let processor = unsafe { video_device.CreateVideoProcessor(&enumerator, 0) }?;

        let source = RECT {
            left: 0,
            top: 0,
            right: input.width as i32,
            bottom: input.height as i32,
        };
        let destination = RECT {
            left: 0,
            top: 0,
            right: output_width as i32,
            bottom: output_height as i32,
        };
        // SAFETY: processor/context/enumerator are live and belong to the same
        // device; stream 0 and both rectangles are within the declared input
        // and output dimensions.
        unsafe {
            video_context.VideoProcessorSetStreamFrameFormat(
                &processor,
                0,
                D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            );
            // Without this the driver is free to denoise, sharpen, or
            // auto-correct contrast on the way through — a resize that
            // quietly edits pixels, and one whose result differs per
            // vendor.
            video_context.VideoProcessorSetStreamAutoProcessingMode(&processor, 0, false);
            // The source rectangle is the frame's *visible* size, which is
            // not the texture's: a decoder's surfaces are padded up to its
            // own alignment, and scaling that padding in would drag it
            // into the picture.
            video_context.VideoProcessorSetStreamSourceRect(&processor, 0, true, Some(&source));
            video_context.VideoProcessorSetStreamDestRect(&processor, 0, true, Some(&destination));
            video_context.VideoProcessorSetOutputTargetRect(&processor, true, Some(&destination));
        }

        Ok(Self {
            enumerator,
            processor,
            input,
            output_width,
            output_height,
            output_format,
            color_space: None,
        })
    }

    /// Whether this processor is the one a frame of `input` needs to reach
    /// `output_width`x`output_height` in `output_format`.
    pub(super) fn matches(
        &self,
        input: InputShape,
        output_width: u32,
        output_height: u32,
        output_format: DXGI_FORMAT,
    ) -> bool {
        self.input == input
            && self.output_width == output_width
            && self.output_height == output_height
            && self.output_format == output_format
    }

    /// Scales one texture-array slice of `source` into `output`.
    ///
    /// `video_context` must be the shared immediate context, with its lock
    /// held for the whole call. The two color spaces are what each side is
    /// tagged with, and their relationship is the whole conversion: equal
    /// values tell the processor nothing about color changes, which is what
    /// keeps a pure resize from turning into one, while a Y'CbCr input
    /// paired with an RGB output is what asks for the conversion.
    pub(super) fn scale(
        &mut self,
        video_device: &ID3D11VideoDevice,
        video_context: &ID3D11VideoContext,
        source: &ID3D11Texture2D,
        array_slice: u32,
        output: &ID3D11Texture2D,
        color_space: BltColorSpaces,
    ) -> Result<(), D3d11ScalerError> {
        if self.color_space != Some(color_space) {
            // Windows 8.1 and later; see `color_space` for why the older
            // setters will not do.
            let video_context: ID3D11VideoContext1 = video_context.cast()?;
            // SAFETY: the processor and video context are live, stream 0 was
            // configured at construction, and both values are DXGI enum
            // values.
            unsafe {
                video_context.VideoProcessorSetStreamColorSpace1(
                    &self.processor,
                    0,
                    color_space.input,
                );
                video_context
                    .VideoProcessorSetOutputColorSpace1(&self.processor, color_space.output);
            }
            self.color_space = Some(color_space);
        }

        let input_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            // Zero means "the texture's own format" — the only correct
            // answer for a DXGI-typed surface, since a FourCC would
            // reinterpret it.
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPIV {
                    MipSlice: 0,
                    ArraySlice: array_slice,
                },
            },
        };
        let mut input_view = None;
        // SAFETY: `source` is a validated texture, `array_slice` is bounded by
        // its description, and `input_view` is a live out-parameter tied to
        // this processor's enumerator.
        unsafe {
            video_device.CreateVideoProcessorInputView(
                source,
                &self.enumerator,
                &input_desc,
                Some(&mut input_view),
            )
        }?;
        let input_view =
            input_view.expect("CreateVideoProcessorInputView succeeded without producing a view");

        let output_desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
            ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 },
            },
        };
        let mut output_view = None;
        // SAFETY: `output` was allocated for this processor/device and
        // `output_view` is a live out-parameter using the matching enumerator.
        unsafe {
            video_device.CreateVideoProcessorOutputView(
                output,
                &self.enumerator,
                &output_desc,
                Some(&mut output_view),
            )
        }?;
        let output_view =
            output_view.expect("CreateVideoProcessorOutputView succeeded without producing a view");

        let mut streams = [D3D11_VIDEO_PROCESSOR_STREAM {
            Enable: true.into(),
            // `pInputSurface` is a `ManuallyDrop`, so the view moved in
            // here is *not* released when `streams` goes out of scope —
            // the explicit drop after the `Blt` below is what releases it.
            pInputSurface: ManuallyDrop::new(Some(input_view)),
            ..Default::default()
        }];
        // SAFETY: input/output views and processor all remain live for the
        // synchronous submission; `streams` contains exactly the configured
        // stream and its manually-held COM reference.
        let result =
            unsafe { video_context.VideoProcessorBlt(&self.processor, &output_view, 0, &streams) };
        // SAFETY: the `ManuallyDrop` owns exactly the reference placed into it
        // above and has not otherwise been dropped.
        unsafe { ManuallyDrop::drop(&mut streams[0].pInputSurface) };
        result?;
        Ok(())
    }
}

/// The colorimetry to hand the processor for a frame tagged this way, in
/// the surface format it is held in.
///
/// Called once per side of a `Blt` (see [`ScaleProcessor::scale`]). For a
/// pure resize both sides pass the same arguments and so get the same
/// answer, which is what tells the processor there is no color change to
/// make; a conversion describes each side for what it actually holds.
///
/// A `DXGI_COLOR_SPACE_TYPE`, set through `ID3D11VideoContext1`, rather
/// than the older `D3D11_VIDEO_PROCESSOR_COLOR_SPACE`, whose one matrix bit
/// says 601 or 709 and nothing else: a BT.2020 frame described that way was
/// converted as BT.601, which measured an encoded (200, 40, 40) back as
/// (190, 32, 40). Unspecified metadata follows the same fallback
/// `D3d11VideoCompositor` uses — BT.601 through 576 lines and BT.709 above
/// it, limited range — so the two elements describe one frame the same way.
///
/// Every answer here is gamma 2.2 and BT.709 primaries apart from BT.2020's
/// own, which is what each matrix is normally paired with; a frame's
/// primaries and transfer tags are not read, and nothing here maps HDR.
pub(super) fn color_space(
    space: ffmpeg::color::Space,
    range: ffmpeg::color::Range,
    height: u32,
    format: DXGI_FORMAT,
) -> DXGI_COLOR_SPACE_TYPE {
    use ffmpeg::color::Space;

    let full_range = range == ffmpeg::color::Range::JPEG;
    if !is_yuv420(format) {
        // Unspecified RGB is full range, as everything that makes it here
        // does.
        return if full_range || range == ffmpeg::color::Range::Unspecified {
            DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709
        } else {
            DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P709
        };
    }
    enum Matrix {
        Bt601,
        Bt709,
        Bt2020,
    }
    let matrix = match space {
        Space::BT709 => Matrix::Bt709,
        Space::BT2020NCL | Space::BT2020CL => Matrix::Bt2020,
        Space::Unspecified if height > 576 => Matrix::Bt709,
        _ => Matrix::Bt601,
    };
    match (matrix, full_range) {
        (Matrix::Bt601, false) => DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601,
        (Matrix::Bt601, true) => DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601,
        (Matrix::Bt709, false) => DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
        (Matrix::Bt709, true) => DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709,
        (Matrix::Bt2020, false) => DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020,
        (Matrix::Bt2020, true) => DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020,
    }
}

/// Whether `format` is 4:2:0 Y'CbCr — NV12, or P010, the same layout at ten
/// bits a sample that a 10-bit stream decodes to. Half-resolution chroma,
/// so neither can have an odd side, and both are described by a matrix and
/// a range rather than as RGB.
pub(super) fn is_yuv420(format: DXGI_FORMAT) -> bool {
    format == DXGI_FORMAT_NV12 || format == DXGI_FORMAT_P010
}
