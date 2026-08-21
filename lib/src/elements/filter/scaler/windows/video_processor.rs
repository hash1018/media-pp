//! The `ID3D11VideoProcessor` that [`super::D3d11Scaler`] resizes with,
//! kept separate from the element the same way `CudaScaler` keeps its
//! `scale_cuda` graph in `scale_graph.rs`: one object configured for one
//! input shape, rebuilt when that shape changes.

use std::mem::ManuallyDrop;

use ffmpeg_next as ffmpeg;
use windows::Win32::{
    Foundation::RECT,
    Graphics::{
        Direct3D11::{
            D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            D3D11_VIDEO_PROCESSOR_COLOR_SPACE, D3D11_VIDEO_PROCESSOR_CONTENT_DESC,
            D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT,
            D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC,
            D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC,
            D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_STREAM,
            D3D11_VIDEO_USAGE_PLAYBACK_NORMAL, D3D11_VPIV_DIMENSION_TEXTURE2D,
            D3D11_VPOV_DIMENSION_TEXTURE2D, ID3D11Texture2D, ID3D11VideoContext, ID3D11VideoDevice,
            ID3D11VideoProcessor, ID3D11VideoProcessorEnumerator,
        },
        Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_NV12, DXGI_RATIONAL},
    },
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
    pub(super) input: D3D11_VIDEO_PROCESSOR_COLOR_SPACE,
    pub(super) output: D3D11_VIDEO_PROCESSOR_COLOR_SPACE,
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
        if output_format == DXGI_FORMAT_NV12
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
            let support = unsafe { enumerator.CheckVideoProcessorFormat(format) }?;
            if support & needed.0 as u32 == 0 {
                return Err(D3d11ScalerError::UnsupportedByVideoProcessor(format));
            }
        }

        // Index 0 is the plain rate-conversion capability every driver
        // exposes; the others exist for frame-rate conversion, which this
        // element does not do.
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
            unsafe {
                video_context.VideoProcessorSetStreamColorSpace(
                    &self.processor,
                    0,
                    &color_space.input,
                );
                video_context
                    .VideoProcessorSetOutputColorSpace(&self.processor, &color_space.output);
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
        let result =
            unsafe { video_context.VideoProcessorBlt(&self.processor, &output_view, 0, &streams) };
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
/// D3D11's description of color here is coarse — the bitfield below has one
/// bit for the YCbCr matrix (601 or 709) and no way to say anything else —
/// so what matters is that a frame's own tags map to one consistent answer
/// rather than that they round-trip. Unspecified metadata follows the same
/// fallback `D3d11VideoCompositor` uses — BT.601 through 576 lines and
/// BT.709 above it, limited range — so the two elements describe one frame
/// the same way.
pub(super) fn color_space(
    space: ffmpeg::color::Space,
    range: ffmpeg::color::Range,
    height: u32,
    format: DXGI_FORMAT,
) -> D3D11_VIDEO_PROCESSOR_COLOR_SPACE {
    // `D3D11_VIDEO_PROCESSOR_COLOR_SPACE`'s bitfield, in declaration
    // order: Usage:1, RGB_Range:1, YCbCr_Matrix:1, YCbCr_xvYCC:1,
    // Nominal_Range:2. windows-rs exposes it as one `u32`.
    const USAGE_VIDEO_PROCESSING: u32 = 1;
    const RGB_RANGE_LIMITED: u32 = 1 << 1;
    const YCBCR_MATRIX_BT709: u32 = 1 << 2;
    const NOMINAL_RANGE_16_235: u32 = 1 << 4;
    const NOMINAL_RANGE_0_255: u32 = 2 << 4;

    let full_range = range == ffmpeg::color::Range::JPEG;
    // Usage 0 asks the driver for playback-tuned output; 1 asks it to
    // process the video faithfully, which is what an element in the middle
    // of a pipeline owes whatever comes next.
    let mut bits = USAGE_VIDEO_PROCESSING;
    if format == DXGI_FORMAT_NV12 {
        let bt709 = match space {
            ffmpeg::color::Space::BT709 => true,
            ffmpeg::color::Space::Unspecified => height > 576,
            _ => false,
        };
        if bt709 {
            bits |= YCBCR_MATRIX_BT709;
        }
        bits |= if full_range {
            NOMINAL_RANGE_0_255
        } else {
            NOMINAL_RANGE_16_235
        };
    } else if !full_range && range != ffmpeg::color::Range::Unspecified {
        bits |= RGB_RANGE_LIMITED;
    }
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE { _bitfield: bits }
}
