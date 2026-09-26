//! [`VideoEncodeBin`] — one video stream's frames encoded into H.264 by
//! whichever encoder takes them: the GPU's own where it has one that opens,
//! and in software otherwise. The encode side of
//! [`VideoDecodeBin`](crate::elements::VideoDecodeBin).

use std::sync::Arc;

use ffmpeg_next as ffmpeg;

#[cfg(feature = "cuda")]
use crate::elements::{
    CudaCodec, CudaDevice, CudaDownload, CudaEncoder, CudaEncoderOptions, CudaFrameFormat,
};
#[cfg(all(target_os = "windows", feature = "d3d11"))]
use crate::elements::{
    D3d11Download, D3d11Gpu, D3d11Scaler, D3d11ScalerFormat, D3d11VideoCodec, D3d11VideoEncoder,
    D3d11VideoEncoderOptions, D3d11VideoInputFormat,
};
#[cfg(feature = "vulkan")]
use crate::elements::{
    VulkanCodec, VulkanDevice, VulkanDownload, VulkanEncoder, VulkanEncoderOptions,
    VulkanFrameFormat,
};
use crate::{
    buffer::MediaBuffer,
    color::ColorDescription,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet,
        PortContract,
    },
    control::ControlMsg,
    element::{Context, Element, ElementType, Filter, Sink, Source, element_pp_log},
    elements::{
        SwEncoder, SwEncoderOptions, SwScaler, TrackFormat, VideoCodec, filter::line::Line,
    },
    error::Result,
    pad::SrcPad,
    pp_log::{PpLog, pp_info},
};

/// What the software path hands its encoder when the frames were RGB:
/// [`SwScaler`] turns RGB into YUV with swscale's own default, BT.601
/// limited range, and the RGB it came from is a screen's or a compositor's —
/// sRGB, whose primaries are BT.709's — so only the matrix is BT.601.
///
/// Said rather than left out: FFmpeg reads an untagged stream as BT.601 and
/// would get it right, but a player that takes an HD stream to be BT.709
/// would not.
const SWSCALE_FROM_RGB: ColorDescription = ColorDescription {
    space: ffmpeg::color::Space::BT470BG,
    range: ffmpeg::color::Range::MPEG,
    primaries: ffmpeg::color::Primaries::BT709,
    transfer: ffmpeg::color::TransferCharacteristic::BT709,
};

/// Why a [`VideoEncodeBin`] refused a frame.
#[derive(Debug, thiserror::Error)]
pub enum VideoEncodeBinError {
    /// A frame in system memory is not in the layout
    /// [`EncodeInput::System`] said every frame would be — which decides how
    /// its colour is converted and described, so it is refused rather than
    /// encoded under the wrong description.
    #[error("the bin was opened for {declared:?} frames, got {got:?}")]
    FormatMismatch {
        /// What `EncodeInput::System` said.
        declared: ffmpeg::format::Pixel,
        /// What the frame is in.
        got: ffmpeg::format::Pixel,
    },
}

/// Where the frames a [`VideoEncodeBin`] is given come from — which decides
/// the encoders it can choose between.
///
/// A device is a reference-counted handle, so cloning one is cheap; it must
/// be the one every other element on that device in the pipeline shares.
#[derive(Clone)]
pub enum EncodeInput {
    /// Frames in system memory, in `format` — whatever a software decode, a
    /// CPU capture or an application's own frames are in. Encoded in
    /// software: there is nothing on a device to hand a hardware encoder.
    System {
        /// The layout every frame arrives in. It decides how the colour is
        /// converted and what the stream says of it, so a frame in another
        /// is refused with [`VideoEncodeBinError::FormatMismatch`].
        format: ffmpeg::format::Pixel,
    },
    /// D3D11 textures on `gpu`, NV12 or BGRA.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    D3d11 {
        /// The device, and its one shared immediate context, every D3D11
        /// element in the pipeline shares.
        gpu: D3d11Gpu,
        /// What every texture holds.
        format: D3d11VideoInputFormat,
    },
    /// CUDA frames on `device`, NV12 or BGRA.
    #[cfg(feature = "cuda")]
    Cuda {
        /// The CUDA context every CUDA element in the pipeline shares.
        device: CudaDevice,
        /// What every surface holds.
        format: CudaFrameFormat,
    },
    /// Vulkan frames on `device`, NV12 or BGRA.
    #[cfg(feature = "vulkan")]
    Vulkan {
        /// The Vulkan device every Vulkan element in the pipeline shares.
        device: VulkanDevice,
        /// What every frame holds.
        format: VulkanFrameFormat,
    },
}

impl EncodeInput {
    /// Where a [`VideoDecodeBin`](crate::elements::VideoDecodeBin) opened
    /// onto `target` puts its frames, in `format` — its
    /// [`output_format`](crate::elements::VideoDecodeBin::output_format) —
    /// as an encode bin after it takes them: the same device and layout,
    /// nothing to convert between the two.
    ///
    /// `None` where no encode bin takes them: D3D12 frames, a device layout
    /// other than NV12 or BGRA, or system memory in a layout the decoder did
    /// not say.
    pub fn for_decoded(
        target: &crate::elements::DecodeTarget,
        format: Option<ffmpeg::format::Pixel>,
    ) -> Option<Self> {
        use crate::elements::DecodeTarget;
        #[cfg(any(
            feature = "cuda",
            feature = "vulkan",
            all(target_os = "windows", feature = "d3d11")
        ))]
        use ffmpeg::format::Pixel;

        match target {
            DecodeTarget::System => format.map(|format| Self::System { format }),
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            DecodeTarget::D3d11 { gpu, .. } => Some(Self::D3d11 {
                gpu: gpu.clone(),
                format: match format? {
                    Pixel::NV12 => D3d11VideoInputFormat::Nv12,
                    Pixel::BGRA => D3d11VideoInputFormat::Bgra,
                    _ => return None,
                },
            }),
            #[cfg(all(target_os = "windows", feature = "d3d12"))]
            DecodeTarget::D3d12 { .. } => None,
            #[cfg(feature = "cuda")]
            DecodeTarget::Cuda { device, .. } => Some(Self::Cuda {
                device: device.clone(),
                format: match format? {
                    Pixel::NV12 => CudaFrameFormat::Nv12,
                    Pixel::BGRA => CudaFrameFormat::Bgra,
                    _ => return None,
                },
            }),
            #[cfg(feature = "vulkan")]
            DecodeTarget::Vulkan { device, .. } => Some(Self::Vulkan {
                device: device.clone(),
                format: match format? {
                    Pixel::NV12 => VulkanFrameFormat::Nv12,
                    Pixel::BGRA => VulkanFrameFormat::Bgra,
                    _ => return None,
                },
            }),
        }
    }
}

/// What a [`VideoEncodeBin`] encodes to. H.264 always: the one codec every
/// encoder it may choose has, and the one a software encoder is always there
/// for.
#[derive(Debug, Clone, Copy)]
pub struct VideoEncodeOptions {
    /// Encoded frame width in pixels — the frames' own on a device, where
    /// nothing is scaled; in software, what they are scaled to.
    pub width: u32,
    /// Encoded frame height in pixels, as `width`.
    pub height: u32,
    /// The nominal rate the encoder works to and writes into the stream —
    /// see [`SwEncoderOptions::frame_rate`].
    pub frame_rate: ffmpeg::Rational,
    /// Target encoded bit rate, in bits per second.
    pub bit_rate: usize,
    /// Frames between keyframes — see [`SwEncoderOptions::gop_size`].
    pub gop_size: u32,
    /// How many consecutive B-frames the encoder may insert, or `None` for
    /// the encoder's own default — see [`SwEncoderOptions::max_b_frames`].
    pub max_b_frames: Option<u32>,
    /// What YUV frames hold — a decoded stream's own description, say —
    /// written into the stream so a player need not guess, or `None` to
    /// write nothing. Not asked for RGB frames: the bin converts those
    /// itself, or has the encoder do it, and says what came of it.
    pub color: Option<ColorDescription>,
}

/// Which encoder a [`VideoEncodeBin`] chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodePath {
    /// `h264_nvenc`, on NVIDIA's encoder.
    Nvenc,
    /// `h264_mf`, on whichever hardware transform Windows offers.
    MediaFoundation,
    /// `h264_vulkan`, on whichever GPU's Vulkan Video.
    Vulkan,
    /// `libx264`, in software.
    X264,
    /// `libopenh264`, in software — where this FFmpeg has no `libx264`.
    OpenH264,
}

impl EncodePath {
    /// Whether the encoding is done on the GPU.
    pub fn is_hardware(self) -> bool {
        matches!(self, Self::Nvenc | Self::MediaFoundation | Self::Vulkan)
    }
}

/// One video stream's frames in, H.264 packets out, encoded by whichever
/// encoder takes them — the encode side of
/// [`VideoDecodeBin`](crate::elements::VideoDecodeBin).
///
/// # When it chooses
///
/// Once, when it is opened, by opening the encoders in turn and keeping the
/// first that opens:
///
/// - D3D11 textures: `h264_nvenc`, then `h264_mf`, then software.
/// - CUDA frames: `h264_nvenc`, then software.
/// - Vulkan frames: `h264_vulkan` for NV12, then software; BGRA in software,
///   nothing on Vulkan here converting RGB to YUV yet.
/// - Frames in system memory: software.
///
/// Software is `libx264` where this FFmpeg has it, and `libopenh264`, which
/// it always has, otherwise. An encoder that does not open — no such GPU, a
/// driver too old, a consumer GPU out of NVENC sessions — is logged with why,
/// and the next is tried. [`Self::path`] says which it came to, before the
/// pipeline runs.
///
/// Never again after that: a muxer takes its stream's headers from the
/// encoder it was opened with ([`TrackFormat::from`] this), so one
/// encoder's packets could not follow another's in the same file.
///
/// # What it holds
///
/// The encoder, and what the encoder needs in front of it: on the software
/// path from a device, the download to system memory (`D3d11Download`,
/// `CudaDownload` or `VulkanDownload`, in the layout the frames are in), and always
/// a `SwScaler` to YUV420P at the size asked for; on `h264_mf` from BGRA
/// textures, a `D3d11Scaler` to NV12. They are ordinary elements, run the
/// way a [`Rack`](crate::elements::Rack) runs what it holds — each logging
/// under its own name, `{name}-encoder`, `{name}-download`, `{name}-convert`,
/// a failure it raises arriving with its own identity — and the bin is the
/// one node in the graph.
///
/// Of a hardware decoder's pictures in front of it — which that decoder's
/// fixed pool has to have room for — it holds: on D3D11, none, NVENC and
/// Media Foundation each copying a picture into a surface of their own as
/// it arrives; on CUDA NVENC, the pictures still being encoded, since it
/// encodes the decoder's surfaces as they are, as many as its delay —
/// more with `max_b_frames`; on the software path, the one picture its
/// download read last, kept to know a repeat of it. With this crate's own
/// fixture — one reference picture, no B-frames — a decode bin given no
/// room downstream at all still fed NVENC to the end, on D3D11 and on CUDA;
/// a stream with more reference pictures or with B-frames is what the
/// count is for.
///
/// # Colour
///
/// Every path says in the stream what its YUV is. From RGB, each converts
/// with a matrix it can state: `h264_nvenc` with BT.601, limited range,
/// which FFmpeg's wrapper writes into the stream by itself; the software
/// path with swscale's default, the same; and `h264_mf`, which converts by
/// the picture's size and says nothing, is given NV12 its scaler made BT.709
/// and told so. YUV frames are converted by nothing, and
/// [`VideoEncodeOptions::color`] is what is written for them.
pub struct VideoEncodeBin {
    pp_log: PpLog,
    name: Arc<str>,
    line: Line,
    /// What goes into `line` on its first use, once the bin knows whether it
    /// is in a pipeline — see [`Element::attach_context`].
    pending: Option<Vec<Box<dyn Filter>>>,
    context: Option<Arc<Context>>,
    pad: SrcPad,
    path: EncodePath,
    input: InputContract,
    /// What `EncodeInput::System` said every frame is in, for `consume` to
    /// hold each one to; `None` on a device, whose encoder checks its own.
    system_format: Option<ffmpeg::format::Pixel>,
    parameters: ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
}

/// What an encoder the bin may choose is built with, and what came of it.
struct Chosen {
    path: EncodePath,
    elements: Vec<Box<dyn Filter>>,
    parameters: ffmpeg::codec::Parameters,
    time_base: ffmpeg::Rational,
}

impl Chosen {
    /// `encoder` behind `before`, taking its stream's parameters first.
    fn of<E>(path: EncodePath, mut before: Vec<Box<dyn Filter>>, encoder: E) -> Self
    where
        E: Filter + 'static,
        for<'a> TrackFormat: From<&'a E>,
    {
        let TrackFormat {
            parameters,
            time_base,
        } = TrackFormat::from(&encoder);
        before.push(Box::new(encoder));
        Self {
            path,
            elements: before,
            parameters,
            time_base,
        }
    }
}

impl VideoEncodeBin {
    /// Chooses an H.264 encoder for frames from `input` and builds what it
    /// needs in front of it — see this type's own docs for the order.
    ///
    /// `name` names the bin; what it holds is called `{name}-encoder`,
    /// `{name}-convert` and `{name}-download`.
    ///
    /// Fails only where no encoder opens at all — not even the software
    /// one, for a size or a rate it refuses — with that encoder's own error.
    /// Nothing is left behind by a failure.
    pub fn open(
        name: impl Into<String>,
        input: EncodeInput,
        options: VideoEncodeOptions,
    ) -> Result<Self> {
        Self::open_passing_over(name, input, options, &[])
    }

    /// [`Self::open`], never choosing what is in `skip` — so a test can
    /// reach every path on a machine where an earlier one opens.
    pub(crate) fn open_passing_over(
        name: impl Into<String>,
        input: EncodeInput,
        options: VideoEncodeOptions,
        skip: &[EncodePath],
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoEncodeBin, &name, None);
        let mut refused = Vec::new();
        let mut tried = Tried {
            skip,
            refused: &mut refused,
        };
        let (chosen, input_contract) = match &input {
            EncodeInput::System { format } => (
                software(&name, Vec::new(), is_rgb(*format), options, &mut tried)?,
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
                    .with_layouts(PixelLayoutSet::of(PixelLayout::of(*format))),
            ),
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            EncodeInput::D3d11 { gpu, format } => (
                d3d11(&name, gpu, *format, options, &mut tried)?,
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d11).with_layouts(
                    match format {
                        D3d11VideoInputFormat::Nv12 => PixelLayoutSet::NV12,
                        D3d11VideoInputFormat::Bgra => PixelLayoutSet::BGRA,
                    },
                ),
            ),
            #[cfg(feature = "cuda")]
            EncodeInput::Cuda { device, format } => (
                cuda(&name, device, *format, options, &mut tried)?,
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                    .with_layouts(format.layouts()),
            ),
            #[cfg(feature = "vulkan")]
            EncodeInput::Vulkan { device, format } => (
                vulkan(&name, device, *format, options, &mut tried)?,
                PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Vulkan)
                    .with_layouts(format.layouts()),
            ),
        };
        for reason in &refused {
            pp_info!(pp_log: &pp_log, "passed over: {reason}");
        }
        pp_info!(pp_log: &pp_log, "opened: {:?}", chosen.path);
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::packet(MediaKind::VideoPacket)),
        );
        Ok(Self {
            pp_log,
            line: Line::new(ElementType::VideoEncodeBin, &name),
            name,
            pending: Some(chosen.elements),
            context: None,
            pad,
            path: chosen.path,
            input: InputContract::Fixed(input_contract),
            system_format: match input {
                EncodeInput::System { format } => Some(format),
                #[allow(unreachable_patterns)]
                _ => None,
            },
            parameters: chosen.parameters,
            time_base: chosen.time_base,
        })
    }

    /// Which encoder it chose. Never changes.
    pub fn path(&self) -> EncodePath {
        self.path
    }

    /// The stream the chosen encoder writes, as a muxer's `add_stream` takes
    /// it — through [`TrackFormat::from`] this.
    pub fn parameters(&self) -> ffmpeg::codec::Parameters {
        self.parameters.clone()
    }

    /// The unit each packet's timestamps are counted in — the chosen
    /// encoder's own.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.time_base
    }

    /// Puts what `open` built into the line, the first time there is a
    /// reason to: wired into a pipeline, or handed something without one.
    fn install(&mut self) {
        if let Some(elements) = self.pending.take()
            && let Some(line) = self.line.fill(elements, self.context.as_ref())
        {
            pp_info!(self, "filled: {line}");
        }
    }
}

/// What the encoders passed over so far were passed over for, and which
/// the caller ruled out before asking.
struct Tried<'a> {
    skip: &'a [EncodePath],
    refused: &'a mut Vec<String>,
}

impl Tried<'_> {
    /// Whether `path` may be tried; one ruled out is noted as passed over.
    fn may(&mut self, path: EncodePath) -> bool {
        let may = !self.skip.contains(&path);
        if !may {
            self.refused.push(format!("{path:?}: ruled out"));
        }
        may
    }

    fn refused(&mut self, what: &str, error: impl std::fmt::Display) {
        self.refused.push(format!("{what}: {error}"));
    }
}

impl From<&VideoEncodeBin> for TrackFormat {
    fn from(bin: &VideoEncodeBin) -> Self {
        Self::new(bin.parameters(), bin.time_base())
    }
}

/// The software path: `before` — what brings a device's frames into system
/// memory, if anything — then a scaler to YUV420P at the size asked for, and
/// `libx264`, or `libopenh264` where there is none. `rgb` is whether what
/// reaches the scaler is RGB, which decides what the stream says it holds.
fn software(
    name: &str,
    before: Vec<Box<dyn Filter>>,
    rgb: bool,
    options: VideoEncodeOptions,
    tried: &mut Tried<'_>,
) -> Result<Chosen> {
    let color = if rgb {
        Some(SWSCALE_FROM_RGB)
    } else {
        options.color
    };
    let open = |codec| {
        let options = SwEncoderOptions {
            codec,
            width: options.width,
            height: options.height,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: options.frame_rate,
            bit_rate: options.bit_rate,
            gop_size: options.gop_size,
            max_b_frames: options.max_b_frames,
        };
        let name = format!("{name}-encoder");
        match color {
            Some(color) => SwEncoder::with_color(name, options, color),
            None => SwEncoder::new(name, options),
        }
    };
    let x264 = if tried.may(EncodePath::X264) {
        open(VideoCodec::H264)
            .inspect_err(|error| tried.refused("libx264", error))
            .ok()
    } else {
        None
    };
    let (path, encoder) = match x264 {
        Some(encoder) => (EncodePath::X264, encoder),
        None => (EncodePath::OpenH264, open(VideoCodec::OpenH264)?),
    };
    let mut elements = before;
    elements.push(Box::new(SwScaler::new(
        format!("{name}-convert"),
        ffmpeg::format::Pixel::YUV420P,
        options.width,
        options.height,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )));
    Ok(Chosen::of(path, elements, encoder))
}

#[cfg(all(target_os = "windows", feature = "d3d11"))]
fn d3d11(
    name: &str,
    gpu: &D3d11Gpu,
    format: D3d11VideoInputFormat,
    options: VideoEncodeOptions,
    tried: &mut Tried<'_>,
) -> Result<Chosen> {
    let hardware = |codec, input_format| D3d11VideoEncoderOptions {
        codec,
        input_format,
        width: options.width,
        height: options.height,
        frame_rate: options.frame_rate,
        bit_rate: options.bit_rate,
        gop_size: options.gop_size,
        max_b_frames: options.max_b_frames,
    };
    let encoder = format!("{name}-encoder");
    let convert = format!("{name}-convert");

    // NVENC converts BGRA itself and says so; YUV is described as asked.
    if tried.may(EncodePath::Nvenc) {
        let nvenc_options = hardware(D3d11VideoCodec::H264Nvenc, format);
        let nvenc = match (format, options.color) {
            (D3d11VideoInputFormat::Nv12, Some(color)) => {
                D3d11VideoEncoder::with_color(&encoder, gpu, nvenc_options, color)
            }
            _ => D3d11VideoEncoder::new(&encoder, gpu, nvenc_options),
        };
        match nvenc {
            Ok(nvenc) => return Ok(Chosen::of(EncodePath::Nvenc, Vec::new(), nvenc)),
            Err(error) => tried.refused("h264_nvenc", error),
        }
    }

    // Media Foundation converts BGRA by the picture's size and says nothing,
    // so it is given NV12 a scaler made BT.709, and told that.
    if tried.may(EncodePath::MediaFoundation) {
        let mf_options = hardware(
            D3d11VideoCodec::H264MediaFoundation,
            D3d11VideoInputFormat::Nv12,
        );
        let mf = match format {
            D3d11VideoInputFormat::Bgra => {
                D3d11Scaler::to_format(&convert, gpu, D3d11ScalerFormat::Nv12)
                    .map_err(crate::Error::from)
                    .and_then(|scaler| {
                        let encoder = D3d11VideoEncoder::with_color(
                            &encoder,
                            gpu,
                            mf_options,
                            ColorDescription::BT709_LIMITED,
                        )?;
                        Ok((vec![Box::new(scaler) as Box<dyn Filter>], encoder))
                    })
            }
            D3d11VideoInputFormat::Nv12 => match options.color {
                Some(color) => D3d11VideoEncoder::with_color(&encoder, gpu, mf_options, color),
                None => D3d11VideoEncoder::new(&encoder, gpu, mf_options),
            }
            .map(|encoder| (Vec::new(), encoder))
            .map_err(crate::Error::from),
        };
        match mf {
            Ok((before, mf)) => return Ok(Chosen::of(EncodePath::MediaFoundation, before, mf)),
            Err(error) => tried.refused("h264_mf", error),
        }
    }

    // Software, from the same layout in system memory: nothing on the GPU
    // is asked of a device that may have no video processor at all.
    let download: Box<dyn Filter> = Box::new(D3d11Download::new(format!("{name}-download"), gpu)?);
    software(
        name,
        vec![download],
        format == D3d11VideoInputFormat::Bgra,
        options,
        tried,
    )
}

#[cfg(feature = "cuda")]
fn cuda(
    name: &str,
    device: &CudaDevice,
    format: CudaFrameFormat,
    options: VideoEncodeOptions,
    tried: &mut Tried<'_>,
) -> Result<Chosen> {
    let nvenc_options = CudaEncoderOptions {
        codec: CudaCodec::H264,
        input_format: format,
        width: options.width,
        height: options.height,
        frame_rate: options.frame_rate,
        bit_rate: options.bit_rate,
        gop_size: options.gop_size,
        max_b_frames: options.max_b_frames,
    };
    let encoder = format!("{name}-encoder");
    if tried.may(EncodePath::Nvenc) {
        let nvenc = match (format, options.color) {
            (CudaFrameFormat::Nv12, Some(color)) => {
                CudaEncoder::with_color(&encoder, device, nvenc_options, color)
            }
            _ => CudaEncoder::new(&encoder, device, nvenc_options),
        };
        match nvenc {
            Ok(nvenc) => return Ok(Chosen::of(EncodePath::Nvenc, Vec::new(), nvenc)),
            Err(error) => tried.refused("h264_nvenc", error),
        }
    }
    let download: Box<dyn Filter> = Box::new(CudaDownload::new(
        format!("{name}-download"),
        device,
        format,
    ));
    software(
        name,
        vec![download],
        format == CudaFrameFormat::Bgra,
        options,
        tried,
    )
}

#[cfg(feature = "vulkan")]
fn vulkan(
    name: &str,
    device: &VulkanDevice,
    format: VulkanFrameFormat,
    options: VideoEncodeOptions,
    tried: &mut Tried<'_>,
) -> Result<Chosen> {
    if format == VulkanFrameFormat::Nv12 && tried.may(EncodePath::Vulkan) {
        let vulkan_options = VulkanEncoderOptions {
            codec: VulkanCodec::H264,
            width: options.width,
            height: options.height,
            frame_rate: options.frame_rate,
            bit_rate: options.bit_rate,
            gop_size: options.gop_size,
            max_b_frames: options.max_b_frames,
        };
        let encoder = format!("{name}-encoder");
        let opened = match options.color {
            Some(color) => VulkanEncoder::with_color(&encoder, device, vulkan_options, color),
            None => VulkanEncoder::new(&encoder, device, vulkan_options),
        };
        match opened {
            Ok(encoder) => return Ok(Chosen::of(EncodePath::Vulkan, Vec::new(), encoder)),
            Err(error) => tried.refused("h264_vulkan", error),
        }
    }
    let download: Box<dyn Filter> =
        Box::new(VulkanDownload::new(format!("{name}-download"), device));
    software(
        name,
        vec![download],
        format == VulkanFrameFormat::Bgra,
        options,
        tried,
    )
}

/// Whether a frame in `got` is one a bin told `declared` should take —
/// the same layout, the full-range J variants being their plain ones.
fn same_layout(declared: ffmpeg::format::Pixel, got: ffmpeg::format::Pixel) -> bool {
    use ffmpeg::format::Pixel;
    let plain = |pixel| match pixel {
        Pixel::YUVJ420P => Pixel::YUV420P,
        Pixel::YUVJ422P => Pixel::YUV422P,
        Pixel::YUVJ444P => Pixel::YUV444P,
        other => other,
    };
    plain(declared) == plain(got)
}

fn is_rgb(format: ffmpeg::format::Pixel) -> bool {
    use ffmpeg::format::Pixel;
    matches!(
        format,
        Pixel::BGRA
            | Pixel::RGBA
            | Pixel::ARGB
            | Pixel::ABGR
            | Pixel::BGR24
            | Pixel::RGB24
            | Pixel::BGRZ
            | Pixel::RGBZ
            | Pixel::ZBGR
            | Pixel::ZRGB
    )
}

impl Element for VideoEncodeBin {
    /// Fills the line now that the pipeline is known, so what is in it logs
    /// and paces as part of that pipeline.
    fn attach_context(&mut self, context: &Arc<Context>) {
        self.context = Some(context.clone());
        self.install();
    }

    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::VideoEncodeBin
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for VideoEncodeBin {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VideoEncodeBin {
    /// Frames from where [`EncodeInput`] said, in the layout it said.
    fn input_contract(&self) -> InputContract {
        self.input
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.install();
        if let (Some(declared), MediaBuffer::Video(frame)) = (self.system_format, &buf)
            && !same_layout(declared, frame.format())
        {
            return Err(VideoEncodeBinError::FormatMismatch {
                declared,
                got: frame.format(),
            }
            .into());
        }
        for made in self.line.consume(buf)? {
            self.pad.push(made)?;
        }
        Ok(())
    }

    fn control(&mut self, msg: &ControlMsg) -> Result<()> {
        self.install();
        self.line.control(msg)
    }
}

impl Drop for VideoEncodeBin {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing what it held");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::set_time_base,
        elements::SwDecoder,
        test_support::{assert_rgb_near, capture},
    };

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const FRAMES: i64 = 30;
    /// (230, 20, 20) — red enough that a matrix read the wrong way shows.
    const RED: [u8; 3] = [230, 20, 20];
    /// The same, as BT.709 limited-range Y'CbCr.
    const RED_709: [u8; 3] = [72, 107, 220];

    fn options(color: Option<ColorDescription>) -> VideoEncodeOptions {
        VideoEncodeOptions {
            width: WIDTH,
            height: HEIGHT,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 2_000_000,
            gop_size: 30,
            max_b_frames: None,
            color,
        }
    }

    fn stamped(mut frame: ffmpeg::frame::Video, pts: i64) -> MediaBuffer {
        frame.set_pts(Some(pts));
        set_time_base(&mut frame, ffmpeg::Rational::new(1, 30));
        MediaBuffer::video(frame)
    }

    fn bgra(pts: i64) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, WIDTH, HEIGHT);
        for pixel in frame.data_mut(0).as_chunks_mut::<4>().0 {
            *pixel = [RED[2], RED[1], RED[0], 255];
        }
        stamped(frame, pts)
    }

    fn yuv420p(pts: i64) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, WIDTH, HEIGHT);
        for (plane, value) in RED_709.into_iter().enumerate() {
            frame.data_mut(plane).fill(value);
        }
        frame.set_color_space(ffmpeg::color::Space::BT709);
        frame.set_color_range(ffmpeg::color::Range::MPEG);
        stamped(frame, pts)
    }

    /// Runs `frames` through `bin` and its end of stream, and answers what
    /// came out.
    fn encode(
        mut bin: VideoEncodeBin,
        frames: impl Iterator<Item = MediaBuffer>,
    ) -> Vec<MediaBuffer> {
        let received = capture(&mut bin);
        for frame in frames {
            bin.consume(frame).expect("a frame encodes");
        }
        bin.consume(MediaBuffer::Eos).expect("the encoder drains");
        std::mem::take(&mut *received.lock().unwrap())
    }

    /// Decodes what `encode` answered, of a stream described by
    /// `parameters`, and answers how many pictures came out, the centre of
    /// the last, read with the colour the stream says it holds, and the
    /// matrix it says.
    fn decoded(
        parameters: ffmpeg::codec::Parameters,
        packets: Vec<MediaBuffer>,
    ) -> (usize, [u8; 3], ffmpeg::color::Space) {
        let mut decoder = SwDecoder::new("check", parameters).expect("an H.264 decoder");
        let received = capture(&mut decoder);
        for packet in packets {
            decoder.consume(packet).expect("the stream decodes");
        }
        let pictures: Vec<_> = received
            .lock()
            .unwrap()
            .iter()
            .filter_map(|buffer| match buffer {
                MediaBuffer::Video(frame) => Some(frame.clone()),
                _ => None,
            })
            .collect();
        let last = pictures.last().expect("pictures came out");
        assert_eq!((last.width(), last.height()), (WIDTH, HEIGHT));
        let (x, y) = (WIDTH as usize / 2, HEIGHT as usize / 2);
        let sample = |plane: usize, x: usize, y: usize| {
            f32::from(last.data(plane)[y * last.stride(plane) + x]) / 255.0
        };
        let (luma, cb, cr) = (
            sample(0, x, y),
            sample(1, x / 2, y / 2),
            sample(2, x / 2, y / 2),
        );
        let rows = crate::color::yuv_to_rgb_rows(last.color_space(), last.color_range(), HEIGHT);
        let rgb = rows.map(|[from_y, from_cb, from_cr, offset]| {
            ((from_y * luma + from_cb * cb + from_cr * cr + offset).clamp(0.0, 1.0) * 255.0 + 0.5)
                as u8
        });
        (pictures.len(), rgb, last.color_space())
    }

    /// Encodes `frames` of red through a bin opened on `input`, never
    /// choosing what is in `skip`, and checks the stream plays as red,
    /// every frame of it. Answers the path it took.
    fn encodes_red(
        input: EncodeInput,
        color: Option<ColorDescription>,
        skip: &[EncodePath],
        frames: impl Iterator<Item = MediaBuffer>,
    ) -> EncodePath {
        let bin = VideoEncodeBin::open_passing_over("encode", input, options(color), skip)
            .expect("some encoder opens");
        let path = bin.path();
        let parameters = bin.parameters();
        let made = encode(bin, frames);
        assert!(
            matches!(made.first(), Some(MediaBuffer::Packet(packet)) if packet.is_key()),
            "{path:?}: the stream starts on a keyframe"
        );
        assert!(
            made.last().is_some_and(MediaBuffer::is_eos),
            "{path:?}: the end of stream follows the packets"
        );
        let (pictures, rgb, space) = decoded(parameters, made);
        // Said, not left to a guess: read untagged, a 320x240 picture is
        // taken to be BT.601 and would come out right by luck.
        assert_ne!(
            space,
            ffmpeg::color::Space::Unspecified,
            "{path:?}: the stream says what its YUV is"
        );
        assert_eq!(
            pictures, FRAMES as usize,
            "{path:?}: every frame was encoded"
        );
        assert_rgb_near(rgb, RED, 6, &format!("{path:?}"));
        path
    }

    /// YUV in system memory is encoded in software, and the stream says
    /// the colour it was told, so red plays as red — on either software
    /// encoder.
    #[test]
    fn yuv_in_system_memory_says_the_colour_it_was_told() {
        for skip in [&[][..], &[EncodePath::X264][..]] {
            let path = encodes_red(
                EncodeInput::System {
                    format: ffmpeg::format::Pixel::YUV420P,
                },
                Some(ColorDescription::BT709_LIMITED),
                skip,
                (0..FRAMES).map(yuv420p),
            );
            assert!(!path.is_hardware(), "{path:?}");
            assert!(!skip.contains(&path), "{path:?} was ruled out");
        }
    }

    /// BGRA in system memory is converted in software, and the stream says
    /// what the conversion made — with no description asked for.
    #[test]
    fn rgb_in_system_memory_says_what_it_was_made_into() {
        let path = encodes_red(
            EncodeInput::System {
                format: ffmpeg::format::Pixel::BGRA,
            },
            None,
            &[],
            (0..FRAMES).map(bgra),
        );
        assert!(!path.is_hardware(), "{path:?}");
    }

    /// What a decode bin puts out, as an encode bin takes it: the layout it
    /// said, and nothing where no encode bin takes that.
    #[test]
    fn a_decode_bins_output_is_an_encode_bins_input() {
        use crate::elements::DecodeTarget;
        use ffmpeg::format::Pixel;

        assert!(matches!(
            EncodeInput::for_decoded(&DecodeTarget::System, Some(Pixel::YUV420P)),
            Some(EncodeInput::System {
                format: Pixel::YUV420P
            })
        ));
        assert!(EncodeInput::for_decoded(&DecodeTarget::System, None).is_none());
        #[cfg(all(target_os = "windows", feature = "d3d11"))]
        if let Some(gpu) = crate::test_support::try_d3d11_gpu() {
            let target = DecodeTarget::D3d11 {
                gpu,
                downstream_hw_frames: 4,
            };
            assert!(matches!(
                EncodeInput::for_decoded(&target, Some(Pixel::BGRA)),
                Some(EncodeInput::D3d11 {
                    format: D3d11VideoInputFormat::Bgra,
                    ..
                })
            ));
            assert!(EncodeInput::for_decoded(&target, Some(Pixel::P010LE)).is_none());
        }
    }

    /// A frame in another layout than the one `EncodeInput::System` said is
    /// refused, not encoded under the wrong colour description.
    #[test]
    fn a_frame_in_another_layout_is_refused() {
        let mut bin = VideoEncodeBin::open(
            "encode",
            EncodeInput::System {
                format: ffmpeg::format::Pixel::NV12,
            },
            options(None),
        )
        .unwrap();
        let received = capture(&mut bin);
        assert!(matches!(
            bin.consume(yuv420p(0)),
            Err(crate::Error::VideoEncodeBinError(
                VideoEncodeBinError::FormatMismatch { .. }
            ))
        ));
        assert!(received.lock().unwrap().is_empty(), "nothing was encoded");
    }

    /// What the bin takes is what it was opened for: frames in system
    /// memory in the layout it was told, and nothing on a device.
    #[test]
    fn it_links_to_what_it_was_opened_for() {
        let bin = VideoEncodeBin::open(
            "encode",
            EncodeInput::System {
                format: ffmpeg::format::Pixel::YUV420P,
            },
            options(None),
        )
        .unwrap();
        let from = |domain, layout| {
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, domain)
                    .with_layouts(PixelLayoutSet::of(layout)),
            )
        };
        let fits =
            |produced| !crate::contract::check_link(&produced, &bin.input_contract()).is_refused();
        assert!(fits(from(MemoryDomain::System, PixelLayout::Yuv420p)));
        assert!(!fits(from(MemoryDomain::D3d11, PixelLayout::Nv12)));
    }

    /// What the bin says of its stream is what a muxer writes the file
    /// from: a recording through it opens again as H.264 of the size and
    /// length it was made at, every frame of it there.
    fn records(
        bin: VideoEncodeBin,
        frames: impl Iterator<Item = MediaBuffer>,
    ) -> Option<ColorDescription> {
        use crate::elements::{FileDemuxer, FileMuxer};

        // A file of its own for every recording, not one per path: tests run
        // at once, and two that took the same path wrote into one file. On
        // a runner with no NVIDIA GPU a BGRA recording falls to software,
        // the path a YUV420P one takes, and CI read the BGRA one's colour
        // back as the YUV420P one's.
        static RECORDINGS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = bin.path();
        let file = std::env::temp_dir().join(format!(
            "media-pp-encode-bin-{}-{}-{path:?}.mp4",
            std::process::id(),
            RECORDINGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let mut muxer = FileMuxer::create(&file).expect("the file opens");
        let track = muxer.add_stream("video", &bin).expect("the track is added");
        let mut sinks = muxer.open().expect("the file is written");
        let mut sink = sinks.take(track).expect("the track's sink");
        for made in encode(bin, frames) {
            sink.consume(made)
                .expect("the muxer takes what the bin made");
        }
        drop(sink);
        drop(sinks);

        let (demuxer, streams) = FileDemuxer::open("check", &file).expect("the file opens again");
        let length = demuxer.duration();
        let video = &streams[0];
        let codec = video.parameters.id();
        // SAFETY: plain fields of parameters this test owns.
        let size = unsafe {
            let raw = video.parameters.as_ptr();
            ((*raw).width as u32, (*raw).height as u32)
        };
        let color = video.color();
        // Closed before it is deleted, which Windows refuses while anything
        // has it open.
        drop(streams);
        drop(demuxer);
        let _ = std::fs::remove_file(&file);
        assert_eq!(codec, ffmpeg::codec::Id::H264, "{path:?}");
        assert_eq!(size, (WIDTH, HEIGHT), "{path:?}");
        let length = length.expect("the file says how long it is");
        assert!(
            length.abs_diff(std::time::Duration::from_secs(1))
                < std::time::Duration::from_millis(100),
            "{path:?}: {length:?} long"
        );
        color
    }

    #[test]
    fn a_software_recording_opens_again() {
        let bin = VideoEncodeBin::open(
            "encode",
            EncodeInput::System {
                format: ffmpeg::format::Pixel::YUV420P,
            },
            options(Some(ColorDescription::BT709_LIMITED)),
        )
        .unwrap();
        let color = records(bin, (0..FRAMES).map(yuv420p));
        // What the stream was told, read back off the file — what a
        // transcode of it hands its own encoder.
        assert_eq!(color, Some(ColorDescription::BT709_LIMITED));
    }

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    mod d3d11 {
        use super::*;
        use crate::{elements::D3d11Upload, repeat::PerFrameTransform};

        /// Frames put on `gpu` as they come.
        fn uploaded(
            gpu: &D3d11Gpu,
            frames: impl Iterator<Item = MediaBuffer>,
        ) -> impl Iterator<Item = MediaBuffer> {
            let mut upload = D3d11Upload::new("upload", gpu);
            frames.map(move |frame| {
                let MediaBuffer::Video(frame) = frame else {
                    unreachable!()
                };
                MediaBuffer::Video(upload.transform(&frame).expect("the frame uploads"))
            })
        }

        /// Each path the bin can take from D3D11 textures, in turn: the first
        /// it chooses, then with NVENC ruled out, then with both hardware
        /// encoders ruled out.
        const IN_TURN: [&[EncodePath]; 3] = [
            &[],
            &[EncodePath::Nvenc],
            &[EncodePath::Nvenc, EncodePath::MediaFoundation],
        ];

        /// BGRA textures — a compositor's — play as red on every path the
        /// bin can take from them: NVENC converting and saying so itself,
        /// Media Foundation given NV12 made BT.709, and software from a
        /// download.
        #[test]
        fn bgra_textures_play_as_red_on_every_path() {
            let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
                return;
            };
            let _session = crate::test_support::encoder_session();
            let mut taken = Vec::new();
            for skip in IN_TURN {
                let input = EncodeInput::D3d11 {
                    gpu: gpu.clone(),
                    format: D3d11VideoInputFormat::Bgra,
                };
                let frames = uploaded(&gpu, (0..FRAMES).map(bgra));
                taken.push(encodes_red(input, None, skip, frames));
            }
            eprintln!("paths taken: {taken:?}");
            assert!(!taken[2].is_hardware(), "software, with both ruled out");
        }

        /// A recording from each hardware path opens again: their streams'
        /// headers are the ones the muxer is given.
        #[test]
        fn a_hardware_recording_opens_again() {
            let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
                return;
            };
            let _session = crate::test_support::encoder_session();
            for skip in &IN_TURN[..2] {
                let input = EncodeInput::D3d11 {
                    gpu: gpu.clone(),
                    format: D3d11VideoInputFormat::Bgra,
                };
                let bin = VideoEncodeBin::open_passing_over("encode", input, options(None), skip)
                    .unwrap();
                records(bin, uploaded(&gpu, (0..FRAMES).map(bgra)));
            }
        }

        /// NV12 textures — a decode's or an upload's — say the colour they
        /// were told, on every path.
        #[test]
        fn nv12_textures_say_the_colour_they_were_told() {
            let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
                return;
            };
            let _session = crate::test_support::encoder_session();
            let mut taken = Vec::new();
            for skip in IN_TURN {
                let input = EncodeInput::D3d11 {
                    gpu: gpu.clone(),
                    format: D3d11VideoInputFormat::Nv12,
                };
                let frames = uploaded(&gpu, (0..FRAMES).map(yuv420p));
                taken.push(encodes_red(
                    input,
                    Some(ColorDescription::BT709_LIMITED),
                    skip,
                    frames,
                ));
            }
            eprintln!("paths taken: {taken:?}");
            assert!(!taken[2].is_hardware(), "software, with both ruled out");
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use crate::{elements::CudaUpload, repeat::PerFrameTransform};

        /// A format, what it holds, and how a frame of red is made in it.
        type Case = (
            CudaFrameFormat,
            Option<ColorDescription>,
            fn(i64) -> MediaBuffer,
        );

        /// BGRA and NV12 on CUDA play as red, on NVENC and in software.
        #[test]
        fn cuda_frames_play_as_red_on_every_path() {
            let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
                return;
            };
            let _session = crate::test_support::encoder_session();
            let cases: [Case; 2] = [
                (CudaFrameFormat::Bgra, None, bgra),
                (
                    CudaFrameFormat::Nv12,
                    Some(ColorDescription::BT709_LIMITED),
                    yuv420p,
                ),
            ];
            for (format, color, make) in cases {
                for skip in [&[][..], &[EncodePath::Nvenc][..]] {
                    let mut upload = CudaUpload::new("upload", &device, format);
                    let frames = (0..FRAMES).map(make).map(move |frame| {
                        let MediaBuffer::Video(frame) = frame else {
                            unreachable!()
                        };
                        MediaBuffer::Video(upload.transform(&frame).expect("the frame uploads"))
                    });
                    let input = EncodeInput::Cuda {
                        device: device.clone(),
                        format,
                    };
                    let path = encodes_red(input, color, skip, frames);
                    eprintln!("{format:?}, passing over {skip:?}: {path:?}");
                }
            }
        }
    }

    #[cfg(feature = "vulkan")]
    mod vulkan {
        use super::*;
        use crate::{elements::VulkanUpload, repeat::PerFrameTransform};

        /// A format, what it holds, and how a frame of red is made in it.
        type Case = (
            VulkanFrameFormat,
            Option<ColorDescription>,
            fn(i64) -> MediaBuffer,
        );

        /// BGRA and NV12 on Vulkan play as red: NV12 on `h264_vulkan` and in
        /// software, BGRA in software after its download.
        #[test]
        fn vulkan_frames_play_as_red_on_every_path() {
            let Some(device) = crate::test_support::try_vulkan_device() else {
                return;
            };
            let _session = crate::test_support::encoder_session();
            let cases: [Case; 2] = [
                (VulkanFrameFormat::Bgra, None, bgra),
                (
                    VulkanFrameFormat::Nv12,
                    Some(ColorDescription::BT709_LIMITED),
                    yuv420p,
                ),
            ];
            for (format, color, make) in cases {
                for skip in [&[][..], &[EncodePath::Vulkan][..]] {
                    let mut upload = VulkanUpload::new("upload", &device);
                    let frames = (0..FRAMES).map(make).map(move |frame| {
                        let MediaBuffer::Video(frame) = frame else {
                            unreachable!()
                        };
                        MediaBuffer::Video(upload.transform(&frame).expect("the frame uploads"))
                    });
                    let input = EncodeInput::Vulkan {
                        device: device.clone(),
                        format,
                    };
                    let path = encodes_red(input, color, skip, frames);
                    eprintln!("{format:?}, passing over {skip:?}: {path:?}");
                    if format == VulkanFrameFormat::Bgra || !skip.is_empty() {
                        assert!(!path.is_hardware(), "{format:?}: in software");
                    }
                }
            }
        }
    }
}
