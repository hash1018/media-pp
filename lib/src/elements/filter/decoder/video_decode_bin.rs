//! A video stream decoded to wherever its pictures are wanted, by whichever
//! path can take it.
//!
//! A hardware decoder is the cheap way to put a stream on a device, and the
//! only one worth having for the streams it takes. It does not take them all:
//! a codec the GPU has no decoder for (ProRes, DNxHD, Motion JPEG through
//! D3D11VA), a picture it cannot hand on (4:2:2 and 4:4:4, where every
//! consumer of a decoded surface here reads 4:2:0), alpha, which no hardware
//! decoder keeps, and a profile the GPU turns out not to have, which nothing
//! says until it is asked to decode one. FFmpeg decodes every one of those in
//! software, and an upload puts the result on the same device, so the stream
//! reaches the same place either way. 10-bit 4:2:0 the hardware decodes, and
//! the GPU brings down to 8 bits after it where the target can.
//!
//! [`VideoDecodeBin`] is one element holding whichever of those lines the
//! stream needs. It chooses when it is opened, from what the stream says
//! about itself, and changes its mind at most once, if the hardware refuses a
//! stream it chose the hardware for. With [`DecodeTarget::System`] it is the
//! software decoder alone, so the same code builds a decoder in a build
//! without any GPU backend.

use std::sync::{Arc, Mutex};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;

#[cfg(all(target_os = "windows", feature = "d3d11"))]
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11DeviceContext};
#[cfg(all(target_os = "windows", feature = "d3d12"))]
use windows::Win32::Graphics::Direct3D12::ID3D12Device;

#[cfg(feature = "cuda")]
use crate::elements::{CudaDevice, CudaFrameFormat};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    control::ControlMsg,
    element::{Context, Element, ElementType, Filter, Sink, Source, element_pp_log},
    elements::{DecodeThreading, SwDecoder, filter::line::Line},
    error::{Error, Result},
    pad::SrcPad,
    pp_log::{PpLog, pp_info, pp_warn},
};

/// Errors specific to [`VideoDecodeBin`]. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]); every other failure is the
/// chosen element's own, from its constructor.
#[derive(Debug, ThisError)]
pub enum VideoDecodeBinError {
    /// The stream is not video.
    #[error("VideoDecodeBin decodes video, but the stream is {0:?}")]
    NotVideo(ffmpeg::media::Type),

    /// The stream does not say how large its pictures are, which the
    /// software path onto a device has to know to make a surface for them.
    #[error("the stream does not say its picture size")]
    UnknownSize,

    /// The software path would put out NV12, which cannot have an odd side,
    /// and the target has no other layout — `DecodeTarget::D3d12`.
    #[error("{0}x{1} has an odd side, which NV12 on D3D12 cannot have")]
    OddSize(u32, u32),
}

/// Where decoded pictures are to end up.
///
/// A target owns its device, and cloning one is cheap: a device is a
/// reference-counted handle. `downstream_hw_frames` is the hardware
/// decoder's surface budget, passed on as it is — see `D3d11Decoder::new` or
/// `CudaDecoder::new` for why it has to cover every frame downstream may
/// hold. The software path has a growable pool and does not need it.
#[derive(Clone)]
pub enum DecodeTarget {
    /// CPU memory, as [`crate::elements::SwDecoder`] puts it out: decoded
    /// in software, in whatever layout the stream decodes to. There is
    /// nothing to choose between, and nothing to fall back to — this is
    /// here so one piece of code can build a decoder whichever backends a
    /// build has.
    System,
    /// D3D11 textures on `device`, as [`crate::elements::D3d11Decoder`] and
    /// [`crate::elements::D3d11Upload`] make them — NV12, or BGRA where the
    /// stream has alpha or an odd side. A 10-bit stream is brought down to
    /// NV12 by a [`crate::elements::D3d11Scaler`], which draws through
    /// `context`: the immediate context every D3D11 element in the pipeline
    /// shares, as that element's own docs require.
    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    D3d11 {
        device: ID3D11Device,
        context: Arc<Mutex<ID3D11DeviceContext>>,
        downstream_hw_frames: i32,
    },
    /// D3D12 resources on `device`, as [`crate::elements::D3d12Decoder`] and
    /// [`crate::elements::D3d12Upload`] make them — always NV12, the one
    /// layout either of them has and the one everything reading D3D12 frames
    /// here reads. So alpha is not kept on this target, and is no reason to
    /// decode in software; a 10-bit stream is decoded in software, there
    /// being nothing on D3D12 here to bring it down to 8 bits after the
    /// hardware; and a software-decoded stream with an odd side
    /// has no layout to go in and is refused with
    /// [`VideoDecodeBinError::OddSize`]. `D3d12Decoder`'s pool grows, so
    /// there is no surface budget to give.
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    D3d12 { device: ID3D12Device },
    /// CUDA frames on `device`, as [`crate::elements::CudaDecoder`] and
    /// [`crate::elements::CudaUpload`] make them — NV12, or BGRA where the
    /// stream has alpha or an odd side. A 10-bit stream is brought down to
    /// NV12 by a [`crate::elements::CudaScaler`].
    #[cfg(feature = "cuda")]
    Cuda {
        device: CudaDevice,
        downstream_hw_frames: i32,
    },
}

/// Which way a [`VideoDecodeBin`] decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodePath {
    /// On the target's own hardware decoder.
    Hardware,
    /// In software, and uploaded where the target is a device — and why.
    Software(SoftwareReason),
}

/// Why a [`VideoDecodeBin`] decodes in software.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftwareReason {
    /// The target is [`DecodeTarget::System`], which software decoding is
    /// how pictures reach.
    SystemMemory,
    /// The stream's pictures carry alpha. No hardware decoder keeps it, so
    /// where the target has a layout that does, these are decoded in
    /// software and uploaded as BGRA.
    Alpha,
    /// This FFmpeg build has no decoder for the codec that decodes on the
    /// target's hardware.
    NoHardwareDecoder(ffmpeg::codec::Id),
    /// The pictures are not 4:2:0 — 4:2:2, 4:4:4 — or are 10-bit 4:2:0 on a
    /// target with nothing to bring them down to 8 bits after the hardware.
    /// A hardware decoder that takes them at all hands on a surface (Y210,
    /// P010, ...) that nothing reading decoded surfaces here takes.
    PixelFormat(ffmpeg::format::Pixel),
    /// The hardware was chosen, and refused the stream once asked to decode
    /// it — a profile or a size the GPU does not have, which a stream's
    /// parameters do not say. The bin went on in software from there.
    HardwareRefused,
}

/// One video stream's packets in, its pictures out where the
/// [`DecodeTarget`] says — decoded by the target's hardware where it takes
/// the stream, and otherwise in software and uploaded.
///
/// # What it holds
///
/// On the hardware path, the target's decoder (`D3d11Decoder`,
/// `D3d12Decoder`, `CudaDecoder`), and for a 10-bit stream its scaler
/// (`D3d11Scaler`, `CudaScaler`) after it, bringing each picture down to
/// NV12 at the same size. On the software path, `SwDecoder` →
/// `SwScaler` → the target's upload, or `SwDecoder` alone for
/// [`DecodeTarget::System`]. They are ordinary elements, run the way a
/// [`Rack`](crate::elements::Rack) runs what it holds: each in the pipeline,
/// logging under its own name, a failure it raises arriving with its own
/// identity. What they are not is graph nodes — the bin is one node in the
/// topology diagram and in [`crate::stats`], and it logs what it holds, at
/// `Info`, each time that changes.
///
/// # When it chooses
///
/// When it is opened, from the stream's parameters: alpha where the target
/// can keep it, a codec without a hardware decoder, or a layout that is not
/// 4:2:0 take the software path, and so does 10-bit on
/// `DecodeTarget::D3d12`; anything else the hardware.
/// [`Self::path`] and [`Self::output_format`] answer before the pipeline
/// runs, so what is built after this can depend on them.
///
/// And once more, at most: a hardware decoder can open for a stream and then
/// refuse it at its first picture, or at a later change of size. The bin
/// then puts the software path in its place, gives it the preroll a seek was
/// in the middle of, and feeds it again every packet since the last
/// keyframe, which it keeps for exactly this — a hardware decoder's
/// reference pictures go with it, so the software one has to start from a
/// keyframe. It logs the change at `Warn`, and [`VideoDecodeBinHandle::path`]
/// then reads [`SoftwareReason::HardwareRefused`].
///
/// What downstream sees does not change: frames on the same device, in the
/// same [`Self::output_format`], at the size the stream was opened with — a
/// stream that changed size is scaled back to it — with PTS and colour
/// description carried through. The pictures in flight when the hardware
/// refused are decoded again, so a few may arrive twice.
///
/// What is kept for a replacement is one keyframe interval of compressed
/// packets, until the next keyframe replaces it; a stream with keyframes
/// far apart keeps more. Nothing is kept on the software path, which has
/// nothing to fall back to.
///
/// # Errors
///
/// Any other failure is the failing element's own, returned from
/// `consume` as it would be without the bin. A replacement that fails
/// leaves the hardware line in place and returns why.
pub struct VideoDecodeBin {
    pp_log: PpLog,
    name: Arc<str>,
    line: Line,
    /// What goes into `line` on its first use, once the bin knows whether it
    /// is in a pipeline — see [`Element::attach_context`].
    pending: Option<Vec<Box<dyn Filter>>>,
    context: Option<Arc<Context>>,
    pad: SrcPad,
    output_format: Option<ffmpeg::format::Pixel>,
    path: Arc<Mutex<DecodePath>>,
    /// Present while the hardware is decoding: what a replacement is built
    /// from, and what it has to be given.
    fallback: Option<Fallback>,
}

// SAFETY: what is not `Send` by its own type is the device a target holds —
// a COM interface to a free-threaded D3D device, or a reference to an FFmpeg
// CUDA device context with no thread affinity — the same reasoning the
// decoders and uploads give for holding one. `&mut self` on every method
// rules out concurrent use.
unsafe impl Send for VideoDecodeBin {}

/// What the software path is built from, should the hardware refuse.
struct Fallback {
    target: DecodeTarget,
    threading: Option<DecodeThreading>,
    params: ffmpeg::codec::Parameters,
    size: Option<(u32, u32)>,
    /// Every packet from the last keyframe on, the one that failed included.
    since_keyframe: Vec<MediaBuffer>,
    /// A preroll the line is in the middle of, which a new line has to be
    /// armed with before it decodes anything — see `PrerollGate`.
    preroll: Option<ControlMsg>,
}

/// Reads which way a [`VideoDecodeBin`] is decoding, from anywhere.
///
/// Cheap to clone; keeps only the answer alive, not the bin, its elements, or
/// its pipeline. After the bin is gone it reads the last path the bin took.
#[derive(Clone)]
pub struct VideoDecodeBinHandle {
    path: Arc<Mutex<DecodePath>>,
}

impl VideoDecodeBinHandle {
    /// Which way the bin is decoding now.
    pub fn path(&self) -> DecodePath {
        *self
            .path
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl VideoDecodeBin {
    /// Chooses how to decode the stream `params` describes onto `target`,
    /// and builds the elements for it.
    ///
    /// `name` names the bin; what it holds is called `{name}-decoder`, and
    /// on the software path onto a device also `{name}-convert` and
    /// `{name}-upload` — or, for a 10-bit stream on the hardware path,
    /// `{name}-convert` alone.
    ///
    /// Fails with [`VideoDecodeBinError`] for a stream that is not video, or
    /// one whose pictures the target cannot be given — a size unknown or odd
    /// where the software path needs one it can make a surface of — and
    /// otherwise with whatever the chosen element's constructor fails with:
    /// `threading` is how many threads a software decode may have and what
    /// it may spend them on — see [`DecodeThreading`] — and `None` leaves it
    /// as [`SwDecoder::new`] does. It is the software path's, including one
    /// put in when the hardware refuses; the hardware decoders have no such
    /// choice to make.
    ///
    /// a codec FFmpeg cannot decode at all is `SwDecoder`'s error. Nothing is
    /// left behind by a failure.
    pub fn open(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        target: DecodeTarget,
        threading: Option<DecodeThreading>,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::VideoDecodeBin, &name, None);
        let stream = Stream::of(&params)?;
        let path = choose(&stream, &target);

        let (elements, output_format, fallback) = match path {
            DecodePath::Hardware => {
                let mut elements =
                    vec![target.hardware_decoder(format!("{name}-decoder"), params.clone())?];
                if stream.format.is_some_and(is_10bit_420) {
                    let (width, height) = stream.size.ok_or(VideoDecodeBinError::UnknownSize)?;
                    elements.extend(target.to_8bit(format!("{name}-convert"), width, height)?);
                }
                let fallback = Fallback {
                    target: target.clone(),
                    threading,
                    params,
                    size: stream.size,
                    since_keyframe: Vec::new(),
                    preroll: None,
                };
                (elements, Some(ffmpeg::format::Pixel::NV12), Some(fallback))
            }
            DecodePath::Software(reason) => {
                let format = target.software_format(reason, stream.size)?;
                let elements = software(&name, params, stream.size, format, &target, threading)?;
                // Where there is no upload, what comes out is what the
                // stream decodes to.
                (elements, format.or(stream.format), None)
            }
        };
        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(MediaKind::VideoFrame, target.domain())),
        );
        pp_info!(pp_log: &pp_log, "opened: {path:?}, putting out {output_format:?}");
        Ok(Self {
            pp_log,
            line: Line::new(ElementType::VideoDecodeBin, &name),
            name,
            pending: Some(elements),
            context: None,
            pad,
            output_format,
            path: Arc::new(Mutex::new(path)),
            fallback,
        })
    }

    /// Which way this decodes now, and why where it is software.
    pub fn path(&self) -> DecodePath {
        self.handle().path()
    }

    /// The layout of the frames this puts out. On a device, `NV12`, or
    /// `BGRA` where the stream has alpha or a side NV12 cannot have and the
    /// target has BGRA. On [`DecodeTarget::System`], whatever the stream
    /// decodes to — `None` where the stream does not say.
    ///
    /// Never changes, not even when the hardware refuses.
    pub fn output_format(&self) -> Option<ffmpeg::format::Pixel> {
        self.output_format
    }

    /// A handle that reads [`Self::path`] after the bin has gone into a
    /// pipeline.
    pub fn handle(&self) -> VideoDecodeBinHandle {
        VideoDecodeBinHandle {
            path: Arc::clone(&self.path),
        }
    }

    /// Puts what `open` built into the line, the first time there is a
    /// reason to: wired into a pipeline, or handed something without one.
    fn install(&mut self) {
        if let Some(elements) = self.pending.take() {
            self.fill(elements);
        }
    }

    fn fill(&mut self, elements: Vec<Box<dyn Filter>>) {
        if let Some(line) = self.line.fill(elements, self.context.as_ref()) {
            pp_info!(self, "filled: {line}");
        }
    }

    fn push(&mut self, made: Vec<MediaBuffer>) -> Result<()> {
        for buf in made {
            self.pad.push(buf)?;
        }
        Ok(())
    }

    /// Replaces the hardware line with the software one, and gives it what
    /// the hardware had been given since the last keyframe.
    fn fall_back(&mut self, error: &Error) -> Result<()> {
        let Some(fallback) = self.fallback.take() else {
            return Ok(());
        };
        let elements = match software(
            &self.name,
            fallback.params.clone(),
            fallback.size,
            self.output_format,
            &fallback.target,
            fallback.threading,
        ) {
            Ok(elements) => elements,
            Err(replacement) => {
                pp_warn!(
                    self,
                    "the hardware decoder refused the stream ({error}), and no software \
                     decoder could replace it: {replacement}"
                );
                self.fallback = Some(fallback);
                return Err(replacement);
            }
        };
        pp_warn!(
            self,
            "the hardware decoder refused the stream ({error}); decoding in software from \
             here, again from the last keyframe ({} packets)",
            fallback.since_keyframe.len()
        );
        self.fill(elements);
        *self
            .path
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            DecodePath::Software(SoftwareReason::HardwareRefused);

        if let Some(preroll) = fallback.preroll {
            self.line.control(preroll)?;
        }
        // One packet that fails does not keep the rest from being decoded,
        // as it would not have in the line being replaced either; the first
        // failure is what is answered.
        let mut first_error = None;
        for buf in fallback.since_keyframe {
            let result = self.line.consume(buf).and_then(|made| self.push(made));
            if let Err(error) = result {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl Element for VideoDecodeBin {
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
        ElementType::VideoDecodeBin
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for VideoDecodeBin {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for VideoDecodeBin {
    /// Encoded video, which is what every decoder it may hold takes.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::packet(MediaKind::VideoPacket))
    }

    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.install();
        let eos = buf.is_eos();
        if let (Some(fallback), MediaBuffer::Packet(packet)) = (&mut self.fallback, &buf) {
            if packet.is_key() {
                fallback.since_keyframe.clear();
            }
            fallback.since_keyframe.push(buf.clone());
        }
        match self.line.consume(buf) {
            Ok(made) => self.push(made),
            Err(error) if self.fallback.is_some() && refused(&error) => {
                self.fall_back(&error)?;
                // An end of stream is not kept with the packets: it goes to
                // the new line after them, as it went to the old one.
                if eos {
                    let made = self.line.consume(MediaBuffer::Eos)?;
                    self.push(made)?;
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        self.install();
        // Mirrors what the decoder inside does with each, so a replacement
        // can be put in the same state: a preroll is armed until Pause,
        // Resume or Stop ends it, and a Flush or Stop drops the packets a
        // decoder has been told to forget.
        if let Some(fallback) = &mut self.fallback {
            match &msg {
                ControlMsg::Preroll(_) => fallback.preroll = Some(msg.clone()),
                ControlMsg::Pause | ControlMsg::Resume => fallback.preroll = None,
                ControlMsg::Flush | ControlMsg::Stop => {
                    fallback.since_keyframe.clear();
                    fallback.preroll = None;
                }
                ControlMsg::CheckSeek(_) | ControlMsg::Seek(_) => {}
            }
        }
        self.line.control(msg.clone())?;
        self.pad.control(msg)
    }
}

impl Drop for VideoDecodeBin {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing what it held");
    }
}

/// Whether `error` is a hardware decoder refusing the stream, however many
/// stages have traced it on its way here.
fn refused(error: &Error) -> bool {
    match error {
        Error::Traced(traced) => refused(&traced.source),
        #[cfg(all(target_os = "windows", feature = "d3d11"))]
        Error::D3d11DecoderError(crate::elements::D3d11DecoderError::HwAccelUnavailable) => true,
        #[cfg(all(target_os = "windows", feature = "d3d12"))]
        Error::D3d12DecoderError(crate::elements::D3d12DecoderError::HwAccelUnavailable) => true,
        #[cfg(feature = "cuda")]
        Error::CudaDecoderError(crate::elements::CudaDecoderError::HwAccelUnavailable) => true,
        _ => false,
    }
}

/// The software line: decode, and where the target is a device, change to
/// `format` at `size` and upload.
fn software(
    name: &str,
    params: ffmpeg::codec::Parameters,
    size: Option<(u32, u32)>,
    format: Option<ffmpeg::format::Pixel>,
    target: &DecodeTarget,
    threading: Option<DecodeThreading>,
) -> Result<Vec<Box<dyn Filter>>> {
    let decoder = format!("{name}-decoder");
    let decoder = match threading {
        Some(threading) => SwDecoder::with_threading(decoder, params, threading)?,
        None => SwDecoder::new(decoder, params)?,
    };
    let mut line: Vec<Box<dyn Filter>> = vec![Box::new(decoder)];
    if let Some(format) = format {
        let (width, height) = size.ok_or(VideoDecodeBinError::UnknownSize)?;
        // The size the stream was opened with: a change of layout, and of
        // size only where the stream itself changed.
        line.push(Box::new(crate::elements::SwScaler::new(
            format!("{name}-convert"),
            format,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )));
        line.extend(target.upload(format!("{name}-upload"), format, width, height)?);
    }
    Ok(line)
}

impl DecodeTarget {
    /// Where frames for this target live.
    fn domain(&self) -> MemoryDomain {
        match self {
            Self::System => MemoryDomain::System,
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            Self::D3d11 { .. } => MemoryDomain::D3d11,
            #[cfg(all(target_os = "windows", feature = "d3d12"))]
            Self::D3d12 { .. } => MemoryDomain::D3d12,
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => MemoryDomain::Cuda,
        }
    }

    /// Whether this FFmpeg build has a decoder for `codec` that runs on the
    /// target's hardware — `None` for a target with no hardware at all.
    fn decodes(&self, codec: ffmpeg::codec::Id) -> Option<bool> {
        match self {
            Self::System => {
                let _ = codec;
                None
            }
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            Self::D3d11 { .. } => Some(crate::elements::D3d11Decoder::supports(codec)),
            #[cfg(all(target_os = "windows", feature = "d3d12"))]
            Self::D3d12 { .. } => Some(crate::elements::D3d12Decoder::supports(codec)),
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => Some(crate::elements::CudaDecoder::supports(codec)),
        }
    }

    /// Whether the target has a layout that keeps alpha.
    fn keeps_alpha(&self) -> bool {
        match self {
            Self::System => true,
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            Self::D3d11 { .. } => true,
            #[cfg(all(target_os = "windows", feature = "d3d12"))]
            Self::D3d12 { .. } => false,
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => true,
        }
    }

    /// Whether the target has a scaler that brings a 10-bit 4:2:0 surface
    /// its hardware decoder makes down to NV12.
    fn converts_10bit(&self) -> bool {
        match self {
            Self::System => false,
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            Self::D3d11 { .. } => true,
            #[cfg(all(target_os = "windows", feature = "d3d12"))]
            Self::D3d12 { .. } => false,
            #[cfg(feature = "cuda")]
            Self::Cuda { .. } => true,
        }
    }

    /// That scaler, putting out NV12 at `width`x`height` — the size the
    /// stream was opened with, so a stream that changes size is scaled back
    /// to it, as on the software path. Only asked for where
    /// [`Self::converts_10bit`]; `None` on a target without one.
    #[cfg_attr(
        not(any(feature = "cuda", all(target_os = "windows", feature = "d3d11"))),
        allow(unused_variables)
    )]
    fn to_8bit(&self, name: String, width: u32, height: u32) -> Result<Option<Box<dyn Filter>>> {
        Ok(match self {
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            Self::D3d11 {
                device, context, ..
            } => Some(Box::new(crate::elements::D3d11Scaler::new(
                name,
                device,
                Arc::clone(context),
                crate::elements::D3d11ScalerFormat::Nv12,
                width,
                height,
            )?)),
            #[cfg(feature = "cuda")]
            Self::Cuda { device, .. } => Some(Box::new(crate::elements::CudaScaler::with_format(
                name,
                device,
                width,
                height,
                crate::elements::CudaScalerInterp::Bilinear,
                CudaFrameFormat::Nv12,
            ))),
            _ => None,
        })
    }

    /// The layout the software path uploads in, for a stream of `size`
    /// decoded in software because of `reason` — `None` where nothing is
    /// uploaded.
    fn software_format(
        &self,
        reason: SoftwareReason,
        size: Option<(u32, u32)>,
    ) -> Result<Option<ffmpeg::format::Pixel>> {
        if matches!(self, Self::System) {
            return Ok(None);
        }
        let (width, height) = size.ok_or(VideoDecodeBinError::UnknownSize)?;
        // NV12 is half-resolution chroma, so a surface of it cannot have an
        // odd side; BGRA can, and is also what keeps alpha.
        let odd = width % 2 != 0 || height % 2 != 0;
        if reason == SoftwareReason::Alpha || odd {
            if self.keeps_alpha() {
                return Ok(Some(ffmpeg::format::Pixel::BGRA));
            }
            if odd {
                return Err(VideoDecodeBinError::OddSize(width, height).into());
            }
        }
        Ok(Some(ffmpeg::format::Pixel::NV12))
    }

    fn hardware_decoder(
        &self,
        name: String,
        params: ffmpeg::codec::Parameters,
    ) -> Result<Box<dyn Filter>> {
        Ok(match self {
            // `choose` never picks the hardware for a target without any.
            Self::System => Box::new(SwDecoder::new(name, params)?),
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            Self::D3d11 {
                device,
                downstream_hw_frames,
                ..
            } => Box::new(crate::elements::D3d11Decoder::new(
                name,
                params,
                device,
                *downstream_hw_frames,
            )?),
            #[cfg(all(target_os = "windows", feature = "d3d12"))]
            Self::D3d12 { device } => {
                Box::new(crate::elements::D3d12Decoder::new(name, params, device)?)
            }
            #[cfg(feature = "cuda")]
            Self::Cuda {
                device,
                downstream_hw_frames,
            } => Box::new(crate::elements::CudaDecoder::new(
                name,
                params,
                device,
                *downstream_hw_frames,
            )?),
        })
    }

    /// The upload onto the target's device — `None` for system memory,
    /// which is where the pictures already are.
    ///
    /// `format` is fixed at construction only for CUDA; `D3d11Upload` takes
    /// NV12 and BGRA alike, frame by frame, and `D3d12Upload` has NV12 only.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn upload(
        &self,
        name: String,
        format: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
    ) -> Result<Option<Box<dyn Filter>>> {
        Ok(match self {
            Self::System => None,
            #[cfg(all(target_os = "windows", feature = "d3d11"))]
            Self::D3d11 { device, .. } => Some(Box::new(crate::elements::D3d11Upload::new(
                name, device, width, height,
            ))),
            #[cfg(all(target_os = "windows", feature = "d3d12"))]
            Self::D3d12 { device } => Some(Box::new(crate::elements::D3d12Upload::new(
                name, device, width, height,
            )?)),
            #[cfg(feature = "cuda")]
            Self::Cuda { device, .. } => {
                let format = if format == ffmpeg::format::Pixel::BGRA {
                    CudaFrameFormat::Bgra
                } else {
                    CudaFrameFormat::Nv12
                };
                Some(Box::new(crate::elements::CudaUpload::new(
                    name, device, format, width, height,
                )?))
            }
        })
    }
}

/// What a stream's parameters say that the choice depends on.
struct Stream {
    codec: ffmpeg::codec::Id,
    /// `None` where the parameters do not say — a stream described only by
    /// its session, before a frame of it has been decoded.
    format: Option<ffmpeg::format::Pixel>,
    size: Option<(u32, u32)>,
}

impl Stream {
    fn of(params: &ffmpeg::codec::Parameters) -> Result<Self> {
        let context = ffmpeg::codec::context::Context::from_parameters(params.clone())?;
        let medium = context.medium();
        if medium != ffmpeg::media::Type::Video {
            return Err(VideoDecodeBinError::NotVideo(medium).into());
        }
        let codec = context.id();
        let video = context.decoder().video()?;
        let format = Some(video.format()).filter(|format| *format != ffmpeg::format::Pixel::None);
        let size = Some((video.width(), video.height())).filter(|(w, h)| *w > 0 && *h > 0);
        Ok(Self {
            codec,
            format,
            size,
        })
    }
}

/// The choice made at `open` — see [`decide`], which is this with the
/// target's answers already read.
fn choose(stream: &Stream, target: &DecodeTarget) -> DecodePath {
    decide(
        stream,
        target.decodes(stream.codec),
        target.keeps_alpha(),
        target.converts_10bit(),
    )
}

/// The choice itself, apart from any device: no hardware at all first; then
/// alpha where the target can keep it, since it decides the output layout as
/// well as the path; then whether the hardware has a decoder; then whether
/// what it would decode to is a layout anything downstream reads, or one the
/// target can bring down to that after it.
fn decide(
    stream: &Stream,
    hardware_decodes: Option<bool>,
    keeps_alpha: bool,
    converts_10bit: bool,
) -> DecodePath {
    let Some(hardware_decodes) = hardware_decodes else {
        return DecodePath::Software(SoftwareReason::SystemMemory);
    };
    if keeps_alpha && stream.format.is_some_and(has_alpha) {
        return DecodePath::Software(SoftwareReason::Alpha);
    }
    if !hardware_decodes {
        return DecodePath::Software(SoftwareReason::NoHardwareDecoder(stream.codec));
    }
    match stream.format {
        // A stream that does not say is left to the hardware, which is what
        // decoding it straight onto the device always did — and if the
        // hardware refuses it, the bin replaces it then.
        Some(format) if !is_8bit_420(format) && !(converts_10bit && is_10bit_420(format)) => {
            DecodePath::Software(SoftwareReason::PixelFormat(format))
        }
        _ => DecodePath::Hardware,
    }
}

/// The layouts a hardware decoder turns into an NV12 surface.
fn is_8bit_420(format: ffmpeg::format::Pixel) -> bool {
    use ffmpeg::format::Pixel;
    matches!(format, Pixel::YUV420P | Pixel::YUVJ420P | Pixel::NV12)
}

/// The layouts a hardware decoder turns into a P010 surface: NV12's own
/// layout at 16 bits a sample, 10 of them used.
fn is_10bit_420(format: ffmpeg::format::Pixel) -> bool {
    use ffmpeg::format::Pixel;
    matches!(
        format,
        Pixel::YUV420P10LE | Pixel::YUV420P10BE | Pixel::P010LE | Pixel::P010BE
    )
}

/// Whether `format` has an alpha channel.
fn has_alpha(format: ffmpeg::format::Pixel) -> bool {
    // SAFETY: a lookup in libavutil's static table of descriptors, which
    // answers null for a format it does not know.
    unsafe {
        let descriptor = ffmpeg::ffi::av_pix_fmt_desc_get(format.into());
        !descriptor.is_null()
            && (*descriptor).flags & (ffmpeg::ffi::AV_PIX_FMT_FLAG_ALPHA as u64) != 0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::elements::AppSink;

    fn stream(codec: ffmpeg::codec::Id, format: Option<ffmpeg::format::Pixel>) -> Stream {
        Stream {
            codec,
            format,
            size: Some((64, 64)),
        }
    }

    /// The choice alone, in the order it is made: no hardware first; alpha
    /// wins where the target keeps it, even where the hardware has a
    /// decoder, and is no reason where it does not; a missing decoder comes
    /// before the layout; 10-bit 4:2:0 goes to the hardware only where the
    /// target can bring it down to 8 bits after it, and 4:2:2 never; and a
    /// stream that does not say its layout is left to the hardware.
    #[test]
    fn the_path_follows_the_target_then_alpha_then_the_decoder_then_the_layout() {
        use ffmpeg::codec::Id;
        use ffmpeg::format::Pixel;

        let prores = stream(Id::PRORES, Some(Pixel::YUVA444P10LE));
        assert_eq!(
            decide(&prores, None, true, true),
            DecodePath::Software(SoftwareReason::SystemMemory)
        );
        assert_eq!(
            decide(&prores, Some(true), true, true),
            DecodePath::Software(SoftwareReason::Alpha)
        );
        assert_eq!(
            decide(&prores, Some(false), false, true),
            DecodePath::Software(SoftwareReason::NoHardwareDecoder(Id::PRORES)),
            "alpha a target cannot keep is no reason of its own"
        );
        assert_eq!(
            decide(
                &stream(Id::MJPEG, Some(Pixel::YUVJ420P)),
                Some(false),
                true,
                true
            ),
            DecodePath::Software(SoftwareReason::NoHardwareDecoder(Id::MJPEG))
        );
        let hevc10 = stream(Id::HEVC, Some(Pixel::YUV420P10LE));
        assert_eq!(
            decide(&hevc10, Some(true), true, true),
            DecodePath::Hardware
        );
        assert_eq!(
            decide(&hevc10, Some(true), false, false),
            DecodePath::Software(SoftwareReason::PixelFormat(Pixel::YUV420P10LE)),
            "nothing to bring it down to 8 bits"
        );
        assert_eq!(
            decide(
                &stream(Id::H264, Some(Pixel::YUV422P)),
                Some(true),
                true,
                true
            ),
            DecodePath::Software(SoftwareReason::PixelFormat(Pixel::YUV422P))
        );
        assert_eq!(
            decide(
                &stream(Id::HEVC, Some(Pixel::YUV422P10LE)),
                Some(true),
                true,
                true
            ),
            DecodePath::Software(SoftwareReason::PixelFormat(Pixel::YUV422P10LE))
        );
        assert_eq!(
            decide(
                &stream(Id::H264, Some(Pixel::YUV420P)),
                Some(true),
                true,
                false
            ),
            DecodePath::Hardware
        );
        assert_eq!(
            decide(&stream(Id::H264, None), Some(true), true, false),
            DecodePath::Hardware
        );
    }

    type Frames = Vec<Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>>>;

    /// Wires `bin`, then `after` if any, into a collector.
    fn collect(bin: &mut VideoDecodeBin, after: Option<Box<dyn Filter>>) -> Arc<Mutex<Frames>> {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let collected = Arc::clone(&frames);
        let collector: Box<dyn Sink> = Box::new(AppSink::new("out", move |buffer| {
            if let MediaBuffer::Video(frame) = buffer {
                collected.lock().unwrap().push(frame);
            }
            Ok(())
        }));
        let downstream = match after {
            Some(mut after) => {
                after.src_pads()[0].link(collector);
                Box::new(after) as Box<dyn Sink>
            }
            None => collector,
        };
        bin.src_pads()[0].link(downstream);
        frames
    }

    /// Runs `bin`, then `after` if any, over `packets` and an end of stream,
    /// and answers every picture that comes out of the end.
    fn run(
        mut bin: VideoDecodeBin,
        after: Option<Box<dyn Filter>>,
        packets: Vec<ffmpeg::Packet>,
    ) -> Result<Frames> {
        let frames = collect(&mut bin, after);
        for packet in packets {
            bin.consume(MediaBuffer::Packet(Arc::new(packet)))?;
        }
        bin.consume(MediaBuffer::Eos)?;
        drop(bin);
        Ok(std::mem::take(&mut *frames.lock().unwrap()))
    }

    /// H.264 through the one encoder this project treats as always present
    /// — see `test_support::synthesize`.
    fn h264() -> (ffmpeg::codec::Parameters, Vec<ffmpeg::Packet>) {
        crate::test_support::try_encoded_packets(
            "libopenh264",
            ffmpeg::format::Pixel::YUV420P,
            (64, 64),
            90,
        )
        .expect("OpenH264 is always present")
    }

    #[test]
    fn a_stream_that_is_not_video_is_refused() {
        crate::init().unwrap();
        let mut params = ffmpeg::codec::Parameters::new();
        // SAFETY: `params` owns a live, freshly allocated AVCodecParameters.
        unsafe {
            let raw = params.as_mut_ptr();
            (*raw).codec_type = ffmpeg::ffi::AVMediaType::AVMEDIA_TYPE_AUDIO;
            (*raw).codec_id = ffmpeg::ffi::AVCodecID::AV_CODEC_ID_AAC;
        }
        let error = VideoDecodeBin::open("audio", params, DecodeTarget::System, None)
            .err()
            .expect("an audio stream has no video decode path");
        assert!(
            matches!(
                error,
                Error::VideoDecodeBinError(VideoDecodeBinError::NotVideo(
                    ffmpeg::media::Type::Audio
                ))
            ),
            "{error}"
        );
    }

    /// System memory is the software decoder alone, putting out what the
    /// stream decodes to — which is what makes the bin usable in a build
    /// with no GPU backend at all.
    #[test]
    fn system_memory_is_the_software_decoder_alone() {
        let (params, packets) = h264();
        let sent = packets.len();
        let bin = VideoDecodeBin::open("h264", params, DecodeTarget::System, None).unwrap();
        assert_eq!(
            bin.path(),
            DecodePath::Software(SoftwareReason::SystemMemory)
        );
        assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::YUV420P));

        let frames = run(bin, None, packets).expect("H.264 decodes");
        assert_eq!(frames.len(), sent);
        assert!(
            frames
                .iter()
                .all(|frame| frame.format() == ffmpeg::format::Pixel::YUV420P)
        );
    }

    /// What the bin is given for threading reaches the software decoder it
    /// builds: on four threads for throughput it holds pictures back until
    /// the end, where with nothing given it decodes on one and holds none.
    #[test]
    fn threading_reaches_the_software_decoder() {
        let held_back = |threading| {
            let (params, packets) = h264();
            let sent = packets.len();
            let mut bin =
                VideoDecodeBin::open("h264", params, DecodeTarget::System, threading).unwrap();
            let frames = collect(&mut bin, None);
            for packet in packets {
                bin.consume(MediaBuffer::Packet(Arc::new(packet))).unwrap();
            }
            let before = frames.lock().unwrap().len();
            bin.consume(MediaBuffer::Eos).unwrap();
            assert_eq!(
                frames.lock().unwrap().len(),
                sent,
                "every picture comes out"
            );
            sent - before
        };
        assert_eq!(held_back(None), 0, "one thread holds nothing back");
        let threading = DecodeThreading {
            threads: std::num::NonZeroU32::new(4),
            kind: crate::elements::DecodeThreadKind::Frame,
        };
        assert!(
            held_back(Some(threading)) > 0,
            "four for several pictures at once do"
        );
    }

    /// Parameters that no longer say what layout the pictures are in — a
    /// stream described only by its session — so the choice cannot see a
    /// layout the hardware would refuse and leaves it to the hardware.
    #[cfg(any(
        feature = "cuda",
        all(target_os = "windows", any(feature = "d3d11", feature = "d3d12"))
    ))]
    fn without_layout(mut params: ffmpeg::codec::Parameters) -> ffmpeg::codec::Parameters {
        // SAFETY: `params` owns a live AVCodecParameters; `-1` is
        // `AV_PIX_FMT_NONE`, what an undescribed stream carries.
        unsafe {
            (*params.as_mut_ptr()).format = -1;
        }
        params
    }

    /// Opens the bin, or answers `None` after saying why on a machine whose
    /// adapter makes a device and does no video decoding — a hosted CI
    /// runner's software adapter, which the decoders' and uploads' own tests
    /// skip on too. Anything else is the failure it looks like.
    #[cfg(any(
        feature = "cuda",
        all(target_os = "windows", any(feature = "d3d11", feature = "d3d12"))
    ))]
    fn open_or_skip(
        name: &str,
        params: ffmpeg::codec::Parameters,
        target: DecodeTarget,
    ) -> Option<VideoDecodeBin> {
        fn no_video_device(error: &Error) -> bool {
            match error {
                Error::Traced(traced) => no_video_device(&traced.source),
                #[cfg(all(target_os = "windows", feature = "d3d11"))]
                Error::D3d11DecoderError(crate::elements::D3d11DecoderError::HwDeviceInit(_)) => {
                    true
                }
                #[cfg(all(target_os = "windows", feature = "d3d12"))]
                Error::D3d12DecoderError(crate::elements::D3d12DecoderError::HwDeviceInit(_)) => {
                    true
                }
                #[cfg(all(target_os = "windows", feature = "d3d12"))]
                Error::D3d12UploadError(crate::elements::D3d12UploadError::HwDeviceInit(_)) => true,
                _ => false,
            }
        }
        match VideoDecodeBin::open(name, params, target, None) {
            Ok(bin) => Some(bin),
            Err(error) if no_video_device(&error) => {
                eprintln!("skipping: this device does not do video decoding ({error})");
                None
            }
            Err(error) => panic!("{name} did not open: {error}"),
        }
    }

    /// VP9 profile 1 — 8-bit 4:4:4 — which no hardware decoder here has a
    /// profile for, with its layout hidden so the bin tries the hardware.
    #[cfg(any(
        feature = "cuda",
        all(target_os = "windows", any(feature = "d3d11", feature = "d3d12"))
    ))]
    fn refused_by_hardware() -> Option<(ffmpeg::codec::Parameters, Vec<ffmpeg::Packet>)> {
        let (params, packets) = crate::test_support::try_encoded_packets(
            "libvpx-vp9",
            ffmpeg::format::Pixel::YUV444P,
            (64, 64),
            90,
        )?;
        Some((without_layout(params), packets))
    }

    /// 10-bit HEVC, every sample 90 at 8 bits, from NVENC — the one 10-bit
    /// HEVC encoder an FFmpeg build here is likely to have.
    #[cfg(any(feature = "cuda", all(target_os = "windows", feature = "d3d11")))]
    fn ten_bit_hevc() -> Option<(ffmpeg::codec::Parameters, Vec<ffmpeg::Packet>)> {
        crate::test_support::try_encoded_packets(
            "hevc_nvenc",
            ffmpeg::format::Pixel::P010LE,
            (256, 144),
            90,
        )
    }

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    mod d3d11 {
        use std::time::Duration;

        use super::*;
        use crate::{
            control::PrerollContext, elements::D3d11Download, test_support::try_encoded_packets,
        };

        fn target(
            device: &ID3D11Device,
            context: &Arc<Mutex<ID3D11DeviceContext>>,
        ) -> DecodeTarget {
            DecodeTarget::D3d11 {
                device: device.clone(),
                context: Arc::clone(context),
                downstream_hw_frames: 4,
            }
        }

        /// A stream the hardware takes goes to it, exactly as a
        /// `D3d11Decoder` on its own would.
        #[test]
        fn h264_decodes_on_the_gpu() {
            let Some((device, context)) = crate::test_support::try_d3d11_device() else {
                return;
            };
            let (params, packets) = h264();
            let sent = packets.len();
            let Some(bin) = open_or_skip("h264", params, target(&device, &context)) else {
                return;
            };
            assert_eq!(bin.path(), DecodePath::Hardware);
            assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::NV12));
            let handle = bin.handle();

            let frames = run(bin, None, packets).expect("an H.264 stream decodes");
            assert_eq!(frames.len(), sent);
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.format() == ffmpeg::format::Pixel::D3D11)
            );
            assert_eq!(handle.path(), DecodePath::Hardware, "and it stayed there");
        }

        /// Motion JPEG has no D3D11VA decoder, so it is decoded in software
        /// and still reaches the device, as NV12.
        #[test]
        fn a_codec_without_a_hardware_decoder_still_reaches_the_gpu() {
            let Some((device, context)) = crate::test_support::try_d3d11_device() else {
                return;
            };
            let Some((params, packets)) =
                try_encoded_packets("mjpeg", ffmpeg::format::Pixel::YUVJ420P, (64, 48), 90)
            else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("mjpeg", params, target(&device, &context)) else {
                return;
            };
            assert_eq!(
                bin.path(),
                DecodePath::Software(SoftwareReason::NoHardwareDecoder(ffmpeg::codec::Id::MJPEG))
            );
            assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::NV12));

            let frames = run(bin, None, packets).expect("a Motion JPEG stream decodes");
            assert_eq!(frames.len(), sent, "every picture comes out");
            for frame in &frames {
                assert_eq!(frame.format(), ffmpeg::format::Pixel::D3D11);
                assert_eq!((frame.width(), frame.height()), (64, 48));
            }
        }

        /// A side NV12 cannot have is uploaded as BGRA instead of failing
        /// at the upload.
        #[test]
        fn an_odd_size_is_uploaded_as_bgra() {
            let Some((device, context)) = crate::test_support::try_d3d11_device() else {
                return;
            };
            let Some((params, packets)) =
                try_encoded_packets("mjpeg", ffmpeg::format::Pixel::YUVJ444P, (65, 49), 90)
            else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("odd", params, target(&device, &context)) else {
                return;
            };
            assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::BGRA));
            let frames = run(bin, None, packets).expect("an odd-sized stream decodes");
            assert_eq!(frames.len(), sent);
            assert!(
                frames
                    .iter()
                    .all(|frame| (frame.width(), frame.height()) == (65, 49))
            );
        }

        /// ProRes 4444 with half-transparent pictures: decoded in software,
        /// uploaded as BGRA, and the alpha is still there when it is read
        /// back off the GPU.
        #[test]
        fn alpha_survives_onto_the_gpu() {
            let Some((device, context)) = crate::test_support::try_d3d11_device() else {
                return;
            };
            // Every 10-bit sample is 0x0202 = 514 of 1023: mid grey, neutral
            // chroma, and an alpha of about a half.
            let Some((params, packets)) = try_encoded_packets(
                "prores_ks",
                ffmpeg::format::Pixel::YUVA444P10LE,
                (64, 48),
                2,
            ) else {
                return;
            };
            let Some(bin) = open_or_skip("prores", params, target(&device, &context)) else {
                return;
            };
            assert_eq!(bin.path(), DecodePath::Software(SoftwareReason::Alpha));
            assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::BGRA));

            let download = D3d11Download::new("read-back", &device, context, 64, 48).unwrap();
            let frames = run(bin, Some(Box::new(download)), packets).expect("ProRes decodes");
            let frame = frames.last().expect("a picture came out");
            assert_eq!(frame.format(), ffmpeg::format::Pixel::BGRA);
            let alpha = frame.data(0)[3];
            assert!(
                (120..=136).contains(&alpha),
                "alpha {alpha} is not the half it was encoded with"
            );
        }

        /// 10-bit HEVC is decoded by D3D11VA, which hands on P010, and
        /// brought down to NV12 on the GPU after it: every picture comes
        /// out as an NV12 texture of the stream's own size.
        #[test]
        fn ten_bit_hevc_decodes_on_the_gpu_and_comes_out_as_nv12() {
            use windows::{
                Win32::Graphics::{
                    Direct3D11::{D3D11_TEXTURE2D_DESC, ID3D11Texture2D},
                    Dxgi::Common::DXGI_FORMAT_NV12,
                },
                core::Interface,
            };

            let Some((device, context)) = crate::test_support::try_d3d11_device() else {
                return;
            };
            let Some((params, packets)) = ten_bit_hevc() else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("hevc10", params, target(&device, &context)) else {
                return;
            };
            assert_eq!(bin.path(), DecodePath::Hardware);
            assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::NV12));
            let handle = bin.handle();

            let frames = run(bin, None, packets).expect("HEVC decodes");
            assert_eq!(handle.path(), DecodePath::Hardware, "and it stayed there");
            assert_eq!(frames.len(), sent, "every picture comes out");
            for frame in &frames {
                let (texture, _) = crate::platform::windows::d3d11va::d3d11va_texture(frame)
                    .expect("a D3D11 frame");
                // SAFETY: the frame is alive for the whole borrow and holds a
                // reference to its texture; nothing here keeps the borrow.
                let texture =
                    unsafe { ID3D11Texture2D::from_raw_borrowed(&texture) }.expect("a texture");
                let mut desc = D3D11_TEXTURE2D_DESC::default();
                // SAFETY: `desc` is a live out-parameter.
                unsafe { texture.GetDesc(&mut desc) };
                assert_eq!(desc.Format, DXGI_FORMAT_NV12);
                assert_eq!((frame.width(), frame.height()), (256, 144));
            }
        }

        /// With the layout hidden, the bin opens the hardware for VP9
        /// profile 1; the hardware refuses at the first picture; and every
        /// picture still comes out, on the GPU, in the layout promised, with
        /// the refusal readable from the handle.
        #[test]
        fn a_stream_the_hardware_refuses_is_decoded_in_software_instead() {
            let Some((device, context)) = crate::test_support::try_d3d11_device() else {
                return;
            };
            let Some((params, packets)) = refused_by_hardware() else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("vp9", params, target(&device, &context)) else {
                return;
            };
            assert_eq!(bin.path(), DecodePath::Hardware, "nothing said otherwise");
            let handle = bin.handle();

            let frames = run(bin, None, packets).expect("the stream decodes after all");
            assert_eq!(
                handle.path(),
                DecodePath::Software(SoftwareReason::HardwareRefused)
            );
            assert_eq!(frames.len(), sent, "every picture comes out, once");
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.format() == ffmpeg::format::Pixel::D3D11)
            );
        }

        /// A seek's preroll that was under way when the hardware refused is
        /// given to the software line too: only the picture the seek asked
        /// for comes out, rather than every one decoded to reach it.
        #[test]
        fn a_replacement_keeps_the_preroll_it_replaced() {
            let Some((device, context)) = crate::test_support::try_d3d11_device() else {
                return;
            };
            let Some((params, packets)) = refused_by_hardware() else {
                return;
            };
            let Some(mut bin) = open_or_skip("vp9", params, target(&device, &context)) else {
                return;
            };
            let frames = collect(&mut bin, None);

            // Frame 3 of five at 30 a second.
            let preroll = PrerollContext::for_seek([], Duration::from_millis(100));
            bin.control(ControlMsg::Preroll(Arc::new(preroll))).unwrap();
            for packet in packets {
                bin.consume(MediaBuffer::Packet(Arc::new(packet))).unwrap();
            }
            bin.consume(MediaBuffer::Eos).unwrap();
            assert_eq!(
                bin.path(),
                DecodePath::Software(SoftwareReason::HardwareRefused)
            );
            let pts: Vec<_> = frames
                .lock()
                .unwrap()
                .iter()
                .map(|frame| frame.pts())
                .collect();
            assert_eq!(pts, vec![Some(3)], "only the picture the seek asked for");
        }
    }

    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    mod d3d12 {
        use super::*;
        use crate::test_support::try_encoded_packets;

        fn target(device: &ID3D12Device) -> DecodeTarget {
            DecodeTarget::D3d12 {
                device: device.clone(),
            }
        }

        #[test]
        fn h264_decodes_on_the_gpu() {
            let Some(device) = crate::test_support::try_d3d12_device() else {
                return;
            };
            let (params, packets) = h264();
            let sent = packets.len();
            let Some(bin) = open_or_skip("h264", params, target(&device)) else {
                return;
            };
            assert_eq!(bin.path(), DecodePath::Hardware);
            let frames = run(bin, None, packets).expect("an H.264 stream decodes");
            assert_eq!(frames.len(), sent);
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.format() == ffmpeg::format::Pixel::D3D12)
            );
        }

        /// D3D12 frames here are NV12 alone, so ProRes 4444 is decoded in
        /// software for want of a hardware decoder — not for its alpha,
        /// which this target has nowhere to keep — and arrives as NV12.
        #[test]
        fn alpha_is_not_kept_and_is_no_reason_of_its_own() {
            let Some(device) = crate::test_support::try_d3d12_device() else {
                return;
            };
            let Some((params, packets)) = try_encoded_packets(
                "prores_ks",
                ffmpeg::format::Pixel::YUVA444P10LE,
                (64, 48),
                2,
            ) else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("prores", params, target(&device)) else {
                return;
            };
            assert_eq!(
                bin.path(),
                DecodePath::Software(SoftwareReason::NoHardwareDecoder(ffmpeg::codec::Id::PRORES))
            );
            assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::NV12));
            let frames = run(bin, None, packets).expect("ProRes decodes");
            assert_eq!(frames.len(), sent);
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.format() == ffmpeg::format::Pixel::D3D12)
            );
        }

        /// An odd side on the software path has no layout on D3D12, and is
        /// refused when the bin is opened rather than at the upload.
        #[test]
        fn an_odd_size_in_software_is_refused() {
            let Some(device) = crate::test_support::try_d3d12_device() else {
                return;
            };
            let Some((params, _)) =
                try_encoded_packets("mjpeg", ffmpeg::format::Pixel::YUVJ444P, (65, 49), 90)
            else {
                return;
            };
            let error = VideoDecodeBin::open("odd", params, target(&device), None)
                .err()
                .expect("NV12 has no odd side");
            assert!(
                matches!(
                    error,
                    Error::VideoDecodeBinError(VideoDecodeBinError::OddSize(65, 49))
                ),
                "{error}"
            );
        }

        #[test]
        fn a_stream_the_hardware_refuses_is_decoded_in_software_instead() {
            let Some(device) = crate::test_support::try_d3d12_device() else {
                return;
            };
            let Some((params, packets)) = refused_by_hardware() else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("vp9", params, target(&device)) else {
                return;
            };
            let handle = bin.handle();
            let frames = run(bin, None, packets).expect("the stream decodes after all");
            assert_eq!(
                handle.path(),
                DecodePath::Software(SoftwareReason::HardwareRefused)
            );
            assert_eq!(frames.len(), sent);
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.format() == ffmpeg::format::Pixel::D3D12)
            );
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use crate::{elements::CudaDownload, test_support::try_encoded_packets};

        fn target(device: &CudaDevice) -> DecodeTarget {
            DecodeTarget::Cuda {
                device: device.clone(),
                downstream_hw_frames: 4,
            }
        }

        /// The same on CUDA: ProRes has no NVDEC decoder, and its alpha
        /// arrives in a BGRA CUDA frame.
        #[test]
        fn alpha_survives_onto_cuda() {
            let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
                return;
            };
            let Some((params, packets)) = try_encoded_packets(
                "prores_ks",
                ffmpeg::format::Pixel::YUVA444P10LE,
                (64, 48),
                2,
            ) else {
                return;
            };
            let Some(bin) = open_or_skip("prores", params, target(&device)) else {
                return;
            };
            assert_eq!(bin.path(), DecodePath::Software(SoftwareReason::Alpha));

            let download = CudaDownload::new("read-back", &device, CudaFrameFormat::Bgra, 64, 48);
            let frames = run(bin, Some(Box::new(download)), packets).expect("ProRes decodes");
            let frame = frames.last().expect("a picture came out");
            let alpha = frame.data(0)[3];
            assert!(
                (120..=136).contains(&alpha),
                "alpha {alpha} is not the half it was encoded with"
            );
        }

        /// The same on CUDA: NVDEC hands on P010, and a `CudaScaler` brings
        /// it down to NV12.
        #[test]
        fn ten_bit_hevc_decodes_on_nvdec_and_comes_out_as_nv12() {
            let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
                return;
            };
            let Some((params, packets)) = ten_bit_hevc() else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("hevc10", params, target(&device)) else {
                return;
            };
            assert_eq!(bin.path(), DecodePath::Hardware);
            assert_eq!(bin.output_format(), Some(ffmpeg::format::Pixel::NV12));
            let handle = bin.handle();

            let download = CudaDownload::new("read-back", &device, CudaFrameFormat::Nv12, 256, 144);
            let frames = run(bin, Some(Box::new(download)), packets).expect("HEVC decodes");
            assert_eq!(handle.path(), DecodePath::Hardware, "and it stayed there");
            assert_eq!(frames.len(), sent, "every picture comes out");
            for frame in &frames {
                assert_eq!(frame.format(), ffmpeg::format::Pixel::NV12);
                let luma = frame.data(0)[frame.stride(0) * 72 + 128];
                assert!(luma.abs_diff(90) <= 2, "luma {luma}, not 90");
            }
        }

        /// NVDEC has no VP9 4:4:4 either: refused, replaced, and every
        /// picture arrives as a CUDA frame.
        #[test]
        fn a_stream_nvdec_refuses_is_decoded_in_software_instead() {
            let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
                return;
            };
            let Some((params, packets)) = refused_by_hardware() else {
                return;
            };
            let sent = packets.len();
            let Some(bin) = open_or_skip("vp9", params, target(&device)) else {
                return;
            };
            let handle = bin.handle();
            let frames = run(bin, None, packets).expect("the stream decodes after all");
            assert_eq!(
                handle.path(),
                DecodePath::Software(SoftwareReason::HardwareRefused)
            );
            assert_eq!(frames.len(), sent);
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.format() == ffmpeg::format::Pixel::CUDA)
            );
        }
    }
}
