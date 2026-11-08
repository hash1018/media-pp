//! [`CudaOrtImageFilter`]: a model that makes a picture of each picture —
//! one that enlarges it, cleans it or brightens it — run on CUDA pictures
//! without them leaving the GPU.

use std::{path::Path, sync::Arc};

use ffmpeg_next::{self as ffmpeg, ffi};
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::{DynTensorValueType, Shape, TensorElementType},
};

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Attached, ElementType, Identity},
    elements::CudaUploadError,
    error::Result,
    frame_size::ForSize,
    orientation::Orientation,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        driver::{
            BgraSurface, CUdeviceptr, CudaDriver, CudaTensor, Fit, FitKernels, Nv12Surface, Unfit,
            YuvToBgra,
        },
        frame::{self, CudaSurfaces, create_hw_frames_ctx},
    },
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    repeat::{PerFrameTransform, RepeatedOutput},
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::{ChannelOrder, ModelInput, OrtError};

/// The side of the square piece a model of open size is run on. Real-ESRGAN
/// x4plus made 1024 by 1024 of one in 2.1 GB of an RTX 3050's memory, and a
/// 1080p picture in 27 s, against 33 s in its own 128 by 128 pieces, whose
/// overlaps are a larger share of them.
const TILE: u32 = 256;
/// How far each piece reaches into its neighbours, in the picture's
/// pixels: a model draws the edge of what it is given worst, and the edge
/// of a piece is not the picture's, so only its middle is kept.
const OVERLAP: u32 = 16;
/// The most pieces run at once by a model that takes any number: one.
/// Four of Real-ESRGAN x4plus's took all 8 GB of an RTX 3050 and spilled
/// into the system's memory, a 1080p picture taking 339 s rather than 27.
const MOST_AT_ONCE: usize = 1;

/// How a [`CudaOrtImageFilter`] runs.
#[derive(Debug, Clone)]
pub struct CudaOrtImageFilterOptions {
    /// The order of the model's planes and how a value from 0 to 1 is
    /// scaled for it. Its output is read back the same way, undone: a
    /// model that takes RGB from -1 to 1 is taken to make RGB from -1 to 1.
    pub values: ModelInput,
    /// What it makes: [`CudaFrameFormat::Bgra`], the default, of NV12, BGRA
    /// or HDR P010 brought to SDR; or [`CudaFrameFormat::P010`] of P010
    /// alone, in ten bits and the picture's own colour — HDR stays HDR, the
    /// model handed its samples as they are coded. NV12 is refused:
    /// [`CudaConverter`](crate::elements::CudaConverter) makes it of BGRA.
    pub output: CudaFrameFormat,
    /// Where TensorRT keeps the engine it builds for this model and GPU,
    /// as [`CudaOrtDetectorOptions::engine_cache`](crate::elements::CudaOrtDetectorOptions::engine_cache)
    /// says; `None` is the user's cache directory.
    #[cfg(feature = "ort-tensorrt")]
    pub engine_cache: Option<std::path::PathBuf>,
}

impl Default for CudaOrtImageFilterOptions {
    /// RGB from 0 to 1, made into BGRA.
    fn default() -> Self {
        Self {
            values: ModelInput::default(),
            output: CudaFrameFormat::Bgra,
            #[cfg(feature = "ort-tensorrt")]
            engine_cache: None,
        }
    }
}

/// Makes each CUDA picture into the picture an ONNX model makes of it — a
/// model that enlarges pictures, cleans them or brightens them, one picture
/// at a time — without the pictures leaving the GPU.
///
/// Any model of one RGB picture in, `[batch, 3, height, width]` floats, and
/// one out, the same or a whole number of times larger, is run the same
/// way: the model file is what makes it one thing or another. Its output is
/// that many times the size of its input, BGRA — which [`CudaConverter`]
/// makes NV12 for an encoder that takes that — or, where
/// [`CudaOrtImageFilterOptions::output`] says, P010 of P010 in ten bits.
/// Needs the `ort-cuda` feature.
///
/// [`CudaConverter`]: crate::elements::CudaConverter
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{
///     CudaConverter, CudaDevice, CudaFrameFormat, CudaOrtImageFilter,
///     CudaOrtImageFilterOptions,
/// };
/// # use media_pp::{element::Context, elements::AppSink, pipeline::DetachedBranch};
/// # fn chain(ctx: &std::sync::Arc<Context>) -> media_pp::Result<DetachedBranch> {
/// # let sink = AppSink::new("pictures", |_buf| Ok(()));
///
/// let device = CudaDevice::new()?;
/// let options = CudaOrtImageFilterOptions::default();
/// let upscale = CudaOrtImageFilter::new("upscale", &device, "realesrgan-x4.onnx", options)?;
/// let to_nv12 = CudaConverter::new("to-nv12", &device, CudaFrameFormat::Nv12)?;
/// let branch = ctx.branch().pipe(upscale).pipe(to_nv12).to(sink)?;
/// # Ok(branch)
/// # }
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// A picture is run through the model in square pieces — 256 by 256 for a
/// model that takes any size, its own size for one that does not — each
/// reaching 16 pixels into its neighbours, of which only the middle is
/// kept, as Real-ESRGAN's own tiling does: a model draws the edge of what it
/// is given worst. A picture smaller than a piece is run in one, the rest
/// of it grey. One piece runs at a time, unless the model takes a fixed
/// number of them: what a model of this kind keeps while it runs grows
/// with every piece, and four of Real-ESRGAN's fill an 8 GB GPU. Each is
/// read into the model's input on the GPU, as the detectors
/// read a picture, run there, and its output written into the picture
/// where it came from. How many times larger the output is comes from the
/// model's first run; one that is not a whole number of times larger, the
/// same across and down, is refused.
///
/// Making BGRA, it takes NV12, BGRA or HDR P010 CUDA pictures from the same
/// [`CudaDevice`] as the rest of the pipeline. An NV12 picture is read as
/// RGB by its own colour tags; an HDR one is brought to SDR first, as the
/// detectors read it, and its output tagged BT.709. What it makes is
/// full-range RGB, tagged so.
///
/// Making P010, it takes P010 alone, SDR or HDR, and is handed its samples
/// as they are coded — PQ or HLG, BT.2020 — read to RGB by its own matrix
/// and range in ten bits, and written back by the same, each block of four
/// pixels the chroma of its mean colour: a picture the model leaves alone
/// comes back the same samples, to a step. Its colour tags, HDR's among
/// them, go on as they were. A model trained on SDR pictures is handed HDR
/// as it was coded, not as light, which is what keeps it in ten bits.
///
/// Either way its timestamps, its other side data and what the picture
/// carried go on with it, and it is stored as the picture was, whichever way
/// it is shown. The same picture handed again — a live source's repeat — is
/// answered with what was made of it.
///
/// # Requirements
///
/// The `ort-cuda` feature, and at run time what
/// [`CudaOrtDetector::runtime`](crate::elements::CudaOrtDetector::runtime)
/// says it needs of the CUDA side. Built with `ort-tensorrt`, the model runs
/// through TensorRT in half precision where TensorRT can run it.
pub struct CudaOrtImageFilter(FilterStage<Filtering>);

filter_stage!(CudaOrtImageFilter);

/// What a [`CudaOrtImageFilter`] does with each picture.
///
/// The device buffer and kernels come before the driver: fields drop in
/// order, and both are freed in the driver's context, which the driver
/// releases.
struct Filtering {
    pp_log: PpLog,
    session: Session,
    input_name: String,
    output_name: String,
    values: ModelInput,
    /// BGRA, or P010 of P010.
    output: CudaFrameFormat,
    /// The size of a piece, the model's input.
    tile: (u32, u32),
    /// How many pieces `tensor` holds, and so run at once.
    capacity: usize,
    /// Whether the model takes any number of pieces at once; one that
    /// does not is handed exactly `capacity`.
    open_batch: bool,
    /// How many times larger the model's output is, from its first run.
    factor: Option<u32>,
    tensor: CudaTensor,
    kernels: FitKernels,
    /// Where an HDR picture is brought to SDR for the model; before the
    /// driver, whose context it is freed in.
    sdr: super::SdrCopy,
    driver: CudaDriver,
    device_memory: MemoryInfo<'static>,
    /// The pool output pictures are taken from, made for their size.
    hw_frames_ctx: ForSize<AvBufferRef>,
    /// Reuses the CPU-side `AVFrame` wrappers; each surface is the pool's.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// What was last made, and of what — see [`PerFrameTransform`].
    repeated: RepeatedOutput,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *mut ffi::AVHWDeviceContext,
    hw_device_ctx: Arc<AvBufferRef>,
}

// SAFETY: the session, buffer, kernels and FFmpeg buffers are used by the
// one thread transforming at a time; `device_ctx` only ever has its address
// compared, and `&mut self` on every method that touches the rest rules out
// concurrent access — the reasoning `CudaOrtEmbedder` gives for its own.
unsafe impl Send for Filtering {}

/// A piece of a picture along one side: where it starts and how long it
/// is, and which part of it is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Piece {
    start: u32,
    length: u32,
    keep_from: u32,
    keep_to: u32,
}

/// The pieces a side `length` long is run in, each `tile` long or the whole
/// side where it is shorter, reaching `overlap` into its neighbours. Every
/// pixel is kept from exactly one piece, at least `overlap` from that
/// piece's edge wherever the picture goes on past it. Starts and the ends
/// of what is kept stay even where `tile` and `length` are, so an NV12
/// piece starts on a chroma sample and a P010 one is written whole blocks
/// of it.
fn pieces(length: u32, tile: u32, overlap: u32) -> Vec<Piece> {
    if length <= tile {
        return vec![Piece {
            start: 0,
            length,
            keep_from: 0,
            keep_to: length,
        }];
    }
    let overlap = overlap.min(tile / 4);
    // Two short of twice the overlap, so that halfway into it, made even,
    // is still the overlap from either piece's edge.
    let stride = tile.saturating_sub(2 * overlap + 2) & !1;
    let mut starts = Vec::new();
    let mut start = 0;
    while start + tile < length {
        starts.push(start);
        start += stride.max(2);
    }
    starts.push(length - tile);
    let mut found = Vec::with_capacity(starts.len());
    let mut kept = 0;
    for (index, &start) in starts.iter().enumerate() {
        let keep_to = match starts.get(index + 1) {
            // Halfway into the overlap with the next, which keeps the rest.
            Some(&next) => ((next + start + tile) / 2) & !1,
            None => length,
        };
        found.push(Piece {
            start,
            length: tile,
            keep_from: kept,
            keep_to,
        });
        kept = keep_to;
    }
    found
}

impl CudaOrtImageFilter {
    /// Loads the model at `model_path` to run on `device`'s GPU — the same
    /// [`CudaDevice`] every other CUDA element in the pipeline was built
    /// from.
    ///
    /// # Errors
    ///
    /// A model that is not one RGB picture in and one out, of floats; an
    /// input of a fixed odd size, whose pieces of an NV12 picture would not
    /// start on a chroma sample; and what loading any ONNX model on CUDA
    /// can fail at.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        model_path: impl AsRef<Path>,
        options: CudaOrtImageFilterOptions,
    ) -> Result<Self> {
        let identity = Identity::new(ElementType::CudaOrtImageFilter, name);
        let pp_log = identity.pp_log().clone();
        if options.output == CudaFrameFormat::Nv12 {
            return Err(OrtError::UnsupportedOutput(options.output).into());
        }
        let path = model_path.as_ref().display().to_string();
        #[cfg(feature = "ort-tensorrt")]
        let engine_cache = options.engine_cache.as_deref();
        #[cfg(not(feature = "ort-tensorrt"))]
        let engine_cache = None;
        let (session, input, provider, linked) = super::object_model_session(
            model_path.as_ref(),
            TILE,
            MOST_AT_ONCE,
            engine_cache,
            &pp_log,
            "filtering pictures",
        )?;
        let (input_name, output_name) = match (session.inputs(), session.outputs()) {
            ([input], [output]) => {
                for (what, value) in [("input", input.dtype()), ("output", output.dtype())] {
                    if value.tensor_type() != Some(TensorElementType::Float32) {
                        return Err(OrtError::UnsupportedModel(format!(
                            "its {what} is not of 32-bit floats"
                        ))
                        .into());
                    }
                }
                (input.name().to_owned(), output.name().to_owned())
            }
            (inputs, outputs) => {
                return Err(OrtError::UnsupportedModel(format!(
                    "{} inputs and {} outputs, where a picture filter has one of each",
                    inputs.len(),
                    outputs.len()
                ))
                .into());
            }
        };
        if !input.size.0.is_multiple_of(2) || !input.size.1.is_multiple_of(2) {
            return Err(OrtError::UnsupportedModel(format!(
                "an input of {}x{}, where a picture filter's sides are even",
                input.size.0, input.size.1
            ))
            .into());
        }
        let capacity = input.batch.unwrap_or(MOST_AT_ONCE).max(1);
        let driver = CudaDriver::retain_primary().map_err(OrtError::from)?;
        let kernels = driver.fit_kernels().map_err(OrtError::from)?;
        let tensor = driver
            .batch_tensor(input.size.0, input.size.1, capacity)
            .map_err(OrtError::from)?;
        let device_memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(OrtError::from)?;
        let hw_device_ctx = device.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition; only the pointer's identity is kept, and the reference
        // held beside it keeps that identity from being reused.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *mut ffi::AVHWDeviceContext };
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, pieces of {}x{}, {} at once, {:?}, making {:?}, on {provider}; {linked}",
            input.size.0,
            input.size.1,
            capacity,
            options.values,
            options.output
        );
        Ok(Self(FilterStage::new(
            identity,
            Filtering {
                pp_log,
                session,
                input_name,
                output_name,
                values: options.values,
                output: options.output,
                tile: input.size,
                capacity,
                open_batch: input.batch.is_none(),
                factor: None,
                tensor,
                kernels,
                sdr: super::SdrCopy::default(),
                driver,
                device_memory,
                hw_frames_ctx: ForSize::new(),
                pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
                repeated: RepeatedOutput::new(),
                device_ctx,
                hw_device_ctx,
            },
        )))
    }
}

/// Where the pixels of a picture are read from.
#[derive(Clone, Copy)]
enum Source {
    Nv12(Nv12Surface),
    P010(Nv12Surface),
    Bgra(BgraSurface),
}

/// A piece across and a piece down: one run's worth of the picture.
#[derive(Clone, Copy)]
struct Patch {
    across: Piece,
    down: Piece,
}

impl Filtering {
    /// The picture the model makes of `frame`.
    fn filter(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, OrtError> {
        let ten_bit = self.output == CudaFrameFormat::P010;
        let surface = frame::validate(
            frame,
            ElementType::CudaOrtImageFilter,
            self.device_ctx,
            if ten_bit {
                CudaSurfaces::P010
            } else {
                CudaSurfaces::NV12_BGRA_OR_P010
            },
        )?;
        let hdr = !ten_bit && surface.layout == ffmpeg::format::Pixel::P010LE;
        let source = if ten_bit {
            Source::P010(Nv12Surface::from_frame(frame).ok_or(OrtError::MissingSurface)?)
        } else if hdr {
            Source::Bgra(
                self.sdr
                    .of(&self.driver, frame, ElementType::CudaOrtImageFilter)?,
            )
        } else if surface.layout == ffmpeg::format::Pixel::NV12 {
            Source::Nv12(Nv12Surface::from_frame(frame).ok_or(OrtError::MissingSurface)?)
        } else {
            Source::Bgra(BgraSurface::from_frame(frame).ok_or(OrtError::MissingSurface)?)
        };
        let colour = if ten_bit {
            YuvToBgra::of_frame_at_depth(frame, 10)
        } else {
            YuvToBgra::of_frame(frame)
        };
        // Written back, ten bits are written as they were read.
        let writing = crate::color::rgb_to_yuv_rows_of_depth(
            frame.color_space(),
            frame.color_range(),
            frame.height(),
            10,
        );
        let size = (frame.width(), frame.height());
        let across = pieces(size.0, self.tile.0, OVERLAP);
        let down = pieces(size.1, self.tile.1, OVERLAP);
        let patches: Vec<Patch> = down
            .iter()
            .flat_map(|&down| across.iter().map(move |&across| Patch { across, down }))
            .collect();

        let planes = self.values.planes();
        let (mul, add) = self.reading();
        let mut destination: Option<(UnboundObjectPoolRef<ffmpeg::frame::Video>, (u32, u32))> =
            None;
        for group in patches.chunks(self.capacity) {
            for (slot, patch) in group.iter().enumerate() {
                self.fit(source, &colour, slot, *patch)?;
            }
            let rows = if self.open_batch {
                group.len()
            } else {
                self.capacity
            };
            if self.values.order == ChannelOrder::Bgr {
                self.driver
                    .swap_planes(&self.kernels, &self.tensor, self.tile, rows)?;
            }
            if !self.values.is_unscaled() {
                self.driver.scale_planes(
                    &self.kernels,
                    &self.tensor,
                    self.tile,
                    rows,
                    self.values.scale,
                    self.values.bias,
                )?;
            }
            // The kernels ran on this driver's context; the model reads the
            // tensor on ONNX Runtime's own stream.
            self.driver.synchronize()?;
            let shape = Shape::new([
                rows as i64,
                3,
                i64::from(self.tile.1),
                i64::from(self.tile.0),
            ]);
            // SAFETY: the pointer is this element's own device allocation of
            // `capacity` inputs, at least `rows`, in the primary context ONNX
            // Runtime's CUDA provider also runs in; it outlives the run, the
            // tensor being dropped before this returns.
            let input = unsafe {
                super::super::device_tensor(
                    &self.device_memory,
                    self.tensor.pointer() as usize as *mut _,
                    &shape,
                )?
            };
            // The output is left where the model wrote it, on the GPU, and
            // written into the picture from there.
            let mut binding = self.session.create_binding()?;
            binding.bind_input(&self.input_name, &input)?;
            binding.bind_output_to_device(&self.output_name, &self.device_memory)?;
            let outputs = self.session.run_binding(&binding)?;
            binding.synchronize_outputs()?;
            let output = outputs[0].downcast_ref::<DynTensorValueType>()?;
            let made: Vec<usize> = output
                .shape()
                .iter()
                .map(|&side| side.max(0) as usize)
                .collect();
            let factor = factor_of(&mut self.factor, self.tile, &made, rows, &self.pp_log)?;
            let (dst, out_size) = match &mut destination {
                Some((dst, out_size)) => (&*dst, *out_size),
                None => {
                    let out_size = (size.0 * factor, size.1 * factor);
                    let made = picture(
                        &mut self.hw_frames_ctx,
                        &self.pool,
                        &self.hw_device_ctx,
                        &self.pp_log,
                        self.output,
                        out_size,
                    )?;
                    let (dst, _) = destination.insert((made, out_size));
                    (&*dst, out_size)
                }
            };
            let dst = if ten_bit {
                Source::P010(Nv12Surface::from_frame(dst).ok_or(OrtError::MissingSurface)?)
            } else {
                Source::Bgra(BgraSurface::from_frame(dst).ok_or(OrtError::MissingSurface)?)
            };
            let out_tile = (self.tile.0 * factor, self.tile.1 * factor);
            for (slot, patch) in group.iter().enumerate() {
                let (across, down) = (patch.across, patch.down);
                let unfit = Unfit {
                    plane: out_tile,
                    region: (
                        (across.keep_from - across.start) * factor,
                        (down.keep_from - down.start) * factor,
                        (across.keep_to - across.keep_from) * factor,
                        (down.keep_to - down.keep_from) * factor,
                    ),
                    at: (across.keep_from * factor, down.keep_from * factor),
                    red: planes[0] as u32,
                    blue: planes[2] as u32,
                    mul,
                    add,
                };
                let at = output.data_ptr() as CUdeviceptr
                    + (slot * 3 * out_tile.0 as usize * out_tile.1 as usize * size_of::<f32>())
                        as CUdeviceptr;
                // SAFETY: `at` is the `slot`th of `rows` outputs of `made`'s
                // shape — three planes of `out_tile`, `factor_of` checked —
                // on this device, written by a run that has finished
                // (`synchronize_outputs`) and kept by `outputs` until the
                // synchronize below has returned.
                unsafe {
                    match dst {
                        Source::P010(dst) => self.driver.unfit_p010(
                            &self.kernels,
                            at,
                            unfit,
                            dst,
                            out_size,
                            writing,
                        )?,
                        Source::Bgra(dst) => {
                            self.driver
                                .unfit_bgra(&self.kernels, at, unfit, dst, out_size)?
                        }
                        Source::Nv12(_) => unreachable!("nothing is written as NV12"),
                    }
                }
            }
            // What was written is read on FFmpeg's stream next, and the
            // output it was written from goes with `outputs`.
            self.driver.synchronize()?;
        }

        let (mut picture, _) = destination.expect("a picture has at least one piece");
        // SAFETY: both frames are live and distinct — `picture` came from
        // the pool, `frame` is the caller's — so `av_frame_copy_props`
        // reads one and writes the other with no aliasing.
        unsafe {
            ffi::av_frame_copy_props(picture.as_mut_ptr(), frame.as_ptr());
        }
        // Ten bits are written in the picture's own colour, which its tags
        // go on saying. Otherwise what the model made is full-range RGB,
        // whatever it was made of; an HDR picture was brought to SDR first.
        if ten_bit {
            return Ok(picture);
        }
        picture.set_color_space(ffmpeg::color::Space::RGB);
        picture.set_color_range(ffmpeg::color::Range::JPEG);
        if hdr {
            picture.set_color_primaries(ffmpeg::color::Primaries::BT709);
            picture.set_color_transfer_characteristic(ffmpeg::color::TransferCharacteristic::BT709);
        }
        Ok(picture)
    }

    /// Reads `patch` of the picture into the `slot`th input, the rest of a
    /// piece larger than the picture grey.
    fn fit(
        &self,
        source: Source,
        colour: &YuvToBgra,
        slot: usize,
        patch: Patch,
    ) -> std::result::Result<(), OrtError> {
        let (x, y) = (u64::from(patch.across.start), u64::from(patch.down.start));
        let cut = (patch.across.length, patch.down.length);
        let fit = Fit {
            model: self.tile,
            offset: (0, 0),
            scaled: cut,
            orientation: Orientation::UPRIGHT,
        };
        match source {
            Source::Nv12(whole) => {
                let crop = Nv12Surface {
                    luma: whole.luma + y * whole.luma_pitch as u64 + x,
                    chroma: whole.chroma + (y / 2) * whole.chroma_pitch as u64 + x,
                    ..whole
                };
                self.driver
                    .fit_nv12(&self.kernels, &self.tensor, slot, fit, crop, cut, colour)?;
            }
            Source::P010(whole) => {
                let crop = Nv12Surface {
                    luma: whole.luma + y * whole.luma_pitch as u64 + x * 2,
                    chroma: whole.chroma + (y / 2) * whole.chroma_pitch as u64 + x * 2,
                    ..whole
                };
                self.driver
                    .fit_p010(&self.kernels, &self.tensor, slot, fit, crop, cut, colour)?;
            }
            Source::Bgra(whole) => {
                let crop = BgraSurface {
                    pixels: whole.pixels + y * whole.pitch as u64 + x * 4,
                    ..whole
                };
                self.driver
                    .fit_bgra(&self.kernels, &self.tensor, slot, fit, crop, cut)?;
            }
        }
        Ok(())
    }

    /// How the model's values are read back to 0 to 1, red, green and blue:
    /// its input's scaling, undone.
    fn reading(&self) -> ([f32; 3], [f32; 3]) {
        let planes = self.values.planes();
        let mut mul = [1.0; 3];
        let mut add = [0.0; 3];
        for (colour, plane) in planes.into_iter().enumerate() {
            mul[colour] = 1.0 / self.values.scale[plane];
            add[colour] = -self.values.bias[plane] / self.values.scale[plane];
        }
        (mul, add)
    }
}

impl PerFrameTransform for Filtering {
    fn repeated(&mut self) -> &mut RepeatedOutput {
        &mut self.repeated
    }

    fn frame_ref_failed(&self, code: i32) -> crate::error::Error {
        pp_error!(self, "av_frame_ref failed: {code}");
        OrtError::from(CudaUploadError::FrameRef(code)).into()
    }

    fn produce(
        &mut self,
        source: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
        self.filter(source)
            .inspect_err(|error| pp_error!(self, "{error}"))
            .map_err(Into::into)
    }
}

impl Filter for Filtering {
    /// Device-resident frames: NV12, BGRA, or P010 read through an SDR
    /// copy — or P010 alone, made into P010.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda).with_layouts(match self
                .output
            {
                CudaFrameFormat::P010 => crate::contract::PixelLayoutSet::P010,
                _ => crate::contract::PixelLayoutSet::GPU_SCALABLE,
            }),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                .with_layouts(self.output.layouts()),
        )
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "CudaOrtImageFilter",
                wanted: "CUDA video frames",
                got: buf.kind(),
            }
            .into());
        };
        let made = PerFrameTransform::transform(self, &frame)?;
        out.push(MediaBuffer::Video(made.into()));
        Ok(())
    }

    /// A seek or a flush: what follows is not a repeat of what came before.
    fn reset(&mut self) {
        self.repeated.clear();
    }

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
    }
}

/// How many times larger than a piece of `tile` the model's output of
/// `shape` is, `rows` pieces of it: the same as every run before, `known`.
fn factor_of(
    known: &mut Option<u32>,
    tile: (u32, u32),
    shape: &[usize],
    rows: usize,
    pp_log: &PpLog,
) -> std::result::Result<u32, OrtError> {
    let (tile_w, tile_h) = (tile.0 as usize, tile.1 as usize);
    let factor = match *shape {
        [batch, 3, height, width]
            if batch == rows
                && height.is_multiple_of(tile_h)
                && width.is_multiple_of(tile_w)
                && height / tile_h == width / tile_w
                && height >= tile_h =>
        {
            u32::try_from(height / tile_h).ok()
        }
        _ => None,
    };
    let Some(factor) = factor.filter(|factor| known.is_none_or(|was| was == *factor)) else {
        return Err(OrtError::UnsupportedModel(format!(
            "its output is {shape:?} for {rows} pieces of {tile_w}x{tile_h}, where a \
             picture filter's is the same pictures a whole number of times larger"
        )));
    };
    if known.is_none() {
        pp_info!(pp_log: pp_log, "the model makes pictures {factor} times larger");
    }
    *known = Some(factor);
    Ok(factor)
}

/// A picture of `format` and `size` from `pool`, its surface from `frames`,
/// made for that size on `device` where it is not.
fn picture(
    frames: &mut ForSize<AvBufferRef>,
    pool: &UnboundObjectPool<ffmpeg::frame::Video>,
    device: &AvBufferRef,
    pp_log: &PpLog,
    format: CudaFrameFormat,
    size: (u32, u32),
) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, OrtError> {
    let frames_ctx = frames
        .try_get(size.0, size.1, |width, height| {
            pp_info!(pp_log: pp_log, "allocating a {width}x{height} pool");
            // SAFETY: `create_hw_frames_ctx`'s contract is a live device
            // context, which is what the owned `AvBufferRef` is.
            unsafe { create_hw_frames_ctx(device, format, width, height) }
                .map_err(CudaUploadError::from)
        })?
        .as_ptr();
    let mut picture = pool.get();
    // SAFETY: `picture` is the pooled wrapper's own `AVFrame`; the unref
    // hands its previous surface back to its pool before another is
    // taken, and the frames context is this element's own, held for its
    // life.
    unsafe {
        let dst = picture.as_mut_ptr();
        ffi::av_frame_unref(dst);
        let code = ffi::av_hwframe_get_buffer(frames_ctx, dst, 0);
        if code < 0 {
            return Err(CudaUploadError::HwFrameGet(code).into());
        }
    }
    Ok(picture)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pixel of a side is kept from exactly one piece, at least the
    /// overlap from that piece's edge wherever the picture goes on past it;
    /// pieces start even and lie inside the side.
    #[test]
    fn pieces_keep_every_pixel_once_away_from_their_edges() {
        for length in [2, 100, 256, 258, 300, 480, 1080, 1920, 3840] {
            let found = pieces(length, 256, 16);
            let mut kept = 0;
            for piece in &found {
                assert_eq!(piece.keep_from, kept, "{length}: {found:?}");
                assert!(piece.start.is_multiple_of(2), "{length}: {found:?}");
                assert!(piece.start + piece.length <= length, "{length}: {found:?}");
                assert!(
                    piece.keep_from >= piece.start && piece.keep_to <= piece.start + piece.length
                );
                if piece.keep_from > 0 {
                    assert!(piece.keep_from >= piece.start + 16, "{length}: {found:?}");
                }
                if piece.keep_to < length {
                    assert!(
                        piece.keep_to + 16 <= piece.start + piece.length,
                        "{length}: {found:?}"
                    );
                }
                assert!(piece.keep_to.is_multiple_of(2), "{length}: {found:?}");
                kept = piece.keep_to;
            }
            assert_eq!(kept, length, "{length}: {found:?}");
        }
        assert_eq!(pieces(100, 256, 16).len(), 1, "shorter than a piece: one");
    }

    /// An ONNX model of one `Resize` by `factor`, nearest pixel, of a
    /// picture of any size: each output pixel `(x, y)` is input pixel
    /// `(x / 2, y / 2)` where `factor` is 2. Written out here, so that a
    /// test of a picture filter needs no model of anyone's.
    fn resize_model(factor: f32) -> Vec<u8> {
        fn varint(out: &mut Vec<u8>, mut value: u64) {
            while value >= 0x80 {
                out.push(value as u8 | 0x80);
                value >>= 7;
            }
            out.push(value as u8);
        }
        fn int(out: &mut Vec<u8>, field: u64, value: u64) {
            varint(out, field << 3);
            varint(out, value);
        }
        fn bytes(out: &mut Vec<u8>, field: u64, value: &[u8]) {
            varint(out, (field << 3) | 2);
            varint(out, value.len() as u64);
            out.extend_from_slice(value);
        }
        // A float tensor of `[batch, 3, height, width]`, each left open.
        let value_info = |name: &str| {
            let mut shape = Vec::new();
            for (open, fixed) in [(Some("n"), 0), (None, 3), (Some("h"), 0), (Some("w"), 0)] {
                let mut dim = Vec::new();
                match open {
                    Some(name) => bytes(&mut dim, 2, name.as_bytes()),
                    None => int(&mut dim, 1, fixed),
                }
                bytes(&mut shape, 1, &dim);
            }
            let mut tensor = Vec::new();
            int(&mut tensor, 1, 1);
            bytes(&mut tensor, 2, &shape);
            let mut ty = Vec::new();
            bytes(&mut ty, 1, &tensor);
            let mut info = Vec::new();
            bytes(&mut info, 1, name.as_bytes());
            bytes(&mut info, 2, &ty);
            info
        };
        let mut node = Vec::new();
        for input in ["input", "", "scales"] {
            bytes(&mut node, 1, input.as_bytes());
        }
        bytes(&mut node, 2, b"output");
        bytes(&mut node, 4, b"Resize");
        let mut scales = Vec::new();
        int(&mut scales, 1, 4);
        int(&mut scales, 2, 1);
        let floats: Vec<u8> = [1.0f32, 1.0, factor, factor]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        bytes(&mut scales, 4, &floats);
        bytes(&mut scales, 8, b"scales");
        let mut graph = Vec::new();
        bytes(&mut graph, 1, &node);
        bytes(&mut graph, 2, b"resize");
        bytes(&mut graph, 5, &scales);
        bytes(&mut graph, 11, &value_info("input"));
        bytes(&mut graph, 12, &value_info("output"));
        let mut opset = Vec::new();
        bytes(&mut opset, 1, b"");
        int(&mut opset, 2, 13);
        let mut model = Vec::new();
        int(&mut model, 1, 8);
        bytes(&mut model, 8, &opset);
        bytes(&mut model, 7, &graph);
        model
    }

    /// `model` written to a file of its own, removed when dropped.
    struct ModelFile(std::path::PathBuf);

    impl ModelFile {
        fn new(model: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "media_pp_resize_{}_{}.onnx",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::write(&path, model).expect("written");
            Self(path)
        }
    }

    impl Drop for ModelFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn pixel(x: u32, y: u32) -> [u8; 4] {
        [
            ((x * 7 + y * 3) % 256) as u8,
            ((x * 13 + y * 5) % 256) as u8,
            ((x * 29 + y * 11) % 256) as u8,
            255,
        ]
    }

    /// A `width` by `height` picture of [`pixel`], uploaded as `format`,
    /// shown at PTS 42.
    fn uploaded(
        device: &CudaDevice,
        width: u32,
        height: u32,
        format: CudaFrameFormat,
    ) -> MediaBuffer {
        use crate::element::RawSink;
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
        let stride = frame.stride(0);
        for y in 0..height {
            for x in 0..width {
                frame.data_mut(0)[y as usize * stride + x as usize * 4..][..4]
                    .copy_from_slice(&pixel(x, y));
            }
        }
        frame.set_pts(Some(42));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational(1, 30));
        let frame = if format == CudaFrameFormat::Nv12 {
            let mut scaler = ffmpeg::software::scaling::Context::get(
                ffmpeg::format::Pixel::BGRA,
                width,
                height,
                ffmpeg::format::Pixel::NV12,
                width,
                height,
                ffmpeg::software::scaling::Flags::POINT,
            )
            .expect("a scaler");
            let mut nv12 = ffmpeg::frame::Video::empty();
            scaler.run(&frame, &mut nv12).expect("scaled");
            nv12.set_pts(Some(42));
            crate::buffer::set_time_base(&mut nv12, ffmpeg::Rational(1, 30));
            nv12
        } else {
            frame
        };
        let mut upload = crate::elements::CudaUpload::new("upload", device, format);
        let received = crate::test_support::capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("uploads");
        received.lock().unwrap().remove(0)
    }

    /// `buf`, a CUDA BGRA picture, in system memory.
    fn downloaded(device: &CudaDevice, buf: MediaBuffer) -> ffmpeg::frame::Video {
        use crate::element::RawSink;
        let mut download =
            crate::elements::CudaDownload::new("download", device, CudaFrameFormat::Bgra);
        let received = crate::test_support::capture(&mut download);
        download.consume(buf).expect("downloads");
        let MediaBuffer::Video(frame) = received.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        (**frame).clone()
    }

    fn filtered(filter: &mut CudaOrtImageFilter, buf: MediaBuffer) -> Result<MediaBuffer> {
        use crate::element::RawSink;
        let received = crate::test_support::capture(filter);
        filter.consume(buf)?;
        Ok(received.lock().unwrap().remove(0))
    }

    /// The model's picture of each piece goes back where the piece came
    /// from, and the picture comes out twice the size, every pixel the one
    /// it was made of: across pieces and their overlaps, from BGRA exactly
    /// and from NV12 as its colours read, the PTS kept and the colour tagged
    /// RGB. Handed again, the same picture is answered with what was made.
    #[test]
    fn a_picture_comes_out_as_the_model_makes_it_piece_by_piece() {
        let Some((device, _serial)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let model = ModelFile::new(&resize_model(2.0));
        let mut filter = match CudaOrtImageFilter::new(
            "twice",
            &device,
            &model.0,
            CudaOrtImageFilterOptions::default(),
        ) {
            Ok(filter) => filter,
            Err(error) => {
                eprintln!("skipping: the CUDA runtime cannot be used here: {error}");
                return;
            }
        };
        // Two pieces across, one of them shifted back to end at the edge,
        // and one down, larger than the picture.
        let (width, height) = (300u32, 200u32);
        let bgra = uploaded(&device, width, height, CudaFrameFormat::Bgra);
        let made = filtered(&mut filter, bgra.clone()).expect("filters");
        let MediaBuffer::Video(made_frame) = &made else {
            panic!("a picture");
        };
        assert_eq!(
            (made_frame.pts(), made_frame.color_space()),
            (Some(42), ffmpeg::color::Space::RGB)
        );
        let out = downloaded(&device, made.clone());
        assert_eq!((out.width(), out.height()), (width * 2, height * 2));
        let stride = out.stride(0);
        for y in 0..height * 2 {
            for x in 0..width * 2 {
                let at = &out.data(0)[y as usize * stride + x as usize * 4..][..4];
                assert_eq!(at, pixel(x / 2, y / 2), "({x}, {y})");
            }
        }
        let again = filtered(&mut filter, bgra).expect("filters again");
        let id = |buf: &MediaBuffer| match buf {
            MediaBuffer::Video(frame) => crate::buffer::picture_id(frame),
            _ => panic!("a picture"),
        };
        assert_eq!(
            id(&again),
            id(&made),
            "a repeat is answered with what was made"
        );

        // NV12, read by its colour tags as `CudaConverter` reads it.
        let nv12 = uploaded(&device, width, height, CudaFrameFormat::Nv12);
        let reference = {
            use crate::element::RawSink;
            let mut converter =
                crate::elements::CudaConverter::new("to-bgra", &device, CudaFrameFormat::Bgra)
                    .expect("a converter");
            let received = crate::test_support::capture(&mut converter);
            converter.consume(nv12.clone()).expect("converts");
            let converted = received.lock().unwrap().remove(0);
            downloaded(&device, converted)
        };
        let out = downloaded(&device, filtered(&mut filter, nv12).expect("filters NV12"));
        let (stride, reference_stride) = (out.stride(0), reference.stride(0));
        let mut worst = 0;
        for y in 0..height * 2 {
            for x in 0..width * 2 {
                let at = &out.data(0)[y as usize * stride + x as usize * 4..][..4];
                let expected = &reference.data(0)
                    [(y / 2) as usize * reference_stride + (x / 2) as usize * 4..][..4];
                for (made, read) in at.iter().zip(expected) {
                    worst = worst.max(made.abs_diff(*read));
                }
            }
        }
        assert!(
            worst <= 1,
            "NV12 read {worst} away from CudaConverter's reading"
        );
    }

    /// Wiring takes CUDA pictures in and not system memory's, and what
    /// comes out goes into a converter to NV12 and not into one to BGRA.
    #[test]
    fn it_links_from_cuda_pictures_into_what_takes_bgra() {
        use crate::contract::{OutputContract, PixelLayoutSet, check_link};
        use crate::element::{RawSink, SrcPads};
        use crate::elements::CudaConverter;
        let Some((device, _serial)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let model = ModelFile::new(&resize_model(2.0));
        let Ok(mut filter) = CudaOrtImageFilter::new(
            "twice",
            &device,
            &model.0,
            CudaOrtImageFilterOptions::default(),
        ) else {
            eprintln!("skipping: the CUDA runtime cannot be used here");
            return;
        };
        let frames = |domain| {
            OutputContract::Fixed(
                PortContract::frame(MediaKind::VideoFrame, domain)
                    .with_layouts(PixelLayoutSet::GPU_SCALABLE),
            )
        };
        let accepted = filter.input_contract();
        assert!(!check_link(&frames(MemoryDomain::Cuda), &accepted).is_refused());
        assert!(check_link(&frames(MemoryDomain::System), &accepted).is_refused());
        let made = filter.src_pads()[0].contract();
        let into = |format| {
            CudaConverter::new("convert", &device, format)
                .expect("a converter")
                .input_contract()
        };
        assert!(!check_link(&made, &into(CudaFrameFormat::Nv12)).is_refused());
        assert!(check_link(&made, &into(CudaFrameFormat::Bgra)).is_refused());
    }

    /// Made into P010, an HDR picture stays HDR in ten bits: its tags go on
    /// with it, and each pixel of a two-times enlargement is the ten-bit
    /// samples of the one it was made of, across pieces. An 8-bit picture
    /// is refused where ten bits are to be made, and NV12 is no output.
    #[test]
    fn ten_bits_come_out_as_they_went_in() {
        use crate::element::RawSink;
        let Some((device, _serial)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let model = ModelFile::new(&resize_model(2.0));
        let options = CudaOrtImageFilterOptions {
            output: CudaFrameFormat::P010,
            ..CudaOrtImageFilterOptions::default()
        };
        let Ok(mut filter) = CudaOrtImageFilter::new("twice", &device, &model.0, options) else {
            eprintln!("skipping: the CUDA runtime cannot be used here");
            return;
        };
        assert!(matches!(
            CudaOrtImageFilter::new(
                "nv12",
                &device,
                &model.0,
                CudaOrtImageFilterOptions {
                    output: CudaFrameFormat::Nv12,
                    ..CudaOrtImageFilterOptions::default()
                }
            ),
            Err(crate::error::Error::OrtDetectorError(
                OrtError::UnsupportedOutput(CudaFrameFormat::Nv12)
            ))
        ));

        // Colours each block of four shares, its pixels lifted alike in
        // every colour, so that its chroma is the block's exactly.
        let (width, height) = (300u32, 200u32);
        let write = crate::color::rgb_to_yuv_rows_of_depth(
            ffmpeg::color::Space::BT2020NCL,
            ffmpeg::color::Range::MPEG,
            height,
            10,
        );
        let code = |row: [f32; 4], [r, g, b]: [f32; 3]| {
            ((row[0] * r + row[1] * g + row[2] * b + row[3]) * 1023.0).round() as u16
        };
        let base = |bx: u32, by: u32| {
            [
                0.2 + 0.6 * ((bx * 7 + by * 3) % 11) as f32 / 10.0,
                0.2 + 0.6 * ((bx * 5 + by * 9) % 11) as f32 / 10.0,
                0.2 + 0.6 * ((bx * 3 + by * 7) % 11) as f32 / 10.0,
            ]
        };
        let luma = |x: u32, y: u32| {
            let lift = 0.04 * ((x + 2 * y) % 5) as f32 / 4.0 - 0.02;
            code(write[0], base(x / 2, y / 2).map(|c| c + lift))
        };
        let chroma = |bx: u32, by: u32, c: usize| code(write[1 + c], base(bx, by));
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::P010LE, width, height);
        frame.set_color_space(ffmpeg::color::Space::BT2020NCL);
        frame.set_color_range(ffmpeg::color::Range::MPEG);
        frame.set_color_primaries(ffmpeg::color::Primaries::BT2020);
        frame.set_color_transfer_characteristic(ffmpeg::color::TransferCharacteristic::SMPTE2084);
        frame.set_pts(Some(42));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational(1, 30));
        let stride = frame.stride(0);
        for y in 0..height {
            for x in 0..width {
                frame.data_mut(0)[y as usize * stride + x as usize * 2..][..2]
                    .copy_from_slice(&(luma(x, y) << 6).to_le_bytes());
            }
        }
        let stride = frame.stride(1);
        for by in 0..height / 2 {
            for bx in 0..width / 2 {
                for c in 0..2 {
                    frame.data_mut(1)[by as usize * stride + (bx as usize * 2 + c) * 2..][..2]
                        .copy_from_slice(&(chroma(bx, by, c) << 6).to_le_bytes());
                }
            }
        }
        let mut upload = crate::elements::CudaUpload::new("upload", &device, CudaFrameFormat::P010);
        let received = crate::test_support::capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("uploads");
        let picture = received.lock().unwrap().remove(0);

        let made = filtered(&mut filter, picture).expect("filters");
        let mut download =
            crate::elements::CudaDownload::new("download", &device, CudaFrameFormat::P010);
        let received = crate::test_support::capture(&mut download);
        download.consume(made).expect("downloads");
        let MediaBuffer::Video(out) = received.lock().unwrap().remove(0) else {
            panic!("a picture");
        };
        assert_eq!((out.width(), out.height()), (width * 2, height * 2));
        assert_eq!(out.format(), ffmpeg::format::Pixel::P010LE);
        assert_eq!(
            (
                out.pts(),
                out.color_space(),
                out.color_transfer_characteristic()
            ),
            (
                Some(42),
                ffmpeg::color::Space::BT2020NCL,
                ffmpeg::color::TransferCharacteristic::SMPTE2084
            ),
            "the picture's own colour, HDR and all"
        );
        let sample = |plane: usize, x: u32, y: u32| {
            let at = y as usize * out.stride(plane) + x as usize * 2;
            u16::from_le_bytes([out.data(plane)[at], out.data(plane)[at + 1]]) >> 6
        };
        let mut worst = 0;
        for y in 0..height * 2 {
            for x in 0..width * 2 {
                worst = worst.max(sample(0, x, y).abs_diff(luma(x / 2, y / 2)));
            }
        }
        // Each block of the output is one pixel of the input, four times:
        // the chroma of that pixel's colour, which is its block's.
        for by in 0..height {
            for bx in 0..width {
                for c in 0..2 {
                    let now = sample(1, bx * 2 + c as u32, by);
                    worst = worst.max(now.abs_diff(chroma(bx / 2, by / 2, c)));
                }
            }
        }
        assert!(
            worst <= 1,
            "samples {worst} steps from what they were made of"
        );

        let eight_bit = uploaded(&device, 64, 48, CudaFrameFormat::Nv12);
        assert!(
            filtered(&mut filter, eight_bit).is_err(),
            "NV12 is not ten bits"
        );
    }

    /// A model whose output is not a whole number of times larger is
    /// refused at its first picture, and every picture after; a packet
    /// is refused.
    #[test]
    fn a_model_that_is_not_a_whole_number_of_times_larger_is_refused() {
        let Some((device, _serial)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let model = ModelFile::new(&resize_model(1.5));
        let Ok(mut filter) = CudaOrtImageFilter::new(
            "one-and-a-half",
            &device,
            &model.0,
            CudaOrtImageFilterOptions::default(),
        ) else {
            eprintln!("skipping: the CUDA runtime cannot be used here");
            return;
        };
        for _ in 0..2 {
            let picture = uploaded(&device, 64, 48, CudaFrameFormat::Bgra);
            let Err(error) = filtered(&mut filter, picture) else {
                panic!("refused");
            };
            let error = error.to_string();
            assert!(error.contains("whole number"), "{error}");
        }
        let packet = MediaBuffer::packet(ffmpeg::Packet::new(4));
        assert!(filtered(&mut filter, packet).is_err());
    }
}
