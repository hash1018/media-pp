//! [`CudaOrtDetector`]: object detection on CUDA pictures, each handed on
//! with what was found in it, without the picture leaving the GPU.

#[cfg(feature = "ort-tensorrt")]
use std::path::PathBuf;
use std::{path::Path, sync::Arc};

use ffmpeg_next::{self as ffmpeg, ffi};
#[cfg(feature = "ort-tensorrt")]
use ort::ep::TensorRT;
use ort::{
    ep::CUDA,
    inputs,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::{Shape, TensorRefMut},
};

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    element::{Context, Element, ElementType, element_pp_log},
    error::Result,
    platform::cuda::{
        CudaDevice,
        driver::{BgraSurface, CudaDriver, CudaTensor, Fit, FitKernels, Nv12Surface, YuvToBgra},
        frame::{self, CudaSurfaces},
    },
    platform::ffmpeg::AvBufferRef,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use crate::elements::{BatchSlot, Detection, Detections};

use super::super::{
    Interval, Letterbox, OrtDetectorError, OrtDetectorOptions, decode_batch, labels, model_input,
};
use super::runtime::{self, CudaRuntime};

/// Whether a [`CudaOrtDetector`] runs its model through TensorRT, with
/// the `ort-tensorrt` feature.
///
/// TensorRT compiles the model into an engine for this GPU — fused layers,
/// the fastest kernel for each found by timing them, half precision where
/// [`CudaOrtDetectorOptions::fp16`] allows — and runs it several times
/// faster than CUDA does, at the cost of minutes building the engine the
/// first time and TensorRT's libraries at run time. CUDA alone runs the
/// model operator by operator on cuDNN and cuBLAS, and starts at once.
/// Either way an operator TensorRT cannot run falls to CUDA.
///
/// TensorRT's libraries are linked with the feature, so a program built
/// with it does not start without them; what is left to decide is a
/// TensorRT older than the one taken, and a provider that will not start.
#[cfg(feature = "ort-tensorrt")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UseTensorRtPolicy {
    /// CUDA alone.
    Off,
    /// TensorRT where it can run; CUDA alone where it cannot — with a
    /// warning where TensorRT is too old, and without one where ONNX
    /// Runtime passes over a provider that will not start.
    #[default]
    Preferred,
    /// TensorRT, or no detector: [`CudaOrtDetector::new`] refuses with
    /// [`OrtDetectorError::TensorRtUnavailable`] where TensorRT is too old,
    /// and with the provider's own error where it will not start.
    Required,
}

/// Whether to run on TensorRT, from what the machine has and what was
/// asked; the warning, if one is due, is the caller's to say.
#[cfg(feature = "ort-tensorrt")]
fn choose(
    runtime: CudaRuntime,
    policy: UseTensorRtPolicy,
) -> std::result::Result<bool, OrtDetectorError> {
    match (runtime, policy) {
        (CudaRuntime::Unavailable { cuda }, _) => {
            Err(OrtDetectorError::CudaRuntimeUnavailable(cuda))
        }
        (_, UseTensorRtPolicy::Off) => Ok(false),
        (CudaRuntime::TensorRt, _) => Ok(true),
        (CudaRuntime::CudaOnly { tensorrt }, UseTensorRtPolicy::Required) => {
            Err(OrtDetectorError::TensorRtUnavailable(tensorrt))
        }
        (CudaRuntime::CudaOnly { .. }, UseTensorRtPolicy::Preferred) => Ok(false),
    }
}

/// How a [`CudaOrtDetector`] runs its model, beside what every detector is
/// told.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaOrtDetectorOptions {
    /// Thresholds and labels, as every detector takes them.
    pub detector: OrtDetectorOptions,
    /// The most pictures the model is run on at once: the pictures of a
    /// [`StreamMux`](crate::elements::StreamMux)'s batch, up to this many,
    /// go through it together — DeepStream's `batch-size`. A small model
    /// leaves most of the GPU idle on one picture, and a batch fills it:
    /// on an RTX 3050, YOLO11n through TensorRT ran about 460 pictures a
    /// second one at a time and 880 eight at a time. Give it the mux's
    /// `max_batch`; a larger batch is run in parts this size.
    ///
    /// It takes a model whose batch is left open, as an Ultralytics export
    /// with `dynamic=True` is; a model made for one picture at a time —
    /// the YOLOv10n release — is run so, with a warning. With TensorRT the
    /// engine is built for every batch from 1 to this, and built again —
    /// minutes — when this changes. The default, 1, is a detector of one
    /// picture at a time.
    pub max_batch: usize,
    /// Whether TensorRT runs the model, and what happens where it cannot.
    #[cfg(feature = "ort-tensorrt")]
    pub tensorrt: UseTensorRtPolicy,
    /// Whether TensorRT may run in half precision — about twice as fast on
    /// a GPU with tensor cores, at a detector's tolerance.
    #[cfg(feature = "ort-tensorrt")]
    pub fp16: bool,
    /// Where TensorRT keeps the engine it builds for this model and GPU.
    /// Building one takes minutes; the first detector to start builds it
    /// and every later one loads it in under a second. `None` keeps it in
    /// the user's cache directory, under `media-pp/tensorrt`.
    #[cfg(feature = "ort-tensorrt")]
    pub engine_cache: Option<PathBuf>,
}

impl Default for CudaOrtDetectorOptions {
    fn default() -> Self {
        Self {
            detector: OrtDetectorOptions::default(),
            max_batch: 1,
            #[cfg(feature = "ort-tensorrt")]
            tensorrt: UseTensorRtPolicy::default(),
            #[cfg(feature = "ort-tensorrt")]
            fp16: true,
            #[cfg(feature = "ort-tensorrt")]
            engine_cache: None,
        }
    }
}

/// Runs a YOLO detector on each CUDA picture and hands the picture on
/// unchanged, carrying the [`Detections`] found in it — what
/// [`SwOrtDetector`](super::super::SwOrtDetector) does on the CPU, with the
/// picture never leaving the GPU: a kernel fits it into the model's input
/// in device memory, and ONNX Runtime's CUDA provider — or with
/// `ort-tensorrt`, TensorRT — reads it there.
///
/// It takes NV12 or BGRA CUDA pictures of any size from the same
/// [`CudaDevice`] as the rest of the pipeline — a decoder's, a compositor's
/// — and the boxes it finds are fractions of each picture, as
/// [`Detection`](crate::elements::Detection) describes. The models it
/// reads are those [`OrtDetectorOptions`] describes.
///
/// # Requirements
///
/// The `ort-cuda` feature, which fetches ONNX Runtime's CUDA build and
/// links CUDA 13's runtime, cuBLAS and cuRAND, and cuDNN 9, into the
/// program; `ort-tensorrt` adds TensorRT 10 and its ONNX parser. Building
/// needs them where the linker finds them, and a program built with them
/// does not start without them, as it does not without FFmpeg. At run time
/// it also needs a driver for CUDA 13.0, and versions no older than ONNX
/// Runtime's build was made against: the CUDA 13.2 runtime, cuDNN 9.23.2
/// and TensorRT 10.15.1 — [`CudaOrtDetector::runtime`] says which falls
/// short, before a model is loaded.
///
/// The first start with TensorRT builds an engine for the model and GPU,
/// which takes minutes — see `CudaOrtDetectorOptions::engine_cache`.
pub struct CudaOrtDetector(FilterStage<Detecting>);

filter_stage!(CudaOrtDetector);

/// What a [`CudaOrtDetector`] does with each picture.
///
/// The device buffers and kernels come before the driver: fields drop in
/// order, and both are freed in the driver's context, which the driver
/// releases.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtDetectorOptions,
    labels: Arc<[Arc<str>]>,
    model: (u32, u32),
    /// Which pictures it looks at.
    interval: Interval,
    /// How many pictures the tensor holds, and the model is run on at
    /// once.
    max_batch: usize,
    /// The pictures of the batch under way, in the order they came, each
    /// looked at fitted into the next slot of the tensor.
    held: Vec<Held>,
    /// The pipeline's, to report a picture's failure on where others must
    /// go on in the same call.
    context: Option<Arc<Context>>,
    tensor: CudaTensor,
    kernels: FitKernels,
    driver: CudaDriver,
    memory: MemoryInfo<'static>,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *mut ffi::AVHWDeviceContext,
    _hw_device_ctx: Arc<AvBufferRef>,
}

// SAFETY: the session, buffers and kernels are used by the one thread
// transforming at a time; `device_ctx` only ever has its address compared,
// and `&mut self` on every method that touches the rest rules out
// concurrent access — the reasoning `CudaConverter` gives for its own.
unsafe impl Send for Detecting {}

impl CudaOrtDetector {
    /// What this machine can run a detector on, without a model, a device
    /// or a session: whether the driver is there, and whether it and the
    /// linked libraries are new enough — each asked its own version. For an
    /// application deciding whether to offer detection on the GPU at all;
    /// [`Self::new`] asks the same question first.
    ///
    /// The linked libraries are there if the program started, so what this
    /// can find short of them is age: the versions taken are those ONNX
    /// Runtime's build was made against — a driver for CUDA 13.0 or newer,
    /// the CUDA 13.2 runtime, cuDNN 9.23.2 and TensorRT 10.15.1. Whether
    /// there is a GPU, [`CudaDevice::new`] answers.
    pub fn runtime() -> CudaRuntime {
        runtime::runtime(&runtime::Linked::ask())
    }

    /// Loads the model at `model_path` to run on `device`'s GPU — the same
    /// [`CudaDevice`] every other CUDA element in the pipeline was built
    /// from.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        model_path: impl AsRef<Path>,
        options: CudaOrtDetectorOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaOrtDetector, &name, None);
        let path = model_path.as_ref().display().to_string();
        if options.max_batch == 0 {
            return Err(OrtDetectorError::ZeroMaxBatch.into());
        }

        let linked = runtime::Linked::ask();
        let runtime = runtime::runtime(&linked);
        #[cfg(feature = "ort-tensorrt")]
        let tensorrt = {
            if let (CudaRuntime::CudaOnly { tensorrt }, UseTensorRtPolicy::Preferred) =
                (&runtime, options.tensorrt)
            {
                pp_warn!(pp_log: &pp_log, "TensorRT cannot be used ({tensorrt}): running on CUDA alone");
            }
            choose(runtime, options.tensorrt)?
        };
        #[cfg(not(feature = "ort-tensorrt"))]
        if let CudaRuntime::Unavailable { cuda } = runtime {
            return Err(OrtDetectorError::CudaRuntimeUnavailable(cuda).into());
        }

        let mut providers = Vec::new();
        #[cfg(feature = "ort-tensorrt")]
        let provider = if tensorrt {
            let cache = options
                .engine_cache
                .clone()
                .unwrap_or_else(default_engine_cache);
            if let Err(error) = std::fs::create_dir_all(&cache) {
                pp_warn!(pp_log: &pp_log, "no engine cache at {}: {error}", cache.display());
            }
            // Device 0 for both providers: the one `CudaDevice` opens, which
            // takes no ordinal, and so where every incoming picture is.
            let mut provider = TensorRT::default()
                .with_device_id(0)
                .with_fp16(options.fp16)
                .with_engine_cache(true)
                .with_engine_cache_path(cache.display())
                .with_timing_cache(true)
                .with_timing_cache_path(cache.display());
            // TensorRT builds an engine for the shapes it is told of, and
            // builds it again for a batch outside them: told every batch up
            // to `max_batch`, a short batch costs no rebuild.
            if options.max_batch > 1
                && let Some(input) = batch_input(model_path.as_ref())?
            {
                let (w, h) = (input.width, input.height);
                let shape = |batch| format!("{}:{batch}x3x{h}x{w}", input.name);
                provider = provider
                    .with_profile_min_shapes(shape(1))
                    .with_profile_opt_shapes(shape(options.max_batch))
                    .with_profile_max_shapes(shape(options.max_batch));
            }
            let provider = provider.build();
            // ONNX Runtime passes over a provider that fails to register and
            // tries the next, which would be CUDA running in TensorRT's place.
            providers.push(if options.tensorrt == UseTensorRtPolicy::Required {
                provider.error_on_failure()
            } else {
                provider
            });
            format!(
                "TensorRT, fp16={}, engine_cache={}",
                options.fp16,
                cache.display()
            )
        } else {
            "CUDA".to_owned()
        };
        #[cfg(not(feature = "ort-tensorrt"))]
        let provider = "CUDA";
        providers.push(CUDA::default().with_device_id(0).build().error_on_failure());
        let session = Session::builder()
            .map_err(OrtDetectorError::from)?
            .with_execution_providers(providers)
            .map_err(|error| OrtDetectorError::Ort(error.into()))?
            .commit_from_file(model_path)
            .map_err(OrtDetectorError::from)?;

        let model = model_input(&session)?;
        let labels = labels(options.detector.labels.as_deref(), &session);
        let driver = CudaDriver::retain_primary().map_err(OrtDetectorError::from)?;
        let kernels = driver.fit_kernels().map_err(OrtDetectorError::from)?;
        let max_batch = if open_batch(&session) {
            options.max_batch
        } else {
            if options.max_batch > 1 {
                pp_warn!(
                    pp_log: &pp_log,
                    "the model takes one picture at a time: a max_batch of {} is run one by one",
                    options.max_batch
                );
            }
            1
        };
        let tensor = driver
            .batch_tensor(model.0, model.1, max_batch)
            .map_err(OrtDetectorError::from)?;
        let memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(OrtDetectorError::from)?;
        let hw_device_ctx = device.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition; only the pointer's identity is kept, and the reference
        // held beside it keeps that identity from being reused.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *mut ffi::AVHWDeviceContext };

        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batches of up to {max_batch}, {} labels, on {provider}; {linked}",
            model.0,
            model.1,
            labels.len()
        );
        if labels.is_empty() {
            pp_warn!(
                pp_log: &pp_log,
                "the model names no classes and none were given: detections carry class numbers only"
            );
        }
        let interval = Interval::new(options.detector.interval);
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            session,
            options: options.detector,
            labels,
            model,
            tensor,
            kernels,
            driver,
            memory,
            device_ctx,
            _hw_device_ctx: hw_device_ctx,
            interval,
            max_batch,
            held: Vec::new(),
            context: None,
        })))
    }
}

/// A picture of the batch under way: how it was fitted into the tensor,
/// where it is looked at, or `None` where it is let by.
struct Held {
    buf: MediaBuffer,
    letterbox: Option<Letterbox>,
}

/// Whether `session`'s first input leaves its batch open, so that the model
/// can be run on several pictures at once.
fn open_batch(session: &Session) -> bool {
    session
        .inputs()
        .first()
        .and_then(|input| input.dtype().tensor_shape())
        .and_then(|shape| shape.iter().next().copied())
        .is_some_and(|batch| batch < 0)
}

/// What TensorRT's profile names: the model's first input.
#[cfg(feature = "ort-tensorrt")]
struct BatchInput {
    name: String,
    width: u32,
    height: u32,
}

/// The model's first input, where its batch is left open — read by loading
/// the model on the CPU first, as TensorRT has to be told it before the
/// session that runs the model is made.
#[cfg(feature = "ort-tensorrt")]
fn batch_input(model: &Path) -> std::result::Result<Option<BatchInput>, OrtDetectorError> {
    let probe = Session::builder()?.commit_from_file(model)?;
    if !open_batch(&probe) {
        return Ok(None);
    }
    let (width, height) = model_input(&probe)?;
    Ok(Some(BatchInput {
        name: probe.inputs()[0].name().to_owned(),
        width,
        height,
    }))
}

/// `$XDG_CACHE_HOME/media-pp/tensorrt`, or the platform's own cache
/// directory, or the temporary one where there is neither.
#[cfg(feature = "ort-tensorrt")]
fn default_engine_cache() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("media-pp").join("tensorrt")
}

impl Detecting {
    /// Fits `frame` into the `slot`th input of the device tensor and says
    /// how it was fitted. The kernel is launched, not waited for: the run
    /// waits once for every picture of its batch.
    fn fit(
        &mut self,
        frame: &ffmpeg::frame::Video,
        slot: usize,
    ) -> std::result::Result<Letterbox, OrtDetectorError> {
        let surface = frame::validate(
            frame,
            ElementType::CudaOrtDetector,
            self.device_ctx,
            CudaSurfaces::NV12_OR_BGRA,
        )?;
        let size = (frame.width(), frame.height());
        let letterbox = Letterbox::new(size, self.model);
        let fit = Fit {
            model: self.model,
            offset: letterbox.offset,
            scaled: letterbox.scaled,
        };
        if surface.layout == ffmpeg::format::Pixel::NV12 {
            let source = Nv12Surface::from_frame(frame).ok_or(OrtDetectorError::MissingSurface)?;
            self.driver.fit_nv12(
                &self.kernels,
                &self.tensor,
                slot,
                fit,
                source,
                size,
                &YuvToBgra::of_frame(frame),
            )?;
        } else {
            let source = BgraSurface::from_frame(frame).ok_or(OrtDetectorError::MissingSurface)?;
            self.driver
                .fit_bgra(&self.kernels, &self.tensor, slot, fit, source, size)?;
        }
        Ok(letterbox)
    }

    /// How many of the held pictures are fitted into the tensor.
    fn fitted(&self) -> usize {
        self.held
            .iter()
            .filter(|held| held.letterbox.is_some())
            .count()
    }

    /// Runs the model on the pictures fitted into the first `letterboxes`
    /// inputs of the tensor, and says what it found in each.
    fn detect(
        &mut self,
        letterboxes: &[Letterbox],
    ) -> std::result::Result<Vec<Vec<Detection>>, OrtDetectorError> {
        // The kernels ran on this driver's context; the model reads the
        // tensor on ONNX Runtime's own stream. Waiting here is what puts
        // every fitted picture before the read.
        self.driver.synchronize()?;
        let pictures = letterboxes.len();
        let shape = Shape::new([
            pictures as i64,
            3,
            i64::from(self.model.1),
            i64::from(self.model.0),
        ]);
        debug_assert!(
            pictures <= self.max_batch
                && self.tensor.floats()
                    == 3 * (self.model.0 * self.model.1) as usize * self.max_batch
        );
        // SAFETY: the pointer is this element's own device allocation of
        // `tensor.floats()` floats — room for `max_batch` inputs, of which
        // `shape` takes the first `pictures` — in the primary context ONNX
        // Runtime's CUDA provider also runs in; it outlives the run, the
        // view being dropped before this returns.
        let input = unsafe {
            TensorRefMut::<f32>::from_raw(
                self.memory.clone(),
                self.tensor.pointer() as usize as *mut _,
                shape,
            )?
        };
        let outputs = self.session.run(inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        decode_batch(output, letterboxes, &self.options)
    }

    /// Runs the model on what is held, and hands every held picture on in
    /// the order it came — those looked at carrying what was found in them.
    /// Where the run fails, the pictures it was for go with it.
    fn run_held(&mut self, out: &mut Output) -> std::result::Result<(), OrtDetectorError> {
        let held = std::mem::take(&mut self.held);
        let letterboxes: Vec<Letterbox> = held.iter().filter_map(|held| held.letterbox).collect();
        let mut found = if letterboxes.is_empty() {
            Vec::new()
        } else {
            self.detect(&letterboxes)?
        }
        .into_iter();
        for Held { buf, letterbox } in held {
            let items = letterbox.and_then(|_| found.next());
            match items {
                Some(items) => {
                    let detections =
                        Detections::new(Arc::clone(&self.name), Arc::clone(&self.labels), items);
                    out.push(detections.attach_to(buf));
                }
                // Let by unlooked-at, carrying nothing, which says so.
                None => out.push(buf),
            }
        }
        Ok(())
    }

    /// Puts `error` — one picture's, where the others of its batch go on in
    /// the same call — on the pipeline's bus.
    fn report(&self, error: OrtDetectorError) {
        match &self.context {
            Some(context) => context.bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::CudaOrtDetector,
                    name: Arc::clone(&self.name),
                    error: error.into(),
                },
            ),
            None => pp_error!(pp_log: &self.pp_log, "{error}"),
        }
    }
}

impl Element for Detecting {
    fn attach_context(&mut self, context: &Arc<Context>) {
        self.context = Some(Arc::clone(context));
    }

    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaOrtDetector
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Detecting {
    /// Decoded NV12 or BGRA video on the GPU.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::Cuda,
        ))
    }

    /// A picture with no [`BatchSlot`] is a batch of its own, run at once.
    /// The pictures of a batch are held, each fitted as it comes, and run
    /// together when its last has come, or as soon as `max_batch` of them
    /// are.
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtDetectorError::UnsupportedBuffer {
                detector: "CudaOrtDetector",
                wanted: "CUDA video frames",
                got: buf.kind(),
            }
            .into());
        };
        let slot = buf
            .metadata()
            .and_then(|metadata| metadata.get::<BatchSlot>())
            .copied();
        // What is held is of another batch, whose last picture never came:
        // it is run as it is rather than kept waiting.
        let held_batch = self.held.first().and_then(|held| {
            held.buf
                .metadata()
                .and_then(|metadata| metadata.get::<BatchSlot>())
                .map(|slot| slot.batch)
        });
        if !self.held.is_empty() && held_batch != slot.map(|slot| slot.batch) {
            self.run_held(out)?;
        }
        let last = slot.is_none_or(|slot| slot.is_last());

        let letterbox = if self.interval.look(&buf) {
            match self.fit(frame, self.fitted()) {
                Ok(letterbox) => Some(letterbox),
                // This picture alone failed. Where nothing else is to go on
                // from this call, the failure is this call's; where the
                // batch it closes is, the batch goes on and the failure is
                // reported beside it.
                Err(error) if !(last && !self.held.is_empty()) => return Err(error.into()),
                Err(error) => {
                    self.report(error);
                    self.run_held(out)?;
                    return Ok(());
                }
            }
        } else {
            None
        };
        self.held.push(Held { buf, letterbox });
        // A full tensor is run at once: what it holds need not wait for the
        // rest of a batch larger than it.
        if last || self.fitted() == self.max_batch {
            self.run_held(out)?;
        }
        Ok(())
    }

    /// The end of the stream: a batch cut short is run as it is.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        Ok(self.run_held(out)?)
    }

    fn reset(&mut self) {
        self.held.clear();
        self.interval.restart();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    #[cfg(feature = "ort-tensorrt")]
    use super::super::runtime::LibraryVersion;
    use super::super::runtime::RuntimeShortfall;
    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, CudaUpload, SwOrtDetector};
    use crate::platform::cuda::CudaFrameFormat;

    #[cfg(feature = "ort-tensorrt")]
    #[test]
    fn tensorrt_is_used_where_the_machine_has_it_and_policy_allows() {
        let old_tensorrt = RuntimeShortfall::Outdated {
            library: "TensorRT",
            found: LibraryVersion {
                major: 10,
                minor: 3,
                patch: 0,
            },
            needed: LibraryVersion {
                major: 10,
                minor: 15,
                patch: 1,
            },
        };
        let cuda_only = || CudaRuntime::CudaOnly {
            tensorrt: old_tensorrt.clone(),
        };
        let unavailable = || CudaRuntime::Unavailable {
            cuda: RuntimeShortfall::Missing {
                libraries: vec!["libcudart.so.13"],
            },
        };
        use UseTensorRtPolicy::{Off, Preferred, Required};

        for policy in [Off, Preferred, Required] {
            assert!(matches!(
                choose(unavailable(), policy),
                Err(OrtDetectorError::CudaRuntimeUnavailable(
                    RuntimeShortfall::Missing { .. }
                ))
            ));
        }
        assert!(!choose(CudaRuntime::TensorRt, Off).unwrap());
        assert!(choose(CudaRuntime::TensorRt, Preferred).unwrap());
        assert!(choose(CudaRuntime::TensorRt, Required).unwrap());
        assert!(!choose(cuda_only(), Off).unwrap());
        assert!(!choose(cuda_only(), Preferred).unwrap());
        assert!(matches!(
            choose(cuda_only(), Required),
            Err(OrtDetectorError::TensorRtUnavailable(shortfall)) if shortfall == old_tensorrt
        ));
    }

    /// Everything `stage` hands on, kept.
    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// What `detector` found in `buf`, the picture it handed on checked to
    /// be the one it was given.
    fn found(detector: &mut dyn RawSinkAndPads, buf: MediaBuffer) -> Detections {
        let kept = capture(detector.pads());
        detector.sink().consume(buf).expect("detects");
        let kept = kept.lock().unwrap();
        let MediaBuffer::Video(frame) = &kept[0] else {
            panic!("a picture goes on");
        };
        assert_eq!(frame.pts(), Some(150), "the picture it was given");
        kept[0]
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .expect("carries Detections")
            .clone()
    }

    trait RawSinkAndPads {
        fn sink(&mut self) -> &mut dyn RawSink;
        fn pads(&mut self) -> &mut dyn SrcPads;
    }

    impl<T: RawSink + SrcPads> RawSinkAndPads for T {
        fn sink(&mut self) -> &mut dyn RawSink {
            self
        }

        fn pads(&mut self) -> &mut dyn SrcPads {
            self
        }
    }

    fn iou(a: &crate::elements::Detection, b: &crate::elements::Detection) -> f32 {
        let w = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0);
        let h = ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
        let inter = w * h;
        inter / (a.width * a.height + b.width * b.height - inter)
    }

    /// Whatever this machine has, the answer is one of the three, and asking
    /// costs a load of the driver — printed, for whoever runs it. A build
    /// without `ort-tensorrt` never answers TensorRT.
    #[test]
    fn the_runtime_is_told_without_a_model() {
        let asked = std::time::Instant::now();
        let runtime = CudaOrtDetector::runtime();
        eprintln!("{runtime:?} in {:?}", asked.elapsed());
        if let CudaRuntime::Unavailable {
            cuda: RuntimeShortfall::Missing { libraries },
        } = &runtime
        {
            assert!(!libraries.is_empty());
        }
        #[cfg(not(feature = "ort-tensorrt"))]
        assert!(!matches!(runtime, CudaRuntime::TensorRt));
    }

    /// The point of the element: on the GPU it finds what the CPU detector
    /// finds in the same picture — the same objects, in the same places —
    /// from NV12 and from BGRA alike. Needs a model, a video, CUDA and the
    /// runtime libraries; skipped, saying so, without them. Runs on CUDA
    /// alone, which starts at once: TensorRT reads the same tensor.
    #[test]
    fn it_finds_on_the_gpu_what_the_cpu_finds() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Ok(device) = CudaDevice::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        if let CudaRuntime::Unavailable { cuda } = CudaOrtDetector::runtime() {
            eprintln!("skipping: the CUDA runtime cannot be used ({cuda})");
            return;
        }
        let options = CudaOrtDetectorOptions {
            #[cfg(feature = "ort-tensorrt")]
            tensorrt: UseTensorRtPolicy::Off,
            ..CudaOrtDetectorOptions::default()
        };
        let mut cpu = SwOrtDetector::new("cpu", &model, options.detector.clone()).expect("loads");
        let expected = found(
            &mut cpu,
            MediaBuffer::video(crate::test_support::nth_picture(
                &video,
                150,
                ffmpeg::format::Pixel::NV12,
            )),
        );
        assert!(
            !expected.items.is_empty(),
            "the picture has something in it"
        );

        for (format, layout) in [
            (CudaFrameFormat::Nv12, ffmpeg::format::Pixel::NV12),
            (CudaFrameFormat::Bgra, ffmpeg::format::Pixel::BGRA),
        ] {
            let mut upload = CudaUpload::new("upload", &device, format);
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(crate::test_support::nth_picture(
                    &video, 150, layout,
                )))
                .expect("uploads");
            let on_gpu = uploaded.lock().unwrap().remove(0);
            let mut gpu = CudaOrtDetector::new("gpu", &device, &model, options.clone())
                .expect("loads on CUDA");
            let actual = found(&mut gpu, on_gpu);
            for wanted in expected.items.iter().filter(|found| found.score > 0.5) {
                let best = actual
                    .items
                    .iter()
                    .filter(|found| found.class_id == wanted.class_id)
                    .map(|found| iou(found, wanted))
                    .fold(0.0, f32::max);
                assert!(
                    best > 0.8,
                    "{format:?}: {wanted:?} found at IoU {best} only"
                );
            }
        }
    }

    /// A picture `n` of `video`, on the GPU, as the `index`th of a batch of
    /// `size` a mux handed on.
    fn in_batch(
        device: &CudaDevice,
        video: &str,
        n: usize,
        index: usize,
        size: usize,
    ) -> MediaBuffer {
        use crate::buffer::Metadata;
        use crate::elements::vision::batch::{StreamId, StreamOrigin};
        let mut upload = CudaUpload::new("upload", device, CudaFrameFormat::Nv12);
        let uploaded = capture(&mut upload);
        upload
            .consume(MediaBuffer::video(crate::test_support::nth_picture(
                video,
                n,
                ffmpeg::format::Pixel::NV12,
            )))
            .expect("uploads");
        let buf = uploaded.lock().unwrap().remove(0);
        buf.with_metadata(
            Metadata::default()
                .with(StreamOrigin {
                    id: StreamId(index as u64),
                    name: Arc::from(format!("camera {index}")),
                    generation: 0,
                })
                .with(BatchSlot {
                    batch: 0,
                    index,
                    size,
                }),
        )
    }

    /// The pictures of a batch are held until its last has come, then run
    /// through the model at once and handed on in order, each carrying what
    /// one picture at a time finds in it; a batch cut short by the end of
    /// the stream is run as it is. Needs a model whose batch is left open —
    /// YOLO11n exported with `dynamic=True` — a video, CUDA and the runtime
    /// libraries; skipped, saying so, without them.
    #[test]
    fn a_batch_finds_what_one_picture_at_a_time_finds() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Ok(device) = CudaDevice::new() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        if let CudaRuntime::Unavailable { cuda } = CudaOrtDetector::runtime() {
            eprintln!("skipping: the CUDA runtime cannot be used ({cuda})");
            return;
        }
        let probe = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(&model))
            .expect("loads");
        if !open_batch(&probe) {
            eprintln!("skipping: {model} takes one picture at a time");
            return;
        }
        let options = |max_batch| CudaOrtDetectorOptions {
            max_batch,
            #[cfg(feature = "ort-tensorrt")]
            tensorrt: UseTensorRtPolicy::Off,
            ..CudaOrtDetectorOptions::default()
        };
        let pictures = [100, 150, 200];
        let found_in = |buf: &MediaBuffer| -> Vec<(usize, f32, f32)> {
            buf.metadata()
                .and_then(|metadata| metadata.get::<Detections>())
                .expect("carries Detections")
                .items
                .iter()
                .filter(|item| item.score > 0.5)
                .map(|item| (item.class_id, item.x, item.y))
                .collect()
        };

        let mut alone = CudaOrtDetector::new("alone", &device, &model, options(1)).expect("loads");
        let one_at_a_time = capture(&mut alone);
        for (index, n) in pictures.into_iter().enumerate() {
            alone
                .consume(in_batch(&device, &video, n, index, pictures.len()))
                .expect("detects");
            assert_eq!(
                one_at_a_time.lock().unwrap().len(),
                index + 1,
                "one at a time, each goes on at once"
            );
        }

        let mut batched =
            CudaOrtDetector::new("batched", &device, &model, options(4)).expect("loads");
        let together = capture(&mut batched);
        for (index, n) in pictures.into_iter().enumerate() {
            batched
                .consume(in_batch(&device, &video, n, index, pictures.len()))
                .expect("detects");
            let handed_on = together.lock().unwrap().len();
            let last = index + 1 == pictures.len();
            assert_eq!(
                handed_on,
                if last { pictures.len() } else { 0 },
                "held to the last"
            );
        }
        let one_at_a_time = one_at_a_time.lock().unwrap();
        let together = together.lock().unwrap();
        for (index, (alone, batched)) in one_at_a_time.iter().zip(together.iter()).enumerate() {
            let MediaBuffer::Video(frame) = batched else {
                panic!("a picture goes on");
            };
            assert_eq!(
                batched
                    .metadata()
                    .and_then(|m| m.get::<BatchSlot>())
                    .map(|s| s.index),
                Some(index),
                "in the order they came"
            );
            let (alone, batched) = (found_in(alone), found_in(batched));
            assert!(!alone.is_empty(), "picture {index} has something in it");
            assert_eq!(alone.len(), batched.len(), "picture {index}");
            for (a, b) in alone.iter().zip(&batched) {
                assert_eq!(a.0, b.0, "picture {index}");
                assert!(
                    (a.1 - b.1).abs() < 0.01 && (a.2 - b.2).abs() < 0.01,
                    "picture {index} ({}x{}): {a:?} against {b:?}",
                    frame.width(),
                    frame.height()
                );
            }
        }
        drop(together);

        // Two of a batch of three, then the end: both go on, looked at.
        let mut cut =
            CudaOrtDetector::new("cut short", &device, &model, options(4)).expect("loads");
        let ended = capture(&mut cut);
        for (index, n) in pictures.into_iter().take(2).enumerate() {
            cut.consume(in_batch(&device, &video, n, index, 3))
                .expect("detects");
        }
        assert!(ended.lock().unwrap().is_empty(), "waiting for the third");
        cut.stream_event(&crate::stream::StreamEvent::Eos)
            .expect("drained");
        let ended = ended.lock().unwrap();
        assert_eq!(ended.len(), 2);
        assert!(ended.iter().all(|buf| !found_in(buf).is_empty()));
    }
}
