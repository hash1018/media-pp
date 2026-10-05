//! [`CudaOrtDetector`]: object detection on CUDA pictures, each handed on
//! with what was found in it, without the picture leaving the GPU.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ffmpeg_next::{self as ffmpeg, ffi};
use ort::{
    ep::{CUDA, TensorRT},
    inputs,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::{Shape, TensorRefMut},
};

use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    element::{Element, ElementType, element_pp_log},
    error::Result,
    platform::cuda::{
        CudaDevice,
        driver::{BgraSurface, CudaDriver, CudaTensor, Fit, FitKernels, Nv12Surface, YuvToBgra},
        frame::{self, CudaSurfaces},
    },
    platform::ffmpeg::AvBufferRef,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::{
    Detections, Letterbox, OrtDetectorError, OrtDetectorOptions, attach, decode, labels,
    model_input,
};
use super::runtime::{self, CudaRuntime};

/// Whether a [`CudaOrtDetector`] runs its model through TensorRT.
///
/// TensorRT compiles the model into an engine for this GPU — fused layers,
/// the fastest kernel for each found by timing them, half precision where
/// [`CudaOrtDetectorOptions::fp16`] allows — and runs it several times
/// faster than CUDA does, at the cost of minutes building the engine the
/// first time and TensorRT's libraries at run time. CUDA alone runs the
/// model operator by operator on cuDNN and cuBLAS, and starts at once.
/// Either way an operator TensorRT cannot run falls to CUDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UseTensorRtPolicy {
    /// CUDA alone.
    Off,
    /// TensorRT where the machine has it; CUDA alone, with a warning,
    /// where it does not.
    #[default]
    Preferred,
    /// TensorRT, or no detector: [`CudaOrtDetector::new`] refuses with
    /// [`OrtDetectorError::TensorRtUnavailable`] where its libraries are
    /// missing or too old, and with the provider's own error where they are
    /// there but it cannot start.
    Required,
}

/// Whether to run on TensorRT, from what the machine has and what was
/// asked; the warning, if one is due, is the caller's to say.
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
    /// Whether TensorRT runs the model, and what happens where it cannot.
    pub tensorrt: UseTensorRtPolicy,
    /// Whether TensorRT may run in half precision — about twice as fast on
    /// a GPU with tensor cores, at a detector's tolerance.
    pub fp16: bool,
    /// Where TensorRT keeps the engine it builds for this model and GPU.
    /// Building one takes minutes; the first detector to start builds it
    /// and every later one loads it in under a second. `None` keeps it in
    /// the user's cache directory, under `media-pp/tensorrt`.
    pub engine_cache: Option<PathBuf>,
}

impl Default for CudaOrtDetectorOptions {
    fn default() -> Self {
        Self {
            detector: OrtDetectorOptions::default(),
            tensorrt: UseTensorRtPolicy::default(),
            fp16: true,
            engine_cache: None,
        }
    }
}

/// Runs a YOLO detector on each CUDA picture and hands the picture on
/// unchanged, carrying the [`Detections`] found in it — what
/// [`SwOrtDetector`](super::super::SwOrtDetector) does on the CPU, with the
/// picture never leaving the GPU: a kernel fits it into the model's input
/// in device memory, and TensorRT reads it there.
///
/// It takes NV12 or BGRA CUDA pictures of any size from the same
/// [`CudaDevice`] as the rest of the pipeline — a decoder's, a compositor's
/// — and the boxes it finds are fractions of each picture, as
/// [`Detections`] describes.
///
/// # Requirements
///
/// The `ort-cuda` feature, which fetches ONNX Runtime's CUDA and TensorRT
/// build, and at run time a driver for CUDA 13, the CUDA 13.2 runtime,
/// cuDNN 9.23 and TensorRT 10.15, or newer within those majors, where the
/// loader finds them (`LD_LIBRARY_PATH` on Linux). Without TensorRT it runs
/// on CUDA alone, saying so, unless [`UseTensorRtPolicy::Required`] says not
/// to; without CUDA it cannot run, and says that —
/// [`CudaOrtDetector::runtime`] tells which before a model is loaded.
///
/// The first start with TensorRT builds an engine for the model and GPU,
/// which takes minutes — see [`CudaOrtDetectorOptions::engine_cache`].
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
    /// What this machine can run a detector on, found by asking the loader
    /// for the libraries ONNX Runtime's providers open — the driver, CUDA 13
    /// with cuBLAS and cuRAND, cuDNN 9, and TensorRT 10 — and those that
    /// say their version for it, without a model, a device or a session.
    /// For an application deciding whether to offer detection on the GPU at
    /// all; [`Self::new`] asks the same question first.
    ///
    /// The versions taken are those ONNX Runtime's build was made against:
    /// a driver for CUDA 13.0 or newer, the CUDA 13.2 runtime, cuDNN 9.23.2
    /// and TensorRT 10.15.1. Whether there is a GPU, [`CudaDevice::new`]
    /// answers.
    pub fn runtime() -> CudaRuntime {
        if let Some(cuda) = runtime::shortfall(runtime::CUDA) {
            return CudaRuntime::Unavailable { cuda };
        }
        match runtime::shortfall(runtime::TENSORRT) {
            None => CudaRuntime::TensorRt,
            Some(tensorrt) => CudaRuntime::CudaOnly { tensorrt },
        }
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
        let cache = options
            .engine_cache
            .clone()
            .unwrap_or_else(default_engine_cache);

        let runtime = Self::runtime();
        if let (CudaRuntime::CudaOnly { tensorrt }, UseTensorRtPolicy::Preferred) =
            (&runtime, options.tensorrt)
        {
            pp_warn!(pp_log: &pp_log, "TensorRT cannot be used ({tensorrt}): running on CUDA alone");
        }
        let tensorrt = choose(runtime, options.tensorrt)?;

        let mut providers = Vec::new();
        if tensorrt {
            if let Err(error) = std::fs::create_dir_all(&cache) {
                pp_warn!(pp_log: &pp_log, "no engine cache at {}: {error}", cache.display());
            }
            // Device 0 for both providers: the one `CudaDevice` opens, which
            // takes no ordinal, and so where every incoming picture is.
            let provider = TensorRT::default()
                .with_device_id(0)
                .with_fp16(options.fp16)
                .with_engine_cache(true)
                .with_engine_cache_path(cache.display())
                .with_timing_cache(true)
                .with_timing_cache_path(cache.display())
                .build();
            // ONNX Runtime passes over a provider that fails to register and
            // tries the next, which would be CUDA running in TensorRT's place.
            providers.push(if options.tensorrt == UseTensorRtPolicy::Required {
                provider.error_on_failure()
            } else {
                provider
            });
        }
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
        let tensor = driver
            .tensor(model.0, model.1)
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
            "model loaded: path={path}, input={}x{}, {} labels, tensorrt={}, fp16={}, engine_cache={}",
            model.0,
            model.1,
            labels.len(),
            tensorrt,
            options.fp16,
            cache.display()
        );
        if labels.is_empty() {
            pp_warn!(
                pp_log: &pp_log,
                "the model names no classes and none were given: detections carry class numbers only"
            );
        }
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
        })))
    }
}

/// `$XDG_CACHE_HOME/media-pp/tensorrt`, or the platform's own cache
/// directory, or the temporary one where there is neither.
fn default_engine_cache() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("media-pp").join("tensorrt")
}

impl Detecting {
    /// Fits `frame` into the device tensor and says how it was fitted.
    fn fit(
        &mut self,
        frame: &ffmpeg::frame::Video,
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
                fit,
                source,
                size,
                &YuvToBgra::of_frame(frame),
            )?;
        } else {
            let source = BgraSurface::from_frame(frame).ok_or(OrtDetectorError::MissingSurface)?;
            self.driver
                .fit_bgra(&self.kernels, &self.tensor, fit, source, size)?;
        }
        // The kernel runs on this driver's context; the model reads the
        // tensor on ONNX Runtime's own stream. Waiting here is what puts the
        // fitted picture before the read.
        self.driver.synchronize()?;
        Ok(letterbox)
    }

    /// Looks at `frame`, and says what it found.
    fn detect(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Detections, OrtDetectorError> {
        let letterbox = self.fit(frame)?;
        let shape = Shape::new([1, 3, i64::from(self.model.1), i64::from(self.model.0)]);
        // SAFETY: the pointer is this element's own device allocation of
        // `tensor.floats()` floats — exactly `shape` — in the primary context
        // ONNX Runtime's CUDA provider also runs in; it outlives the run, the
        // view being dropped before this returns.
        let input = unsafe {
            TensorRefMut::<f32>::from_raw(
                self.memory.clone(),
                self.tensor.pointer() as usize as *mut _,
                shape,
            )?
        };
        debug_assert_eq!(
            self.tensor.floats(),
            3 * (self.model.0 * self.model.1) as usize
        );
        let outputs = self.session.run(inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        Ok(Detections {
            detector: Arc::clone(&self.name),
            labels: Arc::clone(&self.labels),
            items: decode(output, &letterbox, &self.options)?,
        })
    }
}

impl Element for Detecting {
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

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtDetectorError::UnsupportedBuffer {
                detector: "CudaOrtDetector",
                wanted: "CUDA video frames",
                got: buf.kind(),
            }
            .into());
        };
        let detections = self.detect(frame)?;
        out.push(attach(buf, detections));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::super::runtime::{LibraryVersion, RuntimeShortfall};
    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, CudaUpload, SwOrtDetector};
    use crate::platform::cuda::CudaFrameFormat;

    #[test]
    fn tensorrt_is_used_where_the_machine_has_it_and_policy_allows() {
        let old_tensorrt = RuntimeShortfall::Outdated {
            library: "libnvinfer.so.10",
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

    /// The 150th picture of `video`, as `format` in system memory.
    fn picture(video: &str, format: ffmpeg::format::Pixel) -> ffmpeg::frame::Video {
        let mut input = ffmpeg::format::input(video).expect("the video opens");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("a video stream");
        let index = stream.index();
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .and_then(|context| context.decoder().video())
            .expect("a decoder");
        let mut decoded = ffmpeg::frame::Video::empty();
        let mut seen = 0;
        for (stream, packet) in input.packets() {
            if stream.index() != index {
                continue;
            }
            decoder.send_packet(&packet).expect("decodes");
            while decoder.receive_frame(&mut decoded).is_ok() {
                seen += 1;
                if seen == 150 {
                    let mut converted = ffmpeg::frame::Video::empty();
                    ffmpeg::software::scaling::Context::get(
                        decoded.format(),
                        decoded.width(),
                        decoded.height(),
                        format,
                        decoded.width(),
                        decoded.height(),
                        ffmpeg::software::scaling::Flags::BILINEAR,
                    )
                    .and_then(|mut context| context.run(&decoded, &mut converted))
                    .expect("converts");
                    converted.set_pts(Some(150));
                    return converted;
                }
            }
        }
        panic!("the video is shorter than 150 pictures");
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
    /// costs a few loads — printed, for whoever runs it with the libraries
    /// there.
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
            tensorrt: UseTensorRtPolicy::Off,
            ..CudaOrtDetectorOptions::default()
        };
        let mut cpu = SwOrtDetector::new("cpu", &model, options.detector.clone()).expect("loads");
        let expected = found(
            &mut cpu,
            MediaBuffer::video(picture(&video, ffmpeg::format::Pixel::NV12)),
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
                .consume(MediaBuffer::video(picture(&video, layout)))
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
}
