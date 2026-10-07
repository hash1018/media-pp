//! [`CudaOrtClassifier`]: what a detector found, classified by a second
//! model on CUDA pictures without them leaving the GPU.

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

use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::{Classification, Detections},
    error::Result,
    platform::cuda::{
        CudaDevice,
        driver::{BgraSurface, CudaDriver, CudaTensor, Fit, FitKernels, Nv12Surface, YuvToBgra},
        frame::{self, CudaSurfaces},
    },
    platform::ffmpeg::AvBufferRef,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::classify::{
    Crop, Input, Memory, OrtClassifierOptions, Plan, answer, apply, best, classifier_input,
};
use super::super::{OrtError, labels};
use super::runtime::{self, CudaRuntime};
use crate::orientation::{Orientation, Orientations};

/// The batch TensorRT builds a classifier's engine to be quickest at: a
/// picture has a few objects to classify at once, seldom the most there is
/// room for. Every batch from 1 to the most runs on the one engine. On an
/// RTX 3050, MobileNetV2's engine built for 2 ran one to three objects
/// 2–10% faster than one built for 8, and eight 6% slower; a pipeline
/// classifying every object on every picture went 1% faster.
#[cfg(feature = "ort-tensorrt")]
const TENSORRT_OPT_BATCH: usize = 2;

/// The most objects classified in one run of a model that takes any
/// number at once.
const MAX_BATCH: usize = 32;

/// Classifies the objects a detector found on each CUDA picture with a
/// second model, without the picture leaving the GPU — what
/// [`SwOrtClassifier`](crate::elements::SwOrtClassifier) does on the CPU:
/// each object's box is cut from the picture and stretched into the
/// model's input by a kernel, in device memory, several objects to one
/// run, and ONNX Runtime's CUDA provider reads them there.
///
/// It takes NV12 or BGRA CUDA pictures from the same [`CudaDevice`] as the
/// rest of the pipeline. An object's box is taken to whole 2x2 blocks on
/// an NV12 picture. Built with `ort-tensorrt`, it runs the model through
/// TensorRT in half precision where TensorRT can run, and on CUDA alone
/// where it cannot — a detector's `UseTensorRtPolicy::Preferred`. A
/// classifier's batch is however many objects a picture has, so the engine
/// is built for every batch from one to the most it runs at once, rather
/// than rebuilt for each; it is kept beside the detectors' in the same
/// cache. Through TensorRT a small model's run is several times shorter,
/// and shorter still beside a detector on the same GPU, where CUDA's many
/// small kernels each wait their turn behind the detector's.
///
/// # Requirements
///
/// The `ort-cuda` feature, and at run time what
/// [`CudaOrtDetector::runtime`](crate::elements::CudaOrtDetector::runtime)
/// says it needs of the CUDA side.
pub struct CudaOrtClassifier(FilterStage<Classifying>);

filter_stage!(CudaOrtClassifier);

/// What a [`CudaOrtClassifier`] does with each picture.
///
/// The device buffer and kernels come before the driver: fields drop in
/// order, and both are freed in the driver's context, which the driver
/// releases.
struct Classifying {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtClassifierOptions,
    labels: Arc<[Arc<str>]>,
    input: Input,
    /// How many inputs `tensor` holds.
    capacity: usize,
    memory: Memory,
    tensor: CudaTensor,
    kernels: FitKernels,
    /// Where an HDR picture is brought to SDR for the model; before the
    /// driver, whose context it is freed in.
    sdr: super::SdrCopy,
    driver: CudaDriver,
    device_memory: MemoryInfo<'static>,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *mut ffi::AVHWDeviceContext,
    _hw_device_ctx: Arc<AvBufferRef>,
    /// How each picture is turned to be shown.
    orientations: Orientations,
}

// SAFETY: the session, buffer and kernels are used by the one thread
// transforming at a time; `device_ctx` only ever has its address compared,
// and `&mut self` on every method that touches the rest rules out
// concurrent access — the reasoning `CudaOrtDetector` gives for its own.
unsafe impl Send for Classifying {}

impl CudaOrtClassifier {
    /// Loads the classification model at `model_path` to run on `device`'s
    /// GPU — the same [`CudaDevice`] every other CUDA element in the
    /// pipeline was built from.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        model_path: impl AsRef<Path>,
        options: OrtClassifierOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaOrtClassifier, &name, None);
        let path = model_path.as_ref().display().to_string();

        let model_path = model_path.as_ref();

        let linked = runtime::Linked::ask();
        let runtime = runtime::runtime(&linked);
        #[cfg(feature = "ort-tensorrt")]
        if let CudaRuntime::CudaOnly { tensorrt } = &runtime {
            pp_warn!(pp_log: &pp_log, "TensorRT cannot be used ({tensorrt}): classifying on CUDA alone");
        }
        #[cfg(feature = "ort-tensorrt")]
        let tensorrt = runtime == CudaRuntime::TensorRt;
        if let CudaRuntime::Unavailable { cuda } = runtime {
            return Err(OrtError::CudaRuntimeUnavailable(cuda).into());
        }

        // The input is read first, on the CPU: TensorRT is told the batches
        // it will be given before its session is made.
        let probe = Session::builder()
            .map_err(OrtError::from)?
            .commit_from_file(model_path)
            .map_err(OrtError::from)?;
        let input = classifier_input(&probe)?;
        #[cfg(feature = "ort-tensorrt")]
        let input_name = probe.inputs()[0].name().to_owned();
        drop(probe);
        let capacity = input.batch.unwrap_or(MAX_BATCH).max(1);

        let mut providers = Vec::new();
        #[cfg(feature = "ort-tensorrt")]
        let provider = if tensorrt {
            let profile = input.batch.is_none().then(|| super::BatchProfile {
                min: 1,
                opt: TENSORRT_OPT_BATCH.min(capacity),
                max: capacity,
            });
            let root = super::default_engine_cache();
            let cache = super::engine_directory(&root, profile);
            if let Err(error) = std::fs::create_dir_all(&cache) {
                pp_warn!(pp_log: &pp_log, "no engine cache at {}: {error}", cache.display());
            }
            let mut provider = TensorRT::default()
                .with_device_id(0)
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(cache.display())
                .with_timing_cache(true)
                .with_timing_cache_path(root.display());
            if let Some(profile) = profile {
                let (width, height) = input.size;
                let shape = |batch: usize| format!("{input_name}:{batch}x3x{height}x{width}");
                provider = provider
                    .with_profile_min_shapes(shape(profile.min))
                    .with_profile_opt_shapes(shape(profile.opt))
                    .with_profile_max_shapes(shape(profile.max));
            }
            // Preferred: where TensorRT will not start, ONNX Runtime passes
            // over it to CUDA.
            providers.push(provider.build());
            "TensorRT, fp16"
        } else {
            "CUDA"
        };
        #[cfg(not(feature = "ort-tensorrt"))]
        let provider = "CUDA";
        providers.push(CUDA::default().with_device_id(0).build().error_on_failure());
        let session = Session::builder()
            .map_err(OrtError::from)?
            .with_execution_providers(providers)
            .map_err(|error| OrtError::Ort(error.into()))?
            .commit_from_file(model_path)
            .map_err(OrtError::from)?;
        let labels = labels(options.labels.as_deref(), &session);
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
            "model loaded: path={path}, input={}x{}, batch={:?}, {} labels, {:?}, on {provider}; {linked}",
            input.size.0,
            input.size.1,
            input.batch,
            labels.len(),
            options.input
        );
        if labels.is_empty() {
            pp_warn!(
                pp_log: &pp_log,
                "the model names no classes and none were given: answers carry class numbers only"
            );
        }
        Ok(Self(FilterStage::new(Classifying {
            name,
            pp_log,
            session,
            options,
            labels,
            input,
            capacity,
            memory: Memory::default(),
            tensor,
            kernels,
            sdr: super::SdrCopy::default(),
            driver,
            device_memory,
            device_ctx,
            _hw_device_ctx: hw_device_ctx,
            orientations: Orientations::default(),
        })))
    }
}

impl Classifying {
    /// Cuts `crop` of `frame`, turned the way the picture is shown, into
    /// input `slot` of the tensor.
    fn cut(
        &self,
        frame: &ffmpeg::frame::Video,
        sdr: Option<BgraSurface>,
        (left, top, width, height): Crop,
        orientation: Orientation,
        slot: usize,
    ) -> std::result::Result<(), OrtError> {
        let fit = Fit {
            model: self.input.size,
            offset: (0, 0),
            scaled: self.input.size,
            orientation,
        };
        if sdr.is_none() && frame_is_nv12(frame) {
            // On whole 2x2 blocks, so that the chroma the crop starts on is
            // the luma's.
            let (left, top) = (left & !1, top & !1);
            let (width, height) = ((width & !1).max(2), (height & !1).max(2));
            let whole = Nv12Surface::from_frame(frame).ok_or(OrtError::MissingSurface)?;
            let (x, y) = (u64::from(left), u64::from(top));
            let crop = Nv12Surface {
                luma: whole.luma + y * whole.luma_pitch as u64 + x,
                chroma: whole.chroma + (y / 2) * whole.chroma_pitch as u64 + x,
                ..whole
            };
            self.driver.fit_nv12(
                &self.kernels,
                &self.tensor,
                slot,
                fit,
                crop,
                (width, height),
                &YuvToBgra::of_frame(frame),
            )?;
        } else {
            let whole = match sdr {
                Some(sdr) => sdr,
                None => BgraSurface::from_frame(frame).ok_or(OrtError::MissingSurface)?,
            };
            let crop = BgraSurface {
                pixels: whole.pixels + u64::from(top) * whole.pitch as u64 + u64::from(left) * 4,
                ..whole
            };
            self.driver.fit_bgra(
                &self.kernels,
                &self.tensor,
                slot,
                fit,
                crop,
                (width, height),
            )?;
        }
        Ok(())
    }

    /// The model's answer for each of `crops` of `frame`, in order.
    fn classify(
        &mut self,
        frame: &ffmpeg::frame::Video,
        crops: &[Crop],
    ) -> std::result::Result<Vec<Option<(usize, f32)>>, OrtError> {
        let surface = frame::validate(
            frame,
            ElementType::CudaOrtClassifier,
            self.device_ctx,
            CudaSurfaces::NV12_BGRA_OR_P010,
        )?;
        // An HDR picture is cut from its SDR copy, made once for all its
        // objects.
        let sdr = if surface.layout == ffmpeg::format::Pixel::P010LE {
            Some(
                self.sdr
                    .of(&self.driver, frame, ElementType::CudaOrtClassifier)?,
            )
        } else {
            None
        };
        let (width, height) = self.input.size;
        let (scale, bias) = self.options.input.affine();
        let orientation = self.orientations.of(frame, &self.pp_log);
        let mut answers = Vec::with_capacity(crops.len());
        for group in crops.chunks(self.capacity) {
            for (slot, crop) in group.iter().enumerate() {
                self.cut(frame, sdr, *crop, orientation, slot)?;
            }
            // A model of fixed batch is handed exactly that many; one of
            // open batch, as many as there are.
            let rows = self.input.batch.unwrap_or(group.len());
            if (scale, bias) != ([1.0; 3], [0.0; 3]) {
                self.driver.scale_planes(
                    &self.kernels,
                    &self.tensor,
                    (width, height),
                    rows,
                    scale,
                    bias,
                )?;
            }
            // The kernels ran on this driver's context; the model reads the
            // tensor on ONNX Runtime's own stream.
            self.driver.synchronize()?;
            let shape = Shape::new([rows as i64, 3, i64::from(height), i64::from(width)]);
            // SAFETY: the pointer is this element's own device allocation of
            // `capacity` inputs, at least `rows` — exactly `shape` or more — in
            // the primary context ONNX Runtime's CUDA provider also runs in; it
            // outlives the run, the view being dropped before this returns.
            let input = unsafe {
                TensorRefMut::<f32>::from_raw(
                    self.device_memory.clone(),
                    self.tensor.pointer() as usize as *mut _,
                    shape,
                )?
            };
            let outputs = self.session.run(inputs![input])?;
            let output = outputs[0].try_extract_array::<f32>()?;
            let classes = output.shape().last().copied().unwrap_or(0);
            let flat: Vec<f32> = output.iter().copied().collect();
            for slot in 0..group.len() {
                let row = flat
                    .get(slot * classes..(slot + 1) * classes)
                    .unwrap_or(&[]);
                answers.push(best(row));
            }
        }
        Ok(answers)
    }
}

impl Element for Classifying {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaOrtClassifier
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Classifying {
    /// Device-resident frames; which of the two layouts is a runtime value.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                .with_layouts(crate::contract::PixelLayoutSet::GPU_SCALABLE),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "CudaOrtClassifier",
                wanted: "CUDA video frames",
                got: buf.kind(),
            }
            .into());
        };
        let Some(mut detections) = buf
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .cloned()
        else {
            out.push(buf);
            return Ok(());
        };
        let plan: Plan = self.memory.plan(
            &buf,
            &detections,
            &self.options,
            (frame.width(), frame.height()),
        );
        let crops: Vec<Crop> = plan.classify.iter().map(|(_, crop)| *crop).collect();
        let found = if crops.is_empty() {
            Vec::new()
        } else {
            self.classify(frame, &crops)?
        };
        let answers: Vec<(usize, Option<Classification>)> = plan
            .classify
            .iter()
            .zip(found)
            .map(|((index, _), found)| {
                let answer =
                    found.and_then(|found| answer(&self.name, &self.labels, &self.options, found));
                (*index, answer)
            })
            .collect();
        apply(&mut detections, &mut self.memory, plan, answers);
        out.push(detections.attach_to(buf));
        Ok(())
    }

    /// A seek or a flush: the objects after it are numbered anew.
    fn reset(&mut self) {
        self.memory.clear();
    }
}

/// Whether `frame`'s surface is NV12 — its own, validated before.
fn frame_is_nv12(frame: &ffmpeg::frame::Video) -> bool {
    crate::platform::cuda::frame::surface_layout(frame) == Some(ffmpeg::format::Pixel::NV12)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, CudaUpload, Detection, InputScale, SwOrtClassifier};
    use crate::platform::cuda::CudaFrameFormat;
    use crate::test_support::nth_picture;

    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// The top answer for each object `classifier` was handed `buf` with.
    fn answers<T: RawSink + SrcPads>(
        classifier: &mut T,
        buf: MediaBuffer,
    ) -> Vec<Option<(usize, f32)>> {
        let kept = capture(classifier);
        classifier.consume(buf).expect("classifies");
        let kept = kept.lock().unwrap();
        kept[0]
            .metadata()
            .and_then(|m| m.get::<Detections>())
            .expect("carries Detections")
            .items
            .iter()
            .map(|item| {
                item.classes
                    .first()
                    .map(|class| (class.class_id, class.score))
            })
            .collect()
    }

    /// On the GPU it says what the CPU says of the same objects, from NV12
    /// and from BGRA: the same model on the same boxes of the same picture,
    /// each input pixel the mean of those it covers on both. Where the CPU
    /// is sure of a box, the GPU names the same class at much the same
    /// score; where it is not, a thousand classes share the score and the
    /// answer turns on the last digit, so those are not compared. On NV12
    /// the scores differ by up to about 0.1: swscale interpolates the
    /// chroma the GPU takes as it is.
    #[test]
    fn it_says_on_the_gpu_what_the_cpu_says() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_CLASSIFIER"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_CLASSIFIER and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some((device, _serial)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let options = OrtClassifierOptions {
            min_score: 0.0,
            input: InputScale::ImageNet,
            ..OrtClassifierOptions::default()
        };
        let Ok(mut gpu) = CudaOrtClassifier::new("gpu", &device, &model, options.clone()) else {
            eprintln!("skipping: the CUDA runtime cannot be used here");
            return;
        };
        let mut cpu = SwOrtClassifier::new("cpu", &model, options).expect("loads");
        // A grid of boxes across the picture, each a sizeable piece of it.
        let items: Vec<Detection> = (0..6)
            .map(|i| {
                let (column, row) = ((i % 3) as f32, (i / 3) as f32);
                Detection::new(0, 0.9, 0.05 + column * 0.3, 0.1 + row * 0.45, 0.25, 0.4)
            })
            .collect();
        let found = Detections::new("detector", Arc::from([]), items);
        // An answer this sure is the model's, not the last digit's.
        const SURE: f32 = 0.3;

        // The CPU is handed the very picture the GPU is, so both read its
        // colours alike: an RGB24 one converted here would be swscale's
        // BT.601 where the GPU reads an untagged HD picture as BT.709.
        for (format, layout) in [
            (CudaFrameFormat::Nv12, ffmpeg::format::Pixel::NV12),
            (CudaFrameFormat::Bgra, ffmpeg::format::Pixel::BGRA),
        ] {
            let picture = nth_picture(&video, 150, layout);
            let expected = answers(
                &mut cpu,
                found.clone().attach_to(MediaBuffer::video(picture.clone())),
            );
            assert!(expected.iter().all(Option::is_some), "{expected:?}");
            let mut upload = CudaUpload::new("upload", &device, format);
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(picture))
                .expect("uploads");
            let on_gpu = uploaded.lock().unwrap().remove(0);
            let actual = answers(&mut gpu, found.clone().attach_to(on_gpu));
            eprintln!("{format:?}: cpu {expected:?}, gpu {actual:?}");
            let sure: Vec<_> = expected
                .iter()
                .zip(&actual)
                .filter_map(|(cpu, gpu)| Some(((*cpu).filter(|(_, score)| *score >= SURE)?, *gpu)))
                .collect();
            assert!(!sure.is_empty(), "{format:?}: the CPU is sure of no box");
            for ((class, score), gpu) in sure {
                let (gpu_class, gpu_score) = gpu.expect("the GPU answers too");
                assert_eq!(
                    gpu_class, class,
                    "{format:?}: the CPU said {class} at {score}"
                );
                assert!(
                    (gpu_score - score).abs() <= 0.15,
                    "{format:?}: class {class} at {gpu_score} on the GPU, {score} on the CPU"
                );
            }
        }
    }
}
