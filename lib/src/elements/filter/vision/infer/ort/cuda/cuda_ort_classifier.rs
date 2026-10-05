//! [`CudaOrtClassifier`]: what a detector found, classified by a second
//! model on CUDA pictures without them leaving the GPU.

use std::{path::Path, sync::Arc};

use ffmpeg_next::{self as ffmpeg, ffi};
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
/// an NV12 picture. It runs on CUDA alone, not TensorRT: a classifier's
/// batch is however many objects a picture has, which a TensorRT engine
/// built for one shape would be rebuilt for.
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
    driver: CudaDriver,
    device_memory: MemoryInfo<'static>,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *mut ffi::AVHWDeviceContext,
    _hw_device_ctx: Arc<AvBufferRef>,
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

        let linked = runtime::Linked::ask();
        if let CudaRuntime::Unavailable { cuda } = runtime::runtime(&linked) {
            return Err(OrtError::CudaRuntimeUnavailable(cuda).into());
        }
        let session = Session::builder()
            .map_err(OrtError::from)?
            .with_execution_providers([CUDA::default()
                .with_device_id(0)
                .build()
                .error_on_failure()])
            .map_err(|error| OrtError::Ort(error.into()))?
            .commit_from_file(model_path)
            .map_err(OrtError::from)?;
        let input = classifier_input(&session)?;
        let labels = labels(options.labels.as_deref(), &session);
        let capacity = input.batch.unwrap_or(MAX_BATCH).max(1);
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
            "model loaded: path={path}, input={}x{}, batch={:?}, {} labels, {:?}; {linked}",
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
            driver,
            device_memory,
            device_ctx,
            _hw_device_ctx: hw_device_ctx,
        })))
    }
}

impl Classifying {
    /// Cuts `crop` of `frame` into input `slot` of the tensor.
    fn cut(
        &self,
        frame: &ffmpeg::frame::Video,
        nv12: bool,
        (left, top, width, height): Crop,
        slot: usize,
    ) -> std::result::Result<(), OrtError> {
        let fit = Fit {
            model: self.input.size,
            offset: (0, 0),
            scaled: self.input.size,
        };
        if nv12 {
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
            let whole = BgraSurface::from_frame(frame).ok_or(OrtError::MissingSurface)?;
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
            CudaSurfaces::NV12_OR_BGRA,
        )?;
        let nv12 = surface.layout == ffmpeg::format::Pixel::NV12;
        let (width, height) = self.input.size;
        let (scale, bias) = self.options.input.affine();
        let mut answers = Vec::with_capacity(crops.len());
        for group in crops.chunks(self.capacity) {
            for (slot, crop) in group.iter().enumerate() {
                self.cut(frame, nv12, *crop, slot)?;
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
                .with_layouts(crate::contract::PixelLayoutSet::NV12_OR_BGRA),
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
        let plan: Plan =
            self.memory
                .plan(&detections, &self.options, (frame.width(), frame.height()));
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
    fn answers<T: RawSink + SrcPads>(classifier: &mut T, buf: MediaBuffer) -> Vec<Option<usize>> {
        let kept = capture(classifier);
        classifier.consume(buf).expect("classifies");
        let kept = kept.lock().unwrap();
        kept[0]
            .metadata()
            .and_then(|m| m.get::<Detections>())
            .expect("carries Detections")
            .items
            .iter()
            .map(|item| item.classes.first().map(|class| class.class_id))
            .collect()
    }

    /// On the GPU it says what the CPU says of the same objects, from NV12
    /// and from BGRA: the same model on the same boxes, cut by a kernel
    /// rather than on the CPU. The cuts differ — nearest samples against
    /// bilinear — so most, not all, must agree.
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
        let expected = answers(
            &mut cpu,
            found.clone().attach_to(MediaBuffer::video(nth_picture(
                &video,
                150,
                ffmpeg::format::Pixel::RGB24,
            ))),
        );
        assert!(expected.iter().all(Option::is_some), "{expected:?}");

        for (format, layout) in [
            (CudaFrameFormat::Nv12, ffmpeg::format::Pixel::NV12),
            (CudaFrameFormat::Bgra, ffmpeg::format::Pixel::BGRA),
        ] {
            let mut upload = CudaUpload::new("upload", &device, format);
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(nth_picture(&video, 150, layout)))
                .expect("uploads");
            let on_gpu = uploaded.lock().unwrap().remove(0);
            let actual = answers(&mut gpu, found.clone().attach_to(on_gpu));
            let agree = expected.iter().zip(&actual).filter(|(a, b)| a == b).count();
            eprintln!("{format:?}: cpu {expected:?}, gpu {actual:?}");
            assert!(agree >= 4, "{format:?}: {agree} of 6 agree");
        }
    }
}
