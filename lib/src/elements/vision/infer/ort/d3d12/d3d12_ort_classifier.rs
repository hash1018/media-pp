//! [`D3d12OrtClassifier`]: what a detector found, classified by a second
//! model on D3D12 pictures, each object cut from the picture on the GPU and
//! the model run by DirectML.

use std::{path::Path, sync::Arc};

use ort::{
    ep::directml::{DMLResource, resource_from_d3d},
    memory::MemoryInfo,
    session::Session,
};
use windows::core::Interface;

use crate::ffmpeg;
use crate::orientation::Orientations;
use crate::platform::windows::d3d12_gpu::D3d12Gpu;
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::{ChannelOrder, Classification, Detections, ModelInput},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::classify::{
    Crop, Input, Memory, OrtClassifierOptions, Plan, answer, apply, best,
};
use super::super::device_tensor;
use super::super::{OrtError, labels};
use super::fitting::{Cut, Fitting};
use super::{DIRECTML_BATCH, directml_memory, host_memory, object_model_session};

/// Classifies the objects a detector found on each D3D12 picture with a
/// second model — what [`SwOrtClassifier`](crate::elements::SwOrtClassifier)
/// does on the CPU, with each object's box cut from the picture and
/// stretched into the model's input by a compute shader where the picture
/// is, several objects to one run, and the model run by ONNX Runtime's
/// DirectML provider on the same GPU.
///
/// It goes after a detector — a
/// [`D3d12OrtDetector`](crate::elements::D3d12OrtDetector), or any — and
/// best after an [`ObjectTracker`](crate::elements::ObjectTracker), which
/// lets a followed object be classified once and its answer kept, as
/// [`OrtClassifierOptions::reclassify`] says. It takes NV12 D3D12 pictures
/// made on the [`D3d12Gpu`] it is given. The model is an image classifier
/// with one input, `[batch, 3, height, width]` — 224 for a side it leaves
/// open — and one output of a score per class, probabilities or raw. A
/// model that takes any number of objects at once is fixed to take eight at
/// a time, as DirectML is fastest with every dimension fixed; a picture's
/// objects go in eights, the inputs past them unused.
///
/// # Requirements
///
/// The `ort-directml` feature, on Windows.
pub struct D3d12OrtClassifier(FilterStage<Classifying>);

filter_stage!(D3d12OrtClassifier);

/// What a [`D3d12OrtClassifier`] does with each picture. The session goes
/// before the fitting, whose buffer it reads, and the input's allocation
/// before both.
struct Classifying {
    name: Arc<str>,
    pp_log: PpLog,
    /// The fitting's buffer as DirectML reads it.
    allocation: DMLResource,
    session: Session,
    options: OrtClassifierOptions,
    labels: Arc<[Arc<str>]>,
    input: Input,
    memory: Memory,
    fitting: Fitting,
    /// How each picture is turned to be shown.
    orientations: Orientations,
    /// Where the input is, and where the output is copied.
    on_gpu: MemoryInfo<'static>,
    on_host: MemoryInfo<'static>,
}

// SAFETY: the session, the buffer and its allocation are used by the one
// thread transforming at a time, through `&mut self`; the D3D12 and
// DirectML objects are free-threaded.
unsafe impl Send for Classifying {}

impl D3d12OrtClassifier {
    /// Loads the classification model at `model_path` to run through
    /// DirectML on `gpu` — the same [`D3d12Gpu`] every other D3D12 element
    /// of the pipeline was made with, whose pictures it reads.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        model_path: impl AsRef<Path>,
        options: OrtClassifierOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12OrtClassifier, &name, None);
        let path = model_path.as_ref().display().to_string();
        let (session, input) = object_model_session(gpu, model_path.as_ref(), 224, &pp_log)?;
        let labels = labels(options.labels.as_deref(), &session);
        let fitting = Fitting::new(gpu, input.size, input.batch.unwrap_or(DIRECTML_BATCH))?;
        // SAFETY: the fitting's live buffer, on the device DirectML was made
        // on; the allocation is dropped before the fitting.
        let allocation = unsafe { resource_from_d3d(fitting.tensor().as_raw().cast()) }
            .map_err(OrtError::from)?;
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batch={:?}, {} labels, {:?}, on DirectML",
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
            allocation,
            session,
            options,
            labels,
            input,
            memory: Memory::default(),
            fitting,
            orientations: Orientations::default(),
            on_gpu: directml_memory()?,
            on_host: host_memory()?,
        })))
    }
}

impl Classifying {
    /// The model's answer for each of `crops` of `frame`, in order.
    fn classify(
        &mut self,
        frame: &ffmpeg::frame::Video,
        crops: &[Crop],
    ) -> std::result::Result<Vec<Option<(usize, f32)>>, OrtError> {
        let picture = self.fitting.picture(frame)?;
        let orientation = self.orientations.of(frame, &self.pp_log);
        let (width, height) = self.input.size;
        let (scale, bias) = self.options.input.affine();
        let values = ModelInput {
            order: ChannelOrder::Rgb,
            scale,
            bias,
        };
        let mut answers = Vec::with_capacity(crops.len());
        for group in crops.chunks(self.fitting.capacity()) {
            // Each box stretched to the whole input, as the other
            // classifiers cut theirs.
            let cuts: Vec<Cut> = group
                .iter()
                .enumerate()
                .map(|(slot, &crop)| Cut {
                    crop,
                    orientation,
                    offset: (0, 0),
                    scaled: self.input.size,
                    slot,
                })
                .collect();
            self.fitting.fit_each(&[(&picture, &cuts)], values)?;
            // A model of fixed batch is handed exactly that many; one of
            // open batch, as many as there are.
            let rows = self.input.batch.unwrap_or(group.len());
            // SAFETY: the allocation is the fitting's buffer of `capacity`
            // inputs, of which the shape takes the first `rows`; it outlives
            // the run, the tensor being dropped before this returns.
            let tensor = unsafe {
                device_tensor(
                    &self.on_gpu,
                    *self.allocation,
                    &[rows as i64, 3, i64::from(height), i64::from(width)],
                )?
            };
            let mut binding = self.session.create_binding()?;
            binding.bind_input(self.session.inputs()[0].name().to_owned(), &tensor)?;
            let output = self.session.outputs()[0].name().to_owned();
            binding.bind_output_to_device(&output, &self.on_host)?;
            let outputs = self.session.run_binding(&binding)?;
            let (shape, data) = outputs[output.as_str()].try_extract_tensor::<f32>()?;
            let classes = shape.last().map_or(0, |&classes| classes.max(0) as usize);
            for slot in 0..group.len() {
                let row = data
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
        ElementType::D3d12OrtClassifier
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Classifying {
    /// Decoded NV12 video in D3D12 textures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "D3d12OrtClassifier",
                wanted: "D3D12 video frames",
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
        let mut plan: Plan = self.memory.plan(
            &buf,
            &detections,
            &self.options,
            (frame.width(), frame.height()),
        );
        // A box of no width or height, which a `min_size` of 0 lets by, has
        // nothing to cut.
        plan.classify
            .retain(|(_, (_, _, width, height))| *width > 0 && *height > 0);
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
    use crate::elements::{AppSink, D3d12Upload, Detection, InputScale, SwOrtClassifier};
    use crate::test_support::{nth_picture, try_d3d12_gpu};

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

    /// Through DirectML it says what the CPU says of the same objects: the
    /// same model on the same boxes of the same NV12 picture. Where the CPU
    /// is sure of a box, the GPU names the same class at much the same
    /// score; where it is not, a thousand classes share the score and the
    /// answer turns on the last digit, so those are not compared. Ten
    /// boxes, more than one run of eight takes: on a model of open batch
    /// the last two go in a second. Needs a model, a video and a D3D12
    /// device; skipped, saying so, without them.
    #[test]
    fn it_says_through_directml_what_the_cpu_says() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_CLASSIFIER"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_CLASSIFIER and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let options = OrtClassifierOptions {
            min_score: 0.0,
            input: InputScale::ImageNet,
            ..OrtClassifierOptions::default()
        };
        let mut on_gpu =
            D3d12OrtClassifier::new("gpu", &gpu, &model, options.clone()).expect("loads");
        let mut cpu = SwOrtClassifier::new("cpu", &model, options).expect("loads");
        // A grid of boxes across the picture, each a sizeable piece of it.
        let items: Vec<Detection> = (0..10)
            .map(|i| {
                let (column, row) = ((i % 5) as f32, (i / 5) as f32);
                Detection::new(0, 0.9, 0.02 + column * 0.19, 0.1 + row * 0.45, 0.18, 0.4)
            })
            .collect();
        let found = Detections::new("detector", Arc::from([]), items);
        // An answer this sure is the model's, not the last digit's.
        const SURE: f32 = 0.3;

        // The CPU is handed the very picture the GPU is, so both read its
        // colours alike.
        let picture = nth_picture(&video, 150, ffmpeg::format::Pixel::NV12);
        let expected = answers(
            &mut cpu,
            found.clone().attach_to(MediaBuffer::video(picture.clone())),
        );
        assert!(expected.iter().all(Option::is_some), "{expected:?}");
        let Ok(mut upload) = D3d12Upload::new("upload", &gpu) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return;
        };
        let uploaded = capture(&mut upload);
        upload
            .consume(MediaBuffer::video(picture))
            .expect("uploads");
        let buf = uploaded.lock().unwrap().remove(0);
        let actual = answers(&mut on_gpu, found.clone().attach_to(buf));
        eprintln!("cpu {expected:?}, directml {actual:?}");
        let sure: Vec<_> = expected
            .iter()
            .zip(&actual)
            .filter_map(|(cpu, gpu)| Some(((*cpu).filter(|(_, score)| *score >= SURE)?, *gpu)))
            .collect();
        assert!(!sure.is_empty(), "the CPU is sure of no box");
        for ((class, score), gpu) in sure {
            let (gpu_class, gpu_score) = gpu.expect("the GPU answers too");
            assert_eq!(gpu_class, class, "the CPU said {class} at {score}");
            assert!(
                (gpu_score - score).abs() <= 0.15,
                "class {class} at {gpu_score} through DirectML, {score} on the CPU"
            );
        }
    }
}
