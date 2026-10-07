//! [`MetalOrtClassifier`]: what a detector found, classified by a second
//! model on VideoToolbox pictures, each object cut from the picture on the
//! GPU.

use crate::orientation::Orientations;
use std::{path::Path, sync::Arc};

use ort::{inputs, session::Session, value::TensorRef};

use crate::ffmpeg;
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::{Classification, Detections},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::classify::{
    Crop, Input, Memory, OrtClassifierOptions, Plan, answer, apply, best,
};
use super::super::{OrtError, labels};
use super::fitting::{Cut, Fitting, Picture};
use super::{CORE_ML_BATCH, object_model_session};

/// Classifies the objects a detector found on each VideoToolbox picture
/// with a second model — what
/// [`SwOrtClassifier`](crate::elements::SwOrtClassifier) does on the CPU,
/// with each object's box cut from the picture and stretched into the
/// model's input by a Metal kernel where the picture is, several objects to
/// one run, and the model run by ONNX Runtime's Core ML provider on the GPU
/// or the Neural Engine.
///
/// It goes after a detector — a
/// [`MetalOrtDetector`](crate::elements::MetalOrtDetector), or any — and
/// best after an [`ObjectTracker`](crate::elements::ObjectTracker), which
/// lets a followed object be classified once and its answer kept, as
/// [`OrtClassifierOptions::reclassify`] says. It takes NV12 or BGRA
/// VideoToolbox pictures from any device, since a pixel buffer belongs to
/// none. The model is an image classifier with one input, `[batch, 3,
/// height, width]`, and one output of a score per class, probabilities or
/// raw.
///
/// As the detector does, it runs through Core ML or not at all:
/// [`Self::new`] refuses where the provider does not start. The cut inputs
/// are written to memory the CPU and GPU share, and handed to ONNX Runtime
/// from there. A model that takes any number of pictures at once is fixed
/// to take four, which Core ML compiles once and runs on the Neural Engine;
/// a picture's objects go in fours, the inputs past them unused.
///
/// # Requirements
///
/// The `ort-coreml` feature, on an Apple silicon Mac.
pub struct MetalOrtClassifier(FilterStage<Classifying>);

filter_stage!(MetalOrtClassifier);

/// What a [`MetalOrtClassifier`] does with each picture.
struct Classifying {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtClassifierOptions,
    labels: Arc<[Arc<str>]>,
    input: Input,
    memory: Memory,
    fitting: Fitting,
    /// How each picture is turned to be shown.
    orientations: Orientations,
}

impl MetalOrtClassifier {
    /// Loads the classification model at `model_path` to run through Core
    /// ML.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        options: OrtClassifierOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalOrtClassifier, &name, None);
        let path = model_path.as_ref().display().to_string();
        let (session, input) = object_model_session(model_path.as_ref(), 224, &pp_log)?;
        let labels = labels(options.labels.as_deref(), &session);
        let fitting = Fitting::new(input.size, input.batch.unwrap_or(CORE_ML_BATCH))?;
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batch={:?}, {} labels, {:?}, on Core ML",
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
            memory: Memory::default(),
            fitting,
            orientations: Orientations::default(),
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
        let picture = Picture::of(frame)?;
        let orientation = self.orientations.of(frame, &self.pp_log);
        let (width, height) = self.input.size;
        let affine = self.options.input.affine();
        let mut answers = Vec::with_capacity(crops.len());
        for group in crops.chunks(self.fitting.capacity()) {
            // Each box stretched to the whole input, as the CPU and CUDA
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
            self.fitting.fit(&picture, &cuts, affine)?;
            // A model of fixed batch is handed exactly that many; one of
            // open batch, as many as there are.
            let rows = self.input.batch.unwrap_or(group.len());
            let input = TensorRef::from_array_view((
                [rows, 3, height as usize, width as usize],
                self.fitting.input(rows),
            ))?;
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
        ElementType::MetalOrtClassifier
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Classifying {
    /// Decoded NV12 or BGRA video in VideoToolbox pixel buffers.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "MetalOrtClassifier",
                wanted: "VideoToolbox video frames",
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
    use crate::elements::{AppSink, Detection, InputScale, SwOrtClassifier, VideoToolboxUpload};
    use crate::test_support::{nth_picture, try_videotoolbox_device};

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

    /// Through Core ML it says what the CPU says of the same objects, from
    /// NV12 and from BGRA: the same model on the same boxes, cut by a Metal
    /// kernel rather than on the CPU. The cuts differ — nearest samples
    /// against bilinear — so most, not all, must agree. Needs a model and a
    /// video; skipped, saying so, without them.
    #[test]
    fn it_says_through_core_ml_what_the_cpu_says() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_CLASSIFIER"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_CLASSIFIER and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let options = OrtClassifierOptions {
            min_score: 0.0,
            input: InputScale::ImageNet,
            ..OrtClassifierOptions::default()
        };
        let mut gpu =
            MetalOrtClassifier::new("gpu", &model, options.clone()).expect("loads on Core ML");
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

        for layout in [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA] {
            let mut upload = VideoToolboxUpload::new("upload", &device);
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(nth_picture(&video, 150, layout)))
                .expect("uploads");
            let on_gpu = uploaded.lock().unwrap().remove(0);
            let actual = answers(&mut gpu, found.clone().attach_to(on_gpu));
            let agree = expected.iter().zip(&actual).filter(|(a, b)| a == b).count();
            eprintln!("{layout:?}: cpu {expected:?}, core ml {actual:?}");
            assert!(agree >= 4, "{layout:?}: {agree} of 6 agree");
        }
    }
}
