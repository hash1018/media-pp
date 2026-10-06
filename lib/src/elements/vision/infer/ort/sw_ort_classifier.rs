//! [`SwOrtClassifier`]: what a detector found, classified by a second model
//! on the CPU.

use std::{path::Path, sync::Arc};

use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

use crate::ffmpeg;
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::{Classification, Detections},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::classify::{
    Crop, Input, Memory, OrtClassifierOptions, Plan, answer, apply, best, classifier_input,
};
use super::sw_ort_detector::{is_hardware, to_rgb24};
use super::{OrtError, labels};

/// The most objects classified in one run of a model that takes any
/// number at once.
const MAX_BATCH: usize = 32;

/// Classifies the objects a detector found on each picture with a second
/// model, on the CPU, and hands the picture on with each answer in its
/// object's [`Detection::classes`](crate::elements::Detection::classes) —
/// DeepStream's secondary inference: a car's make, a person's clothes.
///
/// It goes after a detector, and best after an
/// [`ObjectTracker`](crate::elements::ObjectTracker): an object the tracker
/// numbered is classified once and its answer kept, on every picture it is
/// followed through — those a detector let by too — until
/// [`OrtClassifierOptions::reclassify`] says to ask again. Each object's
/// box is cut from the picture and stretched to the model's input.
///
/// It takes decoded pictures in system memory, any format. The model is an
/// image classifier with one input, `[batch, 3, height, width]`, and one
/// output of a score per class, probabilities or raw; objects are
/// classified together where its batch is open, one at a time where it is
/// fixed.
pub struct SwOrtClassifier(FilterStage<Classifying>);

filter_stage!(SwOrtClassifier);

/// The conversion of one picture shape to RGB24 at its own size, and the
/// picture it makes.
struct Converting {
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    context: ffmpeg::software::scaling::Context,
    rgb: ffmpeg::frame::Video,
}

// SAFETY: the scaling context and frame are heap allocations owned solely by
// this conversion, used only by the one thread transforming at a time, as the
// detector's fitting holds its own.
unsafe impl Send for Converting {}

/// What an [`SwOrtClassifier`] does with each picture.
struct Classifying {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtClassifierOptions,
    labels: Arc<[Arc<str>]>,
    input: Input,
    memory: Memory,
    converting: Option<Converting>,
}

impl SwOrtClassifier {
    /// Loads the classification model at `model_path` to run on the CPU.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        options: OrtClassifierOptions,
    ) -> Result<Self> {
        let path = model_path.as_ref().display().to_string();
        let session = Session::builder()
            .map_err(OrtError::from)?
            .commit_from_file(model_path)
            .map_err(OrtError::from)?;
        let input = classifier_input(&session)?;
        let labels = labels(options.labels.as_deref(), &session);
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwOrtClassifier, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batch={:?}, {} labels, {:?}",
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
            converting: None,
        })))
    }
}

impl Classifying {
    /// `frame` as RGB24 at its own size, through a conversion kept while
    /// pictures keep their shape.
    fn rgb(&mut self, frame: &ffmpeg::frame::Video) -> Result<&ffmpeg::frame::Video> {
        let from = (
            frame.format(),
            frame.width(),
            frame.height(),
            frame.color_space(),
            frame.color_range(),
        );
        if self
            .converting
            .as_ref()
            .is_none_or(|converting| converting.from != from)
        {
            self.converting = Some(Converting {
                from,
                context: to_rgb24(from, (from.1, from.2))?,
                rgb: ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, from.1, from.2),
            });
        }
        let converting = self.converting.as_mut().expect("made above");
        converting
            .context
            .run(frame, &mut converting.rgb)
            .map_err(OrtError::Convert)?;
        Ok(&converting.rgb)
    }

    /// The model's answer for each of `crops` of `frame`, in order.
    fn classify(
        &mut self,
        frame: &ffmpeg::frame::Video,
        crops: &[Crop],
    ) -> Result<Vec<Option<(usize, f32)>>> {
        let (width, height) = self.input.size;
        let (scale, bias) = self.options.input.affine();
        let batch = self.input.batch.unwrap_or(MAX_BATCH).max(1);
        let rgb = self.rgb(frame)?.clone();
        let mut answers = Vec::with_capacity(crops.len());
        for group in crops.chunks(batch) {
            // A model of fixed batch is handed exactly that many, the rest
            // left blank; one of open batch, as many as there are.
            let rows = self.input.batch.unwrap_or(group.len());
            let mut tensor = Array4::<f32>::zeros((rows, 3, height as usize, width as usize));
            for (slot, crop) in group.iter().enumerate() {
                cut(&rgb, *crop, (width, height), scale, bias, &mut tensor, slot);
            }
            let outputs = self
                .session
                .run(inputs![
                    TensorRef::from_array_view(&tensor).map_err(OrtError::from)?
                ])
                .map_err(OrtError::from)?;
            let output = outputs[0]
                .try_extract_array::<f32>()
                .map_err(OrtError::from)?;
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

/// `crop` of the RGB24 picture `rgb`, stretched to `size` and scaled per
/// channel, into input `slot` of `tensor`.
///
/// Each input pixel is the mean of the source pixels it covers — from
/// `d * source / size` to `(d + 1) * source / size` rounded up, at least
/// one — as the CUDA and Metal kernels fit a box: a bilinear sample at its
/// centre read four pixels of the dozen a person's box gives each input
/// pixel, so the same box reached the model as another picture on the CPU
/// than on the GPU, and an unsure model named it otherwise.
fn cut(
    rgb: &ffmpeg::frame::Video,
    (left, top, width, height): Crop,
    size: (u32, u32),
    scale: [f32; 3],
    bias: [f32; 3],
    tensor: &mut Array4<f32>,
    slot: usize,
) {
    let stride = rgb.stride(0);
    let data = rgb.data(0);
    let covered = |d: u32, source: u32, scaled: u32| {
        let first = u64::from(d) * u64::from(source) / u64::from(scaled);
        let last = (u64::from(d + 1) * u64::from(source)).div_ceil(u64::from(scaled));
        let last = last.min(u64::from(source)).max(first + 1);
        (first as usize, last as usize)
    };
    for y in 0..size.1 {
        let (y0, y1) = covered(y, height, size.1);
        for x in 0..size.0 {
            let (x0, x1) = covered(x, width, size.0);
            let mut sum = [0u32; 3];
            for row in y0..y1 {
                let at = (top as usize + row) * stride + (left as usize + x0) * 3;
                for pixel in data[at..at + (x1 - x0) * 3].as_chunks::<3>().0 {
                    for channel in 0..3 {
                        sum[channel] += u32::from(pixel[channel]);
                    }
                }
            }
            let count = ((x1 - x0) * (y1 - y0)) as f32;
            for channel in 0..3 {
                tensor[[slot, channel, y as usize, x as usize]] =
                    sum[channel] as f32 / count / 255.0 * scale[channel] + bias[channel];
            }
        }
    }
}

impl Element for Classifying {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwOrtClassifier
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Classifying {
    /// Decoded video in system memory, any pixel layout.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ))
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let refused = |got| OrtError::UnsupportedBuffer {
            detector: "SwOrtClassifier",
            wanted: "video frames in system memory",
            got,
        };
        let MediaBuffer::Video(frame) = &buf else {
            return Err(refused(buf.kind()).into());
        };
        if is_hardware(frame.format()) {
            return Err(refused("a hardware video frame").into());
        }
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, Detection, InputScale};

    /// The classifier this machine's tests can run, where it has one: an
    /// image classifier named by `MEDIA_PP_TEST_CLASSIFIER`, ImageNet's
    /// normalisation, and its labels one a line beside it as
    /// `MEDIA_PP_TEST_CLASSIFIER_LABELS`.
    pub(crate) fn classifier() -> Option<(String, Option<Vec<String>>)> {
        let path = std::env::var("MEDIA_PP_TEST_CLASSIFIER").ok()?;
        let labels = std::env::var("MEDIA_PP_TEST_CLASSIFIER_LABELS")
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|text| text.lines().map(str::to_owned).collect());
        Path::new(&path).is_file().then_some((path, labels))
    }

    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// Each detected object comes out with the classifier's answer; a
    /// followed one keeps it on the next picture without being asked again,
    /// and a picture with no detections goes through as it came.
    #[test]
    fn objects_come_out_classified_and_followed_ones_keep_it() {
        let Some((model, labels)) = classifier() else {
            eprintln!("skipping: set MEDIA_PP_TEST_CLASSIFIER to an image classifier to run this");
            return;
        };
        let options = OrtClassifierOptions {
            labels,
            min_score: 0.0,
            input: InputScale::ImageNet,
            ..OrtClassifierOptions::default()
        };
        let mut classifier = SwOrtClassifier::new("classifier", &model, options).expect("loads");
        let kept = capture(&mut classifier);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 320, 240);
        for (i, byte) in frame.data_mut(0).iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        let items = vec![
            Detection {
                track_id: Some(4),
                ..Detection::new(0, 0.9, 0.1, 0.1, 0.4, 0.5)
            },
            Detection::new(0, 0.9, 0.5, 0.4, 0.3, 0.4),
        ];
        let found = Detections::new("detector", Arc::from([]), items);
        let picture = || found.clone().attach_to(MediaBuffer::video(frame.clone()));
        classifier.consume(picture()).expect("classified");
        classifier.consume(picture()).expect("remembered");
        classifier
            .consume(MediaBuffer::video(frame.clone()))
            .expect("nothing to classify");

        let kept = kept.lock().unwrap();
        let classes = |buf: &MediaBuffer| -> Vec<Vec<Classification>> {
            buf.metadata()
                .and_then(|m| m.get::<Detections>())
                .unwrap()
                .items
                .iter()
                .map(|item| item.classes.clone())
                .collect()
        };
        let first = classes(&kept[0]);
        assert!(first.iter().all(|c| c.len() == 1), "{first:?}");
        assert_eq!(first[0][0].classifier.as_ref(), "classifier");
        eprintln!(
            "answered: {:?}",
            first
                .iter()
                .map(|c| (c[0].label(), c[0].score))
                .collect::<Vec<_>>()
        );
        let second = classes(&kept[1]);
        assert_eq!(
            second[0], first[0],
            "the followed object's answer, remembered"
        );
        assert!(kept[2].metadata().is_none());
    }
}
