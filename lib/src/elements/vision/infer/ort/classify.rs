//! What the classifiers share: their options, how a classification model's
//! input and output are read, which objects to classify on a picture, and
//! what is remembered of each followed one — DeepStream's secondary
//! inference, and its asynchronous mode, where an object is classified once
//! and the answer kept for as long as it is followed.

use std::collections::HashMap;
use std::sync::Arc;

use ort::session::Session;

use super::OrtError;
use crate::buffer::MediaBuffer;
use crate::elements::vision::batch::StreamId;
use crate::elements::vision::batch::per_stream::{PerStream, stream_of};
use crate::elements::{Classification, Detections};

/// How a classifier decides what to classify, and what its answers are.
#[derive(Debug, Clone, PartialEq)]
pub struct OrtClassifierOptions {
    /// The model's class names, in its class order. `None` reads them from
    /// the model's `names` metadata, as an Ultralytics export carries.
    pub labels: Option<Vec<String>>,
    /// Which detected classes it classifies — a car model's classifier, the
    /// detector's cars — or `None` for every one.
    pub classes: Option<Vec<usize>>,
    /// The fewest pixels an object's box may have across or down to be
    /// classified: a smaller one is too little picture to tell by.
    pub min_size: u32,
    /// The lowest score an answer is kept at; below it, an object is left
    /// unclassified, and tried again on the next picture it is detected on.
    pub min_score: f32,
    /// How the model wants its input's values.
    pub input: InputScale,
    /// How many pictures a followed object's answer is kept before it is
    /// classified again; 0 keeps the first answer for as long as it is
    /// followed. An object with no tracker's number is classified on every
    /// picture it is detected on.
    pub reclassify: u32,
}

impl Default for OrtClassifierOptions {
    /// The model's own labels, every class, boxes of 16 pixels or more,
    /// answers at 0.5 or more, inputs from 0 to 1, and an answer kept for a
    /// second at 30 frames a second.
    fn default() -> Self {
        Self {
            labels: None,
            classes: None,
            min_size: 16,
            min_score: 0.5,
            input: InputScale::Unit,
            reclassify: 30,
        }
    }
}

/// How a model wants its input's values.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InputScale {
    /// RGB from 0 to 1 — an Ultralytics export's.
    #[default]
    Unit,
    /// RGB from 0 to 1, less ImageNet's mean and over its deviation per
    /// channel — torchvision's and most ImageNet models'.
    ImageNet,
}

impl InputScale {
    /// The `scale` and `bias` that make a value from 0 to 1 the model's,
    /// per channel: `x · scale + bias`.
    pub(crate) fn affine(self) -> ([f32; 3], [f32; 3]) {
        match self {
            Self::Unit => ([1.0; 3], [0.0; 3]),
            Self::ImageNet => {
                const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
                const DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];
                (
                    std::array::from_fn(|c| 1.0 / DEVIATION[c]),
                    std::array::from_fn(|c| -MEAN[c] / DEVIATION[c]),
                )
            }
        }
    }
}

/// A classification model's input: its size, and how many pictures it takes
/// at once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Input {
    pub(crate) size: (u32, u32),
    /// `None` where the model takes any number — its first dimension left
    /// open — and the batch is what is classified at once.
    pub(crate) batch: Option<usize>,
}

/// The model's input, from its first input's `[batch, 3, height, width]`;
/// 224 for a side the model leaves open.
pub(crate) fn classifier_input(session: &Session) -> Result<Input, OrtError> {
    let input = session
        .inputs()
        .first()
        .ok_or_else(|| OrtError::UnsupportedModel("the model has no input".into()))?;
    let shape = input
        .dtype()
        .tensor_shape()
        .ok_or_else(|| OrtError::UnsupportedModel("its input is not a tensor".into()))?;
    let side = |value: i64| if value > 0 { value as u32 } else { 224 };
    match shape.iter().as_slice() {
        [batch, 3, height, width] => Ok(Input {
            size: (side(*width), side(*height)),
            batch: (*batch > 0).then_some(*batch as usize),
        }),
        other => Err(OrtError::UnsupportedModel(format!(
            "its input is {other:?}, not [batch, 3, height, width]"
        ))),
    }
}

/// The class a row of a model's output chose, and how sure it is: the row as
/// it is where it is already probabilities — none negative, summing to 1 —
/// and through a softmax where it is a model's raw scores.
pub(crate) fn best(row: &[f32]) -> Option<(usize, f32)> {
    let (class, &top) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    let sum: f32 = row.iter().sum();
    if row.iter().all(|&value| value >= 0.0) && (sum - 1.0).abs() < 0.01 {
        return Some((class, top));
    }
    let total: f32 = row.iter().map(|&value| (value - top).exp()).sum();
    Some((class, 1.0 / total))
}

/// An object's box in pixels, cut to the picture: left, top, width, height.
pub(crate) type Crop = (u32, u32, u32, u32);

/// What a classifier does with one picture's detections: which to classify
/// now, by index and box, and which carry a remembered answer.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    pub(crate) classify: Vec<(usize, Crop)>,
    pub(crate) remembered: Vec<(usize, Classification)>,
    /// The stream the picture is of, whose memory its answers go into.
    stream: Option<StreamId>,
}

/// What a classifier knows of the objects it has classified, kept for each
/// stream after a [`StreamMux`](crate::elements::StreamMux): what an answer
/// ages by is its own stream's pictures, not every stream's.
#[derive(Default)]
pub(crate) struct Memory {
    streams: PerStream<Answers>,
}

/// What a classifier knows of one stream's objects, by a tracker's number.
#[derive(Debug, Default)]
struct Answers {
    /// The stream's pictures so far.
    picture: u64,
    /// Each followed object's answer, the picture it was given on, and the
    /// last picture the object was seen on.
    answers: HashMap<u64, (Classification, u64, u64)>,
}

/// How many pictures an object may go unseen before its answer is
/// forgotten.
const FORGET_AFTER: u64 = 300;

impl Memory {
    /// What to do with `detections`, on `buf`, a picture `width` by
    /// `height`.
    pub(crate) fn plan(
        &mut self,
        buf: &MediaBuffer,
        detections: &Detections,
        options: &OrtClassifierOptions,
        size: (u32, u32),
    ) -> Plan {
        // A stream sought keeps what it knew: its objects after the seek
        // are numbered anew, and the answers of those before age out.
        let (answers, _) = self.streams.get(buf, |_| Answers::default());
        Plan {
            stream: stream_of(buf).0,
            ..answers.plan(detections, options, size)
        }
    }

    /// Remembers `answer` for object `id`, given on the picture `plan` was
    /// made for.
    fn remember(&mut self, plan: &Plan, id: u64, answer: Classification) {
        if let Some(answers) = self.streams.of(plan.stream) {
            answers.remember(id, answer);
        }
    }

    /// Forgets everything: after a seek, the numbers are a new tracker's.
    pub(crate) fn clear(&mut self) {
        self.streams.clear();
    }
}

impl Answers {
    fn plan(
        &mut self,
        detections: &Detections,
        options: &OrtClassifierOptions,
        (width, height): (u32, u32),
    ) -> Plan {
        self.picture += 1;
        let picture = self.picture;
        let mut plan = Plan::default();
        for (index, item) in detections.items.iter().enumerate() {
            if let Some(id) = item.track_id
                && let Some((answer, given, seen)) = self.answers.get_mut(&id)
            {
                *seen = picture;
                let fresh =
                    options.reclassify == 0 || picture - *given < u64::from(options.reclassify);
                if fresh || detections.predicted {
                    plan.remembered.push((index, answer.clone()));
                    continue;
                }
            }
            // Only a detector's own boxes are classified: a tracker's
            // expectation is near the object, not on it.
            if detections.predicted {
                continue;
            }
            if options
                .classes
                .as_ref()
                .is_some_and(|classes| !classes.contains(&item.class_id))
            {
                continue;
            }
            // A thousandth of a pixel either way is arithmetic, not the box,
            // as the overlays place theirs.
            let at = |fraction: f32, size: u32, round: fn(f64) -> f64, nudge: f64| {
                (round(f64::from(fraction) * f64::from(size) + nudge).max(0.0) as u32).min(size)
            };
            let left = at(item.x, width, f64::floor, 1e-3);
            let top = at(item.y, height, f64::floor, 1e-3);
            let right = at(item.x + item.width, width, f64::ceil, -1e-3);
            let bottom = at(item.y + item.height, height, f64::ceil, -1e-3);
            let (w, h) = (right.saturating_sub(left), bottom.saturating_sub(top));
            if w < options.min_size || h < options.min_size {
                continue;
            }
            plan.classify.push((index, (left, top, w, h)));
        }
        self.answers
            .retain(|_, (_, _, seen)| picture - *seen <= FORGET_AFTER);
        plan
    }

    /// Remembers `answer` for object `id`, given on this picture.
    fn remember(&mut self, id: u64, answer: Classification) {
        self.answers
            .insert(id, (answer, self.picture, self.picture));
    }
}

/// Puts `answers` — by index, from the model or remembered — on
/// `detections`' items, remembering the new ones of followed objects.
pub(crate) fn apply(
    detections: &mut Detections,
    memory: &mut Memory,
    mut plan: Plan,
    answers: Vec<(usize, Option<Classification>)>,
) {
    for (index, answer) in std::mem::take(&mut plan.remembered) {
        detections.items[index].classes.push(answer);
    }
    for (index, answer) in answers {
        let Some(answer) = answer else {
            continue;
        };
        if let Some(id) = detections.items[index].track_id {
            memory.remember(&plan, id, answer.clone());
        }
        detections.items[index].classes.push(answer);
    }
}

/// `class_id` at `score` as `classifier`'s answer, where it is sure
/// enough.
pub(crate) fn answer(
    classifier: &Arc<str>,
    labels: &Arc<[Arc<str>]>,
    options: &OrtClassifierOptions,
    (class_id, score): (usize, f32),
) -> Option<Classification> {
    (score >= options.min_score)
        .then(|| Classification::new(Arc::clone(classifier), Arc::clone(labels), class_id, score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::Detection;

    fn found(items: Vec<Detection>, predicted: bool) -> Detections {
        Detections {
            predicted,
            ..Detections::new("detector", Arc::from([]), items)
        }
    }

    fn tracked(id: Option<u64>, class_id: usize) -> Detection {
        Detection {
            track_id: id,
            ..Detection::new(class_id, 0.9, 0.1, 0.1, 0.2, 0.4)
        }
    }

    fn red() -> Classification {
        Classification::new("classifier", Arc::from([Arc::from("red")]), 0, 0.8)
    }

    /// A picture of `stream`, as a mux hands it on, or of no mux's.
    fn on(stream: Option<u64>) -> MediaBuffer {
        use crate::buffer::Metadata;
        use crate::elements::vision::batch::StreamOrigin;
        let buf = MediaBuffer::video(crate::ffmpeg::frame::Video::empty());
        match stream {
            Some(stream) => buf.with_metadata(Metadata::default().with(StreamOrigin {
                id: StreamId(stream),
                name: Arc::from(format!("camera {stream}")),
                generation: 0,
            })),
            None => buf,
        }
    }

    /// An answer ages by its own stream's pictures: another stream's coming
    /// between do not make it due again.
    #[test]
    fn an_answer_ages_by_its_own_streams_pictures() {
        let options = OrtClassifierOptions {
            reclassify: 3,
            ..OrtClassifierOptions::default()
        };
        let mut memory = Memory::default();
        let size = (640, 360);
        let mut first = found(vec![tracked(Some(7), 0)], false);
        let plan = memory.plan(&on(Some(1)), &first, &options, size);
        apply(&mut first, &mut memory, plan, vec![(0, Some(red()))]);
        for _ in 0..5 {
            let other = found(vec![tracked(Some(8), 0)], false);
            memory.plan(&on(Some(2)), &other, &options, size);
        }
        let again = found(vec![tracked(Some(7), 0)], false);
        let plan = memory.plan(&on(Some(1)), &again, &options, size);
        assert!(plan.classify.is_empty(), "one picture of its own on");
        assert_eq!(plan.remembered.len(), 1);
        memory.plan(&on(Some(1)), &again, &options, size);
        let plan = memory.plan(&on(Some(1)), &again, &options, size);
        assert_eq!(plan.classify.len(), 1, "three of its own on, asked again");
    }

    #[test]
    fn scores_are_read_as_probabilities_or_through_a_softmax() {
        assert_eq!(best(&[0.1, 0.7, 0.2]), Some((1, 0.7)));
        let (class, score) = best(&[1.0, 3.0, 0.0]).unwrap();
        assert_eq!(class, 1);
        let expected = 3f32.exp() / (1f32.exp() + 3f32.exp() + 1.0);
        assert!((score - expected).abs() < 1e-5, "{score}");
        assert_eq!(best(&[]), None);
    }

    #[test]
    fn imagenet_input_is_less_the_mean_over_the_deviation() {
        let (scale, bias) = InputScale::ImageNet.affine();
        let red = 0.485 * scale[0] + bias[0];
        assert!(red.abs() < 1e-6, "the mean is 0");
        assert!(((1.0 * scale[0] + bias[0]) - (1.0 - 0.485) / 0.229).abs() < 1e-5);
        assert_eq!(InputScale::Unit.affine(), ([1.0; 3], [0.0; 3]));
    }

    /// A followed object is classified once, its answer carried on later
    /// pictures — those a detector let by too — and asked again after
    /// `reclassify`; one with no number every time.
    #[test]
    fn a_followed_object_is_classified_once_and_remembered() {
        let options = OrtClassifierOptions {
            reclassify: 3,
            ..OrtClassifierOptions::default()
        };
        let mut memory = Memory::default();
        let size = (640, 360);
        let mut picture = found(vec![tracked(Some(7), 0), tracked(None, 0)], false);
        let plan = memory.plan(&on(None), &picture, &options, size);
        assert_eq!(plan.classify.len(), 2);
        let classify: Vec<usize> = plan.classify.iter().map(|(i, _)| *i).collect();
        apply(
            &mut picture,
            &mut memory,
            plan,
            classify.into_iter().map(|i| (i, Some(red()))).collect(),
        );
        assert_eq!(picture.items[0].classes, vec![red()]);

        let expected = found(vec![tracked(Some(7), 0)], true);
        let plan = memory.plan(&on(None), &expected, &options, size);
        assert!(plan.classify.is_empty(), "not on a tracker's expectation");
        assert_eq!(plan.remembered.len(), 1, "but remembered there");

        let again = found(vec![tracked(Some(7), 0), tracked(None, 0)], false);
        let plan = memory.plan(&on(None), &again, &options, size);
        assert_eq!(plan.remembered.len(), 1);
        assert_eq!(
            plan.classify.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [1]
        );

        let plan = memory.plan(&on(None), &again, &options, size);
        assert_eq!(
            plan.classify.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 1],
            "three pictures on, asked again"
        );
    }

    #[test]
    fn only_the_classes_asked_for_and_boxes_big_enough_are_classified() {
        let options = OrtClassifierOptions {
            classes: Some(vec![2]),
            min_size: 50,
            ..OrtClassifierOptions::default()
        };
        let mut memory = Memory::default();
        let picture = found(
            vec![
                tracked(None, 0),
                tracked(None, 2),
                Detection::new(2, 0.9, 0.0, 0.0, 0.05, 0.05),
            ],
            false,
        );
        let plan = memory.plan(&on(None), &picture, &options, (640, 360));
        assert_eq!(plan.classify.len(), 1);
        let (index, (left, top, width, height)) = plan.classify[0];
        assert_eq!(index, 1);
        assert_eq!((left, top, width, height), (64, 36, 128, 144));
    }
}
