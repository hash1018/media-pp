//! What the classifiers share: their options, how a classification model's
//! input and output are read, which objects to classify on a picture, and
//! what is remembered of each followed one — DeepStream's secondary
//! inference, and its asynchronous mode, where an object is classified once
//! and the answer kept for as long as it is followed. The embedders choose
//! and remember their objects the same way, an answer being a vector.

use std::collections::HashMap;
use std::sync::Arc;

use ort::session::Session;

use super::OrtError;
use crate::buffer::MediaBuffer;
use crate::elements::vision::batch::StreamId;
use crate::elements::vision::batch::per_stream::{PerStream, stream_of};
use crate::elements::{Classification, Detection, Detections};

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
    /// The lowest detection score an object is classified at. A detector
    /// before a tracker is told to keep unsure boxes, for the tracker to
    /// match a partly hidden object with, and the tracker hands on those it
    /// numbered nothing for as they came; without this each of them was
    /// classified again on every picture, as an object with no number is.
    /// An answer already given a followed object is kept whatever its box
    /// scores now.
    pub min_detection_score: f32,
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
    /// The model's own labels, every class, boxes of 16 pixels or more
    /// detected at 0.25 or more — Ultralytics' own confidence threshold —
    /// answers at 0.5 or more, inputs from 0 to 1, and an answer kept for a
    /// second at 30 frames a second.
    fn default() -> Self {
        Self {
            labels: None,
            classes: None,
            min_size: 16,
            min_detection_score: 0.25,
            min_score: 0.5,
            input: InputScale::Unit,
            reclassify: 30,
        }
    }
}

/// Which of a detector's objects a second model looks at, and how often
/// again: what a classifier's and an embedder's options both say.
pub(crate) trait Selects {
    /// The detected classes looked at, or `None` for every one.
    fn classes(&self) -> Option<&[usize]>;
    /// The fewest pixels across or down a box may have.
    fn min_size(&self) -> u32;
    /// The lowest detection score looked at.
    fn min_detection_score(&self) -> f32;
    /// How many pictures a followed object's answer is kept; 0 for as long
    /// as it is followed.
    fn refresh(&self) -> u32;
}

impl Selects for OrtClassifierOptions {
    fn classes(&self) -> Option<&[usize]> {
        self.classes.as_deref()
    }

    fn min_size(&self) -> u32 {
        self.min_size
    }

    fn min_detection_score(&self) -> f32 {
        self.min_detection_score
    }

    fn refresh(&self) -> u32 {
        self.reclassify
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
    image_input(session, 224)
}

/// [`classifier_input`], `open` for a side the model leaves open.
pub(crate) fn image_input(session: &Session, open: u32) -> Result<Input, OrtError> {
    let input = session
        .inputs()
        .first()
        .ok_or_else(|| OrtError::UnsupportedModel("the model has no input".into()))?;
    let shape = input
        .dtype()
        .tensor_shape()
        .ok_or_else(|| OrtError::UnsupportedModel("its input is not a tensor".into()))?;
    let side = |value: i64| if value > 0 { value as u32 } else { open };
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

/// What a classifier — or an embedder, whose answer `T` is a vector — does
/// with one picture's detections: which to look at now, by index and box,
/// and which carry a remembered answer.
#[derive(Debug)]
pub(crate) struct Plan<T = Classification> {
    pub(crate) classify: Vec<(usize, Crop)>,
    pub(crate) remembered: Vec<(usize, T)>,
    /// The stream the picture is of, whose memory its answers go into.
    stream: Option<StreamId>,
}

impl<T> Default for Plan<T> {
    fn default() -> Self {
        Self {
            classify: Vec::new(),
            remembered: Vec::new(),
            stream: None,
        }
    }
}

/// What a classifier knows of the objects it has classified, kept for each
/// stream after a [`StreamMux`](crate::elements::StreamMux): what an answer
/// ages by is its own stream's pictures, not every stream's.
pub(crate) struct Memory<T = Classification> {
    streams: PerStream<Answers<T>>,
}

impl<T> Default for Memory<T> {
    fn default() -> Self {
        Self {
            streams: PerStream::default(),
        }
    }
}

/// What a classifier knows of one stream's objects, by a tracker's number.
#[derive(Debug)]
struct Answers<T> {
    /// The stream's pictures so far.
    picture: u64,
    /// Each followed object's answer, the picture it was given on, and the
    /// last picture the object was seen on.
    answers: HashMap<u64, (T, u64, u64)>,
}

impl<T> Default for Answers<T> {
    fn default() -> Self {
        Self {
            picture: 0,
            answers: HashMap::new(),
        }
    }
}

/// How many pictures an object may go unseen before its answer is
/// forgotten.
const FORGET_AFTER: u64 = 300;

impl<T: Clone> Memory<T> {
    /// What to do with `detections`, on `buf`, a picture `width` by
    /// `height`.
    pub(crate) fn plan(
        &mut self,
        buf: &MediaBuffer,
        detections: &Detections,
        options: &impl Selects,
        size: (u32, u32),
    ) -> Plan<T> {
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
    fn remember(&mut self, plan: &Plan<T>, id: u64, answer: T) {
        if let Some(answers) = self.streams.of(plan.stream) {
            answers.remember(id, answer);
        }
    }

    /// Forgets everything: after a seek, the numbers are a new tracker's.
    pub(crate) fn clear(&mut self) {
        self.streams.clear();
    }
}

impl<T: Clone> Answers<T> {
    fn plan(
        &mut self,
        detections: &Detections,
        options: &impl Selects,
        (width, height): (u32, u32),
    ) -> Plan<T> {
        self.picture += 1;
        let picture = self.picture;
        let mut plan = Plan::default();
        for (index, item) in detections.items.iter().enumerate() {
            if let Some(id) = item.track_id
                && let Some((answer, given, seen)) = self.answers.get_mut(&id)
            {
                *seen = picture;
                let fresh =
                    options.refresh() == 0 || picture - *given < u64::from(options.refresh());
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
            if item.score < options.min_detection_score()
                || options
                    .classes()
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
            if w < options.min_size() || h < options.min_size() {
                continue;
            }
            plan.classify.push((index, (left, top, w, h)));
        }
        self.answers
            .retain(|_, (_, _, seen)| picture - *seen <= FORGET_AFTER);
        plan
    }

    /// Remembers `answer` for object `id`, given on this picture.
    fn remember(&mut self, id: u64, answer: T) {
        self.answers
            .insert(id, (answer, self.picture, self.picture));
    }
}

/// Puts `answers` — by index, from the model or remembered — on
/// `detections`' items, remembering the new ones of followed objects.
pub(crate) fn apply(
    detections: &mut Detections,
    memory: &mut Memory,
    plan: Plan,
    answers: Vec<(usize, Option<Classification>)>,
) {
    apply_each(detections, memory, plan, answers, |item, answer| {
        item.classes.push(answer)
    });
}

/// [`apply`] for answers of any kind, each put on its item by `put`.
pub(crate) fn apply_each<T: Clone>(
    detections: &mut Detections,
    memory: &mut Memory<T>,
    mut plan: Plan<T>,
    answers: Vec<(usize, Option<T>)>,
    put: impl Fn(&mut Detection, T),
) {
    for (index, answer) in std::mem::take(&mut plan.remembered) {
        put(&mut detections.items[index], answer);
    }
    for (index, answer) in answers {
        let Some(answer) = answer else {
            continue;
        };
        if let Some(id) = detections.items[index].track_id {
            memory.remember(&plan, id, answer.clone());
        }
        put(&mut detections.items[index], answer);
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
        let mut memory: Memory = Memory::default();
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

    /// The unsure boxes a tracker is handed and hands on unnumbered are not
    /// classified; a followed object keeps its answer when its box is unsure
    /// on one picture.
    #[test]
    fn unsure_detections_are_not_classified_and_answers_are_kept() {
        let options = OrtClassifierOptions {
            reclassify: 0,
            ..OrtClassifierOptions::default()
        };
        let unsure = |id: Option<u64>| Detection {
            score: 0.15,
            ..tracked(id, 0)
        };
        let mut memory = Memory::default();
        let plan = memory.plan(
            &on(None),
            &found(vec![unsure(None), tracked(Some(1), 0)], false),
            &options,
            (640, 360),
        );
        let classified: Vec<usize> = plan.classify.iter().map(|(index, _)| *index).collect();
        assert_eq!(classified, [1], "the unsure, unnumbered box is left");
        apply(
            &mut found(vec![unsure(None), tracked(Some(1), 0)], false),
            &mut memory,
            plan,
            vec![(1, Some(red()))],
        );

        let plan = memory.plan(
            &on(None),
            &found(vec![unsure(Some(1))], false),
            &options,
            (640, 360),
        );
        assert!(plan.classify.is_empty());
        assert_eq!(plan.remembered.len(), 1, "its answer is kept");
    }
}
