//! What a detector knows of the kind of model it runs: how a picture's
//! values are put into the model's input, and how what the model outputs is
//! read into boxes — [`DetectorModel`], one of the kinds this crate reads or
//! an application's own, read by its own [`DetectorDecoder`]: DeepStream's
//! custom bounding-box parser.
//!
//! Every kind is fitted alike — turned the way it is shown, scaled to fit
//! inside the model's input, proportions kept, centred, the rest grey — and
//! every box is mapped back alike, through the [`Letterbox`] it was fitted
//! by; what differs is the values the model wants, and its output.

use std::fmt;
use std::sync::Arc;

use super::{OrtDetectorOptions, OrtError, yolo::Yolo};
use crate::elements::Detection;
use crate::orientation::Orientation;

/// The kind of detection model a detector runs: how its input is filled
/// and its output read.
#[derive(Clone, Default)]
#[non_exhaustive]
pub enum DetectorModel {
    /// An Ultralytics YOLO detector, in either layout it exports — see
    /// [`OrtDetectorOptions`].
    #[default]
    Yolo,
    /// A model this crate does not read: its input filled as `input` says,
    /// and its output read by `decoder`.
    Custom {
        /// The values the model wants.
        input: ModelInput,
        /// What reads its output.
        decoder: Arc<dyn DetectorDecoder>,
    },
}

impl fmt::Debug for DetectorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Yolo => f.write_str("Yolo"),
            Self::Custom { input, decoder } => f
                .debug_struct("Custom")
                .field("input", input)
                .field("decoder", decoder)
                .finish(),
        }
    }
}

/// Two custom models are the same where they fill the same input and share
/// one decoder.
impl PartialEq for DetectorModel {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Yolo, Self::Yolo) => true,
            (
                Self::Custom { input, decoder },
                Self::Custom {
                    input: other_input,
                    decoder: other_decoder,
                },
            ) => input == other_input && Arc::ptr_eq(decoder, other_decoder),
            _ => false,
        }
    }
}

impl DetectorModel {
    /// The values the model wants.
    pub(crate) fn input(&self) -> ModelInput {
        match self {
            Self::Yolo => ModelInput::default(),
            Self::Custom { input, .. } => *input,
        }
    }

    /// What reads the output of a model of this kind.
    pub(crate) fn decoder(&self) -> Arc<dyn DetectorDecoder> {
        match self {
            Self::Yolo => Arc::new(Yolo),
            Self::Custom { decoder, .. } => Arc::clone(decoder),
        }
    }

    /// Whether this is a YOLO model — whose boxes' best classes the GPU
    /// detectors find where the output is, rather than copying it down.
    #[cfg(any(feature = "ort-cuda", all(target_os = "macos", feature = "ort-coreml")))]
    pub(crate) fn is_yolo(&self) -> bool {
        matches!(self, Self::Yolo)
    }
}

/// The values a model wants in its input, made from each pixel's red, green
/// and blue as 0 to 1: put in `order`, then each times `scale` plus `bias`,
/// both given in that order. The margin around a picture fitted inside the
/// input — Ultralytics' grey, 114 of 255 — is made the same way.
///
/// The default is RGB from 0 to 1, as Ultralytics exports a model. A model
/// trained in OpenCV's BGR from 0 to 255 less a mean per channel is
/// `ModelInput { order: ChannelOrder::Bgr, scale: [255.0; 3], bias: [-104.0,
/// -117.0, -123.0] }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelInput {
    /// The order of the input's three planes.
    pub order: ChannelOrder,
    /// What each plane's value, 0 to 1, is multiplied by.
    pub scale: [f32; 3],
    /// What is then added to it.
    pub bias: [f32; 3],
}

impl Default for ModelInput {
    /// RGB from 0 to 1.
    fn default() -> Self {
        Self {
            order: ChannelOrder::Rgb,
            scale: [1.0; 3],
            bias: [0.0; 3],
        }
    }
}

impl ModelInput {
    /// Whether the values are put in as they are, scaled by nothing.
    #[cfg(feature = "ort-cuda")]
    pub(crate) fn is_unscaled(&self) -> bool {
        (self.scale, self.bias) == ([1.0; 3], [0.0; 3])
    }

    /// Which plane each of red, green and blue goes in.
    pub(crate) fn planes(&self) -> [usize; 3] {
        match self.order {
            ChannelOrder::Rgb => [0, 1, 2],
            ChannelOrder::Bgr => [2, 1, 0],
        }
    }
}

/// The order of a model's three colour planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelOrder {
    /// Red, green, blue — PyTorch's and Ultralytics'.
    #[default]
    Rgb,
    /// Blue, green, red — OpenCV's, and so a model trained on what it reads.
    Bgr,
}

/// What reads a detection model's output into boxes, for a model this
/// crate does not read itself: a [`DetectorModel::Custom`]'s. One decoder is
/// shared by every detector it is given to, and called from each one's own
/// thread.
///
/// It is handed one picture's outputs at a time and says what the model
/// found in that picture, in pixels of the model's input; the detector maps
/// each box back onto the picture through how it fitted the picture, turned
/// and scaled, so a decoder never sees the picture itself.
pub trait DetectorDecoder: fmt::Debug + Send + Sync {
    /// What the model found in one picture, from every one of its
    /// `outputs`, in the model's order: those boxes confident enough by
    /// `options.conf_threshold`, less the duplicates — see
    /// [`non_max_suppression`] — in any order. `input` is the model's input
    /// size, width by height, which the boxes are in.
    ///
    /// An output that is not what the decoder reads is
    /// [`OrtError::UnsupportedModel`], which fails the pictures run with it.
    fn decode(
        &self,
        outputs: &[ModelOutput<'_>],
        input: (u32, u32),
        options: &OrtDetectorOptions,
    ) -> Result<Vec<ModelBox>, OrtError>;
}

/// One of a model's outputs for one picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelOutput<'a> {
    /// Its name in the model.
    pub name: &'a str,
    /// Its shape without the batch: the model's output less its first
    /// dimension, which is the picture's.
    pub shape: &'a [usize],
    /// Its values, the last dimension's adjacent.
    pub data: &'a [f32],
}

/// One object a model found, in pixels of its input.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ModelBox {
    /// Its class, by the model's numbering.
    pub class_id: usize,
    /// How sure the model is.
    pub score: f32,
    /// Its left, top, right and bottom edges.
    pub corners: [f32; 4],
    /// Points the model found on it — a face's eyes, say — each `(x, y)`,
    /// in the model's own order; none for a model that finds none.
    pub landmarks: Vec<(f32, f32)>,
}

impl ModelBox {
    /// An object of `class_id` found at `score`, between `corners`: left,
    /// top, right and bottom.
    pub fn new(class_id: usize, score: f32, corners: [f32; 4]) -> Self {
        Self {
            class_id,
            score,
            corners,
            landmarks: Vec::new(),
        }
    }

    /// The same, with `landmarks` found on it.
    pub fn with_landmarks(self, landmarks: Vec<(f32, f32)>) -> Self {
        Self { landmarks, ..self }
    }
}

/// `boxes` less the duplicates: highest score first, each box kept unless
/// it overlaps one kept already of its own class by more than
/// `iou_threshold`, as intersection over union — Ultralytics' default,
/// per-class suppression. What is kept is most confident first.
pub fn non_max_suppression(mut boxes: Vec<ModelBox>, iou_threshold: f32) -> Vec<ModelBox> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<ModelBox> = Vec::with_capacity(boxes.len());
    for candidate in boxes {
        let duplicate = kept.iter().any(|kept| {
            kept.class_id == candidate.class_id
                && iou(&kept.corners, &candidate.corners) > iou_threshold
        });
        if !duplicate {
            kept.push(candidate);
        }
    }
    kept
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let overlap_w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let overlap_h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = overlap_w * overlap_h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// How a picture is fitted inside the model's input: turned the way it is
/// shown, scaled to fit, proportions kept, centred, the rest grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Letterbox {
    /// The picture's size, as stored.
    pub(crate) frame: (u32, u32),
    /// The model's input size.
    pub(crate) model: (u32, u32),
    /// The size the picture is scaled to inside the input, turned.
    pub(crate) scaled: (u32, u32),
    /// Where the scaled picture's top-left corner sits in the input.
    pub(crate) offset: (u32, u32),
    /// How the picture is turned to be shown, and so into the input.
    pub(crate) orientation: Orientation,
}

impl Letterbox {
    /// A picture shown as stored.
    #[cfg(test)]
    pub(crate) fn new(frame: (u32, u32), model: (u32, u32)) -> Self {
        Self::shown(frame, model, Orientation::UPRIGHT)
    }

    /// A picture stored `frame` in size and shown turned as `orientation`
    /// says: the model is handed it the right way up, as it was trained on
    /// pictures, and what it finds is read back onto the stored picture.
    pub(crate) fn shown(frame: (u32, u32), model: (u32, u32), orientation: Orientation) -> Self {
        let shown = orientation.display_size(frame.0, frame.1);
        let scale =
            (model.0 as f32 / shown.0.max(1) as f32).min(model.1 as f32 / shown.1.max(1) as f32);
        let scaled = (
            ((shown.0 as f32 * scale).round() as u32).clamp(1, model.0),
            ((shown.1 as f32 * scale).round() as u32).clamp(1, model.1),
        );
        Self {
            frame,
            model,
            scaled,
            offset: ((model.0 - scaled.0) / 2, (model.1 - scaled.1) / 2),
            orientation,
        }
    }

    /// A point of the model's input, as fractions of the picture as shown,
    /// clamped to it.
    fn onto_frame(&self, x: f32, y: f32) -> (f32, f32) {
        (
            ((x - self.offset.0 as f32) / self.scaled.0 as f32).clamp(0.0, 1.0),
            ((y - self.offset.1 as f32) / self.scaled.1 as f32).clamp(0.0, 1.0),
        )
    }

    /// A point of the model's input, as fractions of the stored picture.
    fn point(&self, x: f32, y: f32) -> (f32, f32) {
        let (x, y) = self.onto_frame(x, y);
        let [x, y, _, _] = self.orientation.from_display([x, y, 0.0, 0.0]);
        (x, y)
    }

    /// A box the model found, as a [`Detection`] on the stored picture.
    pub(crate) fn detection(&self, found: ModelBox) -> Detection {
        let [left, top, right, bottom] = found.corners;
        let (x1, y1) = self.onto_frame(left, top);
        let (x2, y2) = self.onto_frame(right, bottom);
        let [x, y, width, height] = self.orientation.from_display([x1, y1, x2 - x1, y2 - y1]);
        Detection {
            landmarks: found
                .landmarks
                .iter()
                .map(|&(x, y)| self.point(x, y))
                .collect(),
            ..Detection::new(found.class_id, found.score, x, y, width, height)
        }
    }

    /// What was found in the picture, `found` mapped onto it, most
    /// confident first.
    pub(crate) fn detections(&self, found: Vec<ModelBox>) -> Vec<Detection> {
        let mut found: Vec<Detection> = found
            .into_iter()
            .map(|found| self.detection(found))
            .collect();
        found.sort_by(|a, b| b.score.total_cmp(&a.score));
        found
    }
}

/// What `decoder` reads `outputs` — each a whole run's, its first dimension
/// the batch — to hold for each of the pictures `letterboxes` fitted, one
/// after another from the first of the batch, mapped onto each picture.
///
/// A batch may hold more than the pictures — a model of fixed batch handed
/// fewer — and what it holds past them is no picture's.
pub(crate) fn decode_batch(
    decoder: &dyn DetectorDecoder,
    outputs: &[ModelOutput<'_>],
    letterboxes: &[Letterbox],
    input: (u32, u32),
    options: &OrtDetectorOptions,
) -> Result<Vec<Vec<Detection>>, OrtError> {
    let pictures = letterboxes.len();
    // Each output's own shape and how many values a picture's takes.
    let per_picture = outputs
        .iter()
        .map(|output| match output.shape.split_first() {
            Some((&batch, shape)) if batch >= pictures => {
                let floats: usize = shape.iter().product();
                if output.data.len() < pictures * floats {
                    return Err(OrtError::UnsupportedModel(format!(
                        "its output {} holds {} values, not the {:?} it says",
                        output.name,
                        output.data.len(),
                        output.shape
                    )));
                }
                Ok((shape, floats))
            }
            _ => Err(OrtError::UnsupportedModel(format!(
                "its output {} is {:?}, not a batch of {pictures}",
                output.name, output.shape
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    letterboxes
        .iter()
        .enumerate()
        .map(|(picture, letterbox)| {
            let mine: Vec<ModelOutput<'_>> = outputs
                .iter()
                .zip(&per_picture)
                .map(|(output, &(shape, floats))| ModelOutput {
                    name: output.name,
                    shape,
                    data: &output.data[picture * floats..(picture + 1) * floats],
                })
                .collect();
            Ok(letterbox.detections(decoder.decode(&mine, input, options)?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_picture_is_fitted_with_grey_above_and_below() {
        let letterbox = Letterbox::new((1920, 1080), (640, 640));
        assert_eq!(letterbox.scaled, (640, 360));
        assert_eq!(letterbox.offset, (0, 140));
    }

    /// A portrait recording stored on its side is fitted the right way up —
    /// tall in the input, as it is shown — and what is found in it is read
    /// back onto the stored picture: a box at the top of what is shown is at
    /// the stored picture's left, for a quarter turn clockwise, and so is a
    /// point on it.
    #[test]
    fn a_turned_picture_is_fitted_upright_and_read_back_as_stored() {
        use crate::orientation::Rotation;
        let turned = Letterbox::shown(
            (1920, 1080),
            (640, 640),
            Orientation::rotated(Rotation::Clockwise90),
        );
        assert_eq!(turned.scaled, (360, 640));
        assert_eq!(turned.offset, (140, 0));
        // The top tenth of what is shown, across its whole width, and a
        // point at its top-left corner.
        let found = turned.detection(
            ModelBox::new(0, 0.9, [140.0, 0.0, 500.0, 64.0]).with_landmarks(vec![(140.0, 0.0)]),
        );
        let near = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(
            near(found.x, 0.0)
                && near(found.y, 0.0)
                && near(found.width, 0.1)
                && near(found.height, 1.0),
            "{found:?}"
        );
        // Shown top-left is stored top-right, a quarter turn clockwise
        // having taken the stored top edge to the shown right.
        let (x, y) = found.landmarks[0];
        assert!(near(x, 0.0) && near(y, 1.0), "{found:?}");
    }

    /// Landmarks are mapped through the letterbox as the box is.
    #[test]
    fn landmarks_land_where_the_box_does() {
        let letterbox = Letterbox::new((1920, 1080), (640, 640));
        let found = letterbox.detection(
            ModelBox::new(0, 0.9, [320.0, 140.0, 640.0, 500.0])
                .with_landmarks(vec![(320.0, 140.0), (640.0, 500.0)]),
        );
        assert_eq!(found.landmarks, vec![(0.5, 0.0), (1.0, 1.0)]);
        assert_eq!((found.x, found.y), (0.5, 0.0));
    }

    #[test]
    fn overlapping_boxes_of_one_class_are_one_object() {
        let boxes = vec![
            ModelBox::new(0, 0.8, [77.0, 76.0, 127.0, 126.0]),
            ModelBox::new(0, 0.9, [75.0, 75.0, 125.0, 125.0]),
            ModelBox::new(1, 0.7, [76.0, 75.0, 126.0, 125.0]),
        ];
        let kept: Vec<_> = non_max_suppression(boxes, 0.45)
            .iter()
            .map(|found| (found.class_id, found.score))
            .collect();
        assert_eq!(kept, vec![(0, 0.9), (1, 0.7)]);
    }

    /// A decoder of the application's own, run over a batch: each picture
    /// is handed its own part of every output, and what it finds is mapped
    /// through its own letterbox.
    #[test]
    fn a_custom_decoder_reads_each_picture_of_a_batch() {
        /// Reads `[1, 5]`: corners and a score, and a `[2]` of a point.
        #[derive(Debug)]
        struct OneBox;
        impl DetectorDecoder for OneBox {
            fn decode(
                &self,
                outputs: &[ModelOutput<'_>],
                _input: (u32, u32),
                options: &OrtDetectorOptions,
            ) -> Result<Vec<ModelBox>, OrtError> {
                let [boxes, points] = outputs else {
                    return Err(OrtError::UnsupportedModel("two outputs".into()));
                };
                assert_eq!((boxes.shape, points.shape), (&[1, 5][..], &[2][..]));
                let row = boxes.data;
                Ok((row[4] >= options.conf_threshold)
                    .then(|| {
                        ModelBox::new(7, row[4], [row[0], row[1], row[2], row[3]])
                            .with_landmarks(vec![(points.data[0], points.data[1])])
                    })
                    .into_iter()
                    .collect())
            }
        }
        let wide = Letterbox::new((1920, 1080), (640, 640));
        let tall = Letterbox::new((1080, 1920), (640, 640));
        let boxes = [
            320.0, 320.0, 480.0, 480.0, 0.9, //
            0.0, 0.0, 10.0, 10.0, 0.1,
        ];
        let points = [400.0, 400.0, 0.0, 0.0];
        let outputs = [
            ModelOutput {
                name: "boxes",
                shape: &[2, 1, 5],
                data: &boxes,
            },
            ModelOutput {
                name: "points",
                shape: &[2, 2],
                data: &points,
            },
        ];
        let options = OrtDetectorOptions::default();
        let found =
            decode_batch(&OneBox, &outputs, &[wide, tall], (640, 640), &options).expect("reads");
        assert_eq!(found[1], Vec::new(), "the second's box is below threshold");
        assert_eq!(found[0].len(), 1);
        assert_eq!(found[0][0].class_id, 7);
        assert_eq!(found[0][0].landmarks.len(), 1);
        // One output short of the batch is the model's fault, said so.
        let short = [ModelOutput {
            name: "boxes",
            shape: &[1, 1, 5],
            data: &boxes[..5],
        }];
        let error = decode_batch(&OneBox, &short, &[wide, tall], (640, 640), &options)
            .expect_err("a batch of one for two pictures");
        assert!(error.to_string().contains("boxes"), "{error}");
    }

    #[test]
    fn custom_models_are_equal_by_their_decoder() {
        #[derive(Debug)]
        struct Nothing;
        impl DetectorDecoder for Nothing {
            fn decode(
                &self,
                _: &[ModelOutput<'_>],
                _: (u32, u32),
                _: &OrtDetectorOptions,
            ) -> Result<Vec<ModelBox>, OrtError> {
                Ok(Vec::new())
            }
        }
        let decoder: Arc<dyn DetectorDecoder> = Arc::new(Nothing);
        let custom = |decoder: &Arc<dyn DetectorDecoder>| DetectorModel::Custom {
            input: ModelInput::default(),
            decoder: Arc::clone(decoder),
        };
        assert_eq!(custom(&decoder), custom(&decoder));
        assert_ne!(custom(&decoder), custom(&(Arc::new(Nothing) as _)));
        assert_ne!(custom(&decoder), DetectorModel::Yolo);
        assert_eq!(DetectorModel::default(), DetectorModel::Yolo);
    }
}
