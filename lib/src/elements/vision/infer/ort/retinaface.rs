//! Reading a RetinaFace model's output — see [`DetectorModel::RetinaFace`].
//!
//! [`DetectorModel::RetinaFace`]: super::DetectorModel::RetinaFace

use super::model::{
    ChannelOrder, DetectorDecoder, ModelBox, ModelInput, ModelOutput, non_max_suppression,
};
use super::{OrtDetectorOptions, OrtError};

/// The values RetinaFace was trained on: OpenCV's BGR from 0 to 255, less
/// the mean of each channel.
pub(crate) const INPUT: ModelInput = ModelInput {
    order: ChannelOrder::Bgr,
    scale: [255.0; 3],
    bias: [-104.0, -117.0, -123.0],
};

/// The anchors' sizes at each of the three levels the model looks at, in
/// pixels of its input, and each level's step — biubug6's `cfg_mnet` and
/// `cfg_re50` alike, and every export of them.
const LEVELS: [([f32; 2], u32); 3] = [([16.0, 32.0], 8), ([64.0, 128.0], 16), ([256.0, 512.0], 32)];

/// How far a box's centre and the log of its size are scaled.
const VARIANCE: [f32; 2] = [0.1, 0.2];

/// What reads a RetinaFace model's output: three outputs of one row per
/// anchor — the box's offsets from its anchor (`[anchors, 4]`), the scores
/// of background and face (`[anchors, 2]`), and five points' offsets
/// (`[anchors, 10]`) — told apart by their last dimension, as exports name
/// them differently.
#[derive(Debug)]
pub(crate) struct RetinaFace {
    /// The model's input size, which the anchors are laid over.
    input: (u32, u32),
    /// Each anchor's centre and size, as fractions of the input, in the
    /// model's order: level by level, row by row, then each size.
    anchors: Vec<[f32; 4]>,
}

impl RetinaFace {
    /// The decoder for a model of `input` size.
    pub(crate) fn new(input: (u32, u32)) -> Self {
        let (width, height) = (input.0 as f32, input.1 as f32);
        let mut anchors = Vec::new();
        for (sizes, step) in LEVELS {
            let (columns, rows) = (input.0.div_ceil(step), input.1.div_ceil(step));
            let step = step as f32;
            for row in 0..rows {
                for column in 0..columns {
                    for size in sizes {
                        anchors.push([
                            (column as f32 + 0.5) * step / width,
                            (row as f32 + 0.5) * step / height,
                            size / width,
                            size / height,
                        ]);
                    }
                }
            }
        }
        Self { input, anchors }
    }
}

impl DetectorDecoder for RetinaFace {
    fn decode(
        &self,
        outputs: &[ModelOutput<'_>],
        _input: (u32, u32),
        options: &OrtDetectorOptions,
    ) -> Result<Vec<ModelBox>, OrtError> {
        let anchors = self.anchors.len();
        let of_width = |columns: usize| {
            outputs
                .iter()
                .find(|output| *output.shape == [anchors, columns])
                .map(|output| output.data)
                .ok_or_else(|| {
                    let shapes: Vec<_> = outputs.iter().map(|output| output.shape).collect();
                    OrtError::UnsupportedModel(format!(
                        "RetinaFace's outputs are {shapes:?}, not [{anchors}, 4], [{anchors}, 2] \
                         and [{anchors}, 10] for its {}x{} input",
                        self.input.0, self.input.1
                    ))
                })
        };
        let (boxes, scores, points) = (of_width(4)?, of_width(2)?, of_width(10)?);
        let (width, height) = (self.input.0 as f32, self.input.1 as f32);
        let [centre, size] = VARIANCE;
        let mut found = Vec::new();
        for (index, anchor) in self.anchors.iter().enumerate() {
            let score = face(&scores[index * 2..index * 2 + 2]);
            if score < options.conf_threshold {
                continue;
            }
            let &[ax, ay, aw, ah] = anchor;
            let offsets = &boxes[index * 4..index * 4 + 4];
            let cx = ax + offsets[0] * centre * aw;
            let cy = ay + offsets[1] * centre * ah;
            let w = aw * (offsets[2] * size).exp();
            let h = ah * (offsets[3] * size).exp();
            let landmarks = points[index * 10..index * 10 + 10]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[dx, dy]| {
                    (
                        (ax + dx * centre * aw) * width,
                        (ay + dy * centre * ah) * height,
                    )
                })
                .collect();
            found.push(
                ModelBox::new(
                    0,
                    score,
                    [
                        (cx - w / 2.0) * width,
                        (cy - h / 2.0) * height,
                        (cx + w / 2.0) * width,
                        (cy + h / 2.0) * height,
                    ],
                )
                .with_landmarks(landmarks),
            );
        }
        Ok(non_max_suppression(found, options.iou_threshold))
    }
}

/// How sure an anchor's two scores — background, face — say it is a face:
/// the second as it is where they are already probabilities, as every
/// export of biubug6's test-phase model makes them, and through a softmax
/// where they are the raw scores.
fn face(scores: &[f32]) -> f32 {
    let (background, face) = (scores[0], scores[1]);
    let probabilities = (0.0..=1.0).contains(&background)
        && (0.0..=1.0).contains(&face)
        && (background + face - 1.0).abs() < 1e-3;
    if probabilities {
        face
    } else {
        1.0 / (1.0 + (background - face).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{Letterbox, decode_batch};
    use super::*;

    #[test]
    fn a_640_input_has_its_16800_anchors_in_the_models_order() {
        let decoder = RetinaFace::new((640, 640));
        assert_eq!(decoder.anchors.len(), (80 * 80 + 40 * 40 + 20 * 20) * 2);
        // The first cell's two sizes, then the next cell along the row.
        let near = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(near(
            decoder.anchors[0],
            [4.0 / 640.0, 4.0 / 640.0, 0.025, 0.025]
        ));
        assert!(near(
            decoder.anchors[1],
            [4.0 / 640.0, 4.0 / 640.0, 0.05, 0.05]
        ));
        assert!(near(
            decoder.anchors[2],
            [12.0 / 640.0, 4.0 / 640.0, 0.025, 0.025]
        ));
        // The last level's last cell, its larger size.
        assert!(near(
            *decoder.anchors.last().unwrap(),
            [624.0 / 640.0, 624.0 / 640.0, 0.8, 0.8]
        ));
        // A side that steps do not divide rounds up, as the model does.
        assert_eq!(
            RetinaFace::new((650, 480)).anchors.len(),
            (82 * 60 + 41 * 30 + 21 * 15) * 2
        );
    }

    /// The outputs of a 64 by 32 model with every anchor background but
    /// one, whose box and points are given as offsets, its scores as `face`.
    fn outputs(face: [f32; 2], at: usize, offsets: [f32; 4], points: [f32; 10]) -> Vec<Vec<f32>> {
        let anchors = RetinaFace::new((64, 32)).anchors.len();
        let mut boxes = vec![0.0; anchors * 4];
        let mut scores = [1.0, 0.0].repeat(anchors);
        let mut landmarks = vec![0.0; anchors * 10];
        boxes[at * 4..at * 4 + 4].copy_from_slice(&offsets);
        scores[at * 2..at * 2 + 2].copy_from_slice(&face);
        landmarks[at * 10..at * 10 + 10].copy_from_slice(&points);
        vec![boxes, scores, landmarks]
    }

    fn found(data: &[Vec<f32>], order: [usize; 3]) -> Result<Vec<ModelBox>, OrtError> {
        let decoder = RetinaFace::new((64, 32));
        let anchors = decoder.anchors.len();
        let shapes = [[anchors, 4], [anchors, 2], [anchors, 10]];
        let outputs: Vec<ModelOutput<'_>> = order
            .iter()
            .map(|&which| ModelOutput {
                name: "output",
                shape: &shapes[which],
                data: &data[which],
            })
            .collect();
        decoder.decode(&outputs, (64, 32), &OrtDetectorOptions::default())
    }

    /// An anchor of no offset is its own box; its points, of no offset, its
    /// centre — whatever order the outputs come in.
    #[test]
    fn an_anchor_unmoved_is_its_own_box() {
        // Anchor 3: the second cell of the first row of the first level,
        // its larger size — centred at (12, 4), 32 across.
        let data = outputs([0.1, 0.9], 3, [0.0; 4], [0.0; 10]);
        for order in [[0, 1, 2], [2, 0, 1]] {
            let found = found(&data, order).expect("reads");
            assert_eq!(found.len(), 1, "{found:?}");
            assert_eq!(found[0].class_id, 0);
            assert_eq!(found[0].score, 0.9);
            assert_eq!(found[0].corners, [-4.0, -12.0, 28.0, 20.0]);
            assert_eq!(found[0].landmarks, vec![(12.0, 4.0); 5]);
        }
    }

    /// Offsets move the centre by a tenth of the anchor's size each and
    /// scale the size by e to a fifth of theirs.
    #[test]
    fn offsets_move_and_scale_the_anchor() {
        let data = outputs(
            [0.1, 0.9],
            3,
            [1.0, -1.0, 5.0 * 2f32.ln(), 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
        );
        let found = found(&data, [0, 1, 2]).expect("reads").remove(0);
        // Centre (12 + 3.2, 4 - 3.2), 64 across and 32 down.
        let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
        let [left, top, right, bottom] = found.corners;
        assert!(
            near(left, 15.2 - 32.0)
                && near(top, 0.8 - 16.0)
                && near(right, 15.2 + 32.0)
                && near(bottom, 0.8 + 16.0),
            "{found:?}"
        );
        let (x, y) = found.landmarks[0];
        assert!(near(x, 15.2) && near(y, 4.0), "{found:?}");
        let (x, y) = found.landmarks[1];
        assert!(near(x, 12.0) && near(y, 7.2), "{found:?}");
        let (x, y) = found.landmarks[4];
        assert!(near(x, 8.8) && near(y, 0.8), "{found:?}");
    }

    /// Raw scores, not yet probabilities, are put through a softmax.
    #[test]
    fn raw_scores_are_made_probabilities() {
        assert_eq!(face(&[0.2, 0.8]), 0.8);
        let raw = face(&[-2.0, 3.0]);
        assert!((raw - 1.0 / (1.0 + (-5.0f32).exp())).abs() < 1e-6, "{raw}");
        let data = outputs([-2.0, -4.0], 3, [0.0; 4], [0.0; 10]);
        assert!(
            found(&data, [0, 1, 2]).expect("reads").is_empty(),
            "below 0.25"
        );
    }

    #[test]
    fn outputs_of_another_number_of_anchors_are_refused() {
        let decoder = RetinaFace::new((640, 640));
        let data = vec![0.0; 100 * 4];
        let error = decoder
            .decode(
                &[ModelOutput {
                    name: "loc",
                    shape: &[100, 4],
                    data: &data,
                }],
                (640, 640),
                &OrtDetectorOptions::default(),
            )
            .expect_err("100 anchors");
        assert!(error.to_string().contains("[16800, 4]"), "{error}");
    }

    /// Through a detector's letterbox, a face's points land on the picture
    /// with its box.
    #[test]
    fn a_face_and_its_points_land_on_the_picture() {
        let decoder = RetinaFace::new((64, 32));
        let anchors = decoder.anchors.len();
        let data = outputs([0.0, 1.0], 3, [0.0; 4], [0.0; 10]);
        let shapes = [[1, anchors, 4], [1, anchors, 2], [1, anchors, 10]];
        let outputs: Vec<ModelOutput<'_>> = (0..3)
            .map(|which| ModelOutput {
                name: "output",
                shape: &shapes[which],
                data: &data[which],
            })
            .collect();
        // A 128 by 64 picture, fitted at half its size.
        let letterbox = Letterbox::new((128, 64), (64, 32));
        let found = decode_batch(
            &decoder,
            &outputs,
            &[letterbox],
            (64, 32),
            &OrtDetectorOptions::default(),
        )
        .expect("reads")
        .remove(0);
        assert_eq!(found[0].landmarks, vec![(12.0 / 64.0, 4.0 / 32.0); 5]);
        assert_eq!(
            (found[0].x, found[0].y),
            (0.0, 0.0),
            "clamped to the picture"
        );
    }
}
