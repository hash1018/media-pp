//! Reading a DETR-family detector's output — see [`DetectorModel::Detr`].
//!
//! [`DetectorModel::Detr`]: super::DetectorModel::Detr

use super::model::{DetectorDecoder, ModelBox, ModelOutput, non_max_suppression};
use super::{OrtDetectorOptions, OrtError};

/// What reads a DETR-family model's output: two outputs of one row per
/// query — each class's logit (`[queries, classes]`) and the box's centre
/// and size as fractions of the input (`[queries, 4]`) — found by their
/// names, `logits` and `pred_boxes`, as transformers exports them, or
/// failing that by their shapes.
#[derive(Debug)]
pub(crate) struct Detr;

impl DetectorDecoder for Detr {
    fn decode(
        &self,
        outputs: &[ModelOutput<'_>],
        input: (u32, u32),
        options: &OrtDetectorOptions,
    ) -> Result<Vec<ModelBox>, OrtError> {
        let (logits, boxes, classes) = read(outputs)?;
        let (width, height) = (input.0 as f32, input.1 as f32);
        let mut found = Vec::new();
        for (query, row) in logits.chunks_exact(classes).enumerate() {
            // The best class by its logit, then that logit through the
            // sigmoid each class is scored by on its own.
            let (class_id, logit) = row.iter().copied().enumerate().fold(
                (0, f32::NEG_INFINITY),
                |best, (class_id, logit)| {
                    if logit > best.1 {
                        (class_id, logit)
                    } else {
                        best
                    }
                },
            );
            let score = 1.0 / (1.0 + (-logit).exp());
            if score < options.conf_threshold {
                continue;
            }
            let [cx, cy, w, h] = [0, 1, 2, 3].map(|i| boxes[query * 4 + i]);
            found.push(ModelBox::new(
                class_id,
                score,
                [
                    (cx - w / 2.0) * width,
                    (cy - h / 2.0) * height,
                    (cx + w / 2.0) * width,
                    (cy + h / 2.0) * height,
                ],
            ));
        }
        Ok(non_max_suppression(found, options.iou_threshold))
    }
}

/// The logits, the boxes and how many classes there are: the outputs named
/// `logits` and `pred_boxes`, or where they are not so named, the one
/// `[queries, 4]` and the one `[queries, classes]` of as many queries.
fn read<'a>(outputs: &[ModelOutput<'a>]) -> Result<(&'a [f32], &'a [f32], usize), OrtError> {
    let named = |name: &str| outputs.iter().find(|output| output.name == name);
    let (logits, boxes) = match (named("logits"), named("pred_boxes")) {
        (Some(logits), Some(boxes)) => (logits, boxes),
        _ => {
            let boxes: Vec<_> = outputs
                .iter()
                .filter(|o| o.shape.last() == Some(&4))
                .collect();
            let logits: Vec<_> = outputs
                .iter()
                .filter(|o| o.shape.last() != Some(&4))
                .collect();
            match (&logits[..], &boxes[..]) {
                ([logits], [boxes]) => (*logits, *boxes),
                _ => return Err(unsupported(outputs)),
            }
        }
    };
    match (logits.shape, boxes.shape) {
        (&[queries, classes], &[box_queries, 4])
            if queries == box_queries
                && classes > 0
                && logits.data.len() >= queries * classes
                && boxes.data.len() >= queries * 4 =>
        {
            Ok((
                &logits.data[..queries * classes],
                &boxes.data[..queries * 4],
                classes,
            ))
        }
        _ => Err(unsupported(outputs)),
    }
}

fn unsupported(outputs: &[ModelOutput<'_>]) -> OrtError {
    let shapes: Vec<_> = outputs
        .iter()
        .map(|output| (output.name, output.shape))
        .collect();
    OrtError::UnsupportedModel(format!(
        "a DETR's outputs are {shapes:?}, not logits [queries, classes] and \
         pred_boxes [queries, 4]"
    ))
}

#[cfg(test)]
mod tests {
    use super::super::model::{Letterbox, decode_batch};
    use super::*;

    /// The logit a sigmoid makes `score` of.
    fn logit(score: f32) -> f32 {
        (score / (1.0 - score)).ln()
    }

    /// Three queries of three classes: a sure person in the middle, a
    /// second class unsure, and nothing.
    fn outputs() -> (Vec<f32>, Vec<f32>) {
        let logits = vec![
            logit(0.9),
            logit(0.2),
            logit(0.1), // query 0: class 0 at 0.9
            logit(0.1),
            logit(0.15),
            logit(0.05), // query 1: class 1 at 0.15
            -20.0,
            -20.0,
            -20.0, // query 2: nothing
        ];
        let boxes = vec![
            0.5, 0.5, 0.25, 0.5, // centred, a quarter across, half down
            0.25, 0.25, 0.1, 0.1, //
            0.0, 0.0, 0.0, 0.0,
        ];
        (logits, boxes)
    }

    fn decode(
        names: [&str; 2],
        order: [usize; 2],
        conf_threshold: f32,
    ) -> Result<Vec<ModelBox>, OrtError> {
        let (logits, boxes) = outputs();
        let data = [logits, boxes];
        let shapes = [[3, 3], [3, 4]];
        let outputs: Vec<ModelOutput<'_>> = order
            .iter()
            .map(|&which| ModelOutput {
                name: names[which],
                shape: &shapes[which],
                data: &data[which],
            })
            .collect();
        let options = OrtDetectorOptions {
            conf_threshold,
            ..OrtDetectorOptions::default()
        };
        Detr.decode(&outputs, (200, 100), &options)
    }

    /// A query is its best class, scored by that class's sigmoid, its box
    /// the centre and size made pixels of the input; a query under the
    /// threshold is nothing.
    #[test]
    fn a_query_is_its_best_class_in_pixels_of_the_input() {
        let found = decode(["logits", "pred_boxes"], [0, 1], 0.25).expect("reads");
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].class_id, 0);
        assert!((found[0].score - 0.9).abs() < 1e-5, "{found:?}");
        assert_eq!(found[0].corners, [75.0, 25.0, 125.0, 75.0]);

        let unsure = decode(["logits", "pred_boxes"], [0, 1], 0.1).expect("reads");
        assert_eq!(unsure.len(), 2, "{unsure:?}");
        assert_eq!(unsure[1].class_id, 1);
        assert!((unsure[1].score - 0.15).abs() < 1e-5, "{unsure:?}");
    }

    /// Outputs named otherwise, in either order, are told apart by their
    /// shapes; outputs that are not a DETR's are refused.
    #[test]
    fn unnamed_outputs_are_read_by_their_shapes() {
        for order in [[0, 1], [1, 0]] {
            let found = decode(["scores", "boxes"], order, 0.25).expect("reads");
            assert_eq!(found[0].corners, [75.0, 25.0, 125.0, 75.0]);
        }
        let data = vec![0.0; 12];
        let only = [ModelOutput {
            name: "output0",
            shape: &[3, 4],
            data: &data,
        }];
        assert!(matches!(
            Detr.decode(&only, (200, 100), &OrtDetectorOptions::default()),
            Err(OrtError::UnsupportedModel(_))
        ));
    }

    /// A picture fitted into the input with a margin is read back onto the
    /// picture.
    #[test]
    fn a_box_is_read_back_through_the_letterbox() {
        let (logits, mut boxes) = outputs();
        // Centred, a quarter across and an eighth down.
        boxes[..4].copy_from_slice(&[0.5, 0.5, 0.25, 0.125]);
        let outputs = [
            ModelOutput {
                name: "logits",
                shape: &[1, 3, 3],
                data: &logits,
            },
            ModelOutput {
                name: "pred_boxes",
                shape: &[1, 3, 4],
                data: &boxes,
            },
        ];
        // A 400 by 100 picture in a 200 by 200 input: halved, 75 down.
        let letterboxes = [Letterbox::new((400, 100), (200, 200))];
        let found = decode_batch(
            &Detr,
            &outputs,
            &letterboxes,
            (200, 200),
            &OrtDetectorOptions::default(),
        )
        .expect("reads");
        let [person] = &found[0][..] else {
            panic!("{found:?}");
        };
        // Input (75, 87.5)–(125, 112.5): the picture's (150, 25)–(250, 75), as
        // fractions of it.
        let near = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(
            near(person.x, 0.375)
                && near(person.y, 0.25)
                && near(person.width, 0.25)
                && near(person.height, 0.5),
            "{person:?}"
        );
    }
}
