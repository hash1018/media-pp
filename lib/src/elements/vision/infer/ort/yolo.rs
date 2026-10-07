//! Reading an Ultralytics YOLO model's output — see [`OrtDetectorOptions`]
//! for the two layouts.

#[cfg(any(feature = "ort-cuda", all(target_os = "macos", feature = "ort-coreml")))]
use super::model::Letterbox;
use super::model::{DetectorDecoder, ModelBox, ModelOutput, non_max_suppression};
use super::{OrtDetectorOptions, OrtError};
#[cfg(any(feature = "ort-cuda", all(target_os = "macos", feature = "ort-coreml")))]
use crate::elements::Detection;

/// What reads a YOLO model's output: [`DetectorModel::Yolo`]'s decoder.
///
/// [`DetectorModel::Yolo`]: super::DetectorModel::Yolo
#[derive(Debug)]
pub(crate) struct Yolo;

impl DetectorDecoder for Yolo {
    fn decode(
        &self,
        outputs: &[ModelOutput<'_>],
        _input: (u32, u32),
        options: &OrtDetectorOptions,
    ) -> Result<Vec<ModelBox>, OrtError> {
        let output = outputs
            .first()
            .ok_or_else(|| OrtError::UnsupportedModel("the model has no output".into()))?;
        match *output.shape {
            // `[boxes, 6]`: corners, score, class — already suppressed.
            [_, 6] => Ok(output
                .data
                .as_chunks::<6>()
                .0
                .iter()
                .filter(|row| row[4] >= options.conf_threshold)
                .map(|row| {
                    ModelBox::new(
                        row[5].max(0.0) as usize,
                        row[4],
                        [row[0], row[1], row[2], row[3]],
                    )
                })
                .collect()),
            // `[4 + classes, boxes]`: a column per box. Each box's best class
            // is found a row at a time — one class's score for every box,
            // side by side in memory — rather than a column at a time, whose
            // values lie a whole row apart: over YOLO11's 8400 boxes and 80
            // classes, reading by column took more than a millisecond a
            // picture.
            [rows, boxes] if rows > 4 => {
                let row = |row: usize| &output.data[row * boxes..(row + 1) * boxes];
                let mut best = vec![(0, f32::NEG_INFINITY); boxes];
                for class_id in 0..rows - 4 {
                    for (best, &score) in best.iter_mut().zip(row(4 + class_id)) {
                        if score > best.1 {
                            *best = (class_id, score);
                        }
                    }
                }
                let (cx, cy, w, h) = (row(0), row(1), row(2), row(3));
                let boxes = best.iter().enumerate().map(|(index, &(class_id, score))| {
                    ([cx[index], cy[index], w[index], h[index]], class_id, score)
                });
                Ok(suppressed(boxes, options))
            }
            ref other => Err(OrtError::UnsupportedModel(format!(
                "its output is {other:?} a picture, neither [4 + classes, boxes] nor [boxes, 6]"
            ))),
        }
    }
}

/// What a `[batch, 4 + classes, boxes]` output whose boxes' best classes
/// were found where it is, on the GPU — six floats a box: centre, width and
/// height, the best class's score, that class; `boxes` boxes a picture, one
/// picture after another — holds for each picture `letterboxes` fitted.
#[cfg(any(feature = "ort-cuda", all(target_os = "macos", feature = "ort-coreml")))]
pub(crate) fn decode_best(
    best: &[f32],
    boxes: usize,
    letterboxes: &[Letterbox],
    options: &OrtDetectorOptions,
) -> Vec<Vec<Detection>> {
    letterboxes
        .iter()
        .enumerate()
        .map(|(picture, letterbox)| {
            let rows = best
                .get(picture * boxes * 6..(picture + 1) * boxes * 6)
                .unwrap_or_default();
            let boxes = rows
                .as_chunks::<6>()
                .0
                .iter()
                .map(|&[cx, cy, w, h, score, class]| {
                    ([cx, cy, w, h], class.max(0.0) as usize, score)
                });
            letterbox.detections(suppressed(boxes, options))
        })
        .collect()
}

/// What a `[4 + classes, boxes]` output's boxes — each its centre and size
/// in the model's input, its best class and that class's score — come to:
/// those confident enough, less the duplicates.
fn suppressed(
    boxes: impl Iterator<Item = ([f32; 4], usize, f32)>,
    options: &OrtDetectorOptions,
) -> Vec<ModelBox> {
    let candidates = boxes
        .filter(|(_, _, score)| *score >= options.conf_threshold)
        .map(|([cx, cy, w, h], class_id, score)| {
            ModelBox::new(
                class_id,
                score,
                [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            )
        })
        .collect();
    non_max_suppression(candidates, options.iou_threshold)
}

#[cfg(test)]
mod tests {
    use super::super::model::{Letterbox, decode_batch};
    use super::*;

    /// One output of `shape`, the whole run's.
    fn output<'a>(shape: &'a [usize], data: &'a [f32]) -> [ModelOutput<'a>; 1] {
        [ModelOutput {
            name: "output0",
            shape,
            data,
        }]
    }

    /// `[batch, 4 + classes, boxes]` from each picture's boxes, a row each.
    fn columns(pictures: &[&[[f32; 6]]]) -> Vec<f32> {
        let mut data = Vec::new();
        for boxes in pictures {
            for row in 0..6 {
                data.extend(boxes.iter().map(|column| column[row]));
            }
        }
        data
    }

    /// A box drawn on the fitted input lands on the same place of the
    /// picture, as fractions of it.
    #[test]
    fn an_end_to_end_row_is_mapped_back_onto_the_picture() {
        let letterbox = Letterbox::new((1920, 1080), (640, 640));
        // The right half of the picture, top to bottom, in input pixels.
        let data = [
            320.0, 140.0, 640.0, 500.0, 0.9, 2.0, //
            0.0, 0.0, 10.0, 10.0, 0.1, 0.0,
        ];
        let found = decode_batch(
            &Yolo,
            &output(&[1, 2, 6], &data),
            &[letterbox],
            (640, 640),
            &OrtDetectorOptions::default(),
        )
        .unwrap()
        .remove(0);
        assert_eq!(found.len(), 1, "the 0.1 row is below the threshold");
        let car = &found[0];
        assert_eq!(car.class_id, 2);
        assert!((car.x - 0.5).abs() < 1e-6 && car.y.abs() < 1e-6, "{car:?}");
        assert!(
            (car.width - 0.5).abs() < 1e-6 && (car.height - 1.0).abs() < 1e-6,
            "{car:?}"
        );
    }

    /// The v8 layout: two overlapping boxes of one class are one object,
    /// and a box of another class beside them is not suppressed by them.
    #[test]
    fn overlapping_boxes_of_one_class_are_one_object() {
        let letterbox = Letterbox::new((640, 640), (640, 640));
        // Rows: cx, cy, w, h, then a score for each of two classes.
        let data = columns(&[&[
            [100.0, 100.0, 50.0, 50.0, 0.9, 0.0],
            [102.0, 101.0, 50.0, 50.0, 0.8, 0.0],
            [101.0, 100.0, 50.0, 50.0, 0.0, 0.7],
        ]]);
        let found = decode_batch(
            &Yolo,
            &output(&[1, 6, 3], &data),
            &[letterbox],
            (640, 640),
            &OrtDetectorOptions::default(),
        )
        .unwrap()
        .remove(0);
        let classes: Vec<_> = found
            .iter()
            .map(|found| (found.class_id, found.score))
            .collect();
        assert_eq!(classes, vec![(0, 0.9), (1, 0.7)]);
    }

    /// A batch's output is read a picture at a time, each through its own
    /// letterbox: the same box of the input is a different size on a wide
    /// picture and on a tall one.
    #[test]
    fn each_picture_of_a_batch_is_read_through_its_own_letterbox() {
        let wide = Letterbox::new((1920, 1080), (640, 640));
        let tall = Letterbox::new((1080, 1920), (640, 640));
        let row = [320.0, 320.0, 400.0, 400.0, 0.9, 0.0];
        let data = [row, row].concat();
        let options = OrtDetectorOptions::default();
        let found = decode_batch(
            &Yolo,
            &output(&[2, 1, 6], &data),
            &[wide, tall],
            (640, 640),
            &options,
        )
        .unwrap();
        let one = |letterbox: Letterbox| {
            decode_batch(
                &Yolo,
                &output(&[1, 1, 6], &row),
                &[letterbox],
                (640, 640),
                &options,
            )
            .unwrap()
            .remove(0)
        };
        assert_eq!(found, vec![one(wide), one(tall)]);
        assert_ne!(found[0][0].width, found[1][0].width, "{found:?}");
    }

    /// The boxes' best classes found on the GPU — six floats a box, as
    /// `best_class` writes them — read to what the whole output reads to on
    /// the CPU, ties and all.
    #[cfg(any(feature = "ort-cuda", all(target_os = "macos", feature = "ort-coreml")))]
    #[test]
    fn best_classes_found_on_the_gpu_read_as_the_whole_output_does() {
        let letterboxes = [
            Letterbox::new((1920, 1080), (640, 640)),
            Letterbox::new((640, 640), (640, 640)),
        ];
        // Two pictures of three boxes: centre, size, then two classes.
        let pictures: [&[[f32; 6]]; 2] = [
            &[
                [100.0, 100.0, 50.0, 50.0, 0.9, 0.2],
                [102.0, 101.0, 50.0, 50.0, 0.3, 0.8],
                [400.0, 300.0, 80.0, 40.0, 0.1, 0.05],
            ],
            &[
                [320.0, 320.0, 100.0, 200.0, 0.6, 0.6],
                [10.0, 10.0, 5.0, 5.0, 0.0, 0.0],
                [500.0, 500.0, 60.0, 60.0, 0.2, 0.7],
            ],
        ];
        let mut best = Vec::new();
        for boxes in pictures {
            for column in boxes {
                let (class, score) = if column[5] > column[4] {
                    (1.0, column[5])
                } else {
                    (0.0, column[4])
                };
                best.extend([column[0], column[1], column[2], column[3], score, class]);
            }
        }
        let options = OrtDetectorOptions::default();
        let data = columns(&pictures);
        let whole = decode_batch(
            &Yolo,
            &output(&[2, 6, 3], &data),
            &letterboxes,
            (640, 640),
            &options,
        )
        .unwrap();
        assert_eq!(decode_best(&best, 3, &letterboxes, &options), whole);
        assert!(
            whole[1]
                .iter()
                .any(|found| found.score == 0.6 && found.class_id == 0),
            "a tie is the first class's: {whole:?}"
        );
    }

    #[test]
    fn an_output_of_another_shape_is_refused() {
        let letterbox = Letterbox::new((640, 640), (640, 640));
        let data = [0.0; 15];
        let error = decode_batch(
            &Yolo,
            &output(&[1, 3, 5], &data),
            &[letterbox],
            (640, 640),
            &OrtDetectorOptions::default(),
        )
        .expect_err("three rows of five");
        assert!(error.to_string().contains("[3, 5]"), "{error}");
    }
}
