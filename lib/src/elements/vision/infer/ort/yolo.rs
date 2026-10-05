//! Reading an Ultralytics YOLO model's output, and fitting a picture to its
//! input the way Ultralytics trains it — see [`OrtDetectorOptions`] for the
//! two layouts.

use ndarray::{ArrayView2, ArrayViewD, Axis, Ix3};

use super::{OrtDetectorError, OrtDetectorOptions};
use crate::elements::Detection;

/// How a picture is fitted inside the model's input: scaled to fit,
/// proportions kept, centred, the rest grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Letterbox {
    /// The picture's size.
    pub(crate) frame: (u32, u32),
    /// The model's input size.
    pub(crate) model: (u32, u32),
    /// The size the picture is scaled to inside the input.
    pub(crate) scaled: (u32, u32),
    /// Where the scaled picture's top-left corner sits in the input.
    pub(crate) offset: (u32, u32),
}

impl Letterbox {
    pub(crate) fn new(frame: (u32, u32), model: (u32, u32)) -> Self {
        let scale =
            (model.0 as f32 / frame.0.max(1) as f32).min(model.1 as f32 / frame.1.max(1) as f32);
        let scaled = (
            ((frame.0 as f32 * scale).round() as u32).clamp(1, model.0),
            ((frame.1 as f32 * scale).round() as u32).clamp(1, model.1),
        );
        Self {
            frame,
            model,
            scaled,
            offset: ((model.0 - scaled.0) / 2, (model.1 - scaled.1) / 2),
        }
    }

    /// A point of the model's input, as fractions of the picture, clamped
    /// to it.
    fn onto_frame(&self, x: f32, y: f32) -> (f32, f32) {
        (
            ((x - self.offset.0 as f32) / self.scaled.0 as f32).clamp(0.0, 1.0),
            ((y - self.offset.1 as f32) / self.scaled.1 as f32).clamp(0.0, 1.0),
        )
    }

    /// A box of the model's input by its corners, as a [`Detection`] on the
    /// picture.
    fn detection(&self, class_id: usize, score: f32, corners: [f32; 4]) -> Detection {
        let (x1, y1) = self.onto_frame(corners[0], corners[1]);
        let (x2, y2) = self.onto_frame(corners[2], corners[3]);
        Detection::new(class_id, score, x1, y1, x2 - x1, y2 - y1)
    }
}

/// Reads a model's output into what it found in the picture `letterbox`
/// fitted, most confident first — see [`OrtDetectorOptions`] for the two
/// layouts.
pub(crate) fn decode(
    output: ArrayViewD<'_, f32>,
    letterbox: &Letterbox,
    options: &OrtDetectorOptions,
) -> Result<Vec<Detection>, OrtDetectorError> {
    Ok(
        decode_batch(output, std::slice::from_ref(letterbox), options)?
            .pop()
            .expect("one picture"),
    )
}

/// [`decode`] for a batch: the model's output for as many pictures as
/// `letterboxes` fitted, one after another, read into what each was found
/// to hold.
pub(crate) fn decode_batch(
    output: ArrayViewD<'_, f32>,
    letterboxes: &[Letterbox],
    options: &OrtDetectorOptions,
) -> Result<Vec<Vec<Detection>>, OrtDetectorError> {
    let shape = output.shape().to_vec();
    match shape.as_slice() {
        [pictures, _, _] if *pictures == letterboxes.len() => {}
        other => {
            return Err(OrtDetectorError::UnsupportedModel(format!(
                "its output is {other:?}, not [{}, rows, columns]",
                letterboxes.len()
            )));
        }
    }
    // Three axes, known as such: walked through a view of a dimension
    // known only at run time, every step costs an index of its own.
    let output = output
        .into_dimensionality::<Ix3>()
        .map_err(|error| OrtDetectorError::UnsupportedModel(error.to_string()))?;
    letterboxes
        .iter()
        .enumerate()
        .map(|(picture, letterbox)| {
            decode_one(
                output.index_axis(Axis(0), picture),
                &shape,
                letterbox,
                options,
            )
        })
        .collect()
}

/// One picture's rows of an output of `shape`.
fn decode_one(
    output: ArrayView2<'_, f32>,
    shape: &[usize],
    letterbox: &Letterbox,
    options: &OrtDetectorOptions,
) -> Result<Vec<Detection>, OrtDetectorError> {
    let mut found: Vec<Detection> = if shape[2] == 6 {
        // `[boxes, 6]`: corners, score, class — already suppressed.
        output
            .axis_iter(Axis(0))
            .filter(|row| row[4] >= options.conf_threshold)
            .map(|row| {
                letterbox.detection(
                    row[5].max(0.0) as usize,
                    row[4],
                    [row[0], row[1], row[2], row[3]],
                )
            })
            .collect()
    } else if shape[1] > 4 {
        // `[4 + classes, boxes]`: a column per box. Each box's best class
        // is found a row at a time — one class's score for every box, side
        // by side in memory — rather than a column at a time, whose values
        // lie a whole row apart: over YOLO11's 8400 boxes and 80 classes,
        // reading by column took more than a millisecond a picture.
        let mut best = vec![(0, f32::NEG_INFINITY); shape[2]];
        for (class_id, row) in output.axis_iter(Axis(0)).skip(4).enumerate() {
            let mut keep = |(best, &score): (&mut (usize, f32), &f32)| {
                if score > best.1 {
                    *best = (class_id, score);
                }
            };
            match row.as_slice() {
                Some(row) => best.iter_mut().zip(row).for_each(&mut keep),
                None => best.iter_mut().zip(row.iter()).for_each(&mut keep),
            }
        }
        let boxes = best.iter().enumerate().map(|(index, &(class_id, score))| {
            let at = |row: usize| output[[row, index]];
            ([at(0), at(1), at(2), at(3)], class_id, score)
        });
        suppressed(boxes, letterbox, options)
    } else {
        return Err(OrtDetectorError::UnsupportedModel(format!(
            "its output is {shape:?}, neither [batch, 4 + classes, boxes] nor [batch, boxes, 6]"
        )));
    };
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(found)
}

/// [`decode_batch`] for a `[batch, 4 + classes, boxes]` output whose boxes'
/// best classes were found where it is, on the GPU: six floats a box —
/// centre, width and height, the best class's score, that class — `boxes`
/// boxes a picture, one picture after another.
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
            let mut found = suppressed(boxes, letterbox, options);
            found.sort_by(|a, b| b.score.total_cmp(&a.score));
            found
        })
        .collect()
}

/// What a `[4 + classes, boxes]` output's boxes — each its centre and size
/// in the model's input, its best class and that class's score — come to on
/// the picture: those confident enough, less the duplicates.
fn suppressed(
    boxes: impl Iterator<Item = ([f32; 4], usize, f32)>,
    letterbox: &Letterbox,
    options: &OrtDetectorOptions,
) -> Vec<Detection> {
    let candidates = boxes
        .filter(|(_, _, score)| *score >= options.conf_threshold)
        .map(|([cx, cy, w, h], class_id, score)| {
            (
                class_id,
                score,
                [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            )
        })
        .collect();
    non_max_suppression(candidates, options.iou_threshold)
        .into_iter()
        .map(|(class_id, score, corners)| letterbox.detection(class_id, score, corners))
        .collect()
}

/// One candidate: class, score, corners in the model's input.
type Candidate = (usize, f32, [f32; 4]);

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

/// Highest score first, keeping each box that overlaps no kept box of its
/// own class past `iou_threshold` — Ultralytics' default, per-class NMS.
fn non_max_suppression(mut candidates: Vec<Candidate>, iou_threshold: f32) -> Vec<Candidate> {
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut kept: Vec<Candidate> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let duplicate = kept
            .iter()
            .any(|kept| kept.0 == candidate.0 && iou(&kept.2, &candidate.2) > iou_threshold);
        if !duplicate {
            kept.push(candidate);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use super::*;

    #[test]
    fn a_wide_picture_is_fitted_with_grey_above_and_below() {
        let letterbox = Letterbox::new((1920, 1080), (640, 640));
        assert_eq!(letterbox.scaled, (640, 360));
        assert_eq!(letterbox.offset, (0, 140));
    }

    /// A box drawn on the fitted input lands on the same place of the
    /// picture, as fractions of it.
    #[test]
    fn an_end_to_end_row_is_mapped_back_onto_the_picture() {
        let letterbox = Letterbox::new((1920, 1080), (640, 640));
        // The right half of the picture, top to bottom, in input pixels.
        let mut output = Array3::<f32>::zeros((1, 2, 6));
        output
            .slice_mut(ndarray::s![0, 0, ..])
            .assign(&ndarray::arr1(&[320.0, 140.0, 640.0, 500.0, 0.9, 2.0]));
        output
            .slice_mut(ndarray::s![0, 1, ..])
            .assign(&ndarray::arr1(&[0.0, 0.0, 10.0, 10.0, 0.1, 0.0]));
        let found = decode(
            output.view().into_dyn(),
            &letterbox,
            &OrtDetectorOptions::default(),
        )
        .unwrap();
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
        let columns = [
            [100.0, 100.0, 50.0, 50.0, 0.9, 0.0],
            [102.0, 101.0, 50.0, 50.0, 0.8, 0.0],
            [101.0, 100.0, 50.0, 50.0, 0.0, 0.7],
        ];
        let mut output = Array3::<f32>::zeros((1, 6, columns.len()));
        for (box_index, column) in columns.iter().enumerate() {
            for (row, value) in column.iter().enumerate() {
                output[[0, row, box_index]] = *value;
            }
        }
        let found = decode(
            output.view().into_dyn(),
            &letterbox,
            &OrtDetectorOptions::default(),
        )
        .unwrap();
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
        let mut output = Array3::<f32>::zeros((2, 1, 6));
        for picture in 0..2 {
            output
                .slice_mut(ndarray::s![picture, 0, ..])
                .assign(&ndarray::arr1(&[320.0, 320.0, 400.0, 400.0, 0.9, 0.0]));
        }
        let found = decode_batch(
            output.view().into_dyn(),
            &[wide, tall],
            &OrtDetectorOptions::default(),
        )
        .unwrap();
        let one = |n: usize| {
            decode(
                output.slice(ndarray::s![n..n + 1, .., ..]).into_dyn(),
                &[wide, tall][n],
                &OrtDetectorOptions::default(),
            )
            .unwrap()
        };
        assert_eq!(found, vec![one(0), one(1)]);
        assert_ne!(found[0][0].width, found[1][0].width, "{found:?}");
        assert!(
            decode_batch(
                output.view().into_dyn(),
                &[wide],
                &OrtDetectorOptions::default()
            )
            .is_err(),
            "two pictures' output for one letterbox"
        );
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
        let pictures = [
            [
                [100.0, 100.0, 50.0, 50.0, 0.9, 0.2],
                [102.0, 101.0, 50.0, 50.0, 0.3, 0.8],
                [400.0, 300.0, 80.0, 40.0, 0.1, 0.05],
            ],
            [
                [320.0, 320.0, 100.0, 200.0, 0.6, 0.6],
                [10.0, 10.0, 5.0, 5.0, 0.0, 0.0],
                [500.0, 500.0, 60.0, 60.0, 0.2, 0.7],
            ],
        ];
        let mut output = Array3::<f32>::zeros((2, 6, 3));
        let mut best = Vec::new();
        for (picture, boxes) in pictures.iter().enumerate() {
            for (index, column) in boxes.iter().enumerate() {
                for (row, value) in column.iter().enumerate() {
                    output[[picture, row, index]] = *value;
                }
                let (class, score) = if column[5] > column[4] {
                    (1.0, column[5])
                } else {
                    (0.0, column[4])
                };
                best.extend([column[0], column[1], column[2], column[3], score, class]);
            }
        }
        let options = OrtDetectorOptions::default();
        let whole = decode_batch(output.view().into_dyn(), &letterboxes, &options).unwrap();
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
        let output = Array3::<f32>::zeros((2, 3, 6));
        let letterbox = Letterbox::new((640, 640), (640, 640));
        assert!(
            decode(
                output.view().into_dyn(),
                &letterbox,
                &OrtDetectorOptions::default()
            )
            .is_err()
        );
    }
}
