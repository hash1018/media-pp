//! What the embedders share: their options, how an object is cut from the
//! picture for the model — straightened by its five points or stretched
//! from its box — and how the model's output is read into vectors. Which
//! objects are looked at, and what is remembered of each followed one, is
//! the classifiers' (`classify`).

use std::sync::Arc;

use super::OrtError;
use super::classify::Selects;
use super::model::ModelInput;
use crate::elements::{Cutout, Detection, Embedding};
use crate::orientation::Orientation;

/// How an embedder decides what to look at, and how it fits each object to
/// its model.
#[derive(Debug, Clone, PartialEq)]
pub struct OrtEmbedderOptions {
    /// The values the model wants. A face model's are its own: OpenCV's
    /// SFace takes RGB from 0 to 255, `ModelInput { scale: [255.0; 3],
    /// ..ModelInput::default() }`, and AdaFace BGR from -1 to 1,
    /// `ModelInput { order: ChannelOrder::Bgr, scale: [2.0; 3], bias: [-1.0;
    /// 3] }`. Values a model was not trained on give vectors that tell
    /// nothing apart.
    pub input: ModelInput,
    /// How each object is cut from the picture.
    pub align: Alignment,
    /// Which detected classes it embeds — a face model, the detector's
    /// faces — or `None` for every one.
    pub classes: Option<Vec<usize>>,
    /// The fewest pixels an object's box may have across or down to be
    /// embedded: a smaller one is too little picture to tell by.
    pub min_size: u32,
    /// The lowest detection score an object is embedded at — a detector
    /// before a tracker keeps unsure boxes, which are not looked at.
    pub min_detection_score: f32,
    /// How many pictures a followed object's vector is kept before it is
    /// embedded again; 0 keeps the first for as long as it is followed. An
    /// object with no tracker's number is embedded on every picture it is
    /// detected on.
    pub reembed: u32,
    /// Whether each vector carries the picture it was made from
    /// ([`Embedding::cutout`](crate::elements::Embedding::cutout)): the
    /// object as the model saw it, for a person to know it by — a face's
    /// thumbnail. It is read back from the model's own input, so it costs
    /// a copy of that input for every object embedded, which is once a
    /// [`reembed`](Self::reembed) for an object followed.
    pub cutouts: bool,
}

impl Default for OrtEmbedderOptions {
    /// RGB from 0 to 1, straightened by five points, every class, boxes of
    /// 32 pixels or more detected at 0.5 or more, a vector kept for a second
    /// at 30 frames a second.
    fn default() -> Self {
        Self {
            input: ModelInput::default(),
            align: Alignment::FivePoints,
            classes: None,
            min_size: 32,
            min_detection_score: 0.5,
            reembed: 30,
            cutouts: false,
        }
    }
}

impl Selects for OrtEmbedderOptions {
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
        self.reembed
    }
}

/// How an object is cut from the picture for an embedding model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Alignment {
    /// Turned, scaled and moved so that its five
    /// [`landmarks`](Detection::landmarks) — eyes, nose, the corners of the
    /// mouth, as a face detector such as
    /// [`DetectorModel::RetinaFace`](super::DetectorModel::RetinaFace)
    /// finds them — fall as near as they can on where the ArcFace family of
    /// face models expects them: the 112 by 112 template every one of them
    /// is trained on, scaled to the model's input. An object without five
    /// points is not embedded.
    #[default]
    FivePoints,
    /// Its box, the right way up, stretched to the model's input — for a
    /// model of whole objects, such as a person re-identification model.
    Box,
}

/// Where the ArcFace family expects a face's eyes, nose and mouth corners,
/// in pixels of a 112 by 112 input, each pixel's centre at whole numbers.
pub(crate) const ARCFACE_TEMPLATE: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];

/// Where each pixel of a model's input is read from in the picture as
/// stored: input pixel `(u, v)` from `(m[0] u + m[1] v + m[2], m[3] u + m[4]
/// v + m[5])`, both with each pixel's centre at whole numbers, sampled
/// between the four pixels around it and black outside the picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Warp(pub(crate) [f32; 6]);

impl Warp {
    /// The point of the picture input pixel `(u, v)` is read from.
    pub(crate) fn source(&self, u: f32, v: f32) -> (f32, f32) {
        let m = self.0;
        (m[0] * u + m[1] * v + m[2], m[3] * u + m[4] * v + m[5])
    }

    /// How `object`, on a picture `frame` in size, is cut for a model of
    /// input `model` as `align` says; `None` where it cannot be.
    pub(crate) fn of(
        object: &Detection,
        align: Alignment,
        frame: (u32, u32),
        orientation: Orientation,
        model: (u32, u32),
    ) -> Option<Self> {
        match align {
            Alignment::FivePoints => Self::five_points(&object.landmarks, frame, model),
            Alignment::Box => Some(Self::of_box(object, frame, orientation, model)),
        }
    }

    /// The face whose five points are `landmarks` — fractions of a picture
    /// `frame` in size — straightened onto the template scaled to `model`.
    pub(crate) fn five_points(
        landmarks: &[(f32, f32)],
        frame: (u32, u32),
        model: (u32, u32),
    ) -> Option<Self> {
        if landmarks.len() != 5 {
            return None;
        }
        // In pixels, each pixel's centre at whole numbers, as the template's.
        let points: Vec<(f32, f32)> = landmarks
            .iter()
            .map(|&(x, y)| (x * frame.0 as f32 - 0.5, y * frame.1 as f32 - 0.5))
            .collect();
        let (sx, sy) = (model.0 as f32 / 112.0, model.1 as f32 / 112.0);
        let template: Vec<(f32, f32)> = ARCFACE_TEMPLATE
            .iter()
            .map(|&(x, y)| ((x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5))
            .collect();
        // The input from the picture, then the other way round: what is
        // sampled is where each input pixel comes from.
        let [c, d, tx, ty] = similarity(&points, &template)?;
        let length = c * c + d * d;
        let (ic, id) = (c / length, d / length);
        Some(Self([
            ic,
            id,
            -(ic * tx + id * ty),
            -id,
            ic,
            id * tx - ic * ty,
        ]))
    }

    /// `object`'s box, the right way up as the picture is shown, stretched
    /// over the whole input.
    fn of_box(
        object: &Detection,
        frame: (u32, u32),
        orientation: Orientation,
        model: (u32, u32),
    ) -> Self {
        let [x, y, w, h] =
            orientation.to_display([object.x, object.y, object.width, object.height]);
        // Three corners of the box as shown, where they are stored, in
        // pixels.
        let stored = |x: f32, y: f32| {
            let [x, y, _, _] = orientation.from_display([x, y, 0.0, 0.0]);
            (x * frame.0 as f32, y * frame.1 as f32)
        };
        let origin = stored(x, y);
        let right = stored(x + w, y);
        let down = stored(x, y + h);
        let (mw, mh) = (model.0 as f32, model.1 as f32);
        let across = ((right.0 - origin.0) / mw, (right.1 - origin.1) / mw);
        let below = ((down.0 - origin.0) / mh, (down.1 - origin.1) / mh);
        // Input pixel (u, v)'s centre, u + 0.5 of mw across, as a pixel of
        // the picture with its centre at whole numbers.
        Self([
            across.0,
            below.0,
            origin.0 + 0.5 * across.0 + 0.5 * below.0 - 0.5,
            across.1,
            below.1,
            origin.1 + 0.5 * across.1 + 0.5 * below.1 - 0.5,
        ])
    }
}

/// The turn, scale and move that take `from` nearest `to` by least squares:
/// `[c, d, tx, ty]`, a point `(x, y)` going to `(c x - d y + tx, d x + c y +
/// ty)` — Umeyama's estimate, as insightface aligns faces. `None` where the
/// points do not span a direction.
pub(crate) fn similarity(from: &[(f32, f32)], to: &[(f32, f32)]) -> Option<[f32; 4]> {
    let n = from.len().min(to.len()) as f64;
    if n < 2.0 {
        return None;
    }
    let mean = |points: &[(f32, f32)]| {
        let (x, y) = points.iter().fold((0.0, 0.0), |(x, y), p| {
            (x + f64::from(p.0), y + f64::from(p.1))
        });
        (x / n, y / n)
    };
    let (fx, fy) = mean(from);
    let (tx, ty) = mean(to);
    // As complex numbers, to = w from + t: w is the sum of conj(from) to
    // over that of |from|², each about its mean.
    let (mut re, mut im, mut spread) = (0.0, 0.0, 0.0);
    for (a, b) in from.iter().zip(to) {
        let (ax, ay) = (f64::from(a.0) - fx, f64::from(a.1) - fy);
        let (bx, by) = (f64::from(b.0) - tx, f64::from(b.1) - ty);
        re += ax * bx + ay * by;
        im += ax * by - ay * bx;
        spread += ax * ax + ay * ay;
    }
    if spread < 1e-9 {
        return None;
    }
    let (c, d) = (re / spread, im / spread);
    Some([
        c as f32,
        d as f32,
        (tx - (c * fx - d * fy)) as f32,
        (ty - (d * fx + c * fy)) as f32,
    ])
}

/// The pictures `rows` inputs of a model were made from, `size` each,
/// read back from `tensor` — the inputs, one after another, three planes
/// each — by undoing `values`: each plane less its bias, over its scale,
/// put back in RGB order and made eight bits.
pub(crate) fn cutouts(
    tensor: &[f32],
    rows: usize,
    size: (u32, u32),
    values: ModelInput,
) -> Vec<Arc<Cutout>> {
    let (width, height) = (size.0 as usize, size.1 as usize);
    let plane = width * height;
    let planes = values.planes();
    tensor
        .chunks(plane * 3)
        .take(rows)
        .filter_map(|input| {
            let mut rgb = vec![0u8; plane * 3];
            for (channel, &p) in planes.iter().enumerate() {
                let (scale, bias) = (values.scale[p], values.bias[p]);
                let from = input.get(p * plane..(p + 1) * plane)?;
                for (pixel, &value) in from.iter().enumerate() {
                    let unit = if scale == 0.0 {
                        0.0
                    } else {
                        (value - bias) / scale
                    };
                    rgb[pixel * 3 + channel] = (unit.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
            Cutout::new(size.0, size.1, rgb).map(Arc::new)
        })
        .collect()
}

/// `found`, each with the picture its vector was made from — `cutouts` in
/// the same order; one with no vector is left so.
pub(crate) fn with_cutouts(
    found: Vec<Option<Embedding>>,
    cutouts: Vec<Arc<Cutout>>,
) -> Vec<Option<Embedding>> {
    let mut cutouts = cutouts.into_iter();
    found
        .into_iter()
        .map(|embedding| {
            let cutout = cutouts.next();
            match (embedding, cutout) {
                (Some(embedding), Some(cutout)) => Some(embedding.with_cutout(cutout)),
                (embedding, _) => embedding,
            }
        })
        .collect()
}

/// Each picture's vector from a model's first output of `shape` — `[batch,
/// dim]`, or `[batch, dim, 1, 1]` as some export it — the first `rows` of
/// it, made `embedder`'s.
pub(crate) fn embeddings(
    embedder: &Arc<str>,
    shape: &[usize],
    data: &[f32],
    rows: usize,
) -> Result<Vec<Option<Embedding>>, OrtError> {
    let batch = shape.first().copied().unwrap_or(0);
    let dim: usize = shape.iter().skip(1).product();
    if batch < rows || dim == 0 || data.len() < rows * dim {
        return Err(OrtError::UnsupportedModel(format!(
            "its output is {shape:?}, not a vector for each of {rows}"
        )));
    }
    Ok(data[..rows * dim]
        .chunks(dim)
        .map(|row| Embedding::new(Arc::clone(embedder), row))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orientation::Rotation;

    fn near(a: (f32, f32), b: (f32, f32), within: f32) -> bool {
        (a.0 - b.0).abs() < within && (a.1 - b.1).abs() < within
    }

    /// A model's input made from known pixels — in BGR, scaled and moved —
    /// gives back those pixels, input by input.
    #[test]
    fn a_cutout_is_the_picture_the_input_was_made_from() {
        let values = ModelInput {
            order: super::super::ChannelOrder::Bgr,
            scale: [2.0; 3],
            bias: [-1.0; 3],
        };
        // Two inputs of 2 by 1 pixels: red then grey; blue then white.
        let pixels: [[[u8; 3]; 2]; 2] = [[[255, 0, 0], [128, 128, 128]], [[0, 0, 255], [255; 3]]];
        let planes = values.planes();
        let mut tensor = vec![0.0f32; 2 * 3 * 2];
        for (row, input) in pixels.iter().enumerate() {
            for (pixel, rgb) in input.iter().enumerate() {
                for (channel, &value) in rgb.iter().enumerate() {
                    let p = planes[channel];
                    tensor[row * 6 + p * 2 + pixel] =
                        f32::from(value) / 255.0 * values.scale[p] + values.bias[p];
                }
            }
        }
        let back = cutouts(&tensor, 2, (2, 1), values);
        assert_eq!(back.len(), 2);
        for (cutout, input) in back.iter().zip(&pixels) {
            assert_eq!((cutout.width, cutout.height), (2, 1));
            assert_eq!(&cutout.rgb[..], input.as_flattened());
        }
    }

    /// Points turned a third of a quarter, doubled and moved are brought
    /// back by the transform found from them.
    #[test]
    fn a_turn_scale_and_move_is_recovered() {
        let (angle, scale, shift) = (0.5f32, 2.0f32, (30.0f32, -12.0f32));
        let (c, d) = (scale * angle.cos(), scale * angle.sin());
        let from = ARCFACE_TEMPLATE.to_vec();
        let to: Vec<(f32, f32)> = from
            .iter()
            .map(|&(x, y)| (c * x - d * y + shift.0, d * x + c * y + shift.1))
            .collect();
        let found = similarity(&from, &to).expect("found");
        for (got, want) in found.iter().zip([c, d, shift.0, shift.1]) {
            assert!((got - want).abs() < 1e-3, "{found:?}");
        }
        assert!(similarity(&[(1.0, 1.0); 5], &to).is_none(), "one point");
    }

    /// A face whose points are the template's, at the template's size, is
    /// read as it is; one twice the size at twice the place is read from
    /// twice as far — and the points land on the template.
    #[test]
    fn five_points_on_the_template_read_it_where_it_is() {
        let frame = (224u32, 224u32);
        // The template's points, as fractions of a picture twice its size.
        let landmarks: Vec<(f32, f32)> = ARCFACE_TEMPLATE
            .iter()
            .map(|&(x, y)| ((2.0 * x + 1.0) / 224.0, (2.0 * y + 1.0) / 224.0))
            .collect();
        let warp = Warp::five_points(&landmarks, frame, (112, 112)).expect("five");
        for &(x, y) in &ARCFACE_TEMPLATE {
            let source = warp.source(x, y);
            assert!(
                near(source, (2.0 * x + 0.5, 2.0 * y + 0.5), 1e-3),
                "{source:?}"
            );
        }
        assert!(Warp::five_points(&landmarks[..4], frame, (112, 112)).is_none());
    }

    /// A box is stretched over the whole input: the input's corners read
    /// the box's, the right way up for a picture shown turned.
    #[test]
    fn a_box_is_stretched_over_the_input_the_right_way_up() {
        let object = Detection::new(0, 0.9, 0.25, 0.5, 0.5, 0.25);
        let upright = Warp::of(
            &object,
            Alignment::Box,
            (400, 200),
            Orientation::UPRIGHT,
            (100, 50),
        )
        .unwrap();
        // Input pixel (0, 0)'s centre is a hundredth of the box in: pixel
        // 100 + 1 across, 100 + 0.5 down, less half a pixel.
        assert!(near(upright.source(0.0, 0.0), (100.5, 100.0), 1e-4));
        assert!(near(upright.source(99.0, 49.0), (298.5, 149.0), 1e-4));

        // Stored on its side and shown a quarter turn clockwise: the top of
        // what is shown is the stored left edge, so the input's top-left
        // reads the stored bottom-left.
        let turned = Warp::of(
            &Detection::new(0, 0.9, 0.0, 0.0, 0.5, 1.0),
            Alignment::Box,
            (400, 200),
            Orientation::rotated(Rotation::Clockwise90),
            (50, 50),
        )
        .unwrap();
        let (x, y) = turned.source(0.0, 0.0);
        assert!(x < 10.0 && y > 190.0, "({x}, {y})");
        let (x, y) = turned.source(49.0, 0.0);
        assert!(x < 10.0 && y < 10.0, "({x}, {y})");
    }

    #[test]
    fn each_row_of_the_output_is_a_vector() {
        let name: Arc<str> = Arc::from("e");
        let found = embeddings(&name, &[2, 2, 1, 1], &[3.0, 4.0, 0.0, 2.0], 2).unwrap();
        assert_eq!(&*found[0].as_ref().unwrap().vector, [0.6, 0.8]);
        assert_eq!(&*found[1].as_ref().unwrap().vector, [0.0, 1.0]);
        // A fixed batch of four, two of them asked for.
        assert_eq!(embeddings(&name, &[4, 1], &[1.0; 4], 2).unwrap().len(), 2);
        assert!(embeddings(&name, &[1, 2], &[1.0; 2], 2).is_err());
    }
}
