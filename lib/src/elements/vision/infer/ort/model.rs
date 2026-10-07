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

use super::{OrtDetectorOptions, OrtError, retinaface, retinaface::RetinaFace, yolo::Yolo};
use crate::buffer::Metadata;
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
    /// A RetinaFace face detector, as biubug6's `Pytorch_Retinaface`
    /// trains one and its exports to ONNX take and make it — MobileNet or
    /// ResNet: one input of OpenCV's BGR from 0 to 255 less a mean per
    /// channel, and three outputs of a row per anchor, the box's offsets
    /// from its anchor (`[batch, anchors, 4]`), the scores of background
    /// and face (`[batch, anchors, 2]`, probabilities or not) and the
    /// offsets of five points (`[batch, anchors, 10]`), in any order and by
    /// any name. Every box is a face, class 0, named `face` unless
    /// [`OrtDetectorOptions::labels`] names it otherwise, and carries the
    /// five points as its [`Detection::landmarks`]: the eyes, the nose and
    /// the corners of the mouth, as the model orders them. Duplicates are
    /// suppressed by [`OrtDetectorOptions::iou_threshold`].
    RetinaFace,
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
            Self::RetinaFace => f.write_str("RetinaFace"),
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
            (Self::Yolo, Self::Yolo) | (Self::RetinaFace, Self::RetinaFace) => true,
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
            Self::RetinaFace => retinaface::INPUT,
            Self::Custom { input, .. } => *input,
        }
    }

    /// What reads the output of a model of this kind, whose input is
    /// `input` in size.
    pub(crate) fn decoder(&self, input: (u32, u32)) -> Arc<dyn DetectorDecoder> {
        match self {
            Self::Yolo => Arc::new(Yolo),
            Self::RetinaFace => Arc::new(RetinaFace::new(input)),
            Self::Custom { decoder, .. } => Arc::clone(decoder),
        }
    }

    /// The class names a model of this kind has where it names none
    /// itself and none were given: `face`, for a face detector's one class.
    pub(crate) fn labels(&self, read: Arc<[Arc<str>]>) -> Arc<[Arc<str>]> {
        match self {
            Self::RetinaFace if read.is_empty() => Arc::from([Arc::from("face")]),
            _ => read,
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

/// How a picture — or a rectangle of it, one of its [`Tiles`] — is fitted
/// inside the model's input: turned the way it is shown, scaled to fit,
/// proportions kept, centred, the rest grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Letterbox {
    /// The picture's size, as stored.
    pub(crate) frame: (u32, u32),
    /// The rectangle of it fitted — left, top, width, height, in pixels of
    /// the picture as stored; the whole of it but for a tile.
    pub(crate) crop: (u32, u32, u32, u32),
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
        Self::cropped(frame, (0, 0, frame.0, frame.1), model, orientation)
    }

    /// `crop` of a picture stored `frame` in size, fitted as a picture of
    /// its own would be: turned as `orientation` says, as the picture is.
    pub(crate) fn cropped(
        frame: (u32, u32),
        crop: (u32, u32, u32, u32),
        model: (u32, u32),
        orientation: Orientation,
    ) -> Self {
        let shown = orientation.display_size(crop.2, crop.3);
        let scale =
            (model.0 as f32 / shown.0.max(1) as f32).min(model.1 as f32 / shown.1.max(1) as f32);
        let scaled = (
            ((shown.0 as f32 * scale).round() as u32).clamp(1, model.0),
            ((shown.1 as f32 * scale).round() as u32).clamp(1, model.1),
        );
        Self {
            frame,
            crop,
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

    /// A rectangle as fractions of the crop as stored, as fractions of the
    /// whole stored picture.
    fn uncrop(&self, [x, y, width, height]: [f32; 4]) -> [f32; 4] {
        let (left, top, w, h) = self.crop;
        let (fw, fh) = (self.frame.0 as f32, self.frame.1 as f32);
        [
            (left as f32 + x * w as f32) / fw,
            (top as f32 + y * h as f32) / fh,
            width * w as f32 / fw,
            height * h as f32 / fh,
        ]
    }

    /// A point of the model's input, as fractions of the stored picture.
    fn point(&self, x: f32, y: f32) -> (f32, f32) {
        let (x, y) = self.onto_frame(x, y);
        let [x, y, _, _] = self.uncrop(self.orientation.from_display([x, y, 0.0, 0.0]));
        (x, y)
    }

    /// A box the model found, as a [`Detection`] on the stored picture.
    pub(crate) fn detection(&self, found: ModelBox) -> Detection {
        let [left, top, right, bottom] = found.corners;
        let (x1, y1) = self.onto_frame(left, top);
        let (x2, y2) = self.onto_frame(right, bottom);
        let [x, y, width, height] =
            self.uncrop(self.orientation.from_display([x1, y1, x2 - x1, y2 - y1]));
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

/// Looking at a picture in overlapping tiles as well as whole: each tile is
/// fitted to the model's input as a picture of its own would be, so that a
/// face too small to find in the whole picture shrunk to the model's input
/// is found in a tile shrunk less — DeepStream's `nvdspreprocess` regions,
/// SAHI's slices.
///
/// On eleven public videos, a RetinaFace ResNet50 looking at 3 by 2 tiles
/// with a quarter overlap as well as the whole picture missed 19% of 976
/// faces where it missed 46% without, and found more than half of the faces
/// under 24 pixels where it found none — at seven runs of the model a
/// picture instead of one, and more boxes that are no face. Where one face
/// is found both in the whole picture and in a tile, the whole picture's box
/// is kept, with the more confident score: a tile can cut a large face in
/// two.
///
/// Each tile is looked in on every picture, unless [`Tiles::choose`] says
/// otherwise picture by picture.
#[derive(Debug, Clone)]
pub struct Tiles {
    /// Tiles across a picture shown wider than tall; down one shown taller,
    /// so that a portrait picture is cut the same way turned.
    pub across: u32,
    /// Tiles down a picture shown wider than tall; across one shown taller.
    pub down: u32,
    /// How much of a tile its neighbour covers too, as a fraction of it,
    /// from 0 to less than 1: a face on the line between two is whole in one
    /// of them where it is no wider than this.
    pub overlap: f32,
    /// Which tiles each picture is looked in — from what it carries in, the
    /// boxes an earlier detector found on it, say — or `None` for every one.
    /// The whole picture is always looked at.
    pub choose: Option<Arc<dyn TileChooser>>,
}

impl Default for Tiles {
    /// Three across and two down, a quarter overlapping — the arrangement
    /// measured above — every one looked in.
    fn default() -> Self {
        Self {
            across: 3,
            down: 2,
            overlap: 0.25,
            choose: None,
        }
    }
}

impl PartialEq for Tiles {
    /// The same arrangement, choosing by the same chooser — the same one,
    /// not one alike.
    fn eq(&self, other: &Self) -> bool {
        self.across == other.across
            && self.down == other.down
            && self.overlap == other.overlap
            && match (&self.choose, &other.choose) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
    }
}

/// What picks the tiles a detector looks in, picture by picture —
/// [`Tiles::choose`]. A tile can be passed over where nothing it could find
/// is there: in front of a face detector, where a person detector found no
/// small person, a tile holds no face the whole picture misses, and running
/// the model on it is time spent for nothing.
///
/// One chooser is shared by every detector it is given to, and called from
/// each one's own thread, once for each tile of each picture looked at.
///
/// On eleven public videos with 20 to 80 people a picture, looking with a
/// RetinaFace ResNet50 only in the tiles where a person detector's box put a
/// small head missed exactly the faces all six tiles missed, at five runs of
/// the model a picture instead of seven; where a few people were, three.
pub trait TileChooser: fmt::Debug + Send + Sync {
    /// Whether to look in `tile` of `picture` — left, top, width and height
    /// as fractions of the picture as stored, as a [`Detection`]'s box is.
    fn look_in(&self, picture: &TiledPicture<'_>, tile: [f32; 4]) -> bool;
}

/// A picture about to be looked at in tiles, as a [`TileChooser`] is told
/// of it.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TiledPicture<'a> {
    /// What the picture carries in, as earlier elements left it: what an
    /// earlier detector found is its `Detections`, and an element of the
    /// application's own may have put anything beside it.
    pub metadata: Option<&'a Metadata>,
    /// Its size as stored, in pixels.
    pub size: (u32, u32),
    /// Which way up it is shown: on a picture stored on its side, a box's
    /// top is not where a person's head is.
    pub orientation: Orientation,
    /// How many pixels of the model's input one of the picture's comes to
    /// where the picture is looked at whole: what decides whether a thing is
    /// too small to be found without a tile.
    pub whole_scale: f32,
}

impl Tiles {
    /// Whether these can cut a picture: at least one tile each way, and an
    /// overlap that leaves each tile something of its own.
    pub(crate) fn check(&self) -> Result<(), OrtError> {
        if self.across == 0 || self.down == 0 || !(0.0..0.9).contains(&self.overlap) {
            return Err(OrtError::InvalidTiles(self.clone()));
        }
        Ok(())
    }

    /// The rectangles of a picture stored `frame` in size and shown as
    /// `orientation` says, each left, top, width and height in stored
    /// pixels, starting on even ones and even in size, so that an NV12
    /// tile's chroma is its luma's.
    pub(crate) fn rectangles(
        &self,
        frame: (u32, u32),
        orientation: Orientation,
    ) -> Vec<(u32, u32, u32, u32)> {
        let shown = orientation.display_size(frame.0, frame.1);
        // Across and down as shown, then as stored.
        let (across, down) = if shown.0 >= shown.1 {
            (self.across, self.down)
        } else {
            (self.down, self.across)
        };
        let (across, down) = if shown == frame || shown.0 == shown.1 {
            (across, down)
        } else {
            (down, across)
        };
        let side = |length: u32, count: u32| {
            let size = length as f32 / (count as f32 - (count - 1) as f32 * self.overlap);
            ((size as u32) & !1).clamp(2.min(length), length)
        };
        let (width, height) = (side(frame.0, across), side(frame.1, down));
        let start = |index: u32, count: u32, length: u32, size: u32| {
            if count < 2 {
                0
            } else {
                (((length - size) as f32 * index as f32 / (count - 1) as f32) as u32) & !1
            }
        };
        let mut found = Vec::with_capacity((across * down) as usize);
        for row in 0..down {
            for column in 0..across {
                found.push((
                    start(column, across, frame.0, width),
                    start(row, down, frame.1, height),
                    width,
                    height,
                ));
            }
        }
        found
    }
}

/// Every way a detector looks at a picture stored `frame` in size, carrying
/// `metadata` in: the whole of it first, then each of `tiles` their chooser
/// picks.
pub(crate) fn looks(
    frame: (u32, u32),
    model: (u32, u32),
    orientation: Orientation,
    tiles: Option<&Tiles>,
    metadata: Option<&Metadata>,
) -> Vec<Letterbox> {
    let whole = Letterbox::shown(frame, model, orientation);
    let Some(tiles) = tiles else {
        return vec![whole];
    };
    let shown = orientation.display_size(frame.0, frame.1);
    let picture = TiledPicture {
        metadata,
        size: frame,
        orientation,
        whole_scale: whole.scaled.0 as f32 / shown.0.max(1) as f32,
    };
    let (width, height) = (frame.0.max(1) as f32, frame.1.max(1) as f32);
    let chosen = |&(x, y, w, h): &(u32, u32, u32, u32)| {
        tiles.choose.as_ref().is_none_or(|choose| {
            choose.look_in(
                &picture,
                [
                    x as f32 / width,
                    y as f32 / height,
                    w as f32 / width,
                    h as f32 / height,
                ],
            )
        })
    };
    let mut found = vec![whole];
    found.extend(
        tiles
            .rectangles(frame, orientation)
            .into_iter()
            .filter(chosen)
            .map(|crop| Letterbox::cropped(frame, crop, model, orientation)),
    );
    found
}

/// What one picture holds, from what each of its `looks` found — the whole
/// picture's first: where a tile's object is mostly inside one the whole
/// picture found of its class, or holds most of it, it is that one, which
/// keeps its own box and takes the tile's score where higher; the others the
/// tiles found are suppressed among themselves by `iou_threshold`, and kept.
/// Most confident first.
pub(crate) fn merge_looks(mut found: Vec<Vec<Detection>>, iou_threshold: f32) -> Vec<Detection> {
    if found.len() < 2 {
        return found.pop().unwrap_or_default();
    }
    let mut looks = found.into_iter();
    let mut whole = looks.next().unwrap_or_default();
    let mut tiled: Vec<Detection> = Vec::new();
    for item in looks.flatten() {
        let rect = [item.x, item.y, item.width, item.height];
        let same = whole
            .iter_mut()
            .filter(|kept| kept.class_id == item.class_id)
            .filter(|kept| {
                let kept_rect = [kept.x, kept.y, kept.width, kept.height];
                share(&rect, &kept_rect) >= 0.5 || share(&kept_rect, &rect) >= 0.5
            })
            .max_by(|a, b| {
                let iou_of = |d: &Detection| overlap(&rect, &[d.x, d.y, d.width, d.height]);
                iou_of(a).total_cmp(&iou_of(b))
            });
        match same {
            Some(kept) => kept.score = kept.score.max(item.score),
            None => tiled.push(item),
        }
    }
    tiled.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept_tiles: Vec<Detection> = Vec::with_capacity(tiled.len());
    for item in tiled {
        let rect = [item.x, item.y, item.width, item.height];
        let duplicate = kept_tiles.iter().any(|kept| {
            kept.class_id == item.class_id
                && overlap(&rect, &[kept.x, kept.y, kept.width, kept.height]) > iou_threshold
        });
        if !duplicate {
            kept_tiles.push(item);
        }
    }
    whole.extend(kept_tiles);
    whole.sort_by(|a, b| b.score.total_cmp(&a.score));
    whole
}

/// How much of rectangle `a` — left, top, width, height — lies in `b`.
fn share(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let area = a[2] * a[3];
    if area <= 0.0 {
        return 0.0;
    }
    intersection(a, b) / area
}

/// Intersection over union of two rectangles — left, top, width, height.
fn overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let inter = intersection(a, b);
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn intersection(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.0);
    let h = ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.0);
    w * h
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

    /// Three across and two down a wide picture, overlapping a quarter, on
    /// even pixels, covering it; a tall one is cut two across and three
    /// down, and one stored on its side but shown tall the same, as stored.
    #[test]
    fn tiles_cover_the_picture_as_it_is_shown() {
        use crate::orientation::Rotation;
        let tiles = Tiles::default();
        let wide = tiles.rectangles((1920, 1080), Orientation::UPRIGHT);
        assert_eq!(wide.len(), 6);
        // 1920 / (3 - 2 * 0.25) = 768; 1080 / (2 - 0.25) = 617, so 616.
        assert_eq!(wide[0], (0, 0, 768, 616));
        assert_eq!(wide[2], (1152, 0, 768, 616));
        assert_eq!(wide[5], (1152, 464, 768, 616));
        for (left, top, width, height) in &wide {
            assert!(left % 2 == 0 && top % 2 == 0 && width % 2 == 0 && height % 2 == 0);
            assert!(left + width <= 1920 && top + height <= 1080);
        }
        let tall = tiles.rectangles((1080, 1920), Orientation::UPRIGHT);
        assert_eq!((tall[0].2, tall[0].3), (616, 768), "two across, three down");
        // Stored wide, shown tall: two across and three down as shown are
        // three across and two down as stored.
        let turned = tiles.rectangles((1920, 1080), Orientation::rotated(Rotation::Clockwise90));
        assert_eq!(turned, wide);
        assert!(
            Tiles {
                across: 0,
                ..tiles.clone()
            }
            .check()
            .is_err()
        );
        assert!(
            Tiles {
                overlap: 0.95,
                ..tiles.clone()
            }
            .check()
            .is_err()
        );
        assert!(tiles.check().is_ok());
    }

    /// What a tile's look finds lands on the whole picture where the tile
    /// is: the tile's own top-left corner is its corner.
    #[test]
    fn a_tiles_box_lands_where_the_tile_is() {
        let tile = Letterbox::cropped(
            (1920, 1080),
            (1152, 464, 768, 616),
            (640, 640),
            Orientation::UPRIGHT,
        );
        // 768 by 616 fitted to 640: 640 by 513, 63 down.
        assert_eq!((tile.scaled, tile.offset), ((640, 513), (0, 63)));
        let found = tile.detection(
            ModelBox::new(0, 0.9, [0.0, 63.0, 320.0, 576.0]).with_landmarks(vec![(640.0, 576.0)]),
        );
        let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
        assert!(
            near(found.x, 1152.0 / 1920.0) && near(found.y, 464.0 / 1080.0),
            "{found:?}"
        );
        assert!(
            near(found.width, 384.0 / 1920.0) && near(found.height, 616.0 / 1080.0),
            "{found:?}"
        );
        let (x, y) = found.landmarks[0];
        assert!(
            near(x, 1.0) && near(y, 1.0),
            "the bottom-right corner: {found:?}"
        );
        assert_eq!(
            looks((1920, 1080), (640, 640), Orientation::UPRIGHT, None, None).len(),
            1
        );
        let all = looks(
            (1920, 1080),
            (640, 640),
            Orientation::UPRIGHT,
            Some(&Tiles::default()),
            None,
        );
        assert_eq!(all.len(), 7);
        assert_eq!(all[0].crop, (0, 0, 1920, 1080), "the whole picture first");
    }

    /// Looks in the tiles that hold the centre of a box the picture carries
    /// in.
    #[derive(Debug)]
    struct WhereFound;

    impl TileChooser for WhereFound {
        fn look_in(&self, picture: &TiledPicture<'_>, [x, y, w, h]: [f32; 4]) -> bool {
            assert!((picture.whole_scale - 640.0 / 1920.0).abs() < 1e-6);
            picture
                .metadata
                .and_then(|metadata| metadata.get::<crate::elements::Detections>())
                .is_some_and(|found| {
                    found.items.iter().any(|item| {
                        let (cx, cy) = (item.x + item.width / 2.0, item.y + item.height / 2.0);
                        (x..x + w).contains(&cx) && (y..y + h).contains(&cy)
                    })
                })
        }
    }

    /// A chooser is asked of each tile, from what the picture carries, and
    /// only the tiles it picks are looked in — the whole picture always.
    #[test]
    fn a_chooser_picks_the_tiles_from_what_the_picture_carries() {
        let tiles = Tiles {
            choose: Some(Arc::new(WhereFound)),
            ..Tiles::default()
        };
        let looked = |metadata: Option<&Metadata>| {
            looks(
                (1920, 1080),
                (640, 640),
                Orientation::UPRIGHT,
                Some(&tiles),
                metadata,
            )
        };
        assert_eq!(looked(None).len(), 1, "nothing carried, the whole alone");
        // Top left, and bottom right: one tile each, as 3 by 2 tiles of a
        // quarter's overlap meet nowhere near either.
        let found = crate::elements::Detections::new(
            "people",
            Arc::from([]),
            vec![
                Detection::new(0, 0.9, 0.02, 0.02, 0.04, 0.04),
                Detection::new(0, 0.9, 0.94, 0.94, 0.04, 0.04),
            ],
        );
        let metadata = Metadata::default().with(found);
        let all = looked(Some(&metadata));
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].crop, (0, 0, 1920, 1080), "the whole picture first");
        assert_eq!((all[1].crop.0, all[1].crop.1), (0, 0));
        assert_eq!(all[2].crop.0 + all[2].crop.2, 1920);
        assert_eq!(all[2].crop.1 + all[2].crop.3, 1080);
        assert_ne!(
            tiles,
            Tiles::default(),
            "a chooser is part of what tiles are"
        );
        assert_eq!(tiles, tiles.clone());
    }

    /// A face found whole and in a tile keeps the whole picture's box and
    /// the higher score — a tile can cut it in two; one found only in tiles
    /// is kept once; another class beside it is its own.
    #[test]
    fn tiles_add_what_the_whole_picture_missed() {
        let at = |class_id, score, x, y, w, h| Detection::new(class_id, score, x, y, w, h);
        let whole = vec![at(0, 0.6, 0.10, 0.10, 0.20, 0.20)];
        let left_tile = vec![
            // The same face, its right half cut off by the tile's edge.
            at(0, 0.9, 0.10, 0.10, 0.10, 0.20),
            // A small face the whole picture missed, and another class on it.
            at(0, 0.7, 0.50, 0.50, 0.02, 0.03),
            at(1, 0.5, 0.50, 0.50, 0.02, 0.03),
        ];
        let right_tile = vec![at(0, 0.8, 0.501, 0.50, 0.02, 0.03)];
        let merged = merge_looks(vec![whole, left_tile, right_tile], 0.45);
        let summary: Vec<_> = merged
            .iter()
            .map(|d| (d.class_id, d.score, d.width))
            .collect();
        assert_eq!(
            summary,
            vec![(0, 0.9, 0.20), (0, 0.8, 0.02), (1, 0.5, 0.02)]
        );
        assert_eq!(
            merge_looks(vec![vec![at(0, 0.5, 0.0, 0.0, 0.1, 0.1)]], 0.45).len(),
            1
        );
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
