//! Drawing what a detector found onto the picture it found it in, as
//! filters: each picture that carries [`Detections`] is handed on with a
//! box around each object, and with a label above it where a font is given
//! — and, as [`OverlayParts`] says, the zones and lines an
//! [`ObjectAnalytics`](crate::elements::ObjectAnalytics) counted from its
//! [`Analytics`] — or with a [`Redaction`], each box hidden by a mosaic, a
//! blur or a fill before anything is drawn over it.
//!
//! One element per place a picture lives, as the detectors are:
//! [`SwDetectionOverlay`] draws on pictures in system memory, and
//! `CudaDetectionOverlay` on CUDA pictures and `MetalDetectionOverlay` on
//! VideoToolbox ones without them leaving the GPU.
//! What they share is here: the options, where each box and label goes in a
//! picture of a given size, and the colours.
//!
//! Each draws on a copy, never on the picture it was handed: a picture is
//! shared, and a Tee before the overlay hands the same one to a branch that
//! must not see the boxes. A picture with nothing to draw is handed on as
//! it came. What it carries goes on with the drawn copy, so whatever comes
//! after can still read it.

mod sw_detection_overlay;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDetectionOverlay, CudaDetectionOverlayError};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal::{MetalDetectionOverlay, MetalDetectionOverlayError};
pub use sw_detection_overlay::{SwDetectionOverlay, SwDetectionOverlayError};

use std::num::NonZeroU32;

use thiserror::Error as ThisError;

use crate::color::Color;
use crate::elements::source::{TextMask, TextRasterError, rasterize_coverage};
use crate::elements::{Analytics, Detection, Detections};
use crate::orientation::Orientation;

/// What a detection overlay does with each picture: per class of what was
/// found, whether its box is drawn and how, whether it is hidden and how,
/// and from what score — and beside them, the font labels are drawn in and
/// the zones and lines of an
/// [`ObjectAnalytics`](crate::elements::ObjectAnalytics).
///
/// By default every detection's box is drawn, two pixels thick in its
/// class's colour, with no label.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DetectionOverlayOptions {
    /// Raw TrueType or OpenType font bytes, which every label is drawn in —
    /// boxes', zones' and lines' alike — read once for them all. This crate
    /// bundles no font of its own, as with a compositor's text layers. A
    /// [`BoxStyle::label`] with no font here is refused.
    pub font: Option<Vec<u8>>,
    /// The zones and lines an analytics counted, drawn where asked for.
    pub parts: OverlayParts,
    /// What is done with the detections of the classes named here, the
    /// first rule naming a class being the one it follows. A class named by
    /// two rules is refused.
    pub rules: Vec<ClassRule>,
    /// What is done with the detections of every class no rule names.
    pub others: Treatment,
}

/// What a detection overlay does with one class's detections.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassRule {
    /// The class.
    pub class: ClassId,
    /// What is done with its detections.
    pub treatment: Treatment,
}

impl ClassRule {
    /// `treatment` for `class` — a name, `"face"`, or a number, `0`.
    pub fn new(class: impl Into<ClassId>, treatment: Treatment) -> Self {
        Self {
            class: class.into(),
            treatment,
        }
    }
}

/// A class of a detector's model, by name or by number.
///
/// A name is what the model calls the class — the label the
/// [`Detections`] carry for it — so that a rule follows the class when
/// the model is changed for one numbering its classes otherwise; it
/// matches nothing where the model names no classes, which the overlay
/// warns of once. A number is the class's place in the model's output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClassId {
    /// The class the model calls this.
    Name(String),
    /// The class in this place of the model's output.
    Id(usize),
}

impl From<&str> for ClassId {
    fn from(name: &str) -> Self {
        Self::Name(name.to_owned())
    }
}

impl From<String> for ClassId {
    fn from(name: String) -> Self {
        Self::Name(name)
    }
}

impl From<usize> for ClassId {
    fn from(id: usize) -> Self {
        Self::Id(id)
    }
}

impl std::fmt::Display for ClassId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(name) => write!(f, "{name:?}"),
            Self::Id(id) => write!(f, "{id}"),
        }
    }
}

/// What a detection overlay does with a detection: hides it, draws its box
/// over that, both or neither — each only where the detection's score is
/// at least `min_score`.
#[derive(Debug, Clone, PartialEq)]
pub struct Treatment {
    /// How its box and label are drawn, or `None` to draw none.
    pub draw: Option<BoxStyle>,
    /// How it is hidden, before any box is drawn over it, or `None` to
    /// leave it as it is.
    pub hide: Option<Hiding>,
    /// Detections less confident than this are left alone. Hiding wants it
    /// low — a face half seen is still a face — and drawing often higher.
    pub min_score: f32,
}

impl Treatment {
    /// Its box drawn as `style` says, nothing hidden.
    pub fn boxes(style: BoxStyle) -> Self {
        Self {
            draw: Some(style),
            hide: None,
            min_score: 0.0,
        }
    }

    /// Hidden as `style` says, a tenth's margin, no box drawn.
    pub fn hidden(style: RedactStyle) -> Self {
        Self {
            draw: None,
            hide: Some(Hiding::new(style)),
            min_score: 0.0,
        }
    }

    /// Nothing done.
    pub fn none() -> Self {
        Self {
            draw: None,
            hide: None,
            min_score: 0.0,
        }
    }
}

impl Default for Treatment {
    /// Its box drawn as [`BoxStyle::default`] says.
    fn default() -> Self {
        Self::boxes(BoxStyle::default())
    }
}

/// How a box is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxStyle {
    /// How thick its lines are, in pixels of the picture drawn on. Rounded
    /// up to even on a picture whose colour is stored per 2x2 block, NV12
    /// or YUV 4:2:0, so that a line's colour does not bleed half a block
    /// outside it.
    pub line_width: u32,
    /// What colour it is.
    pub color: BoxColors,
    /// The label above it, or `None` for the box alone. Drawn in
    /// [`DetectionOverlayOptions::font`].
    pub label: Option<LabelStyle>,
}

impl Default for BoxStyle {
    /// Two pixels thick, in its class's colour, with no label.
    fn default() -> Self {
        Self {
            line_width: 2,
            color: BoxColors::ByClass,
            label: None,
        }
    }
}

/// How a detection is hidden: the box grown by `margin` and covered as
/// `style` says, on the copy of the picture the overlay draws on —
/// DeepStream's redaction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hiding {
    /// How it is covered.
    pub style: RedactStyle,
    /// How far the box is grown before it is covered, each way, as a
    /// fraction of its own width and height: 0.1 covers a tenth more on
    /// every side, which a box drawn a little tight — or a face moving
    /// between two detections — would otherwise leave showing. 0 or more.
    pub margin: f32,
    /// What of the grown box is covered: all of it, or the ellipse inside
    /// it, which hides a face as well with less of the picture around it.
    pub shape: HideShape,
}

impl Hiding {
    /// `style`, with a tenth's margin, the whole box.
    pub fn new(style: RedactStyle) -> Self {
        Self {
            style,
            margin: 0.1,
            shape: HideShape::Rectangle,
        }
    }
}

/// What of a box a [`Hiding`] covers.
///
/// An ellipse's edge is hard: a pixel — or, where the picture's colour is
/// stored per 2x2 block, a block — is covered where its centre is inside
/// the ellipse the box is drawn round, and left as it was where not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HideShape {
    /// The whole box.
    #[default]
    Rectangle,
    /// The ellipse inside the box, touching its four sides.
    Ellipse,
}

/// Whether the centre of sample `(x, y)` of a `width` by `height` plane is
/// inside the ellipse that touches its four sides — computed in this
/// order, each step rounded as `f32` is, as the CUDA kernel `cell_paint`
/// computes it, so that the two cover the same samples. A plane of colour
/// stored per 2x2 block, half the size, so asks of each block's centre.
pub(crate) fn inside_ellipse(x: u32, y: u32, width: u32, height: u32) -> bool {
    let dx = (x as f32 + 0.5) * 2.0 / width as f32 - 1.0;
    let dy = (y as f32 + 0.5) * 2.0 / height as f32 - 1.0;
    dx * dx + dy * dy <= 1.0
}

/// How a [`Hiding`] covers a box.
///
/// A mosaic's or a blur's cells are sized by the box, not fixed in pixels:
/// a face's shorter side is cut into `cells`, so that a face close to the
/// camera is hidden as well as one far off — a cell fixed in pixels leaves
/// the near face most of its features. `min_cell` keeps a small face's
/// cells from shrinking to a pixel or two, where a mosaic hides nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RedactStyle {
    /// Cells of one colour each, the mean of the pixels under it: the box's
    /// shorter side cut into `cells`, but no cell smaller than `min_cell`
    /// pixels, the longer side into as many cells as come out nearest to
    /// square. Fewer, larger cells hide more.
    Mosaic {
        /// Cells across the box's shorter side.
        cells: NonZeroU32,
        /// The smallest a cell may be, in pixels.
        min_cell: u32,
    },
    /// The same cells' means, each blended smoothly into its neighbours
    /// rather than drawn flat: a softer look that hides as much as the
    /// mosaic of the same cells — which an ordinary blur too light to hide
    /// a face does not, as it can be partly undone.
    Blur {
        /// Cells across the box's shorter side.
        cells: NonZeroU32,
        /// The smallest a cell may be, in pixels.
        min_cell: u32,
    },
    /// The box filled with one colour: nothing of what was under it is
    /// left.
    Fill(Color),
}

impl RedactStyle {
    /// Six cells across a box's shorter side, none smaller than eight
    /// pixels.
    pub fn mosaic() -> Self {
        Self::Mosaic {
            cells: NonZeroU32::new(6).expect("not zero"),
            min_cell: 8,
        }
    }

    /// [`Self::mosaic`]'s cells, blended.
    pub fn blur() -> Self {
        Self::Blur {
            cells: NonZeroU32::new(6).expect("not zero"),
            min_cell: 8,
        }
    }
}

/// Why a detection overlay cannot be made of the options it was given.
#[derive(Debug, Clone, PartialEq, ThisError)]
#[non_exhaustive]
pub enum DetectionOverlayOptionsError {
    /// A label asked for with no [`DetectionOverlayOptions::font`] to draw
    /// it in.
    #[error("a label is asked for with no font to draw it in")]
    NoFont,
    /// A label size that is not a positive number of pixels.
    #[error("label size {0} is not a positive number of pixels")]
    LabelSize(f32),
    /// A class two rules name.
    #[error("class {0} is named by more than one rule")]
    DuplicateClass(ClassId),
    /// A hiding margin that is not a number of 0 or more.
    #[error("hiding margin {0} is not a number of 0 or more")]
    Margin(f32),
    /// A minimum score that is not a number.
    #[error("min_score is not a number")]
    MinScore,
}

/// The zones and lines of an
/// [`ObjectAnalytics`](crate::elements::ObjectAnalytics) an overlay draws,
/// from the picture's [`Analytics`], each labelled where
/// [`DetectionOverlayOptions::font`] gives a font.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlayParts {
    /// Each zone: its outline in amber — red while it is crowded —
    /// labelled with its name and how many objects are in it, `door 2`.
    pub zones: bool,
    /// Each line: drawn across the picture in cyan, labelled with its name
    /// and its crossings forward and backward, `gate 12 / 3`.
    pub lines: bool,
    /// How thick the zones' outlines and the lines are, in pixels, rounded
    /// up to even as a box's are.
    pub line_width: u32,
    /// Their labels' pixel height.
    pub label_size: f32,
}

impl Default for OverlayParts {
    /// Neither drawn; two pixels thick and labelled 16 high where they are.
    fn default() -> Self {
        Self {
            zones: false,
            lines: false,
            line_width: 2,
            label_size: 16.0,
        }
    }
}

impl OverlayParts {
    /// Zones and lines alike.
    pub fn all() -> Self {
        Self {
            zones: true,
            lines: true,
            ..Self::default()
        }
    }
}

/// What colour a box is drawn in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoxColors {
    /// One of a dozen distinct colours, chosen by the detection's class, so
    /// that every object of a class is the same colour in every picture.
    #[default]
    ByClass,
    /// The same colour for every box.
    One(Color),
    /// One of the same dozen, chosen by the number an
    /// [`ObjectTracker`](crate::elements::ObjectTracker) gave the object, so
    /// that one object is one colour throughout; by class where it has
    /// none.
    ByTrack,
}

/// How the label above a box is drawn, and what it says — of the class
/// name, the number a tracker gave the object, the score and what
/// classifiers said of it, each as asked for, `car #7 0.87 | minivan` with
/// all four — on a band of the box's colour, in
/// [`DetectionOverlayOptions::font`]. A box whose label would say nothing
/// has none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelStyle {
    /// The text's pixel height, in the picture drawn on.
    pub size: f32,
    /// Whether the class's name is said — or `class` and its number where
    /// the model names none.
    pub class: bool,
    /// Whether the number an [`ObjectTracker`](crate::elements::ObjectTracker)
    /// gave the object follows, as in `person #7`, where it has one.
    pub track_id: bool,
    /// Whether the score follows, as in `person 0.87`.
    pub score: bool,
    /// Whether what classifiers said of the object follows, as in
    /// `car | minivan`.
    pub classes: bool,
}

impl LabelStyle {
    /// A label `size` pixels high, saying all it can.
    pub fn new(size: f32) -> Self {
        Self {
            size,
            class: true,
            track_id: true,
            score: true,
            classes: true,
        }
    }
}

impl Default for LabelStyle {
    /// 16 pixels high, saying all it can.
    fn default() -> Self {
        Self::new(16.0)
    }
}

impl DetectionOverlayOptions {
    /// Every treatment these give, the rules' and `others`.
    fn treatments(&self) -> impl Iterator<Item = &Treatment> {
        self.rules
            .iter()
            .map(|rule| &rule.treatment)
            .chain(std::iter::once(&self.others))
    }

    /// Why an overlay cannot be made of these, if it cannot.
    pub(crate) fn check(&self) -> Result<(), DetectionOverlayOptionsError> {
        let mut named = std::collections::HashSet::new();
        for rule in &self.rules {
            if !named.insert(&rule.class) {
                return Err(DetectionOverlayOptionsError::DuplicateClass(
                    rule.class.clone(),
                ));
            }
        }
        let sizes = self
            .treatments()
            .filter_map(|treatment| treatment.draw.as_ref()?.label.map(|label| label.size));
        for size in sizes {
            if self.font.is_none() {
                return Err(DetectionOverlayOptionsError::NoFont);
            }
            if !size.is_finite() || size <= 0.0 {
                return Err(DetectionOverlayOptionsError::LabelSize(size));
            }
        }
        let size = self.parts.label_size;
        if (self.parts.zones || self.parts.lines) && (!size.is_finite() || size <= 0.0) {
            return Err(DetectionOverlayOptionsError::LabelSize(size));
        }
        for treatment in self.treatments() {
            if !treatment.min_score.is_finite() {
                return Err(DetectionOverlayOptionsError::MinScore);
            }
            if let Some(hiding) = treatment.hide
                && (!hiding.margin.is_finite() || hiding.margin < 0.0)
            {
                return Err(DetectionOverlayOptionsError::Margin(hiding.margin));
            }
        }
        Ok(())
    }

    /// Whether any treatment hides.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn hides_any(&self) -> bool {
        self.treatments().any(|treatment| treatment.hide.is_some())
    }

    /// Whether any treatment hides by painting cells: a mosaic or a blur,
    /// or a fill cut to an ellipse, which is painted as one cell its
    /// colour.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuts_cells(&self) -> bool {
        self.treatments().any(|treatment| {
            treatment.hide.is_some_and(|hiding| {
                hiding.shape == HideShape::Ellipse
                    || matches!(
                        hiding.style,
                        RedactStyle::Mosaic { .. } | RedactStyle::Blur { .. }
                    )
            })
        })
    }

    /// Whether any treatment hides an ellipse.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn hides_ellipses(&self) -> bool {
        self.treatments().any(|treatment| {
            treatment
                .hide
                .is_some_and(|hiding| hiding.shape == HideShape::Ellipse)
        })
    }

    /// Whether a rule names a class where `detections` carry no names, so
    /// that it matches nothing: worth a warning, once.
    pub(crate) fn names_unmatched(&self, detections: &Detections) -> bool {
        detections.labels.is_empty()
            && self
                .rules
                .iter()
                .any(|rule| matches!(rule.class, ClassId::Name(_)))
    }

    /// What is done with `detection`, one of `detections`: the first rule
    /// naming its class, or `others` — and `None` where its score is below
    /// that treatment's.
    pub(crate) fn treatment(
        &self,
        detections: &Detections,
        detection: &Detection,
    ) -> Option<&Treatment> {
        let named = detections.label(detection);
        let treatment = self
            .rules
            .iter()
            .find(|rule| match &rule.class {
                ClassId::Id(id) => *id == detection.class_id,
                ClassId::Name(name) => named == Some(name.as_str()),
            })
            .map_or(&self.others, |rule| &rule.treatment);
        (detection.score >= treatment.min_score).then_some(treatment)
    }
}

/// A rectangle of a picture, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Rect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Rect {
    pub(crate) fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// The size of a picture being drawn on, and the block its colour is
/// stored in: 2 for NV12 and YUV 4:2:0, where every rectangle drawn starts
/// and ends on an even pixel, 1 for packed RGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Canvas {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) block: u32,
}

impl Canvas {
    fn down(&self, value: u32) -> u32 {
        value - value % self.block
    }

    fn up(&self, value: u32) -> u32 {
        value.div_ceil(self.block) * self.block
    }

    /// The width and height that whole blocks cover.
    fn limit(&self) -> (u32, u32) {
        (self.down(self.width), self.down(self.height))
    }

    /// Where `detection` is in this picture: its box scaled to the picture,
    /// grown out to whole blocks and cut to the picture. `None` where none
    /// of it is in the picture.
    pub(crate) fn place(&self, detection: &Detection) -> Option<Rect> {
        let (limit_x, limit_y) = self.limit();
        // A thousandth of a pixel either way is arithmetic, not the box: a
        // fraction like 0.3 lands a hair past 30 pixels, and rounding that
        // outward would grow every such box by one.
        let edge = |fraction: f32, size: u32, limit: u32, round: fn(f32) -> f32, nudge: f32| {
            (round(fraction * size as f32 + nudge).max(0.0) as u32).min(limit)
        };
        let left = self.down(edge(detection.x, self.width, limit_x, f32::floor, 1e-3));
        let top = self.down(edge(detection.y, self.height, limit_y, f32::floor, 1e-3));
        let right = self
            .up(edge(
                detection.x + detection.width,
                self.width,
                limit_x,
                f32::ceil,
                -1e-3,
            ))
            .min(limit_x);
        let bottom = self
            .up(edge(
                detection.y + detection.height,
                self.height,
                limit_y,
                f32::ceil,
                -1e-3,
            ))
            .min(limit_y);
        let placed = Rect {
            x: left,
            y: top,
            width: right.saturating_sub(left),
            height: bottom.saturating_sub(top),
        };
        (!placed.is_empty()).then_some(placed)
    }

    /// The rectangle `points` — in pixels — span, grown by `reach` each
    /// way and out to whole blocks, and cut to the picture. `None` where
    /// none of it is in the picture.
    fn bounds(&self, points: &[(f32, f32)], reach: f32) -> Option<Rect> {
        let (limit_x, limit_y) = self.limit();
        let fold = |pick: fn(&(f32, f32)) -> f32, start: f32, keep: fn(f32, f32) -> f32| {
            points.iter().map(pick).fold(start, keep)
        };
        let (left, right) = (
            fold(|p| p.0, f32::INFINITY, f32::min) - reach,
            fold(|p| p.0, f32::NEG_INFINITY, f32::max) + reach,
        );
        let (top, bottom) = (
            fold(|p| p.1, f32::INFINITY, f32::min) - reach,
            fold(|p| p.1, f32::NEG_INFINITY, f32::max) + reach,
        );
        let clamp = |value: f32, limit: u32| (value.max(0.0) as u32).min(limit);
        let x = self.down(clamp(left.floor(), limit_x));
        let y = self.down(clamp(top.floor(), limit_y));
        let placed = Rect {
            x,
            y,
            width: self
                .up(clamp(right.ceil(), limit_x))
                .min(limit_x)
                .saturating_sub(x),
            height: self
                .up(clamp(bottom.ceil(), limit_y))
                .min(limit_y)
                .saturating_sub(y),
        };
        (!placed.is_empty()).then_some(placed)
    }

    /// The four lines of a box `line_width` thick, inside it. A box thinner
    /// than two lines is filled; the lines that would be empty are.
    pub(crate) fn edges(&self, placed: Rect, line_width: u32) -> [Rect; 4] {
        let line = self.up(line_width.max(1));
        let across = line.min(placed.height);
        let down = line.min(placed.width);
        [
            Rect {
                height: across,
                ..placed
            },
            Rect {
                y: placed.y + placed.height - across,
                height: across,
                ..placed
            },
            Rect {
                width: down,
                ..placed
            },
            Rect {
                x: placed.x + placed.width - down,
                width: down,
                ..placed
            },
        ]
    }

    /// Where a label `text_width` by `text_height` goes for a box placed at
    /// `placed`: the band behind it, and the text's origin inside the band.
    /// Above the box where there is room, else just inside its top; moved
    /// left to stay in the picture, and cut to it where it is wider.
    pub(crate) fn label(&self, placed: Rect, text_width: u32, text_height: u32) -> (Rect, Rect) {
        let (limit_x, limit_y) = self.limit();
        let pad = self.up(2);
        let width = self.up(text_width + 2 * pad).min(limit_x);
        let height = self.up(text_height + 2 * pad).min(limit_y);
        let x = placed.x.min(limit_x - width);
        let y = if placed.y >= height {
            placed.y - height
        } else {
            placed.y.min(limit_y - height)
        };
        let band = Rect {
            x,
            y,
            width,
            height,
        };
        // The text within the band, cut to it, on whole blocks so that a
        // plane drawn at half resolution starts where the text does.
        let text = Rect {
            x: x + pad,
            y: y + pad,
            width: self.down(text_width.min(width.saturating_sub(2 * pad))),
            height: self.down(text_height.min(height.saturating_sub(2 * pad))),
        };
        (band, text)
    }
}

/// A dozen colours far enough apart to tell boxes apart by, picked by class.
const PALETTE: [Color; 12] = [
    Color::new(0, 200, 83),
    Color::new(255, 82, 82),
    Color::new(41, 121, 255),
    Color::new(255, 196, 0),
    Color::new(213, 0, 249),
    Color::new(0, 229, 255),
    Color::new(255, 109, 0),
    Color::new(118, 255, 3),
    Color::new(255, 64, 129),
    Color::new(101, 31, 255),
    Color::new(29, 233, 182),
    Color::new(198, 255, 0),
];

/// The colour of `detection`'s box.
pub(crate) fn box_color(colors: BoxColors, detection: &Detection) -> Color {
    let by_class = PALETTE[detection.class_id % PALETTE.len()];
    match colors {
        BoxColors::ByClass => by_class,
        BoxColors::One(color) => color,
        BoxColors::ByTrack => detection
            .track_id
            .map_or(by_class, |id| PALETTE[(id % PALETTE.len() as u64) as usize]),
    }
}

/// Black or white, whichever reads better on `background`.
pub(crate) fn text_color(background: Color) -> Color {
    let luma = 0.2126 * f32::from(background.red)
        + 0.7152 * f32::from(background.green)
        + 0.0722 * f32::from(background.blue);
    if luma > 140.0 {
        Color::BLACK
    } else {
        Color::WHITE
    }
}

/// What a label says, of what `parts` asks for: the class's name, or
/// `class` and its number where the model names none; then the object's
/// number where a tracker gave it one, the score, and what classifiers said
/// of it — `car #7 0.87 | minivan` with all four. Empty where it says
/// nothing.
pub(crate) fn label_text(
    detections: &Detections,
    detection: &Detection,
    parts: &LabelStyle,
) -> String {
    let mut words = Vec::new();
    if parts.class {
        words.push(match detections.label(detection) {
            Some(name) => name.to_owned(),
            None => format!("class {}", detection.class_id),
        });
    }
    if parts.track_id
        && let Some(id) = detection.track_id
    {
        words.push(format!("#{id}"));
    }
    if parts.score {
        words.push(format!("{:.2}", detection.score));
    }
    let mut text = words.join(" ");
    if parts.classes {
        for class in &detection.classes {
            if !text.is_empty() {
                text.push(' ');
            }
            match class.label() {
                Some(label) => text.push_str(&format!("| {label}")),
                None => text.push_str(&format!("| class {}", class.class_id)),
            }
        }
    }
    text
}

/// BT.709 limited-range Y'CbCr for an sRGB colour — what a YUV picture
/// stores for it.
pub(crate) fn bt709_limited(color: Color) -> (u8, u8, u8) {
    let (r, g, b) = (
        f32::from(color.red) / 255.0,
        f32::from(color.green) / 255.0,
        f32::from(color.blue) / 255.0,
    );
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let cb = (b - y) / 1.8556;
    let cr = (r - y) / 1.5748;
    let byte = |value: f32| value.round().clamp(0.0, 255.0) as u8;
    (
        byte(16.0 + 219.0 * y),
        byte(128.0 + 224.0 * cb),
        byte(128.0 + 224.0 * cr),
    )
}

/// How many rasterized labels an overlay keeps before forgetting them all:
/// a label with a score is a new string most pictures.
pub(crate) const LABEL_CACHE: usize = 256;

/// A zone's outline, and its label's band.
const ZONE: Color = Color::new(255, 196, 0);
/// A zone's while it is crowded.
const CROWDED: Color = Color::new(255, 64, 64);
/// A line, and its label's band.
const LINE: Color = Color::new(0, 229, 255);

/// How long a piece of a slanting line is at most, in pixels: each is
/// painted through a coverage mask of the rectangle around it, so a line
/// across the picture costs its length times this, not its rectangle.
const PIECE: f32 = 64.0;

/// A coverage mask an overlay paints through, and what it is cached by.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MaskKey {
    /// A label's text, in the overlay's font, `size` — an `f32`'s bits —
    /// pixels high.
    Text { text: String, size: u32 },
    /// A piece of a line `width` pixels thick, from `from` to `to` — in
    /// sixteenths of a pixel of the picture — through exactly `rect`, which
    /// the mask is the size of.
    Stroke {
        rect: Rect,
        from: (i32, i32),
        to: (i32, i32),
        width: u32,
    },
    /// The top-left `width` by `height` of `key`'s mask, made for the
    /// picture as shown, turned back the way the picture is stored — what a
    /// label or a line laid out on a turned picture is painted through.
    Turned {
        key: Box<MaskKey>,
        width: u32,
        height: u32,
        orientation: Orientation,
    },
}

/// What a mask is made of: a label in the overlay's font, at the size its
/// key says, where it has one — `None` draws no label — and a line's
/// piece from its key alone.
pub(crate) fn rasterize(
    key: &MaskKey,
    font: Option<&ab_glyph::FontArc>,
) -> Result<Option<TextMask>, TextRasterError> {
    match key {
        MaskKey::Text { text, size } => match font {
            Some(font) => rasterize_coverage(font, f32::from_bits(*size), text),
            None => Ok(None),
        },
        &MaskKey::Stroke {
            rect,
            from,
            to,
            width,
        } => Ok(Some(stroke_mask(rect, from, to, width))),
        MaskKey::Turned {
            key,
            width,
            height,
            orientation,
        } => Ok(rasterize(key, font)?.map(|shown| turned(&shown, *width, *height, *orientation))),
    }
}

/// The top-left `width` by `height` of `shown`, a mask made for the
/// picture as shown, as it lies in the picture as stored.
fn turned(shown: &TextMask, width: u32, height: u32, orientation: Orientation) -> TextMask {
    let (width, height) = (width.min(shown.width), height.min(shown.height));
    let stored = orientation.display_size(width, height);
    let [xx, xy, x0, yx, yy, y0] = orientation.sampling(stored.0, stored.1);
    let mut coverage = vec![0; (stored.0 * stored.1) as usize];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let (sx, sy) = (xx * x + xy * y + x0, yx * x + yy * y + y0);
            coverage[(sy as u32 * stored.0 + sx as u32) as usize] =
                shown.coverage[(y as u32 * shown.width + x as u32) as usize];
        }
    }
    TextMask {
        width: stored.0,
        height: stored.1,
        coverage,
    }
}

/// How much of each pixel of `rect` a line `width` thick from `from` to
/// `to` — in sixteenths of a pixel — covers, with round ends and a pixel's
/// fade at its edges.
fn stroke_mask(rect: Rect, from: (i32, i32), to: (i32, i32), width: u32) -> TextMask {
    let point = |(x, y): (i32, i32)| (x as f32 / 16.0, y as f32 / 16.0);
    let (a, b) = (point(from), point(to));
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let length = dx * dx + dy * dy;
    let half = width as f32 / 2.0;
    let mut coverage = Vec::with_capacity((rect.width * rect.height) as usize);
    for y in 0..rect.height {
        for x in 0..rect.width {
            let p = ((rect.x + x) as f32 + 0.5, (rect.y + y) as f32 + 0.5);
            let t = if length > 0.0 {
                (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let distance = (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy);
            coverage.push(((half + 0.5 - distance).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    TextMask {
        width: rect.width,
        height: rect.height,
        coverage,
    }
}

/// One thing an overlay paints, in order: a rectangle of the picture in a
/// colour, solid or through a mask.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mark {
    pub(crate) rect: Rect,
    pub(crate) color: Color,
    /// The mask, of at least the rectangle's size and read from its
    /// top-left, or `None` to paint the rectangle solid.
    pub(crate) mask: Option<MaskKey>,
}

/// One box to hide, in pixels of the picture: all of `rect`, or with
/// `ellipse` the samples [`inside_ellipse`] of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Hide {
    /// `rect` cut into `cells` — across, down — each painted the mean of
    /// the pixels under it, or with `smooth` blended into its neighbours.
    Cells {
        rect: Rect,
        cells: (u32, u32),
        smooth: bool,
        ellipse: bool,
    },
    /// `rect` filled with one colour.
    Fill {
        rect: Rect,
        color: Color,
        ellipse: bool,
    },
}

/// What `options` hide of `detections` on a picture `canvas` describes:
/// each box its treatment hides grown by the margin, cut to the picture
/// and out to whole blocks, and for a mosaic or a blur the cells it is cut
/// into.
pub(crate) fn hides(
    canvas: Canvas,
    options: &DetectionOverlayOptions,
    detections: &Detections,
) -> Vec<Hide> {
    detections
        .items
        .iter()
        .filter_map(|detection| {
            let hiding = options.treatment(detections, detection)?.hide?;
            let margin = hiding.margin;
            let grown = Detection::new(
                detection.class_id,
                detection.score,
                detection.x - detection.width * margin,
                detection.y - detection.height * margin,
                detection.width * (1.0 + 2.0 * margin),
                detection.height * (1.0 + 2.0 * margin),
            );
            let rect = canvas.place(&grown)?;
            let ellipse = hiding.shape == HideShape::Ellipse;
            let (cells, min_cell, smooth) = match hiding.style {
                RedactStyle::Fill(color) => {
                    return Some(Hide::Fill {
                        rect,
                        color,
                        ellipse,
                    });
                }
                RedactStyle::Mosaic { cells, min_cell } => (cells, min_cell, false),
                RedactStyle::Blur { cells, min_cell } => (cells, min_cell, true),
            };
            let shorter = rect.width.min(rect.height);
            let cell = shorter.div_ceil(cells.get()).max(min_cell).max(1);
            // As many cells along each side as come nearest to `cell`, each
            // at least one sample of the picture's colour planes.
            let along =
                |side: u32| ((side + cell / 2) / cell).clamp(1, (side / canvas.block).max(1));
            Some(Hide::Cells {
                rect,
                cells: (along(rect.width), along(rect.height)),
                smooth,
                ellipse,
            })
        })
        .collect()
}

/// The `index`th of `cells` stretches a side `length` long is
/// `[edge(index), edge(index + 1))`: as near equal as whole samples allow.
pub(crate) fn cell_edge(index: u32, cells: u32, length: u32) -> u32 {
    (u64::from(index) * u64::from(length) / u64::from(cells)) as u32
}

impl DetectionOverlayOptions {
    /// Whether a picture carrying `detections` and `analytics` has anything
    /// to draw or hide: a picture with nothing is handed on as it came.
    pub(crate) fn draws(
        &self,
        detections: Option<&Detections>,
        analytics: Option<&Analytics>,
    ) -> bool {
        detections.is_some_and(|detections| {
            detections.items.iter().any(|detection| {
                self.treatment(detections, detection)
                    .is_some_and(|treatment| treatment.draw.is_some() || treatment.hide.is_some())
            })
        }) || analytics.is_some_and(|analytics| {
            (self.parts.zones && !analytics.zones.is_empty())
                || (self.parts.lines && !analytics.lines.is_empty())
        })
    }
}

/// What to paint for `detections` and `analytics` on a picture `canvas`
/// describes, in order: the zones and lines, then each box its treatment
/// draws — the most confident last, so that it is drawn over the rest —
/// and its label's band and text. `mask` makes — or finds — the mask a key
/// names, and says its size, or `None` where it has nothing to draw.
pub(crate) fn marks(
    canvas: Canvas,
    options: &DetectionOverlayOptions,
    detections: Option<&Detections>,
    analytics: Option<&Analytics>,
    mask: &mut dyn FnMut(&MaskKey) -> Option<(u32, u32)>,
) -> Vec<Mark> {
    let mut marks = Vec::new();
    let parts = options.parts;
    if let Some(analytics) = analytics {
        let pixels = |(x, y): (f32, f32)| (x * canvas.width as f32, y * canvas.height as f32);
        let zones = analytics.zones.iter().filter(|_| parts.zones);
        for zone in zones {
            let corners: Vec<(f32, f32)> = zone.corners.iter().copied().map(pixels).collect();
            let color = if zone.crowded { CROWDED } else { ZONE };
            for (index, &from) in corners.iter().enumerate() {
                let to = corners[(index + 1) % corners.len()];
                stroke(canvas, parts.line_width, from, to, color, &mut marks, mask);
            }
            if let Some(anchor) = canvas.bounds(&corners, 0.0) {
                let text = format!("{} {}", zone.name, zone.objects.len());
                let label = (text, parts.label_size);
                labelled(canvas, anchor, label, color, &mut marks, mask);
            }
        }
        let lines = analytics.lines.iter().filter(|_| parts.lines);
        for line in lines {
            let (from, to) = (pixels(line.start), pixels(line.end));
            stroke(canvas, parts.line_width, from, to, LINE, &mut marks, mask);
            if let Some(anchor) = canvas.bounds(&[from, to], 0.0) {
                let text = format!("{} {} / {}", line.name, line.forward, line.backward);
                let label = (text, parts.label_size);
                labelled(canvas, anchor, label, LINE, &mut marks, mask);
            }
        }
    }
    let Some(detections) = detections else {
        return marks;
    };
    let mut drawn: Vec<(&Detection, &BoxStyle)> = detections
        .items
        .iter()
        .filter_map(|detection| {
            let style = options.treatment(detections, detection)?.draw.as_ref()?;
            Some((detection, style))
        })
        .collect();
    drawn.sort_by(|a, b| a.0.score.total_cmp(&b.0.score));
    for (detection, style) in drawn {
        let Some(placed) = canvas.place(detection) else {
            continue;
        };
        let color = box_color(style.color, detection);
        marks.extend(
            canvas
                .edges(placed, style.line_width)
                .into_iter()
                .filter(|edge| !edge.is_empty())
                .map(|rect| Mark {
                    rect,
                    color,
                    mask: None,
                }),
        );
        if let Some(label) = &style.label {
            let text = label_text(detections, detection, label);
            if !text.is_empty() {
                labelled(canvas, placed, (text, label.size), color, &mut marks, mask);
            }
        }
    }
    marks
}

/// [`marks`] for a picture `canvas` describes as stored and shown turned
/// as `orientation` says: laid out on the picture as shown — so that a
/// label is the right way up and above its box as the picture is seen, a
/// line's label likewise — and each mark then put where it lies in the
/// picture as stored, its mask turned with it. The boxes, zones and lines
/// are all of the picture as stored, as everything a pipeline says of a
/// picture is.
pub(crate) fn marks_turned(
    canvas: Canvas,
    orientation: Orientation,
    options: &DetectionOverlayOptions,
    detections: Option<&Detections>,
    analytics: Option<&Analytics>,
    mask: &mut dyn FnMut(&MaskKey) -> Option<(u32, u32)>,
) -> Vec<Mark> {
    if orientation.is_upright() {
        return marks(canvas, options, detections, analytics, mask);
    }
    let (width, height) = orientation.display_size(canvas.width, canvas.height);
    let shown = Canvas {
        width,
        height,
        block: canvas.block,
    };
    let detections = detections.map(|detections| {
        let mut turned = detections.clone();
        for item in &mut turned.items {
            [item.x, item.y, item.width, item.height] =
                orientation.to_display([item.x, item.y, item.width, item.height]);
        }
        turned
    });
    let point = |(x, y): (f32, f32)| {
        let [x, y, ..] = orientation.to_display([x, y, 0.0, 0.0]);
        (x, y)
    };
    let analytics = analytics.map(|analytics| {
        let mut turned = analytics.clone();
        for zone in &mut turned.zones {
            zone.corners = zone.corners.iter().copied().map(point).collect();
        }
        for line in &mut turned.lines {
            (line.start, line.end) = (point(line.start), point(line.end));
        }
        turned
    });
    let laid = marks(
        shown,
        options,
        detections.as_ref(),
        analytics.as_ref(),
        mask,
    );
    // Where each shown pixel is stored: a rectangle's two far corners
    // there, and the rectangle between them.
    let [xx, xy, x0, yx, yy, y0] = orientation.sampling(canvas.width, canvas.height);
    let stored = |x: u32, y: u32| {
        let (x, y) = (x as i32, y as i32);
        ((xx * x + xy * y + x0) as u32, (yx * x + yy * y + y0) as u32)
    };
    laid.into_iter()
        .filter_map(|mark| {
            let rect = mark.rect;
            let a = stored(rect.x, rect.y);
            let b = stored(rect.x + rect.width - 1, rect.y + rect.height - 1);
            let placed = Rect {
                x: a.0.min(b.0),
                y: a.1.min(b.1),
                width: a.0.abs_diff(b.0) + 1,
                height: a.1.abs_diff(b.1) + 1,
            };
            let mask_key = match mark.mask {
                None => None,
                Some(key) => {
                    let key = MaskKey::Turned {
                        key: Box::new(key),
                        width: rect.width,
                        height: rect.height,
                        orientation,
                    };
                    // Made now, as every mask a mark names is.
                    mask(&key)?;
                    Some(key)
                }
            };
            Some(Mark {
                rect: placed,
                mask: mask_key,
                ..mark
            })
        })
        .collect()
}

/// A line `line_width` thick from `from` to `to`, in pixels, as pieces each
/// painted through a mask of the rectangle around it.
fn stroke(
    canvas: Canvas,
    line_width: u32,
    from: (f32, f32),
    to: (f32, f32),
    color: Color,
    marks: &mut Vec<Mark>,
    mask: &mut dyn FnMut(&MaskKey) -> Option<(u32, u32)>,
) {
    let width = canvas.up(line_width.max(1));
    let length = (to.0 - from.0).hypot(to.1 - from.1);
    let pieces = (length / PIECE).ceil().max(1.0) as usize;
    let at = |t: f32| (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
    let fixed = |(x, y): (f32, f32)| ((x * 16.0).round() as i32, (y * 16.0).round() as i32);
    for piece in 0..pieces {
        let (a, b) = (
            at(piece as f32 / pieces as f32),
            at((piece + 1) as f32 / pieces as f32),
        );
        let Some(rect) = canvas.bounds(&[a, b], width as f32 / 2.0 + 1.0) else {
            continue;
        };
        let key = MaskKey::Stroke {
            rect,
            from: fixed(a),
            to: fixed(b),
            width,
        };
        // Cut to the mask as it was kept: one stored on whole blocks may
        // have lost an odd last row or column of its fade.
        let Some((width, height)) = mask(&key) else {
            continue;
        };
        let rect = Rect {
            width: rect.width.min(width),
            height: rect.height.min(height),
            ..rect
        };
        if !rect.is_empty() {
            marks.push(Mark {
                rect,
                color,
                mask: Some(key),
            });
        }
    }
}

/// The label `text`, `size` pixels high, for something placed at `anchor`:
/// its band in `color`, and its text in black or white on it.
fn labelled(
    canvas: Canvas,
    anchor: Rect,
    (text, size): (String, f32),
    color: Color,
    marks: &mut Vec<Mark>,
    mask: &mut dyn FnMut(&MaskKey) -> Option<(u32, u32)>,
) {
    let key = MaskKey::Text {
        text,
        size: size.to_bits(),
    };
    let Some((width, height)) = mask(&key) else {
        return;
    };
    let (band, area) = canvas.label(anchor, width, height);
    if band.is_empty() {
        return;
    }
    marks.push(Mark {
        rect: band,
        color,
        mask: None,
    });
    if !area.is_empty() {
        marks.push(Mark {
            rect: area,
            color: text_color(color),
            mask: Some(key),
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A font this machine has, to draw labels in, or `None` — as the
    /// compositors' text tests find one.
    pub(super) fn system_font() -> Option<Vec<u8>> {
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        .iter()
        .find_map(|path| std::fs::read(path).ok())
    }

    /// `frame`, an NV12 or BGRA picture in system memory as shown, stored
    /// as `orientation` turns it and carrying the display matrix that says
    /// so — or, `back`, a stored one as it is shown, carrying none.
    pub(super) fn turn(
        frame: &ffmpeg_next::frame::Video,
        orientation: Orientation,
        back: bool,
    ) -> ffmpeg_next::frame::Video {
        use ffmpeg_next::{ffi, format::Pixel};
        let (width, height) = orientation.display_size(frame.width(), frame.height());
        let mut out = ffmpeg_next::frame::Video::new(frame.format(), width, height);
        let stored = if back {
            (frame.width(), frame.height())
        } else {
            (width, height)
        };
        let [xx, xy, x0, yx, yy, y0] = orientation.sampling(stored.0, stored.1);
        let at = |x: u32, y: u32| {
            let (x, y) = (x as i32, y as i32);
            (
                (xx * x + xy * y + x0) as usize,
                (yx * x + yy * y + y0) as usize,
            )
        };
        // Each shown pixel (x, y) and where it is stored; a 2x2 block's
        // chroma goes where its top-left pixel does, in the block that is.
        let shown = if back {
            (width, height)
        } else {
            (frame.width(), frame.height())
        };
        let nv12 = frame.format() == Pixel::NV12;
        let bytes = if nv12 { 1 } else { 4 };
        for y in 0..shown.1 {
            for x in 0..shown.0 {
                let (sx, sy) = at(x, y);
                let (from, to) = if back {
                    ((sx, sy), (x as usize, y as usize))
                } else {
                    ((x as usize, y as usize), (sx, sy))
                };
                let (fs, ts) = (frame.stride(0), out.stride(0));
                for byte in 0..bytes {
                    out.data_mut(0)[to.1 * ts + to.0 * bytes + byte] =
                        frame.data(0)[from.1 * fs + from.0 * bytes + byte];
                }
                if nv12 && x % 2 == 0 && y % 2 == 0 {
                    let (fs, ts) = (frame.stride(1), out.stride(1));
                    for byte in 0..2 {
                        out.data_mut(1)[to.1 / 2 * ts + to.0 / 2 * 2 + byte] =
                            frame.data(1)[from.1 / 2 * fs + from.0 / 2 * 2 + byte];
                    }
                }
            }
        }
        out.set_pts(frame.pts());
        if !back && !orientation.is_upright() {
            let matrix = orientation.matrix();
            // SAFETY: `out` is a live frame; FFmpeg allocates the 36 bytes
            // asked for, which the matrix fills.
            unsafe {
                let data = ffi::av_frame_new_side_data(
                    out.as_mut_ptr(),
                    ffi::AVFrameSideDataType::AV_FRAME_DATA_DISPLAYMATRIX,
                    size_of_val(&matrix),
                );
                assert!(!data.is_null());
                std::ptr::copy_nonoverlapping(
                    matrix.as_ptr().cast::<u8>(),
                    (*data).data,
                    size_of_val(&matrix),
                );
            }
        }
        out
    }

    /// Every way a picture can be turned.
    pub(super) fn every_orientation() -> Vec<Orientation> {
        use crate::orientation::Rotation;
        [
            Rotation::None,
            Rotation::Clockwise90,
            Rotation::Half,
            Rotation::Clockwise270,
        ]
        .into_iter()
        .flat_map(|rotation| [false, true].map(|mirrored| Orientation { rotation, mirrored }))
        .collect()
    }

    fn detection(x: f32, y: f32, width: f32, height: f32) -> Detection {
        Detection::new(0, 0.9, x, y, width, height)
    }

    const RGB: Canvas = Canvas {
        width: 100,
        height: 50,
        block: 1,
    };
    const NV12: Canvas = Canvas {
        width: 100,
        height: 50,
        block: 2,
    };

    #[test]
    fn a_box_is_scaled_to_the_picture_and_grown_out_to_its_pixels() {
        let placed = RGB.place(&detection(0.105, 0.21, 0.5, 0.5)).unwrap();
        assert_eq!(
            placed,
            Rect {
                x: 10,
                y: 10,
                width: 51,
                height: 26
            }
        );
    }

    #[test]
    fn on_nv12_every_edge_is_even() {
        let placed = NV12.place(&detection(0.105, 0.21, 0.5, 0.5)).unwrap();
        assert_eq!(
            placed,
            Rect {
                x: 10,
                y: 10,
                width: 52,
                height: 26
            }
        );
        for edge in NV12.edges(placed, 3) {
            assert_eq!(edge.x % 2, 0);
            assert_eq!(edge.y % 2, 0);
            assert_eq!(edge.width % 2, 0);
            assert_eq!(edge.height % 2, 0);
        }
    }

    #[test]
    fn a_box_past_the_picture_is_cut_to_it_and_one_outside_is_not_drawn() {
        let placed = RGB.place(&detection(-0.2, 0.8, 0.5, 0.5)).unwrap();
        assert_eq!(
            placed,
            Rect {
                x: 0,
                y: 40,
                width: 30,
                height: 10
            }
        );
        assert_eq!(RGB.place(&detection(1.2, 0.1, 0.1, 0.1)), None);
        assert_eq!(RGB.place(&detection(0.1, 0.1, 0.0, 0.1)), None);
    }

    #[test]
    fn the_lines_are_inside_the_box_and_a_thin_box_is_filled() {
        let placed = Rect {
            x: 10,
            y: 10,
            width: 20,
            height: 10,
        };
        let [top, bottom, left, right] = RGB.edges(placed, 2);
        assert_eq!(
            top,
            Rect {
                height: 2,
                ..placed
            }
        );
        assert_eq!(
            bottom,
            Rect {
                y: 18,
                height: 2,
                ..placed
            }
        );
        assert_eq!(left, Rect { width: 2, ..placed });
        assert_eq!(
            right,
            Rect {
                x: 28,
                width: 2,
                ..placed
            }
        );
        let thin = Rect {
            height: 1,
            ..placed
        };
        assert_eq!(RGB.edges(thin, 4)[0], thin);
    }

    #[test]
    fn a_label_goes_above_its_box_where_there_is_room_and_inside_where_not() {
        let placed = Rect {
            x: 20,
            y: 30,
            width: 40,
            height: 10,
        };
        let (band, text) = RGB.label(placed, 30, 10);
        assert_eq!(
            band,
            Rect {
                x: 20,
                y: 16,
                width: 34,
                height: 14
            }
        );
        assert_eq!(
            text,
            Rect {
                x: 22,
                y: 18,
                width: 30,
                height: 10
            }
        );
        let (band, _) = RGB.label(Rect { y: 4, ..placed }, 30, 10);
        assert_eq!(band.y, 4, "inside the top of a box at the picture's top");
    }

    #[test]
    fn a_label_stays_in_the_picture() {
        let at_right = Rect {
            x: 90,
            y: 30,
            width: 10,
            height: 10,
        };
        let (band, text) = RGB.label(at_right, 30, 10);
        assert_eq!(band.x + band.width, 100);
        let (band, text_wide) = RGB.label(at_right, 300, 10);
        assert_eq!((band.x, band.width), (0, 100));
        assert!(text_wide.x + text_wide.width <= 100);
        assert!(text.x + text.width <= 100);
    }

    /// On a picture stored on its side and shown a quarter turn clockwise,
    /// a box's label is laid out as the picture is seen — above the box —
    /// which in the picture as stored is beside it on the left, ending
    /// where the box does at the bottom; its text is painted through its
    /// mask turned, and the box itself is where it is stored.
    #[test]
    fn a_label_on_a_turned_picture_is_above_its_box_as_shown() {
        use crate::orientation::Rotation;
        let canvas = Canvas {
            width: 200,
            height: 320,
            block: 2,
        };
        let options = DetectionOverlayOptions {
            others: Treatment::boxes(BoxStyle {
                label: Some(LabelStyle::new(16.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        let detections = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("person")]),
            vec![Detection::new(0, 0.9, 0.5, 0.25, 0.2, 0.5)],
        );
        let mut measure = |key: &MaskKey| match key {
            MaskKey::Text { text, .. } => Some((text.len() as u32 * 8, 12)),
            MaskKey::Stroke { rect, .. } => Some((rect.width, rect.height)),
            MaskKey::Turned {
                width,
                height,
                orientation,
                ..
            } => Some(orientation.display_size(*width, *height)),
        };
        let turned = Orientation::rotated(Rotation::Clockwise90);
        let marks = marks_turned(
            canvas,
            turned,
            &options,
            Some(&detections),
            None,
            &mut measure,
        );
        let placed = canvas.place(&detections.items[0]).unwrap();
        // The four edges, then the band, then the text.
        assert_eq!(marks.len(), 6, "{marks:?}");
        let [top, bottom, left, right] = [0, 1, 2, 3].map(|n| marks[n].rect);
        let edges = [top, bottom, left, right];
        let span = |pick: fn(&Rect) -> (u32, u32)| {
            edges
                .iter()
                .map(pick)
                .fold((u32::MAX, 0), |(lo, hi), (a, b)| (lo.min(a), hi.max(b)))
        };
        assert_eq!(
            span(|r| (r.x, r.x + r.width)),
            (placed.x, placed.x + placed.width)
        );
        assert_eq!(
            span(|r| (r.y, r.y + r.height)),
            (placed.y, placed.y + placed.height)
        );
        let (band, text) = (marks[4].rect, marks[5].rect);
        assert_eq!(
            band.x + band.width,
            placed.x,
            "beside the box: {band:?} {placed:?}"
        );
        assert_eq!(
            band.y + band.height,
            placed.y + placed.height,
            "{band:?} {placed:?}"
        );
        // "person 90%" is 88 by 12 as laid out: 12 across and 88 down, stored.
        assert_eq!((text.width, text.height), (12, 88));
        assert!(matches!(
            &marks[5].mask,
            Some(MaskKey::Turned { width: 88, height: 12, orientation, .. }) if *orientation == turned
        ));
    }

    /// A mask turned back the way a picture is stored puts each shown
    /// pixel's coverage where that pixel is stored, for each of the eight
    /// ways a picture can be turned, and keeps only the part asked for.
    #[test]
    fn a_turned_mask_lies_where_the_picture_is_stored() {
        use crate::orientation::Rotation;
        let shown = TextMask {
            width: 5,
            height: 3,
            coverage: (0..15).collect(),
        };
        for rotation in [
            Rotation::None,
            Rotation::Clockwise90,
            Rotation::Half,
            Rotation::Clockwise270,
        ] {
            for mirrored in [false, true] {
                let orientation = Orientation { rotation, mirrored };
                // The left four columns of the five.
                let stored = turned(&shown, 4, 3, orientation);
                assert_eq!(
                    (stored.width, stored.height),
                    orientation.display_size(4, 3)
                );
                for y in 0..stored.height {
                    for x in 0..stored.width {
                        let [left, top, ..] = orientation.to_display([
                            (x as f32 + 0.5) / stored.width as f32,
                            (y as f32 + 0.5) / stored.height as f32,
                            0.0,
                            0.0,
                        ]);
                        let (sx, sy) = ((left * 4.0) as u32, (top * 3.0) as u32);
                        assert_eq!(
                            stored.coverage[(y * stored.width + x) as usize],
                            shown.coverage[(sy * 5 + sx) as usize],
                            "{orientation:?} at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    /// A label says what its style asks for, each part in its place, and
    /// nothing at all where it asks for none.
    #[test]
    fn a_label_says_what_it_is_asked_to() {
        let detections =
            Detections::new("test", Arc::from(vec![Arc::<str>::from("person")]), vec![]);
        let all = &LabelStyle::default();
        let without_score = &LabelStyle {
            score: false,
            ..*all
        };
        let mut found = detection(0.0, 0.0, 0.1, 0.1);
        found.score = 0.876;
        assert_eq!(label_text(&detections, &found, all), "person 0.88");
        assert_eq!(label_text(&detections, &found, without_score), "person");
        found.track_id = Some(12);
        assert_eq!(label_text(&detections, &found, all), "person #12 0.88");
        found.class_id = 7;
        assert_eq!(
            label_text(&detections, &found, without_score),
            "class 7 #12"
        );
        found.classes.push(crate::elements::Classification::new(
            "classifier",
            Arc::from([Arc::from("minivan")]),
            0,
            0.7,
        ));
        assert_eq!(
            label_text(&detections, &found, without_score),
            "class 7 #12 | minivan"
        );
        let silent = LabelStyle {
            class: false,
            track_id: false,
            score: false,
            classes: false,
            ..LabelStyle::default()
        };
        let number_alone = &LabelStyle {
            track_id: true,
            ..silent
        };
        assert_eq!(label_text(&detections, &found, number_alone), "#12");
        let classified_alone = &LabelStyle {
            classes: true,
            ..silent
        };
        assert_eq!(
            label_text(&detections, &found, classified_alone),
            "| minivan"
        );
        assert_eq!(label_text(&detections, &found, &silent), "");
    }

    /// What `options` draw of one person and one car on a 400 by 200
    /// picture, with the zones and lines of [`watched`]: whether anything
    /// is, the colours painted, and the marks.
    fn drawn_by(options: &DetectionOverlayOptions) -> (bool, Vec<Color>, Vec<Mark>) {
        let canvas = Canvas {
            width: 400,
            height: 200,
            block: 2,
        };
        let detections = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("person"), Arc::<str>::from("car")]),
            vec![
                detection(0.1, 0.1, 0.2, 0.2),
                Detection::new(1, 0.4, 0.5, 0.5, 0.2, 0.2),
            ],
        );
        let analytics = watched();
        let mut measure = |key: &MaskKey| match key {
            MaskKey::Text { text, .. } => Some((text.len() as u32 * 8, 12)),
            MaskKey::Stroke { rect, .. } => Some((rect.width, rect.height)),
            MaskKey::Turned { .. } => None,
        };
        let marks = marks(
            canvas,
            options,
            Some(&detections),
            Some(&analytics),
            &mut measure,
        );
        let colors = marks.iter().map(|mark| mark.color).collect();
        (
            options.draws(Some(&detections), Some(&analytics)),
            colors,
            marks,
        )
    }

    fn person() -> Detection {
        detection(0.1, 0.1, 0.2, 0.2)
    }

    fn car() -> Detection {
        Detection::new(1, 0.4, 0.5, 0.5, 0.2, 0.2)
    }

    /// Every box is drawn by default, and zones and lines where asked for,
    /// each apart; nothing drawn where nothing is asked for.
    #[test]
    fn each_part_is_drawn_where_asked_for() {
        let (person, car) = (
            box_color(BoxColors::ByClass, &person()),
            box_color(BoxColors::ByClass, &car()),
        );
        let (draws, colors, _) = drawn_by(&DetectionOverlayOptions::default());
        assert!(draws && colors.contains(&person) && colors.contains(&car));
        assert!(!colors.contains(&ZONE) && !colors.contains(&LINE));

        let analytics_alone = |parts| DetectionOverlayOptions {
            parts,
            others: Treatment::none(),
            ..DetectionOverlayOptions::default()
        };
        let (draws, colors, _) = drawn_by(&analytics_alone(OverlayParts {
            lines: true,
            ..OverlayParts::default()
        }));
        assert!(draws && colors.contains(&LINE));
        assert!(!colors.contains(&ZONE) && !colors.contains(&person));
        let (_, colors, _) = drawn_by(&analytics_alone(OverlayParts {
            zones: true,
            ..OverlayParts::default()
        }));
        assert!(colors.contains(&ZONE) && !colors.contains(&LINE));
        let (draws, colors, _) = drawn_by(&analytics_alone(OverlayParts::default()));
        assert!(!draws && colors.is_empty());
    }

    /// A rule names a class by name or by number, the first naming it
    /// being the one it follows; each treatment draws its own way, from its
    /// own score; a class no rule names follows `others`.
    #[test]
    fn each_class_is_drawn_as_its_rule_says() {
        let (person, car) = (person(), car());
        let yellow = Color::new(255, 220, 0);
        let options = DetectionOverlayOptions {
            rules: vec![
                ClassRule::new(
                    "person",
                    Treatment::boxes(BoxStyle {
                        color: BoxColors::One(yellow),
                        line_width: 6,
                        ..BoxStyle::default()
                    }),
                ),
                ClassRule::new(1, Treatment::none()),
            ],
            others: Treatment::none(),
            ..DetectionOverlayOptions::default()
        };
        let (draws, colors, marks) = drawn_by(&options);
        assert!(draws);
        assert!(colors.iter().all(|color| *color == yellow), "{colors:?}");
        assert!(
            marks.iter().any(|mark| mark.rect.height == 6),
            "its own line width: {marks:?}"
        );
        assert!(!colors.contains(&box_color(BoxColors::ByClass, &car)));

        let sure = DetectionOverlayOptions {
            others: Treatment {
                min_score: 0.5,
                ..Treatment::default()
            },
            ..DetectionOverlayOptions::default()
        };
        let (_, colors, _) = drawn_by(&sure);
        assert!(colors.contains(&box_color(BoxColors::ByClass, &person)));
        assert!(
            !colors.contains(&box_color(BoxColors::ByClass, &car)),
            "the car's 0.4 is below its treatment's score"
        );

        let silent = LabelStyle {
            class: false,
            track_id: false,
            score: false,
            classes: false,
            ..LabelStyle::default()
        };
        let labelled = |label| DetectionOverlayOptions {
            font: Some(Vec::new()),
            rules: vec![ClassRule::new(
                "person",
                Treatment::boxes(BoxStyle {
                    label: Some(label),
                    ..BoxStyle::default()
                }),
            )],
            others: Treatment::none(),
            ..DetectionOverlayOptions::default()
        };
        let (_, _, marks) = drawn_by(&labelled(silent));
        assert!(
            marks.iter().all(|mark| mark.mask.is_none()),
            "a label saying nothing is not drawn: {marks:?}"
        );
        let (_, _, marks) = drawn_by(&labelled(LabelStyle::new(20.0)));
        let said = MaskKey::Text {
            text: "person 0.90".into(),
            size: 20.0f32.to_bits(),
        };
        assert!(marks.iter().any(|mark| mark.mask.as_ref() == Some(&said)));
    }

    /// Options an overlay cannot be made of are refused: a class two rules
    /// name, a label with no font or of no size, a margin below 0.
    #[test]
    fn options_that_cannot_be_drawn_with_are_refused() {
        let twice = DetectionOverlayOptions {
            rules: vec![
                ClassRule::new("face", Treatment::none()),
                ClassRule::new("face", Treatment::hidden(RedactStyle::mosaic())),
            ],
            ..DetectionOverlayOptions::default()
        };
        assert_eq!(
            twice.check(),
            Err(DetectionOverlayOptionsError::DuplicateClass("face".into()))
        );
        let labelled = |font, size| DetectionOverlayOptions {
            font,
            others: Treatment::boxes(BoxStyle {
                label: Some(LabelStyle::new(size)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        assert_eq!(
            labelled(None, 16.0).check(),
            Err(DetectionOverlayOptionsError::NoFont)
        );
        assert_eq!(
            labelled(Some(Vec::new()), 0.0).check(),
            Err(DetectionOverlayOptionsError::LabelSize(0.0))
        );
        assert_eq!(labelled(Some(Vec::new()), 16.0).check(), Ok(()));
        let below = DetectionOverlayOptions {
            others: Treatment {
                hide: Some(Hiding {
                    margin: -0.1,
                    ..Hiding::new(RedactStyle::mosaic())
                }),
                ..Treatment::none()
            },
            ..DetectionOverlayOptions::default()
        };
        assert_eq!(
            below.check(),
            Err(DetectionOverlayOptionsError::Margin(-0.1))
        );
        assert!(DetectionOverlayOptions::default().check().is_ok());
    }

    #[test]
    fn classes_keep_their_colour_and_text_reads_on_it() {
        let of_class = |class_id| Detection::new(class_id, 0.9, 0.0, 0.0, 0.1, 0.1);
        assert_eq!(
            box_color(BoxColors::ByClass, &of_class(3)),
            box_color(BoxColors::ByClass, &of_class(15))
        );
        assert_ne!(
            box_color(BoxColors::ByClass, &of_class(0)),
            box_color(BoxColors::ByClass, &of_class(1))
        );
        let tracked = |id| Detection {
            track_id: Some(id),
            ..of_class(0)
        };
        assert_ne!(
            box_color(BoxColors::ByTrack, &tracked(1)),
            box_color(BoxColors::ByTrack, &tracked(2)),
            "two objects of one class"
        );
        assert_eq!(
            box_color(BoxColors::ByTrack, &of_class(4)),
            box_color(BoxColors::ByClass, &of_class(4)),
            "by class where untracked"
        );
        assert_eq!(text_color(Color::new(255, 196, 0)), Color::BLACK);
        assert_eq!(text_color(Color::new(41, 121, 255)), Color::WHITE);
    }

    #[test]
    fn white_and_black_are_limited_range() {
        assert_eq!(bt709_limited(Color::WHITE), (235, 128, 128));
        assert_eq!(bt709_limited(Color::BLACK), (16, 128, 128));
    }

    /// A zone of two objects and a line crossed three times forward and
    /// once back — in fractions, as an `ObjectAnalytics` puts them.
    pub(crate) fn watched() -> Analytics {
        use crate::elements::{LineCount, ZoneCount};
        Analytics {
            zones: vec![ZoneCount {
                name: Arc::from("door"),
                corners: Arc::from(vec![(0.1, 0.1), (0.4, 0.1), (0.4, 0.8), (0.1, 0.8)]),
                objects: vec![0, 1],
                crowded: false,
            }],
            lines: vec![LineCount {
                name: Arc::from("gate"),
                start: (0.5, 0.2),
                end: (0.9, 0.8),
                forward: 3,
                backward: 1,
                crossed: Vec::new(),
            }],
        }
    }

    fn drawing_analytics() -> DetectionOverlayOptions {
        DetectionOverlayOptions {
            parts: OverlayParts {
                line_width: 4,
                ..OverlayParts::all()
            },
            others: Treatment::none(),
            ..DetectionOverlayOptions::default()
        }
    }

    /// A slanting line is painted in pieces, each through a mask of the
    /// rectangle around it alone, whose coverage is whole all along the
    /// line and nothing a few pixels off it.
    #[test]
    fn a_slanting_line_is_painted_along_its_length_and_nowhere_else() {
        let canvas = Canvas {
            width: 400,
            height: 200,
            block: 2,
        };
        let mut analytics = watched();
        analytics.zones.clear();
        let mut masks = std::collections::HashMap::new();
        let marks = marks(
            canvas,
            &drawing_analytics(),
            None,
            Some(&analytics),
            &mut |key| {
                let mask = rasterize(key, None).unwrap()?;
                let size = (mask.width, mask.height);
                masks.insert(key.clone(), mask);
                Some(size)
            },
        );
        assert!(marks.len() > 1, "in pieces: {marks:?}");
        assert!(marks.iter().all(|mark| mark.color == LINE));
        let coverage = |x: f32, y: f32| -> u8 {
            let (x, y) = (x as u32, y as u32);
            marks
                .iter()
                .filter(|mark| {
                    let rect = mark.rect;
                    x >= rect.x
                        && y >= rect.y
                        && x < rect.x + rect.width
                        && y < rect.y + rect.height
                })
                .map(|mark| {
                    let mask = &masks[mark.mask.as_ref().expect("through a mask")];
                    mask.coverage[((y - mark.rect.y) * mask.width + x - mark.rect.x) as usize]
                })
                .max()
                .unwrap_or(0)
        };
        let (from, to) = ((200.0, 40.0), (360.0, 160.0));
        let (length, across) = (200.0f32, ((120.0 / 200.0) as f32, (-160.0 / 200.0) as f32));
        for step in 0..=50 {
            let t = step as f32 / 50.0;
            let at = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
            assert_eq!(coverage(at.0, at.1), 255, "on the line at {at:?}");
            let off = (at.0 + across.0 * 6.0, at.1 + across.1 * 6.0);
            assert_eq!(coverage(off.0, off.1), 0, "six pixels off it at {off:?}");
        }
        let painted: u32 = marks
            .iter()
            .map(|mark| mark.rect.width * mark.rect.height)
            .sum();
        assert!(
            (painted as f32) < length * 4.0 * 16.0,
            "the pieces' rectangles, not the line's: {painted} pixels"
        );
    }

    /// A zone is outlined in amber, red while crowded, and labelled with
    /// how many objects are in it; a line is labelled with its crossings
    /// each way. Neither is drawn unless asked for.
    #[test]
    fn zones_and_lines_are_labelled_with_their_counts() {
        let canvas = Canvas {
            width: 400,
            height: 200,
            block: 2,
        };
        let mut measure = |key: &MaskKey| match key {
            MaskKey::Text { text, .. } => Some((text.len() as u32 * 8, 12)),
            MaskKey::Stroke { rect, .. } => Some((rect.width, rect.height)),
            MaskKey::Turned { .. } => None,
        };
        let mut analytics = watched();
        let drawn = marks(
            canvas,
            &drawing_analytics(),
            None,
            Some(&analytics),
            &mut measure,
        );
        assert!(drawn.iter().any(|mark| mark.color == ZONE));
        assert!(!drawn.iter().any(|mark| mark.color == CROWDED));
        analytics.zones[0].crowded = true;
        let crowded = marks(
            canvas,
            &drawing_analytics(),
            None,
            Some(&analytics),
            &mut measure,
        );
        assert!(crowded.iter().any(|mark| mark.color == CROWDED));
        assert!(!crowded.iter().any(|mark| mark.color == ZONE));
        let texts: Vec<&MaskKey> = drawn.iter().filter_map(|mark| mark.mask.as_ref()).collect();
        for label in ["door 2", "gate 3 / 1"] {
            assert!(
                texts.contains(&&MaskKey::Text {
                    text: label.to_owned(),
                    size: 16.0f32.to_bits(),
                }),
                "{label}: {texts:?}"
            );
        }

        let not_asked = DetectionOverlayOptions::default();
        assert!(!not_asked.draws(None, Some(&analytics)));
        assert!(marks(canvas, &not_asked, None, Some(&analytics), &mut measure).is_empty());
        assert!(drawing_analytics().draws(None, Some(&analytics)));
        assert!(!drawing_analytics().draws(None, Some(&Analytics::default())));
    }
}
