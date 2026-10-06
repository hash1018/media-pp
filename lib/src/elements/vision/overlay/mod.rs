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

use crate::color::Color;
use crate::elements::source::{TextMask, TextRasterError, rasterize_coverage};
use crate::elements::{Analytics, Detection, Detections};

/// How a detection overlay draws, beside the picture and what was found
/// in it.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionOverlayOptions {
    /// How thick each box's lines are, in pixels of the picture drawn on.
    /// Rounded up to even on a picture whose colour is stored per 2x2
    /// block, NV12 or YUV 4:2:0, so that a line's colour does not bleed
    /// half a block outside it.
    pub line_width: u32,
    /// Detections less confident than this are not drawn.
    pub min_score: f32,
    /// What colour each box is.
    pub colors: BoxColors,
    /// The labels above the boxes, or `None` for boxes alone.
    pub labels: Option<LabelStyle>,
    /// Which of what a picture carries are drawn: by default the boxes
    /// alone.
    pub parts: OverlayParts,
    /// What is found hidden — mosaicked, blurred or filled — before
    /// anything is drawn over it, or `None` to hide nothing. To hand on
    /// the hidden picture alone, set `parts.boxes` to `false`.
    pub redact: Option<Redaction>,
}

impl Default for DetectionOverlayOptions {
    fn default() -> Self {
        Self {
            line_width: 2,
            min_score: 0.0,
            colors: BoxColors::ByClass,
            labels: None,
            parts: OverlayParts::default(),
            redact: None,
        }
    }
}

/// How an overlay hides what was found — faces, number plates — in each
/// box, grown by [`Redaction::margin`]: DeepStream's redaction, on a copy
/// of the picture as everything an overlay draws is.
///
/// A mosaic or a blur is made of cells sized by the box, not fixed in
/// pixels: a face's shorter side is cut into [`RedactStyle::Mosaic`]'s
/// `cells`, so that a face close to the camera is hidden as well as one far
/// off — a cell fixed in pixels leaves the near face most of its features.
/// `min_cell` keeps a small face's cells from shrinking to a pixel or two,
/// where a mosaic hides nothing.
#[derive(Debug, Clone, PartialEq)]
pub struct Redaction {
    /// How each box is hidden.
    pub style: RedactStyle,
    /// How far each box is grown before it is hidden, each way, as a
    /// fraction of its own width and height: 0.1 hides a tenth more on
    /// every side, which a box drawn a little tight — or a face moving
    /// between two detections — would otherwise leave showing. 0 or more.
    pub margin: f32,
    /// Detections less confident than this are not hidden. Separate from
    /// [`DetectionOverlayOptions::min_score`], as hiding wants to err the
    /// other way from drawing: by default everything the detector kept.
    pub min_score: f32,
    /// The classes hidden, or `None` for every class.
    pub classes: Option<Vec<usize>>,
}

impl Redaction {
    /// `style`, a tenth's margin, every class and every score.
    pub fn new(style: RedactStyle) -> Self {
        Self {
            style,
            margin: 0.1,
            min_score: 0.0,
            classes: None,
        }
    }

    /// Why these cannot be hidden by, if they cannot.
    pub(crate) fn check(&self) -> Result<(), &'static str> {
        if !self.margin.is_finite() || self.margin < 0.0 {
            return Err("its margin is not a number of 0 or more");
        }
        if !self.min_score.is_finite() {
            return Err("its min_score is not a number");
        }
        Ok(())
    }
}

/// How a [`Redaction`] hides a box.
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

/// Which of what a picture carries an overlay draws. Each label goes with
/// what it labels, where [`DetectionOverlayOptions::labels`] gives a font.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayParts {
    /// Each object's box, and its label — what [`LabelStyle`] says goes in
    /// it — from the picture's [`Detections`].
    pub boxes: bool,
    /// Each zone of an [`ObjectAnalytics`](crate::elements::ObjectAnalytics),
    /// from the picture's [`Analytics`]: its outline in amber — red while it
    /// is crowded — `line_width` thick, labelled with its name and how many
    /// objects are in it, `door 2`.
    pub zones: bool,
    /// Each line of an [`ObjectAnalytics`](crate::elements::ObjectAnalytics),
    /// from the picture's [`Analytics`]: drawn across the picture in cyan,
    /// `line_width` thick, labelled with its name and its crossings forward
    /// and backward, `gate 12 / 3`.
    pub lines: bool,
}

impl Default for OverlayParts {
    /// The boxes alone: zones and lines are drawn where asked for.
    fn default() -> Self {
        Self {
            boxes: true,
            zones: false,
            lines: false,
        }
    }
}

impl OverlayParts {
    /// Boxes, zones and lines alike.
    pub fn all() -> Self {
        Self {
            boxes: true,
            zones: true,
            lines: true,
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

/// How the label above each box is drawn, and what it says — of the class
/// name, the number a tracker gave the object, the score and what
/// classifiers said of it, each as asked for, `car #7 0.87 | minivan` with
/// all four — on a band of the box's colour. A box whose label would say
/// nothing has none.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelStyle {
    /// Raw TrueType or OpenType font bytes. This crate bundles no font of
    /// its own, as with a compositor's text layers.
    pub font_data: Vec<u8>,
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
    /// A label in `font_data`, 16 pixels high, saying all it can.
    pub fn new(font_data: Vec<u8>) -> Self {
        Self {
            font_data,
            size: 16.0,
            class: true,
            track_id: true,
            score: true,
            classes: true,
        }
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
    parts: LabelParts,
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

/// The detections an overlay draws, most confident last so that it is drawn
/// over the rest.
pub(crate) fn drawn(detections: &Detections, min_score: f32) -> Vec<&Detection> {
    let mut drawn: Vec<&Detection> = detections
        .items
        .iter()
        .filter(|detection| detection.score >= min_score)
        .collect();
    drawn.sort_by(|a, b| a.score.total_cmp(&b.score));
    drawn
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
    /// A label's text, in the overlay's font.
    Text(String),
    /// A piece of a line `width` pixels thick, from `from` to `to` — in
    /// sixteenths of a pixel of the picture — through exactly `rect`, which
    /// the mask is the size of.
    Stroke {
        rect: Rect,
        from: (i32, i32),
        to: (i32, i32),
        width: u32,
    },
}

/// What a mask is made of: a label in the overlay's font and size where it
/// has one — `None` draws no label — and a line's piece from its key
/// alone.
pub(crate) fn rasterize(
    key: &MaskKey,
    font: Option<(&ab_glyph::FontArc, f32)>,
) -> Result<Option<TextMask>, TextRasterError> {
    match key {
        MaskKey::Text(text) => match font {
            Some((font, size)) => rasterize_coverage(font, size, text),
            None => Ok(None),
        },
        &MaskKey::Stroke {
            rect,
            from,
            to,
            width,
        } => Ok(Some(stroke_mask(rect, from, to, width))),
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

/// What an overlay draws with, of its options: all but the font.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Style {
    line_width: u32,
    min_score: f32,
    colors: BoxColors,
    parts: OverlayParts,
    label: LabelParts,
}

/// What a box's label says, of a [`LabelStyle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LabelParts {
    class: bool,
    track_id: bool,
    score: bool,
    classes: bool,
}

/// One box to hide, in pixels of the picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Hide {
    /// `rect` cut into `cells` — across, down — each painted the mean of
    /// the pixels under it, or with `smooth` blended into its neighbours.
    Cells {
        rect: Rect,
        cells: (u32, u32),
        smooth: bool,
    },
    /// `rect` filled with one colour.
    Fill { rect: Rect, color: Color },
}

impl Redaction {
    /// Whether `detection` is one this hides.
    fn hides(&self, detection: &Detection) -> bool {
        detection.score >= self.min_score
            && self
                .classes
                .as_ref()
                .is_none_or(|classes| classes.contains(&detection.class_id))
    }
}

/// What `redaction` hides of `detections` on a picture `canvas` describes:
/// each box grown by the margin, cut to the picture and out to whole
/// blocks, and for a mosaic or a blur the cells it is cut into.
pub(crate) fn hides(canvas: Canvas, redaction: &Redaction, detections: &Detections) -> Vec<Hide> {
    let margin = redaction.margin;
    detections
        .items
        .iter()
        .filter(|detection| redaction.hides(detection))
        .filter_map(|detection| {
            let grown = Detection::new(
                detection.class_id,
                detection.score,
                detection.x - detection.width * margin,
                detection.y - detection.height * margin,
                detection.width * (1.0 + 2.0 * margin),
                detection.height * (1.0 + 2.0 * margin),
            );
            let rect = canvas.place(&grown)?;
            let (cells, min_cell, smooth) = match redaction.style {
                RedactStyle::Fill(color) => return Some(Hide::Fill { rect, color }),
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
        self.style().draws(detections, analytics)
            || self.redact.as_ref().is_some_and(|redaction| {
                detections.is_some_and(|detections| {
                    detections
                        .items
                        .iter()
                        .any(|detection| redaction.hides(detection))
                })
            })
    }

    pub(crate) fn style(&self) -> Style {
        Style {
            line_width: self.line_width,
            min_score: self.min_score,
            colors: self.colors,
            parts: self.parts,
            // Without a font no label is drawn, whatever it would say.
            label: self
                .labels
                .as_ref()
                .map_or_else(LabelParts::default, |style| LabelParts {
                    class: style.class,
                    track_id: style.track_id,
                    score: style.score,
                    classes: style.classes,
                }),
        }
    }
}

impl Style {
    /// Whether a picture carrying `detections` and `analytics` has anything
    /// to draw: a picture with nothing is handed on as it came.
    pub(crate) fn draws(
        &self,
        detections: Option<&Detections>,
        analytics: Option<&Analytics>,
    ) -> bool {
        (self.parts.boxes
            && detections.is_some_and(|detections| {
                detections
                    .items
                    .iter()
                    .any(|detection| detection.score >= self.min_score)
            }))
            || analytics.is_some_and(|analytics| {
                (self.parts.zones && !analytics.zones.is_empty())
                    || (self.parts.lines && !analytics.lines.is_empty())
            })
    }
}

/// What to paint for `detections` and `analytics` on a picture `canvas`
/// describes, in order: the zones and lines, then each box's lines, then
/// its label's band and text. `mask` makes — or finds — the mask a key
/// names, and says its size, or `None` where it has nothing to draw.
pub(crate) fn marks(
    canvas: Canvas,
    style: Style,
    detections: Option<&Detections>,
    analytics: Option<&Analytics>,
    mask: &mut dyn FnMut(&MaskKey) -> Option<(u32, u32)>,
) -> Vec<Mark> {
    let mut marks = Vec::new();
    if let Some(analytics) = analytics {
        let pixels = |(x, y): (f32, f32)| (x * canvas.width as f32, y * canvas.height as f32);
        let zones = analytics.zones.iter().filter(|_| style.parts.zones);
        for zone in zones {
            let corners: Vec<(f32, f32)> = zone.corners.iter().copied().map(pixels).collect();
            let color = if zone.crowded { CROWDED } else { ZONE };
            for (index, &from) in corners.iter().enumerate() {
                let to = corners[(index + 1) % corners.len()];
                stroke(canvas, style, from, to, color, &mut marks, mask);
            }
            if let Some(anchor) = canvas.bounds(&corners, 0.0) {
                let text = format!("{} {}", zone.name, zone.objects.len());
                labelled(canvas, anchor, text, color, &mut marks, mask);
            }
        }
        let lines = analytics.lines.iter().filter(|_| style.parts.lines);
        for line in lines {
            let (from, to) = (pixels(line.start), pixels(line.end));
            stroke(canvas, style, from, to, LINE, &mut marks, mask);
            if let Some(anchor) = canvas.bounds(&[from, to], 0.0) {
                let text = format!("{} {} / {}", line.name, line.forward, line.backward);
                labelled(canvas, anchor, text, LINE, &mut marks, mask);
            }
        }
    }
    let Some(detections) = detections.filter(|_| style.parts.boxes) else {
        return marks;
    };
    for detection in drawn(detections, style.min_score) {
        let Some(placed) = canvas.place(detection) else {
            continue;
        };
        let color = box_color(style.colors, detection);
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
        let text = label_text(detections, detection, style.label);
        if !text.is_empty() {
            labelled(canvas, placed, text, color, &mut marks, mask);
        }
    }
    marks
}

/// A line from `from` to `to`, in pixels, as pieces each painted through
/// a mask of the rectangle around it.
fn stroke(
    canvas: Canvas,
    style: Style,
    from: (f32, f32),
    to: (f32, f32),
    color: Color,
    marks: &mut Vec<Mark>,
    mask: &mut dyn FnMut(&MaskKey) -> Option<(u32, u32)>,
) {
    let width = canvas.up(style.line_width.max(1));
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

/// The label `text` for something placed at `anchor`: its band in `color`,
/// and its text in black or white on it.
fn labelled(
    canvas: Canvas,
    anchor: Rect,
    text: String,
    color: Color,
    marks: &mut Vec<Mark>,
    mask: &mut dyn FnMut(&MaskKey) -> Option<(u32, u32)>,
) {
    let key = MaskKey::Text(text);
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

    /// A label says what its style asks for, each part in its place, and
    /// nothing at all where it asks for none.
    #[test]
    fn a_label_says_what_it_is_asked_to() {
        let detections =
            Detections::new("test", Arc::from(vec![Arc::<str>::from("person")]), vec![]);
        let all = LabelParts {
            class: true,
            track_id: true,
            score: true,
            classes: true,
        };
        let without_score = LabelParts {
            score: false,
            ..all
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
        let number_alone = LabelParts {
            track_id: true,
            ..LabelParts::default()
        };
        assert_eq!(label_text(&detections, &found, number_alone), "#12");
        let classified_alone = LabelParts {
            classes: true,
            ..LabelParts::default()
        };
        assert_eq!(
            label_text(&detections, &found, classified_alone),
            "| minivan"
        );
        assert_eq!(label_text(&detections, &found, LabelParts::default()), "");
    }

    /// Boxes are drawn by default and zones and lines where asked for, each
    /// apart; an object whose label says nothing has its box alone.
    #[test]
    fn each_part_is_drawn_where_asked_for() {
        let canvas = Canvas {
            width: 400,
            height: 200,
            block: 2,
        };
        let detections = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("person")]),
            vec![detection(0.1, 0.1, 0.2, 0.2)],
        );
        let analytics = watched();
        let mut measure = |key: &MaskKey| match key {
            MaskKey::Text(text) => Some((text.len() as u32 * 8, 12)),
            MaskKey::Stroke { rect, .. } => Some((rect.width, rect.height)),
        };
        let mut drawn = |parts: OverlayParts, labels: Option<LabelStyle>| {
            let options = DetectionOverlayOptions {
                parts,
                labels,
                ..DetectionOverlayOptions::default()
            };
            let style = options.style();
            let marks = marks(
                canvas,
                style,
                Some(&detections),
                Some(&analytics),
                &mut measure,
            );
            let colors: Vec<Color> = marks.iter().map(|mark| mark.color).collect();
            (
                style.draws(Some(&detections), Some(&analytics)),
                colors,
                marks,
            )
        };
        let boxed = box_color(BoxColors::ByClass, &detections.items[0]);
        let (draws, colors, _) = drawn(OverlayParts::default(), None);
        assert!(draws && colors.contains(&boxed));
        assert!(!colors.contains(&ZONE) && !colors.contains(&LINE));
        let lines_alone = OverlayParts {
            boxes: false,
            zones: false,
            lines: true,
        };
        let (draws, colors, _) = drawn(lines_alone, None);
        assert!(draws && colors.contains(&LINE));
        assert!(!colors.contains(&ZONE) && !colors.contains(&boxed));
        let zones_alone = OverlayParts {
            boxes: false,
            zones: true,
            lines: false,
        };
        let (_, colors, _) = drawn(zones_alone, None);
        assert!(colors.contains(&ZONE) && !colors.contains(&LINE));
        let nothing = OverlayParts {
            boxes: false,
            zones: false,
            lines: false,
        };
        let (draws, colors, _) = drawn(nothing, None);
        assert!(!draws && colors.is_empty());

        let silent = LabelStyle {
            class: false,
            track_id: false,
            score: false,
            classes: false,
            ..LabelStyle::new(Vec::new())
        };
        let (_, _, marks) = drawn(OverlayParts::default(), Some(silent));
        assert!(
            marks.iter().all(|mark| mark.mask.is_none()),
            "a label saying nothing is not drawn: {marks:?}"
        );
        let (_, _, marks) = drawn(OverlayParts::default(), Some(LabelStyle::new(Vec::new())));
        assert!(
            marks
                .iter()
                .any(|mark| mark.mask == Some(MaskKey::Text("person 0.90".into())))
        );
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

    fn drawing_analytics() -> Style {
        DetectionOverlayOptions {
            line_width: 4,
            parts: OverlayParts::all(),
            ..DetectionOverlayOptions::default()
        }
        .style()
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
            drawing_analytics(),
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
            MaskKey::Text(text) => Some((text.len() as u32 * 8, 12)),
            MaskKey::Stroke { rect, .. } => Some((rect.width, rect.height)),
        };
        let mut analytics = watched();
        let drawn = marks(
            canvas,
            drawing_analytics(),
            None,
            Some(&analytics),
            &mut measure,
        );
        assert!(drawn.iter().any(|mark| mark.color == ZONE));
        assert!(!drawn.iter().any(|mark| mark.color == CROWDED));
        analytics.zones[0].crowded = true;
        let crowded = marks(
            canvas,
            drawing_analytics(),
            None,
            Some(&analytics),
            &mut measure,
        );
        assert!(crowded.iter().any(|mark| mark.color == CROWDED));
        assert!(!crowded.iter().any(|mark| mark.color == ZONE));
        let texts: Vec<&MaskKey> = drawn.iter().filter_map(|mark| mark.mask.as_ref()).collect();
        for label in ["door 2", "gate 3 / 1"] {
            assert!(
                texts.contains(&&MaskKey::Text(label.to_owned())),
                "{label}: {texts:?}"
            );
        }

        let not_asked = DetectionOverlayOptions::default().style();
        assert!(!not_asked.draws(None, Some(&analytics)));
        assert!(marks(canvas, not_asked, None, Some(&analytics), &mut measure).is_empty());
        assert!(drawing_analytics().draws(None, Some(&analytics)));
        assert!(!drawing_analytics().draws(None, Some(&Analytics::default())));
    }
}
