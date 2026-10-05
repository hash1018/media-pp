//! Drawing what a detector found onto the picture it found it in, as
//! filters: each picture that carries [`Detections`] is handed on with a
//! box around each object, and with a label above it where a font is given.
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
//! it came. Its [`Detections`] go on with the drawn copy, so whatever comes
//! after can still read them.

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

use crate::color::Color;
use crate::elements::{Detection, Detections};

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
}

impl Default for DetectionOverlayOptions {
    fn default() -> Self {
        Self {
            line_width: 2,
            min_score: 0.0,
            colors: BoxColors::ByClass,
            labels: None,
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

/// How the label above each box is drawn: the class name, and the score
/// if asked for, on a band of the box's colour.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelStyle {
    /// Raw TrueType or OpenType font bytes. This crate bundles no font of
    /// its own, as with a compositor's text layers.
    pub font_data: Vec<u8>,
    /// The text's pixel height, in the picture drawn on.
    pub size: f32,
    /// Whether the score follows the name, as in `person 0.87`.
    pub score: bool,
}

impl LabelStyle {
    /// A label in `font_data`, 16 pixels high, with the score.
    pub fn new(font_data: Vec<u8>) -> Self {
        Self {
            font_data,
            size: 16.0,
            score: true,
        }
    }
}

/// A rectangle of a picture, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// What a label says: the class's name, or `class` and its number where
/// the model names none; then the object's number where a tracker gave it
/// one, and the score if asked for — `person #7 0.87`.
pub(crate) fn label_text(detections: &Detections, detection: &Detection, score: bool) -> String {
    let mut text = match detections.label(detection) {
        Some(name) => name.to_owned(),
        None => format!("class {}", detection.class_id),
    };
    if let Some(id) = detection.track_id {
        text.push_str(&format!(" #{id}"));
    }
    if score {
        text.push_str(&format!(" {:.2}", detection.score));
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

    #[test]
    fn a_label_says_the_class_and_score() {
        let detections =
            Detections::new("test", Arc::from(vec![Arc::<str>::from("person")]), vec![]);
        let mut found = detection(0.0, 0.0, 0.1, 0.1);
        found.score = 0.876;
        assert_eq!(label_text(&detections, &found, true), "person 0.88");
        assert_eq!(label_text(&detections, &found, false), "person");
        found.track_id = Some(12);
        assert_eq!(label_text(&detections, &found, true), "person #12 0.88");
        found.class_id = 7;
        assert_eq!(label_text(&detections, &found, false), "class 7 #12");
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
}
