//! Making sense of what was found — DeepStream's `nvdsanalytics`:
//! [`ObjectAnalytics`] counts the objects inside zones of the picture and
//! the objects crossing lines across it.
//!
//! It reads [`Detections`](crate::elements::Detections) alone, never a
//! pixel, so one element serves pictures wherever they live. Its zones and
//! lines are placed as the boxes are, in fractions of the picture, so they
//! hold at any size the picture is scaled to.

mod object_analytics;

use thiserror::Error as ThisError;

pub use object_analytics::ObjectAnalytics;

/// The zones and lines an [`ObjectAnalytics`] watches.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnalyticsOptions {
    /// Zones to count the objects inside of.
    pub zones: Vec<Zone>,
    /// Lines to count the objects crossing.
    pub lines: Vec<Line>,
    /// Zones and lines of their own for some of a
    /// [`StreamMux`](crate::elements::StreamMux)'s streams, which watch those
    /// in place of `zones` and `lines` — a camera's doorway is not where
    /// another's is. A stream not named here, and a picture no mux handed
    /// on, is watched with `zones` and `lines`.
    pub streams: Vec<StreamAnalyticsOptions>,
}

/// The zones and lines one of a [`StreamMux`](crate::elements::StreamMux)'s
/// streams is watched with — see [`AnalyticsOptions::streams`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamAnalyticsOptions {
    /// The name the stream was registered under, as its
    /// [`StreamOrigin`](crate::elements::StreamOrigin) carries it.
    pub stream: String,
    /// Zones to count the objects inside of.
    pub zones: Vec<Zone>,
    /// Lines to count the objects crossing.
    pub lines: Vec<Line>,
}

/// A region of the picture, by its corners in fractions of the picture's
/// width and height, in order around it. An object is inside when the
/// bottom middle of its box is — where a person stands, as DeepStream
/// places it.
#[derive(Debug, Clone, PartialEq)]
pub struct Zone {
    /// What the zone is called in the [`Analytics`](crate::elements::Analytics)
    /// it is counted in.
    pub name: String,
    /// Its corners, three or more.
    pub corners: Vec<(f32, f32)>,
    /// The classes counted, or `None` for every class.
    pub classes: Option<Vec<usize>>,
    /// How many objects inside make it crowded, or `None` never to say so.
    pub crowded_at: Option<usize>,
}

impl Zone {
    /// A zone of every class, never said to be crowded.
    pub fn new(name: impl Into<String>, corners: Vec<(f32, f32)>) -> Self {
        Self {
            name: name.into(),
            corners,
            classes: None,
            crowded_at: None,
        }
    }
}

/// A line across the picture, from `start` to `end` in fractions of the
/// picture's width and height. An object crosses it when the bottom middle
/// of its box passes from one side of it to the other, as it is followed
/// through the pictures — which takes an
/// [`ObjectTracker`](crate::elements::ObjectTracker) before the analytics:
/// an object without a [`track_id`](crate::elements::Detection::track_id)
/// has no last place to have crossed from.
///
/// A side counts only once the object is `margin` away from the line, so
/// a box trembling about it is not counted crossing back and forth.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// What the line is called in the [`Analytics`](crate::elements::Analytics)
    /// it is counted in.
    pub name: String,
    /// Where it starts; which way is forward follows from it — see
    /// [`LineCount::forward`](crate::elements::LineCount::forward).
    pub start: (f32, f32),
    /// Where it ends.
    pub end: (f32, f32),
    /// The classes counted, or `None` for every class.
    pub classes: Option<Vec<usize>>,
    /// How far from the line an object must be to be on one side of it, as
    /// a fraction of the object's own box height — so a person far off and
    /// one close to the camera are held to the same. 0 counts every change
    /// of side, however small.
    pub margin: f32,
}

impl Line {
    /// A line counting every class, with a margin of a tenth of each
    /// object's height.
    pub fn new(name: impl Into<String>, start: (f32, f32), end: (f32, f32)) -> Self {
        Self {
            name: name.into(),
            start,
            end,
            classes: None,
            margin: 0.1,
        }
    }
}

/// Why an [`ObjectAnalytics`] could not be made of the zones and lines it
/// was given.
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum ObjectAnalyticsError {
    /// A zone that encloses nothing, or a corner that is not a number.
    #[error("zone {zone:?}: {reason}")]
    InvalidZone {
        /// The zone's name.
        zone: String,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// A line of no length, an end that is not a number, or a margin below
    /// zero.
    #[error("line {line:?}: {reason}")]
    InvalidLine {
        /// The line's name.
        line: String,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// A stream given zones and lines of its own twice.
    #[error("stream {0:?} is given zones and lines twice")]
    DuplicateStream(String),
}
