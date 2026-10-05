//! Following objects from picture to picture — DeepStream's `nvtracker`:
//! [`ObjectTracker`] numbers what a detector found so that one object keeps
//! one number, and puts where it expects them on the pictures a detector
//! let by.
//!
//! It reads and writes [`Detections`](crate::elements::Detections) alone,
//! never a pixel, so one element serves pictures wherever they live.

mod assign;
mod byte_track;
mod dcf;
mod fft;
mod kalman;
mod luma;
mod object_tracker;

pub use object_tracker::ObjectTracker;

/// How an [`ObjectTracker`] decides what is an object and what is the same
/// one — ByteTrack's own defaults.
///
/// A detection at `high_score` or above is matched first and may begin a
/// track; one between `low_score` and `high_score` only keeps a track
/// already followed, as an object partly hidden scores lower and is still
/// there. For those to reach the tracker at all, the detector before it
/// must keep them: give it a confidence threshold of `low_score` —
/// `OrtDetectorOptions::conf_threshold`, with the `ort` feature — and an
/// overlay after it a `min_score` of what is worth drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerOptions {
    /// The score a detection is confident at.
    pub high_score: f32,
    /// The score below which a detection is ignored.
    pub low_score: f32,
    /// The score a detection matching nothing must have to begin a track.
    pub new_track_score: f32,
    /// How many pictures a new object must be detected on before it is
    /// numbered: 2, ByteTrack's, keeps a detection seen once from counting
    /// as an object. Where the detector looks only now and then — a long
    /// `OrtDetectorOptions::interval` — a new
    /// object has moved by its second sighting further than an unmoving
    /// expectation can match, and 1 numbers it at once.
    pub confirm_after: u32,
    /// How many pictures an object may go unmatched before it is forgotten;
    /// one that comes back within them keeps its number.
    pub lost_pictures: u32,
    /// Whether a detection is matched only to a track of its own class,
    /// so that a person's number never passes to a car.
    pub per_class: bool,
    /// Whether each object is also followed by how it looks — a correlation
    /// filter learned from the pixels around it, as DeepStream's NvDCF has
    /// — rather than by its motion alone. Between detections it finds the
    /// object where it went rather than where its speed would have taken
    /// it, which holds through a turn and over a longer
    /// interval; on a detection's picture it puts the track where the
    /// object is before matching, so that one that moved far is still
    /// matched.
    ///
    /// It reads the pixels around each object: of 8-bit YUV, grey and RGB
    /// pictures in system memory; with the `cuda` feature of NV12 and BGRA
    /// CUDA pictures, copying down only those regions; and with `metal` of
    /// NV12 and BGRA VideoToolbox pictures, read where they are, which the
    /// CPU of an Apple silicon Mac shares with its GPU. Pictures it cannot
    /// read are followed by motion alone.
    pub visual: bool,
}

impl Default for TrackerOptions {
    fn default() -> Self {
        Self {
            high_score: 0.5,
            low_score: 0.1,
            new_track_score: 0.6,
            confirm_after: 2,
            lost_pictures: 30,
            per_class: true,
            visual: false,
        }
    }
}
