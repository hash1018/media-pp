//! What an analysis element found in a picture, as the
//! [`Metadata`](crate::buffer::Metadata) it puts on it: the data every
//! detector writes and every element after one reads, whatever does the
//! detecting.

use std::sync::Arc;

use crate::buffer::MediaBuffer;

/// One object a detector found, placed as fractions of the picture it is
/// attached to: `x` and `y` are its top-left corner, `width` and `height`
/// its size, each from 0 to 1. Multiply by the picture's own width and
/// height for pixels — which holds whatever the picture is scaled to
/// afterwards.
///
/// The box is of the picture as it is stored. A phone's portrait recording
/// is stored on its side and shown turned — see [`Orientation`] — and a
/// detector looks at it the right way up, but says where it found things
/// in the stored picture, where every element after it works;
/// [`Orientation::to_display`] puts a box where it is shown.
///
/// [`Orientation`]: crate::orientation::Orientation
/// [`Orientation::to_display`]: crate::orientation::Orientation::to_display
///
/// Non-exhaustive, so that what later elements learn of an object can be
/// added to it — which track it belongs to, what a second model says it is;
/// make one with [`Detection::new`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Detection {
    /// Index into the label set the model was trained on — see
    /// [`COCO_CLASS_LABELS`] for stock Ultralytics weights.
    pub class_id: usize,
    /// The model's confidence, after thresholding.
    pub score: f32,
    /// Left edge, as a fraction of the picture's width.
    pub x: f32,
    /// Top edge, as a fraction of the picture's height.
    pub y: f32,
    /// Width, as a fraction of the picture's width.
    pub width: f32,
    /// Height, as a fraction of the picture's height.
    pub height: f32,
    /// Which object this is across pictures, as an
    /// [`ObjectTracker`](crate::elements::ObjectTracker) numbers them: the
    /// same number on every picture the object is followed through. `None`
    /// before a tracker, and for what it is not yet sure is an object.
    pub track_id: Option<u64>,
    /// What second models said of the object — a classifier's after a
    /// detector, such as `SwOrtClassifier`'s with the `ort` feature, naming
    /// a car's make or a shirt's colour: DeepStream's secondary inference.
    /// One per classifier that had something to say, in the order they ran.
    pub classes: Vec<Classification>,
    /// Points the detector found on the object — a face's eyes, nose and
    /// the corners of its mouth — each `(x, y)` as fractions of the picture
    /// as stored, like the box, in the model's own order. None for a model
    /// that finds none, and on a box a tracker put where it expects the
    /// object to be.
    pub landmarks: Vec<(f32, f32)>,
}

/// What a classifier said of an object a detector found.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Classification {
    /// The name of the element that said it.
    pub classifier: Arc<str>,
    /// The classifier's class names, indexed by `class_id` — shared by every
    /// object it classifies.
    pub labels: Arc<[Arc<str>]>,
    /// The class it chose.
    pub class_id: usize,
    /// How sure it is, 0 to 1.
    pub score: f32,
}

impl Classification {
    /// `classifier`'s answer, class `class_id` of `labels` at `score`.
    pub fn new(
        classifier: impl Into<Arc<str>>,
        labels: Arc<[Arc<str>]>,
        class_id: usize,
        score: f32,
    ) -> Self {
        Self {
            classifier: classifier.into(),
            labels,
            class_id,
            score,
        }
    }

    /// What the class is called, where the classifier names its classes.
    pub fn label(&self) -> Option<&str> {
        self.labels.get(self.class_id).map(|label| &**label)
    }
}

impl Detection {
    /// An object of `class_id` found at `score`, its box at `x`, `y`,
    /// `width` by `height` as fractions of the picture.
    pub fn new(class_id: usize, score: f32, x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            class_id,
            score,
            x,
            y,
            width,
            height,
            track_id: None,
            classes: Vec::new(),
            landmarks: Vec::new(),
        }
    }
}

/// What a detector found in one picture: the
/// [`Metadata`](crate::buffer::Metadata) it puts on the picture it hands on.
/// Read it downstream with `buffer.metadata()?.get::<Detections>()`.
///
/// An empty list is an answer — the detector looked and found nothing —
/// where a picture carrying no `Detections` was not looked at.
///
/// Whatever does the detecting writes one — this crate's own detectors, or
/// an application's, with [`Detections::new`] and [`Detections::attach_to`]
/// — and whatever reads one, as the detection overlays do, takes it from
/// any of them. Non-exhaustive, like [`Detection`], for what later elements
/// add.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Detections {
    /// The name of the element that looked.
    pub detector: Arc<str>,
    /// The model's class names, indexed by [`Detection::class_id`] — shared
    /// by every picture the detector looks at, not copied onto each.
    pub labels: Arc<[Arc<str>]>,
    /// What it found, most confident first.
    pub items: Vec<Detection>,
    /// Whether these are where a tracker expects the objects to be rather
    /// than what a detector saw: a picture the detector did not look at —
    /// see `OrtDetectorOptions::interval`, with the `ort` feature — carries
    /// the tracker's, so that every picture has its boxes.
    pub predicted: bool,
}

impl Detections {
    /// What `detector`, whose classes are named `labels`, found: `items`,
    /// most confident first.
    pub fn new(
        detector: impl Into<Arc<str>>,
        labels: Arc<[Arc<str>]>,
        items: Vec<Detection>,
    ) -> Self {
        Self {
            detector: detector.into(),
            labels,
            items,
            predicted: false,
        }
    }

    /// What `detection`'s class is called, where the model or the caller
    /// named its classes.
    pub fn label(&self, detection: &Detection) -> Option<&str> {
        self.labels.get(detection.class_id).map(|label| &**label)
    }

    /// `buf`, carrying these beside whatever it carried already, in place of
    /// any `Detections` it had.
    pub fn attach_to(self, buf: MediaBuffer) -> MediaBuffer {
        let metadata = buf.metadata().cloned().unwrap_or_default().with(self);
        buf.with_metadata(metadata)
    }
}

/// That a picture begins a new shot of an edited video — the first after a
/// cut, or after a fade through black — as a cut detector such as
/// [`SwCutDetector`](crate::elements::SwCutDetector) puts it on the
/// picture. An [`ObjectTracker`](crate::elements::ObjectTracker) starts
/// over on it, and an [`ObjectAnalytics`](crate::elements::ObjectAnalytics)
/// forgets where each object was, since nothing in the new shot is where
/// anything was. The first picture of a stream carries none.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SceneCut {
    /// The name of the element that found it.
    pub detector: Arc<str>,
    /// How far the picture was from the one before: the mean difference
    /// of their luma, plus that of their chroma, each out of 255.
    pub score: f32,
}

/// What an [`ObjectAnalytics`](crate::elements::ObjectAnalytics) made of one
/// picture's [`Detections`]: for each of its zones the objects inside, and
/// for each of its lines how many objects have crossed it so far and which
/// crossed on this picture — the [`Metadata`](crate::buffer::Metadata) it
/// puts beside them. Read it with `buffer.metadata()?.get::<Analytics>()`.
///
/// The zones and lines are in the order the element was given them.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Analytics {
    /// Each zone, and what is in it on this picture.
    pub zones: Vec<ZoneCount>,
    /// Each line, and what has crossed it.
    pub lines: Vec<LineCount>,
}

/// One zone of an [`Analytics`] on one picture.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ZoneCount {
    /// The zone's name.
    pub name: Arc<str>,
    /// Its corners, in fractions of the picture's width and height, as
    /// the zone was given — so that what draws the zone draws the one
    /// counted, a stream's own included.
    pub corners: Arc<[(f32, f32)]>,
    /// The objects inside it, as indices into the picture's
    /// [`Detections::items`].
    pub objects: Vec<usize>,
    /// Whether at least the zone's `crowded_at` objects are inside.
    pub crowded: bool,
}

/// One line of an [`Analytics`]: how many objects have crossed it each way
/// since the element started, and which crossed on this picture.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LineCount {
    /// The line's name.
    pub name: Arc<str>,
    /// Where it starts, in fractions of the picture's width and height, as
    /// the line was given.
    pub start: (f32, f32),
    /// Where it ends.
    pub end: (f32, f32),
    /// Crossings from its left to its right, as seen going from its start
    /// to its end — so a line drawn left to right counts what moves down
    /// the picture.
    pub forward: u64,
    /// Crossings the other way.
    pub backward: u64,
    /// The crossings on this picture.
    pub crossed: Vec<Crossing>,
}

/// An object crossing a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Crossing {
    /// The object, by [`Detection::track_id`].
    pub track_id: u64,
    /// Its class.
    pub class_id: usize,
    /// Whether it crossed forward — see [`LineCount::forward`].
    pub forward: bool,
}

/// The 80 COCO classes stock Ultralytics weights are trained on, in their
/// order. Meaningless for a model trained on another set: index
/// [`Detection::class_id`] into its own labels instead.
#[rustfmt::skip]
pub const COCO_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Metadata;
    use crate::ffmpeg;

    #[test]
    fn a_detection_is_named_by_its_class() {
        let detections = Detections::new(
            "d",
            Arc::from([Arc::from("person"), Arc::from("bicycle")]),
            Vec::new(),
        );
        let bicycle = Detection::new(1, 1.0, 0.0, 0.0, 1.0, 1.0);
        assert_eq!(detections.label(&bicycle), Some("bicycle"));
        let unnamed = Detection {
            class_id: 7,
            ..bicycle
        };
        assert_eq!(detections.label(&unnamed), None);
    }

    #[test]
    fn detections_go_beside_what_a_buffer_carried() {
        #[derive(Debug, PartialEq)]
        struct Earlier;
        let buf = MediaBuffer::video(ffmpeg::frame::Video::new(
            ffmpeg::format::Pixel::RGB24,
            2,
            2,
        ))
        .with_metadata(Metadata::new().with(Earlier));
        let detections = Detections::new("d", Arc::from([Arc::from("person")]), Vec::new());
        let buf = detections.clone().attach_to(buf);
        assert_eq!(
            buf.metadata().and_then(|m| m.get::<Detections>()),
            Some(&detections)
        );
        assert!(buf.metadata().unwrap().contains::<Earlier>());
    }
}
