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
/// Non-exhaustive, so that what later elements learn of an object — which
/// track it belongs to, what a second model says it is — can be added to
/// it; make one with [`Detection::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
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
