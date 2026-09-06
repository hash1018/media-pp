//! Runtime control shared by both chroma-key backends.

use std::sync::{Arc, Mutex};

use super::options::ChromaKeyOptions;

/// The live settings one chroma-key element reads and its
/// [`ChromaKeyHandle`] writes.
///
/// One lock around the whole [`ChromaKeyOptions`] rather than an atomic per
/// field, which is where this differs from [`AudioVolume`]'s control block:
/// gain and mute are independent, but a key color, its threshold, and its
/// feather width are one setting in three parts, and a frame keyed with a
/// new color against the old threshold is a frame nobody asked for.
///
/// The lock costs nothing where it is taken. An element reads it once per
/// frame — not once per pixel — because both backends rebuild their keying
/// parameters per frame anyway: the D3D11 one writes a constant buffer
/// before each draw, and the software one passes the values into its own
/// per-pixel loop.
///
/// [`AudioVolume`]: crate::elements::AudioVolume
#[derive(Debug)]
pub(super) struct ChromaKeyControl {
    options: Mutex<ChromaKeyOptions>,
}

impl ChromaKeyControl {
    pub(super) fn new(options: ChromaKeyOptions) -> Self {
        Self {
            options: Mutex::new(options),
        }
    }

    /// A consistent copy of every setting, for one frame to be keyed with.
    ///
    /// Take this once and key from the copy. Reading it again mid-frame
    /// could pick up a change made in between, which is the one thing the
    /// single lock is here to prevent.
    pub(super) fn get(&self) -> ChromaKeyOptions {
        *self
            .options
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set(&self, options: ChromaKeyOptions) {
        *self
            .options
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = options;
    }
}

/// Thread-safe runtime control for a chroma-key element, from either
/// backend.
///
/// What it is for is a slider. Keying is tuned by eye — a threshold that
/// works for one room's lighting spills on another's — and rebuilding the
/// element for each nudge would mean reopening whatever produces its frames.
/// For a camera that is a visible stall; for a screen capture on Wayland it
/// is a portal dialog.
///
/// Cloning is cheap, and every clone controls the same element. Retaining
/// one keeps only the small shared settings alive — not the element, its
/// pads, or the pipeline graph. A handle whose element has stopped still
/// accepts writes; nothing reads them, and nothing fails.
#[derive(Debug, Clone)]
pub struct ChromaKeyHandle {
    control: Arc<ChromaKeyControl>,
}

impl ChromaKeyHandle {
    pub(super) fn new(control: Arc<ChromaKeyControl>) -> Self {
        Self { control }
    }

    /// Replaces every setting at once, taking effect on the next frame the
    /// element keys.
    ///
    /// All of them together rather than one method per field: a caller
    /// changing only the threshold reads [`ChromaKeyHandle::options`],
    /// adjusts, and writes back, which is one lock instead of three and
    /// leaves no window where half a change is live.
    ///
    /// Values are not validated, exactly as they are not at construction:
    /// a negative `smoothing` is a hard key and a `threshold` outside
    /// `0.0..=1.0` keys everything or nothing, which are answers rather
    /// than errors.
    pub fn set_options(&self, options: ChromaKeyOptions) {
        self.control.set(options);
    }

    /// The settings in force, which is what a caller adjusts and writes
    /// back through [`ChromaKeyHandle::set_options`].
    pub fn options(&self) -> ChromaKeyOptions {
        self.control.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::elements::ChromaKeyMethod;

    fn options() -> ChromaKeyOptions {
        ChromaKeyOptions {
            method: ChromaKeyMethod::Green,
            threshold: 0.15,
            smoothing: 0.1,
        }
    }

    #[test]
    fn what_an_element_reads_is_what_the_handle_last_wrote() {
        let control = Arc::new(ChromaKeyControl::new(options()));
        let handle = ChromaKeyHandle::new(control.clone());

        assert_eq!(control.get().threshold, 0.15);

        handle.set_options(ChromaKeyOptions {
            method: ChromaKeyMethod::Custom(Color::new(10, 20, 30)),
            threshold: 0.4,
            smoothing: 0.0,
        });

        let live = control.get();
        assert_eq!(live.method, ChromaKeyMethod::Custom(Color::new(10, 20, 30)));
        assert_eq!(live.threshold, 0.4);
        assert_eq!(live.smoothing, 0.0);
    }

    /// The read-adjust-write path this deliberately has instead of one
    /// setter per field.
    #[test]
    fn one_field_can_be_changed_without_disturbing_the_others() {
        let control = Arc::new(ChromaKeyControl::new(options()));
        let handle = ChromaKeyHandle::new(control.clone());

        let mut adjusted = handle.options();
        adjusted.threshold = 0.5;
        handle.set_options(adjusted);

        let live = control.get();
        assert_eq!(live.threshold, 0.5);
        assert_eq!(live.method, ChromaKeyMethod::Green);
        assert_eq!(live.smoothing, 0.1, "smoothing was not part of the change");
    }

    #[test]
    fn every_clone_controls_the_same_element() {
        let control = Arc::new(ChromaKeyControl::new(options()));
        let handle = ChromaKeyHandle::new(control.clone());
        let clone = handle.clone();

        let mut adjusted = clone.options();
        adjusted.threshold = 0.9;
        clone.set_options(adjusted);

        assert_eq!(
            handle.options().threshold,
            0.9,
            "a clone must not have taken a copy of the settings"
        );
        assert_eq!(control.get().threshold, 0.9);
    }

    /// Dropping the element leaves the handle usable rather than panicking
    /// on a dead reference — see this type's own docs.
    #[test]
    fn a_handle_outliving_its_element_still_answers() {
        let control = Arc::new(ChromaKeyControl::new(options()));
        let handle = ChromaKeyHandle::new(control.clone());
        drop(control);

        let mut adjusted = handle.options();
        adjusted.threshold = 0.25;
        handle.set_options(adjusted);

        assert_eq!(handle.options().threshold, 0.25);
    }
}
