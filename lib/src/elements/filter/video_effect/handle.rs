//! Runtime control shared by every video-effect backend.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arc_swap::ArcSwap;

use super::options::VideoEffect;

/// The live effect one element reads and its [`VideoEffectHandle`] writes.
///
/// The whole [`VideoEffect`] in one slot, for the reason
/// `ChromaKeyControl` keeps a whole `ChromaKeyOptions` in one: a contrast
/// and a brightness are one correction in two parts, and a frame drawn with
/// the new one against the old other is a frame nobody asked for. Read once
/// per frame, never per pixel.
#[derive(Debug)]
pub(super) struct VideoEffectControl {
    effect: ArcSwap<VideoEffect>,
    /// Whether to apply it at all — kept apart from the effect, as a chroma
    /// key's is, so turning it off costs it nothing it was tuned to.
    enabled: AtomicBool,
}

impl VideoEffectControl {
    pub(super) fn new(effect: VideoEffect) -> Self {
        Self {
            effect: ArcSwap::from_pointee(effect),
            enabled: AtomicBool::new(true),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// A consistent copy of the effect, for one frame to be drawn with.
    pub(super) fn get(&self) -> VideoEffect {
        **self.effect.load()
    }

    fn set(&self, effect: VideoEffect) {
        self.effect.store(Arc::new(effect));
    }
}

/// Thread-safe runtime control for a video-effect element, from any
/// backend.
///
/// What it is for is a slider: an effect is tuned by watching the picture,
/// and rebuilding the element for each nudge would mean reopening whatever
/// produces its frames.
///
/// Cloning is cheap, and every clone controls the same element. Retaining
/// one keeps only the small shared setting alive — not the element, its
/// pads, or the pipeline graph. A handle whose element has stopped still
/// accepts writes; nothing reads them, and nothing fails.
#[derive(Debug, Clone)]
pub struct VideoEffectHandle {
    control: Arc<VideoEffectControl>,
}

impl VideoEffectHandle {
    pub(super) fn new(control: Arc<VideoEffectControl>) -> Self {
        Self { control }
    }

    /// Replaces the effect, taking effect on the next frame.
    ///
    /// All of it at once, and it may be the other kind of effect: a caller
    /// changing one field reads [`VideoEffectHandle::effect`], adjusts and
    /// writes back, which leaves no window where half a change is live.
    pub fn set_effect(&self, effect: VideoEffect) {
        self.control.set(effect);
    }

    /// The effect in force.
    pub fn effect(&self) -> VideoEffect {
        self.control.get()
    }

    /// Turns the effect on or off without disturbing its settings, taking
    /// effect on the next frame.
    ///
    /// An element that is off is still in the graph and hands each frame
    /// straight through — the same picture, not a copy of it — so a
    /// checkbox beside it costs nothing.
    pub fn set_enabled(&self, enabled: bool) {
        self.control.set_enabled(enabled);
    }

    /// Whether the effect is on. See [`VideoEffectHandle::set_enabled`].
    pub fn enabled(&self) -> bool {
        self.control.enabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::{ColorCorrection, LumaKey};

    #[test]
    fn what_an_element_reads_is_what_the_handle_last_wrote() {
        let control = Arc::new(VideoEffectControl::new(VideoEffect::ColorCorrection(
            ColorCorrection::default(),
        )));
        let handle = VideoEffectHandle::new(control.clone());
        let key = VideoEffect::LumaKey(LumaKey {
            min: 0.3,
            ..LumaKey::default()
        });

        handle.set_effect(key);

        assert_eq!(
            control.get(),
            key,
            "switching kind is a write like any other"
        );
        assert_eq!(
            handle.clone().effect(),
            key,
            "a clone sees the same element"
        );
    }

    #[test]
    fn enabled_is_what_a_new_element_is_and_stays_apart_from_the_effect() {
        let effect = VideoEffect::ColorCorrection(ColorCorrection {
            brightness: 0.4,
            ..ColorCorrection::default()
        });
        let control = Arc::new(VideoEffectControl::new(effect));
        let handle = VideoEffectHandle::new(control.clone());
        assert!(control.enabled());

        handle.set_enabled(false);

        assert!(!control.enabled());
        assert_eq!(
            control.get(),
            effect,
            "turning it off kept what it was tuned to"
        );
    }
}
