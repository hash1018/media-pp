//! What a window renderer takes to put a picture on the screen once it has
//! it, learned now and then and published to the pipeline's playback clock —
//! see `PlaybackClock::presentation_delay`. How each renderer learns it is
//! its own: a present waited for on Vulkan, DXGI's frame statistics on
//! Windows, or an estimate from the display's refresh where neither says.

use std::{collections::VecDeque, time::Duration};

/// The measurements, the schedule they are taken on, and what was published.
///
/// A measurement can cost the renderer — Vulkan holds it until its picture
/// is on the screen — so it is taken often only while little is known, and
/// rarely after: the first frame after a swap chain is made, every tenth
/// until there are five, and every hundred and twentieth after that — a
/// couple of seconds, to follow a display that changes under it. What is
/// published is the median of the last few, so one slow present does not
/// move it.
#[derive(Default)]
pub(crate) struct PresentationDelay {
    samples: VecDeque<Duration>,
    /// Frames presented since the last measurement was due.
    since: u32,
    /// The playback clock's place for it, where there is a pipeline.
    pub(crate) registration: Option<crate::playback_clock::PresenterRegistration>,
    /// What was last published, where it came from, and whether that is
    /// news for the log.
    published: Option<Duration>,
    source: String,
    changed: bool,
}

impl PresentationDelay {
    const KEPT: usize = 9;

    /// Whether the frame about to be presented is one to measure.
    pub(crate) fn due(&mut self) -> bool {
        self.since = self.since.saturating_add(1);
        let every = if self.samples.len() < 5 { 10 } else { 120 };
        if self.samples.is_empty() || self.since >= every {
            self.since = 0;
            true
        } else {
            false
        }
    }

    /// One measurement: a picture reached the screen `delay` after it was
    /// handed over.
    pub(crate) fn record(&mut self, delay: Duration) {
        if self.samples.len() == Self::KEPT {
            self.samples.pop_front();
        }
        self.samples.push_back(delay);
        let mut sorted: Vec<Duration> = self.samples.iter().copied().collect();
        sorted.sort();
        self.publish(sorted[sorted.len() / 2], "measured".into());
    }

    /// Whether anything has been measured since the swap chain was made.
    pub(crate) fn measured(&self) -> bool {
        !self.samples.is_empty()
    }

    /// Publishes an estimate where nothing can be measured: `delay`, for a
    /// display refreshing every `interval`.
    pub(crate) fn estimate(&mut self, delay: Duration, interval: Duration) {
        let hertz = 1.0 / interval.as_secs_f64();
        self.publish(
            delay,
            format!("estimated as two refreshes of a {hertz:.0} Hz display"),
        );
    }

    fn publish(&mut self, delay: Duration, source: String) {
        if let Some(registration) = &self.registration {
            registration.publish(delay);
        }
        // Told once it moves by a millisecond, not on every sample.
        if self
            .published
            .is_none_or(|last| last.abs_diff(delay) >= Duration::from_millis(1))
        {
            self.published = Some(delay);
            self.source = source;
            self.changed = true;
        }
    }

    /// Forgets what was measured, for a swap chain that may present
    /// differently — full screen may skip the compositor.
    pub(crate) fn restart(&mut self) {
        self.samples.clear();
        self.since = 0;
    }

    /// What was published and where it came from, once each time it moves —
    /// for the renderer's log.
    pub(crate) fn take_change(&mut self) -> Option<(Duration, &str)> {
        let delay = self
            .published
            .filter(|_| std::mem::take(&mut self.changed))?;
        Some((delay, &self.source))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{clock::Clock, playback_clock::PlaybackClock};

    fn ms(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    /// The first frame is measured, then every tenth until there are five
    /// measurements, then every hundred and twentieth.
    #[test]
    fn measurements_thin_out_once_there_are_enough() {
        let mut delay = PresentationDelay::default();
        assert!(delay.due(), "the first frame");
        delay.record(ms(20));
        let next = (1..=10).find(|_| delay.due());
        assert_eq!(next, Some(10));
        for _ in 0..4 {
            delay.record(ms(20));
        }
        let next = (1..=200).find(|_| delay.due());
        assert_eq!(next, Some(120));
    }

    /// The median is published, so one slow present does not move it, and
    /// the log hears of it only when it moves by a millisecond.
    #[test]
    fn the_median_is_published_to_the_clock() {
        let clock = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        let mut delay = PresentationDelay {
            registration: Some(clock.register_presenter()),
            ..Default::default()
        };
        for sample in [20, 21, 90] {
            delay.record(ms(sample));
        }
        assert_eq!(clock.presentation_delay(), ms(21));
        assert!(delay.take_change().is_some());
        delay.record(ms(21));
        assert_eq!(clock.presentation_delay(), ms(21));
        assert!(delay.take_change().is_none(), "nothing moved");
        drop(delay);
        assert_eq!(clock.presentation_delay(), Duration::ZERO);
    }
}
