//! Where playback is, for the audio renderers, when it runs at a rate: a
//! record of which media each sample handed to a device stands for.
//!
//! A renderer stretches its sound to the playback rate with
//! [`crate::elements::filter::audio::stretcher`], so each sample the device
//! plays stands for `rate` samples' worth of media, and it says where
//! playback is from what the device has played. Both renderers do it with
//! this, so neither backend has a rate of its own to get right.

// Built for its own tests too, where no audio renderer uses it.
#![cfg_attr(
    not(any(
        all(target_os = "windows", feature = "wasapi-renderer"),
        all(target_os = "linux", feature = "pipewire-audio-renderer")
    )),
    allow(dead_code)
)]

use std::{
    collections::VecDeque,
    sync::{Mutex, MutexGuard},
};

/// Which media the samples handed to a device stand for, from the first on:
/// each span played at a rate covers that many times its own length of
/// media. Where playback is, is where the samples the device has played
/// reach.
///
/// Kept in samples, not nanoseconds, so a long stretch at one rate adds up
/// exactly. Spans the device has played through are forgotten as a
/// position is read past them.
pub(crate) struct PlayedMedia {
    sample_rate: u32,
    /// `(first sample, media there, rate)`, oldest first. Behind a lock so
    /// that reading a position, which a renderer does from `&self`, can
    /// forget the spans played through.
    spans: Mutex<VecDeque<(u64, i64, f64)>>,
    /// Samples handed over so far.
    handed: u64,
}

impl PlayedMedia {
    /// Nothing handed over yet, the first sample to be at `media_ns`.
    pub(crate) fn new(sample_rate: u32, media_ns: i64) -> Self {
        Self {
            sample_rate: sample_rate.max(1),
            spans: Mutex::new(VecDeque::from([(0, media_ns, 1.0)])),
            handed: 0,
        }
    }

    fn spans(&self) -> MutexGuard<'_, VecDeque<(u64, i64, f64)>> {
        self.spans
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// `samples` more handed over, played at `rate`.
    pub(crate) fn push(&mut self, samples: u64, rate: f64) {
        let media_ns = self.media_at(self.handed);
        let mut spans = self.spans();
        if spans
            .back()
            .is_none_or(|&(_, _, last_rate)| last_rate != rate)
        {
            spans.push_back((self.handed, media_ns, rate));
        }
        drop(spans);
        self.handed += samples;
    }

    /// The media reached once the device has played `played` samples, at
    /// most as far as what it has been handed.
    pub(crate) fn media_at(&self, played: u64) -> i64 {
        self.at(&self.spans(), played)
    }

    fn at(&self, spans: &VecDeque<(u64, i64, f64)>, played: u64) -> i64 {
        let played = played.min(self.handed);
        // The first span starts at zero and is never forgotten while it is
        // the only one, so there is always one to find.
        let (start, media_ns, rate) = spans
            .iter()
            .rev()
            .find(|(start, _, _)| *start <= played)
            .copied()
            .unwrap_or((0, 0, 1.0));
        let ns = (played - start) as f64 * 1e9 / f64::from(self.sample_rate);
        media_ns.saturating_add((ns * rate) as i64)
    }

    /// As [`Self::media_at`], forgetting the spans played through: what a
    /// renderer calls as it publishes where playback is.
    pub(crate) fn played(&self, played: u64) -> i64 {
        let mut spans = self.spans();
        while spans.len() > 1 && spans[1].0 <= played {
            spans.pop_front();
        }
        self.at(&spans, played)
    }

    /// How far the media handed over reaches.
    pub(crate) fn handed_until(&self) -> i64 {
        self.media_at(self.handed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;

    /// Where playback is, from what the device has played: at the rate each
    /// span was handed over at.
    #[test]
    fn played_samples_reach_the_media_their_rate_says() {
        let mut map = PlayedMedia::new(RATE, 5_000_000_000);
        map.push(u64::from(RATE), 1.0);
        map.push(u64::from(RATE), 2.0);
        map.push(u64::from(RATE) / 2, 0.5);
        assert_eq!(map.media_at(0), 5_000_000_000);
        assert_eq!(map.media_at(u64::from(RATE)), 6_000_000_000);
        assert_eq!(map.media_at(u64::from(RATE) * 3 / 2), 7_000_000_000);
        assert_eq!(map.handed_until(), 8_250_000_000);
        assert_eq!(
            map.played(u64::from(RATE) * 10),
            8_250_000_000,
            "no further than what was handed over"
        );
        assert_eq!(map.spans().len(), 1, "what was played through is forgotten");
    }
}
