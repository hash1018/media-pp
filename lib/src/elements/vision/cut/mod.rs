//! Finding where one shot of an edited video ends and the next begins —
//! where a tracker has to start over, since nothing in the new shot is
//! where anything in the old one was.
//!
//! Each picture is made a thumbnail — the mean of its luma in 64 by 36
//! cells, and of each chroma channel in 32 by 18 — and compared with the
//! one before it. A cut is a picture far from the one before, both against
//! a floor ([`CutDetectorOptions::min_score`]) and against how far the
//! pictures just before it moved from each other
//! ([`CutDetectorOptions::threshold`]), so that a fast pan, every picture
//! far from the last, is not a string of cuts. A flash — a picture or two
//! unlike the ones on either side — is told from a cut by the pictures
//! after it ([`CutDetectorOptions::lookahead`]); a light switched or swung
//! over the same shot by its luma keeping its shape — the cells correlated
//! as they were, the brighter parts brighter — and a fade through black is
//! a cut where the picture comes back: edited video goes through black
//! between shots, and a tracker starting over where it did not costs
//! little.
//!
//! What a detector does with the thumbnails is the same wherever the
//! pictures live, so it is here; each backend's element only makes them:
//! [`SwCutDetector`] in system memory, `CudaCutDetector` on the GPU, where
//! the cells are averaged by the kernel a CUDA overlay's mosaic is, and
//! only the thumbnail comes down.

#[cfg(feature = "cuda")]
mod cuda_cut_detector;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_cut_detector;
mod sw_cut_detector;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error as ThisError;

use crate::buffer::{MediaBuffer, time_base};
use crate::elements::SceneCut;
use crate::transform::Output;

#[cfg(feature = "cuda")]
pub use cuda_cut_detector::{CudaCutDetector, CudaCutDetectorError};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal_cut_detector::{MetalCutDetector, MetalCutDetectorError};
pub use sw_cut_detector::{SwCutDetector, SwCutDetectorError};

/// The cells a thumbnail's luma is averaged in, across and down; its
/// chroma in half as many each way.
const CELLS: (u32, u32) = (64, 36);

/// How many of the latest scores a cut is weighed against.
const WINDOW: usize = 8;

/// A score this far, and this many times the average before it, is a cut
/// whatever [`CutDetectorOptions::threshold`] says: a cut in the middle of
/// a fight, where every picture is far from the last, stands only so far
/// above them.
const STRONG: (f32, f32) = (40.0, 2.0);

/// A picture whose luma averages below this, out of 255, and varies by
/// less than [`DARK_SPREAD`] across it, is dark — black, or nearly, as a
/// fade passes through.
const DARK_LUMA: f32 = 24.0;
const DARK_SPREAD: f32 = 24.0;

/// How alike in shape — the correlation of their luma cells — a picture
/// far from the last may be to it and still be no cut: a stage light
/// switched or swung, the singer brightened and the dark behind her not.
/// A concert recording had 65 of its 137 cuts so, each with its two
/// pictures correlated over 0.85, and the cut that followed them at about
/// 0; on the hand-marked films the same veto lost no cut and three false
/// ones.
const RELIT: f32 = 0.85;

/// How a cut detector tells a cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutDetectorOptions {
    /// How many times further a picture must be from the one before than
    /// the pictures before it were from each other, on average over the
    /// last eight. Higher misses cuts between alike shots; lower takes a
    /// burst of motion for one. A picture very far from the last — 40 —
    /// needs only twice the average, so that a cut in the middle of a
    /// fight is not missed.
    ///
    /// The defaults found 220 of 222 cuts hand-marked in 19 minutes of
    /// three Blender films, live action and animated, and called 14 that
    /// were none, where ffmpeg's `scdet` at its best found 191 and called
    /// 18: see `docs/benchmarks/vision/cuts.md`.
    pub threshold: f32,
    /// How far a picture must be from the one before to be a cut at all —
    /// what keeps a still shot's flicker from counting: the mean difference
    /// of their luma cells plus that of their chroma cells, each out of 255.
    pub min_score: f32,
    /// How long after a cut no other is taken.
    pub min_scene: Duration,
    /// How many pictures after a possible cut are looked at before it is
    /// called one: where one of them is like the picture before it again,
    /// it was a flash. Each is held that long, so a hardware decoder in
    /// front needs that many more surfaces.
    pub lookahead: u32,
}

impl Default for CutDetectorOptions {
    fn default() -> Self {
        Self {
            threshold: 2.5,
            min_score: 12.0,
            min_scene: Duration::from_millis(200),
            lookahead: 2,
        }
    }
}

/// Options a cut detector cannot be made with.
#[derive(Debug, Clone, PartialEq, ThisError)]
pub enum CutDetectorOptionsError {
    /// The threshold is not a finite number of at least 1.
    #[error("a cut detector's threshold must be at least 1, got {0}")]
    Threshold(f32),
    /// The floor is not a finite number from 0 to 510.
    #[error("a cut detector's min_score must be from 0 to 510, got {0}")]
    MinScore(f32),
    /// More pictures to look ahead than make sense to hold.
    #[error("a cut detector looks at most {max} pictures ahead, asked for {asked}")]
    Lookahead {
        /// What was asked for.
        asked: u32,
        /// The most it takes.
        max: u32,
    },
}

impl CutDetectorOptions {
    /// The most pictures a detector holds back.
    const MAX_LOOKAHEAD: u32 = 30;

    pub(crate) fn check(&self) -> Result<(), CutDetectorOptionsError> {
        if !self.threshold.is_finite() || self.threshold < 1.0 {
            return Err(CutDetectorOptionsError::Threshold(self.threshold));
        }
        if !self.min_score.is_finite() || !(0.0..=510.0).contains(&self.min_score) {
            return Err(CutDetectorOptionsError::MinScore(self.min_score));
        }
        if self.lookahead > Self::MAX_LOOKAHEAD {
            return Err(CutDetectorOptionsError::Lookahead {
                asked: self.lookahead,
                max: Self::MAX_LOOKAHEAD,
            });
        }
        Ok(())
    }
}

/// A picture, small: the mean of its luma in each of `cells` cells, row
/// by row, and of its two chroma channels in each of `chroma_cells`, Cb
/// and Cr side by side — each 0 to 255.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Thumbnail {
    pub(crate) cells: (u32, u32),
    pub(crate) luma: Vec<f32>,
    pub(crate) chroma_cells: (u32, u32),
    pub(crate) chroma: Vec<f32>,
}

impl Thumbnail {
    /// The cells a `width` by `height` picture's luma is averaged in, and
    /// its chroma: fewer where the picture, or its chroma plane, has fewer
    /// samples than cells, so that no cell is empty.
    pub(crate) fn cells_of(width: u32, height: u32) -> ((u32, u32), (u32, u32)) {
        let chroma = (width.div_ceil(2), height.div_ceil(2));
        (
            (CELLS.0.min(width).max(1), CELLS.1.min(height).max(1)),
            (
                (CELLS.0 / 2).min(chroma.0).max(1),
                (CELLS.1 / 2).min(chroma.1).max(1),
            ),
        )
    }

    /// Whether it is dark: a fade at its darkest, or black between shots.
    fn dark(&self) -> bool {
        let mean = self.luma.iter().sum::<f32>() / self.luma.len() as f32;
        let (low, high) = self
            .luma
            .iter()
            .fold((f32::MAX, f32::MIN), |(low, high), &v| {
                (low.min(v), high.max(v))
            });
        mean < DARK_LUMA && high - low < DARK_SPREAD
    }

    /// How alike its luma is to `other`'s in shape: their correlation,
    /// from -1 to 1, which a light brightening or dimming the picture —
    /// the whole of it or the brighter parts more — leaves near 1.
    fn correlation(&self, other: &Thumbnail) -> f32 {
        if self.cells != other.cells {
            return 0.0;
        }
        let n = self.luma.len() as f32;
        let (ma, mb) = (
            self.luma.iter().sum::<f32>() / n,
            other.luma.iter().sum::<f32>() / n,
        );
        let (mut ab, mut aa, mut bb) = (0.0f32, 0.0f32, 0.0f32);
        for (a, b) in self.luma.iter().zip(&other.luma) {
            let (a, b) = (a - ma, b - mb);
            ab += a * b;
            aa += a * a;
            bb += b * b;
        }
        if aa <= 0.0 || bb <= 0.0 {
            return 0.0;
        }
        ab / (aa * bb).sqrt()
    }

    /// How far it is from `other`: the mean difference of the luma cells,
    /// plus that of the chroma cells, each out of 255. A picture of another
    /// shape is as far as there is.
    fn distance(&self, other: &Thumbnail) -> f32 {
        if self.cells != other.cells || self.chroma_cells != other.chroma_cells {
            return 510.0;
        }
        let mean_difference = |a: &[f32], b: &[f32]| {
            a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f32>() / a.len() as f32
        };
        mean_difference(&self.luma, &other.luma) + mean_difference(&self.chroma, &other.chroma)
    }
}

/// The mean of channel `channel` of each of `cells` cells of a plane —
/// `width` by `height` samples of `channels` bytes each, rows `stride`
/// apart — into `means`, row by row: cell `c` of `n` along a side `length`
/// long spans `[c * length / n, (c + 1) * length / n)`. Summed row by row in
/// `f32` and times the reciprocal of the count, as the CUDA kernel
/// `cell_means` does it, so that the two make the same thumbnail.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cell_means(
    data: &[u8],
    stride: usize,
    width: u32,
    height: u32,
    channels: u32,
    channel: u32,
    cells: (u32, u32),
    means: &mut Vec<f32>,
) {
    let edge = |index: u32, cells: u32, length: u32| {
        (u64::from(index) * u64::from(length) / u64::from(cells)) as usize
    };
    for cy in 0..cells.1 {
        let (top, bottom) = (edge(cy, cells.1, height), edge(cy + 1, cells.1, height));
        for cx in 0..cells.0 {
            let (left, right) = (edge(cx, cells.0, width), edge(cx + 1, cells.0, width));
            let mut sum = 0.0f32;
            for row in top..bottom {
                let line = &data[row * stride..];
                for x in left..right {
                    sum += f32::from(line[x * channels as usize + channel as usize]);
                }
            }
            let count = ((right - left) * (bottom - top)) as f32;
            means.push(sum * (1.0 / count));
        }
    }
}

/// A picture waiting for the ones after it, its thumbnail, and when it is.
struct Held {
    buf: MediaBuffer,
    thumbnail: Thumbnail,
    at: Duration,
}

/// Where a stream stands as to dark pictures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dark {
    /// The last picture was not dark.
    No,
    /// The last were dark, after a picture that was not.
    After,
    /// The stream began dark, and is still.
    FromStart,
}

/// What a cut detector keeps of one stream: the pictures held for those
/// after them, and what the pictures already handed on were like.
pub(crate) struct Judge {
    name: Arc<str>,
    options: CutDetectorOptions,
    held: VecDeque<Held>,
    /// The last picture handed on.
    last: Option<Thumbnail>,
    /// The latest scores since the last cut.
    recent: VecDeque<f32>,
    dark: Dark,
    /// How many pictures after the one being handed on are a flash, to be
    /// handed on unjudged and not compared with.
    flash: usize,
    /// When the last cut was, or the stream began.
    since: Option<Duration>,
    /// Pictures seen, to time a picture that says no time.
    pictures: u64,
}

impl Judge {
    pub(crate) fn new(name: Arc<str>, options: CutDetectorOptions) -> Self {
        Self {
            name,
            options,
            held: VecDeque::new(),
            last: None,
            recent: VecDeque::new(),
            dark: Dark::No,
            flash: 0,
            since: None,
            pictures: 0,
        }
    }

    /// Takes `buf`, whose thumbnail is `thumbnail`, and hands on every
    /// picture whose pictures after it have all been seen.
    pub(crate) fn push(&mut self, buf: MediaBuffer, thumbnail: Thumbnail, out: &mut Output) {
        // When it is, by its own time, or at 30 a second where it says none.
        let at = match &buf {
            MediaBuffer::Video(frame) => frame.pts().zip(time_base(frame)),
            _ => None,
        }
        .and_then(|(pts, unit)| {
            let seconds = pts as f64 * f64::from(unit.numerator()) / f64::from(unit.denominator());
            (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds))
        })
        .unwrap_or(Duration::from_secs(self.pictures) / 30);
        self.pictures += 1;
        self.held.push_back(Held { buf, thumbnail, at });
        while self.held.len() > self.options.lookahead as usize {
            self.hand_on(out);
        }
    }

    /// Hands on every picture still held, judged by those after it there
    /// are: the end of the stream.
    pub(crate) fn finish(&mut self, out: &mut Output) {
        while !self.held.is_empty() {
            self.hand_on(out);
        }
    }

    /// Lets go of every picture held, and of what the stream was like: a
    /// seek or a flush, after which it begins again.
    pub(crate) fn clear(&mut self) {
        *self = Self::new(Arc::clone(&self.name), self.options);
    }

    /// Hands on the first picture held, carrying a [`SceneCut`] where it
    /// begins a shot.
    fn hand_on(&mut self, out: &mut Output) {
        let Some(Held { buf, thumbnail, at }) = self.held.pop_front() else {
            return;
        };
        if self.flash > 0 {
            // Part of a flash already judged: neither a cut nor what the
            // next picture is compared with.
            self.flash -= 1;
            out.push(buf);
            return;
        }
        let since = *self.since.get_or_insert(at);
        let settled = at.saturating_sub(since) >= self.options.min_scene;
        let mut cut = None;
        if thumbnail.dark() {
            if self.dark == Dark::No {
                self.dark = if self.last.is_some() {
                    Dark::After
                } else {
                    Dark::FromStart
                };
            }
        } else if self.dark != Dark::No {
            // Back from black, which edited video passes through between
            // shots alone: a cut, unless the stream began there.
            let after = std::mem::replace(&mut self.dark, Dark::No) == Dark::After;
            if after && settled {
                let score = self
                    .last
                    .as_ref()
                    .map_or(0.0, |last| thumbnail.distance(last));
                cut = Some(score);
            }
        } else if let Some(last) = &self.last {
            let score = thumbnail.distance(last);
            let average = (!self.recent.is_empty())
                .then(|| self.recent.iter().sum::<f32>() / self.recent.len() as f32);
            // Far from the last, and further than the pictures before had
            // been from each other: with none before to weigh it by — the
            // stream's second picture — it is not called a cut.
            let (strong, strong_ratio) = STRONG;
            let stands_out = score >= self.options.min_score
                && average.is_some_and(|average| {
                    score >= self.options.threshold * average
                        || (score >= strong && score >= strong_ratio * average)
                });
            // The same shot under another light: the brighter parts brighter
            // or darker, the picture's shape as it was.
            let relit = thumbnail.correlation(last) >= RELIT;
            if stands_out && settled && !relit {
                // A flash: one of the pictures after it is like the one
                // before it again. It, and those up to that one, are let by.
                let back = self.held.iter().position(|after| {
                    !after.thumbnail.dark()
                        && after.thumbnail.distance(last) < self.options.min_score
                });
                // A cut a picture or two on, further from the one before it
                // than this is from the last: this one was the blur running
                // up to it, and the cut is that one.
                let mut before = &thumbnail;
                let later = self.held.iter().any(|after| {
                    let further =
                        !after.thumbnail.dark() && after.thumbnail.distance(before) > score;
                    before = &after.thumbnail;
                    further
                });
                match back {
                    Some(back) => {
                        self.flash = back;
                        out.push(buf);
                        return;
                    }
                    None if later => {
                        self.last = Some(thumbnail);
                        out.push(buf);
                        return;
                    }
                    None => cut = Some(score),
                }
            }
            self.recent.push_back(score);
            if self.recent.len() > WINDOW {
                self.recent.pop_front();
            }
        }
        self.last = Some(thumbnail);
        let buf = match cut {
            Some(score) => {
                self.since = Some(at);
                self.recent.clear();
                let metadata = buf.metadata().cloned().unwrap_or_default().with(SceneCut {
                    detector: Arc::clone(&self.name),
                    score,
                });
                buf.with_metadata(metadata)
            }
            None => buf,
        };
        out.push(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::set_time_base;
    use crate::ffmpeg;

    /// A thumbnail every cell of which is `luma`, chroma grey.
    fn flat(luma: f32) -> Thumbnail {
        Thumbnail {
            cells: (4, 2),
            luma: vec![luma; 8],
            chroma_cells: (2, 1),
            chroma: vec![128.0; 4],
        }
    }

    fn picture(index: i64) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::GRAY8, 4, 2);
        frame.set_pts(Some(index));
        set_time_base(&mut frame, ffmpeg::Rational(1, 30));
        MediaBuffer::video(frame)
    }

    /// The pictures a judge hands on for `thumbnails`, one a picture at 30
    /// a second, and which carry a cut.
    fn judged(options: CutDetectorOptions, thumbnails: Vec<Thumbnail>) -> Vec<usize> {
        let mut judge = Judge::new("cuts".into(), options);
        let mut out = Output::default();
        for (index, thumbnail) in thumbnails.into_iter().enumerate() {
            judge.push(picture(index as i64), thumbnail, &mut out);
        }
        judge.finish(&mut out);
        let made = out.into_buffers();
        assert!(
            made.iter()
                .enumerate()
                .all(|(index, buf)| matches!(buf, MediaBuffer::Video(frame) if frame.pts() == Some(index as i64))),
            "every picture, in order"
        );
        made.iter()
            .enumerate()
            .filter(|(_, buf)| {
                buf.metadata()
                    .is_some_and(|metadata| metadata.get::<SceneCut>().is_some())
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn quick() -> CutDetectorOptions {
        CutDetectorOptions {
            min_scene: Duration::ZERO,
            ..CutDetectorOptions::default()
        }
    }

    /// A shot changing all at once is a cut, on its first picture, and the
    /// first picture of the stream is not.
    #[test]
    fn a_change_all_at_once_is_a_cut_on_the_new_shots_first_picture() {
        let shots = [vec![flat(60.0); 10], vec![flat(180.0); 10]].concat();
        assert_eq!(judged(quick(), shots), vec![10]);
    }

    /// The blur running up to a cut — a picture far from the last, and the
    /// next further still — is not the cut: the furthest is.
    #[test]
    fn a_cut_after_the_blur_running_up_to_it_is_on_the_furthest_picture() {
        let shots = [
            vec![flat(60.0); 10],
            vec![flat(100.0)],
            vec![flat(220.0); 10],
        ]
        .concat();
        assert_eq!(judged(quick(), shots), vec![11]);
    }

    /// In the middle of a fight, every picture far from the last, a cut
    /// twice as far as the average is one when it is far enough by itself.
    #[test]
    fn a_strong_cut_in_heavy_motion_is_a_cut() {
        // Steps of 22 up and down, then one of 50: short of 2.5 times their
        // average, but past 40 and twice it.
        let mut thumbnails: Vec<Thumbnail> = (0..12)
            .map(|n| flat(if n % 2 == 0 { 80.0 } else { 102.0 }))
            .collect();
        thumbnails.extend([flat(152.0), flat(130.0), flat(152.0), flat(130.0)]);
        assert_eq!(judged(quick(), thumbnails), vec![12]);
        // The same step out of steps of 30 is not twice their average.
        let mut calmer: Vec<Thumbnail> = (0..12)
            .map(|n| flat(if n % 2 == 0 { 72.0 } else { 102.0 }))
            .collect();
        calmer.extend([flat(152.0), flat(122.0), flat(152.0), flat(122.0)]);
        assert!(judged(quick(), calmer).is_empty());
    }

    /// A pan — every picture as far from the last as the one before was —
    /// is no cut, though each step is past the floor; a step three times
    /// the others' is.
    #[test]
    fn steady_motion_is_no_cut_and_a_jump_out_of_it_is() {
        let mut thumbnails: Vec<Thumbnail> =
            (0..12).map(|n| flat(40.0 + n as f32 * 16.0)).collect();
        assert!(judged(quick(), thumbnails.clone()).is_empty());
        thumbnails.extend((0..5).map(|n| flat(30.0 + n as f32 * 16.0)));
        assert_eq!(judged(quick(), thumbnails), vec![12]);
    }

    /// A light switched on over the same shot — the bright half brighter,
    /// the dark half as it was — is no cut, though as far from the last as
    /// a cut; the same change to another shape is one.
    #[test]
    fn a_light_over_the_same_shot_is_no_cut() {
        let shaped = |left: f32, right: f32| Thumbnail {
            cells: (4, 2),
            luma: vec![left, left, right, right, left, left, right, right],
            chroma_cells: (2, 1),
            chroma: vec![128.0; 4],
        };
        let mut lit = vec![shaped(40.0, 120.0); 10];
        lit.extend(vec![shaped(40.0, 200.0); 10]);
        assert!(judged(quick(), lit).is_empty());

        let mut cut = vec![shaped(40.0, 120.0); 10];
        cut.extend(vec![shaped(200.0, 40.0); 10]);
        assert_eq!(judged(quick(), cut), vec![10]);
    }

    /// A flash — a picture or two unlike those on either side — is no cut,
    /// nor is the shot it goes back to; the same change kept is a cut.
    #[test]
    fn a_flash_is_no_cut() {
        for long in [1, 2] {
            let mut flash = vec![flat(60.0); 12];
            for picture in &mut flash[5..5 + long] {
                *picture = flat(230.0);
            }
            assert!(judged(quick(), flash).is_empty(), "{long} pictures");
        }
        let kept = [vec![flat(60.0); 5], vec![flat(230.0); 7]].concat();
        assert_eq!(judged(quick(), kept), vec![5]);
    }

    /// Through black: no cut going dark or while dark, one where the
    /// picture comes back — whatever comes back — and none for a stream
    /// that begins dark.
    #[test]
    fn a_fade_through_black_is_a_cut_where_the_picture_comes_back() {
        let fade = |to: f32| {
            let down = (0..6).map(|n| flat(150.0 - n as f32 * 25.0));
            let dark = (0..4).map(|_| flat(16.0));
            let up = (1..=6).map(move |n| flat(16.0 + n as f32 * (to - 16.0) / 6.0));
            down.chain(dark).chain(up).collect::<Vec<_>>()
        };
        assert_eq!(judged(quick(), fade(220.0)), vec![10]);
        assert_eq!(judged(quick(), fade(150.0)), vec![10]);
        let from_black = [vec![flat(16.0); 4], vec![flat(150.0); 4]].concat();
        assert!(judged(quick(), from_black).is_empty());
    }

    /// No cut comes within `min_scene` of the last, or of the start.
    #[test]
    fn no_cut_comes_sooner_than_min_scene() {
        let options = CutDetectorOptions {
            min_scene: Duration::from_millis(300),
            ..CutDetectorOptions::default()
        };
        let shots = [
            vec![flat(60.0); 5],
            vec![flat(180.0); 5],
            vec![flat(60.0); 12],
            vec![flat(180.0); 5],
        ]
        .concat();
        assert_eq!(judged(options, shots), vec![10, 22]);
    }

    /// A seek's leftovers are let go of, and what follows starts afresh.
    #[test]
    fn a_cleared_judge_holds_nothing_and_starts_again() {
        let mut judge = Judge::new("cuts".into(), quick());
        let mut out = Output::default();
        judge.push(picture(0), flat(60.0), &mut out);
        judge.push(picture(1), flat(60.0), &mut out);
        judge.clear();
        judge.finish(&mut out);
        assert!(out.into_buffers().is_empty());
        let mut out = Output::default();
        judge.push(picture(5), flat(200.0), &mut out);
        judge.finish(&mut out);
        let made = out.into_buffers();
        assert_eq!(made.len(), 1);
        assert!(
            made[0].metadata().is_none(),
            "the first picture after is no cut"
        );
    }

    /// Cells are as near equal as whole samples allow, summed in order.
    #[test]
    fn a_cell_is_the_mean_of_its_samples() {
        // 5 by 2, two channels; channel 1 of each sample is ten times x.
        let mut data = Vec::new();
        for _ in 0..2 {
            for x in 0..5u8 {
                data.extend([0, x * 10]);
            }
        }
        let mut means = Vec::new();
        cell_means(&data, 10, 5, 2, 2, 1, (2, 1), &mut means);
        // [0, 2) and [2, 5).
        assert_eq!(means, vec![5.0, 30.0]);
    }

    #[test]
    fn options_that_cannot_judge_are_refused() {
        let refused = |options: CutDetectorOptions| options.check().unwrap_err();
        let defaults = CutDetectorOptions::default();
        assert!(defaults.check().is_ok());
        assert_eq!(
            refused(CutDetectorOptions {
                threshold: 0.5,
                ..defaults
            }),
            CutDetectorOptionsError::Threshold(0.5)
        );
        assert!(matches!(
            refused(CutDetectorOptions {
                min_score: f32::NAN,
                ..defaults
            }),
            CutDetectorOptionsError::MinScore(_)
        ));
        assert!(matches!(
            refused(CutDetectorOptions {
                lookahead: 31,
                ..defaults
            }),
            CutDetectorOptionsError::Lookahead { asked: 31, .. }
        ));
    }
}
