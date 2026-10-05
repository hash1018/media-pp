//! A correlation filter that follows one object by how it looks: MOSSE
//! (Bolme et al., 2010), with the search tried at three scales so that the
//! box grows and shrinks with the object.
//!
//! It learns, from the pixels around the object, the filter whose
//! correlation with that neighbourhood peaks sharply where the object is;
//! on the next picture it correlates the filter with the neighbourhood
//! around where the object is expected, and the peak is where the object
//! went. How sharp the peak stands out of the rest — the peak-to-sidelobe
//! ratio — says whether to believe it. Correlating over every shift at once
//! is a product of Fourier transforms, which is what makes it fast.

use super::fft::{Complex, fft2};
use super::luma::{Luma, Region};

/// The side of the square a neighbourhood is sampled to.
pub(super) const SIZE: usize = 64;
/// How much larger than the box the neighbourhood is, each way: room for
/// the object to have moved, and its surroundings to tell it by.
const PADDING: f64 = 2.0;
/// The smallest neighbourhood side, in pixels, so that a tiny box still
/// has some picture around it.
const MIN_WINDOW: f64 = 24.0;
/// How sharp the wanted peak is, in samples.
const SIGMA: f32 = 2.0;
/// How much of each new look a filter takes in: Bolme's rate.
pub(super) const LEARNING_RATE: f32 = 0.125;
/// Keeps the filter's division away from frequencies with no energy.
const REGULARISER: f32 = 0.01;
/// The scales tried around the last size.
const SCALES: [f64; 3] = [1.0 / 1.05, 1.0, 1.05];
/// The half-side of the area around the peak left out of the sidelobe.
const PEAK_EXCLUSION: isize = 5;

/// Where the filter found its object, and how sure it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Found {
    /// The box, by top-left corner and size, in pixels.
    pub(super) tlwh: [f64; 4],
    /// The peak-to-sidelobe ratio: under about 5 is no object.
    pub(super) psr: f32,
}

/// One object's filter: the running numerator and denominator of
/// `H* = Σ G·F̄ / Σ F·F̄`.
#[derive(Debug, Clone)]
pub(super) struct Dcf {
    numerator: Vec<Complex>,
    denominator: Vec<Complex>,
    /// The box size the filter was last trained at, in pixels.
    size: (f64, f64),
}

/// The wanted response: a Gaussian peak at the centre, transformed.
fn target() -> &'static [Complex] {
    static TARGET: std::sync::OnceLock<Vec<Complex>> = std::sync::OnceLock::new();
    TARGET.get_or_init(|| {
        let centre = (SIZE / 2) as f32;
        let mut g: Vec<Complex> = (0..SIZE * SIZE)
            .map(|i| {
                let (x, y) = ((i % SIZE) as f32 - centre, (i / SIZE) as f32 - centre);
                Complex::new((-(x * x + y * y) / (2.0 * SIGMA * SIGMA)).exp(), 0.0)
            })
            .collect();
        fft2(&mut g, SIZE, false);
        g
    })
}

/// A Hann window, which fades the neighbourhood's edges out so that the
/// transform does not see them as a sharp border.
fn window() -> &'static [f32] {
    static WINDOW: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    WINDOW.get_or_init(|| {
        let hann = |i: usize| {
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (SIZE - 1) as f32).cos()
        };
        (0..SIZE * SIZE)
            .map(|i| hann(i % SIZE) * hann(i / SIZE))
            .collect()
    })
}

/// The window around a box of `size`, in pixels.
fn window_of(size: (f64, f64)) -> (f64, f64) {
    (
        (size.0 * PADDING).max(MIN_WINDOW),
        (size.1 * PADDING).max(MIN_WINDOW),
    )
}

/// `region` sampled at `SIZE` by `SIZE` across `width` by `height` around
/// `centre`, made ready to correlate — logarithm, zero mean, unit energy,
/// window — and transformed.
fn features(region: &Region, centre: (f64, f64), width: f64, height: f64) -> Vec<Complex> {
    let mut patch: Vec<f32> = (0..SIZE * SIZE)
        .map(|i| {
            let (u, v) = ((i % SIZE) as f64, (i / SIZE) as f64);
            let x = centre.0 + ((u + 0.5) / SIZE as f64 - 0.5) * width;
            let y = centre.1 + ((v + 0.5) / SIZE as f64 - 0.5) * height;
            (region.at(x, y) + 1.0).ln()
        })
        .collect();
    let mean = patch.iter().sum::<f32>() / patch.len() as f32;
    let energy = patch
        .iter()
        .map(|p| (p - mean) * (p - mean))
        .sum::<f32>()
        .sqrt();
    let scale = if energy > 1e-6 { 1.0 / energy } else { 0.0 };
    let mut transformed: Vec<Complex> = patch
        .iter_mut()
        .zip(window())
        .map(|(p, w)| Complex::new((*p - mean) * scale * w, 0.0))
        .collect();
    fft2(&mut transformed, SIZE, false);
    transformed
}

/// The centre of a box by top-left corner and size.
fn centre_of(tlwh: [f64; 4]) -> (f64, f64) {
    (tlwh[0] + tlwh[2] / 2.0, tlwh[1] + tlwh[3] / 2.0)
}

/// The response's highest point, to a fraction of a sample by a parabola
/// through it and its neighbours either way, and its peak-to-sidelobe
/// ratio.
fn peak(response: &[f32]) -> ((f32, f32), f32, f32) {
    let (index, &top) = response
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a response");
    let (px, py) = ((index % SIZE) as isize, (index / SIZE) as isize);
    let at = |x: isize, y: isize| {
        response
            [(y.rem_euclid(SIZE as isize) as usize) * SIZE + x.rem_euclid(SIZE as isize) as usize]
    };
    let offset = |before: f32, after: f32| {
        let curve = before - 2.0 * top + after;
        if curve.abs() < 1e-12 {
            0.0
        } else {
            0.5 * (before - after) / curve
        }
    };
    let fx = px as f32 + offset(at(px - 1, py), at(px + 1, py));
    let fy = py as f32 + offset(at(px, py - 1), at(px, py + 1));

    let (mut sum, mut squares, mut count) = (0.0f32, 0.0f32, 0usize);
    for y in 0..SIZE as isize {
        for x in 0..SIZE as isize {
            let near = |a: isize, b: isize| {
                let d = (a - b).rem_euclid(SIZE as isize);
                d.min(SIZE as isize - d) <= PEAK_EXCLUSION
            };
            if near(x, px) && near(y, py) {
                continue;
            }
            let value = at(x, y);
            sum += value;
            squares += value * value;
            count += 1;
        }
    }
    let mean = sum / count as f32;
    let deviation = (squares / count as f32 - mean * mean).max(1e-12).sqrt();
    ((fx, fy), top, (top - mean) / deviation)
}

impl Dcf {
    /// A filter for the object in box `tlwh` of the picture `luma` reads;
    /// `None` where none of it is in the picture.
    pub(super) fn new(luma: &mut dyn Luma, tlwh: [f64; 4]) -> Option<Self> {
        let size = (tlwh[2], tlwh[3]);
        let mut dcf = Self {
            numerator: vec![Complex::default(); SIZE * SIZE],
            denominator: vec![Complex::default(); SIZE * SIZE],
            size,
        };
        // Learned from the box as seen and a little nearer and further, so
        // that a first look is not all the filter knows of a change of size.
        let centre = centre_of(tlwh);
        let (width, height) = window_of(size);
        let region = luma.around(centre, width * 1.1, height * 1.1)?;
        let looks = [0.95, 1.0, 1.05];
        for scale in looks {
            let f = features(&region, centre, width * scale, height * scale);
            for (i, (g, f)) in target().iter().zip(&f).enumerate() {
                dcf.numerator[i] = dcf.numerator[i] + *g * f.conj();
                dcf.denominator[i] = dcf.denominator[i] + *f * f.conj();
            }
        }
        let share = 1.0 / looks.len() as f32;
        for (a, b) in dcf.numerator.iter_mut().zip(dcf.denominator.iter_mut()) {
            *a = a.scale(share);
            *b = b.scale(share);
        }
        Some(dcf)
    }

    /// Takes in how the object looks in box `tlwh`, at `rate`.
    pub(super) fn learn(&mut self, luma: &mut dyn Luma, tlwh: [f64; 4], rate: f32) {
        let centre = centre_of(tlwh);
        let (width, height) = window_of((tlwh[2], tlwh[3]));
        let Some(region) = luma.around(centre, width, height) else {
            return;
        };
        let f = features(&region, centre, width, height);
        for (i, (g, f)) in target().iter().zip(&f).enumerate() {
            self.numerator[i] = self.numerator[i].scale(1.0 - rate) + (*g * f.conj()).scale(rate);
            self.denominator[i] =
                self.denominator[i].scale(1.0 - rate) + (*f * f.conj()).scale(rate);
        }
        self.size = (tlwh[2], tlwh[3]);
    }

    /// Looks for the object around `centre`, where it is expected, at the
    /// size it was last seen at and a little nearer and further: where it
    /// is, at the scale it answers best to.
    pub(super) fn find(&self, luma: &mut dyn Luma, centre: (f64, f64)) -> Option<Found> {
        let (width, height) = window_of(self.size);
        let largest = SCALES[SCALES.len() - 1];
        let region = luma.around(centre, width * largest, height * largest)?;
        let mut best: Option<(f32, Found)> = None;
        for scale in SCALES {
            let (w, h) = (width * scale, height * scale);
            let mut response = features(&region, centre, w, h);
            for (i, f) in response.iter_mut().enumerate() {
                let filter =
                    self.numerator[i] / (self.denominator[i] + Complex::new(REGULARISER, 0.0));
                *f = filter * *f;
            }
            fft2(&mut response, SIZE, true);
            let real: Vec<f32> = response.iter().map(|c| c.re).collect();
            let ((fx, fy), top, psr) = peak(&real);
            let half = (SIZE / 2) as f32;
            let shift = |at: f32| {
                let d = at - half;
                if d > half { d - SIZE as f32 } else { d }
            };
            let moved = (
                f64::from(shift(fx)) * w / SIZE as f64,
                f64::from(shift(fy)) * h / SIZE as f64,
            );
            let size = (self.size.0 * scale, self.size.1 * scale);
            let found = Found {
                tlwh: [
                    centre.0 + moved.0 - size.0 / 2.0,
                    centre.1 + moved.1 - size.1 / 2.0,
                    size.0,
                    size.1,
                ],
                psr,
            };
            if best.is_none_or(|(value, _)| top > value) {
                best = Some((top, found));
            }
        }
        best.map(|(_, found)| found)
    }
}

#[cfg(test)]
mod tests {
    use super::super::luma::{Area, Luma, Region};
    use super::*;

    /// A picture of noise with a brighter textured block on it, the block
    /// where it is put.
    struct Scene {
        size: (u32, u32),
        block: [f64; 4],
    }

    fn noise(x: u32, y: u32, seed: u32) -> f32 {
        let mut h = x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263) ^ seed;
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        ((h ^ (h >> 16)) & 0xff) as f32
    }

    impl Luma for Scene {
        fn size(&self) -> (u32, u32) {
            self.size
        }

        fn read(&mut self, area: Area) -> Option<Region> {
            let [bx, by, bw, bh] = self.block;
            let pixels = (area.y..area.y + area.height)
                .flat_map(|y| (area.x..area.x + area.width).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let (fx, fy) = (f64::from(x), f64::from(y));
                    if fx >= bx && fx < bx + bw && fy >= by && fy < by + bh {
                        // The block's own texture moves with it.
                        let (u, v) = ((fx - bx) as u32, (fy - by) as u32);
                        100.0 + noise(u / 3, v / 3, 7) * 0.6
                    } else {
                        noise(x, y, 1) * 0.3
                    }
                })
                .collect();
            Some(Region { area, pixels })
        }
    }

    /// The block moves 5 pixels right and 3 down; the filter learned where
    /// it was finds it where it went, sure of it.
    #[test]
    fn a_moved_object_is_found_where_it_went() {
        let block = [100.0, 80.0, 40.0, 60.0];
        let mut scene = Scene {
            size: (320, 240),
            block,
        };
        let dcf = Dcf::new(&mut scene, block).unwrap();
        scene.block = [105.0, 83.0, 40.0, 60.0];
        let found = dcf.find(&mut scene, (120.0, 110.0)).unwrap();
        assert!((found.tlwh[0] - 105.0).abs() < 1.5, "{found:?}");
        assert!((found.tlwh[1] - 83.0).abs() < 1.5, "{found:?}");
        assert!(found.psr > 8.0, "{found:?}");
    }

    /// Followed over many pictures with what it learns on the way, a block
    /// turning a corner — which a constant velocity would overshoot — is
    /// kept.
    #[test]
    fn an_object_turning_is_followed() {
        let block = [60.0, 60.0, 40.0, 60.0];
        let mut scene = Scene {
            size: (400, 300),
            block,
        };
        let mut dcf = Dcf::new(&mut scene, block).unwrap();
        let mut expected = super::centre_of(scene.block);
        for step in 1..=40 {
            // Right for twenty pictures, then down.
            let (dx, dy) = if step <= 20 { (4.0, 0.0) } else { (0.0, 4.0) };
            scene.block[0] += dx;
            scene.block[1] += dy;
            let found = dcf.find(&mut scene, expected).unwrap();
            assert!(found.psr > 5.0, "picture {step}: {found:?}");
            dcf.learn(&mut scene, found.tlwh, LEARNING_RATE);
            expected = super::centre_of(found.tlwh);
        }
        let [x, y, _, _] = scene.block;
        assert!(
            (expected.0 - (x + 20.0)).abs() < 3.0 && (expected.1 - (y + 30.0)).abs() < 3.0,
            "ended at {expected:?}, the block's centre at ({}, {})",
            x + 20.0,
            y + 30.0
        );
    }

    /// Where the object is not, nothing stands out.
    #[test]
    fn nothing_there_is_not_sure() {
        let block = [100.0, 80.0, 40.0, 60.0];
        let mut scene = Scene {
            size: (320, 240),
            block,
        };
        let dcf = Dcf::new(&mut scene, block).unwrap();
        scene.block = [-500.0, -500.0, 1.0, 1.0];
        let found = dcf.find(&mut scene, (120.0, 110.0)).unwrap();
        assert!(found.psr < 6.0, "{found:?}");
    }
}
