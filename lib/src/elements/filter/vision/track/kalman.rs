//! The Kalman filter a box is followed with: its centre, its width and its
//! height, each moving at a constant velocity of its own from one picture
//! to the next.
//!
//! Width and height, as BoT-SORT has them, rather than SORT's and
//! ByteTrack's aspect ratio and height: their aspect ratio is given a
//! fixed uncertainty a hundredth of a ratio near 0.4, so it all but never
//! follows a person turning or a car seen from a new side, and a box
//! carried between detections kept the width it first had — half again
//! too wide, measured on people walking, against a detector's box.
//!
//! The matrix arithmetic is written out by index, as the formulas are: an
//! iterator per row and column would hide which product is which.
#![allow(clippy::needless_range_loop)]

/// A box as the filter measures it: centre x, centre y, width and height,
/// in pixels.
pub(super) type Measurement = [f64; 4];

type Vector = [f64; 8];
type Matrix = [[f64; 8]; 8];
type Small = [[f64; 4]; 4];

/// How uncertain a position is, relative to the box's size along it —
/// ByteTrack's own weights, which make the filter the same for a near box
/// and a far one.
const POSITION_WEIGHT: f64 = 1.0 / 20.0;
/// The same for a velocity.
const VELOCITY_WEIGHT: f64 = 1.0 / 160.0;

/// Where a box is, how fast each part of it is changing, and how sure the
/// filter is of both.
#[derive(Debug, Clone)]
pub(super) struct Kalman {
    /// `[x, y, w, h, vx, vy, vw, vh]`.
    mean: Vector,
    covariance: Matrix,
}

fn diagonal(values: Vector) -> Matrix {
    let mut matrix = [[0.0; 8]; 8];
    for (i, value) in values.into_iter().enumerate() {
        matrix[i][i] = value * value;
    }
    matrix
}

impl Kalman {
    /// A box first seen at `measurement`, standing still as far as is known.
    pub(super) fn new(measurement: Measurement) -> Self {
        let (w, h) = (measurement[2], measurement[3]);
        let mut mean = [0.0; 8];
        mean[..4].copy_from_slice(&measurement);
        let (pw, ph) = (POSITION_WEIGHT * w, POSITION_WEIGHT * h);
        let (vw, vh) = (VELOCITY_WEIGHT * w, VELOCITY_WEIGHT * h);
        Self {
            mean,
            covariance: diagonal([
                2.0 * pw,
                2.0 * ph,
                2.0 * pw,
                2.0 * ph,
                10.0 * vw,
                10.0 * vh,
                10.0 * vw,
                10.0 * vh,
            ]),
        }
    }

    /// Where the box is, as measured.
    pub(super) fn measurement(&self) -> Measurement {
        [self.mean[0], self.mean[1], self.mean[2], self.mean[3]]
    }

    /// Stops the box growing or shrinking — what ByteTrack does to a box it
    /// has lost, whose size would otherwise run on unchecked.
    pub(super) fn hold_size(&mut self) {
        self.mean[6] = 0.0;
        self.mean[7] = 0.0;
    }

    /// One picture on: everything moves by its velocity, and the filter is
    /// less sure of where it all is.
    pub(super) fn predict(&mut self) {
        let (w, h) = (self.mean[2], self.mean[3]);
        for i in 0..4 {
            self.mean[i] += self.mean[i + 4];
        }
        // F · P · Fᵀ, with F the identity plus the velocities in the top
        // right: each position row and column gains its velocity's.
        let p = &mut self.covariance;
        for row in p.iter_mut() {
            for i in 0..4 {
                row[i] += row[i + 4];
            }
        }
        for i in 0..4 {
            let velocity = p[i + 4];
            for (cell, add) in p[i].iter_mut().zip(velocity) {
                *cell += add;
            }
        }
        let (pw, ph) = (POSITION_WEIGHT * w, POSITION_WEIGHT * h);
        let (vw, vh) = (VELOCITY_WEIGHT * w, VELOCITY_WEIGHT * h);
        let noise = [pw, ph, pw, ph, vw, vh, vw, vh];
        for (i, std) in noise.into_iter().enumerate() {
            p[i][i] += std * std;
        }
    }

    /// Corrects the box toward where it was measured, by how much the
    /// filter trusts the measurement over its own expectation.
    pub(super) fn update(&mut self, measurement: Measurement) {
        self.update_trusting(measurement, 1.0);
    }

    /// [`Self::update`], for a measurement `doubt` times as uncertain as a
    /// detection's — one a visual tracker made, which the motion is weighed
    /// against rather than overruled by.
    pub(super) fn update_trusting(&mut self, measurement: Measurement, doubt: f64) {
        let (w, h) = (self.mean[2], self.mean[3]);
        let std = [
            doubt * POSITION_WEIGHT * w,
            doubt * POSITION_WEIGHT * h,
            doubt * POSITION_WEIGHT * w,
            doubt * POSITION_WEIGHT * h,
        ];
        // S = H · P · Hᵀ + R: the top-left 4x4 of P, plus the noise.
        let mut s: Small = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                s[i][j] = self.covariance[i][j];
            }
            s[i][i] += std[i] * std[i];
        }
        let Some(s_inverse) = invert(s) else {
            return;
        };
        // K = P · Hᵀ · S⁻¹: P's first four columns times S⁻¹.
        let mut gain = [[0.0; 4]; 8];
        for (i, row) in gain.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..4)
                    .map(|k| self.covariance[i][k] * s_inverse[k][j])
                    .sum();
            }
        }
        let innovation: [f64; 4] = std::array::from_fn(|i| measurement[i] - self.mean[i]);
        for (i, row) in gain.iter().enumerate() {
            self.mean[i] += (0..4).map(|j| row[j] * innovation[j]).sum::<f64>();
        }
        // P -= K · S · Kᵀ, which is K · (H · P): K times P's first four rows.
        let top = [
            self.covariance[0],
            self.covariance[1],
            self.covariance[2],
            self.covariance[3],
        ];
        for (i, row) in gain.iter().enumerate() {
            for j in 0..8 {
                self.covariance[i][j] -= (0..4).map(|k| row[k] * top[k][j]).sum::<f64>();
            }
        }
    }
}

/// The inverse of a 4x4 matrix, by Gauss-Jordan elimination with partial
/// pivoting; `None` for one that has none.
fn invert(mut m: Small) -> Option<Small> {
    let mut inverse: Small = std::array::from_fn(|i| std::array::from_fn(|j| f64::from(i == j)));
    for column in 0..4 {
        let pivot =
            (column..4).max_by(|&a, &b| m[a][column].abs().total_cmp(&m[b][column].abs()))?;
        if m[pivot][column].abs() < 1e-12 {
            return None;
        }
        m.swap(column, pivot);
        inverse.swap(column, pivot);
        let scale = m[column][column];
        for j in 0..4 {
            m[column][j] /= scale;
            inverse[column][j] /= scale;
        }
        for row in 0..4 {
            if row != column {
                let factor = m[row][column];
                for j in 0..4 {
                    m[row][j] -= factor * m[column][j];
                    inverse[row][j] -= factor * inverse[column][j];
                }
            }
        }
    }
    Some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_times_its_inverse_is_the_identity() {
        let m = [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.2, 0.1],
            [0.5, 0.2, 2.0, 0.3],
            [0.0, 0.1, 0.3, 1.0],
        ];
        let inverse = invert(m).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let product: f64 = (0..4).map(|k| m[i][k] * inverse[k][j]).sum();
                assert!(
                    (product - f64::from(i == j)).abs() < 1e-9,
                    "{i},{j}: {product}"
                );
            }
        }
        assert!(invert([[0.0; 4]; 4]).is_none());
    }

    /// Fed a box moving 5 pixels a picture, it learns the velocity and
    /// predicts where the box will be next.
    #[test]
    fn a_steady_motion_is_learned_and_carried_on() {
        let mut kalman = Kalman::new([100.0, 50.0, 0.5, 80.0]);
        for step in 1..=20 {
            kalman.predict();
            kalman.update([100.0 + 5.0 * f64::from(step), 50.0, 0.5, 80.0]);
        }
        kalman.predict();
        let [x, y, a, h] = kalman.measurement();
        assert!((x - 205.0).abs() < 1.0, "x {x}");
        assert!((y - 50.0).abs() < 0.5, "y {y}");
        assert!(
            (a - 0.5).abs() < 0.01 && (h - 80.0).abs() < 0.5,
            "a {a}, h {h}"
        );
    }

    /// A measurement pulls the box toward it without jumping all the way: the
    /// filter weighs what it saw against what it expected.
    #[test]
    fn a_measurement_is_weighed_against_the_expectation() {
        let mut kalman = Kalman::new([100.0, 100.0, 50.0, 100.0]);
        for _ in 0..10 {
            kalman.predict();
            kalman.update([100.0, 100.0, 50.0, 100.0]);
        }
        kalman.predict();
        kalman.update([110.0, 100.0, 50.0, 100.0]);
        let x = kalman.measurement()[0];
        assert!(x > 100.0 && x < 110.0, "x {x}");
    }
}
