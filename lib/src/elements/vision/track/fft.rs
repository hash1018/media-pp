//! The discrete Fourier transform of a square power-of-two image, which a
//! correlation filter is learned and applied through: a correlation over
//! every shift at once is a product of transforms.

use std::ops::{Add, Div, Mul, Sub};

/// A complex number.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Complex {
    pub(super) re: f32,
    pub(super) im: f32,
}

impl Complex {
    pub(super) const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub(super) fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    pub(super) fn scale(self, by: f32) -> Self {
        Self::new(self.re * by, self.im * by)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let norm = other.re * other.re + other.im * other.im;
        (self * other.conj()).scale(1.0 / norm)
    }
}

/// An in-place radix-2 transform of `data`, whose length is a power of
/// two; the inverse, unscaled, where `inverse`.
fn fft(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reversed order, so the butterflies below combine neighbours.
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let angle = sign * 2.0 * std::f32::consts::PI / len as f32;
        let step = Complex::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut twiddle = Complex::new(1.0, 0.0);
            for k in 0..len / 2 {
                let even = data[start + k];
                let odd = data[start + k + len / 2] * twiddle;
                data[start + k] = even + odd;
                data[start + k + len / 2] = even - odd;
                twiddle = twiddle * step;
            }
        }
        len <<= 1;
    }
}

/// The transform of a `size` by `size` image, row-major, in place: every
/// row, then every column.
pub(super) fn fft2(data: &mut [Complex], size: usize, inverse: bool) {
    debug_assert_eq!(data.len(), size * size);
    for row in data.chunks_exact_mut(size) {
        fft(row, inverse);
    }
    let mut column = vec![Complex::default(); size];
    for x in 0..size {
        for (y, cell) in column.iter_mut().enumerate() {
            *cell = data[y * size + x];
        }
        fft(&mut column, inverse);
        for (y, cell) in column.iter().enumerate() {
            data[y * size + x] = *cell;
        }
    }
    if inverse {
        let scale = 1.0 / (size * size) as f32;
        for cell in data.iter_mut() {
            *cell = cell.scale(scale);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform agrees with the definition, term by term, and the
    /// inverse brings the image back.
    #[test]
    fn the_transform_is_the_definition_and_inverts() {
        let size = 8;
        let image: Vec<Complex> = (0..size * size)
            .map(|i| Complex::new(((i * 7) % 11) as f32 - 5.0, 0.0))
            .collect();
        let mut transformed = image.clone();
        fft2(&mut transformed, size, false);
        for (v, u) in [(0, 0), (1, 3), (5, 2), (7, 7)] {
            let mut sum = Complex::default();
            for y in 0..size {
                for x in 0..size {
                    let angle =
                        -2.0 * std::f32::consts::PI * ((u * x + v * y) as f32) / size as f32;
                    sum = sum + image[y * size + x] * Complex::new(angle.cos(), angle.sin());
                }
            }
            let got = transformed[v * size + u];
            assert!(
                (got.re - sum.re).abs() < 1e-3 && (got.im - sum.im).abs() < 1e-3,
                "({u}, {v}): {got:?} against {sum:?}"
            );
        }
        fft2(&mut transformed, size, true);
        for (back, original) in transformed.iter().zip(&image) {
            assert!((back.re - original.re).abs() < 1e-4 && back.im.abs() < 1e-4);
        }
    }
}
