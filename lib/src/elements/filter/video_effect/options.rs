//! What a video effect does, and the one set of numbers every backend
//! evaluates it from.
//!
//! Each effect is described in its own terms — a brightness, a hue, a luma
//! range — and resolved here into an [`EffectParams`]: a colour matrix, an
//! exponent, an opacity and a luma mask. The D3D11 shader, the CUDA kernel
//! and the software loop all evaluate that one block and nothing else, so an
//! effect is added here once rather than three times, and the three cannot
//! drift into disagreeing about what it means.

/// BT.709 luma weights for red, green and blue — the definition this crate
/// already uses wherever it turns RGB into brightness.
pub(crate) const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// One effect a video-effect element applies, with its settings.
///
/// Passed at construction and changed afterwards through
/// [`VideoEffectHandle`](super::VideoEffectHandle), which may also switch
/// one effect for the other: both resolve to the same parameters, so the
/// element does not care which it is running.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VideoEffect {
    /// Brightness, contrast, saturation, hue, gamma and opacity.
    ColorCorrection(ColorCorrection),
    /// Transparency by brightness: what is darker or brighter than a range
    /// is cut out.
    LumaKey(LumaKey),
}

/// A picture's tone and colour, adjusted.
///
/// Applied in the order the fields are listed: gamma first, then saturation
/// and hue, then contrast and brightness, and the result clamped. Every
/// field's neutral value leaves the picture exactly as it was, and
/// [`ColorCorrection::default`] is all of them.
///
/// Values are not validated, as a chroma key's are not: a contrast below
/// zero is read as zero and a gamma at or below zero as a very small one,
/// which are answers rather than errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorCorrection {
    /// Added to every channel, as a fraction of full scale. `0.0` is neutral;
    /// `-1.0..=1.0` is the range that means anything.
    pub brightness: f32,
    /// How far each channel is from mid grey, as a multiple. `1.0` is
    /// neutral, `0.0` flattens the picture to grey, `2.0` doubles it.
    pub contrast: f32,
    /// How far each pixel is from its own grey, as a multiple. `1.0` is
    /// neutral, `0.0` is greyscale, `2.0` twice as saturated.
    pub saturation: f32,
    /// Rotation of every colour around the grey axis, in degrees. `0.0` is
    /// neutral; greys stay grey whatever it is.
    pub hue_degrees: f32,
    /// Each channel becomes `channel^(1 / gamma)`. `1.0` is neutral; above it
    /// lifts the mid-tones, below it darkens them, and black and white stay
    /// where they are.
    pub gamma: f32,
    /// Multiplies the picture's alpha. `1.0` is neutral, `0.0` invisible.
    pub opacity: f32,
}

impl Default for ColorCorrection {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            hue_degrees: 0.0,
            gamma: 1.0,
            opacity: 1.0,
        }
    }
}

/// Cuts out what is too dark or too bright — a black or white backdrop, a
/// logo on either, a flame against a dark room.
///
/// Brightness is BT.709 luma of the pixel as it arrives, `0.0` for black and
/// `1.0` for white. A pixel inside `min..=max` keeps its alpha; outside, it
/// fades to transparent over the smoothing distance beyond each end.
/// [`LumaKey::default`] keeps everything, so a key added and not yet tuned
/// changes nothing.
///
/// The key multiplies the alpha a pixel already has rather than replacing
/// it, so it can follow a chroma key and cut out what that one left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LumaKey {
    /// Pixels darker than this become transparent. `0.0` keeps them all.
    pub min: f32,
    /// How far below `min` the fade to transparent reaches. `0.0` is a hard
    /// edge.
    pub min_smoothing: f32,
    /// Pixels brighter than this become transparent. `1.0` keeps them all.
    pub max: f32,
    /// How far above `max` the fade to transparent reaches. `0.0` is a hard
    /// edge.
    pub max_smoothing: f32,
}

impl Default for LumaKey {
    fn default() -> Self {
        Self {
            min: 0.0,
            min_smoothing: 0.0,
            max: 1.0,
            max_smoothing: 0.0,
        }
    }
}

/// What every backend evaluates, per pixel, with each channel in `0.0..=1.0`:
///
/// ```text
/// luma  = saturate(dot(LUMA, rgb))
/// mask  = saturate((luma - luma_low) * luma_low_inv + 1)
///       * saturate((luma_high - luma) * luma_high_inv + 1)
/// c     = rgb ^ exponent                      (skipped when exponent is 1)
/// rgb'  = saturate(rows · [c, 1])
/// alpha = saturate(alpha * opacity * mask)
/// ```
///
/// The mask is read from the pixel as it arrived, before the colour changes.
/// A hard edge arrives as an effectively infinite `*_inv`, which the `+ 1`
/// turns into "inside keeps, outside goes" — the same no-branch, no-division
/// trick the chroma key's feather band uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EffectParams {
    /// One row per output channel — red, green, blue — each
    /// `[from_red, from_green, from_blue, offset]`.
    pub(crate) rows: [[f32; 4]; 3],
    pub(crate) exponent: f32,
    pub(crate) opacity: f32,
    pub(crate) luma_low: f32,
    pub(crate) luma_low_inv: f32,
    pub(crate) luma_high: f32,
    pub(crate) luma_high_inv: f32,
}

impl EffectParams {
    const IDENTITY_ROWS: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];

    /// Leaves every pixel exactly as it was, so an element may hand the frame
    /// on untouched rather than make a copy that is the same.
    pub(crate) fn is_identity(&self) -> bool {
        self.rows == Self::IDENTITY_ROWS
            && self.exponent == 1.0
            && self.opacity == 1.0
            && self.luma_low <= 0.0
            && self.luma_high >= 1.0
    }
}

impl VideoEffect {
    /// The parameters every backend runs this effect with.
    pub(crate) fn params(&self) -> EffectParams {
        match self {
            Self::ColorCorrection(correction) => correction.params(),
            Self::LumaKey(key) => key.params(),
        }
    }

    /// Which effect this is, for a log line.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::ColorCorrection(_) => "color correction",
            Self::LumaKey(_) => "luma key",
        }
    }
}

impl ColorCorrection {
    fn params(&self) -> EffectParams {
        let saturation = self.saturation.max(0.0);
        let contrast = self.contrast.max(0.0);

        // Saturation: each channel moved toward the pixel's own luma, by
        // `1 - saturation` — a matrix whose every row is the luma weights,
        // blended with the identity.
        let mut saturate = [[0.0f32; 3]; 3];
        for (row, channel) in saturate.iter_mut().enumerate() {
            for (column, weight) in channel.iter_mut().enumerate() {
                let identity = if row == column { 1.0 } else { 0.0 };
                *weight = (1.0 - saturation) * LUMA[column] + saturation * identity;
            }
        }

        // Hue: a rotation about the grey axis (1, 1, 1), so a grey stays the
        // grey it was however far the colours around it turn.
        let (sin, cos) = self.hue_degrees.to_radians().sin_cos();
        let a = (1.0 - cos) / 3.0;
        let b = sin / 3f32.sqrt();
        let rotate = [
            [cos + a, a - b, a + b],
            [a + b, cos + a, a - b],
            [a - b, a + b, cos + a],
        ];

        // Contrast about mid grey, then brightness: a scale and one offset
        // shared by all three channels.
        let offset = 0.5 * (1.0 - contrast) + self.brightness;
        let mut rows = [[0.0f32; 4]; 3];
        for (row, out) in rows.iter_mut().enumerate() {
            for (column, weight) in out.iter_mut().take(3).enumerate() {
                let rotated: f32 = (0..3).map(|k| rotate[row][k] * saturate[k][column]).sum();
                *weight = contrast * rotated;
            }
            out[3] = offset;
        }
        // Exactly the identity when nothing was asked for, rather than an
        // identity plus whatever the trigonometry left over: that is what
        // lets a neutral correction cost nothing.
        if self.saturation == 1.0 && self.hue_degrees == 0.0 && self.contrast == 1.0 {
            rows = EffectParams::IDENTITY_ROWS;
            for row in &mut rows {
                row[3] = self.brightness;
            }
        }

        EffectParams {
            rows,
            exponent: 1.0 / self.gamma.max(1e-3),
            opacity: self.opacity,
            // No mask: every luma is inside.
            luma_low: 0.0,
            luma_low_inv: 1.0,
            luma_high: 1.0,
            luma_high_inv: 1.0,
        }
    }
}

impl LumaKey {
    fn params(&self) -> EffectParams {
        EffectParams {
            rows: EffectParams::IDENTITY_ROWS,
            exponent: 1.0,
            opacity: 1.0,
            luma_low: self.min,
            luma_low_inv: inverse_width(self.min_smoothing),
            luma_high: self.max,
            luma_high_inv: inverse_width(self.max_smoothing),
        }
    }
}

/// `1 / smoothing`, or for a hard edge an inverse so large that anything
/// past the edge by however little saturates to fully transparent.
fn inverse_width(smoothing: f32) -> f32 {
    if smoothing > 0.0 {
        1.0 / smoothing
    } else {
        f32::MAX
    }
}

/// Channel bytes as they are written back: nearest, the way a GPU's
/// float-to-UNORM conversion rounds and the CUDA kernel's `+ 0.5` does.
fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// One BGRA pixel through `params` — what the software element does to every
/// pixel, and what the GPU backends' tests compare their own output with.
pub(crate) fn apply(params: &EffectParams, [b, g, r, a]: [u8; 4]) -> [u8; 4] {
    let channel = |byte: u8| f32::from(byte) / 255.0;
    let rgb = [channel(r), channel(g), channel(b)];
    let luma = (LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]).clamp(0.0, 1.0);
    let low = ((luma - params.luma_low) * params.luma_low_inv + 1.0).clamp(0.0, 1.0);
    let high = ((params.luma_high - luma) * params.luma_high_inv + 1.0).clamp(0.0, 1.0);
    let mask = low * high;

    let lifted = if params.exponent == 1.0 {
        rgb
    } else {
        rgb.map(|value| value.powf(params.exponent))
    };
    let out = params
        .rows
        .map(|row| row[0] * lifted[0] + row[1] * lifted[1] + row[2] * lifted[2] + row[3]);
    let alpha = channel(a) * params.opacity * mask;
    [
        to_byte(out[2]),
        to_byte(out[1]),
        to_byte(out[0]),
        to_byte(alpha),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: [[u8; 4]; 6] = [
        [0, 0, 0, 255],
        [255, 255, 255, 255],
        [30, 140, 220, 255],
        [200, 60, 10, 128],
        [128, 128, 128, 0],
        [1, 254, 77, 200],
    ];

    fn correct(correction: ColorCorrection, pixel: [u8; 4]) -> [u8; 4] {
        apply(&VideoEffect::ColorCorrection(correction).params(), pixel)
    }

    /// Both defaults leave every pixel exactly as it was, and say so — which
    /// is what lets an element hand the frame through untouched.
    #[test]
    fn the_defaults_change_nothing_and_are_known_to() {
        for effect in [
            VideoEffect::ColorCorrection(ColorCorrection::default()),
            VideoEffect::LumaKey(LumaKey::default()),
        ] {
            let params = effect.params();
            assert!(params.is_identity(), "{effect:?}");
            for pixel in SAMPLES {
                assert_eq!(apply(&params, pixel), pixel, "{effect:?} on {pixel:?}");
            }
        }
    }

    #[test]
    fn brightness_is_added_to_every_channel() {
        let brighter = ColorCorrection {
            brightness: 0.2,
            ..ColorCorrection::default()
        };
        // 0.2 of full scale is 51.
        assert_eq!(correct(brighter, [10, 20, 30, 255]), [61, 71, 81, 255]);
        assert_eq!(
            correct(brighter, [250, 250, 250, 255]),
            [255, 255, 255, 255]
        );
    }

    #[test]
    fn contrast_scales_about_mid_grey() {
        let flat = ColorCorrection {
            contrast: 0.0,
            ..ColorCorrection::default()
        };
        assert_eq!(correct(flat, [0, 255, 40, 255]), [128, 128, 128, 255]);

        let more = ColorCorrection {
            contrast: 1.5,
            ..ColorCorrection::default()
        };
        // Each channel half as far again from 0.5: 200 -> 236.25,
        // 128 -> 128.25, 100 -> 86.25, none of them near a rounding edge.
        assert_eq!(correct(more, [200, 128, 100, 255]), [236, 128, 86, 255]);
    }

    #[test]
    fn no_saturation_is_greyscale_by_luma() {
        let grey = ColorCorrection {
            saturation: 0.0,
            ..ColorCorrection::default()
        };
        let [b, g, r, a] = correct(grey, [0, 0, 255, 255]);
        assert_eq!((b, g, r, a), (b, b, b, 255), "every channel the same");
        // Pure red's luma is 0.2126 of full scale: 54.
        assert_eq!(b, 54);
    }

    /// A hue rotation turns colours and leaves greys where they are; a whole
    /// third of a turn carries each primary onto the next.
    #[test]
    fn hue_turns_colours_about_the_grey_axis() {
        let third = ColorCorrection {
            hue_degrees: 120.0,
            ..ColorCorrection::default()
        };
        assert_eq!(correct(third, [90, 90, 90, 255]), [90, 90, 90, 255]);
        let [b, g, r, _] = correct(third, [0, 0, 255, 255]);
        assert!(r <= 1 && g >= 254 && b <= 1, "red turned to {r},{g},{b}");
    }

    #[test]
    fn gamma_lifts_the_mid_tones_and_leaves_the_ends() {
        let lifted = ColorCorrection {
            gamma: 2.0,
            ..ColorCorrection::default()
        };
        assert_eq!(correct(lifted, [0, 255, 64, 255])[..2], [0, 255]);
        // sqrt(64 / 255) of full scale is 128.
        assert_eq!(correct(lifted, [0, 255, 64, 255])[2], 128);
    }

    #[test]
    fn opacity_scales_the_alpha_already_there() {
        let half = ColorCorrection {
            opacity: 0.5,
            ..ColorCorrection::default()
        };
        assert_eq!(correct(half, [10, 20, 30, 255]), [10, 20, 30, 128]);
        assert_eq!(correct(half, [10, 20, 30, 100])[3], 50);
    }

    fn key(key: LumaKey, pixel: [u8; 4]) -> [u8; 4] {
        apply(&VideoEffect::LumaKey(key).params(), pixel)
    }

    /// Inside the range keeps its alpha and its colour; outside goes, at
    /// whichever end.
    #[test]
    fn a_luma_key_cuts_what_is_outside_its_range() {
        let range = LumaKey {
            min: 0.2,
            max: 0.8,
            ..LumaKey::default()
        };
        assert_eq!(key(range, [0, 0, 0, 255])[3], 0, "black is below");
        assert_eq!(key(range, [255, 255, 255, 255])[3], 0, "white is above");
        assert_eq!(key(range, [128, 128, 128, 255]), [128, 128, 128, 255]);
        assert_eq!(
            key(range, [56, 56, 56, 255])[3],
            255,
            "just inside the low end"
        );
        assert_eq!(
            key(range, [199, 199, 199, 255])[3],
            255,
            "just inside the high end"
        );
    }

    #[test]
    fn a_luma_key_fades_over_its_smoothing() {
        let soft = LumaKey {
            min: 0.4,
            min_smoothing: 0.2,
            ..LumaKey::default()
        };
        // 0.3 is halfway down the 0.2..0.4 fade.
        let alpha = key(soft, [77, 77, 77, 255])[3];
        assert!((126..=130).contains(&alpha), "got {alpha}");
        assert_eq!(key(soft, [26, 26, 26, 255])[3], 0, "0.1 is past the fade");
    }

    /// It multiplies what alpha there was, so it composes with a chroma key
    /// in front of it rather than undoing it.
    #[test]
    fn a_luma_key_keeps_the_alpha_a_pixel_already_had() {
        let range = LumaKey {
            min: 0.2,
            ..LumaKey::default()
        };
        assert_eq!(key(range, [128, 128, 128, 90])[3], 90);
        assert_eq!(key(range, [128, 128, 128, 0])[3], 0);
    }

    /// Anything asked for is not the identity, so it is not skipped.
    #[test]
    fn a_setting_moved_off_neutral_is_not_the_identity() {
        for correction in [
            ColorCorrection {
                brightness: 0.01,
                ..ColorCorrection::default()
            },
            ColorCorrection {
                gamma: 1.1,
                ..ColorCorrection::default()
            },
            ColorCorrection {
                opacity: 0.9,
                ..ColorCorrection::default()
            },
            ColorCorrection {
                hue_degrees: 5.0,
                ..ColorCorrection::default()
            },
        ] {
            assert!(
                !VideoEffect::ColorCorrection(correction)
                    .params()
                    .is_identity()
            );
        }
        assert!(
            !VideoEffect::LumaKey(LumaKey {
                max: 0.9,
                ..LumaKey::default()
            })
            .params()
            .is_identity()
        );
    }
}
