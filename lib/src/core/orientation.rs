//! Which way up a picture is shown: a phone held upright records its
//! sensor's landscape pictures and says, beside them, to turn them a
//! quarter round — the container's display matrix, which every decoder of
//! this crate hands on as each picture's own.
//!
//! The pixels stay as they were stored, and so does everything this crate
//! says about where things are in them: a [`Detection`]'s box, a tracker's
//! object, an overlay's mosaic are all of the stored picture, so nothing
//! has to turn a picture to work on it. What turns is a model's input,
//! which is fitted the right way up, and whatever shows the picture.
//!
//! [`Detection`]: crate::elements::Detection

use ffmpeg_next::{ffi, frame::Video};
use thiserror::Error as ThisError;

/// A quarter turn, clockwise, as many times as it says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Rotation {
    /// Not turned.
    #[default]
    None,
    /// A quarter turn clockwise.
    Clockwise90,
    /// Upside down.
    Half,
    /// Three quarters clockwise: a quarter turn the other way.
    Clockwise270,
}

/// How a stored picture is turned to be shown: mirrored left to right where
/// it says so, then turned clockwise by `rotation`.
///
/// One of the eight a picture can be turned by without resampling it — the
/// only ones a phone or a camera writes. [`Orientation::of`] reads a
/// picture's, [`StreamInfo::orientation`] a stream's, and
/// [`TrackFormat::with_orientation`] writes one into a file a picture is
/// re-encoded to, where the container keeps it beside the pictures as the
/// file being copied had.
///
/// [`StreamInfo::orientation`]: crate::elements::StreamInfo::orientation
/// [`TrackFormat::with_orientation`]: crate::elements::TrackFormat::with_orientation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Orientation {
    /// The turn, after any mirroring.
    pub rotation: Rotation,
    /// Whether it is mirrored left to right first.
    pub mirrored: bool,
}

/// A display matrix that turns a picture by something other than quarter
/// turns, or skews it — which nothing here can follow.
#[derive(Debug, Clone, Copy, PartialEq, ThisError)]
#[error("the picture is turned {degrees:.1} degrees counterclockwise, not by quarter turns")]
pub struct UnsupportedOrientation {
    /// The angle FFmpeg reads from the matrix, counterclockwise.
    pub degrees: f64,
}

/// Where a stored point goes when shown, about the picture's centre with
/// each side running from -1 to 1: `x' = a x + c y`, `y' = b x + d y`, as
/// FFmpeg lays a display matrix out, each of `a`, `b`, `c`, `d` -1, 0 or 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Linear {
    a: i32,
    b: i32,
    c: i32,
    d: i32,
}

impl Linear {
    /// The map that undoes this one: a signed permutation's transpose.
    fn inverse(self) -> Self {
        Self {
            a: self.a,
            b: self.c,
            c: self.b,
            d: self.d,
        }
    }

    /// Whether a shown picture's sides are the stored one's the other way
    /// round.
    fn transposes(self) -> bool {
        self.a == 0
    }

    /// `(x, y)`, each -1 to 1, mapped.
    fn apply(self, (x, y): (f32, f32)) -> (f32, f32) {
        (
            self.a as f32 * x + self.c as f32 * y,
            self.b as f32 * x + self.d as f32 * y,
        )
    }
}

impl Orientation {
    /// Shown as stored.
    pub const UPRIGHT: Self = Self {
        rotation: Rotation::None,
        mirrored: false,
    };

    /// Turned by `rotation`, not mirrored.
    pub const fn rotated(rotation: Rotation) -> Self {
        Self {
            rotation,
            mirrored: false,
        }
    }

    /// Whether it is shown as stored.
    pub fn is_upright(self) -> bool {
        self == Self::UPRIGHT
    }

    /// The picture's: upright where it carries no display matrix.
    ///
    /// # Errors
    ///
    /// Where its matrix turns it by other than quarter turns.
    pub fn of(frame: &Video) -> Result<Self, UnsupportedOrientation> {
        // SAFETY: `frame` is a live `AVFrame`; its side data is FFmpeg's own,
        // and a display matrix's is nine `i32`s by FFmpeg's definition.
        unsafe {
            let data = ffi::av_frame_get_side_data(
                frame.as_ptr(),
                ffi::AVFrameSideDataType::AV_FRAME_DATA_DISPLAYMATRIX,
            );
            if data.is_null() || (*data).size < 9 * size_of::<i32>() {
                return Ok(Self::UPRIGHT);
            }
            Self::of_matrix(&*((*data).data as *const [i32; 9]))
        }
    }

    /// What FFmpeg's display `matrix` says.
    pub(crate) fn of_matrix(matrix: &[i32; 9]) -> Result<Self, UnsupportedOrientation> {
        let refused = || UnsupportedOrientation {
            // SAFETY: nine `i32`s, which is what the function reads.
            degrees: unsafe { ffi::av_display_rotation_get(matrix.as_ptr()) },
        };
        let [a, b, _, c, d, ..] = *matrix;
        // Each entry is 16.16 fixed point; a scaled matrix still says which
        // way each axis goes, by which entries are far from naught.
        let largest = [a, b, c, d]
            .map(|value| u64::from(value.unsigned_abs()))
            .into_iter()
            .max()
            .unwrap_or(0);
        if largest == 0 {
            return Err(refused());
        }
        let sign = |value: i32| -> Option<i32> {
            let magnitude = u64::from(value.unsigned_abs());
            if magnitude * 4 < largest {
                Some(0)
            } else if magnitude * 4 > largest * 3 {
                Some(value.signum())
            } else {
                None
            }
        };
        let linear = Linear {
            a: sign(a).ok_or_else(refused)?,
            b: sign(b).ok_or_else(refused)?,
            c: sign(c).ok_or_else(refused)?,
            d: sign(d).ok_or_else(refused)?,
        };
        Self::ALL
            .into_iter()
            .find(|orientation| orientation.linear() == linear)
            .ok_or_else(refused)
    }

    /// The display matrix FFmpeg writes for it.
    pub(crate) fn matrix(self) -> [i32; 9] {
        let Linear { a, b, c, d } = self.linear();
        [a << 16, b << 16, 0, c << 16, d << 16, 0, 0, 0, 1 << 30]
    }

    /// Every one there is.
    const ALL: [Self; 8] = {
        let rotations = [
            Rotation::None,
            Rotation::Clockwise90,
            Rotation::Half,
            Rotation::Clockwise270,
        ];
        let mut all = [Self::UPRIGHT; 8];
        let mut index = 0;
        while index < 8 {
            all[index] = Self {
                rotation: rotations[index % 4],
                mirrored: index >= 4,
            };
            index += 1;
        }
        all
    };

    fn linear(self) -> Linear {
        // In image coordinates, y downward: a quarter turn clockwise takes
        // (x, y) to (-y, x).
        let turned = match self.rotation {
            Rotation::None => Linear {
                a: 1,
                b: 0,
                c: 0,
                d: 1,
            },
            Rotation::Clockwise90 => Linear {
                a: 0,
                b: 1,
                c: -1,
                d: 0,
            },
            Rotation::Half => Linear {
                a: -1,
                b: 0,
                c: 0,
                d: -1,
            },
            Rotation::Clockwise270 => Linear {
                a: 0,
                b: -1,
                c: 1,
                d: 0,
            },
        };
        if self.mirrored {
            // The mirror first, (x, y) to (-x, y): the turn's x column negated.
            Linear {
                a: -turned.a,
                b: -turned.b,
                ..turned
            }
        } else {
            turned
        }
    }

    /// The size a `width` by `height` stored picture is shown at.
    pub fn display_size(self, width: u32, height: u32) -> (u32, u32) {
        if self.linear().transposes() {
            (height, width)
        } else {
            (width, height)
        }
    }

    /// A box of the stored picture — left, top, width and height, as
    /// fractions of it — where it is shown, as fractions of what is shown.
    pub fn to_display(self, rect: [f32; 4]) -> [f32; 4] {
        map(self.linear(), rect)
    }

    /// A box of what is shown — left, top, width and height, as fractions
    /// of it — where it is in the stored picture.
    pub fn from_display(self, rect: [f32; 4]) -> [f32; 4] {
        map(self.linear().inverse(), rect)
    }

    /// Where the shown picture's pixel `(x, y)` is stored, for a
    /// `width` by `height` stored picture: `[x_of_x, x_of_y, x_0, y_of_x,
    /// y_of_y, y_0]`, the stored pixel being `(x_of_x * x + x_of_y * y +
    /// x_0, y_of_x * x + y_of_y * y + y_0)`. What a kernel fitting the
    /// picture the right way up reads it by.
    pub(crate) fn sampling(self, width: u32, height: u32) -> [i32; 6] {
        // The stored point, about the centre, is the inverse of the shown
        // one: x = a x' + b y', y = c x' + d y'. A pixel's -1 end is pixel
        // 0 and its 1 end pixel `size - 1`.
        let Linear { a, b, c, d } = self.linear();
        let start = |of_x: i32, of_y: i32, size: u32| {
            if of_x < 0 || of_y < 0 {
                size as i32 - 1
            } else {
                0
            }
        };
        [a, b, start(a, b, width), c, d, start(c, d, height)]
    }
}

/// How each picture an element is handed is turned to be shown, read from
/// the picture itself: one turned by other than quarter turns is said once,
/// and taken as it is stored.
#[derive(Default)]
pub(crate) struct Orientations {
    warned: bool,
}

impl Orientations {
    /// `frame`'s.
    pub(crate) fn of(&mut self, frame: &Video, pp_log: &crate::pp_log::PpLog) -> Orientation {
        Orientation::of(frame).unwrap_or_else(|error| {
            if !self.warned {
                crate::pp_log::pp_warn!(pp_log: pp_log, "{error}: taken as it is stored");
                self.warned = true;
            }
            Orientation::UPRIGHT
        })
    }
}

/// `rect`, as fractions, through `linear` about the centre.
fn map(linear: Linear, [x, y, width, height]: [f32; 4]) -> [f32; 4] {
    let centred = |value: f32| value * 2.0 - 1.0;
    let (x0, y0) = linear.apply((centred(x), centred(y)));
    let (x1, y1) = linear.apply((centred(x + width), centred(y + height)));
    let fraction = |value: f32| (value + 1.0) / 2.0;
    let (left, right) = (x0.min(x1), x0.max(x1));
    let (top, bottom) = (y0.min(y1), y0.max(y1));
    [
        fraction(left),
        fraction(top),
        (right - left) / 2.0,
        (bottom - top) / 2.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-6)
    }

    /// FFmpeg reads each turn's matrix as the turn it makes:
    /// `av_display_rotation_get` — what ffprobe prints — gives the angle
    /// counterclockwise, so -90 for a quarter turn clockwise, which is what
    /// a phone's portrait recording carries. (`av_display_rotation_set` is
    /// not its inverse: it writes that matrix for 90.)
    #[test]
    fn ffmpeg_reads_each_turn_as_the_angle_it_makes() {
        for (degrees, rotation) in [
            (0.0, Rotation::None),
            (-90.0, Rotation::Clockwise90),
            (180.0, Rotation::Half),
            (90.0, Rotation::Clockwise270),
        ] {
            let matrix = Orientation::rotated(rotation).matrix();
            // SAFETY: nine `i32`s, which is what the function reads.
            let read = unsafe { ffi::av_display_rotation_get(matrix.as_ptr()) };
            assert!(
                (read.abs() - f64::abs(degrees)).abs() < 1e-9
                    && (degrees == 180.0 || read == degrees),
                "{rotation:?} reads as {read}"
            );
        }
        let mut skewed = [0i32; 9];
        // SAFETY: nine `i32`s, which is what the function writes.
        unsafe { ffi::av_display_rotation_set(skewed.as_mut_ptr(), 30.0) };
        assert!(Orientation::of_matrix(&skewed).is_err());
    }

    /// Each of the eight reads back from the matrix it writes.
    #[test]
    fn each_orientation_survives_its_matrix() {
        for orientation in Orientation::ALL {
            assert_eq!(
                Orientation::of_matrix(&orientation.matrix()),
                Ok(orientation)
            );
        }
    }

    /// A quarter turn clockwise puts the stored top-left corner top right,
    /// with the sides the other way round; mirrored first, bottom right.
    #[test]
    fn a_box_goes_where_the_turn_takes_it() {
        let corner = [0.0, 0.0, 0.25, 0.5];
        let turned = Orientation::rotated(Rotation::Clockwise90);
        assert_eq!(turned.display_size(1080, 1920), (1920, 1080));
        assert!(near(turned.to_display(corner), [0.5, 0.0, 0.5, 0.25]));
        let mirrored = Orientation {
            rotation: Rotation::Clockwise90,
            mirrored: true,
        };
        assert!(near(mirrored.to_display(corner), [0.5, 0.75, 0.5, 0.25]));
        for orientation in Orientation::ALL {
            let shown = orientation.to_display([0.1, 0.2, 0.3, 0.4]);
            assert!(near(orientation.from_display(shown), [0.1, 0.2, 0.3, 0.4]));
        }
    }

    /// The sampling reads each shown pixel from where the turn took it.
    #[test]
    fn sampling_finds_each_shown_pixel_where_it_is_stored() {
        let (width, height) = (6u32, 4u32);
        for orientation in Orientation::ALL {
            let [xx, xy, x0, yx, yy, y0] = orientation.sampling(width, height);
            let shown = orientation.display_size(width, height);
            for y in 0..shown.1 as i32 {
                for x in 0..shown.0 as i32 {
                    let stored = (xx * x + xy * y + x0, yx * x + yy * y + y0);
                    // The stored pixel's centre, shown, is this pixel's.
                    let [left, top, ..] = orientation.to_display([
                        (stored.0 as f32 + 0.5) / width as f32,
                        (stored.1 as f32 + 0.5) / height as f32,
                        0.0,
                        0.0,
                    ]);
                    assert_eq!(
                        (
                            (left * shown.0 as f32 - 0.5).round() as i32,
                            (top * shown.1 as f32 - 0.5).round() as i32
                        ),
                        (x, y),
                        "{orientation:?}"
                    );
                }
            }
        }
    }
}
