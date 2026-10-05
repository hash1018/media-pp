//! A picture's brightness, read a region at a time — all a visual tracker
//! needs of it, wherever the picture lives: a plane read in place in system
//! memory, a rectangle copied down from a CUDA surface.

use ffmpeg_next::{self as ffmpeg, format::Pixel};

/// A rectangle of a picture, in whole pixels, inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Area {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

/// The brightness of an [`Area`], 0 to 255, row-major.
#[derive(Debug, Clone)]
pub(super) struct Region {
    pub(super) area: Area,
    pub(super) pixels: Vec<f32>,
}

impl Region {
    /// The brightness at `(x, y)` in picture coordinates, between pixel
    /// centres, and as the nearest edge's beyond the region.
    pub(super) fn at(&self, x: f64, y: f64) -> f32 {
        let Area {
            x: left,
            y: top,
            width,
            height,
        } = self.area;
        let fx = (x - f64::from(left) - 0.5).clamp(0.0, f64::from(width - 1));
        let fy = (y - f64::from(top) - 0.5).clamp(0.0, f64::from(height - 1));
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = (
            (x0 + 1).min(width as usize - 1),
            (y0 + 1).min(height as usize - 1),
        );
        let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
        let w = width as usize;
        let p = |x: usize, y: usize| self.pixels[y * w + x];
        let top_row = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
        let bottom_row = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
        top_row + (bottom_row - top_row) * ty
    }
}

/// A picture whose brightness can be read.
pub(super) trait Luma {
    /// Its width and height.
    fn size(&self) -> (u32, u32);
    /// The brightness of `area`, which is inside the picture; `None` where
    /// it cannot be read.
    fn read(&mut self, area: Area) -> Option<Region>;

    /// The brightness around `centre`, `width` by `height` of it, cut to
    /// the picture, a pixel wider on each side for sampling between them;
    /// `None` where none of it is in the picture.
    fn around(&mut self, centre: (f64, f64), width: f64, height: f64) -> Option<Region> {
        let (picture_width, picture_height) = self.size();
        let left = (centre.0 - width / 2.0 - 1.0).floor().max(0.0) as u32;
        let top = (centre.1 - height / 2.0 - 1.0).floor().max(0.0) as u32;
        let right = ((centre.0 + width / 2.0 + 1.0).ceil().max(0.0) as u32).min(picture_width);
        let bottom = ((centre.1 + height / 2.0 + 1.0).ceil().max(0.0) as u32).min(picture_height);
        if right <= left + 1 || bottom <= top + 1 {
            return None;
        }
        self.read(Area {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        })
    }
}

/// Where a packed RGB pixel's channels are.
#[derive(Debug, Clone, Copy)]
struct Channels {
    bytes: usize,
    red: usize,
    green: usize,
    blue: usize,
}

const BGRA: Channels = Channels {
    bytes: 4,
    red: 2,
    green: 1,
    blue: 0,
};

impl Channels {
    /// Brightness, by BT.709's weights.
    fn luma(&self, pixel: &[u8]) -> f32 {
        0.2126 * f32::from(pixel[self.red])
            + 0.7152 * f32::from(pixel[self.green])
            + 0.0722 * f32::from(pixel[self.blue])
    }
}

/// How a system-memory picture holds its brightness.
#[derive(Debug, Clone, Copy)]
enum Layout {
    /// As its first plane, a byte a pixel.
    Plane,
    /// Packed with its colour.
    Packed(Channels),
}

/// A picture in system memory, read in place.
pub(super) struct SystemLuma<'a> {
    frame: &'a ffmpeg::frame::Video,
    layout: Layout,
}

impl<'a> SystemLuma<'a> {
    /// `frame`'s brightness, where it is in a format this reads: 8-bit
    /// YUV or grey, or packed 8-bit RGB.
    pub(super) fn of(frame: &'a ffmpeg::frame::Video) -> Option<Self> {
        let layout = match frame.format() {
            Pixel::NV12
            | Pixel::NV21
            | Pixel::YUV420P
            | Pixel::YUVJ420P
            | Pixel::YUV422P
            | Pixel::YUV444P
            | Pixel::GRAY8 => Layout::Plane,
            Pixel::RGB24 => Layout::Packed(Channels {
                bytes: 3,
                red: 0,
                green: 1,
                blue: 2,
            }),
            Pixel::BGR24 => Layout::Packed(Channels {
                bytes: 3,
                red: 2,
                green: 1,
                blue: 0,
            }),
            Pixel::BGRA => Layout::Packed(BGRA),
            Pixel::RGBA => Layout::Packed(Channels {
                bytes: 4,
                red: 0,
                green: 1,
                blue: 2,
            }),
            _ => return None,
        };
        Some(Self { frame, layout })
    }
}

impl Luma for SystemLuma<'_> {
    fn size(&self) -> (u32, u32) {
        (self.frame.width(), self.frame.height())
    }

    fn read(&mut self, area: Area) -> Option<Region> {
        let stride = self.frame.stride(0);
        let data = self.frame.data(0);
        let mut pixels = Vec::with_capacity((area.width * area.height) as usize);
        for y in area.y..area.y + area.height {
            let row = &data[y as usize * stride..];
            match self.layout {
                Layout::Plane => pixels.extend(
                    row[area.x as usize..(area.x + area.width) as usize]
                        .iter()
                        .map(|&byte| f32::from(byte)),
                ),
                Layout::Packed(channels) => pixels.extend(
                    (area.x..area.x + area.width)
                        .map(|x| channels.luma(&row[x as usize * channels.bytes..])),
                ),
            }
        }
        Some(Region { area, pixels })
    }
}

/// A CUDA picture, read a rectangle at a time by copying it down.
#[cfg(feature = "cuda")]
pub(super) struct CudaLuma<'a> {
    driver: &'a crate::platform::cuda::driver::CudaDriver,
    /// The plane holding brightness — NV12's luma, or the BGRA pixels —
    /// and its pitch.
    plane: u64,
    pitch: usize,
    bgra: bool,
    size: (u32, u32),
}

#[cfg(feature = "cuda")]
impl<'a> CudaLuma<'a> {
    /// `frame`'s brightness, where it is an NV12 or BGRA CUDA picture.
    pub(super) fn of(
        driver: &'a crate::platform::cuda::driver::CudaDriver,
        frame: &ffmpeg::frame::Video,
    ) -> Option<Self> {
        use crate::platform::cuda::driver::{BgraSurface, Nv12Surface};
        let size = (frame.width(), frame.height());
        match crate::platform::cuda::frame::surface_layout(frame)? {
            Pixel::NV12 => {
                let surface = Nv12Surface::from_frame(frame)?;
                Some(Self {
                    driver,
                    plane: surface.luma,
                    pitch: surface.luma_pitch,
                    bgra: false,
                    size,
                })
            }
            Pixel::BGRA => {
                let surface = BgraSurface::from_frame(frame)?;
                Some(Self {
                    driver,
                    plane: surface.pixels,
                    pitch: surface.pitch,
                    bgra: true,
                    size,
                })
            }
            _ => None,
        }
    }
}

#[cfg(feature = "cuda")]
impl Luma for CudaLuma<'_> {
    fn size(&self) -> (u32, u32) {
        self.size
    }

    fn read(&mut self, area: Area) -> Option<Region> {
        let bytes = if self.bgra { 4 } else { 1 };
        let (width, height) = (area.width as usize, area.height as usize);
        let mut copied = vec![0u8; width * height * bytes];
        self.driver
            .download_rect(
                self.plane,
                self.pitch,
                area.x as usize * bytes,
                area.y as usize,
                width * bytes,
                height,
                &mut copied,
            )
            .ok()?;
        let pixels = if self.bgra {
            let (pixels, _) = copied.as_chunks::<4>();
            pixels.iter().map(|pixel| BGRA.luma(pixel)).collect()
        } else {
            copied.into_iter().map(f32::from).collect()
        };
        Some(Region { area, pixels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_region_is_read_and_sampled_between_its_pixels() {
        let mut frame = ffmpeg::frame::Video::new(Pixel::GRAY8, 8, 4);
        let stride = frame.stride(0);
        for y in 0..4 {
            for x in 0..8 {
                frame.data_mut(0)[y * stride + x] = (x * 10 + y) as u8;
            }
        }
        let mut luma = SystemLuma::of(&frame).unwrap();
        let region = luma
            .read(Area {
                x: 2,
                y: 1,
                width: 3,
                height: 2,
            })
            .unwrap();
        assert_eq!(region.pixels, vec![21.0, 31.0, 41.0, 22.0, 32.0, 42.0]);
        // Pixel (3, 1)'s centre, and halfway to (4, 1).
        assert_eq!(region.at(3.5, 1.5), 31.0);
        assert_eq!(region.at(4.0, 1.5), 36.0);
        // Beyond the region, its edge.
        assert_eq!(region.at(0.0, 1.5), 21.0);
    }

    #[test]
    fn around_a_centre_is_cut_to_the_picture() {
        let frame = ffmpeg::frame::Video::new(Pixel::RGB24, 20, 10);
        let mut luma = SystemLuma::of(&frame).unwrap();
        let region = luma.around((2.0, 2.0), 10.0, 10.0).unwrap();
        assert_eq!((region.area.x, region.area.y), (0, 0));
        assert_eq!((region.area.width, region.area.height), (8, 8));
        assert!(luma.around((-50.0, 5.0), 10.0, 10.0).is_none());
    }
}
