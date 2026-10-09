//! [`SwDetectionOverlay`]: what a detector found, drawn onto pictures in
//! system memory.

use std::collections::HashMap;
use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, format::Pixel};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    color::Color,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet,
        PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::Detections,
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::{
    Canvas, Cut, DetectionOverlayOptions, DetectionOverlayOptionsError, Hide, LABEL_CACHE, MaskKey,
    Rect, bt709_limited, cell_edge, cover, fade, hides, marks_turned, rasterize, ten_bit,
};
use crate::elements::Analytics;
use crate::elements::source::{TextFontError, TextMask, load_font};
use crate::orientation::Orientations;

/// The pixel formats it draws on.
const LAYOUTS: PixelLayoutSet = PixelLayoutSet::from_slice(&[
    PixelLayout::Nv12,
    PixelLayout::Yuv420p,
    PixelLayout::Rgb24,
    PixelLayout::Bgra,
]);

/// Errors specific to `SwDetectionOverlay`. Converts into the crate-wide
/// `Error` via `?`.
#[derive(Debug, ThisError)]
pub enum SwDetectionOverlayError {
    /// Something other than a decoded picture arrived.
    #[error("SwDetectionOverlay only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// A picture in a format it does not draw on.
    #[error(
        "SwDetectionOverlay draws on NV12, YUV 4:2:0, RGB24 or BGRA, and hides in P010 where it draws nothing, not {0:?}"
    )]
    UnsupportedFormat(Pixel),
    /// The label font is not a TrueType or OpenType font.
    #[error("the label font is not a TrueType or OpenType font")]
    LabelFont,
    /// Options it cannot be made of.
    #[error(transparent)]
    Options(#[from] DetectionOverlayOptionsError),
}

impl From<TextFontError> for SwDetectionOverlayError {
    fn from(error: TextFontError) -> Self {
        match error {
            TextFontError::Size(size) => DetectionOverlayOptionsError::LabelSize(size).into(),
            TextFontError::Font(_) => Self::LabelFont,
        }
    }
}

/// Draws the [`Detections`] each picture carries onto a copy of it, in
/// system memory — each class as its
/// [`ClassRule`](super::ClassRule) in the options says: its box and label drawn,
/// or it hidden by a mosaic, a blur or a fill, or both — and
/// hands the copy on, carrying the same `Detections`.
///
/// It takes NV12, YUV 4:2:0, RGB24 or BGRA pictures of any size, and draws
/// at that size: the boxes are fractions of the picture, so it goes before
/// or after a scaler alike. A picture carrying nothing to draw is handed on
/// as it came, not copied. The copy is what keeps the boxes off the picture
/// it was handed, which a Tee in front of it shares with another branch.
pub struct SwDetectionOverlay(FilterStage<Overlaying>);

filter_stage!(SwDetectionOverlay);

/// What a [`SwDetectionOverlay`] does with each picture.
struct Overlaying {
    name: Arc<str>,
    pp_log: PpLog,
    options: DetectionOverlayOptions,
    font: Option<ab_glyph::FontArc>,
    /// Masks already rasterized — labels by their text, the pieces of
    /// zones and lines by where they are; `None` for one with nothing to
    /// draw.
    masks: HashMap<MaskKey, Option<TextMask>>,
    /// Whether it has said that a rule names a class the detections
    /// carry no names for.
    warned_names: bool,
    /// How each picture is turned to be shown.
    orientations: Orientations,
}

impl SwDetectionOverlay {
    /// An overlay drawing as `options` say.
    ///
    /// # Errors
    ///
    /// Options it cannot be made of — a label with no font, a class named
    /// twice — or a font that cannot be read.
    pub fn new(
        name: impl Into<String>,
        options: DetectionOverlayOptions,
    ) -> std::result::Result<Self, SwDetectionOverlayError> {
        let name: Arc<str> = name.into().into();
        options.check()?;
        let pp_log = element_pp_log(ElementType::SwDetectionOverlay, &name, None);
        // Read once at any size: each label is rasterized at its own.
        let font = options
            .font
            .as_ref()
            .map(|font| load_font(font.clone(), 16.0))
            .transpose()?;
        pp_info!(
            pp_log: &pp_log,
            "opened: {} rules, others {:?}, font={}",
            options.rules.len(),
            options.others,
            font.is_some()
        );
        Ok(Self(FilterStage::new(Overlaying {
            name,
            pp_log,
            options,
            font,
            masks: HashMap::new(),
            warned_names: false,
            orientations: Orientations::default(),
        })))
    }
}

/// Where a picture's colour is, and how it is laid out.
#[derive(Debug, Clone, Copy)]
enum Planes {
    /// One plane of `bytes`-byte pixels, red, green and blue at those
    /// offsets, and alpha at `alpha` where there is one.
    Packed {
        bytes: usize,
        red: usize,
        green: usize,
        blue: usize,
        alpha: Option<usize>,
    },
    /// Luma, then Cb and Cr interleaved per 2x2 block.
    Nv12,
    /// Luma, then Cb, then Cr, each per 2x2 block.
    Yuv420p,
    /// As NV12, two bytes a sample, ten bits at the top: hidden in, never
    /// drawn on.
    P010,
}

impl Planes {
    fn of(format: Pixel) -> Option<Self> {
        Some(match format {
            Pixel::RGB24 => Self::Packed {
                bytes: 3,
                red: 0,
                green: 1,
                blue: 2,
                alpha: None,
            },
            Pixel::BGRA => Self::Packed {
                bytes: 4,
                red: 2,
                green: 1,
                blue: 0,
                alpha: Some(3),
            },
            Pixel::NV12 => Self::Nv12,
            Pixel::YUV420P => Self::Yuv420p,
            Pixel::P010LE => Self::P010,
            _ => return None,
        })
    }

    fn block(self) -> u32 {
        match self {
            Self::Packed { .. } => 1,
            Self::Nv12 | Self::Yuv420p | Self::P010 => 2,
        }
    }
}

/// `destination` moved toward `source` by `coverage` of 255.
fn mix(destination: u8, source: u8, coverage: u8) -> u8 {
    let (d, s, a) = (
        u32::from(destination),
        u32::from(source),
        u32::from(coverage),
    );
    ((d * (255 - a) + s * a + 127) / 255) as u8
}

/// Draws `color` over `rect` of `frame`, through `coverage` where it is
/// given — a mask `rect.width` wide, one byte a pixel — and solidly where
/// not. `rect` is on whole blocks of `planes`.
fn paint(
    frame: &mut ffmpeg::frame::Video,
    planes: Planes,
    rect: Rect,
    color: Color,
    coverage: Option<(&[u8], usize)>,
) {
    let at = |x: usize, y: usize| coverage.map_or(255, |(mask, stride)| mask[y * stride + x]);
    let (x0, y0) = (rect.x as usize, rect.y as usize);
    let (width, height) = (rect.width as usize, rect.height as usize);
    match planes {
        Planes::Packed {
            bytes,
            red,
            green,
            blue,
            alpha,
        } => {
            let stride = frame.stride(0);
            let data = frame.data_mut(0);
            for y in 0..height {
                let row = (y0 + y) * stride;
                for x in 0..width {
                    let a = at(x, y);
                    let pixel = &mut data[row + (x0 + x) * bytes..][..bytes];
                    pixel[red] = mix(pixel[red], color.red, a);
                    pixel[green] = mix(pixel[green], color.green, a);
                    pixel[blue] = mix(pixel[blue], color.blue, a);
                    if let Some(alpha) = alpha {
                        pixel[alpha] = pixel[alpha].max(a);
                    }
                }
            }
        }
        Planes::Nv12 | Planes::Yuv420p => {
            let (luma, cb, cr) = bt709_limited(color);
            let stride = frame.stride(0);
            let data = frame.data_mut(0);
            for y in 0..height {
                let row = (y0 + y) * stride + x0;
                for x in 0..width {
                    data[row + x] = mix(data[row + x], luma, at(x, y));
                }
            }
            // A chroma sample covers a 2x2 block; it takes the block's
            // average coverage, as the CUDA masks do.
            let block = |x: usize, y: usize| {
                let sum: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .map(|&(dx, dy)| u32::from(at(x * 2 + dx, y * 2 + dy)))
                    .sum();
                ((sum + 2) / 4) as u8
            };
            let (cx0, cy0) = (x0 / 2, y0 / 2);
            if matches!(planes, Planes::Nv12) {
                let stride = frame.stride(1);
                let data = frame.data_mut(1);
                for y in 0..height / 2 {
                    let row = (cy0 + y) * stride + cx0 * 2;
                    for x in 0..width / 2 {
                        let a = block(x, y);
                        data[row + x * 2] = mix(data[row + x * 2], cb, a);
                        data[row + x * 2 + 1] = mix(data[row + x * 2 + 1], cr, a);
                    }
                }
            } else {
                for (plane, value) in [(1, cb), (2, cr)] {
                    let stride = frame.stride(plane);
                    let data = frame.data_mut(plane);
                    for y in 0..height / 2 {
                        let row = (cy0 + y) * stride + cx0;
                        for x in 0..width / 2 {
                            data[row + x] = mix(data[row + x], value, block(x, y));
                        }
                    }
                }
            }
        }
        // Nothing is drawn on P010, and its fills are painted as cells: see
        // `Overlaying::draw`.
        Planes::P010 => {}
    }
}

/// One plane of a box to hide: `size` samples of `channels` channels each
/// from byte `offset` of a plane, rows `stride` apart, cut into `cells`
/// across and down — and only as much of each sample covered as [`cover`]
/// says of `cut`. A channel is a byte, or with `wide` two, ten bits at the
/// top, as P010 lays them.
#[derive(Clone, Copy)]
struct CellPlane {
    stride: usize,
    offset: usize,
    size: (u32, u32),
    channels: usize,
    cells: (u32, u32),
    smooth: bool,
    cut: Cut,
    wide: bool,
}

/// Paints each cell of a plane its mean — or with `smooth` its mean blended
/// into its neighbours' — as the CUDA overlay's `cell_means` and
/// `cell_paint` do, in the same `f32` operations in the same order, so
/// that the two write the same bytes. With `fill`, each cell is painted
/// those values instead of its mean: a fill cut to an ellipse, as one
/// cell.
fn hide_plane(data: &mut [u8], plane: CellPlane, fill: Option<&[f32]>) {
    let CellPlane {
        stride,
        offset,
        size: (width, height),
        channels,
        cells: (across, down),
        smooth,
        cut,
        wide,
    } = plane;
    let bytes = if wide { 2 } else { 1 };
    let at = |x: u32, y: u32| offset + y as usize * stride + x as usize * channels * bytes;
    // A channel's value: its byte, or its two bytes' top ten bits.
    let read = |data: &[u8], at: usize, channel: usize| {
        if wide {
            f32::from(u16::from_le_bytes([data[at + 2 * channel], data[at + 2 * channel + 1]]) >> 6)
        } else {
            f32::from(data[at + channel])
        }
    };
    let write = |data: &mut [u8], at: usize, channel: usize, value: f32| {
        let value = value.round_ties_even();
        if wide {
            let sample = (value as u16) << 6;
            data[at + 2 * channel..at + 2 * channel + 2].copy_from_slice(&sample.to_le_bytes());
        } else {
            data[at + channel] = value as u8;
        }
    };
    let means = match fill {
        Some(values) => values.to_vec(),
        None => {
            let mut means = vec![0.0f32; across as usize * down as usize * channels];
            for cy in 0..down {
                let (y0, y1) = (cell_edge(cy, down, height), cell_edge(cy + 1, down, height));
                for cx in 0..across {
                    let (x0, x1) = (
                        cell_edge(cx, across, width),
                        cell_edge(cx + 1, across, width),
                    );
                    let mut sums = [0.0f32; 4];
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let sample = at(x, y);
                            for (channel, sum) in sums.iter_mut().enumerate().take(channels) {
                                *sum += read(data, sample, channel);
                            }
                        }
                    }
                    let reciprocal = 1.0f32 / ((x1 - x0) * (y1 - y0)) as f32;
                    let first = (cy * across + cx) as usize * channels;
                    for (mean, sum) in means[first..first + channels].iter_mut().zip(sums) {
                        *mean = sum * reciprocal;
                    }
                }
            }
            means
        }
    };
    // Where a sample sits among the cells' centres along a side: the cells
    // either side and how far it is from the first's centre to the second's.
    let between = |sample: u32, cells: u32, length: u32| {
        let last = (cells - 1) as f32;
        let at = ((sample as f32 + 0.5) * cells as f32 / length as f32 - 0.5)
            .max(0.0)
            .min(last);
        let first = at.floor() as u32;
        (first, (first + 1).min(cells - 1), at - first as f32)
    };
    for y in 0..height {
        for x in 0..width {
            let Some(alpha) = cover(x, y, width, height, cut) else {
                continue;
            };
            let sample = at(x, y);
            // Where the edge fades, the cover mixed over what was there.
            let put = |data: &mut [u8], channel: usize, value: f32| {
                let value = if alpha < 1.0 {
                    fade(read(data, sample, channel), value, alpha)
                } else {
                    value
                };
                write(data, sample, channel, value);
            };
            if smooth {
                let (c0, c1, t) = between(x, across, width);
                let (r0, r1, s) = between(y, down, height);
                let mean = |row: u32, column: u32, channel: usize| {
                    means[(row * across + column) as usize * channels + channel]
                };
                for channel in 0..channels {
                    let (m00, m01) = (mean(r0, c0, channel), mean(r0, c1, channel));
                    let (m10, m11) = (mean(r1, c0, channel), mean(r1, c1, channel));
                    let top = m00 + (m01 - m00) * t;
                    let bottom = m10 + (m11 - m10) * t;
                    put(data, channel, top + (bottom - top) * s);
                }
            } else {
                let cx = ((x + 1) * across - 1) / width;
                let cy = ((y + 1) * down - 1) / height;
                let first = (cy * across + cx) as usize * channels;
                for channel in 0..channels {
                    put(data, channel, means[first + channel]);
                }
            }
        }
    }
}

/// Hides `rect` of `frame` by its cells, plane by plane: a picture whose
/// colour is stored per 2x2 block has its colour cut into the same cells
/// at half the size, `rect` being on whole blocks. With `fill`, as one
/// cell that colour — a fill cut to an ellipse, or faded at its edge.
fn hide_cells(
    frame: &mut ffmpeg::frame::Video,
    planes: Planes,
    rect: Rect,
    (cells, smooth, cut): ((u32, u32), bool, Cut),
    fill: Option<Color>,
) {
    let (x, y) = (rect.x as usize, rect.y as usize);
    let size = (rect.width, rect.height);
    let half = (rect.width / 2, rect.height / 2);
    let plane = |stride: usize, offset: usize, size: (u32, u32), channels: usize| CellPlane {
        stride,
        offset,
        size,
        channels,
        cells,
        smooth,
        cut,
        wide: false,
    };
    let transfer = frame.color_transfer_characteristic();
    let (luma, cb, cr) = fill.map_or((0, 0, 0), bt709_limited);
    let values = |bytes: &[u8]| {
        bytes
            .iter()
            .map(|&byte| f32::from(byte))
            .collect::<Vec<_>>()
    };
    match planes {
        Planes::Packed {
            bytes,
            red,
            green,
            blue,
            alpha,
        } => {
            let color = fill.map(|color| {
                let mut pixel = vec![0u8; bytes];
                pixel[red] = color.red;
                pixel[green] = color.green;
                pixel[blue] = color.blue;
                if let Some(alpha) = alpha {
                    pixel[alpha] = 255;
                }
                values(&pixel)
            });
            let stride = frame.stride(0);
            let offset = y * stride + x * bytes;
            hide_plane(
                frame.data_mut(0),
                plane(stride, offset, size, bytes),
                color.as_deref(),
            );
        }
        Planes::Nv12 => {
            let stride = frame.stride(0);
            let luma = fill.map(|_| values(&[luma]));
            hide_plane(
                frame.data_mut(0),
                plane(stride, y * stride + x, size, 1),
                luma.as_deref(),
            );
            let stride = frame.stride(1);
            let chroma = fill.map(|_| values(&[cb, cr]));
            hide_plane(
                frame.data_mut(1),
                plane(stride, (y / 2) * stride + x, half, 2),
                chroma.as_deref(),
            );
        }
        Planes::P010 => {
            // Two bytes a sample: `x` samples are `2x` bytes in, luma and
            // chroma pairs alike.
            let [luma, cb, cr] = fill.map_or([0.0; 3], |color| ten_bit(color, transfer));
            let stride = frame.stride(0);
            hide_plane(
                frame.data_mut(0),
                CellPlane {
                    wide: true,
                    ..plane(stride, y * stride + 2 * x, size, 1)
                },
                fill.map(|_| [luma]).as_ref().map(|values| &values[..]),
            );
            let stride = frame.stride(1);
            hide_plane(
                frame.data_mut(1),
                CellPlane {
                    wide: true,
                    ..plane(stride, (y / 2) * stride + 2 * x, half, 2)
                },
                fill.map(|_| [cb, cr]).as_ref().map(|values| &values[..]),
            );
        }
        Planes::Yuv420p => {
            let stride = frame.stride(0);
            let luma = fill.map(|_| values(&[luma]));
            hide_plane(
                frame.data_mut(0),
                plane(stride, y * stride + x, size, 1),
                luma.as_deref(),
            );
            for (index, value) in [(1, cb), (2, cr)] {
                let stride = frame.stride(index);
                let offset = (y / 2) * stride + x / 2;
                let colour = fill.map(|_| values(&[value]));
                hide_plane(
                    frame.data_mut(index),
                    plane(stride, offset, half, 1),
                    colour.as_deref(),
                );
            }
        }
    }
}

/// The mask `key` names, from `masks` where it is there, else made in
/// `font` and kept there.
fn mask<'a>(
    masks: &'a mut HashMap<MaskKey, Option<TextMask>>,
    font: Option<&ab_glyph::FontArc>,
    pp_log: &PpLog,
    key: &MaskKey,
) -> Option<&'a TextMask> {
    if !masks.contains_key(key) {
        let made = rasterize(key, font)
            .inspect_err(|error| pp_error!(pp_log: pp_log, "{key:?} not drawn: {error:?}"))
            .ok()
            .flatten();
        masks.insert(key.clone(), made);
    }
    masks.get(key)?.as_ref()
}

impl Overlaying {
    /// A copy of `frame` with `detections` and `analytics` drawn on it.
    fn draw(
        &mut self,
        frame: &ffmpeg::frame::Video,
        detections: Option<&Detections>,
        analytics: Option<&Analytics>,
    ) -> std::result::Result<ffmpeg::frame::Video, SwDetectionOverlayError> {
        let planes = Planes::of(frame.format())
            .filter(|planes| !matches!(planes, Planes::P010) || !self.options.draws_any())
            .ok_or(SwDetectionOverlayError::UnsupportedFormat(frame.format()))?;
        let canvas = Canvas {
            width: frame.width(),
            height: frame.height(),
            block: planes.block(),
        };
        // The cache is emptied between pictures, never while one's marks are
        // placed: every mask made for this picture is read after all of
        // them are, and one emptied away in between was drawn as a solid
        // block, or on CUDA panicked the element's thread.
        if self.masks.len() >= LABEL_CACHE {
            self.masks.clear();
        }
        let orientation = self.orientations.of(frame, &self.pp_log);
        let (masks, font, pp_log) = (&mut self.masks, self.font.as_ref(), &self.pp_log);
        let marks = marks_turned(
            canvas,
            orientation,
            &self.options,
            detections,
            analytics,
            &mut |key| mask(masks, font, pp_log, key).map(|mask| (mask.width, mask.height)),
        );
        let mut copy = frame.clone();
        // What is hidden first, so that a box and its label are drawn over
        // it where they are asked for.
        if let Some(detections) = detections {
            for hide in hides(canvas, &self.options, detections) {
                match hide {
                    Hide::Fill { rect, color, cut }
                        if cut.is_box() && !matches!(planes, Planes::P010) =>
                    {
                        paint(&mut copy, planes, rect, color, None)
                    }
                    Hide::Fill { rect, color, cut } => {
                        hide_cells(&mut copy, planes, rect, ((1, 1), false, cut), Some(color))
                    }
                    Hide::Cells {
                        rect,
                        cells,
                        smooth,
                        cut,
                    } => hide_cells(&mut copy, planes, rect, (cells, smooth, cut), None),
                }
            }
        }
        for mark in marks {
            // A mark with a mask is drawn through it or not at all: drawn
            // without, a label was a solid block of the box's colour.
            let mask = match &mark.mask {
                None => None,
                Some(key) => match self.masks.get(key).and_then(Option::as_ref) {
                    Some(mask) => Some(mask),
                    None => {
                        pp_error!(self, "{key:?} not drawn: its mask is gone from the cache");
                        continue;
                    }
                },
            };
            // A mask is read row by row from the rectangle's top-left: the
            // rectangle is never wider than it, so its stride is the mask's.
            let mask = mask.map(|mask| (mask.coverage.as_slice(), mask.width as usize));
            paint(&mut copy, planes, mark.rect, mark.color, mask);
        }
        Ok(copy)
    }
}

impl Element for Overlaying {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwDetectionOverlay
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Overlaying {
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System).with_layouts(LAYOUTS),
        )
    }

    /// The picture it was handed, or a copy of it in the same format:
    /// whatever was promised upstream holds after it, as after a queue.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(SwDetectionOverlayError::UnsupportedBuffer(kind).into());
        };
        let metadata = buf.metadata_arc().cloned();
        let detections = metadata
            .as_deref()
            .and_then(|metadata| metadata.get::<Detections>());
        let analytics = metadata
            .as_deref()
            .and_then(|metadata| metadata.get::<Analytics>());
        if let Some(detections) = detections
            && !self.warned_names
            && self.options.names_unmatched(detections)
        {
            pp_warn!(
                self,
                "a rule names a class, and the detections carry no class names: it matches nothing"
            );
            self.warned_names = true;
        }
        if !self.options.draws(detections, analytics) {
            // Nothing to draw: the same picture, not a copy of it.
            out.push(buf);
            return Ok(());
        }
        let drawn = self
            .draw(frame, detections, analytics)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        let mut output = MediaBuffer::video(drawn);
        output.set_metadata(metadata);
        out.push(output);
        Ok(())
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use std::num::NonZeroU32;

    use super::super::{
        BoxStyle, ClassRule, Hide, Hiding, OverlayParts, RedactStyle, Treatment, box_color, hides,
    };
    use super::*;
    use crate::buffer::Metadata;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, Detection};

    /// Everything `stage` hands on, kept.
    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// A grey `format` picture, 40 by 20.
    fn grey(format: Pixel) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(format, 40, 20);
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        frame.set_pts(Some(3));
        frame
    }

    /// One detection of class 0, from (10, 4) to (30, 16) of a 40 by 20
    /// picture.
    fn found() -> Detections {
        Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("thing")]),
            vec![Detection::new(0, 0.9, 0.25, 0.2, 0.5, 0.6)],
        )
    }

    fn carrying(frame: ffmpeg::frame::Video, detections: Detections) -> MediaBuffer {
        MediaBuffer::video(frame).with_metadata(Metadata::new().with(detections))
    }

    fn luma(buf: &MediaBuffer, x: usize, y: usize) -> u8 {
        let MediaBuffer::Video(frame) = buf else {
            panic!("expected a picture");
        };
        frame.data(0)[y * frame.stride(0) + x]
    }

    #[test]
    fn a_box_is_drawn_on_a_copy_and_the_picture_handed_in_is_untouched() {
        let mut overlay =
            SwDetectionOverlay::new("overlay", DetectionOverlayOptions::default()).unwrap();
        let kept = capture(&mut overlay);
        let input = carrying(grey(Pixel::NV12), found());
        overlay.consume(input.clone()).expect("drawn");

        let output = kept.lock().unwrap().remove(0);
        let (box_luma, _, _) = bt709_limited(box_color(
            super::super::BoxColors::ByClass,
            &found().items[0],
        ));
        assert_eq!(luma(&output, 10, 4), box_luma, "the top-left corner");
        assert_eq!(luma(&output, 29, 15), box_luma, "the bottom-right corner");
        assert_eq!(luma(&output, 20, 10), 128, "inside the box");
        assert_eq!(luma(&output, 5, 10), 128, "outside it");
        assert_eq!(luma(&input, 10, 4), 128, "the picture handed in");
        let MediaBuffer::Video(frame) = &output else {
            unreachable!();
        };
        assert_eq!(frame.pts(), Some(3));
        assert_eq!(
            output.metadata().and_then(|m| m.get::<Detections>()),
            Some(&found()),
            "the detections go on with the copy"
        );
    }

    #[test]
    fn every_format_it_takes_is_drawn_on() {
        let color = Color::new(255, 0, 0);
        let options = DetectionOverlayOptions {
            others: Treatment::boxes(BoxStyle {
                color: super::super::BoxColors::One(color),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        for format in [Pixel::NV12, Pixel::YUV420P, Pixel::RGB24, Pixel::BGRA] {
            let mut overlay = SwDetectionOverlay::new("overlay", options.clone()).unwrap();
            let kept = capture(&mut overlay);
            overlay
                .consume(carrying(grey(format), found()))
                .expect("drawn");
            let MediaBuffer::Video(frame) = kept.lock().unwrap().remove(0) else {
                unreachable!();
            };
            let pixel = |x: usize, y: usize| -> Vec<u8> {
                match format {
                    Pixel::RGB24 => frame.data(0)[y * frame.stride(0) + x * 3..][..3].to_vec(),
                    Pixel::BGRA => frame.data(0)[y * frame.stride(0) + x * 4..][..3].to_vec(),
                    _ => vec![frame.data(0)[y * frame.stride(0) + x]],
                }
            };
            let expected = match format {
                Pixel::RGB24 => vec![255, 0, 0],
                Pixel::BGRA => vec![0, 0, 255],
                _ => vec![bt709_limited(color).0],
            };
            assert_eq!(pixel(10, 4), expected, "{format:?}");
            assert_ne!(pixel(20, 10), expected, "{format:?} inside");
        }
    }

    #[test]
    fn a_picture_with_nothing_to_draw_is_handed_on_as_it_came() {
        let options = DetectionOverlayOptions {
            others: Treatment {
                min_score: 0.95,
                ..Treatment::default()
            },
            ..DetectionOverlayOptions::default()
        };
        let mut overlay = SwDetectionOverlay::new("overlay", options).unwrap();
        let kept = capture(&mut overlay);
        let bare = MediaBuffer::video(grey(Pixel::NV12));
        let unsure = carrying(grey(Pixel::NV12), found());
        overlay.consume(bare.clone()).expect("bare");
        overlay
            .consume(unsure.clone())
            .expect("below the threshold");
        let kept = kept.lock().unwrap();
        for (input, output) in [bare, unsure].iter().zip(kept.iter()) {
            let (MediaBuffer::Video(input), MediaBuffer::Video(output)) = (input, output) else {
                unreachable!();
            };
            assert_eq!(
                crate::buffer::picture_id(input),
                crate::buffer::picture_id(output)
            );
        }
    }

    #[test]
    fn a_format_it_does_not_draw_on_is_refused_and_a_bad_font_too() {
        let mut overlay =
            SwDetectionOverlay::new("overlay", DetectionOverlayOptions::default()).unwrap();
        let error = overlay
            .consume(carrying(grey(Pixel::YUV444P), found()))
            .expect_err("refused");
        assert!(error.to_string().contains("YUV444P"), "{error}");

        let options = DetectionOverlayOptions {
            font: Some(b"not a font".to_vec()),
            ..DetectionOverlayOptions::default()
        };
        assert!(matches!(
            SwDetectionOverlay::new("overlay", options),
            Err(SwDetectionOverlayError::LabelFont)
        ));
    }

    /// With a font, the band above the box is the box's colour and the text
    /// on it is not.
    #[test]
    fn a_label_is_drawn_above_its_box() {
        let Some(font) = super::super::tests::system_font() else {
            eprintln!("skipping: no system font");
            return;
        };
        let color = Color::new(255, 196, 0);
        let options = DetectionOverlayOptions {
            font: Some(font),
            others: Treatment::boxes(BoxStyle {
                color: super::super::BoxColors::One(color),
                label: Some(super::super::LabelStyle::new(12.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        let mut overlay = SwDetectionOverlay::new("overlay", options).unwrap();
        let kept = capture(&mut overlay);
        let mut frame = ffmpeg::frame::Video::new(Pixel::RGB24, 120, 60);
        frame.data_mut(0).fill(128);
        let mut detections = found();
        detections.items[0].y = 0.5;
        detections.items[0].height = 0.4;
        overlay.consume(carrying(frame, detections)).expect("drawn");
        let MediaBuffer::Video(drawn) = kept.lock().unwrap().remove(0) else {
            unreachable!();
        };
        let green = |x: usize, y: usize| drawn.data(0)[y * drawn.stride(0) + x * 3 + 1];
        // The box starts at (30, 30); the band ends on the row above it.
        assert_eq!(green(30, 29), 196, "the band's padding");
        assert_eq!(green(30, 30), 196, "the box's line");
        let top = (0..30).find(|&y| green(30, y) == 196).expect("a band");
        let right = (30..120)
            .take_while(|&x| green(x, top) == 196)
            .last()
            .unwrap();
        let written = (top..30)
            .flat_map(|y| (30..=right).map(move |x| (x, y)))
            .filter(|&(x, y)| green(x, y) < 100)
            .count();
        assert!(written > 10, "the text is drawn ({written} dark pixels)");
    }

    /// A picture's labels are all still there when it is drawn, the cache
    /// filling up on the way: it was emptied mid-picture, and the masks
    /// made before were drawn as solid blocks.
    /// A picture stored turned is drawn on as it is shown: boxes and their
    /// labels, drawn on it and turned back the way it is shown, are the
    /// very bytes drawn on the picture stored the right way up — the
    /// labels upright and above their boxes as the picture is seen — for
    /// each of the eight ways it can be turned.
    #[test]
    fn a_turned_picture_is_drawn_on_as_it_is_shown() {
        use super::super::tests::{every_orientation, system_font, turn};
        let Some(font) = system_font() else {
            eprintln!("skipping: no system font to draw labels in");
            return;
        };
        let options = DetectionOverlayOptions {
            font: Some(font),
            others: Treatment::boxes(BoxStyle {
                line_width: 2,
                label: Some(super::super::LabelStyle::new(14.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        // As shown: one box low in the picture, one at its top edge, whose
        // label goes inside it.
        let shown = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("thing")]),
            vec![
                Detection::new(0, 0.9, 0.25, 0.5, 0.5, 0.375),
                Detection::new(0, 0.6, 0.625, 0.0, 0.25, 0.25),
            ],
        );
        for format in [Pixel::NV12, Pixel::BGRA] {
            let mut upright = ffmpeg::frame::Video::new(format, 160, 96);
            for plane in 0..upright.planes() {
                for (index, byte) in upright.data_mut(plane).iter_mut().enumerate() {
                    *byte = (index * 7 % 251) as u8;
                }
            }
            let draw = |picture: ffmpeg::frame::Video, found: Detections| {
                let mut overlay = SwDetectionOverlay::new("overlay", options.clone()).unwrap();
                let kept = capture(&mut overlay);
                overlay.consume(carrying(picture, found)).expect("drawn");
                let MediaBuffer::Video(drawn) = kept.lock().unwrap().remove(0) else {
                    panic!("a picture");
                };
                (**drawn).clone()
            };
            let expected = draw(upright.clone(), shown.clone());
            for orientation in every_orientation() {
                let mut stored = shown.clone();
                for item in &mut stored.items {
                    [item.x, item.y, item.width, item.height] =
                        orientation.from_display([item.x, item.y, item.width, item.height]);
                }
                let drawn = draw(turn(&upright, orientation, false), stored);
                let back = turn(&drawn, orientation, true);
                for plane in 0..expected.planes() {
                    let rows = if plane == 0 { 96 } else { 48 };
                    let bytes = if format == Pixel::NV12 { 160 } else { 640 };
                    for row in 0..rows {
                        let at = |frame: &ffmpeg::frame::Video| {
                            frame.data(plane)[row * frame.stride(plane)..][..bytes].to_vec()
                        };
                        assert_eq!(
                            at(&back),
                            at(&expected),
                            "{format:?} {orientation:?}: plane {plane}, row {row}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_full_cache_keeps_every_mask_of_the_picture_being_drawn() {
        let Some(font) = super::super::tests::system_font() else {
            eprintln!("skipping: no system font");
            return;
        };
        let options = DetectionOverlayOptions {
            font: Some(font),
            others: Treatment::boxes(BoxStyle {
                label: Some(super::super::LabelStyle::new(12.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        let font = options
            .font
            .as_ref()
            .map(|font| load_font(font.clone(), 16.0))
            .transpose()
            .unwrap();
        let mut overlaying = Overlaying {
            name: Arc::from("overlay"),
            pp_log: element_pp_log(ElementType::SwDetectionOverlay, "overlay", None),
            options,
            font,
            masks: (1..LABEL_CACHE)
                .map(|n| {
                    let text = format!("earlier {n}");
                    (MaskKey::Text { text, size: 0 }, None)
                })
                .collect(),
            warned_names: false,
            orientations: Orientations::default(),
        };
        let names: Vec<Arc<str>> = ["one", "two", "three"].map(Arc::from).to_vec();
        let items = (0..3)
            .map(|class| Detection::new(class, 0.9, 0.1 + class as f32 * 0.3, 0.5, 0.2, 0.4))
            .collect();
        let detections = Detections::new("test", Arc::from(names), items);
        let frame = ffmpeg::frame::Video::new(Pixel::RGB24, 320, 120);
        overlaying
            .draw(&frame, Some(&detections), None)
            .expect("drawn");
        let labels = overlaying
            .masks
            .keys()
            .filter(|key| matches!(key, MaskKey::Text { text, .. } if !text.starts_with("earlier")))
            .count();
        assert_eq!(labels, 3, "every label of the picture is kept");
    }

    #[test]
    fn mixing_is_exact_at_the_ends() {
        assert_eq!(mix(10, 200, 0), 10);
        assert_eq!(mix(10, 200, 255), 200);
        assert_eq!(mix(0, 255, 128), 128);
    }

    /// With `analytics`, a picture carrying zones and lines and no box is
    /// drawn on: the line in cyan where it was counted, the zone's edge in
    /// amber, the rest as it was. Without, it is handed on as it came.
    #[test]
    fn zones_and_lines_are_drawn_where_they_were_counted() {
        use crate::elements::{Analytics, LineCount};
        let mut analytics = super::super::tests::watched();
        analytics.lines = vec![LineCount {
            name: Arc::from("gate"),
            start: (0.5, 0.5),
            end: (0.9, 0.5),
            forward: 0,
            backward: 0,
            crossed: Vec::new(),
        }];
        let picture = || {
            let mut frame = ffmpeg::frame::Video::new(Pixel::BGRA, 80, 40);
            frame.data_mut(0).fill(128);
            MediaBuffer::video(frame).with_metadata(Metadata::new().with(analytics.clone()))
        };
        let options = DetectionOverlayOptions {
            parts: OverlayParts::all(),
            ..DetectionOverlayOptions::default()
        };
        let mut overlay = SwDetectionOverlay::new("overlay", options).unwrap();
        let kept = capture(&mut overlay);
        overlay.consume(picture()).expect("drawn");
        let output = kept.lock().unwrap().remove(0);
        let MediaBuffer::Video(frame) = &output else {
            unreachable!();
        };
        let pixel = |x: usize, y: usize| frame.data(0)[y * frame.stride(0) + x * 4..][..3].to_vec();
        assert_eq!(pixel(56, 20), vec![255, 229, 0], "on the line");
        assert_eq!(pixel(56, 19), vec![255, 229, 0], "its other row");
        assert_eq!(pixel(56, 23), vec![128, 128, 128], "off it");
        assert_eq!(pixel(20, 4), vec![0, 196, 255], "the zone's top edge");
        assert_eq!(pixel(20, 20), vec![128, 128, 128], "inside the zone");
        assert!(
            output
                .metadata()
                .and_then(|m| m.get::<Analytics>())
                .is_some(),
            "the analytics go on with the copy"
        );

        let mut plain =
            SwDetectionOverlay::new("plain", DetectionOverlayOptions::default()).unwrap();
        let kept = capture(&mut plain);
        let input = picture();
        let MediaBuffer::Video(before) = &input else {
            unreachable!();
        };
        let id = crate::buffer::picture_id(before);
        plain.consume(input).expect("handed on");
        let MediaBuffer::Video(after) = kept.lock().unwrap().remove(0) else {
            unreachable!();
        };
        assert_eq!(
            crate::buffer::picture_id(&after),
            id,
            "not asked: as it came"
        );
    }

    /// The first byte of the sample at (`x`, `y`): luma, or a packed
    /// picture's first colour.
    fn sample(buf: &MediaBuffer, x: usize, y: usize) -> u8 {
        let MediaBuffer::Video(frame) = buf else {
            panic!("expected a picture");
        };
        let bytes = if frame.format() == Pixel::RGB24 { 3 } else { 1 };
        frame.data(0)[y * frame.stride(0) + x * bytes]
    }

    /// A picture whose every third column, from the second, is white and
    /// the rest black, 48 by 24, the box from (12, 6) to (36, 18) on it.
    fn stripes(format: Pixel) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(format, 48, 24);
        let white = |x: usize| if x % 3 == 1 { 255 } else { 0 };
        let stride = frame.stride(0);
        let bytes = if format == Pixel::RGB24 { 3 } else { 1 };
        for y in 0..24 {
            for x in 0..48 {
                for byte in 0..bytes {
                    frame.data_mut(0)[y * stride + x * bytes + byte] = white(x);
                }
            }
        }
        for plane in 1..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        carrying(frame, face())
    }

    /// One face of class 0, from (12, 6) to (36, 18) of a 48 by 24 picture.
    fn face() -> Detections {
        Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("face")]),
            vec![Detection::new(0, 0.9, 0.25, 0.25, 0.5, 0.5)],
        )
    }

    /// Options that hide by `style`, with no margin, and draw nothing.
    fn hiding(style: RedactStyle) -> DetectionOverlayOptions {
        DetectionOverlayOptions {
            others: Treatment {
                hide: Some(Hiding {
                    margin: 0.0,
                    ..Hiding::new(style)
                }),
                ..Treatment::none()
            },
            ..DetectionOverlayOptions::default()
        }
    }

    /// What an overlay hiding by `style` makes of `input`.
    fn hidden(style: RedactStyle, input: &MediaBuffer) -> MediaBuffer {
        let mut overlay = SwDetectionOverlay::new("overlay", hiding(style)).unwrap();
        let kept = capture(&mut overlay);
        overlay.consume(input.clone()).expect("hidden");
        kept.lock().unwrap().remove(0)
    }

    /// A mosaic paints every cell the mean of what was under it: the box,
    /// 24 by 12, cut into two cells down its shorter side and four across,
    /// six columns each — two white of every six — reads a third white all
    /// over, whatever the stripes were; outside it, and on the picture
    /// handed in, the stripes are as they were.
    #[test]
    fn a_mosaic_paints_each_cell_its_mean_on_a_copy() {
        let style = RedactStyle::Mosaic {
            cells: NonZeroU32::new(2).unwrap(),
            min_cell: 1,
        };
        for format in [Pixel::RGB24, Pixel::NV12] {
            let input = stripes(format);
            let output = hidden(style, &input);
            for y in 6..18 {
                for x in 12..36 {
                    assert_eq!(sample(&output, x, y), 85, "{format:?} at ({x}, {y})");
                }
            }
            assert_eq!(sample(&output, 10, 10), 255, "{format:?}: outside the box");
            assert_eq!(sample(&output, 13, 3), 255, "{format:?}: above it");
            assert_eq!(
                sample(&input, 13, 10),
                255,
                "{format:?}: the picture handed in"
            );
        }
    }

    /// A blur paints the same cells' means blended into each other: on a
    /// box whose left half is black and right half white, the mosaic jumps
    /// from one to the other at the cells' edge, and the blur passes
    /// through greys between the two cells' centres.
    #[test]
    fn a_blur_blends_the_cells_where_a_mosaic_steps() {
        let mut frame = ffmpeg::frame::Video::new(Pixel::RGB24, 48, 24);
        let stride = frame.stride(0);
        for y in 0..24 {
            for x in 0..48 {
                let value = if x < 24 { 0 } else { 255 };
                frame.data_mut(0)[y * stride + x * 3..][..3].fill(value);
            }
        }
        let input = carrying(frame, face());
        let row = |style| {
            let output = hidden(style, &input);
            (12..36)
                .map(|x| sample(&output, x, 12))
                .collect::<Vec<u8>>()
        };
        let cells = NonZeroU32::new(1).unwrap();
        let mosaic = row(RedactStyle::Mosaic {
            cells,
            min_cell: 12,
        });
        let blur = row(RedactStyle::Blur {
            cells,
            min_cell: 12,
        });
        assert_eq!(mosaic, [[0u8; 12], [255u8; 12]].concat(), "two cells, flat");
        assert!(
            blur.windows(2).all(|pair| pair[0] <= pair[1]),
            "rising: {blur:?}"
        );
        let greys = blur.iter().filter(|&&v| v != 0 && v != 255).count();
        assert!(greys >= 10, "blended between the centres: {blur:?}");
        assert_eq!(
            (blur[0], blur[23]),
            (0, 255),
            "each cell's own mean at its centre and beyond"
        );
    }

    /// A fill leaves nothing of what was under the box.
    #[test]
    fn a_fill_covers_the_box() {
        let output = hidden(RedactStyle::Fill(Color::BLACK), &stripes(Pixel::RGB24));
        for y in 6..18 {
            for x in 12..36 {
                assert_eq!(sample(&output, x, y), 0, "at ({x}, {y})");
            }
        }
    }

    /// An ellipse covers what is inside it and leaves the box's corners: a
    /// fill of the box at (12, 6) to (36, 18) blackens its middle and every
    /// pixel whose centre is inside the ellipse, and nothing else — on
    /// packed RGB pixel by pixel, on NV12 the colour a 2x2 block at a time.
    #[test]
    fn an_ellipse_covers_only_what_is_inside_it() {
        for format in [Pixel::RGB24, Pixel::NV12] {
            let input = stripes(format);
            let options = DetectionOverlayOptions {
                others: Treatment {
                    hide: Some(Hiding {
                        margin: 0.0,
                        shape: super::super::HideShape::Ellipse,
                        ..Hiding::new(RedactStyle::Fill(Color::BLACK))
                    }),
                    ..Treatment::none()
                },
                ..DetectionOverlayOptions::default()
            };
            let mut overlay = SwDetectionOverlay::new("overlay", options).unwrap();
            let kept = capture(&mut overlay);
            overlay.consume(input.clone()).expect("hidden");
            let output = kept.lock().unwrap().remove(0);
            let black = if format == Pixel::NV12 { 16 } else { 0 };
            for y in 0..24u32 {
                for x in 0..48u32 {
                    let inside = (12..36).contains(&x)
                        && (6..18).contains(&y)
                        && super::super::inside_ellipse(x - 12, y - 6, 24, 12);
                    let (now, was) = (
                        sample(&output, x as usize, y as usize),
                        sample(&input, x as usize, y as usize),
                    );
                    if inside {
                        assert_eq!(now, black, "{format:?} ({x}, {y}) is covered");
                    } else {
                        assert_eq!(now, was, "{format:?} ({x}, {y}) is left");
                    }
                }
            }
            // The middle is covered, the corners of the box are not.
            assert!(super::super::inside_ellipse(12, 6, 24, 12));
            assert!(!super::super::inside_ellipse(0, 0, 24, 12));
        }
    }

    /// In a P010 picture — ten bits at the top of two bytes — a mosaic paints
    /// each cell the mean of its ten-bit values, the six bits below left
    /// naught, and a fill puts black at ten-bit black; a picture of it is
    /// refused where anything would be drawn.
    #[test]
    fn a_p010_picture_is_hidden_in_ten_bits() {
        let mut frame = ffmpeg::frame::Video::new(Pixel::P010LE, 48, 24);
        for plane in 0..2 {
            let stride = frame.stride(plane);
            let rows = if plane == 0 { 24 } else { 12 };
            for y in 0..rows {
                for x in 0..48 {
                    // Luma 100 or 700 in stripes, chroma 512; each with
                    // low bits set, which a mean must not read.
                    let value: u16 = if plane == 0 && x % 2 == 1 {
                        700
                    } else if plane == 0 {
                        100
                    } else {
                        512
                    };
                    let sample = (value << 6) | 0b10_1010;
                    frame.data_mut(plane)[y * stride + x * 2..][..2]
                        .copy_from_slice(&sample.to_le_bytes());
                }
            }
        }
        let input = carrying(frame, face());
        let read = |buf: &MediaBuffer, x: usize, y: usize| {
            let MediaBuffer::Video(frame) = buf else {
                panic!("a picture");
            };
            u16::from_le_bytes([
                frame.data(0)[y * frame.stride(0) + x * 2],
                frame.data(0)[y * frame.stride(0) + x * 2 + 1],
            ])
        };
        let mosaic = hidden(RedactStyle::mosaic(), &input);
        // The box from (12, 6) to (36, 18): every sample the stripes' mean,
        // 400, in its top ten bits.
        for y in 6..18 {
            for x in 12..36 {
                assert_eq!(read(&mosaic, x, y), 400 << 6, "({x}, {y})");
            }
        }
        assert_eq!(
            read(&mosaic, 0, 0),
            (100 << 6) | 0b10_1010,
            "outside, as it was"
        );
        let filled = hidden(RedactStyle::Fill(Color::BLACK), &input);
        assert_eq!(read(&filled, 20, 10), 64 << 6, "ten-bit black");
        let drawing =
            SwDetectionOverlay::new("overlay", DetectionOverlayOptions::default()).unwrap();
        let mut drawing = drawing;
        assert!(
            drawing.consume(input.clone()).is_err(),
            "boxes are not drawn on P010"
        );
    }

    /// The cells follow the box: its shorter side cut into `cells`, but
    /// none smaller than `min_cell`; the margin grows the box first; a
    /// class no rule hides, or a score below its rule's, is left as it is.
    #[test]
    fn the_cells_follow_the_box_and_the_margin_grows_it() {
        let canvas = Canvas {
            width: 1000,
            height: 1000,
            block: 2,
        };
        let face = |x, y, side| Detection::new(0, 0.9, x, y, side, side);
        let style = RedactStyle::Mosaic {
            cells: NonZeroU32::new(6).unwrap(),
            min_cell: 8,
        };
        let faces = |treatment: Treatment| DetectionOverlayOptions {
            rules: vec![ClassRule::new("face", treatment)],
            others: Treatment::none(),
            ..DetectionOverlayOptions::default()
        };
        let cut = |options: &DetectionOverlayOptions, detection: Detection| {
            let found = Detections::new(
                "test",
                Arc::from(vec![Arc::<str>::from("face"), Arc::<str>::from("plate")]),
                vec![detection],
            );
            hides(canvas, options, &found)
        };
        let square = |x, side| Rect {
            x,
            y: x,
            width: side,
            height: side,
        };
        let near = faces(Treatment {
            hide: Some(Hiding {
                margin: 0.0,
                ..Hiding::new(style)
            }),
            ..Treatment::none()
        });
        assert_eq!(
            cut(&near, face(0.1, 0.1, 0.3)),
            vec![Hide::Cells {
                rect: square(100, 300),
                cells: (6, 6),
                smooth: false,
                cut: Cut {
                    ellipse: false,
                    feather: 0.0,
                },
            }],
            "a near face: six cells of fifty"
        );
        assert_eq!(
            cut(&near, face(0.1, 0.1, 0.02)),
            vec![Hide::Cells {
                rect: square(100, 20),
                cells: (3, 3),
                smooth: false,
                cut: Cut {
                    ellipse: false,
                    feather: 0.0,
                },
            }],
            "a far face: cells of at least eight, about three across"
        );
        let grown = faces(Treatment::hidden(style));
        let Hide::Cells { rect, .. } = cut(&grown, face(0.5, 0.5, 0.2))[0] else {
            panic!("cells");
        };
        assert_eq!(rect, square(480, 240), "a tenth more each way");
        let plate = Detection {
            class_id: 1,
            ..face(0.5, 0.5, 0.2)
        };
        assert!(cut(&grown, plate).is_empty(), "a class no rule hides");
        let sure = faces(Treatment {
            min_score: 0.95,
            ..Treatment::hidden(style)
        });
        assert!(
            cut(&sure, face(0.5, 0.5, 0.2)).is_empty(),
            "below its score"
        );
    }

    #[test]
    fn options_that_cannot_be_hidden_by_are_refused() {
        let options = |margin| DetectionOverlayOptions {
            others: Treatment {
                hide: Some(Hiding {
                    margin,
                    ..Hiding::new(RedactStyle::mosaic())
                }),
                ..Treatment::none()
            },
            ..DetectionOverlayOptions::default()
        };
        for margin in [-0.1, f32::NAN] {
            assert!(matches!(
                SwDetectionOverlay::new("overlay", options(margin)),
                Err(SwDetectionOverlayError::Options(
                    DetectionOverlayOptionsError::Margin(_)
                ))
            ));
        }
        assert!(SwDetectionOverlay::new("overlay", options(0.0)).is_ok());
    }
}
