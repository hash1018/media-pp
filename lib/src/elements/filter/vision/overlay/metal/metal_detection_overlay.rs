//! [`MetalDetectionOverlay`]: what a detector found, drawn onto
//! VideoToolbox pictures with Metal.

use std::collections::HashMap;
use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use objc2_metal::{MTLPixelFormat, MTLTexture, MTLTextureUsage};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    color::Color,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::{Detections, VideoToolboxDevice},
    error::Result,
    frame_size::ForSize,
    platform::ffmpeg::AvBufferRef,
    platform::macos::{
        metal::{Kernel, MetalError, MetalGpu, Texture, write_texture},
        pixel_buffer::PixelBuffer,
        videotoolbox::{NotVideoToolbox, create_frames_ctx, sw_format_of},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::{
    Canvas, DetectionOverlayOptions, LABEL_CACHE, Rect, box_color, bt709_limited, drawn,
    label_text, text_color,
};
use crate::elements::source::{TextFontError, load_font, rasterize_coverage};

const SHADER: &str = include_str!("../../../../../shaders/metal/overlay.metal");

/// Errors specific to `MetalDetectionOverlay`. Converts into the crate-wide
/// `Error` via `?`.
#[derive(Debug, ThisError)]
pub enum MetalDetectionOverlayError {
    /// Something other than a decoded picture arrived.
    #[error("MetalDetectionOverlay only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// Not an NV12 or BGRA VideoToolbox picture: what it is, or for a
    /// VideoToolbox picture what it holds.
    #[error("MetalDetectionOverlay takes NV12 or BGRA VideoToolbox pictures, got {0:?}")]
    UnsupportedPicture(ffmpeg::format::Pixel),
    /// A VideoToolbox picture with no frames context to say what it holds,
    /// or no pixel buffer in it.
    #[error("a VideoToolbox picture has no {0}")]
    MissingPixelBuffer(&'static str),
    /// A VideoToolbox picture larger than the pixel buffer behind it.
    #[error("a {picture:?} picture is larger than its {surface:?} pixel buffer")]
    PictureOutsideSurface {
        /// The picture's size.
        picture: (u32, u32),
        /// The pixel buffer's.
        surface: (u32, u32),
    },
    /// Making the frames the copies are drawn on failed.
    #[error("could not make the frames to draw on: {0}")]
    Frames(String),
    /// Taking a frame from those frames failed.
    #[error("failed to take a frame from the VideoToolbox pool (code {0})")]
    FrameGet(i32),
    /// Metal refused a kernel, a texture or a pass.
    #[error(transparent)]
    Metal(#[from] MetalError),
    /// The label size is not a positive number of pixels.
    #[error("label size {0} is not a positive number of pixels")]
    LabelSize(f32),
    /// The label font is not a TrueType or OpenType font.
    #[error("the label font is not a TrueType or OpenType font")]
    LabelFont,
}

impl From<TextFontError> for MetalDetectionOverlayError {
    fn from(error: TextFontError) -> Self {
        match error {
            TextFontError::Size(size) => Self::LabelSize(size),
            TextFontError::Font(_) => Self::LabelFont,
        }
    }
}

/// Draws the [`Detections`] each VideoToolbox picture carries onto a copy
/// of it, on the GPU with Metal — a box around each object, and with
/// [`DetectionOverlayOptions::labels`] its class and score above it — and
/// hands the copy on, carrying the same `Detections`: what
/// [`SwDetectionOverlay`](super::super::SwDetectionOverlay) does in system
/// memory, so that a detector's pictures go on to an encoder, a renderer or
/// a compositor without leaving the GPU.
///
/// It takes NV12 or BGRA VideoToolbox pictures of any size, from any
/// device, since a pixel buffer belongs to none, and draws at that size.
/// The copy is a frame of `device`'s, from a pool of this element's own, and
/// is what keeps the boxes off the picture it was handed, which a Tee in
/// front of it shares with another branch. A picture carrying nothing to
/// draw is handed on as it came.
///
/// Lines and label bands are rectangle fills; a label's text is rasterized
/// on the CPU once per distinct string, uploaded as a coverage mask and
/// blended on the GPU, as the Metal compositor's text layers are. An NV12
/// picture is painted in BT.709 limited-range colour, as the other
/// overlays paint it.
pub struct MetalDetectionOverlay(FilterStage<Overlaying>);

filter_stage!(MetalDetectionOverlay);

/// The kernels, by name.
struct Kernels {
    copy_nv12: Kernel,
    copy_bgra: Kernel,
    paint_nv12: Kernel,
    paint_bgra: Kernel,
}

/// What a [`MetalDetectionOverlay`] does with each picture.
struct Overlaying {
    name: Arc<str>,
    pp_log: PpLog,
    options: DetectionOverlayOptions,
    font: Option<ab_glyph::FontArc>,
    /// Labels already rasterized and uploaded, by their text; `None` for
    /// one with nothing to draw.
    labels: HashMap<String, Option<Texture>>,
    gpu: MetalGpu,
    kernels: Kernels,
    /// What a solid paint binds where a label's mask goes: it is not read,
    /// but a kernel's every texture is bound.
    solid: Texture,
    /// This element's own reference to the device its copies are made on.
    hw_device_ctx: Arc<AvBufferRef>,
    /// The pools the copies come from, one per format, each made for the
    /// size of the pictures arriving — see `ForSize`.
    nv12_frames: ForSize<AvBufferRef>,
    bgra_frames: ForSize<AvBufferRef>,
    /// Reuses only the CPU-side `AVFrame` wrapper; each pixel buffer comes
    /// from a frames pool.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the FFmpeg buffers have no thread affinity, and the Metal objects
// are thread-safe and touched only through `&mut self` — the reasoning
// `MetalPass` gives for its own.
unsafe impl Send for Overlaying {}

impl MetalDetectionOverlay {
    /// An overlay drawing as `options` say, its copies made on `device`.
    ///
    /// # Errors
    ///
    /// The label font or size where `options.labels` gives one that cannot
    /// be drawn with, and Metal's where there is no GPU or a kernel will
    /// not compile.
    pub fn new(
        name: impl Into<String>,
        device: &VideoToolboxDevice,
        options: DetectionOverlayOptions,
    ) -> std::result::Result<Self, MetalDetectionOverlayError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalDetectionOverlay, &name, None);
        let font = options
            .labels
            .as_ref()
            .map(|style| load_font(style.font_data.clone(), style.size))
            .transpose()?;
        let gpu = MetalGpu::new()?;
        let [copy_nv12, copy_bgra, paint_nv12, paint_bgra] = <[Kernel; 4]>::try_from(gpu.kernels(
            SHADER,
            &["copy_nv12", "copy_bgra", "paint_nv12", "paint_bgra"],
        )?)
        .unwrap_or_else(|_| unreachable!("four kernels for four names"));
        let solid = gpu.texture(
            MTLPixelFormat::R8Unorm,
            1,
            1,
            MTLTextureUsage::ShaderRead,
            true,
        )?;
        write_texture(&solid, &[255], 1, 1, 1);
        pp_info!(
            pp_log: &pp_log,
            "opened: NV12 or BGRA, line_width={}, min_score={}, labels={}",
            options.line_width,
            options.min_score,
            font.is_some()
        );
        Ok(Self(FilterStage::new(Overlaying {
            name,
            pp_log,
            options,
            font,
            labels: HashMap::new(),
            gpu,
            kernels: Kernels {
                copy_nv12,
                copy_bgra,
                paint_nv12,
                paint_bgra,
            },
            solid,
            hw_device_ctx: device.retain(),
            nv12_frames: ForSize::new(),
            bgra_frames: ForSize::new(),
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })))
    }
}

/// One rectangle to paint: where, in what colour, and through which label
/// mask, if any.
struct Stroke {
    rect: Rect,
    color: Color,
    mask: Option<String>,
}

/// The kernel parameters for `rect` in `colour`, as the shader's `Paint`.
fn paint(rect: Rect, colour: [f32; 3], masked: bool) -> Vec<u8> {
    [rect.x, rect.y, rect.width, rect.height]
        .iter()
        .flat_map(|word| word.to_ne_bytes())
        .chain(
            [colour[0], colour[1], colour[2], 0.0]
                .iter()
                .flat_map(|value| value.to_ne_bytes()),
        )
        .chain(
            [u32::from(masked), 0, 0, 0]
                .iter()
                .flat_map(|word| word.to_ne_bytes()),
        )
        .collect()
}

impl Overlaying {
    /// Whether the label `text` has a mask to draw, rasterizing and
    /// uploading it where it is not cached yet.
    fn label(&mut self, text: &str) -> Option<(u32, u32)> {
        let font = self.font.as_ref()?;
        let size = self.options.labels.as_ref()?.size;
        if !self.labels.contains_key(text) {
            if self.labels.len() >= LABEL_CACHE {
                self.labels.clear();
            }
            let mask = match rasterize_coverage(font, size, text) {
                Ok(Some(raster)) => self
                    .gpu
                    .texture(
                        MTLPixelFormat::R8Unorm,
                        raster.width,
                        raster.height,
                        MTLTextureUsage::ShaderRead,
                        true,
                    )
                    .inspect(|texture| {
                        write_texture(
                            texture,
                            &raster.coverage,
                            raster.width as usize,
                            raster.width,
                            raster.height,
                        );
                    })
                    .inspect_err(|error| pp_error!(self, "label {text:?} not drawn: {error}"))
                    .ok(),
                Ok(None) => None,
                Err(error) => {
                    pp_error!(self, "label {text:?} not drawn: {error:?}");
                    None
                }
            };
            self.labels.insert(text.to_owned(), mask);
        }
        let mask = self.labels.get(text)?.as_ref()?;
        Some((mask.width() as u32, mask.height() as u32))
    }

    /// A frame of `layout` at `width` x `height` from this element's pool.
    fn frame(
        &mut self,
        layout: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MetalDetectionOverlayError>
    {
        let (device, pp_log) = (&self.hw_device_ctx, &self.pp_log);
        let pool = if layout == ffmpeg::format::Pixel::NV12 {
            &mut self.nv12_frames
        } else {
            &mut self.bgra_frames
        };
        let frames_ctx = pool
            .try_get(width, height, |width, height| {
                pp_info!(pp_log: pp_log, "drawing {layout:?} {width}x{height}");
                // SAFETY: `create_frames_ctx`'s contract is a live device
                // context, which is what the owned `AvBufferRef` is.
                unsafe { create_frames_ctx(device, layout, width, height) }
                    .map_err(|error| MetalDetectionOverlayError::Frames(error.to_string()))
            })?
            .as_ptr();
        let mut destination = self.pool.get();
        // SAFETY: the pooled wrapper's own `AVFrame`, its previous pixel
        // buffer handed back first; the frames context is this element's
        // own, held for its life.
        unsafe {
            let dst = destination.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames_ctx, dst, 0);
            if code < 0 {
                return Err(MetalDetectionOverlayError::FrameGet(code));
            }
        }
        Ok(destination)
    }

    /// What to paint for `detections` on a picture `canvas` describes, in
    /// order: each box's lines, then its label's band and text.
    fn strokes(&mut self, canvas: Canvas, detections: &Detections) -> Vec<Stroke> {
        let score = self
            .options
            .labels
            .as_ref()
            .is_some_and(|style| style.score);
        let mut strokes = Vec::new();
        for detection in drawn(detections, self.options.min_score) {
            let Some(placed) = canvas.place(detection) else {
                continue;
            };
            let color = box_color(self.options.colors, detection.class_id);
            strokes.extend(
                canvas
                    .edges(placed, self.options.line_width)
                    .into_iter()
                    .filter(|edge| !edge.is_empty())
                    .map(|rect| Stroke {
                        rect,
                        color,
                        mask: None,
                    }),
            );
            let text = label_text(detections, detection, score);
            let Some((mask_width, mask_height)) = self.label(&text) else {
                continue;
            };
            let (band, area) = canvas.label(placed, mask_width, mask_height);
            if band.is_empty() {
                continue;
            }
            strokes.push(Stroke {
                rect: band,
                color,
                mask: None,
            });
            if !area.is_empty() {
                strokes.push(Stroke {
                    rect: area,
                    color: text_color(color),
                    mask: Some(text),
                });
            }
        }
        strokes
    }

    /// A copy of `source` with `detections` drawn on it.
    fn draw(
        &mut self,
        source: &ffmpeg::frame::Video,
        detections: &Detections,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MetalDetectionOverlayError>
    {
        let layout = match sw_format_of(source) {
            Ok(layout @ (ffmpeg::format::Pixel::NV12 | ffmpeg::format::Pixel::BGRA)) => layout,
            Ok(other) | Err(NotVideoToolbox::Format(other)) => {
                return Err(MetalDetectionOverlayError::UnsupportedPicture(other));
            }
            Err(NotVideoToolbox::NoFramesContext) => {
                return Err(MetalDetectionOverlayError::MissingPixelBuffer(
                    "frames context",
                ));
            }
        };
        let from = PixelBuffer::of_frame(source)
            .ok_or(MetalDetectionOverlayError::MissingPixelBuffer("pixels"))?;
        let (width, height) = (source.width(), source.height());
        let surface = from.size();
        if width > surface.0 || height > surface.1 {
            return Err(MetalDetectionOverlayError::PictureOutsideSurface {
                picture: (width, height),
                surface,
            });
        }
        let nv12 = layout == ffmpeg::format::Pixel::NV12;
        let canvas = Canvas {
            width,
            height,
            block: if nv12 { 2 } else { 1 },
        };
        let strokes = self.strokes(canvas, detections);
        let mut destination = self.frame(layout, width, height)?;
        let to = PixelBuffer::of_frame(&destination).expect("a frame of this element's own pool");

        // Every texture made before anything is encoded, so a failure leaves
        // no pass half written; each lives until the pass has finished.
        let (read, both) = (
            MTLTextureUsage::ShaderRead,
            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        );
        let (targets, sources) = if nv12 {
            (
                vec![
                    self.gpu.plane(&to, 0, MTLPixelFormat::R8Unorm, both)?,
                    self.gpu.plane(&to, 1, MTLPixelFormat::RG8Unorm, both)?,
                ],
                vec![
                    self.gpu.plane(&from, 0, MTLPixelFormat::R8Unorm, read)?,
                    self.gpu.plane(&from, 1, MTLPixelFormat::RG8Unorm, read)?,
                ],
            )
        } else {
            (
                vec![self.gpu.plane(&to, 0, MTLPixelFormat::BGRA8Unorm, both)?],
                vec![self.gpu.plane(&from, 0, MTLPixelFormat::BGRA8Unorm, read)?],
            )
        };
        let (copy, paint_kernel) = if nv12 {
            (&self.kernels.copy_nv12, &self.kernels.paint_nv12)
        } else {
            (&self.kernels.copy_bgra, &self.kernels.paint_bgra)
        };

        let mut pass = self.gpu.pass()?;
        let whole = Rect {
            x: 0,
            y: 0,
            width,
            height,
        };
        let bound: Vec<&Texture> = targets.iter().chain(&sources).collect();
        pass.dispatch(
            copy,
            &bound,
            Some(&paint(whole, [0.0; 3], false)),
            (width, height),
        );
        for stroke in &strokes {
            let colour = if nv12 {
                let (y, cb, cr) = bt709_limited(stroke.color);
                [y, cb, cr].map(|value| f32::from(value) / 255.0)
            } else {
                [stroke.color.red, stroke.color.green, stroke.color.blue]
                    .map(|value| f32::from(value) / 255.0)
            };
            let mask = stroke
                .mask
                .as_ref()
                .and_then(|text| self.labels.get(text))
                .and_then(Option::as_ref);
            let bound: Vec<&Texture> = targets
                .iter()
                .chain(std::iter::once(mask.unwrap_or(&self.solid)))
                .collect();
            pass.dispatch(
                paint_kernel,
                &bound,
                Some(&paint(stroke.rect, colour, mask.is_some())),
                (stroke.rect.width, stroke.rect.height),
            );
        }
        pass.finish()?;

        // SAFETY: two distinct live frames; props are timing, colour and side
        // data, not buffers.
        unsafe {
            ffi::av_frame_copy_props(destination.as_mut_ptr(), source.as_ptr());
        }
        Ok(destination)
    }
}

impl Element for Overlaying {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalDetectionOverlay
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Overlaying {
    /// VideoToolbox frames; which of the two layouts is a runtime value.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    /// The picture it was handed, or a copy of it in the same format:
    /// whatever was promised upstream holds after it — an NV12 decoder's
    /// pictures still meet an encoder that takes NV12 alone.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(MetalDetectionOverlayError::UnsupportedBuffer(kind).into());
        };
        let metadata = buf.metadata_arc().cloned();
        let detections = metadata
            .as_deref()
            .and_then(|metadata| metadata.get::<Detections>())
            .filter(|detections| {
                detections
                    .items
                    .iter()
                    .any(|detection| detection.score >= self.options.min_score)
            });
        let Some(detections) = detections else {
            // Nothing to draw: the same picture, not a copy of it.
            out.push(buf);
            return Ok(());
        };
        let drawn = self
            .draw(frame, detections)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        let mut output = MediaBuffer::Video(Arc::new(drawn).into());
        output.set_metadata(metadata);
        out.push(output);
        Ok(())
    }

    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::super::super::{BoxColors, LabelStyle, tests::system_font};
    use super::*;
    use crate::buffer::Metadata;
    use crate::element::RawSink;
    use crate::elements::{Detection, VideoToolboxDownload, VideoToolboxUpload};
    use crate::test_support::{capture, try_videotoolbox_device};

    /// One detection of class 0, from (16, 8) to (48, 24) of a 64 by 32
    /// picture.
    fn found() -> Detections {
        Detections {
            detector: "test".into(),
            labels: Arc::from(vec![Arc::<str>::from("thing")]),
            items: vec![Detection {
                class_id: 0,
                score: 0.9,
                x: 0.25,
                y: 0.25,
                width: 0.5,
                height: 0.5,
            }],
        }
    }

    /// A grey `format` VideoToolbox picture, `width` by `height`, carrying
    /// `detections`.
    fn grey(
        device: &VideoToolboxDevice,
        format: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
        detections: Option<Detections>,
    ) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        frame.set_pts(Some(5));
        let mut upload = VideoToolboxUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("upload");
        let buf = uploaded.lock().unwrap().remove(0);
        match detections {
            Some(detections) => buf.with_metadata(Metadata::new().with(detections)),
            None => buf,
        }
    }

    fn download(buf: MediaBuffer) -> ffmpeg::frame::Video {
        let mut download = VideoToolboxDownload::new("download");
        let received = capture(&mut download);
        download.consume(buf).expect("download");
        let MediaBuffer::Video(frame) = received.lock().unwrap().remove(0) else {
            panic!("expected a picture");
        };
        (**frame).clone()
    }

    const FORMATS: [ffmpeg::format::Pixel; 2] =
        [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA];

    /// The lines are painted in the box's colour on a copy — limited-range
    /// BT.709 luma on NV12, the colour itself on BGRA — with the picture's
    /// timing and its detections, and the picture handed in is untouched.
    #[test]
    fn boxes_are_drawn_on_a_copy_in_both_formats() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let color = Color::new(255, 0, 0);
        let options = DetectionOverlayOptions {
            colors: BoxColors::One(color),
            ..DetectionOverlayOptions::default()
        };
        for format in FORMATS {
            let mut overlay =
                MetalDetectionOverlay::new("overlay", &device, options.clone()).expect("opens");
            let kept = capture(&mut overlay);
            let input = grey(&device, format, 64, 32, Some(found()));
            overlay.consume(input.clone()).expect("drawn");
            let output = kept.lock().unwrap().remove(0);
            assert_eq!(
                output.metadata().and_then(|m| m.get::<Detections>()),
                Some(&found()),
                "{format:?}: the detections go on with the copy"
            );

            let drawn = download(output);
            let untouched = download(input);
            let pixel = |frame: &ffmpeg::frame::Video, x: usize, y: usize| -> Vec<u8> {
                if format == ffmpeg::format::Pixel::NV12 {
                    let chroma = &frame.data(1)[(y / 2) * frame.stride(1) + (x / 2) * 2..];
                    vec![frame.data(0)[y * frame.stride(0) + x], chroma[0], chroma[1]]
                } else {
                    frame.data(0)[y * frame.stride(0) + x * 4..][..4].to_vec()
                }
            };
            let line = if format == ffmpeg::format::Pixel::NV12 {
                let (y, cb, cr) = bt709_limited(color);
                vec![y, cb, cr]
            } else {
                vec![0, 0, 255, 255]
            };
            assert_eq!(pixel(&drawn, 16, 8), line, "{format:?}: top-left corner");
            assert_eq!(
                pixel(&drawn, 47, 23),
                line,
                "{format:?}: bottom-right corner"
            );
            assert_eq!(
                pixel(&drawn, 32, 16),
                pixel(&untouched, 32, 16),
                "{format:?}: inside the box, the picture as it was"
            );
            assert_eq!(
                pixel(&drawn, 2, 2),
                pixel(&untouched, 2, 2),
                "{format:?}: outside the box, the picture as it was"
            );
            assert_eq!(drawn.pts(), Some(5), "{format:?}: the picture's timing");
            assert_ne!(
                pixel(&untouched, 16, 8),
                line,
                "{format:?}: the picture handed in"
            );
        }
    }

    #[test]
    fn a_picture_with_nothing_to_draw_is_handed_on_as_it_came() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut overlay =
            MetalDetectionOverlay::new("overlay", &device, DetectionOverlayOptions::default())
                .expect("opens");
        let kept = capture(&mut overlay);
        let bare = grey(&device, ffmpeg::format::Pixel::NV12, 64, 32, None);
        let MediaBuffer::Video(input) = &bare else {
            unreachable!();
        };
        let id = crate::buffer::picture_id(input);
        overlay.consume(bare).expect("bare");
        let MediaBuffer::Video(output) = kept.lock().unwrap().remove(0) else {
            unreachable!();
        };
        assert_eq!(crate::buffer::picture_id(&output), id);
    }

    /// A picture in system memory carrying something to draw is refused,
    /// saying what it is, and the overlay draws the next one.
    #[test]
    fn a_picture_in_system_memory_is_refused() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut overlay =
            MetalDetectionOverlay::new("overlay", &device, DetectionOverlayOptions::default())
                .expect("opens");
        let kept = capture(&mut overlay);
        let frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 32);
        let error = overlay
            .consume(MediaBuffer::video(frame).with_metadata(Metadata::new().with(found())))
            .expect_err("refused");
        assert!(
            error.to_string().contains("got NV12"),
            "says what it was: {error}"
        );
        overlay
            .consume(grey(
                &device,
                ffmpeg::format::Pixel::BGRA,
                64,
                32,
                Some(found()),
            ))
            .expect("drawn after");
        assert_eq!(kept.lock().unwrap().len(), 1);
    }

    /// With a font, the band above the box is the box's colour and the text
    /// on it is not: written, and on the picture's top where the box is.
    #[test]
    fn a_label_is_drawn_above_its_box() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let Some(font) = system_font() else {
            eprintln!("skipping: no system font");
            return;
        };
        let color = Color::new(255, 196, 0);
        let options = DetectionOverlayOptions {
            colors: BoxColors::One(color),
            labels: Some(LabelStyle {
                size: 12.0,
                ..LabelStyle::new(font)
            }),
            ..DetectionOverlayOptions::default()
        };
        for format in FORMATS {
            let mut overlay =
                MetalDetectionOverlay::new("overlay", &device, options.clone()).expect("opens");
            let kept = capture(&mut overlay);
            let mut detections = found();
            detections.items[0].y = 0.5;
            detections.items[0].height = 0.4;
            overlay
                .consume(grey(&device, format, 128, 64, Some(detections)))
                .expect("drawn");
            let drawn = download(kept.lock().unwrap().remove(0));
            let luma = |x: usize, y: usize| {
                if format == ffmpeg::format::Pixel::NV12 {
                    drawn.data(0)[y * drawn.stride(0) + x]
                } else {
                    // Green alone tells the band (196) from black text and grey.
                    drawn.data(0)[y * drawn.stride(0) + x * 4 + 1]
                }
            };
            let band = if format == ffmpeg::format::Pixel::NV12 {
                bt709_limited(color).0
            } else {
                196
            };
            // The box starts at (32, 32); its band is above it, and its
            // padding row is the band's colour.
            let (top, bottom) = (0..32).fold((None, None), |(top, bottom), y| {
                if luma(32, y) == band {
                    (top.or(Some(y)), Some(y))
                } else {
                    (top, bottom)
                }
            });
            let (Some(top), Some(bottom)) = (top, bottom) else {
                panic!("{format:?}: no band above the box");
            };
            assert_eq!(bottom, 31, "{format:?}: the band ends where the box starts");
            let right = (32..128)
                .take_while(|&x| luma(x, top) == band)
                .last()
                .unwrap();
            let written = (top..=bottom)
                .flat_map(|y| (32..=right).map(move |x| (x, y)))
                .filter(|&(x, y)| luma(x, y) < band.saturating_sub(40))
                .count();
            assert!(
                written > 10,
                "{format:?}: the text is drawn ({written} dark pixels)"
            );
        }
    }
}
