//! [`CudaDetectionOverlay`]: what a detector found, drawn onto CUDA
//! pictures without them leaving the GPU.

use std::collections::HashMap;
use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    color::Color,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::{CudaDriverError, CudaUploadError, Detections},
    error::Result,
    frame_size::ForSize,
    platform::cuda::{
        CudaDevice, CudaFrameFormat,
        driver::{
            BgraSurface, CellMeans, CellPlane, CudaDriver, CudaMask, Nv12Region, Nv12Surface,
            RedactKernels,
        },
        frame::{self, CudaFrameError, CudaSurfaces, create_hw_frames_ctx},
    },
    platform::ffmpeg::AvBufferRef,
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::{
    Canvas, DetectionOverlayOptions, DetectionOverlayOptionsError, Hide, LABEL_CACHE, MaskKey,
    Rect, hides, marks_turned, rasterize,
};
use crate::elements::Analytics;
use crate::elements::source::{TextFontError, load_font};
use crate::orientation::Orientations;

/// Errors specific to `CudaDetectionOverlay`. Converts into the crate-wide
/// `Error` via `?`.
#[derive(Debug, ThisError)]
pub enum CudaDetectionOverlayError {
    /// Something other than a decoded picture arrived.
    #[error("CudaDetectionOverlay only accepts Video buffers, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// The picture is not an NV12 or BGRA surface from this element's own
    /// device.
    #[error(transparent)]
    Frame(#[from] CudaFrameError),
    /// An NV12 picture of odd size, whose last row or column of colour has
    /// no whole 2x2 block to be copied with.
    #[error("an NV12 picture of odd size, {width}x{height}")]
    OddNv12 {
        /// Its width.
        width: u32,
        /// Its height.
        height: u32,
    },
    /// A surface arrived with no device pointer to read.
    #[error("a surface arrived with no device pointer")]
    MissingSurface,
    /// Allocating the frames the copies are drawn on failed.
    #[error(transparent)]
    Frames(#[from] CudaUploadError),
    /// The CUDA driver refused a copy, a fill or a launch.
    #[error(transparent)]
    Driver(#[from] CudaDriverError),
    /// The label font is not a TrueType or OpenType font.
    #[error("the label font is not a TrueType or OpenType font")]
    LabelFont,
    /// Options it cannot be made of.
    #[error(transparent)]
    Options(#[from] DetectionOverlayOptionsError),
}

impl From<TextFontError> for CudaDetectionOverlayError {
    fn from(error: TextFontError) -> Self {
        match error {
            TextFontError::Size(size) => DetectionOverlayOptionsError::LabelSize(size).into(),
            TextFontError::Font(_) => Self::LabelFont,
        }
    }
}

/// Draws the [`Detections`] each CUDA picture carries onto a copy of it, on
/// the GPU — each class as its
/// [`ClassRule`](super::super::ClassRule) in the options says: its box and label drawn,
/// or it hidden by a mosaic, a blur or a fill, or both — and
/// hands the copy on, carrying the same `Detections`: what
/// [`SwDetectionOverlay`](super::super::SwDetectionOverlay) does in system
/// memory, so that a detector's pictures can go on to an encoder, a
/// renderer or a compositor without leaving the GPU.
///
/// It takes NV12 or BGRA CUDA pictures of any size from the same
/// [`CudaDevice`] as the rest of the pipeline, and draws at that size. The
/// copy is made in device memory from a pool of this element's own, and is
/// what keeps the boxes off the picture it was handed, which a Tee in
/// front of it shares with another branch. A picture carrying nothing to
/// draw is handed on as it came.
///
/// Lines and label bands are rectangle fills; a label's text is rasterized
/// on the CPU once per distinct string, uploaded as a mask and blended on
/// the GPU, as a compositor's text layers are.
pub struct CudaDetectionOverlay(FilterStage<Overlaying>);

filter_stage!(CudaDetectionOverlay);

/// What a [`CudaDetectionOverlay`] does with each picture.
///
/// The masks come before the driver: fields drop in order, and the masks
/// are freed in the driver's context, which the driver releases.
struct Overlaying {
    name: Arc<str>,
    pp_log: PpLog,
    options: DetectionOverlayOptions,
    font: Option<ab_glyph::FontArc>,
    /// Masks already rasterized and uploaded — labels by their text, the
    /// pieces of zones and lines by where they are; `None` for one with
    /// nothing to draw.
    masks: HashMap<MaskKey, Option<CudaMask>>,
    /// Whether it has said that a rule names a class the detections
    /// carry no names for.
    warned_names: bool,
    /// How each picture is turned to be shown.
    orientations: Orientations,
    /// The kernels a mosaic or a blur is painted with, and the buffer its
    /// cells' means go through, grown as a larger box needs: made with the
    /// element where it hides by cells.
    cells: Option<(RedactKernels, CellMeans)>,
    driver: CudaDriver,
    /// This element's own reference to the shared context.
    hw_device_ctx: Arc<AvBufferRef>,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *mut ffi::AVHWDeviceContext,
    /// The pools the copies come from, one per format, each made for the
    /// size of the pictures arriving — see `ForSize`.
    nv12_frames: ForSize<AvBufferRef>,
    bgra_frames: ForSize<AvBufferRef>,
    /// Reuses only the CPU-side `AVFrame` wrapper; each surface comes from
    /// a frames pool.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the buffers and masks are device allocations and FFmpeg buffers
// with no thread affinity, `device_ctx` only ever has its address compared,
// and `&mut self` on every method that touches the rest rules out
// concurrent access — the reasoning `CudaVideoEffect` gives for its own.
unsafe impl Send for Overlaying {}

impl CudaDetectionOverlay {
    /// An overlay drawing as `options` say, on pictures from `device` —
    /// the same [`CudaDevice`] every other CUDA element in the pipeline was
    /// built from.
    ///
    /// # Errors
    ///
    /// Options it cannot be made of — a label with no font, a class named
    /// twice — or a font that cannot be read, and the driver's where it cannot be opened.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        options: DetectionOverlayOptions,
    ) -> std::result::Result<Self, CudaDetectionOverlayError> {
        let name: Arc<str> = name.into().into();
        options.check()?;
        let pp_log = element_pp_log(ElementType::CudaDetectionOverlay, &name, None);
        // Read once at any size: each label is rasterized at its own.
        let font = options
            .font
            .as_ref()
            .map(|font| load_font(font.clone(), 16.0))
            .transpose()?;
        let driver = CudaDriver::retain_primary()?;
        // Made now, where the style needs them, so that a driver that cannot
        // JIT them refuses the element rather than its first picture.
        let cells = if options.cuts_cells() {
            Some((driver.redact_kernels()?, driver.cell_means(4 * 64)?))
        } else {
            None
        };
        let hw_device_ctx = device.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition; only the pointer's identity is kept, and the reference
        // held beside it keeps that identity from being reused.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *mut ffi::AVHWDeviceContext };
        pp_info!(
            pp_log: &pp_log,
            "opened: NV12 or BGRA, {} rules, others {:?}, font={}",
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
            cells,
            driver,
            hw_device_ctx,
            device_ctx,
            nv12_frames: ForSize::new(),
            bgra_frames: ForSize::new(),
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
        })))
    }
}

/// A picture being drawn on: one of the two surfaces it takes.
#[derive(Clone, Copy)]
enum Surface {
    Nv12(Nv12Surface),
    Bgra(BgraSurface),
}

/// The mask `key` names, rasterized in `font` and uploaded by `driver`,
/// from `masks` where it is there.
fn mask<'a>(
    masks: &'a mut HashMap<MaskKey, Option<CudaMask>>,
    font: Option<&ab_glyph::FontArc>,
    driver: &CudaDriver,
    pp_log: &PpLog,
    key: &MaskKey,
) -> Option<&'a CudaMask> {
    if !masks.contains_key(key) {
        let made = match rasterize(key, font) {
            Ok(Some(raster)) => driver
                .upload_mask(&raster.coverage, raster.width, raster.height)
                // A mask under two pixels either way has no whole block
                // to draw; it is left out, not an error.
                .ok(),
            Ok(None) => None,
            Err(error) => {
                pp_error!(pp_log: pp_log, "{key:?} not drawn: {error:?}");
                None
            }
        };
        masks.insert(key.clone(), made);
    }
    masks.get(key)?.as_ref()
}

impl Overlaying {
    /// A frame from this element's pool for `format` at `width` x `height`.
    fn frame(
        &mut self,
        format: CudaFrameFormat,
        width: u32,
        height: u32,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, CudaDetectionOverlayError>
    {
        let (device, pp_log) = (&self.hw_device_ctx, &self.pp_log);
        let pool = match format {
            CudaFrameFormat::Nv12 => &mut self.nv12_frames,
            _ => &mut self.bgra_frames,
        };
        let frames_ctx = pool
            .try_get(width, height, |width, height| {
                pp_info!(pp_log: pp_log, "drawing {format:?} {width}x{height}");
                // SAFETY: `create_hw_frames_ctx`'s contract is a live device
                // context, which is what the owned `AvBufferRef` is.
                unsafe { create_hw_frames_ctx(device, format, width, height) }
                    .map_err(CudaUploadError::from)
            })?
            .as_ptr();
        let mut destination = self.pool.get();
        // SAFETY: `dst` is the pooled wrapper's own `AVFrame`; unreferencing it
        // first hands its previous surface back to the frames pool. The frames
        // context is this element's own, held for its life.
        unsafe {
            let dst = destination.as_mut_ptr();
            ffi::av_frame_unref(dst);
            let code = ffi::av_hwframe_get_buffer(frames_ctx, dst, 0);
            if code < 0 {
                return Err(CudaUploadError::HwFrameGet(code).into());
            }
        }
        Ok(destination)
    }

    /// Fills `rect` of `surface` with `color`.
    fn fill(
        &self,
        surface: Surface,
        rect: Rect,
        color: Color,
    ) -> std::result::Result<(), CudaDriverError> {
        let (x, y) = (u64::from(rect.x), u64::from(rect.y));
        match surface {
            Surface::Nv12(nv12) => {
                // `rect` is on whole 2x2 blocks, so its chroma starts `x`
                // bytes into the row half its height down.
                let at = Nv12Surface {
                    luma: nv12.luma + y * nv12.luma_pitch as u64 + x,
                    chroma: nv12.chroma + (y / 2) * nv12.chroma_pitch as u64 + x,
                    ..nv12
                };
                self.driver.fill_nv12(at, rect.width, rect.height, color)
            }
            Surface::Bgra(bgra) => {
                let at = BgraSurface {
                    pixels: bgra.pixels + y * bgra.pitch as u64 + x * 4,
                    ..bgra
                };
                self.driver
                    .fill_bgra(at, rect.width, rect.height, color, 255)
            }
        }
    }

    /// Paints `rect` of `surface` cell by cell, plane by plane: NV12's
    /// colour, a sample per 2x2 block, in the same cells at half the size,
    /// `rect` being on whole blocks.
    fn hide_cells(
        &mut self,
        surface: Surface,
        rect: Rect,
        cells: (u32, u32),
        smooth: bool,
    ) -> std::result::Result<(), CudaDriverError> {
        let Some((kernels, means)) = &mut self.cells else {
            // Made with the element wherever its style cuts cells.
            return Err(CudaDriverError::KernelRejected(
                "no hiding kernels for a style that cuts cells".into(),
            ));
        };
        let needed = cells.0 as usize * cells.1 as usize * 4;
        if means.floats() < needed {
            *means = self.driver.cell_means(needed)?;
        }
        let (x, y) = (u64::from(rect.x), u64::from(rect.y));
        let planes = match surface {
            Surface::Nv12(nv12) => vec![
                CellPlane {
                    plane: nv12.luma + y * nv12.luma_pitch as u64 + x,
                    pitch: nv12.luma_pitch,
                    width: rect.width,
                    height: rect.height,
                    channels: 1,
                    cells,
                },
                CellPlane {
                    plane: nv12.chroma + (y / 2) * nv12.chroma_pitch as u64 + x,
                    pitch: nv12.chroma_pitch,
                    width: rect.width / 2,
                    height: rect.height / 2,
                    channels: 2,
                    cells,
                },
            ],
            Surface::Bgra(bgra) => vec![CellPlane {
                plane: bgra.pixels + y * bgra.pitch as u64 + x * 4,
                pitch: bgra.pitch,
                width: rect.width,
                height: rect.height,
                channels: 4,
                cells,
            }],
        };
        for plane in planes {
            self.driver.hide_cells(kernels, means, plane, smooth)?;
        }
        Ok(())
    }

    /// A copy of `source` with `detections` and `analytics` drawn on it.
    fn draw(
        &mut self,
        source: &ffmpeg::frame::Video,
        detections: Option<&Detections>,
        analytics: Option<&Analytics>,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, CudaDetectionOverlayError>
    {
        let layout = frame::validate(
            source,
            ElementType::CudaDetectionOverlay,
            self.device_ctx,
            CudaSurfaces::NV12_OR_BGRA,
        )?
        .layout;
        let (width, height) = (source.width(), source.height());
        let nv12 = layout == ffmpeg::format::Pixel::NV12;
        if nv12 && (width % 2 == 1 || height % 2 == 1) {
            return Err(CudaDetectionOverlayError::OddNv12 { width, height });
        }
        let format = if nv12 {
            CudaFrameFormat::Nv12
        } else {
            CudaFrameFormat::Bgra
        };
        let mut destination = self.frame(format, width, height)?;

        let surface = if nv12 {
            let from =
                Nv12Surface::from_frame(source).ok_or(CudaDetectionOverlayError::MissingSurface)?;
            let to = Nv12Surface::from_frame(&destination)
                .ok_or(CudaDetectionOverlayError::MissingSurface)?;
            self.driver.blit_nv12(
                from,
                to,
                Nv12Region {
                    source_x: 0,
                    source_y: 0,
                    destination_x: 0,
                    destination_y: 0,
                    width,
                    height,
                },
            )?;
            Surface::Nv12(to)
        } else {
            let from =
                BgraSurface::from_frame(source).ok_or(CudaDetectionOverlayError::MissingSurface)?;
            let to = BgraSurface::from_frame(&destination)
                .ok_or(CudaDetectionOverlayError::MissingSurface)?;
            self.driver.blit_bgra(from, to, 0, 0, width, height)?;
            Surface::Bgra(to)
        };

        let canvas = Canvas {
            width,
            height,
            block: if nv12 { 2 } else { 1 },
        };
        // The cache is emptied between pictures, never while one's marks are
        // placed: every mask made for this picture is read after all of
        // them are, and one emptied away in between was drawn as a solid
        // block, or on CUDA panicked the element's thread.
        if self.masks.len() >= LABEL_CACHE {
            self.masks.clear();
        }
        // Placed by each uploaded mask's size, which is the rasterized one
        // cut to whole blocks, so nothing is read past it.
        let (masks, font, driver, pp_log) = (
            &mut self.masks,
            self.font.as_ref(),
            &self.driver,
            &self.pp_log,
        );
        let orientation = self.orientations.of(source, &self.pp_log);
        let marks = marks_turned(
            canvas,
            orientation,
            &self.options,
            detections,
            analytics,
            &mut |key| mask(masks, font, driver, pp_log, key).map(|mask| (mask.width, mask.height)),
        );
        // What is hidden first, so that a box and its label are drawn over
        // it where they are asked for.
        if let Some(detections) = detections {
            for hide in hides(canvas, &self.options, detections) {
                match hide {
                    Hide::Fill { rect, color } => self.fill(surface, rect, color)?,
                    Hide::Cells {
                        rect,
                        cells,
                        smooth,
                    } => self.hide_cells(surface, rect, cells, smooth)?,
                }
            }
        }
        for mark in marks {
            let Some(key) = &mark.mask else {
                self.fill(surface, mark.rect, mark.color)?;
                continue;
            };
            // Placed only where its mask was made, and the cache is emptied
            // only between pictures; were it missing all the same, the mark
            // is left out rather than the thread panicking.
            let Some(mask) = self.masks.get(key).and_then(Option::as_ref) else {
                pp_error!(self, "{key:?} not drawn: its mask is gone from the cache");
                continue;
            };
            let area = mark.rect;
            match surface {
                Surface::Nv12(nv12) => self.driver.blend_mask_nv12(
                    nv12,
                    area.x,
                    area.y,
                    mask,
                    0,
                    0,
                    area.width,
                    area.height,
                    mark.color,
                    255,
                )?,
                Surface::Bgra(bgra) => self.driver.blend_mask_bgra(
                    bgra,
                    area.x,
                    area.y,
                    mask.full_at(0, 0),
                    mask.width,
                    area.width,
                    area.height,
                    mark.color,
                    255,
                )?,
            }
        }
        // The copies and fills run on this driver's context while whatever
        // reads the result next goes through FFmpeg's stream: one
        // synchronize per picture makes it visible to both, as
        // `CudaVideoEffect` does.
        self.driver.synchronize()?;

        // SAFETY: both frames are live and distinct — `destination` came from
        // the pool, `source` is the caller's.
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
        ElementType::CudaDetectionOverlay
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Overlaying {
    /// Device-resident frames; which of the two layouts is a runtime value.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
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
            return Err(CudaDetectionOverlayError::UnsupportedBuffer(kind).into());
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
        let mut output = MediaBuffer::Video(Arc::new(drawn).into());
        output.set_metadata(metadata);
        out.push(output);
        Ok(())
    }

    fn reset(&mut self) {}
}

impl Drop for Overlaying {
    fn drop(&mut self) {
        pp_info!(self, "dropped: releasing hw contexts");
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        BoxColors, BoxStyle, LabelStyle, RedactStyle, Treatment, tests::system_font,
    };
    use super::*;
    use crate::buffer::Metadata;
    use crate::element::RawSink;
    use crate::elements::{CudaDownload, CudaUpload, Detection};
    use crate::platform::cuda::driver::bt709_limited as driver_yuv;
    use crate::test_support::{capture, try_cuda_device};

    /// The luma the driver fills `color` with.
    fn bt709_limited(color: Color) -> (u8, u8, u8) {
        driver_yuv(
            f32::from(color.red),
            f32::from(color.green),
            f32::from(color.blue),
        )
    }

    /// One detection of class 0, from (16, 8) to (48, 24) of a 64 by 32
    /// picture.
    fn found() -> Detections {
        Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("thing")]),
            vec![Detection::new(0, 0.9, 0.25, 0.25, 0.5, 0.5)],
        )
    }

    /// A grey `format` picture on the GPU, `width` by `height`, carrying
    /// `detections`.
    fn grey(
        device: &CudaDevice,
        format: CudaFrameFormat,
        width: u32,
        height: u32,
        detections: Option<Detections>,
    ) -> MediaBuffer {
        let pixel = match format {
            CudaFrameFormat::Nv12 => ffmpeg::format::Pixel::NV12,
            _ => ffmpeg::format::Pixel::BGRA,
        };
        let mut frame = ffmpeg::frame::Video::new(pixel, width, height);
        for plane in 0..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        frame.set_pts(Some(5));
        let mut upload = CudaUpload::new("upload", device, format);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("upload");
        let buf = uploaded.lock().unwrap().remove(0);
        match detections {
            Some(detections) => buf.with_metadata(Metadata::new().with(detections)),
            None => buf,
        }
    }

    fn download(
        device: &CudaDevice,
        format: CudaFrameFormat,
        buf: MediaBuffer,
    ) -> ffmpeg::frame::Video {
        let mut download = CudaDownload::new("download", device, format);
        let received = capture(&mut download);
        download.consume(buf).expect("download");
        let MediaBuffer::Video(frame) = received.lock().unwrap().remove(0) else {
            panic!("expected a picture");
        };
        (**frame).clone()
    }

    #[test]
    fn boxes_are_drawn_on_a_copy_in_both_formats() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let color = Color::new(255, 0, 0);
        let options = DetectionOverlayOptions {
            others: Treatment::boxes(BoxStyle {
                color: BoxColors::One(color),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        for format in [CudaFrameFormat::Nv12, CudaFrameFormat::Bgra] {
            let Ok(mut overlay) = CudaDetectionOverlay::new("overlay", &device, options.clone())
            else {
                eprintln!("skipping: no usable CUDA driver here");
                return;
            };
            let kept = capture(&mut overlay);
            let input = grey(&device, format, 64, 32, Some(found()));
            overlay.consume(input.clone()).expect("drawn");
            let output = kept.lock().unwrap().remove(0);
            assert_eq!(
                output.metadata().and_then(|m| m.get::<Detections>()),
                Some(&found()),
                "{format:?}: the detections go on with the copy"
            );

            let drawn = download(&device, format, output);
            let untouched = download(&device, format, input);
            let pixel = |frame: &ffmpeg::frame::Video, x: usize, y: usize| -> Vec<u8> {
                match format {
                    CudaFrameFormat::Nv12 => vec![frame.data(0)[y * frame.stride(0) + x]],
                    _ => frame.data(0)[y * frame.stride(0) + x * 4..][..3].to_vec(),
                }
            };
            let line = match format {
                CudaFrameFormat::Nv12 => vec![bt709_limited(color).0],
                _ => vec![0, 0, 255],
            };
            assert_eq!(pixel(&drawn, 16, 8), line, "{format:?}: top-left corner");
            assert_eq!(
                pixel(&drawn, 47, 23),
                line,
                "{format:?}: bottom-right corner"
            );
            assert_ne!(pixel(&drawn, 32, 16), line, "{format:?}: inside the box");
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
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let Ok(mut overlay) =
            CudaDetectionOverlay::new("overlay", &device, DetectionOverlayOptions::default())
        else {
            return;
        };
        let kept = capture(&mut overlay);
        let bare = grey(&device, CudaFrameFormat::Nv12, 64, 32, None);
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

    /// With a font, the band above the box is the box's colour and the text
    /// on it is not: written, and on the picture's top where the box is.
    #[test]
    fn a_label_is_drawn_above_its_box() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let Some(font) = system_font() else {
            eprintln!("skipping: no system font");
            return;
        };
        let color = Color::new(255, 196, 0);
        let options = DetectionOverlayOptions {
            font: Some(font),
            others: Treatment::boxes(BoxStyle {
                color: BoxColors::One(color),
                label: Some(LabelStyle::new(12.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        for format in [CudaFrameFormat::Nv12, CudaFrameFormat::Bgra] {
            let Ok(mut overlay) = CudaDetectionOverlay::new("overlay", &device, options.clone())
            else {
                return;
            };
            let kept = capture(&mut overlay);
            let mut detections = found();
            detections.items[0].y = 0.5;
            detections.items[0].height = 0.4;
            overlay
                .consume(grey(&device, format, 128, 64, Some(detections)))
                .expect("drawn");
            let output = kept.lock().unwrap().remove(0);
            let drawn = download(&device, format, output);
            let luma = |x: usize, y: usize| match format {
                CudaFrameFormat::Nv12 => drawn.data(0)[y * drawn.stride(0) + x],
                // Green alone tells the band (196) from black text and grey.
                _ => drawn.data(0)[y * drawn.stride(0) + x * 4 + 1],
            };
            let band = match format {
                CudaFrameFormat::Nv12 => bt709_limited(color).0,
                _ => 196,
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
            // The band's top row is padding, its colour all the way across.
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

    /// A `pixel` picture of noise, 96 by 48, the same every time.
    fn noise(pixel: ffmpeg::format::Pixel) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(pixel, 96, 48);
        for plane in 0..frame.planes() {
            for (index, byte) in frame.data_mut(plane).iter_mut().enumerate() {
                let mut v = (index as u32).wrapping_mul(2_654_435_761) ^ (plane as u32 * 977);
                v ^= v >> 15;
                *byte = (v.wrapping_mul(2_246_822_519) >> 24) as u8;
            }
        }
        frame.set_pts(Some(7));
        frame
    }

    /// The bytes a picture shows, plane by plane, row by row, without the
    /// padding at the end of each row.
    fn shown(frame: &ffmpeg::frame::Video) -> Vec<Vec<u8>> {
        let (width, height) = (frame.width() as usize, frame.height() as usize);
        let rows: Vec<(usize, usize)> = match frame.format() {
            ffmpeg::format::Pixel::NV12 => vec![(height, width), (height / 2, width)],
            _ => vec![(height, width * 4)],
        };
        rows.into_iter()
            .enumerate()
            .map(|(plane, (rows, bytes))| {
                let stride = frame.stride(plane);
                (0..rows)
                    .flat_map(|row| frame.data(plane)[row * stride..][..bytes].to_vec())
                    .collect()
            })
            .collect()
    }

    /// The point of the kernels: a mosaic and a blur on the GPU write the
    /// same bytes as the CPU's overlay on the same picture, NV12 and BGRA
    /// alike — one face, grown by the margin, cut into cells of uneven
    /// width, its colour at half the size.
    #[test]
    fn a_box_is_hidden_on_the_gpu_as_on_the_cpu() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let faces = || {
            Detections::new(
                "test",
                Arc::from(vec![Arc::<str>::from("face")]),
                vec![
                    Detection::new(0, 0.9, 0.1, 0.2, 0.37, 0.55),
                    Detection::new(0, 0.8, 0.62, 0.05, 0.2, 0.3),
                ],
            )
        };
        let styles = [
            RedactStyle::mosaic(),
            RedactStyle::blur(),
            RedactStyle::Mosaic {
                cells: std::num::NonZeroU32::new(5).unwrap(),
                min_cell: 2,
            },
            RedactStyle::Blur {
                cells: std::num::NonZeroU32::new(5).unwrap(),
                min_cell: 2,
            },
        ];
        for (format, pixel) in [
            (CudaFrameFormat::Nv12, ffmpeg::format::Pixel::NV12),
            (CudaFrameFormat::Bgra, ffmpeg::format::Pixel::BGRA),
        ] {
            for style in styles {
                let options = DetectionOverlayOptions {
                    others: Treatment::hidden(style),
                    ..DetectionOverlayOptions::default()
                };
                let picture = noise(pixel);

                let mut cpu = crate::elements::SwDetectionOverlay::new("cpu", options.clone())
                    .expect("opens");
                let on_cpu = capture(&mut cpu);
                cpu.consume(
                    MediaBuffer::video(picture.clone())
                        .with_metadata(Metadata::new().with(faces())),
                )
                .expect("hidden on the CPU");
                let MediaBuffer::Video(expected) = on_cpu.lock().unwrap().remove(0) else {
                    panic!("a picture");
                };

                let mut upload = CudaUpload::new("upload", &device, format);
                let uploaded = capture(&mut upload);
                upload
                    .consume(MediaBuffer::video(picture.clone()))
                    .expect("uploads");
                let input = uploaded
                    .lock()
                    .unwrap()
                    .remove(0)
                    .with_metadata(Metadata::new().with(faces()));
                let mut gpu = CudaDetectionOverlay::new("gpu", &device, options).expect("opens");
                let on_gpu = capture(&mut gpu);
                gpu.consume(input).expect("hidden on the GPU");
                let actual = download(&device, format, on_gpu.lock().unwrap().remove(0));

                let (expected, actual) = (shown(&expected), shown(&actual));
                assert_ne!(
                    expected,
                    shown(&picture),
                    "{format:?} {style:?}: something hidden"
                );
                for (plane, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                    let differ = expected
                        .iter()
                        .zip(actual)
                        .filter(|(expected, actual)| expected != actual)
                        .count();
                    assert_eq!(
                        differ, 0,
                        "{format:?} {style:?}, plane {plane}: {differ} bytes differ"
                    );
                }
            }
        }
    }

    /// A picture stored turned is drawn on as it is shown, on the GPU as on
    /// the CPU: boxes and labels drawn on it and turned back are the bytes
    /// drawn on the picture stored the right way up, for each of the eight
    /// ways it can be turned.
    #[test]
    fn a_turned_picture_is_drawn_on_as_it_is_shown() {
        use super::super::super::tests::{every_orientation, system_font, turn};
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let Some(font) = system_font() else {
            eprintln!("skipping: no system font to draw labels in");
            return;
        };
        let options = DetectionOverlayOptions {
            font: Some(font),
            others: Treatment::boxes(BoxStyle {
                label: Some(LabelStyle::new(14.0)),
                ..BoxStyle::default()
            }),
            ..DetectionOverlayOptions::default()
        };
        let found = Detections::new(
            "test",
            Arc::from(vec![Arc::<str>::from("thing")]),
            vec![
                Detection::new(0, 0.9, 0.25, 0.5, 0.5, 0.375),
                Detection::new(0, 0.6, 0.625, 0.0, 0.25, 0.25),
            ],
        );
        for (format, pixel) in [
            (CudaFrameFormat::Nv12, ffmpeg::format::Pixel::NV12),
            (CudaFrameFormat::Bgra, ffmpeg::format::Pixel::BGRA),
        ] {
            let draw = |picture: ffmpeg::frame::Video, found: Detections| {
                let mut upload = CudaUpload::new("upload", &device, format);
                let uploaded = capture(&mut upload);
                upload
                    .consume(MediaBuffer::video(picture))
                    .expect("uploads");
                let input = uploaded
                    .lock()
                    .unwrap()
                    .remove(0)
                    .with_metadata(Metadata::new().with(found));
                let mut gpu =
                    CudaDetectionOverlay::new("gpu", &device, options.clone()).expect("opens");
                let on_gpu = capture(&mut gpu);
                gpu.consume(input).expect("drawn");
                download(&device, format, on_gpu.lock().unwrap().remove(0))
            };
            let upright = noise(pixel);
            let expected = shown(&draw(upright.clone(), found.clone()));
            for orientation in every_orientation() {
                let mut stored = found.clone();
                for item in &mut stored.items {
                    [item.x, item.y, item.width, item.height] =
                        orientation.from_display([item.x, item.y, item.width, item.height]);
                }
                let drawn = draw(turn(&upright, orientation, false), stored);
                assert_eq!(
                    shown(&turn(&drawn, orientation, true)),
                    expected,
                    "{format:?} {orientation:?}"
                );
            }
        }
    }
}
