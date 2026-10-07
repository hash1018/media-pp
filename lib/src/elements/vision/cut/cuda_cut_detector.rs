//! [`CudaCutDetector`]: where one shot ends and the next begins, in CUDA
//! pictures, without them leaving the GPU.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, ffi, format::Pixel};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::vision::batch::PerStream,
    elements::{CudaDevice, CudaDriverError, CudaFrameError, CudaSurfaces},
    error::Result,
    platform::{
        cuda::driver::{CellMeans, CellPlane, CudaDriver, Nv12Surface, RedactKernels},
        cuda::frame,
        ffmpeg::AvBufferRef,
    },
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::{CutDetectorOptions, CutDetectorOptionsError, Judge, Thumbnail};

/// The floats a thumbnail's cells take on the GPU at most: the luma's at
/// two bytes a sample for P010, and the chroma's at four.
const MEANS: usize = 64 * 36 * 2 + 32 * 18 * 4;

/// Errors from a [`CudaCutDetector`].
#[derive(Debug, ThisError)]
pub enum CudaCutDetectorError {
    /// The options were refused.
    #[error(transparent)]
    Options(#[from] CutDetectorOptionsError),
    /// It was handed something other than a picture.
    #[error("CudaCutDetector takes CUDA pictures, got {0}")]
    UnsupportedBuffer(&'static str),
    /// A picture it cannot read: not CUDA, not this device's, not NV12 or
    /// P010.
    #[error(transparent)]
    Frame(#[from] CudaFrameError),
    /// A CUDA frame with no planes behind it.
    #[error("CudaCutDetector was handed a CUDA frame with no surface")]
    MissingSurface,
    /// The driver failed.
    #[error(transparent)]
    Driver(#[from] CudaDriverError),
}

/// Finds where one shot of an edited video ends and the next begins, in
/// NV12 or P010 CUDA pictures from the same [`CudaDevice`] as the rest of
/// the pipeline, and hands each picture on — the first of each shot after
/// the stream's first carrying a [`SceneCut`](crate::elements::SceneCut) —
/// as [`SwCutDetector`](crate::elements::SwCutDetector) does in system
/// memory, the two finding the same cuts in the same pictures.
///
/// The cells of each picture's thumbnail are averaged on the GPU, by the
/// kernel a [`CudaDetectionOverlay`](crate::elements::CudaDetectionOverlay)
/// cuts a mosaic with; only the thumbnail, under 7,000 floats, comes down,
/// once each picture's are averaged. It holds
/// [`CutDetectorOptions::lookahead`] pictures of each stream, so a
/// decoder in front of it needs that many more surfaces: see
/// [`CudaDecoder`](crate::elements::CudaDecoder)'s budget.
pub struct CudaCutDetector(FilterStage<Detecting>);

filter_stage!(CudaCutDetector);

/// What a [`CudaCutDetector`] does with each picture.
///
/// The kernels and buffer come before the driver: fields drop in order,
/// and they are freed in the driver's context, which the driver releases.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    options: CutDetectorOptions,
    judges: PerStream<Judge>,
    kernels: RedactKernels,
    means: CellMeans,
    /// Where the means come down to.
    read: Vec<f32>,
    driver: CudaDriver,
    /// This element's own reference to the shared context.
    _hw_device_ctx: Arc<AvBufferRef>,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *const ffi::AVHWDeviceContext,
}

// SAFETY: the kernels and buffer are device objects with no thread affinity,
// `device_ctx` is only ever compared, and `&mut self` on every method that
// uses the rest rules out concurrent access — as `CudaDetectionOverlay`'s.
unsafe impl Send for Detecting {}

impl CudaCutDetector {
    /// A cut detector judging as `options` say, on pictures from `device` —
    /// the same [`CudaDevice`] every other CUDA element in the pipeline was
    /// built from.
    ///
    /// # Errors
    ///
    /// Options it cannot judge with, and the driver's where its kernels
    /// cannot be loaded.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        options: CutDetectorOptions,
    ) -> std::result::Result<Self, CudaCutDetectorError> {
        options.check()?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaCutDetector, &name, None);
        let driver = CudaDriver::retain_primary()?;
        let kernels = driver.redact_kernels()?;
        let means = driver.cell_means(MEANS)?;
        pp_info!(pp_log: &pp_log, "opened: {options:?}");
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            options,
            judges: PerStream::default(),
            kernels,
            means,
            read: vec![0.0; MEANS],
            driver,
            _hw_device_ctx: device.retain(),
            device_ctx: device.device_ctx(),
        })))
    }
}

impl Detecting {
    /// `frame`'s thumbnail, its cells averaged on the GPU.
    fn thumbnail(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Thumbnail, CudaCutDetectorError> {
        let layout = frame::validate(
            frame,
            ElementType::CudaCutDetector,
            self.device_ctx,
            CudaSurfaces::NV12_OR_P010,
        )?
        .layout;
        let surface = Nv12Surface::from_frame(frame).ok_or(CudaCutDetectorError::MissingSurface)?;
        let (width, height) = (frame.width(), frame.height());
        let (cells, chroma_cells) = Thumbnail::cells_of(width, height);
        // Bytes a luma sample and a chroma pair take; P010's are read by
        // their high bytes.
        let (luma_bytes, chroma_bytes) = if layout == Pixel::P010LE {
            (2, 4)
        } else {
            (1, 2)
        };
        let luma_floats = (cells.0 * cells.1 * luma_bytes) as usize;
        let chroma_floats = (chroma_cells.0 * chroma_cells.1 * chroma_bytes) as usize;
        self.driver.measure_cells(
            &self.kernels,
            &self.means,
            0,
            CellPlane {
                plane: surface.luma,
                pitch: surface.luma_pitch,
                width,
                height,
                channels: luma_bytes,
                cells,
            },
        )?;
        self.driver.measure_cells(
            &self.kernels,
            &self.means,
            luma_floats,
            CellPlane {
                plane: surface.chroma,
                pitch: surface.chroma_pitch,
                width: width.div_ceil(2),
                height: height.div_ceil(2),
                channels: chroma_bytes,
                cells: chroma_cells,
            },
        )?;
        let read = &mut self.read[..luma_floats + chroma_floats];
        self.driver.read_means(&self.means, read)?;
        let (luma, chroma) = read.split_at(luma_floats);
        let luma = luma
            .chunks_exact(luma_bytes as usize)
            .map(|cell| cell[luma_bytes as usize - 1])
            .collect();
        let chroma = chroma
            .chunks_exact(chroma_bytes as usize)
            .flat_map(|cell| {
                // Cb then Cr, each its high byte where it has two.
                let step = chroma_bytes as usize / 2;
                [cell[step - 1], cell[2 * step - 1]]
            })
            .collect();
        Ok(Thumbnail {
            cells,
            luma,
            chroma_cells,
            chroma,
        })
    }
}

impl Element for Detecting {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CudaCutDetector
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Detecting {
    /// NV12 or P010 CUDA pictures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                .with_layouts(PixelLayoutSet::YUV420),
        )
    }

    /// The pictures it was handed.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(CudaCutDetectorError::UnsupportedBuffer(kind).into());
        };
        let thumbnail = self
            .thumbnail(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        let (name, options) = (&self.name, self.options);
        let (judge, moved) = self
            .judges
            .get(&buf, |_| Judge::new(Arc::clone(name), options));
        if moved {
            // That stream was sought: what it held is from before.
            judge.clear();
        }
        judge.push(buf, thumbnail, out);
        Ok(())
    }

    /// Every picture still held, judged by those after it there are.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        for judge in self.judges.values_mut() {
            judge.finish(out);
        }
        Ok(())
    }

    /// A seek or a flush: what was held is let go of.
    fn reset(&mut self) {
        self.judges.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::sw_cut_detector::thumbnail_of;
    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, CudaDownload, CudaFrameFormat, CudaUpload};
    use crate::test_support::{capture, try_cuda_device};

    /// A picture in which no sample is its neighbour's.
    fn noise(width: u32, height: u32, seed: u32) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(Pixel::NV12, width, height);
        for plane in 0..2 {
            for (index, byte) in frame.data_mut(plane).iter_mut().enumerate() {
                let mut v =
                    (index as u32 ^ seed).wrapping_mul(2_654_435_761) ^ (plane as u32 * 977);
                v ^= v >> 15;
                *byte = (v.wrapping_mul(2_246_822_519) >> 24) as u8;
            }
        }
        frame
    }

    /// The GPU's thumbnail of a picture is the CPU's, float for float, at a
    /// size the cells divide evenly and one they do not.
    #[test]
    fn the_gpu_makes_the_cpus_thumbnail() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        for (width, height) in [(128, 72), (1918, 1078)] {
            let picture = noise(width, height, width);
            let mut upload = CudaUpload::new("upload", &device, CudaFrameFormat::Nv12);
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(picture.clone()))
                .expect("uploads");
            let MediaBuffer::Video(frame) = uploaded.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            // The CPU reads what came down, which the upload may have padded.
            let mut download = CudaDownload::new("download", &device, CudaFrameFormat::Nv12);
            let downloaded = capture(&mut download);
            download
                .consume(MediaBuffer::Video(frame.clone()))
                .expect("downloads");
            let MediaBuffer::Video(down) = downloaded.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            let detector =
                CudaCutDetector::new("cuts", &device, CutDetectorOptions::default()).unwrap();
            let mut detecting = detector.0;
            let gpu = detecting.inner_mut().thumbnail(&frame).expect("thumbnail");
            assert_eq!(Some(gpu), thumbnail_of(&down), "{width}x{height}");
        }
    }

    /// A picture from another device context is refused, naming the
    /// element.
    #[test]
    fn a_picture_from_elsewhere_is_refused() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let mut detector =
            CudaCutDetector::new("cuts", &device, CutDetectorOptions::default()).unwrap();
        let kept = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        detector.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        let refused = detector.consume(MediaBuffer::video(noise(64, 36, 1)));
        assert!(refused.is_err(), "a CPU picture is no CUDA one");
        assert!(kept.lock().unwrap().is_empty());
    }
}
