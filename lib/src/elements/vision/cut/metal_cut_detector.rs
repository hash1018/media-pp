//! [`MetalCutDetector`]: where one shot ends and the next begins, in
//! VideoToolbox pictures, their thumbnails made on the GPU.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, format::Pixel};
use objc2_metal::{MTLBuffer, MTLPixelFormat, MTLTextureUsage};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::vision::batch::PerStream,
    error::Result,
    platform::macos::{
        metal::{Buffer, Kernel, MetalError, MetalGpu},
        pixel_buffer::PixelBuffer,
        videotoolbox::{NotVideoToolbox, sw_format_of},
    },
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::{CELLS, CutDetectorOptions, CutDetectorOptionsError, Judge, Thumbnail};

const REDACT: &str = include_str!("../../../shaders/metal/redact.metal");

/// The cells a thumbnail has at most, luma and chroma, each a `float4` of
/// means.
const MEANS: usize = (CELLS.0 * CELLS.1 + (CELLS.0 / 2) * (CELLS.1 / 2)) as usize;

/// Errors from a [`MetalCutDetector`].
#[derive(Debug, ThisError)]
pub enum MetalCutDetectorError {
    /// The options were refused.
    #[error(transparent)]
    Options(#[from] CutDetectorOptionsError),
    /// It was handed something other than a picture.
    #[error("MetalCutDetector takes VideoToolbox pictures, got {0}")]
    UnsupportedBuffer(&'static str),
    /// A picture that is not a VideoToolbox one: what it is instead.
    #[error("MetalCutDetector takes VideoToolbox pictures, got {0:?}")]
    NotVideoToolbox(Pixel),
    /// A VideoToolbox picture holding a layout other than NV12 or P010, or
    /// with no frames context to say which.
    #[error("MetalCutDetector reads NV12 or P010 VideoToolbox pictures, got {0:?}")]
    UnsupportedLayout(Option<Pixel>),
    /// A VideoToolbox picture with no pixel buffer behind it.
    #[error("MetalCutDetector was handed a VideoToolbox picture with no pixel buffer")]
    NoPixelBuffer,
    /// Metal failed.
    #[error(transparent)]
    Metal(#[from] MetalError),
}

/// Finds where one shot of an edited video ends and the next begins, in
/// NV12 or P010 VideoToolbox pictures, and hands each picture on — the
/// first of each shot after the stream's first carrying a
/// [`SceneCut`](crate::elements::SceneCut) — as
/// [`SwCutDetector`](crate::elements::SwCutDetector) does in system
/// memory, the two finding the same cuts in the same pictures.
///
/// The cells of each picture's thumbnail are averaged on the GPU, by the
/// kernel a [`MetalDetectionOverlay`](crate::elements::MetalDetectionOverlay)
/// cuts a mosaic with, into memory the CPU reads, where the picture is: a
/// P010 picture's samples are read byte by byte, their high bytes kept, as
/// the CPU reads them. It takes pictures from any device, since a pixel
/// buffer belongs to none, and holds [`CutDetectorOptions::lookahead`]
/// pictures of each stream.
pub struct MetalCutDetector(FilterStage<Detecting>);

filter_stage!(MetalCutDetector);

/// What a [`MetalCutDetector`] does with each picture.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    options: CutDetectorOptions,
    judges: PerStream<Judge>,
    gpu: MetalGpu,
    /// `cell_means`, compiled as the overlay's is, to make the CPU's floats.
    means_kernel: Kernel,
    /// Each cell's means, luma's then chroma's, in memory the CPU reads.
    means: Buffer,
}

// SAFETY: Metal's objects are thread-safe, and the buffer is written only by
// a pass this waits for before it is read, by the one thread transforming at
// a time — as the Metal overlay's.
unsafe impl Send for Detecting {}

impl MetalCutDetector {
    /// A cut detector judging as `options` say.
    ///
    /// # Errors
    ///
    /// Options it cannot judge with, and Metal's where there is no GPU or
    /// its kernel will not compile.
    pub fn new(
        name: impl Into<String>,
        options: CutDetectorOptions,
    ) -> std::result::Result<Self, MetalCutDetectorError> {
        options.check()?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalCutDetector, &name, None);
        let gpu = MetalGpu::new()?;
        let [means_kernel] = gpu.precise_kernel_array(REDACT, ["cell_means"])?;
        let means = gpu.shared_buffer(MEANS * 4 * size_of::<f32>())?;
        pp_info!(pp_log: &pp_log, "opened: {options:?}");
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            options,
            judges: PerStream::default(),
            gpu,
            means_kernel,
            means,
        })))
    }
}

/// The parameters of one plane's cells over the whole of it, as the
/// redaction shader's `Cells`.
fn whole_plane(size: (u32, u32), cells: (u32, u32)) -> Vec<u8> {
    [0, 0, size.0, size.1, cells.0, cells.1, 0, 0, 0, 0]
        .iter()
        .flat_map(|word| word.to_ne_bytes())
        .collect()
}

impl Detecting {
    /// `frame`'s thumbnail, its cells averaged on the GPU.
    fn thumbnail(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Thumbnail, MetalCutDetectorError> {
        let ten_bits = match sw_format_of(frame) {
            Ok(Pixel::NV12) => false,
            Ok(Pixel::P010LE) => true,
            Ok(other) => return Err(MetalCutDetectorError::UnsupportedLayout(Some(other))),
            Err(NotVideoToolbox::Format(other)) => {
                return Err(MetalCutDetectorError::NotVideoToolbox(other));
            }
            Err(NotVideoToolbox::NoFramesContext) => {
                return Err(MetalCutDetectorError::UnsupportedLayout(None));
            }
        };
        let buffer = PixelBuffer::of_frame(frame).ok_or(MetalCutDetectorError::NoPixelBuffer)?;
        let (width, height) = (frame.width(), frame.height());
        let chroma_size = (width.div_ceil(2), height.div_ceil(2));
        let (cells, chroma_cells) = Thumbnail::cells_of(width, height);
        // Each sample's bytes as channels: P010's low byte, then its high
        // one, which is what is kept.
        let (luma_format, chroma_format) = if ten_bits {
            (MTLPixelFormat::RG8Uint, MTLPixelFormat::RGBA8Uint)
        } else {
            (MTLPixelFormat::R8Uint, MTLPixelFormat::RG8Uint)
        };
        let read = MTLTextureUsage::ShaderRead;
        let luma = self.gpu.plane(&buffer, 0, luma_format, read)?;
        let chroma = self.gpu.plane(&buffer, 1, chroma_format, read)?;
        let luma_cells = (cells.0 * cells.1) as usize;
        let mut pass = self.gpu.pass()?;
        for (plane, size, count, at) in [
            (&luma, (width, height), cells, 0),
            (&chroma, chroma_size, chroma_cells, luma_cells),
        ] {
            let n = (count.0 * count.1) as usize;
            pass.dispatch_groups(
                &self.means_kernel,
                &[plane],
                &[(&self.means, at * 4 * size_of::<f32>())],
                Some(&whole_plane(size, count)),
                (n.div_ceil(64), 1, 1),
                (64, 1, 1),
            );
        }
        pass.finish()?;
        let chroma_count = (chroma_cells.0 * chroma_cells.1) as usize;
        // SAFETY: the buffer is this element's own, `MEANS` cells of four
        // `f32` in shared memory, of which the thumbnail's are read; the pass
        // that wrote them has finished, and nothing writes them again until
        // the next picture's, which takes `&mut self`.
        let means = unsafe {
            std::slice::from_raw_parts(
                self.means.contents().as_ptr().cast::<[f32; 4]>(),
                luma_cells + chroma_count,
            )
        };
        let (luma, chroma) = means.split_at(luma_cells);
        let (y, cb, cr) = if ten_bits { (1, 1, 3) } else { (0, 0, 1) };
        Ok(Thumbnail {
            cells,
            luma: luma.iter().map(|cell| cell[y]).collect(),
            chroma_cells,
            chroma: chroma
                .iter()
                .flat_map(|cell| [cell[cb], cell[cr]])
                .collect(),
        })
    }
}

impl Element for Detecting {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalCutDetector
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Detecting {
    /// NV12 or P010 VideoToolbox pictures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
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
            return Err(MetalCutDetectorError::UnsupportedBuffer(kind).into());
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
    use crate::elements::{AppSink, VideoToolboxDownload, VideoToolboxUpload};
    use crate::test_support::{capture, try_videotoolbox_device};

    /// A picture in which no sample is its neighbour's.
    fn noise(width: u32, height: u32, seed: u32) -> ffmpeg::frame::Video {
        noise_of(Pixel::NV12, width, height, seed)
    }

    /// The same in `format`: NV12 or P010, every byte of it.
    fn noise_of(format: Pixel, width: u32, height: u32, seed: u32) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
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
    /// size the cells divide evenly and one they do not, NV12 and P010 —
    /// whose high bytes are what is read.
    #[test]
    fn the_gpu_makes_the_cpus_thumbnail() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut detecting = MetalCutDetector::new("cuts", CutDetectorOptions::default())
            .unwrap()
            .0;
        for (format, width, height) in [
            (Pixel::NV12, 128, 72),
            (Pixel::NV12, 1918, 1078),
            (Pixel::P010LE, 128, 72),
            (Pixel::P010LE, 1918, 1078),
        ] {
            let mut upload = VideoToolboxUpload::new("upload", &device);
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(noise_of(format, width, height, width)))
                .expect("uploads");
            let MediaBuffer::Video(frame) = uploaded.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            // The CPU reads what came down.
            let mut download = VideoToolboxDownload::new("download");
            let downloaded = capture(&mut download);
            download
                .consume(MediaBuffer::Video(frame.clone()))
                .expect("downloads");
            let MediaBuffer::Video(down) = downloaded.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            let gpu = detecting.inner_mut().thumbnail(&frame).expect("thumbnail");
            assert_eq!(down.format(), format);
            assert_eq!(
                Some(gpu),
                thumbnail_of(&down),
                "{format:?} {width}x{height}"
            );
        }
    }

    /// A picture in system memory is refused, and nothing is handed on.
    #[test]
    fn a_picture_in_system_memory_is_refused() {
        let mut detector = MetalCutDetector::new("cuts", CutDetectorOptions::default()).unwrap();
        let kept = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        detector.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        let refused = detector.consume(MediaBuffer::video(noise(64, 36, 1)));
        assert!(refused.is_err(), "a CPU picture is no VideoToolbox one");
        assert!(kept.lock().unwrap().is_empty());
    }
}
