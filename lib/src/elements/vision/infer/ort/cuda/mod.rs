//! [`CudaOrtDetector`]: detection on CUDA pictures, on CUDA or through
//! TensorRT.

mod cuda_ort_classifier;
mod cuda_ort_detector;
mod runtime;

pub use cuda_ort_classifier::CudaOrtClassifier;
#[cfg(feature = "ort-tensorrt")]
pub use cuda_ort_detector::UseTensorRtPolicy;
pub use cuda_ort_detector::{CudaOrtDetector, CudaOrtDetectorOptions};
pub use runtime::{CudaRuntime, LibraryVersion, RuntimeShortfall};

use crate::element::ElementType;
use crate::ffmpeg;
use crate::platform::cuda::{
    CudaFrameError, CudaSurfaces,
    driver::{BgraSurface, CudaBgraScratch, CudaDriver, CudaDriverError, Nv12Surface},
};

/// An HDR picture — P010, tagged HLG or PQ, as NVDEC decodes an iPhone's —
/// brought to SDR BGRA for a model, which was trained on SDR pictures: by
/// `core/tone_map.rs`'s definition, the one `CudaConverter` brings HDR to
/// SDR by, into a buffer kept for the next picture of its size. What the
/// model finds goes on the ten-bit picture itself.
#[derive(Default)]
struct SdrCopy {
    scratch: Option<CudaBgraScratch>,
}

impl SdrCopy {
    /// `frame`, a P010 picture of this element's device, in SDR BGRA, the
    /// conversion launched and not waited for — a fitting launched after
    /// it on the same stream reads it once it is written.
    ///
    /// # Errors
    ///
    /// A P010 picture tagged neither HLG nor PQ, which has no tone map:
    /// [`crate::elements::CudaScaler`] brings SDR P010 to NV12.
    fn of(
        &mut self,
        driver: &CudaDriver,
        frame: &ffmpeg::frame::Video,
        element: ElementType,
    ) -> Result<BgraSurface, SdrCopyError> {
        let tone_map = crate::tone_map::ToneMap::of_frame(frame).ok_or(SdrCopyError::Frame(
            CudaFrameError::UnsupportedSurfaceFormat {
                element,
                accepts: CudaSurfaces::NV12_OR_BGRA,
                actual: ffmpeg::format::Pixel::P010LE,
            },
        ))?;
        let (width, height) = (frame.width(), frame.height());
        let scratch = match self.scratch.take() {
            Some(scratch) if (scratch.width, scratch.height) == (width, height) => scratch,
            _ => driver.bgra_scratch(width, height)?,
        };
        let source = Nv12Surface::from_frame(frame).ok_or(SdrCopyError::MissingSurface)?;
        let surface = scratch.surface();
        driver.hdr_to_bgra(source, true, surface, width, height, &tone_map)?;
        self.scratch = Some(scratch);
        Ok(surface)
    }
}

/// Why an [`SdrCopy`] could not be made.
#[derive(Debug, thiserror::Error)]
enum SdrCopyError {
    #[error(transparent)]
    Frame(#[from] CudaFrameError),
    #[error(transparent)]
    Driver(#[from] CudaDriverError),
    #[error("a P010 picture has no surface behind it")]
    MissingSurface,
}

impl From<SdrCopyError> for super::OrtError {
    fn from(error: SdrCopyError) -> Self {
        match error {
            SdrCopyError::Frame(error) => error.into(),
            SdrCopyError::Driver(error) => error.into(),
            SdrCopyError::MissingSurface => Self::MissingSurface,
        }
    }
}

/// Where TensorRT keeps the engines it builds, unless told otherwise:
/// `$XDG_CACHE_HOME/media-pp/tensorrt`, or the platform's own cache
/// directory, or the temporary one where there is neither. ONNX Runtime
/// names an engine after its model's graph, so a detector's and a
/// classifier's sit side by side.
#[cfg(feature = "ort-tensorrt")]
fn default_engine_cache() -> std::path::PathBuf {
    use std::path::PathBuf;
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("media-pp").join("tensorrt")
}

/// The batches a TensorRT engine is built for: the fewest, the one it is
/// fastest at, and the most.
#[cfg(feature = "ort-tensorrt")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchProfile {
    min: usize,
    opt: usize,
    max: usize,
}

/// Where in the cache directory `root` the engine for `profile` is kept.
///
/// ONNX Runtime names an engine after its model's graph and precision, not
/// after the batches it was built for: two detectors of one model, one
/// picture at a time and four batched, would find each other's engine in
/// one directory, find it built for other shapes, and each build its own
/// again on every start — minutes. An engine built for a range of batches
/// is kept in a directory of that range's own, `batch-1-4-4`; one built
/// for no range, a model's own batch, in `root` itself. The timing cache
/// stays in `root`, shared: it makes any engine's build quicker, whatever
/// its shapes.
#[cfg(feature = "ort-tensorrt")]
fn engine_directory(root: &std::path::Path, profile: Option<BatchProfile>) -> std::path::PathBuf {
    match profile {
        None => root.to_path_buf(),
        Some(BatchProfile { min, opt, max }) => root.join(format!("batch-{min}-{opt}-{max}")),
    }
}

#[cfg(all(test, feature = "ort-tensorrt"))]
mod tests {
    use std::path::Path;

    use super::{BatchProfile, engine_directory};

    /// Each range of batches has a directory of its own, so that engines
    /// built for two do not take each other's place; no range is the cache
    /// itself, where engines built before ranges were kept apart still are.
    #[test]
    fn each_range_of_batches_keeps_its_engine_apart() {
        let root = Path::new("/cache/tensorrt");
        let range = |min, opt, max| engine_directory(root, Some(BatchProfile { min, opt, max }));
        assert_eq!(engine_directory(root, None), root);
        assert_eq!(range(1, 4, 4), root.join("batch-1-4-4"));
        assert_ne!(range(1, 4, 4), range(1, 8, 8));
        assert_ne!(range(1, 8, 32), range(1, 32, 32));
    }
}

#[cfg(test)]
mod sdr_copy_tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::buffer::MediaBuffer;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, CudaConverter, CudaDownload, CudaUpload};
    use crate::platform::cuda::CudaFrameFormat;
    use crate::test_support::try_cuda_device;

    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// A P010 picture tagged `transfer`, uploaded.
    fn p010(
        device: &crate::elements::CudaDevice,
        transfer: ffmpeg::color::TransferCharacteristic,
    ) -> MediaBuffer {
        let (width, height) = (64u32, 36u32);
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::P010LE, width, height);
        for plane in 0..2 {
            let stride = frame.stride(plane);
            let rows = if plane == 0 { height } else { height / 2 } as usize;
            for y in 0..rows {
                for x in 0..width as usize {
                    let value = (((x * 13 + y * 29) % 900 + 64) as u16) << 6;
                    frame.data_mut(plane)[y * stride + x * 2..][..2]
                        .copy_from_slice(&value.to_le_bytes());
                }
            }
        }
        frame.set_color_space(ffmpeg::color::Space::BT2020NCL);
        frame.set_color_primaries(ffmpeg::color::Primaries::BT2020);
        frame.set_color_transfer_characteristic(transfer);
        frame.set_pts(Some(0));
        crate::buffer::set_time_base(&mut frame, ffmpeg::Rational(1, 30));
        let mut upload = CudaUpload::new("upload", device, CudaFrameFormat::P010);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("uploads");
        uploaded.lock().unwrap().remove(0)
    }

    /// An HDR picture is brought to SDR for a model exactly as
    /// `CudaConverter` brings it — the same bytes — whether HLG or PQ; a
    /// P010 picture tagged neither is refused, naming the element.
    #[test]
    fn an_hdr_picture_is_brought_to_sdr_as_the_converter_brings_it() {
        let Some((device, _serial)) = try_cuda_device() else {
            return;
        };
        let driver = CudaDriver::retain_primary().expect("driver");
        for transfer in [
            ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
            ffmpeg::color::TransferCharacteristic::SMPTE2084,
        ] {
            let MediaBuffer::Video(frame) = p010(&device, transfer) else {
                panic!("a picture");
            };
            let mut copy = SdrCopy::default();
            let surface = copy
                .of(&driver, &frame, ElementType::CudaOrtDetector)
                .expect("an SDR copy");
            driver.synchronize().expect("made");
            let (width, height) = (frame.width() as usize, frame.height() as usize);
            let mut ours = vec![0u8; width * height * 4];
            driver
                .download_rect(
                    surface.pixels,
                    surface.pitch,
                    0,
                    0,
                    width * 4,
                    height,
                    &mut ours,
                )
                .expect("read back");

            let mut converter =
                CudaConverter::new("converter", &device, CudaFrameFormat::Bgra).expect("converter");
            let converted = capture(&mut converter);
            converter
                .consume(MediaBuffer::Video(frame.clone()))
                .expect("converts");
            let mut download = CudaDownload::new("download", &device, CudaFrameFormat::Bgra);
            let downloaded = capture(&mut download);
            download
                .consume(converted.lock().unwrap().remove(0))
                .expect("downloads");
            let MediaBuffer::Video(theirs) = downloaded.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            for y in 0..height {
                let row = &theirs.data(0)[y * theirs.stride(0)..][..width * 4];
                assert_eq!(
                    &ours[y * width * 4..][..width * 4],
                    row,
                    "{transfer:?}, row {y}"
                );
            }
        }
        let MediaBuffer::Video(sdr) = p010(&device, ffmpeg::color::TransferCharacteristic::BT709)
        else {
            panic!("a picture");
        };
        assert!(matches!(
            SdrCopy::default().of(&driver, &sdr, ElementType::CudaOrtDetector),
            Err(SdrCopyError::Frame(
                CudaFrameError::UnsupportedSurfaceFormat {
                    element: ElementType::CudaOrtDetector,
                    ..
                }
            ))
        ));
    }
}
