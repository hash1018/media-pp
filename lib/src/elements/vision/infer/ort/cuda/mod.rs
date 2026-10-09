//! [`CudaOrtDetector`]: detection on CUDA pictures, on CUDA or through
//! TensorRT.

mod cuda_ort_classifier;
mod cuda_ort_detector;
mod cuda_ort_embedder;
mod runtime;

pub use cuda_ort_classifier::CudaOrtClassifier;
#[cfg(feature = "ort-tensorrt")]
pub use cuda_ort_detector::UseTensorRtPolicy;
pub use cuda_ort_detector::{CudaOrtDetector, CudaOrtDetectorOptions};
pub use cuda_ort_embedder::{CudaOrtEmbedder, CudaOrtEmbedderOptions};
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
    /// Whether `frame`, a P010 picture, can be brought to SDR: whether it
    /// is tagged HLG or PQ.
    fn check(frame: &ffmpeg::frame::Video, element: ElementType) -> Result<(), SdrCopyError> {
        Self::tone_map(frame, element).map(|_| ())
    }

    /// How `frame`, a P010 picture, is brought to SDR.
    fn tone_map(
        frame: &ffmpeg::frame::Video,
        element: ElementType,
    ) -> Result<crate::tone_map::ToneMap, SdrCopyError> {
        crate::tone_map::ToneMap::of_frame(frame).ok_or(SdrCopyError::Frame(
            CudaFrameError::UnsupportedSurfaceFormat {
                element,
                accepts: CudaSurfaces::NV12_OR_BGRA,
                actual: ffmpeg::format::Pixel::P010LE,
            },
        ))
    }

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
        let tone_map = Self::tone_map(frame, element)?;
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

/// The batch TensorRT builds a second model's engine to be quickest at: a
/// picture has a few objects to classify at once, seldom the most there is
/// room for. Every batch from 1 to the most runs on the one engine. On an
/// RTX 3050, MobileNetV2's engine built for 2 ran one to three objects
/// 2–10% faster than one built for 8, and eight 6% slower; a pipeline
/// classifying every object on every picture went 1% faster.
#[cfg(feature = "ort-tensorrt")]
const TENSORRT_OPT_BATCH: usize = 2;

/// A session for a model run on the objects a detector found — a
/// classifier's, an embedder's — on CUDA, or through TensorRT in half
/// precision where it can run: its input, `open` for a side it leaves open,
/// what it runs on, for the log, and the libraries asked. Its engine is
/// built for every batch from one to `most`, where the model's batch is
/// open, and kept in `engine_cache` where one is given (see
/// [`engine_paths`]); `doing` is what a warning says runs on CUDA alone.
fn object_model_session(
    model_path: &std::path::Path,
    open: u32,
    most: usize,
    engine_cache: Option<&std::path::Path>,
    pp_log: &crate::pp_log::PpLog,
    doing: &str,
) -> Result<
    (
        ort::session::Session,
        super::classify::Input,
        &'static str,
        runtime::Linked,
    ),
    super::OrtError,
> {
    #[cfg(feature = "ort-tensorrt")]
    use crate::pp_log::pp_warn;
    #[cfg(feature = "ort-tensorrt")]
    use ort::ep::TensorRT;
    use ort::{ep::CUDA, session::Session};

    use super::OrtError;
    use runtime::CudaRuntime;

    let linked = runtime::Linked::ask();
    let runtime = runtime::runtime(&linked);
    #[cfg(feature = "ort-tensorrt")]
    if let CudaRuntime::CudaOnly { tensorrt } = &runtime {
        pp_warn!(pp_log: pp_log, "TensorRT cannot be used ({tensorrt}): {doing} on CUDA alone");
    }
    #[cfg(not(feature = "ort-tensorrt"))]
    let _ = (pp_log, doing, engine_cache);
    #[cfg(feature = "ort-tensorrt")]
    let tensorrt = runtime == CudaRuntime::TensorRt;
    if let CudaRuntime::Unavailable { cuda } = runtime {
        return Err(OrtError::CudaRuntimeUnavailable(cuda));
    }

    // The input is read first, on the CPU: TensorRT is told the batches
    // it will be given before its session is made.
    let probe = Session::builder()?.commit_from_file(model_path)?;
    let input = super::classify::image_input(&probe, open)?;
    #[cfg(feature = "ort-tensorrt")]
    let input_name = probe.inputs()[0].name().to_owned();
    drop(probe);
    #[cfg(not(feature = "ort-tensorrt"))]
    let _ = most;

    let mut providers = Vec::new();
    #[cfg(feature = "ort-tensorrt")]
    let provider = if tensorrt {
        let capacity = input.batch.unwrap_or(most).max(1);
        let profile = input.batch.is_none().then(|| BatchProfile {
            min: 1,
            opt: TENSORRT_OPT_BATCH.min(capacity),
            max: capacity,
        });
        let (cache, timing) = engine_paths(engine_cache, profile);
        if let Err(error) = std::fs::create_dir_all(&cache) {
            pp_warn!(pp_log: pp_log, "no engine cache at {}: {error}", cache.display());
        }
        let mut provider = TensorRT::default()
            .with_device_id(0)
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path(cache.display())
            .with_timing_cache(true)
            .with_timing_cache_path(timing.display());
        if let Some(profile) = profile {
            let (width, height) = input.size;
            let shape = |batch: usize| format!("{input_name}:{batch}x3x{height}x{width}");
            provider = provider
                .with_profile_min_shapes(shape(profile.min))
                .with_profile_opt_shapes(shape(profile.opt))
                .with_profile_max_shapes(shape(profile.max));
        }
        // Preferred: where TensorRT will not start, ONNX Runtime passes
        // over it to CUDA.
        providers.push(provider.build());
        "TensorRT, fp16"
    } else {
        "CUDA"
    };
    #[cfg(not(feature = "ort-tensorrt"))]
    let provider = "CUDA";
    providers.push(CUDA::default().with_device_id(0).build().error_on_failure());
    let session = Session::builder()?
        .with_execution_providers(providers)
        .map_err(|error| OrtError::Ort(error.into()))?
        .commit_from_file(model_path)?;
    Ok((session, input, provider, linked))
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

/// Where an engine built for `profile` is kept, and the timing cache: in
/// `given` both, where a directory is given — it is used as it is, and the
/// caller keeps engines of one model built for different batches apart —
/// and otherwise in [`default_engine_cache`], the engine in a directory of
/// its batches' own ([`engine_directory`]).
#[cfg(feature = "ort-tensorrt")]
fn engine_paths(
    given: Option<&std::path::Path>,
    profile: Option<BatchProfile>,
) -> (std::path::PathBuf, std::path::PathBuf) {
    match given {
        Some(dir) => (dir.to_path_buf(), dir.to_path_buf()),
        None => {
            let root = default_engine_cache();
            (engine_directory(&root, profile), root)
        }
    }
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

    use super::{BatchProfile, engine_directory, engine_paths};

    /// A directory given keeps the engine and the timing cache both, as it
    /// is, whatever the batches; none given is the user's cache, the engine
    /// apart by its batches.
    #[test]
    fn a_directory_given_keeps_engine_and_timing_as_it_is() {
        let given = Path::new("/app/cache/tensorrt");
        let profile = Some(BatchProfile {
            min: 1,
            opt: 2,
            max: 32,
        });
        assert_eq!(
            engine_paths(Some(given), profile),
            (given.to_path_buf(), given.to_path_buf())
        );
        let (engine, timing) = engine_paths(None, profile);
        assert_eq!(engine, timing.join("batch-1-2-32"));
        assert!(timing.ends_with("media-pp/tensorrt"));
    }

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
