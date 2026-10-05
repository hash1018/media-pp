//! [`SwOrtDetector`]: object detection on the CPU, each picture handed on
//! with what was found in it.

use std::{path::Path, sync::Arc};

use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

use crate::ffmpeg;
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::filter::scaler::{is_rgb, matrix},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use crate::elements::Detections;

use super::{Letterbox, OrtDetectorError, OrtDetectorOptions, decode, labels, model_input};

/// The grey a fitted picture's margins are filled with, as Ultralytics
/// trains on: 114 of 255.
const MARGIN: f32 = 114.0 / 255.0;

/// Runs a YOLO detector on each picture on the CPU and hands the picture on
/// unchanged, carrying the [`Detections`] found in it — read them
/// downstream with `buffer.metadata()?.get::<Detections>()`.
///
/// It takes decoded video in system memory, any pixel format and size: each
/// picture is fitted to the model's input inside the element (see
/// [`OrtDetectorOptions`] for the models it reads and how), and the
/// boxes are mapped back onto the picture as fractions of it. What goes on
/// is the picture it was handed, not the fitted copy.
///
/// Every picture is looked at, as fast as the CPU runs the model — tens of
/// milliseconds a picture for a small model. Where pictures come faster
/// than that, a dropping queue in front of it keeps the rest of the
/// pipeline from waiting on it.
pub struct SwOrtDetector(FilterStage<Detecting>);

filter_stage!(SwOrtDetector);

/// What an [`SwOrtDetector`] does with each picture.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtDetectorOptions,
    labels: Arc<[Arc<str>]>,
    model: (u32, u32),
    fitting: Option<Fitting>,
    /// The model's input, kept from one picture to the next.
    input: Array4<f32>,
}

/// One picture shape's conversion to the model's, and the picture it makes.
struct Fitting {
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    letterbox: Letterbox,
    context: ffmpeg::software::scaling::Context,
    scaled: ffmpeg::frame::Video,
}

// SAFETY: the scaling context and frame are heap allocations owned solely by
// this fitting, used only by the one thread transforming at a time, exactly
// as `SwScaler` holds its own.
unsafe impl Send for Fitting {}

impl SwOrtDetector {
    /// Loads the model at `model_path` to run on the CPU.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        options: OrtDetectorOptions,
    ) -> Result<Self> {
        let path = model_path.as_ref().display().to_string();
        let session = Session::builder()
            .map_err(OrtDetectorError::from)?
            .commit_from_file(model_path)
            .map_err(OrtDetectorError::from)?;
        let model = model_input(&session)?;
        let labels = labels(options.labels.as_deref(), &session);
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwOrtDetector, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, {} labels, conf_threshold={}, iou_threshold={}",
            model.0,
            model.1,
            labels.len(),
            options.conf_threshold,
            options.iou_threshold
        );
        if labels.is_empty() {
            pp_warn!(
                pp_log: &pp_log,
                "the model names no classes and none were given: detections carry class numbers only"
            );
        }
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            session,
            options,
            labels,
            model,
            fitting: None,
            input: Array4::zeros((1, 3, model.1 as usize, model.0 as usize)),
        })))
    }
}

impl Detecting {
    /// Fits `frame` into the model's input, reusing the conversion while
    /// pictures keep their shape, and says how it was fitted.
    fn fit(&mut self, frame: &ffmpeg::frame::Video) -> Result<Letterbox> {
        let from = (
            frame.format(),
            frame.width(),
            frame.height(),
            frame.color_space(),
            frame.color_range(),
        );
        if self
            .fitting
            .as_ref()
            .is_none_or(|fitting| fitting.from != from)
        {
            self.fitting = Some(fitting(from, self.model)?);
        }
        let fitting = self.fitting.as_mut().expect("made above");
        fitting
            .context
            .run(frame, &mut fitting.scaled)
            .map_err(OrtDetectorError::Convert)?;

        let letterbox = fitting.letterbox;
        let (offset_x, offset_y) = (letterbox.offset.0 as usize, letterbox.offset.1 as usize);
        let (width, height) = (letterbox.scaled.0 as usize, letterbox.scaled.1 as usize);
        let stride = fitting.scaled.stride(0);
        let data = fitting.scaled.data(0);
        self.input.fill(MARGIN);
        for y in 0..height {
            let row = &data[y * stride..y * stride + width * 3];
            for (x, pixel) in row.as_chunks::<3>().0.iter().enumerate() {
                for (channel, value) in pixel.iter().enumerate() {
                    self.input[[0, channel, offset_y + y, offset_x + x]] =
                        f32::from(*value) / 255.0;
                }
            }
        }
        Ok(letterbox)
    }

    /// Looks at `frame`, and says what it found.
    fn detect(&mut self, frame: &ffmpeg::frame::Video) -> Result<Detections> {
        let letterbox = self.fit(frame)?;
        let outputs = self
            .session
            .run(inputs![
                TensorRef::from_array_view(&self.input).map_err(OrtDetectorError::from)?
            ])
            .map_err(OrtDetectorError::from)?;
        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(OrtDetectorError::from)?;
        Ok(Detections {
            detector: Arc::clone(&self.name),
            labels: Arc::clone(&self.labels),
            items: decode(output, &letterbox, &self.options)?,
        })
    }
}

/// A conversion of pictures shaped `from` to RGB24 at the size they fit the
/// model's input at.
fn fitting(
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    model: (u32, u32),
) -> std::result::Result<Fitting, OrtDetectorError> {
    let (format, width, height, space, range) = from;
    let letterbox = Letterbox::new((width, height), model);
    let mut context = ffmpeg::software::scaling::Context::get(
        format,
        width,
        height,
        ffmpeg::format::Pixel::RGB24,
        letterbox.scaled.0,
        letterbox.scaled.1,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;
    if !is_rgb(format) {
        // SAFETY: `context` is a live `SwsContext` this fitting owns, and the
        // tables are swscale's own static ones.
        unsafe {
            let source = ffmpeg::ffi::sws_getCoefficients(matrix(space));
            let full = i32::from(range == ffmpeg::color::Range::JPEG);
            ffmpeg::ffi::sws_setColorspaceDetails(
                context.as_mut_ptr(),
                source,
                full,
                source,
                1,
                0,
                1 << 16,
                1 << 16,
            );
        }
    }
    Ok(Fitting {
        from,
        letterbox,
        context,
        scaled: ffmpeg::frame::Video::new(
            ffmpeg::format::Pixel::RGB24,
            letterbox.scaled.0,
            letterbox.scaled.1,
        ),
    })
}

/// Whether `format` is a hardware frame's, whose pixels are not in it.
fn is_hardware(format: ffmpeg::format::Pixel) -> bool {
    // SAFETY: a lookup in libavutil's static table of descriptors.
    unsafe {
        let descriptor = ffmpeg::ffi::av_pix_fmt_desc_get(format.into());
        !descriptor.is_null()
            && (*descriptor).flags & (ffmpeg::ffi::AV_PIX_FMT_FLAG_HWACCEL as u64) != 0
    }
}

impl Element for Detecting {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwOrtDetector
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Detecting {
    /// Decoded video in system memory, any pixel layout.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ))
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let refused = |got| OrtDetectorError::UnsupportedBuffer {
            detector: "SwOrtDetector",
            wanted: "video frames in system memory",
            got,
        };
        let MediaBuffer::Video(frame) = &buf else {
            return Err(refused(buf.kind()).into());
        };
        if is_hardware(frame.format()) {
            return Err(refused("a hardware video frame").into());
        }
        let detections = self.detect(frame)?;
        out.push(detections.attach_to(buf));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{RawSink, SrcPads};

    /// The model this machine's tests can run, where it has one: a YOLO
    /// ONNX export named by `MEDIA_PP_TEST_YOLO`.
    fn model() -> Option<String> {
        let path = std::env::var("MEDIA_PP_TEST_YOLO").ok()?;
        Path::new(&path).is_file().then_some(path)
    }

    /// The real thing, where a model is given: a picture goes on unchanged,
    /// carrying what was found in it — for a flat grey picture, nothing,
    /// which is still an answer.
    #[test]
    fn a_picture_goes_on_carrying_what_was_found() {
        let Some(model) = model() else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO to a YOLO ONNX model to run this");
            return;
        };
        let mut detector =
            SwOrtDetector::new("detector", &model, OrtDetectorOptions::default()).expect("loads");
        let kept = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = kept.clone();
        detector.src_pads()[0].link(Box::new(crate::elements::AppSink::new(
            "kept",
            move |buf| {
                sink.lock().unwrap().push(buf);
                Ok(())
            },
        )));
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 1280, 720);
        frame.data_mut(0).fill(128);
        frame.data_mut(1).fill(128);
        frame.set_pts(Some(7));
        detector
            .consume(MediaBuffer::video(frame))
            .expect("detects");
        let kept = kept.lock().unwrap();
        let MediaBuffer::Video(out) = &kept[0] else {
            panic!("a picture goes on");
        };
        assert_eq!((out.width(), out.height(), out.pts()), (1280, 720, Some(7)));
        let found = kept[0]
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .expect("carries Detections");
        assert_eq!(&*found.detector, "detector");
    }

    #[test]
    fn a_missing_model_says_so() {
        let error = SwOrtDetector::new(
            "detector",
            "/nonexistent.onnx",
            OrtDetectorOptions::default(),
        )
        .err()
        .expect("no model there");
        assert!(error.to_string().contains("onnxruntime"), "{error}");
    }
}
