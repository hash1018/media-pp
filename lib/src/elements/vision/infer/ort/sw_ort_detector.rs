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

use super::{
    DetectorDecoder, Interval, Letterbox, ModelInput, ModelOutput, OrtDetectorError,
    OrtDetectorOptions, decode_batch, labels, model_input,
};
use crate::orientation::{Orientation, Orientations};

/// The grey a fitted picture's margins are filled with, as Ultralytics
/// trains on: 114 of 255.
const MARGIN: f32 = 114.0 / 255.0;

/// Runs a detector — YOLO, or the kind [`OrtDetectorOptions::model`]
/// says — on each picture on the CPU and hands the picture on
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
    /// The values the model wants.
    values: ModelInput,
    /// What reads its output.
    decoder: Arc<dyn DetectorDecoder>,
    /// Which pictures it looks at.
    interval: Interval,
    fitting: Option<Fitting>,
    /// How each picture is turned to be shown.
    orientations: Orientations,
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
    orientation: Orientation,
    letterbox: Letterbox,
    context: ffmpeg::software::scaling::Context,
    /// The picture at the size it is fitted at, still as it is stored.
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
        let interval = Interval::new(options.interval);
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            session,
            values: options.model.input(),
            decoder: options.model.decoder(),
            options,
            labels,
            model,
            fitting: None,
            orientations: Orientations::default(),
            input: Array4::zeros((1, 3, model.1 as usize, model.0 as usize)),
            interval,
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
        let orientation = self.orientations.of(frame, &self.pp_log);
        let fitting = match &mut self.fitting {
            Some(fitting) if fitting.from == from && fitting.orientation == orientation => fitting,
            slot => slot.insert(fitting(from, orientation, self.model)?),
        };
        fitting
            .context
            .run(frame, &mut fitting.scaled)
            .map_err(OrtDetectorError::Convert)?;

        let letterbox = fitting.letterbox;
        let (offset_x, offset_y) = (letterbox.offset.0 as usize, letterbox.offset.1 as usize);
        let (width, height) = (letterbox.scaled.0 as usize, letterbox.scaled.1 as usize);
        let stride = fitting.scaled.stride(0);
        let data = fitting.scaled.data(0);
        // Red, green and blue each into its plane of the model's, and each
        // plane's value made the model's.
        let planes = self.values.planes();
        let (scale, bias) = (self.values.scale, self.values.bias);
        let value = |plane: usize, value: f32| value * scale[plane] + bias[plane];
        for plane in 0..3 {
            self.input
                .index_axis_mut(ndarray::Axis(1), plane)
                .fill(value(plane, MARGIN));
        }
        if orientation.is_upright() {
            for y in 0..height {
                let row = &data[y * stride..y * stride + width * 3];
                for (x, pixel) in row.as_chunks::<3>().0.iter().enumerate() {
                    for (channel, sample) in pixel.iter().enumerate() {
                        let plane = planes[channel];
                        self.input[[0, plane, offset_y + y, offset_x + x]] =
                            value(plane, f32::from(*sample) / 255.0);
                    }
                }
            }
        } else {
            // Each pixel of the input from where the turn stored it.
            let [xx, xy, x0, yx, yy, y0] =
                orientation.sampling(fitting.scaled.width(), fitting.scaled.height());
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let at = (yx * x + yy * y + y0) as usize * stride
                        + (xx * x + xy * y + x0) as usize * 3;
                    for (channel, &plane) in planes.iter().enumerate() {
                        self.input[[0, plane, offset_y + y as usize, offset_x + x as usize]] =
                            value(plane, f32::from(data[at + channel]) / 255.0);
                    }
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
        let names: Vec<&str> = outputs.iter().map(|(name, _)| name).collect();
        let tensors = names
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                let (shape, data) = outputs[index].try_extract_tensor::<f32>()?;
                let shape: Vec<usize> = shape.iter().map(|&side| side.max(0) as usize).collect();
                Ok((name, shape, data))
            })
            .collect::<std::result::Result<Vec<_>, OrtDetectorError>>()?;
        let outputs: Vec<ModelOutput<'_>> = tensors
            .iter()
            .map(|(name, shape, data)| ModelOutput { name, shape, data })
            .collect();
        let found = decode_batch(
            &*self.decoder,
            &outputs,
            &[letterbox],
            self.model,
            &self.options,
        )?
        .pop()
        .unwrap_or_default();
        Ok(Detections::new(
            Arc::clone(&self.name),
            Arc::clone(&self.labels),
            found,
        ))
    }
}

/// A conversion of pictures shaped `from`, shown turned as `orientation`
/// says, to RGB24 at the size they fit the model's input at — still as they
/// are stored, the input being read from it turned.
fn fitting(
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    orientation: Orientation,
    model: (u32, u32),
) -> std::result::Result<Fitting, OrtDetectorError> {
    let (width, height) = (from.1, from.2);
    let letterbox = Letterbox::shown((width, height), model, orientation);
    // The turned size, turned back: a quarter turn swaps the sides back.
    let stored = orientation.display_size(letterbox.scaled.0, letterbox.scaled.1);
    let context = to_rgb24(from, stored)?;
    Ok(Fitting {
        from,
        orientation,
        letterbox,
        context,
        scaled: ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, stored.0, stored.1),
    })
}

/// A conversion of pictures shaped `from` to RGB24 at `size`, by the
/// colours they say they are in — or, untagged, those the CUDA and Metal
/// paths guess for them too (`color::matrix_of`): a detector on the CPU and
/// one on the GPU are handed the same colours of the same picture, where
/// swscale's own reading, BT.601 whatever the size, had an untagged HD
/// picture's scores differ by about 0.05.
pub(super) fn to_rgb24(
    from: (
        ffmpeg::format::Pixel,
        u32,
        u32,
        ffmpeg::color::Space,
        ffmpeg::color::Range,
    ),
    size: (u32, u32),
) -> std::result::Result<ffmpeg::software::scaling::Context, OrtDetectorError> {
    let (format, width, height, space, range) = from;
    let mut context = ffmpeg::software::scaling::Context::get(
        format,
        width,
        height,
        ffmpeg::format::Pixel::RGB24,
        size.0,
        size.1,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;
    if !is_rgb(format) {
        // SAFETY: `context` is a live `SwsContext` this fitting owns, and the
        // tables are swscale's own static ones.
        unsafe {
            let source =
                ffmpeg::ffi::sws_getCoefficients(matrix(crate::color::matrix_of(space, height)));
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
    Ok(context)
}

/// Whether `format` is a hardware frame's, whose pixels are not in it.
pub(super) fn is_hardware(format: ffmpeg::format::Pixel) -> bool {
    crate::elements::vision::is_hardware(format)
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
        if !self.interval.look(&buf) {
            // Let by unlooked-at, carrying nothing, which says so.
            out.push(buf);
            return Ok(());
        }
        let detections = self.detect(frame)?;
        out.push(detections.attach_to(buf));
        Ok(())
    }

    fn reset(&mut self) {
        self.interval.restart();
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

    /// An untagged picture is read in the colours the GPU paths guess for
    /// it: one 1080 lines tall as BT.709 — whose red, (72, 107, 220) as
    /// Y'CbCr, comes out red — and one of 480 as BT.601, each as the rows
    /// the CUDA fitting is handed read the same samples, within rounding.
    #[test]
    fn an_untagged_picture_is_read_as_the_gpu_reads_it() {
        for height in [1080, 480] {
            let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, height);
            let (luma, chroma) = (frame.stride(0), frame.stride(1));
            frame.data_mut(0)[..luma * height as usize].fill(72);
            for pair in frame.data_mut(1)[..chroma * (height / 2) as usize]
                .as_chunks_mut::<2>()
                .0
            {
                *pair = [107, 220];
            }
            let from = (
                frame.format(),
                64,
                height,
                frame.color_space(),
                frame.color_range(),
            );
            assert_eq!(from.3, ffmpeg::color::Space::Unspecified);
            let mut rgb = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 64, height);
            to_rgb24(from, (64, height))
                .expect("a conversion")
                .run(&frame, &mut rgb)
                .expect("converted");
            let got = &rgb.data(0)[..3];

            let rows = crate::color::yuv_to_rgb_rows(from.3, from.4, height);
            let sample = [72.0, 107.0, 220.0, 255.0].map(|v: f32| v / 255.0);
            let want: Vec<u8> = rows
                .iter()
                .map(|row| {
                    let v: f32 = row.iter().zip(sample).map(|(k, s)| k * s).sum();
                    (v.clamp(0.0, 1.0) * 255.0).round() as u8
                })
                .collect();
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(got, want)| got.abs_diff(*want) <= 2),
                "{height} lines: {got:?} against {want:?}"
            );
            if height == 1080 {
                assert!(got[0] > 200 && got[1] < 40 && got[2] < 40, "red: {got:?}");
            }
        }
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
