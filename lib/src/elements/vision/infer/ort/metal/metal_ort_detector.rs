//! [`MetalOrtDetector`]: object detection on VideoToolbox pictures, each
//! handed on with what was found in it, the picture fitted to the model on
//! the GPU.

use std::{path::Path, sync::Arc};

use ndarray::{Axis, Slice};
use ort::{inputs, session::Session, value::TensorRef, value::ValueType};

use crate::ffmpeg;
use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    bus::BusEvent,
    contract::{InputContract, MediaKind, MemoryDomain, PixelLayoutSet, PortContract},
    element::{Context, Element, ElementType, element_pp_log},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use crate::elements::{BatchSlot, Detection, Detections};

use super::super::{
    Interval, Letterbox, OrtDetectorOptions, OrtError, decode_batch, labels, model_input,
};
use super::core_ml_session;
use super::fitting::{Cut, Fitting, Picture};

/// How a [`MetalOrtDetector`] runs its model, beside what every detector is
/// told.
#[derive(Debug, Clone, PartialEq)]
pub struct MetalOrtDetectorOptions {
    /// Thresholds and labels, as every detector takes them.
    pub detector: OrtDetectorOptions,
    /// The most pictures the model is run on at once: the pictures of a
    /// [`StreamMux`](crate::elements::StreamMux)'s batch, up to this many,
    /// are fitted in one Metal pass and go through the model together —
    /// DeepStream's `batch-size`. Give it the mux's `max_batch`; a larger
    /// batch is run in parts this size.
    ///
    /// It takes a model whose batch is left open, as an Ultralytics export
    /// with `dynamic=True` is, and fixes that batch at this size for Core
    /// ML — the height and width such an export leaves open beside it at
    /// 640 — as Core ML compiles a model of open shape again for each new
    /// batch, and runs much of one it has no bound for on the CPU: a
    /// batch cut short is run at the full size, the inputs past its pictures
    /// left as they were. A model made for one picture at a time — the
    /// YOLOv10n release — is run so, with a warning. The default, 1, is a
    /// detector of one picture at a time.
    pub max_batch: usize,
}

impl Default for MetalOrtDetectorOptions {
    fn default() -> Self {
        Self {
            detector: OrtDetectorOptions::default(),
            max_batch: 1,
        }
    }
}

/// Runs a YOLO detector on each VideoToolbox picture and hands the picture
/// on unchanged, carrying the [`Detections`] found in it — what
/// [`SwOrtDetector`](super::super::SwOrtDetector) does on the CPU, with the
/// picture fitted to the model's input by a Metal kernel where it is, and
/// the model run by ONNX Runtime's Core ML provider on the GPU or the Neural
/// Engine.
///
/// It takes NV12 or BGRA VideoToolbox pictures of any size — a decoder's, a
/// camera's, a screen's — from any device, since a pixel buffer belongs to
/// none; the boxes it finds are fractions of each picture, as
/// [`Detection`](crate::elements::Detection) describes. The models it reads
/// are those [`OrtDetectorOptions`] describes. The pictures of a
/// [`StreamMux`](crate::elements::StreamMux)'s batch are run together, as
/// [`MetalOrtDetectorOptions::max_batch`] says.
///
/// # Where the model runs
///
/// Core ML decides, layer by layer, between the Neural Engine, the GPU and
/// the CPU, and ONNX Runtime gives the CPU whatever operator Core ML does
/// not take — both allowed, as either is still this machine's best. What is
/// not is running without Core ML at all: [`Self::new`] refuses where the
/// provider does not start, rather than run the whole model on the CPU,
/// which is [`SwOrtDetector`](super::super::SwOrtDetector)'s job.
///
/// The fitted input is written to memory the CPU and GPU share, and handed
/// to ONNX Runtime from there: Core ML's provider takes its input in system
/// memory, which on Apple silicon is the GPU's too, so nothing crosses a
/// bus.
///
/// # Requirements
///
/// The `ort-coreml` feature, on an Apple silicon Mac: ONNX Runtime's build
/// with Core ML is fetched as the CPU one is, and there is none for Intel.
pub struct MetalOrtDetector(FilterStage<Detecting>);

filter_stage!(MetalOrtDetector);

/// What a [`MetalOrtDetector`] does with each picture.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtDetectorOptions,
    labels: Arc<[Arc<str>]>,
    /// Which pictures it looks at.
    interval: Interval,
    /// How many pictures the model is run on at once.
    max_batch: usize,
    /// How many inputs the model is handed each run where its batch is
    /// fixed — by the model or for Core ML — or `None` where it takes as
    /// many as there are pictures.
    rows: Option<usize>,
    /// The pictures of the batch under way, in the order they came.
    held: Vec<Held>,
    /// The pipeline's, to report a picture's failure on where others must
    /// go on in the same call.
    context: Option<Arc<Context>>,
    fitting: Fitting,
}

impl MetalOrtDetector {
    /// Loads the model at `model_path` to run through Core ML.
    ///
    /// Core ML compiles the model for this Mac as the session is made,
    /// which takes a moment for a small model and longer for a large one.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        options: MetalOrtDetectorOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalOrtDetector, &name, None);
        let path = model_path.as_ref().display().to_string();
        if options.max_batch == 0 {
            return Err(OrtError::ZeroMaxBatch.into());
        }

        // Read on the CPU first, which compiles nothing: whether the model
        // leaves its batch open, and by what name, to fix it for Core ML —
        // and its height and width, which an Ultralytics export with
        // `dynamic=True` leaves open too, and which Core ML, given no bound
        // for them, hands much of the model back to the CPU over.
        let open = Session::builder()
            .map_err(OrtError::from)?
            .commit_from_file(model_path.as_ref())
            .map_err(OrtError::from)?;
        let (width, height) = model_input(&open)?;
        let (batch, mut fixed) = match open.inputs().first().map(|input| input.dtype()) {
            Some(ValueType::Tensor {
                shape,
                dimension_symbols,
                ..
            }) => {
                let named = |axis: usize| {
                    dimension_symbols
                        .get(axis)
                        .filter(|symbol| !symbol.is_empty() && shape[axis] <= 0)
                        .cloned()
                };
                let batch = match shape.iter().next().copied() {
                    Some(fixed) if fixed > 0 => Batch::Fixed(fixed as usize),
                    _ => named(0).map_or(Batch::Unnamed, Batch::Named),
                };
                let sides = [(2, height), (3, width)]
                    .into_iter()
                    .filter_map(|(axis, side)| Some((named(axis)?, i64::from(side))))
                    .collect::<Vec<_>>();
                (batch, sides)
            }
            _ => (Batch::Fixed(1), Vec::new()),
        };
        drop(open);
        let (max_batch, rows) = match &batch {
            Batch::Fixed(fixed) => {
                if options.max_batch > *fixed {
                    pp_warn!(
                        pp_log: &pp_log,
                        "the model takes {fixed} picture(s) at a time: a max_batch of {} is run so",
                        options.max_batch
                    );
                }
                (options.max_batch.min(*fixed), Some(*fixed))
            }
            Batch::Named(_) => (options.max_batch, Some(options.max_batch)),
            Batch::Unnamed => {
                pp_warn!(
                    pp_log: &pp_log,
                    "the model leaves its batch open without a name to fix it by: \
                     Core ML compiles it again for each number of pictures"
                );
                (options.max_batch, None)
            }
        };
        if let Batch::Named(symbol) = &batch {
            fixed.push((symbol.clone(), options.max_batch as i64));
        }
        let fixed: Vec<(&str, i64)> = fixed
            .iter()
            .map(|(symbol, size)| (symbol.as_str(), *size))
            .collect();
        let session = core_ml_session(model_path, &fixed)?;
        let model = model_input(&session)?;
        let labels = labels(options.detector.labels.as_deref(), &session);
        let fitting = Fitting::new(model, rows.unwrap_or(max_batch))?;

        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batches of up to {max_batch}, {} labels, on Core ML",
            model.0,
            model.1,
            labels.len()
        );
        if labels.is_empty() {
            pp_warn!(
                pp_log: &pp_log,
                "the model names no classes and none were given: detections carry class numbers only"
            );
        }
        let interval = Interval::new(options.detector.interval);
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            session,
            options: options.detector,
            labels,
            interval,
            max_batch,
            rows,
            held: Vec::new(),
            context: None,
            fitting,
        })))
    }
}

/// What a model says of its batch.
enum Batch {
    /// Fixed at this many pictures.
    Fixed(usize),
    /// Left open, under this name.
    Named(String),
    /// Left open, with no name to fix it by.
    Unnamed,
}

/// A picture of the batch under way, and where it is looked at, the
/// picture to fit and how it is to be fitted — or `None` where it is let
/// by.
struct Held {
    buf: MediaBuffer,
    look: Option<(Picture, Letterbox)>,
}

/// How the whole of `picture` is fitted into input `slot` as Ultralytics
/// trains on — scaled to fit, proportions kept, grey around it.
fn whole(picture: &Picture, letterbox: &Letterbox, slot: usize) -> Cut {
    Cut {
        crop: (0, 0, picture.size.0, picture.size.1),
        offset: letterbox.offset,
        scaled: letterbox.scaled,
        slot,
    }
}

/// Fits the whole of `frame` into `fitting`'s first input, and says how it
/// was fitted.
#[cfg(test)]
fn fit_whole(
    fitting: &mut Fitting,
    frame: &ffmpeg::frame::Video,
) -> std::result::Result<Letterbox, OrtError> {
    let picture = Picture::of(frame)?;
    let letterbox = Letterbox::new(picture.size, fitting.model);
    fitting.fit(
        &picture,
        &[whole(&picture, &letterbox, 0)],
        ([1.0; 3], [0.0; 3]),
    )?;
    Ok(letterbox)
}

impl Detecting {
    /// `frame`, to be fitted, and how.
    fn look(
        &self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<(Picture, Letterbox), OrtError> {
        let picture = Picture::of(frame)?;
        let letterbox = Letterbox::new(picture.size, self.fitting.model);
        Ok((picture, letterbox))
    }

    /// How many of the held pictures are looked at.
    fn looked_at(&self) -> usize {
        self.held.iter().filter(|held| held.look.is_some()).count()
    }

    /// Fits `looks` into the first inputs in one pass, runs the model on
    /// them, and says what it found in each.
    fn detect(
        &mut self,
        looks: &[(&Picture, Letterbox)],
    ) -> std::result::Result<Vec<Vec<Detection>>, OrtError> {
        let cuts: Vec<[Cut; 1]> = looks
            .iter()
            .enumerate()
            .map(|(slot, (picture, letterbox))| [whole(picture, letterbox, slot)])
            .collect();
        let pictures: Vec<(&Picture, &[Cut])> = looks
            .iter()
            .zip(&cuts)
            .map(|((picture, _), cut)| (*picture, cut.as_slice()))
            .collect();
        self.fitting.fit_each(&pictures, ([1.0; 3], [0.0; 3]))?;
        // A model of fixed batch is handed exactly that many; one of open
        // batch, as many as there are.
        let rows = self.rows.unwrap_or(looks.len());
        let (width, height) = (self.fitting.model.0 as usize, self.fitting.model.1 as usize);
        let input =
            TensorRef::from_array_view(([rows, 3, height, width], self.fitting.input(rows)))?;
        let outputs = self.session.run(inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        // The inputs past the batch's pictures hold what they held; what
        // the model found in them is no picture's.
        let output = if output.ndim() > 0 && output.shape()[0] > looks.len() {
            output.slice_axis(Axis(0), Slice::from(0..looks.len()))
        } else {
            output
        };
        let letterboxes: Vec<Letterbox> = looks.iter().map(|(_, letterbox)| *letterbox).collect();
        decode_batch(output, &letterboxes, &self.options)
    }

    /// Runs the model on what is held, and hands every held picture on in
    /// the order it came — those looked at carrying what was found in them.
    /// Where the run fails, the pictures it was for go with it.
    fn run_held(&mut self, out: &mut Output) -> std::result::Result<(), OrtError> {
        let held = std::mem::take(&mut self.held);
        let looks: Vec<(&Picture, Letterbox)> = held
            .iter()
            .filter_map(|held| held.look.as_ref())
            .map(|(picture, letterbox)| (picture, *letterbox))
            .collect();
        let mut found = if looks.is_empty() {
            Vec::new()
        } else {
            self.detect(&looks)?
        }
        .into_iter();
        drop(looks);
        for Held { buf, look } in held {
            match look.and_then(|_| found.next()) {
                Some(items) => {
                    let detections =
                        Detections::new(Arc::clone(&self.name), Arc::clone(&self.labels), items);
                    out.push(detections.attach_to(buf));
                }
                // Let by unlooked-at, carrying nothing, which says so.
                None => out.push(buf),
            }
        }
        Ok(())
    }

    /// Puts `error` — one picture's, where the others of its batch go on in
    /// the same call — on the pipeline's bus.
    fn report(&self, error: OrtError) {
        match &self.context {
            Some(context) => context.bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::MetalOrtDetector,
                    name: Arc::clone(&self.name),
                    error: error.into(),
                },
            ),
            None => pp_error!(pp_log: &self.pp_log, "{error}"),
        }
    }
}

impl Element for Detecting {
    fn attach_context(&mut self, context: &Arc<Context>) {
        self.context = Some(Arc::clone(context));
    }

    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MetalOrtDetector
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Detecting {
    /// Decoded NV12 or BGRA video in VideoToolbox pixel buffers.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::NV12_OR_BGRA),
        )
    }

    /// A picture with no [`BatchSlot`] is a batch of its own, run at once.
    /// The pictures of a batch are held and run together when its last has
    /// come, or as soon as `max_batch` of them are looked at.
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "MetalOrtDetector",
                wanted: "VideoToolbox video frames",
                got: buf.kind(),
            }
            .into());
        };
        let slot = buf
            .metadata()
            .and_then(|metadata| metadata.get::<BatchSlot>())
            .copied();
        // What is held is of another batch, whose last picture never came:
        // it is run as it is rather than kept waiting.
        let held_batch = self.held.first().and_then(|held| {
            held.buf
                .metadata()
                .and_then(|metadata| metadata.get::<BatchSlot>())
                .map(|slot| slot.batch)
        });
        if !self.held.is_empty() && held_batch != slot.map(|slot| slot.batch) {
            self.run_held(out)?;
        }
        let last = slot.is_none_or(|slot| slot.is_last());

        let look = if self.interval.look(&buf) {
            match self.look(frame) {
                Ok(look) => Some(look),
                // This picture alone failed. Where nothing else is to go on
                // from this call, the failure is this call's; where the
                // batch it closes is, the batch goes on and the failure is
                // reported beside it.
                Err(error) if !(last && !self.held.is_empty()) => return Err(error.into()),
                Err(error) => {
                    self.report(error);
                    self.run_held(out)?;
                    return Ok(());
                }
            }
        } else {
            None
        };
        self.held.push(Held { buf, look });
        // A full batch is run at once: what it holds need not wait for the
        // rest of a mux's batch larger than it.
        if last || self.looked_at() == self.max_batch {
            self.run_held(out)?;
        }
        Ok(())
    }

    /// The end of the stream: a batch cut short is run as it is.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        Ok(self.run_held(out)?)
    }

    fn reset(&mut self) {
        self.held.clear();
        self.interval.restart();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::super::fitting::tests::{covered, mean_rgb, pattern, upload};
    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, SwOrtDetector};
    use crate::test_support::{nth_picture, try_videotoolbox_device};

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

    /// What the kernels are to write for `frame`: each pixel of the scaled
    /// picture the mean of the pixels it covers, made RGB by the frame's
    /// own rows, and the margins grey — three planes, R, G and B.
    fn expected(frame: &ffmpeg::frame::Video, model: (u32, u32)) -> Vec<f32> {
        let (width, height) = (frame.width(), frame.height());
        let letterbox = Letterbox::new((width, height), model);
        let plane = (model.0 * model.1) as usize;
        let mut tensor = vec![114.0 / 255.0; 3 * plane];
        for y in 0..letterbox.scaled.1 {
            for x in 0..letterbox.scaled.0 {
                let (x0, x1) = covered(x, width, letterbox.scaled.0);
                let (y0, y1) = covered(y, height, letterbox.scaled.1);
                let rgb = mean_rgb(frame, x0..x1, y0..y1);
                let at = ((letterbox.offset.1 + y) * model.0 + letterbox.offset.0 + x) as usize;
                for (channel, value) in rgb.into_iter().enumerate() {
                    tensor[channel * plane + at] = value;
                }
            }
        }
        tensor
    }

    /// The kernels write, from NV12 and from BGRA alike, the very input the
    /// CPU reads the picture as: the mean of the pixels each covers in the
    /// right channel and plane, made RGB by the picture's own colour, and
    /// grey around it. Needs no model; skipped without VideoToolbox.
    #[test]
    fn a_picture_is_fitted_as_the_cpu_reads_it() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let model = (64, 64);
        let mut fitting = Fitting::new(model, 1).expect("the kernels compile");
        for format in [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA] {
            // Twice as wide as high: grey above and below, and each input
            // pixel covering two by two.
            let picture = pattern(format, 128, 64);
            let wanted = expected(&picture, model);
            let MediaBuffer::Video(frame) = upload(&device, picture) else {
                panic!("a picture is uploaded");
            };
            let letterbox = fit_whole(&mut fitting, &frame).expect("fits");
            assert_eq!(letterbox, Letterbox::new((128, 64), model));
            for (index, (actual, wanted)) in fitting.input(1).iter().zip(&wanted).enumerate() {
                assert!(
                    (actual - wanted).abs() < 1e-3,
                    "{format:?}: float {index} is {actual}, not {wanted}"
                );
            }
        }
    }

    /// Pictures fitted together in one pass land each in its own input,
    /// as each fitted alone does. Needs no model; skipped without
    /// VideoToolbox.
    #[test]
    fn pictures_fitted_together_are_each_fitted_as_alone() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let model = (64, 64);
        let mut alone = Fitting::new(model, 1).expect("the kernels compile");
        let mut together = Fitting::new(model, 2).expect("the kernels compile");
        let frames: Vec<MediaBuffer> = [
            pattern(ffmpeg::format::Pixel::NV12, 128, 64),
            pattern(ffmpeg::format::Pixel::BGRA, 64, 128),
        ]
        .into_iter()
        .map(|picture| upload(&device, picture))
        .collect();
        let pictures: Vec<Picture> = frames
            .iter()
            .map(|buf| {
                let MediaBuffer::Video(frame) = buf else {
                    panic!("a picture is uploaded");
                };
                Picture::of(frame).expect("a VideoToolbox picture")
            })
            .collect();
        let cuts: Vec<[Cut; 1]> = pictures
            .iter()
            .enumerate()
            .map(|(slot, picture)| [whole(picture, &Letterbox::new(picture.size, model), slot)])
            .collect();
        let each: Vec<(&Picture, &[Cut])> = pictures
            .iter()
            .zip(&cuts)
            .map(|(picture, cut)| (picture, cut.as_slice()))
            .collect();
        together
            .fit_each(&each, ([1.0; 3], [0.0; 3]))
            .expect("fits");
        let floats = 3 * (model.0 * model.1) as usize;
        for (slot, buf) in frames.iter().enumerate() {
            let MediaBuffer::Video(frame) = buf else {
                unreachable!();
            };
            fit_whole(&mut alone, frame).expect("fits");
            assert_eq!(
                &together.input(2)[slot * floats..(slot + 1) * floats],
                alone.input(1),
                "input {slot}"
            );
        }
    }

    /// A picture in system memory is refused, saying what it is, rather
    /// than read as a pixel buffer it does not hold.
    #[test]
    fn a_picture_in_system_memory_is_refused() {
        if try_videotoolbox_device().is_none() {
            return;
        }
        let picture = pattern(ffmpeg::format::Pixel::NV12, 128, 64);
        assert!(matches!(
            Picture::of(&picture),
            Err(OrtError::UnsupportedPicture(ffmpeg::format::Pixel::NV12))
        ));
    }

    /// What `detector` found in `buf`, the picture it handed on checked to
    /// be the one it was given.
    fn found<T: RawSink + SrcPads>(detector: &mut T, buf: MediaBuffer) -> Detections {
        let kept = capture(detector);
        detector.consume(buf).expect("detects");
        let kept = kept.lock().unwrap();
        let MediaBuffer::Video(frame) = &kept[0] else {
            panic!("a picture goes on");
        };
        assert_eq!(frame.pts(), Some(150), "the picture it was given");
        kept[0]
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .expect("carries Detections")
            .clone()
    }

    fn iou(a: &crate::elements::Detection, b: &crate::elements::Detection) -> f32 {
        let w = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0);
        let h = ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
        let inter = w * h;
        inter / (a.width * a.height + b.width * b.height - inter)
    }

    /// The point of the element: through Core ML it finds what the CPU
    /// detector finds in the same picture — the same objects, in the same
    /// places — from NV12 and from BGRA alike. Needs a model and a video
    /// with something in its 150th picture; skipped, saying so, without
    /// them.
    #[test]
    fn it_finds_through_core_ml_what_the_cpu_finds() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut cpu =
            SwOrtDetector::new("cpu", &model, OrtDetectorOptions::default()).expect("loads");
        let expected = found(
            &mut cpu,
            MediaBuffer::video(nth_picture(&video, 150, ffmpeg::format::Pixel::NV12)),
        );
        assert!(
            expected.items.iter().any(|found| found.score > 0.5),
            "the picture has something in it"
        );

        let mut gpu = MetalOrtDetector::new("gpu", &model, MetalOrtDetectorOptions::default())
            .expect("loads on Core ML");
        for format in [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA] {
            let actual = found(&mut gpu, upload(&device, nth_picture(&video, 150, format)));
            for wanted in expected.items.iter().filter(|found| found.score > 0.5) {
                let best = actual
                    .items
                    .iter()
                    .filter(|found| found.class_id == wanted.class_id)
                    .map(|found| iou(found, wanted))
                    .fold(0.0, f32::max);
                assert!(
                    best > 0.8,
                    "{format:?}: {wanted:?} found at IoU {best} only"
                );
            }
        }
    }

    /// Picture `n` of `video`, in a VideoToolbox pixel buffer, as the
    /// `index`th of a batch of `size` a mux handed on.
    fn in_batch(
        device: &crate::elements::VideoToolboxDevice,
        video: &str,
        n: usize,
        index: usize,
        size: usize,
    ) -> MediaBuffer {
        use crate::buffer::Metadata;
        use crate::elements::vision::batch::{StreamId, StreamOrigin};
        upload(device, nth_picture(video, n, ffmpeg::format::Pixel::NV12)).with_metadata(
            Metadata::default()
                .with(StreamOrigin {
                    id: StreamId(index as u64),
                    name: Arc::from(format!("camera {index}")),
                    generation: 0,
                })
                .with(BatchSlot {
                    batch: 0,
                    index,
                    size,
                }),
        )
    }

    /// The confident detections on `buf`: class and corner.
    fn found_in(buf: &MediaBuffer) -> Vec<(usize, f32, f32)> {
        buf.metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .expect("carries Detections")
            .items
            .iter()
            .filter(|item| item.score > 0.5)
            .map(|item| (item.class_id, item.x, item.y))
            .collect()
    }

    /// Whether the model at `path` leaves its batch open.
    fn open_batch(path: &str) -> bool {
        let probe = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(path))
            .expect("loads");
        probe
            .inputs()
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .and_then(|shape| shape.iter().next().copied())
            .is_some_and(|batch| batch < 0)
    }

    /// The pictures of a batch are held until its last has come, then
    /// fitted in one pass and run through the model at once, and handed on
    /// in order, each carrying what one picture at a time finds in it; a
    /// batch cut short by the end of the stream is run as it is. Needs a
    /// model whose batch is left open — YOLO11n exported with
    /// `dynamic=True` — and a video; skipped, saying so, without them.
    #[test]
    fn a_batch_finds_what_one_picture_at_a_time_finds() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        if !open_batch(&model) {
            eprintln!("skipping: {model} takes one picture at a time");
            return;
        }
        let options = |max_batch| MetalOrtDetectorOptions {
            max_batch,
            ..MetalOrtDetectorOptions::default()
        };
        let pictures = [100, 150, 200];

        let mut alone = MetalOrtDetector::new("alone", &model, options(1)).expect("loads");
        let one_at_a_time = capture(&mut alone);
        for (index, n) in pictures.into_iter().enumerate() {
            alone
                .consume(in_batch(&device, &video, n, index, pictures.len()))
                .expect("detects");
            assert_eq!(
                one_at_a_time.lock().unwrap().len(),
                index + 1,
                "one at a time, each goes on at once"
            );
        }

        let mut batched = MetalOrtDetector::new("batched", &model, options(4)).expect("loads");
        let together = capture(&mut batched);
        for (index, n) in pictures.into_iter().enumerate() {
            batched
                .consume(in_batch(&device, &video, n, index, pictures.len()))
                .expect("detects");
            let last = index + 1 == pictures.len();
            assert_eq!(
                together.lock().unwrap().len(),
                if last { pictures.len() } else { 0 },
                "held to the last"
            );
        }
        let one_at_a_time = one_at_a_time.lock().unwrap();
        let together = together.lock().unwrap();
        for (index, (alone, batched)) in one_at_a_time.iter().zip(together.iter()).enumerate() {
            assert_eq!(
                batched
                    .metadata()
                    .and_then(|m| m.get::<BatchSlot>())
                    .map(|s| s.index),
                Some(index),
                "in the order they came"
            );
            let (alone, batched) = (found_in(alone), found_in(batched));
            assert!(!alone.is_empty(), "picture {index} has something in it");
            assert_eq!(alone.len(), batched.len(), "picture {index}");
            for (a, b) in alone.iter().zip(&batched) {
                assert_eq!(a.0, b.0, "picture {index}");
                assert!(
                    (a.1 - b.1).abs() < 0.01 && (a.2 - b.2).abs() < 0.01,
                    "picture {index}: {a:?} against {b:?}"
                );
            }
        }
        drop(together);

        // Two of a batch of three, then the end: both go on, looked at.
        let mut cut = MetalOrtDetector::new("cut short", &model, options(4)).expect("loads");
        let ended = capture(&mut cut);
        for (index, n) in pictures.into_iter().take(2).enumerate() {
            cut.consume(in_batch(&device, &video, n, index, 3))
                .expect("detects");
        }
        assert!(ended.lock().unwrap().is_empty(), "waiting for the third");
        cut.stream_event(&crate::stream::StreamEvent::Eos)
            .expect("drained");
        let ended = ended.lock().unwrap();
        assert_eq!(ended.len(), 2);
        assert!(ended.iter().all(|buf| !found_in(buf).is_empty()));
    }

    /// A model made for one picture at a time, given a `max_batch`, runs
    /// each picture of a batch as it comes rather than holding any, and
    /// finds what it finds alone. Needs such a model — the YOLOv10n
    /// release — and a video; skipped, saying so, without them.
    #[test]
    fn a_model_of_one_picture_runs_a_batch_one_by_one() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        if open_batch(&model) {
            eprintln!("skipping: {model} leaves its batch open");
            return;
        }
        let mut detector = MetalOrtDetector::new(
            "fixed",
            &model,
            MetalOrtDetectorOptions {
                max_batch: 4,
                ..MetalOrtDetectorOptions::default()
            },
        )
        .expect("loads");
        let kept = capture(&mut detector);
        for (index, n) in [100, 150, 200].into_iter().enumerate() {
            detector
                .consume(in_batch(&device, &video, n, index, 3))
                .expect("detects");
            assert_eq!(kept.lock().unwrap().len(), index + 1, "goes on at once");
        }
        assert!(
            kept.lock()
                .unwrap()
                .iter()
                .any(|buf| !found_in(buf).is_empty())
        );
    }
}
