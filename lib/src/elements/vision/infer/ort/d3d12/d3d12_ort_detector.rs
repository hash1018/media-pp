//! [`D3d12OrtDetector`]: object detection on D3D12 pictures, each handed on
//! with what was found in it, the picture fitted to the model on the GPU
//! and the model run by DirectML.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ort::{
    ep::directml::{DMLResource, resource_from_d3d},
    memory::MemoryInfo,
    session::Session,
    value::ValueType,
};
use windows::core::Interface;

use crate::ffmpeg;
use crate::orientation::Orientations;
use crate::platform::windows::d3d12_gpu::D3d12Gpu;
use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::{MediaBuffer, Metadata},
    bus::BusEvent,
    contract::{InputContract, MediaKind, MemoryDomain, PixelLayoutSet, PortContract},
    element::{Attached, Context, ElementType, Identity},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use crate::elements::{BatchSlot, Detection, Detections};

use super::super::device_tensor;
use super::super::{
    DetectorDecoder, Interval, Letterbox, ModelInput, ModelOutput, OrtDetectorOptions, OrtError,
    decode_batch, labels, looks, merge_looks, model_input,
};
use super::fitting::{Cut, Fitting, Picture};
use super::{DirectMl, directml_memory, host_memory, warn_past_directml_opset};

/// How a [`D3d12OrtDetector`] runs its model, beside what every detector is
/// told.
#[derive(Debug, Clone, PartialEq)]
pub struct D3d12OrtDetectorOptions {
    /// Thresholds and labels, as every detector takes them.
    pub detector: OrtDetectorOptions,
    /// The most pictures the model is run on at once: the pictures of a
    /// [`StreamMux`](crate::elements::StreamMux)'s batch — or a picture and
    /// its tiles — up to this many, are fitted in one submission and go
    /// through the model together. Give it the mux's `max_batch`, or a
    /// picture's tiles and one; a larger batch is run in parts this size.
    ///
    /// DirectML is fastest with every dimension fixed when the session is
    /// made. A model whose batch is left open by name is fixed at this for
    /// runs of more than one input, and at one for a run of a single input
    /// — a session made the first time one comes — so a picture with no
    /// tiles to look in is not run at the cost of a full batch; the height
    /// and width such a model leaves open are fixed at its input's. A model
    /// made for one picture at a time is run so, with a warning. The
    /// default, 1, is a detector of one picture at a time.
    pub max_batch: usize,
}

impl Default for D3d12OrtDetectorOptions {
    fn default() -> Self {
        Self {
            detector: OrtDetectorOptions::default(),
            max_batch: 1,
        }
    }
}

/// Finds objects in each D3D12 picture with an ONNX model run by DirectML,
/// on any vendor's GPU, and hands the picture on carrying what it found.
///
/// It takes NV12 or HDR P010 D3D12 pictures made on the [`D3d12Gpu`] it is
/// given, and hands each on unchanged with the [`Detections`] found in it,
/// as [`SwOrtDetector`](super::super::SwOrtDetector) does on the CPU.
/// Export the model at ONNX opset 20 or lower: an operator of a later opset
/// runs on the CPU, slower than the CPU detector, with a warning. Needs the
/// `ort-directml` feature, on Windows.
///
/// # Example
///
/// ```no_run
/// # fn main() -> media_pp::Result<()> {
/// use media_pp::elements::{
///     AppSink, D3d12Decoder, D3d12Gpu, D3d12OrtDetector, D3d12OrtDetectorOptions, FileDemuxer,
/// };
/// use media_pp::{ffmpeg::media, pipeline::Pipeline};
///
/// # let (source, _) = FileDemuxer::open("demux", "input.mp4")?;
/// # let video = source.best(media::Type::Video)?;
/// # let params = video.parameters.clone();
/// # let sink = AppSink::new("found", |_buf| Ok(()));
/// let gpu = D3d12Gpu::new()?;
/// let detector = D3d12OrtDetector::new(
///     "detector",
///     &gpu,
///     "yolov10n.onnx",
///     D3d12OrtDetectorOptions::default(),
/// )?;
/// let (pipeline, ()) = Pipeline::new("detect", source, |source, ctx| {
///     let decoder = D3d12Decoder::new("decoder", params, &gpu)?;
///     let branch = ctx.branch().pipe(decoder).queue("frames", 8).pipe(detector).to(sink)?;
///     ctx.attach(source, video.index, branch)?;
///     Ok(())
/// })?;
/// pipeline.run()?;
/// # Ok(())
/// # }
/// ```
///
/// # How it works
///
/// The picture is fitted to the model's input by a compute shader where it
/// is, and the model — YOLO, or the kind [`OrtDetectorOptions::model`] says
/// — is run by ONNX Runtime's DirectML provider on the same GPU: any
/// vendor's, through its own D3D12 driver.
///
/// It takes NV12 D3D12 pictures of any size — a `D3d12Decoder`'s, a
/// `D3d12Upload`'s, a `D3d12Scaler`'s — made on the same [`D3d12Gpu`] it is
/// given, and HDR P010, HLG or PQ, which it looks at through an SDR copy as
/// `CudaOrtDetector` does: a model was trained on SDR pictures, and what it
/// finds goes on the ten-bit picture itself. The boxes it finds are
/// fractions of each picture, as
/// [`Detection`](crate::elements::Detection) describes. The models it reads
/// are those [`OrtDetectorOptions`] describes, with two things DirectML
/// asks of them: an opset of 20 or lower, as an operator of a later opset
/// runs on the CPU, slower than the CPU alone; and, for its speed, the
/// model's own weights in half precision, as DirectML runs the precision
/// the model has — `docs/d3d12-inference.md` measures both.
///
/// # Where the model runs
///
/// DirectML runs on the GPU's queue, the one the fitting is submitted to,
/// so the model reads the fitted input after the fit by the queue's order
/// alone; the outputs are copied to the CPU and read there by the decoders
/// every detector shares. ONNX Runtime gives the CPU an operator DirectML
/// does not take.
///
/// # Requirements
///
/// The `ort-directml` feature, on Windows: ONNX Runtime's build with
/// DirectML is fetched as the CPU one is, and `DirectML.dll` is beside the
/// program.
pub struct D3d12OrtDetector(FilterStage<Detecting>);

filter_stage!(D3d12OrtDetector);

/// What a [`D3d12OrtDetector`] does with each picture. The sessions go
/// before the fitting, whose buffer they read, and the input's allocation
/// before both.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    /// The fitting's buffer as DirectML reads it.
    input: DMLResource,
    /// The model's session, or its sessions, one for each batch size.
    runs: Runs,
    options: OrtDetectorOptions,
    labels: Arc<[Arc<str>]>,
    /// The values the model wants.
    values: ModelInput,
    /// What reads its output.
    decoder: Arc<dyn DetectorDecoder>,
    /// Which pictures it looks at.
    interval: Interval,
    /// How many pictures the model is run on at once.
    max_batch: usize,
    /// The pictures of the batch under way, in the order they came.
    held: Vec<Held>,
    /// The pipeline's, to report a picture's failure on where others must
    /// go on in the same call.
    context: Option<Arc<Context>>,
    fitting: Fitting,
    /// How each picture is turned to be shown.
    orientations: Orientations,
    /// Where the input is, and where the outputs are copied.
    on_gpu: MemoryInfo<'static>,
    on_host: MemoryInfo<'static>,
}

// SAFETY: the sessions, the buffer and its allocation are used by the one
// thread transforming at a time, through `&mut self`; the D3D12 and
// DirectML objects are free-threaded.
unsafe impl Send for Detecting {}

impl D3d12OrtDetector {
    /// Loads the model at `model_path` to run through DirectML on `gpu` —
    /// the same [`D3d12Gpu`] every other D3D12 element of the pipeline was
    /// made with, whose pictures it reads.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        model_path: impl AsRef<Path>,
        options: D3d12OrtDetectorOptions,
    ) -> Result<Self> {
        let identity = Identity::new(ElementType::D3d12OrtDetector, name);
        let name = identity.name();
        let pp_log = identity.pp_log().clone();
        let path = model_path.as_ref().display().to_string();
        if options.max_batch == 0 {
            return Err(OrtError::ZeroMaxBatch.into());
        }
        if let Some(tiles) = &options.detector.tiles {
            tiles.check()?;
        }
        warn_past_directml_opset(model_path.as_ref(), &pp_log);

        // Read on the CPU first: whether the model leaves its batch open,
        // and by what name, to fix it for DirectML — and its height and
        // width, which an Ultralytics export with `dynamic=True` leaves
        // open too.
        let open = Session::builder()
            .map_err(OrtError::from)?
            .commit_from_file(model_path.as_ref())
            .map_err(OrtError::from)?;
        let (width, height) = model_input(&open)?;
        let (batch, sides) = match open.inputs().first().map(|input| input.dtype()) {
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
                     DirectML runs it at each number of pictures as it comes"
                );
                (options.max_batch, None)
            }
        };
        let remake = Remake {
            path: model_path.as_ref().to_path_buf(),
            sides,
            batch: match &batch {
                Batch::Named(symbol) => Some(symbol.clone()),
                _ => None,
            },
            directml: DirectMl::new(gpu)?,
        };
        // The largest first: it reads the model's input and labels, and a
        // batch of every size fits it.
        let session = remake.session(max_batch)?;
        let model = model_input(&session)?;
        let labels = options
            .detector
            .model
            .labels(labels(options.detector.labels.as_deref(), &session));
        let fitting = Fitting::new(gpu, model, rows.unwrap_or(max_batch))?;
        // SAFETY: the fitting's live buffer, on the device DirectML was made
        // on; the allocation is dropped before the fitting.
        let input = unsafe { resource_from_d3d(fitting.tensor().as_raw().cast()) }
            .map_err(OrtError::from)?;
        let runs = match batch {
            Batch::Named(_) => Runs::Open {
                batch: session,
                max_batch,
                alone: None,
                remake,
            },
            _ => Runs::One { session, rows },
        };

        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batches of up to {max_batch}, {} labels, on DirectML",
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
        Ok(Self(FilterStage::new(
            identity,
            Detecting {
                name,
                pp_log,
                input,
                runs,
                values: options.detector.model.input(),
                decoder: options.detector.model.decoder(model),
                options: options.detector,
                labels,
                interval,
                max_batch,
                held: Vec::new(),
                context: None,
                fitting,
                orientations: Orientations::default(),
                on_gpu: directml_memory()?,
                on_host: host_memory()?,
            },
        )))
    }
}

/// The sessions a model is run through, and how many inputs each is
/// handed.
enum Runs {
    /// Every run through one session, handed `rows` inputs — the batch the
    /// model fixes itself — or, where `None`, as many as there are: a batch
    /// left open with no name to fix it by.
    One {
        session: Session,
        rows: Option<usize>,
    },
    /// A batch left open by name, fixed for DirectML: at `max_batch` for
    /// runs of more than one input, and at one for a single input, made the
    /// first time one comes.
    Open {
        batch: Session,
        max_batch: usize,
        alone: Option<Session>,
        remake: Remake,
    },
}

impl Runs {
    /// The session for a run of `n` inputs, and how many it is handed: `n`
    /// or more, the inputs past the run's left as they were.
    fn for_run(
        &mut self,
        n: usize,
        pp_log: &PpLog,
    ) -> std::result::Result<(&mut Session, usize), OrtError> {
        match self {
            Runs::One { session, rows } => Ok((session, rows.unwrap_or(n))),
            Runs::Open {
                batch, max_batch, ..
            } if n > 1 || *max_batch == 1 => Ok((batch, *max_batch)),
            Runs::Open { alone, remake, .. } => {
                let session = match alone {
                    Some(session) => session,
                    empty => {
                        let session = remake.session(1)?;
                        pp_info!(pp_log: pp_log, "a session for one picture at a time made");
                        empty.insert(session)
                    }
                };
                Ok((session, 1))
            }
        }
    }
}

/// How a session on the model is made, at a batch of any size.
struct Remake {
    path: PathBuf,
    /// The sides the model leaves open by name, fixed at its input's.
    sides: Vec<(String, i64)>,
    /// The batch's name, where the model leaves it open by one.
    batch: Option<String>,
    directml: DirectMl,
}

impl Remake {
    /// A session handed `size` inputs a run, where the batch is open by
    /// name; otherwise as the model has it.
    fn session(&self, size: usize) -> std::result::Result<Session, OrtError> {
        let mut fixed: Vec<(&str, i64)> = self
            .sides
            .iter()
            .map(|(symbol, side)| (symbol.as_str(), *side))
            .collect();
        if let Some(batch) = &self.batch {
            fixed.push((batch, size as i64));
        }
        self.directml.session(&self.path, &fixed)
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
/// picture to fit and each way it is fitted — whole, then each tile — or
/// `None` where it is let by.
struct Held {
    buf: MediaBuffer,
    look: Option<(Picture, Vec<Letterbox>)>,
}

/// How what `letterbox` says of a picture — the whole of it or a tile — is
/// fitted into input `slot`.
fn cut(letterbox: &Letterbox, slot: usize) -> Cut {
    Cut {
        crop: letterbox.crop,
        orientation: letterbox.orientation,
        offset: letterbox.offset,
        scaled: letterbox.scaled,
        slot,
    }
}

impl Detecting {
    /// `frame`, to be fitted, and how.
    fn look(
        &mut self,
        frame: &ffmpeg::frame::Video,
        metadata: Option<&Metadata>,
    ) -> std::result::Result<(Picture, Vec<Letterbox>), OrtError> {
        let picture = self.fitting.picture(frame)?;
        let orientation = self.orientations.of(frame, &self.pp_log);
        let looks = looks(
            picture.size,
            self.fitting.model,
            orientation,
            self.options.tiles.as_ref(),
            metadata,
        );
        Ok((picture, looks))
    }

    /// How many inputs the held pictures are looked at in.
    fn looked_at(&self) -> usize {
        self.held
            .iter()
            .filter_map(|held| held.look.as_ref())
            .map(|(_, looks)| looks.len())
            .sum()
    }

    /// Fits `looks` into the first inputs in one submission, runs the model
    /// on them, and says what it found in each.
    fn detect(
        &mut self,
        looks: &[(&Picture, Letterbox)],
    ) -> std::result::Result<Vec<Vec<Detection>>, OrtError> {
        let cuts: Vec<[Cut; 1]> = looks
            .iter()
            .enumerate()
            .map(|(slot, (_, letterbox))| [cut(letterbox, slot)])
            .collect();
        // One picture, its whole and its tiles, is fitted once for all its
        // looks.
        let mut pictures: Vec<(&Picture, Vec<Cut>)> = Vec::new();
        for ((picture, _), [cut]) in looks.iter().zip(&cuts) {
            match pictures.last_mut() {
                Some((last, cuts)) if std::ptr::eq(*last, *picture) => cuts.push(*cut),
                _ => pictures.push((picture, vec![*cut])),
            }
        }
        let pictures: Vec<(&Picture, &[Cut])> = pictures
            .iter()
            .map(|(picture, cuts)| (*picture, cuts.as_slice()))
            .collect();
        self.fitting.fit_each(&pictures, self.values)?;
        let (session, rows) = self.runs.for_run(looks.len(), &self.pp_log)?;
        let (width, height) = (self.fitting.model.0 as i64, self.fitting.model.1 as i64);
        debug_assert!(rows <= self.fitting.capacity());
        // SAFETY: the allocation is the fitting's buffer of `capacity`
        // inputs, of which the shape takes the first `rows`; it outlives
        // the run, the tensor being dropped before this returns.
        let input =
            unsafe { device_tensor(&self.on_gpu, *self.input, &[rows as i64, 3, height, width])? };
        let names: Vec<String> = session
            .outputs()
            .iter()
            .map(|output| output.name().to_owned())
            .collect();
        let mut binding = session.create_binding()?;
        binding.bind_input(session.inputs()[0].name().to_owned(), &input)?;
        for name in &names {
            binding.bind_output_to_device(name, &self.on_host)?;
        }
        let outputs = session.run_binding(&binding)?;
        let letterboxes: Vec<Letterbox> = looks.iter().map(|(_, letterbox)| *letterbox).collect();
        // Every output read as it is; past the batch's pictures, no
        // picture's — the decoders read as many as there are letterboxes.
        let tensors = names
            .iter()
            .map(|name| {
                let (shape, data) = outputs[name.as_str()].try_extract_tensor::<f32>()?;
                let shape: Vec<usize> = shape.iter().map(|&side| side.max(0) as usize).collect();
                Ok((name.as_str(), shape, data))
            })
            .collect::<std::result::Result<Vec<_>, OrtError>>()?;
        let outputs: Vec<ModelOutput<'_>> = tensors
            .iter()
            .map(|(name, shape, data)| ModelOutput { name, shape, data })
            .collect();
        decode_batch(
            &*self.decoder,
            &outputs,
            &letterboxes,
            self.fitting.model,
            &self.options,
        )
    }

    /// Runs the model on what is held, and hands every held picture on in
    /// the order it came — those looked at carrying what was found in them.
    /// Where the run fails, the pictures it was for go with it.
    fn run_held(&mut self, out: &mut Output) -> std::result::Result<(), OrtError> {
        let held = std::mem::take(&mut self.held);
        let looks: Vec<(&Picture, Letterbox)> = held
            .iter()
            .filter_map(|held| held.look.as_ref())
            .flat_map(|(picture, looks)| looks.iter().map(move |look| (picture, *look)))
            .collect();
        // `max_batch` inputs a run, a picture's tiles beside it.
        let mut found = Vec::with_capacity(looks.len());
        for run in looks.chunks(self.max_batch) {
            found.extend(self.detect(run)?);
        }
        let mut found = found.into_iter();
        drop(looks);
        for Held { buf, look } in held {
            match look {
                Some((_, looks)) => {
                    let each: Vec<Vec<Detection>> = found.by_ref().take(looks.len()).collect();
                    let items = merge_looks(each, self.options.iou_threshold);
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
                    element_type: ElementType::D3d12OrtDetector,
                    name: Arc::clone(&self.name),
                    error: error.into(),
                },
            ),
            None => pp_error!(pp_log: &self.pp_log, "{error}"),
        }
    }
}

impl Filter for Detecting {
    /// Decoded NV12, or HDR P010, video in D3D12 textures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::YUV420),
        )
    }

    /// A picture with no [`BatchSlot`] is a batch of its own, run at once.
    /// The pictures of a batch are held and run together when its last has
    /// come, or as soon as `max_batch` of them are looked at.
    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        if !matches!(&buf, MediaBuffer::Video(_)) {
            return Err(OrtError::UnsupportedBuffer {
                detector: "D3d12OrtDetector",
                wanted: "D3D12 video frames",
                got: buf.kind(),
            }
            .into());
        }
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
            let MediaBuffer::Video(frame) = &buf else {
                unreachable!("checked above");
            };
            match self.look(frame, buf.metadata()) {
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
        if last || self.looked_at() >= self.max_batch {
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

    fn attached(&mut self, cx: &Attached<'_>) {
        self.pp_log = cx.pp_log().clone();
        self.context = Some(Arc::clone(cx.context()));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, D3d12Upload, SwOrtDetector, Tiles};
    use crate::test_support::{nth_picture, pictures_with_objects, try_d3d12_gpu};

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

    /// What `detector` found in `buf`, the picture it handed on checked to
    /// be the one it was given.
    fn found<T: RawSink + SrcPads>(detector: &mut T, buf: MediaBuffer) -> Detections {
        let given = match &buf {
            MediaBuffer::Video(frame) => frame.pts(),
            _ => None,
        };
        let kept = capture(detector);
        detector.consume(buf).expect("detects");
        let kept = kept.lock().unwrap();
        let MediaBuffer::Video(frame) = &kept[0] else {
            panic!("a picture goes on");
        };
        assert_eq!(frame.pts(), given, "the picture it was given");
        kept[0]
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .expect("carries Detections")
            .clone()
    }

    fn iou(a: &Detection, b: &Detection) -> f32 {
        let w = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0);
        let h = ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
        let inter = w * h;
        inter / (a.width * a.height + b.width * b.height - inter)
    }

    /// `frame`, in system memory, as a D3D12 picture on `gpu`.
    fn on_gpu(gpu: &D3d12Gpu, frame: ffmpeg::frame::Video) -> Option<MediaBuffer> {
        let Ok(mut upload) = D3d12Upload::new("upload", gpu) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return None;
        };
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("uploads");
        let buf = uploaded.lock().unwrap().remove(0);
        Some(buf)
    }

    /// The point of the element: through DirectML it finds what the CPU
    /// detector finds in the same picture — the same objects, in the same
    /// places. Needs a model and a video, as the CUDA detector's test does,
    /// and a D3D12 device; skipped, saying so, without them.
    #[test]
    fn it_finds_on_the_gpu_what_the_cpu_finds() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(&n) = pictures_with_objects(&model, &video, 1).first() else {
            eprintln!("skipping: nothing in {video} that the CPU is sure of");
            return;
        };
        let options = D3d12OrtDetectorOptions::default();
        let picture = || nth_picture(&video, n, ffmpeg::format::Pixel::NV12);
        let mut cpu = SwOrtDetector::new("cpu", &model, options.detector.clone()).expect("loads");
        let expected = found(&mut cpu, MediaBuffer::video(picture()));
        assert!(
            !expected.items.is_empty(),
            "picture {n} has something in it"
        );

        let Some(buf) = on_gpu(&gpu, picture()) else {
            return;
        };
        let mut detector =
            D3d12OrtDetector::new("gpu", &gpu, &model, options).expect("loads on DirectML");
        let actual = found(&mut detector, buf);
        for wanted in expected.items.iter().filter(|found| found.score > 0.5) {
            let best = actual
                .items
                .iter()
                .filter(|found| found.class_id == wanted.class_id)
                .map(|found| iou(found, wanted))
                .fold(0.0, f32::max);
            assert!(best > 0.8, "{wanted:?} found at IoU {best} only");
        }
    }

    /// Looking in tiles as well as whole, the GPU finds what the CPU finds,
    /// and never fewer than the whole picture alone. Needs a model and a
    /// video, as above.
    #[test]
    fn tiles_find_on_the_gpu_what_they_find_on_the_cpu() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let Some(&n) = pictures_with_objects(&model, &video, 1).first() else {
            eprintln!("skipping: nothing in {video} that the CPU is sure of");
            return;
        };
        let tiled = OrtDetectorOptions {
            tiles: Some(Tiles::default()),
            ..OrtDetectorOptions::default()
        };
        let picture = || nth_picture(&video, n, ffmpeg::format::Pixel::NV12);
        let mut whole =
            SwOrtDetector::new("whole", &model, OrtDetectorOptions::default()).expect("loads");
        let mut cpu = SwOrtDetector::new("cpu", &model, tiled.clone()).expect("loads");
        let alone = found(&mut whole, MediaBuffer::video(picture()));
        let expected = found(&mut cpu, MediaBuffer::video(picture()));

        let Some(buf) = on_gpu(&gpu, picture()) else {
            return;
        };
        let mut detector = D3d12OrtDetector::new(
            "gpu",
            &gpu,
            &model,
            D3d12OrtDetectorOptions {
                detector: tiled,
                max_batch: 7,
            },
        )
        .expect("loads on DirectML");
        let actual = found(&mut detector, buf);
        assert!(
            actual.items.len() >= alone.items.len(),
            "tiles found {} where the whole picture found {}",
            actual.items.len(),
            alone.items.len()
        );
        // The same objects as the CPU's: a small one the whole picture finds
        // near the threshold is boxed by the whole picture on one and by a
        // tile on the other, the two fitting pictures by other filters.
        for wanted in expected.items.iter().filter(|found| found.score > 0.5) {
            let best = actual
                .items
                .iter()
                .filter(|found| found.class_id == wanted.class_id)
                .map(|found| iou(found, wanted))
                .fold(0.0, f32::max);
            assert!(best > 0.4, "{wanted:?} found at IoU {best} only");
        }
    }

    /// `buf` as the `index`th of a batch of `size` a mux handed on.
    fn in_batch(buf: MediaBuffer, index: usize, size: usize) -> MediaBuffer {
        use crate::buffer::Metadata;
        use crate::elements::vision::batch::{StreamId, StreamOrigin};
        buf.with_metadata(
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

    /// The pictures of a batch are held until its last has come, then run
    /// through the model at once and handed on in order, each carrying what
    /// one picture at a time finds in it; a batch cut short by the end of
    /// the stream is run as it is. A model fixed to one picture runs each
    /// as it comes, finding the same. Needs a model and a video, as above.
    #[test]
    fn a_batch_finds_what_one_picture_at_a_time_finds() {
        let (Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_YOLO"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO and MEDIA_PP_TEST_VIDEO to run this");
            return;
        };
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        // Whether the model takes a batch: one fixed to a picture at a time
        // has nothing to hold one for.
        let holds = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(&model))
            .expect("loads")
            .inputs()
            .first()
            .and_then(|input| input.dtype().tensor_shape())
            .and_then(|shape| shape.iter().next().copied())
            .is_some_and(|batch| batch <= 0);
        let pictures = pictures_with_objects(&model, &video, 3);
        if pictures.len() < 3 {
            eprintln!("skipping: fewer than three pictures in {video} with something on them");
            return;
        }
        let options = |max_batch| D3d12OrtDetectorOptions {
            max_batch,
            ..D3d12OrtDetectorOptions::default()
        };
        let mut uploaded = Vec::new();
        for (index, n) in pictures.iter().copied().enumerate() {
            let Some(buf) = on_gpu(&gpu, nth_picture(&video, n, ffmpeg::format::Pixel::NV12))
            else {
                return;
            };
            uploaded.push(in_batch(buf, index, pictures.len()));
        }
        let found_in = |buf: &MediaBuffer| -> Vec<(usize, f32, f32)> {
            buf.metadata()
                .and_then(|metadata| metadata.get::<Detections>())
                .expect("carries Detections")
                .items
                .iter()
                .filter(|item| item.score > 0.5)
                .map(|item| (item.class_id, item.x, item.y))
                .collect()
        };

        let mut alone =
            D3d12OrtDetector::new("alone", &gpu, &model, options(1)).expect("loads on DirectML");
        let one_at_a_time = capture(&mut alone);
        for (index, buf) in uploaded.iter().enumerate() {
            alone.consume(buf.clone()).expect("detects");
            assert_eq!(
                one_at_a_time.lock().unwrap().len(),
                index + 1,
                "one at a time, each goes on at once"
            );
        }

        let mut batched =
            D3d12OrtDetector::new("batched", &gpu, &model, options(4)).expect("loads on DirectML");
        let together = capture(&mut batched);
        for (index, buf) in uploaded.iter().enumerate() {
            batched.consume(buf.clone()).expect("detects");
            let last = index + 1 == uploaded.len();
            let expected = match (holds, last) {
                (false, _) => index + 1,
                (true, true) => uploaded.len(),
                (true, false) => 0,
            };
            assert_eq!(
                together.lock().unwrap().len(),
                expected,
                "held to the last where the model takes a batch"
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

        if !holds {
            return;
        }
        // Two of a batch of three, then the end: both go on, looked at.
        let mut cut = D3d12OrtDetector::new("cut short", &gpu, &model, options(4)).expect("loads");
        let ended = capture(&mut cut);
        for buf in uploaded.iter().take(2) {
            let index = buf
                .metadata()
                .and_then(|m| m.get::<BatchSlot>())
                .unwrap()
                .index;
            cut.consume(in_batch(buf.clone(), index, 3))
                .expect("detects");
        }
        assert!(ended.lock().unwrap().is_empty(), "waiting for the third");
        cut.stream_event(&crate::stream::StreamEvent::Eos)
            .expect("drained");
        let ended = ended.lock().unwrap();
        assert_eq!(ended.len(), 2);
        assert!(ended.iter().all(|buf| !found_in(buf).is_empty()));
    }

    /// A picture in system memory is refused as not one this reads, the
    /// detector left to read the next; and no batch is no detector. Needs
    /// a model, but no video.
    #[test]
    fn what_it_cannot_read_is_refused_as_that() {
        let Ok(model) = std::env::var("MEDIA_PP_TEST_YOLO") else {
            eprintln!("skipping: set MEDIA_PP_TEST_YOLO to run this");
            return;
        };
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let none = D3d12OrtDetectorOptions {
            max_batch: 0,
            ..D3d12OrtDetectorOptions::default()
        };
        assert!(matches!(
            D3d12OrtDetector::new("none", &gpu, &model, none).err(),
            Some(crate::error::Error::OrtDetectorError(
                OrtError::ZeroMaxBatch
            ))
        ));
        let mut detector =
            D3d12OrtDetector::new("gpu", &gpu, &model, D3d12OrtDetectorOptions::default())
                .expect("loads on DirectML");
        let in_memory = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 64);
        let error = detector
            .consume(MediaBuffer::video(in_memory))
            .expect_err("a picture in system memory");
        assert!(
            matches!(
                error,
                crate::error::Error::OrtDetectorError(OrtError::UnsupportedD3d12Picture(_))
            ),
            "{error:?}"
        );
        let Some((_, uploaded)) = super::super::fitting::tests::upload(&gpu, (64, 64)) else {
            return;
        };
        let kept = capture(&mut detector);
        detector.consume(uploaded).expect("reads the next");
        assert_eq!(kept.lock().unwrap().len(), 1, "and hands it on");

        // A ten-bit picture is read through its SDR copy where it is HLG or
        // PQ, and refused where it has no tone map.
        for (transfer, readable) in [
            (ffmpeg::color::TransferCharacteristic::ARIB_STD_B67, true),
            (ffmpeg::color::TransferCharacteristic::BT709, false),
        ] {
            let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::P010LE, 64, 64);
            for plane in 0..2 {
                frame.data_mut(plane).fill(0x80);
            }
            crate::color::ColorDescription {
                space: ffmpeg::color::Space::BT2020NCL,
                range: ffmpeg::color::Range::MPEG,
                primaries: ffmpeg::color::Primaries::BT2020,
                transfer,
            }
            .describe(&mut frame);
            let Some(on_gpu) = super::super::fitting::tests::upload_p010(&gpu, &frame) else {
                return;
            };
            let result = detector.consume(MediaBuffer::video(on_gpu));
            if readable {
                result.expect("reads an HLG picture");
                let kept = kept.lock().unwrap();
                let MediaBuffer::Video(handed) = kept.last().expect("handed on") else {
                    panic!("a picture goes on");
                };
                assert_eq!(
                    handed.format(),
                    ffmpeg::format::Pixel::D3D12,
                    "the picture itself"
                );
            } else {
                assert!(
                    matches!(
                        result,
                        Err(crate::error::Error::OrtDetectorError(
                            OrtError::UnsupportedD3d12Picture(_)
                        ))
                    ),
                    "{result:?}"
                );
            }
        }
    }
}
