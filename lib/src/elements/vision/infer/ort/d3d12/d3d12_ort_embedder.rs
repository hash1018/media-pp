//! [`D3d12OrtEmbedder`]: what a detector found, made into vectors by an
//! embedding model on D3D12 pictures, each object read from the picture on
//! the GPU and the model run by DirectML.

use std::{path::Path, sync::Arc};

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
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{
        InputContract, MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract,
    },
    element::{Element, ElementType, element_pp_log},
    elements::{Detections, Embedding},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::OrtError;
use super::super::classify::{Input, Memory, Plan, apply_each, image_input};
use super::super::device_tensor;
use super::super::embed::{OrtEmbedderOptions, Warp, cutouts, embeddings, with_cutouts};
use super::fitting::Fitting;
use super::{DirectMl, directml_memory, host_memory};

/// How many objects a model that takes any number at once is fixed to take
/// for DirectML, which is fastest with every dimension fixed: a picture's
/// objects go in eights, the inputs past them left as they were.
const DIRECTML_BATCH: usize = 8;

/// Makes each object a detector found on each D3D12 picture into a vector
/// with an embedding model — what
/// [`SwOrtEmbedder`](crate::elements::SwOrtEmbedder) does on the CPU, with
/// each object read into the model's input by a compute shader where the
/// picture is, through the map [`OrtEmbedderOptions::align`] makes for it —
/// a face straightened by its five points, a box stretched — between the
/// four pixels around each point, several objects to one run, and the model
/// run by ONNX Runtime's DirectML provider on the same GPU.
///
/// It goes after a detector — one that finds five points for a face, such
/// as a [`D3d12OrtDetector`](crate::elements::D3d12OrtDetector) of
/// [`DetectorModel::RetinaFace`](crate::elements::DetectorModel::RetinaFace),
/// for [`Alignment::FivePoints`](crate::elements::Alignment::FivePoints) —
/// and best after an [`ObjectTracker`](crate::elements::ObjectTracker),
/// which lets a followed object be embedded once and its vector kept, as
/// [`OrtEmbedderOptions::reembed`] says. It takes NV12 D3D12 pictures made
/// on the [`D3d12Gpu`] it is given. The model has one input, `[batch, 3,
/// height, width]` — 112 for a side it leaves open — and its first output
/// is a vector for each, made of length 1 here whatever the model makes.
/// A model that takes any number of objects at once is fixed to take eight
/// at a time, as DirectML is fastest with every dimension fixed; the models
/// DirectML runs best are those the detector's documentation describes.
///
/// Where [`OrtEmbedderOptions::cutouts`] asks for them, the inputs a run
/// read are copied back to the CPU to make them, which costs a wait for the
/// GPU each run.
///
/// # Requirements
///
/// The `ort-directml` feature, on Windows.
pub struct D3d12OrtEmbedder(FilterStage<Embedder>);

filter_stage!(D3d12OrtEmbedder);

/// What a [`D3d12OrtEmbedder`] does with each picture. The session goes
/// before the fitting, whose buffer it reads, and the input's allocation
/// before both.
struct Embedder {
    name: Arc<str>,
    pp_log: PpLog,
    /// The fitting's buffer as DirectML reads it.
    allocation: DMLResource,
    session: Session,
    options: OrtEmbedderOptions,
    input: Input,
    memory: Memory<Embedding>,
    fitting: Fitting,
    /// How each picture is turned to be shown.
    orientations: Orientations,
    /// Where the input is, and where the output is copied.
    on_gpu: MemoryInfo<'static>,
    on_host: MemoryInfo<'static>,
}

// SAFETY: the session, the buffer and its allocation are used by the one
// thread transforming at a time, through `&mut self`; the D3D12 and
// DirectML objects are free-threaded.
unsafe impl Send for Embedder {}

impl D3d12OrtEmbedder {
    /// Loads the embedding model at `model_path` to run through DirectML on
    /// `gpu` — the same [`D3d12Gpu`] every other D3D12 element of the
    /// pipeline was made with, whose pictures it reads.
    pub fn new(
        name: impl Into<String>,
        gpu: &D3d12Gpu,
        model_path: impl AsRef<Path>,
        options: OrtEmbedderOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d12OrtEmbedder, &name, None);
        let path = model_path.as_ref().display().to_string();

        // Read on the CPU first: its input, and what it calls a batch or a
        // side it leaves open, to fix them for DirectML.
        let probe = Session::builder()
            .map_err(OrtError::from)?
            .commit_from_file(model_path.as_ref())
            .map_err(OrtError::from)?;
        let open = image_input(&probe, 112)?;
        let mut fixed: Vec<(String, i64)> = Vec::new();
        if let Some(ValueType::Tensor {
            shape,
            dimension_symbols,
            ..
        }) = probe.inputs().first().map(|input| input.dtype())
        {
            let named = |axis: usize| {
                dimension_symbols
                    .get(axis)
                    .filter(|symbol| !symbol.is_empty() && shape[axis] <= 0)
                    .cloned()
            };
            for (axis, size) in [(2, open.size.1), (3, open.size.0)] {
                if let Some(symbol) = named(axis) {
                    fixed.push((symbol, i64::from(size)));
                }
            }
            match (open.batch, named(0)) {
                (None, Some(symbol)) => fixed.push((symbol, DIRECTML_BATCH as i64)),
                (None, None) => pp_warn!(
                    pp_log: &pp_log,
                    "the model leaves its batch open without a name to fix it by: \
                     DirectML runs it at each number of objects as it comes"
                ),
                (Some(_), _) => {}
            }
        }
        drop(probe);
        let fixed: Vec<(&str, i64)> = fixed
            .iter()
            .map(|(symbol, size)| (symbol.as_str(), *size))
            .collect();
        let session = DirectMl::new(gpu)?.session(model_path.as_ref(), &fixed)?;
        let input = image_input(&session, 112)?;
        let fitting = Fitting::new(gpu, input.size, input.batch.unwrap_or(DIRECTML_BATCH))?;
        // SAFETY: the fitting's live buffer, on the device DirectML was made
        // on; the allocation is dropped before the fitting.
        let allocation = unsafe { resource_from_d3d(fitting.tensor().as_raw().cast()) }
            .map_err(OrtError::from)?;
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batch={:?}, {:?}, {:?}, on DirectML",
            input.size.0,
            input.size.1,
            input.batch,
            options.input,
            options.align
        );
        Ok(Self(FilterStage::new(Embedder {
            name,
            pp_log,
            allocation,
            session,
            options,
            input,
            memory: Memory::default(),
            fitting,
            orientations: Orientations::default(),
            on_gpu: directml_memory()?,
            on_host: host_memory()?,
        })))
    }
}

impl Embedder {
    /// The model's vector for each of `warps` of `frame`, in order.
    fn embed(
        &mut self,
        frame: &ffmpeg::frame::Video,
        warps: &[Warp],
    ) -> std::result::Result<Vec<Option<Embedding>>, OrtError> {
        let picture = self.fitting.picture(frame)?;
        let (width, height) = self.input.size;
        let mut found = Vec::with_capacity(warps.len());
        for group in warps.chunks(self.fitting.capacity()) {
            let maps: Vec<[f32; 6]> = group.iter().map(|warp| warp.0).collect();
            self.fitting.warp(&picture, &maps, self.options.input)?;
            // A model of fixed batch is handed exactly that many; one of
            // open batch, as many as there are.
            let rows = self.input.batch.unwrap_or(group.len());
            let cut = if self.options.cutouts {
                cutouts(
                    &self.fitting.read_back(group.len())?,
                    group.len(),
                    (width, height),
                    self.options.input,
                )
            } else {
                Vec::new()
            };
            // SAFETY: the allocation is the fitting's buffer of `capacity`
            // inputs, of which the shape takes the first `rows`; it outlives
            // the run, the tensor being dropped before this returns.
            let tensor = unsafe {
                device_tensor(
                    &self.on_gpu,
                    *self.allocation,
                    &[rows as i64, 3, i64::from(height), i64::from(width)],
                )?
            };
            let mut binding = self.session.create_binding()?;
            binding.bind_input(self.session.inputs()[0].name().to_owned(), &tensor)?;
            let output = self.session.outputs()[0].name().to_owned();
            binding.bind_output_to_device(&output, &self.on_host)?;
            let outputs = self.session.run_binding(&binding)?;
            let (shape, data) = outputs[output.as_str()].try_extract_tensor::<f32>()?;
            let shape: Vec<usize> = shape.iter().map(|&side| side.max(0) as usize).collect();
            found.extend(with_cutouts(
                embeddings(&self.name, &shape, data, group.len())?,
                cut,
            ));
        }
        Ok(found)
    }
}

impl Element for Embedder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d12OrtEmbedder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Embedder {
    /// Decoded NV12 video in D3D12 textures.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::D3d12)
                .with_layouts(PixelLayoutSet::NV12),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "D3d12OrtEmbedder",
                wanted: "D3D12 video frames",
                got: buf.kind(),
            }
            .into());
        };
        let Some(mut detections) = buf
            .metadata()
            .and_then(|metadata| metadata.get::<Detections>())
            .cloned()
        else {
            out.push(buf);
            return Ok(());
        };
        let size = (frame.width(), frame.height());
        let plan: Plan<Embedding> = self.memory.plan(&buf, &detections, &self.options, size);
        let orientation = self.orientations.of(frame, &self.pp_log);
        // Those that cannot be cut — a face without its five points — are
        // left without a vector.
        let warps: Vec<(usize, Warp)> = plan
            .classify
            .iter()
            .filter_map(|&(index, _)| {
                let warp = Warp::of(
                    &detections.items[index],
                    self.options.align,
                    size,
                    orientation,
                    self.input.size,
                )?;
                Some((index, warp))
            })
            .collect();
        let found = if warps.is_empty() {
            Vec::new()
        } else {
            let only: Vec<Warp> = warps.iter().map(|(_, warp)| *warp).collect();
            self.embed(frame, &only)?
        };
        let answers = warps.iter().map(|(index, _)| *index).zip(found).collect();
        apply_each(
            &mut detections,
            &mut self.memory,
            plan,
            answers,
            |item, vector| item.embeddings.push(vector),
        );
        out.push(detections.attach_to(buf));
        Ok(())
    }

    /// A seek or a flush: the objects after it are numbered anew.
    fn reset(&mut self) {
        self.memory.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{
        AppSink, ChannelOrder, D3d12Upload, Detection, DetectorModel, ModelInput,
        OrtDetectorOptions, SwOrtDetector, SwOrtEmbedder,
    };
    use crate::test_support::{nth_picture, pictures_found_on, try_d3d12_gpu};

    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// What `stage` handed on of `buf`'s detections.
    fn handed<T: RawSink + SrcPads>(stage: &mut T, buf: MediaBuffer) -> Detections {
        let kept = capture(stage);
        stage.consume(buf).expect("looks");
        let kept = kept.lock().unwrap();
        kept[0]
            .metadata()
            .and_then(|m| m.get::<Detections>())
            .cloned()
            .expect("carries Detections")
    }

    /// Through DirectML it makes the vectors the CPU makes of the same
    /// faces: each straightened by the five points a RetinaFace model
    /// found, read through the same map, and the pictures they were made
    /// from alike. A box with no points is left without one on both. Needs
    /// the models and video the CUDA embedder's test does, and a D3D12
    /// device.
    #[test]
    fn it_makes_on_the_gpu_the_vectors_the_cpu_makes() {
        let (Ok(detector), Ok(model), Ok(video)) = (
            std::env::var("MEDIA_PP_TEST_RETINAFACE"),
            std::env::var("MEDIA_PP_TEST_EMBEDDER"),
            std::env::var("MEDIA_PP_TEST_VIDEO"),
        ) else {
            eprintln!(
                "skipping: set MEDIA_PP_TEST_RETINAFACE, MEDIA_PP_TEST_EMBEDDER and \
                 MEDIA_PP_TEST_VIDEO to run this"
            );
            return;
        };
        let Some(gpu) = try_d3d12_gpu() else {
            return;
        };
        let options = OrtEmbedderOptions {
            input: ModelInput {
                order: ChannelOrder::Bgr,
                scale: [2.0; 3],
                bias: [-1.0; 3],
            },
            cutouts: true,
            ..OrtEmbedderOptions::default()
        };
        let mut on_gpu =
            D3d12OrtEmbedder::new("gpu", &gpu, &model, options.clone()).expect("loads");
        let mut cpu = SwOrtEmbedder::new("cpu", &model, options).expect("loads");
        let faces = OrtDetectorOptions {
            model: DetectorModel::RetinaFace,
            conf_threshold: 0.5,
            ..OrtDetectorOptions::default()
        };
        let Some(&n) = pictures_found_on(&detector, &video, 1, faces.clone()).first() else {
            eprintln!("skipping: no face in {video}");
            return;
        };
        let picture = nth_picture(&video, n, ffmpeg::format::Pixel::NV12);
        let mut finder = SwOrtDetector::new("faces", &detector, faces).expect("loads");
        let mut found = handed(&mut finder, MediaBuffer::video(picture.clone()));
        // A box with no points beside the faces.
        found.items.push(Detection::new(0, 0.9, 0.0, 0.0, 0.3, 0.3));
        let expected = handed(
            &mut cpu,
            found.clone().attach_to(MediaBuffer::video(picture.clone())),
        );
        let Ok(mut upload) = D3d12Upload::new("upload", &gpu) else {
            eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
            return;
        };
        let uploaded = capture(&mut upload);
        upload
            .consume(MediaBuffer::video(picture))
            .expect("uploads");
        let buf = uploaded.lock().unwrap().remove(0);
        let actual = handed(&mut on_gpu, found.clone().attach_to(buf));
        let last = expected.items.len() - 1;
        assert!(
            expected.items[last].embeddings.is_empty() && actual.items[last].embeddings.is_empty(),
            "a box without points has no vector"
        );
        let faces = &expected.items[..last];
        assert!(faces.iter().all(|face| face.embeddings.len() == 1));
        for (cpu, gpu) in faces.iter().zip(&actual.items) {
            let gpu = gpu.embeddings.first().expect("a vector on the GPU too");
            let alike = cpu.embeddings[0].similarity(gpu).expect("of one length");
            eprintln!("{alike}");
            assert!(alike > 0.98, "the vectors are {alike} alike");
            let (cpu, gpu) = (
                cpu.embeddings[0]
                    .cutout
                    .as_ref()
                    .expect("a cutout on the CPU"),
                gpu.cutout.as_ref().expect("a cutout on the GPU"),
            );
            assert_eq!((cpu.width, cpu.height), (gpu.width, gpu.height));
            let apart = cpu
                .rgb
                .iter()
                .zip(gpu.rgb.iter())
                .map(|(a, b)| f64::from(a.abs_diff(*b)))
                .sum::<f64>()
                / cpu.rgb.len() as f64;
            eprintln!("cutouts {apart:.2} apart on average");
            assert!(apart < 4.0, "the cutouts are {apart} apart");
        }
    }
}
