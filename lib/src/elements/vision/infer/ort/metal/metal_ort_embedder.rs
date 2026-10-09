//! [`MetalOrtEmbedder`]: what a detector found, made into vectors by an
//! embedding model on VideoToolbox pictures, each object read from the
//! picture on the GPU.

use std::{path::Path, sync::Arc};

use ort::{inputs, session::Session, value::TensorRef};

use crate::ffmpeg;
use crate::orientation::Orientations;
use crate::pp_log::{PpLog, pp_info};
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
use super::super::classify::{Input, Memory, Plan, apply_each};
use super::super::embed::{OrtEmbedderOptions, Warp, cutouts, embeddings, with_cutouts};
use super::fitting::Fitting;
use super::{CORE_ML_BATCH, object_model_session};

/// Makes each object a detector found on each VideoToolbox picture into a
/// vector with an embedding model — what
/// [`SwOrtEmbedder`](crate::elements::SwOrtEmbedder) does on the CPU, with
/// each object read into the model's input by a Metal kernel where the
/// picture is, through the map [`OrtEmbedderOptions::align`] makes for it —
/// a face straightened by its five points, a box stretched — between the
/// four pixels around each point, several objects to one run, and the model
/// run by ONNX Runtime's Core ML provider on the GPU or the Neural Engine.
///
/// It goes after a detector — one that finds five points for a face, such
/// as a [`MetalOrtDetector`](crate::elements::MetalOrtDetector) of
/// [`DetectorModel::RetinaFace`](crate::elements::DetectorModel::RetinaFace),
/// for [`Alignment::FivePoints`](crate::elements::Alignment::FivePoints) —
/// and best after an [`ObjectTracker`](crate::elements::ObjectTracker),
/// which lets a followed object be embedded once and its vector kept, as
/// [`OrtEmbedderOptions::reembed`] says. It takes NV12 or BGRA VideoToolbox
/// pictures from any device, since a pixel buffer belongs to none, and HDR
/// P010 through an SDR copy, as the detector does. The
/// model has one input, `[batch, 3, height, width]` — 112 for a side it
/// leaves open — and its first output is a vector for each, made of length
/// 1 here whatever the model makes.
///
/// As the classifier does, it runs through Core ML or not at all, and a
/// model that takes any number of objects at once is fixed to take four,
/// which Core ML compiles once; a picture's objects go in fours.
///
/// # Requirements
///
/// The `ort-coreml` feature, on an Apple silicon Mac.
pub struct MetalOrtEmbedder(FilterStage<Embedder>);

filter_stage!(MetalOrtEmbedder);

/// What a [`MetalOrtEmbedder`] does with each picture.
struct Embedder {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtEmbedderOptions,
    input: Input,
    memory: Memory<Embedding>,
    fitting: Fitting,
    /// How each picture is turned to be shown.
    orientations: Orientations,
}

impl MetalOrtEmbedder {
    /// Loads the embedding model at `model_path` to run through Core ML.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        options: OrtEmbedderOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalOrtEmbedder, &name, None);
        let path = model_path.as_ref().display().to_string();
        let (session, input) = object_model_session(model_path.as_ref(), 112, &pp_log)?;
        let fitting = Fitting::new(input.size, input.batch.unwrap_or(CORE_ML_BATCH))?;
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batch={:?}, {:?}, {:?}, on Core ML",
            input.size.0,
            input.size.1,
            input.batch,
            options.input,
            options.align
        );
        Ok(Self(FilterStage::new(Embedder {
            name,
            pp_log,
            session,
            options,
            input,
            memory: Memory::default(),
            fitting,
            orientations: Orientations::default(),
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
                    self.fitting.input(rows),
                    group.len(),
                    (width, height),
                    self.options.input,
                )
            } else {
                Vec::new()
            };
            let input = TensorRef::from_array_view((
                [rows, 3, height as usize, width as usize],
                self.fitting.input(rows),
            ))?;
            let outputs = self.session.run(inputs![input])?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
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
        ElementType::MetalOrtEmbedder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Embedder {
    /// Decoded NV12, BGRA or HDR P010 video in VideoToolbox pixel buffers.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::VideoToolbox)
                .with_layouts(PixelLayoutSet::GPU_SCALABLE),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "MetalOrtEmbedder",
                wanted: "VideoToolbox video frames",
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
        AppSink, ChannelOrder, Detection, DetectorModel, ModelInput, OrtDetectorOptions,
        SwOrtDetector, SwOrtEmbedder, VideoToolboxUpload,
    };
    use crate::test_support::{nth_picture, try_videotoolbox_device};

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

    /// Through Core ML it makes the vectors the CPU makes of the same faces,
    /// from NV12 and from BGRA: each face straightened by the five points a
    /// RetinaFace model found, read through the same map by a Metal kernel.
    /// A box with no points is left without one on both. Needs a RetinaFace
    /// model, an embedding model and a video with faces — the CUDA
    /// embedder's test's.
    #[test]
    fn it_makes_through_core_ml_the_vectors_the_cpu_makes() {
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
        let Some(device) = try_videotoolbox_device() else {
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
        let mut gpu =
            MetalOrtEmbedder::new("gpu", &model, options.clone()).expect("loads on Core ML");
        let mut cpu = SwOrtEmbedder::new("cpu", &model, options).expect("loads");
        let faces = OrtDetectorOptions {
            model: DetectorModel::RetinaFace,
            conf_threshold: 0.5,
            ..OrtDetectorOptions::default()
        };
        let Some(&n) =
            crate::test_support::pictures_found_on(&detector, &video, 1, faces.clone()).first()
        else {
            eprintln!("skipping: no face in {video}");
            return;
        };
        let mut finder = SwOrtDetector::new("faces", &detector, faces).expect("loads");
        let mut found = handed(
            &mut finder,
            MediaBuffer::video(nth_picture(&video, n, ffmpeg::format::Pixel::NV12)),
        );
        // A box with no points beside the faces.
        found.items.push(Detection::new(0, 0.9, 0.0, 0.0, 0.3, 0.3));

        for layout in [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA] {
            let picture = nth_picture(&video, n, layout);
            let expected = handed(
                &mut cpu,
                found.clone().attach_to(MediaBuffer::video(picture.clone())),
            );
            let mut upload = VideoToolboxUpload::new("upload", &device);
            let uploaded = capture(&mut upload);
            upload
                .consume(MediaBuffer::video(picture))
                .expect("uploads");
            let on_gpu = uploaded.lock().unwrap().remove(0);
            let actual = handed(&mut gpu, found.clone().attach_to(on_gpu));
            let last = expected.items.len() - 1;
            assert!(
                expected.items[last].embeddings.is_empty()
                    && actual.items[last].embeddings.is_empty(),
                "a box without points has no vector"
            );
            let faces = &expected.items[..last];
            assert!(!faces.is_empty());
            assert!(faces.iter().all(|face| face.embeddings.len() == 1));
            for (cpu, gpu) in faces.iter().zip(&actual.items) {
                let gpu = gpu.embeddings.first().expect("a vector on the GPU too");
                let alike = cpu.embeddings[0].similarity(gpu).expect("of one length");
                eprintln!("{layout:?}: {alike}");
                assert!(alike > 0.98, "{layout:?}: the vectors are {alike} alike");
                // The pictures they were made from are the same picture.
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
                eprintln!("{layout:?}: cutouts {apart:.2} apart on average");
                assert!(apart < 4.0, "{layout:?}: the cutouts are {apart} apart");
            }
        }
    }
}
