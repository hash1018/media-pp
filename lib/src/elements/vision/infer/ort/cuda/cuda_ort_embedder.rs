//! [`CudaOrtEmbedder`]: what a detector found, made into vectors by an
//! embedding model on CUDA pictures without them leaving the GPU.

use std::{path::Path, sync::Arc};

use ffmpeg_next::{self as ffmpeg, ffi};
use ort::{
    inputs,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::{Shape, TensorRefMut},
};

use crate::pp_log::{PpLog, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::{Detections, Embedding},
    error::Result,
    platform::cuda::{
        CudaDevice,
        driver::{BgraSurface, CudaDriver, CudaTensor, FitKernels, Nv12Surface, YuvToBgra},
        frame::{self, CudaSurfaces},
    },
    platform::ffmpeg::AvBufferRef,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::super::ChannelOrder;
use super::super::OrtError;
use super::super::classify::{Input, Memory, Plan, apply_each};
use super::super::embed::{OrtEmbedderOptions, Warp, cutouts, embeddings, with_cutouts};
use crate::orientation::Orientations;

/// The most objects embedded in one run of a model that takes any number
/// at once.
const MAX_BATCH: usize = 32;

/// Makes each object a detector found on each CUDA picture into a vector
/// with an embedding model, without the picture leaving the GPU — what
/// [`SwOrtEmbedder`](crate::elements::SwOrtEmbedder) does on the CPU: a
/// kernel reads each object into the model's input through the map
/// [`OrtEmbedderOptions::align`] makes for it — a face straightened by its
/// five points, a box stretched — between the four pixels around each
/// point, several objects to one run, and ONNX Runtime's CUDA provider
/// reads them there.
///
/// It takes NV12, BGRA or HDR P010 CUDA pictures from the same
/// [`CudaDevice`] as the rest of the pipeline; an HDR picture is read from
/// an SDR copy, as the detectors read it. Built with `ort-tensorrt`, it
/// runs the model through TensorRT in half precision where TensorRT can
/// run, and on CUDA alone where it cannot, as the classifiers do.
///
/// # Requirements
///
/// The `ort-cuda` feature, and at run time what
/// [`CudaOrtDetector::runtime`](crate::elements::CudaOrtDetector::runtime)
/// says it needs of the CUDA side.
pub struct CudaOrtEmbedder(FilterStage<Embedder>);

filter_stage!(CudaOrtEmbedder);

/// What a [`CudaOrtEmbedder`] does with each picture.
///
/// The device buffer and kernels come before the driver: fields drop in
/// order, and both are freed in the driver's context, which the driver
/// releases.
struct Embedder {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtEmbedderOptions,
    input: Input,
    /// How many inputs `tensor` holds.
    capacity: usize,
    memory: Memory<Embedding>,
    tensor: CudaTensor,
    kernels: FitKernels,
    /// Where an HDR picture is brought to SDR for the model; before the
    /// driver, whose context it is freed in.
    sdr: super::SdrCopy,
    driver: CudaDriver,
    device_memory: MemoryInfo<'static>,
    /// The device context incoming frames must belong to, compared by
    /// pointer.
    device_ctx: *mut ffi::AVHWDeviceContext,
    _hw_device_ctx: Arc<AvBufferRef>,
    /// How each picture is turned to be shown.
    orientations: Orientations,
}

// SAFETY: the session, buffer and kernels are used by the one thread
// transforming at a time; `device_ctx` only ever has its address compared,
// and `&mut self` on every method that touches the rest rules out
// concurrent access — the reasoning `CudaOrtClassifier` gives for its own.
unsafe impl Send for Embedder {}

impl CudaOrtEmbedder {
    /// Loads the embedding model at `model_path` to run on `device`'s GPU —
    /// the same [`CudaDevice`] every other CUDA element in the pipeline was
    /// built from.
    pub fn new(
        name: impl Into<String>,
        device: &CudaDevice,
        model_path: impl AsRef<Path>,
        options: OrtEmbedderOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CudaOrtEmbedder, &name, None);
        let path = model_path.as_ref().display().to_string();
        let (session, input, provider, linked) =
            super::object_model_session(model_path.as_ref(), 112, MAX_BATCH, &pp_log, "embedding")?;
        let capacity = input.batch.unwrap_or(MAX_BATCH).max(1);
        let driver = CudaDriver::retain_primary().map_err(OrtError::from)?;
        let kernels = driver.fit_kernels().map_err(OrtError::from)?;
        let tensor = driver
            .batch_tensor(input.size.0, input.size.1, capacity)
            .map_err(OrtError::from)?;
        let device_memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(OrtError::from)?;
        let hw_device_ctx = device.retain();
        // SAFETY: `hw_device_ctx` owns a live `AVBufferRef` for a CUDA device
        // context, whose `data` is that `AVHWDeviceContext` by FFmpeg's own
        // definition; only the pointer's identity is kept, and the reference
        // held beside it keeps that identity from being reused.
        let device_ctx = unsafe { (*hw_device_ctx.as_ptr()).data as *mut ffi::AVHWDeviceContext };
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batch={:?}, {:?}, {:?}, on {provider}; {linked}",
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
            capacity,
            memory: Memory::default(),
            tensor,
            kernels,
            sdr: super::SdrCopy::default(),
            driver,
            device_memory,
            device_ctx,
            _hw_device_ctx: hw_device_ctx,
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
        let surface = frame::validate(
            frame,
            ElementType::CudaOrtEmbedder,
            self.device_ctx,
            CudaSurfaces::NV12_BGRA_OR_P010,
        )?;
        // An HDR picture is read from its SDR copy, made once for all its
        // objects.
        let sdr = if surface.layout == ffmpeg::format::Pixel::P010LE {
            Some(
                self.sdr
                    .of(&self.driver, frame, ElementType::CudaOrtEmbedder)?,
            )
        } else {
            None
        };
        let size = (frame.width(), frame.height());
        let model = self.input.size;
        let values = self.options.input;
        let mut found = Vec::with_capacity(warps.len());
        for group in warps.chunks(self.capacity) {
            for (slot, warp) in group.iter().enumerate() {
                match sdr {
                    None if surface.layout == ffmpeg::format::Pixel::NV12 => {
                        let source =
                            Nv12Surface::from_frame(frame).ok_or(OrtError::MissingSurface)?;
                        self.driver.warp_nv12(
                            &self.kernels,
                            &self.tensor,
                            slot,
                            model,
                            warp.0,
                            source,
                            size,
                            &YuvToBgra::of_frame(frame),
                        )?;
                    }
                    _ => {
                        let source = match sdr {
                            Some(sdr) => sdr,
                            None => {
                                BgraSurface::from_frame(frame).ok_or(OrtError::MissingSurface)?
                            }
                        };
                        self.driver.warp_bgra(
                            &self.kernels,
                            &self.tensor,
                            slot,
                            model,
                            warp.0,
                            source,
                            size,
                        )?;
                    }
                }
            }
            // A model of fixed batch is handed exactly that many; one of
            // open batch, as many as there are.
            let rows = self.input.batch.unwrap_or(group.len());
            if values.order == ChannelOrder::Bgr {
                self.driver
                    .swap_planes(&self.kernels, &self.tensor, model, rows)?;
            }
            if !values.is_unscaled() {
                self.driver.scale_planes(
                    &self.kernels,
                    &self.tensor,
                    model,
                    rows,
                    values.scale,
                    values.bias,
                )?;
            }
            // The kernels ran on this driver's context; the model reads the
            // tensor on ONNX Runtime's own stream.
            self.driver.synchronize()?;
            // The pictures, read back from the inputs the model is about
            // to be handed.
            let cut = if self.options.cutouts {
                let mut inputs =
                    vec![0.0f32; group.len() * 3 * model.0 as usize * model.1 as usize];
                self.driver.download(&self.tensor, &mut inputs)?;
                cutouts(&inputs, group.len(), model, values)
            } else {
                Vec::new()
            };
            let shape = Shape::new([rows as i64, 3, i64::from(model.1), i64::from(model.0)]);
            // SAFETY: the pointer is this element's own device allocation of
            // `capacity` inputs, at least `rows` — exactly `shape` or more — in
            // the primary context ONNX Runtime's CUDA provider also runs in; it
            // outlives the run, the view being dropped before this returns.
            let input = unsafe {
                TensorRefMut::<f32>::from_raw(
                    self.device_memory.clone(),
                    self.tensor.pointer() as usize as *mut _,
                    shape,
                )?
            };
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
        ElementType::CudaOrtEmbedder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Embedder {
    /// Device-resident frames: NV12, BGRA, or P010 read through an SDR
    /// copy.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, MemoryDomain::Cuda)
                .with_layouts(crate::contract::PixelLayoutSet::GPU_SCALABLE),
        )
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtError::UnsupportedBuffer {
                detector: "CudaOrtEmbedder",
                wanted: "CUDA video frames",
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
        AppSink, CudaUpload, Detection, DetectorModel, OrtDetectorOptions, SwOrtDetector,
        SwOrtEmbedder,
    };
    use crate::platform::cuda::CudaFrameFormat;
    use crate::test_support::nth_picture;

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

    /// On the GPU it makes the vectors the CPU makes of the same faces, from
    /// NV12 and from BGRA: each face straightened by the five points a
    /// RetinaFace model found, read through the same map. A box with no
    /// points is left without one on both. Needs a RetinaFace model, an
    /// embedding model that takes BGR from -1 to 1 — AdaFace's — and a
    /// video with faces.
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
        let Some((device, _serial)) = crate::test_support::try_cuda_device() else {
            return;
        };
        let options = OrtEmbedderOptions {
            input: super::super::super::ModelInput {
                order: ChannelOrder::Bgr,
                scale: [2.0; 3],
                bias: [-1.0; 3],
            },
            cutouts: true,
            ..OrtEmbedderOptions::default()
        };
        let Ok(mut gpu) = CudaOrtEmbedder::new("gpu", &device, &model, options.clone()) else {
            eprintln!("skipping: the CUDA runtime cannot be used here");
            return;
        };
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

        for (format, layout) in [
            (CudaFrameFormat::Nv12, ffmpeg::format::Pixel::NV12),
            (CudaFrameFormat::Bgra, ffmpeg::format::Pixel::BGRA),
        ] {
            let picture = nth_picture(&video, n, layout);
            let expected = handed(
                &mut cpu,
                found.clone().attach_to(MediaBuffer::video(picture.clone())),
            );
            let mut upload = CudaUpload::new("upload", &device, format);
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
            assert!(faces.iter().all(|face| face.embeddings.len() == 1));
            for (cpu, gpu) in faces.iter().zip(&actual.items) {
                let gpu = gpu.embeddings.first().expect("a vector on the GPU too");
                let alike = cpu.embeddings[0].similarity(gpu).expect("of one length");
                eprintln!("{format:?}: {alike}");
                assert!(alike > 0.98, "{format:?}: the vectors are {alike} alike");
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
                eprintln!("{format:?}: cutouts {apart:.2} apart on average");
                assert!(apart < 4.0, "{format:?}: the cutouts are {apart} apart");
            }
        }
    }
}
