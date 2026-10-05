//! [`MetalOrtDetector`]: object detection on VideoToolbox pictures, each
//! handed on with what was found in it, the picture fitted to the model on
//! the GPU.

use std::{path::Path, sync::Arc};

use objc2_metal::{MTLBuffer, MTLPixelFormat, MTLTextureUsage};
use ort::{ep::CoreML, ep::coreml::ModelFormat, inputs, session::Session, value::TensorRef};

use crate::ffmpeg;
use crate::pp_log::{PpLog, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    color::ColorDescription,
    contract::{InputContract, MediaKind, MemoryDomain, PixelLayoutSet, PortContract},
    element::{Element, ElementType, element_pp_log},
    error::Result,
    platform::macos::{
        metal::{Buffer, Kernel, MetalGpu, Texture},
        pixel_buffer::PixelBuffer,
        videotoolbox::{NotVideoToolbox, sw_format_of},
    },
    transform::{Filter, FilterStage, Output, filter_stage},
};

use crate::elements::Detections;

use super::super::{Letterbox, OrtDetectorError, OrtDetectorOptions, decode, labels, model_input};

const SHADER: &str = include_str!("../../../../../../shaders/metal/fit.metal");

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
/// are those [`OrtDetectorOptions`] describes.
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
    fitting: Fitting,
}

/// The kernels that fit a picture into a model's input, and the input they
/// fit it into — apart from the session, so a test can check them without
/// a model.
struct Fitting {
    model: (u32, u32),
    gpu: MetalGpu,
    nv12: Kernel,
    bgra: Kernel,
    /// The model's input, three planes of `f32`, which the kernels write and
    /// the session reads.
    tensor: Buffer,
}

// SAFETY: the kernels and buffer are used by the one thread transforming at
// a time, through `&mut self`; Metal's objects are thread-safe, and the
// buffer is written only by a pass this waits for before it is read — the
// reasoning `MetalPass` gives for its own.
unsafe impl Send for Fitting {}

impl MetalOrtDetector {
    /// Loads the model at `model_path` to run through Core ML.
    ///
    /// Core ML compiles the model for this Mac as the session is made,
    /// which takes a moment for a small model and longer for a large one.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        options: OrtDetectorOptions,
    ) -> Result<Self> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MetalOrtDetector, &name, None);
        let path = model_path.as_ref().display().to_string();

        // ONNX Runtime passes over a provider that fails to register and
        // runs on the CPU in its place, which is what this refuses.
        let provider = CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .build()
            .error_on_failure();
        let session = Session::builder()
            .map_err(OrtDetectorError::from)?
            .with_execution_providers([provider])
            .map_err(|error| OrtDetectorError::Ort(error.into()))?
            .commit_from_file(model_path)
            .map_err(OrtDetectorError::from)?;

        let model = model_input(&session)?;
        let labels = labels(options.labels.as_deref(), &session);
        let fitting = Fitting::new(model)?;

        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, {} labels, on Core ML",
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
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            session,
            options,
            labels,
            fitting,
        })))
    }
}

impl Fitting {
    /// The kernels, and an input of `model`'s size in shared memory.
    fn new(model: (u32, u32)) -> std::result::Result<Self, OrtDetectorError> {
        let gpu = MetalGpu::new()?;
        let [nv12, bgra] = <[Kernel; 2]>::try_from(gpu.kernels(SHADER, &["fit_nv12", "fit_bgra"])?)
            .unwrap_or_else(|_| unreachable!("two kernels for two names"));
        let tensor =
            gpu.shared_buffer(3 * model.0 as usize * model.1 as usize * size_of::<f32>())?;
        Ok(Self {
            model,
            gpu,
            nv12,
            bgra,
            tensor,
        })
    }

    /// Fits `frame` into the input and says how it was fitted.
    fn fit(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Letterbox, OrtDetectorError> {
        let layout = match sw_format_of(frame) {
            Ok(layout @ (ffmpeg::format::Pixel::NV12 | ffmpeg::format::Pixel::BGRA)) => layout,
            Ok(other) | Err(NotVideoToolbox::Format(other)) => {
                return Err(OrtDetectorError::UnsupportedPicture(other));
            }
            Err(NotVideoToolbox::NoFramesContext) => {
                return Err(OrtDetectorError::MissingPixelBuffer("frames context"));
            }
        };
        let buffer =
            PixelBuffer::of_frame(frame).ok_or(OrtDetectorError::MissingPixelBuffer("pixels"))?;
        let size = (frame.width(), frame.height());
        let surface = buffer.size();
        if size.0 > surface.0 || size.1 > surface.1 {
            return Err(OrtDetectorError::PictureOutsideSurface {
                picture: size,
                surface,
            });
        }

        let letterbox = Letterbox::new(size, self.model);
        let rows = if layout == ffmpeg::format::Pixel::NV12 {
            ColorDescription::of(frame).yuv_to_rgb_rows(frame.height())
        } else {
            [[0.0; 4]; 3]
        };
        let mut parameters: Vec<u8> = [
            self.model.0,
            self.model.1,
            letterbox.offset.0,
            letterbox.offset.1,
            letterbox.scaled.0,
            letterbox.scaled.1,
            size.0,
            size.1,
        ]
        .iter()
        .flat_map(|word| word.to_ne_bytes())
        .collect();
        parameters.extend(rows.iter().flatten().flat_map(|value| value.to_ne_bytes()));

        let read = MTLTextureUsage::ShaderRead;
        let (kernel, textures): (&Kernel, Vec<Texture>) = if layout == ffmpeg::format::Pixel::NV12 {
            (
                &self.nv12,
                vec![
                    self.gpu.plane(&buffer, 0, MTLPixelFormat::R8Unorm, read)?,
                    self.gpu.plane(&buffer, 1, MTLPixelFormat::RG8Unorm, read)?,
                ],
            )
        } else {
            (
                &self.bgra,
                vec![
                    self.gpu
                        .plane(&buffer, 0, MTLPixelFormat::BGRA8Unorm, read)?,
                ],
            )
        };
        let bound: Vec<&Texture> = textures.iter().collect();
        let mut pass = self.gpu.pass()?;
        pass.bind_buffer(&self.tensor, 1);
        pass.dispatch(kernel, &bound, Some(&parameters), self.model);
        pass.finish()?;
        Ok(letterbox)
    }

    /// The input, as the last [`Self::fit`] left it: R, G and B planes of
    /// the model's width by height.
    fn input(&self) -> &[f32] {
        // SAFETY: the buffer is this fitting's own, three planes of the
        // model's size of `f32` in shared memory, and the pass that wrote it
        // has finished; nothing writes it again until the next `fit`, which
        // takes `&mut self` and so waits for this borrow to end.
        unsafe {
            std::slice::from_raw_parts(
                self.tensor.contents().as_ptr().cast::<f32>(),
                3 * self.model.0 as usize * self.model.1 as usize,
            )
        }
    }
}

impl Detecting {
    /// Looks at `frame`, and says what it found.
    fn detect(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Detections, OrtDetectorError> {
        let letterbox = self.fitting.fit(frame)?;
        let (width, height) = (self.fitting.model.0 as usize, self.fitting.model.1 as usize);
        let input = TensorRef::from_array_view(([1, 3, height, width], self.fitting.input()))?;
        let outputs = self.session.run(inputs![input])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        Ok(Detections {
            detector: Arc::clone(&self.name),
            labels: Arc::clone(&self.labels),
            items: decode(output, &letterbox, &self.options)?,
        })
    }
}

impl Element for Detecting {
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

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            return Err(OrtDetectorError::UnsupportedBuffer {
                detector: "MetalOrtDetector",
                wanted: "VideoToolbox video frames",
                got: buf.kind(),
            }
            .into());
        };
        let detections = self.detect(frame)?;
        out.push(detections.attach_to(buf));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, SwOrtDetector, VideoToolboxDevice, VideoToolboxUpload};
    use crate::test_support::try_videotoolbox_device;

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

    /// `frame` uploaded to `device`, as a VideoToolbox picture of its layout.
    fn upload(device: &VideoToolboxDevice, frame: ffmpeg::frame::Video) -> MediaBuffer {
        let mut upload = VideoToolboxUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("uploads");
        uploaded.lock().unwrap().remove(0)
    }

    /// A `format` picture in system memory in which no sample is its
    /// neighbour's, so a kernel reading the wrong one is caught.
    fn pattern(format: ffmpeg::format::Pixel, width: u32, height: u32) -> ffmpeg::frame::Video {
        let mut frame = ffmpeg::frame::Video::new(format, width, height);
        frame.set_pts(Some(150));
        let (width, height) = (width as usize, height as usize);
        if format == ffmpeg::format::Pixel::NV12 {
            let stride = frame.stride(0);
            for (y, row) in frame
                .data_mut(0)
                .chunks_mut(stride)
                .take(height)
                .enumerate()
            {
                for (x, luma) in row[..width].iter_mut().enumerate() {
                    *luma = (x * 7 + y * 13) as u8;
                }
            }
            let stride = frame.stride(1);
            for (y, row) in frame
                .data_mut(1)
                .chunks_mut(stride)
                .take(height / 2)
                .enumerate()
            {
                for (x, pair) in row[..width].chunks_mut(2).enumerate() {
                    pair[0] = (x * 11 + 40) as u8;
                    pair[1] = (y * 17 + 60) as u8;
                }
            }
        } else {
            let stride = frame.stride(0);
            for (y, row) in frame
                .data_mut(0)
                .chunks_mut(stride)
                .take(height)
                .enumerate()
            {
                for (x, pixel) in row[..width * 4].chunks_mut(4).enumerate() {
                    pixel.copy_from_slice(&[(x * 5) as u8, (y * 9) as u8, (x + y * 3) as u8, 255]);
                }
            }
        }
        frame
    }

    /// What the kernels are to write for `frame`: each pixel of the scaled
    /// picture the sample under its centre, made RGB by the frame's own
    /// rows, and the margins grey — three planes, R, G and B.
    fn expected(frame: &ffmpeg::frame::Video, model: (u32, u32)) -> Vec<f32> {
        let (width, height) = (frame.width(), frame.height());
        let letterbox = Letterbox::new((width, height), model);
        let rows = ColorDescription::of(frame).yuv_to_rgb_rows(height);
        let plane = (model.0 * model.1) as usize;
        let mut tensor = vec![114.0 / 255.0; 3 * plane];
        for y in 0..letterbox.scaled.1 {
            for x in 0..letterbox.scaled.0 {
                let sx = (((x as f32 + 0.5) * width as f32 / letterbox.scaled.0 as f32) as u32)
                    .min(width - 1) as usize;
                let sy = (((y as f32 + 0.5) * height as f32 / letterbox.scaled.1 as f32) as u32)
                    .min(height - 1) as usize;
                let rgb = if frame.format() == ffmpeg::format::Pixel::NV12 {
                    let luma = frame.data(0)[sy * frame.stride(0) + sx];
                    let chroma = &frame.data(1)[(sy / 2) * frame.stride(1) + (sx / 2) * 2..];
                    let yuv = [
                        f32::from(luma) / 255.0,
                        f32::from(chroma[0]) / 255.0,
                        f32::from(chroma[1]) / 255.0,
                        1.0,
                    ];
                    rows.map(|row| {
                        row.iter()
                            .zip(yuv)
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            .clamp(0.0, 1.0)
                    })
                } else {
                    let bgra = &frame.data(0)[sy * frame.stride(0) + sx * 4..];
                    [bgra[2], bgra[1], bgra[0]].map(|value| f32::from(value) / 255.0)
                };
                let at = ((letterbox.offset.1 + y) * model.0 + letterbox.offset.0 + x) as usize;
                for (channel, value) in rgb.into_iter().enumerate() {
                    tensor[channel * plane + at] = value;
                }
            }
        }
        tensor
    }

    /// The kernels write, from NV12 and from BGRA alike, the very input the
    /// CPU reads the picture as: the sample under each pixel's centre in
    /// the right channel and plane, made RGB by the picture's own colour,
    /// and grey around it. Needs no model; skipped without VideoToolbox.
    #[test]
    fn a_picture_is_fitted_as_the_cpu_reads_it() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let model = (64, 64);
        let mut fitting = Fitting::new(model).expect("the kernels compile");
        for format in [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA] {
            // Twice as wide as high: grey above and below, and every pixel
            // centre at an exact sample, so no rounding can pick another.
            let picture = pattern(format, 128, 64);
            let wanted = expected(&picture, model);
            let MediaBuffer::Video(frame) = upload(&device, picture) else {
                panic!("a picture is uploaded");
            };
            let letterbox = fitting.fit(&frame).expect("fits");
            assert_eq!(letterbox, Letterbox::new((128, 64), model));
            for (index, (actual, wanted)) in fitting.input().iter().zip(&wanted).enumerate() {
                assert!(
                    (actual - wanted).abs() < 1e-3,
                    "{format:?}: float {index} is {actual}, not {wanted}"
                );
            }
        }
    }

    /// A picture in system memory is refused, saying what it is, rather
    /// than read as a pixel buffer it does not hold.
    #[test]
    fn a_picture_in_system_memory_is_refused() {
        if try_videotoolbox_device().is_none() {
            return;
        }
        let mut fitting = Fitting::new((64, 64)).expect("the kernels compile");
        let picture = pattern(ffmpeg::format::Pixel::NV12, 128, 64);
        assert!(matches!(
            fitting.fit(&picture),
            Err(OrtDetectorError::UnsupportedPicture(
                ffmpeg::format::Pixel::NV12
            ))
        ));
    }

    /// The 150th picture of `video`, as `format` in system memory.
    fn picture(video: &str, format: ffmpeg::format::Pixel) -> ffmpeg::frame::Video {
        let mut input = ffmpeg::format::input(video).expect("the video opens");
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("a video stream");
        let index = stream.index();
        let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .and_then(|context| context.decoder().video())
            .expect("a decoder");
        let mut decoded = ffmpeg::frame::Video::empty();
        let mut seen = 0;
        for (stream, packet) in input.packets() {
            if stream.index() != index {
                continue;
            }
            decoder.send_packet(&packet).expect("decodes");
            while decoder.receive_frame(&mut decoded).is_ok() {
                seen += 1;
                if seen == 150 {
                    let mut converted = ffmpeg::frame::Video::empty();
                    ffmpeg::software::scaling::Context::get(
                        decoded.format(),
                        decoded.width(),
                        decoded.height(),
                        format,
                        decoded.width(),
                        decoded.height(),
                        ffmpeg::software::scaling::Flags::BILINEAR,
                    )
                    .and_then(|mut context| context.run(&decoded, &mut converted))
                    .expect("converts");
                    converted.set_pts(Some(150));
                    return converted;
                }
            }
        }
        panic!("the video is shorter than 150 pictures");
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
            MediaBuffer::video(picture(&video, ffmpeg::format::Pixel::NV12)),
        );
        assert!(
            expected.items.iter().any(|found| found.score > 0.5),
            "the picture has something in it"
        );

        let mut gpu = MetalOrtDetector::new("gpu", &model, OrtDetectorOptions::default())
            .expect("loads on Core ML");
        for format in [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA] {
            let actual = found(&mut gpu, upload(&device, picture(&video, format)));
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
}
