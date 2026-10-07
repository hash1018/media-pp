//! [`SwOrtEmbedder`]: what a detector found, made into vectors by an
//! embedding model on the CPU.

use std::{path::Path, sync::Arc};

use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

use crate::ffmpeg;
use crate::pp_log::{PpLog, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::{Detections, Embedding},
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::OrtError;
use super::classify::{Input, Memory, Plan, apply_each, image_input};
use super::embed::{OrtEmbedderOptions, Warp, embeddings};
use super::model::ModelInput;
use super::sw_ort_detector::{ToRgb, is_hardware};
use crate::orientation::Orientations;

/// The most objects embedded in one run of a model that takes any number
/// at once.
const MAX_BATCH: usize = 32;

/// Makes each object a detector found on each picture into a vector with an
/// embedding model, on the CPU, and hands the picture on with each vector
/// in its object's [`Detection::embeddings`](crate::elements::Detection::embeddings)
/// — vectors that lie near each other where two objects look alike, for
/// telling which boxes across a video are the same thing.
///
/// It goes after a detector, and best after an
/// [`ObjectTracker`](crate::elements::ObjectTracker): a followed object is
/// embedded once and its vector kept on every picture it is followed
/// through, until [`OrtEmbedderOptions::reembed`] says to look again. Each
/// object is cut from the picture as [`OrtEmbedderOptions::align`] says —
/// a face straightened by its five points, as a face recognition model
/// expects, or a whole object's box stretched.
///
/// It takes decoded pictures in system memory, any format. The model has
/// one input, `[batch, 3, height, width]` — 112 for a side it leaves open —
/// and its first output is a vector for each, made of length 1 here
/// whatever the model makes; objects are embedded together where its batch
/// is open, one at a time where it is fixed. The vectors are the model's:
/// only those of one model compare.
pub struct SwOrtEmbedder(FilterStage<Embedder>);

filter_stage!(SwOrtEmbedder);

/// What an [`SwOrtEmbedder`] does with each picture.
struct Embedder {
    name: Arc<str>,
    pp_log: PpLog,
    session: Session,
    options: OrtEmbedderOptions,
    input: Input,
    memory: Memory<Embedding>,
    converting: ToRgb,
    /// How each picture is turned to be shown.
    orientations: Orientations,
}

impl SwOrtEmbedder {
    /// Loads the embedding model at `model_path` to run on the CPU.
    pub fn new(
        name: impl Into<String>,
        model_path: impl AsRef<Path>,
        options: OrtEmbedderOptions,
    ) -> Result<Self> {
        let path = model_path.as_ref().display().to_string();
        let session = Session::builder()
            .map_err(OrtError::from)?
            .commit_from_file(model_path)
            .map_err(OrtError::from)?;
        let input = image_input(&session, 112)?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwOrtEmbedder, &name, None);
        pp_info!(
            pp_log: &pp_log,
            "model loaded: path={path}, input={}x{}, batch={:?}, {:?}, {:?}",
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
            converting: ToRgb::default(),
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
    ) -> Result<Vec<Option<Embedding>>> {
        let (width, height) = self.input.size;
        let batch = self.input.batch.unwrap_or(MAX_BATCH).max(1);
        let values = self.options.input;
        let rgb = self.converting.run(frame)?.clone();
        let mut found = Vec::with_capacity(warps.len());
        for group in warps.chunks(batch) {
            // A model of fixed batch is handed exactly that many, the rest
            // left blank; one of open batch, as many as there are.
            let rows = self.input.batch.unwrap_or(group.len());
            let mut tensor = Array4::<f32>::zeros((rows, 3, height as usize, width as usize));
            for (slot, warp) in group.iter().enumerate() {
                warp_into(&rgb, *warp, (width, height), values, &mut tensor, slot);
            }
            let outputs = self
                .session
                .run(inputs![
                    TensorRef::from_array_view(&tensor).map_err(OrtError::from)?
                ])
                .map_err(OrtError::from)?;
            let (shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(OrtError::from)?;
            let shape: Vec<usize> = shape.iter().map(|&side| side.max(0) as usize).collect();
            found.extend(embeddings(&self.name, &shape, data, group.len())?);
        }
        Ok(found)
    }
}

/// What the RGB24 picture `rgb` holds at `(x, y)` — a pixel's centre at
/// whole numbers — between the four pixels around it, 0 to 1, black
/// outside it: what the CUDA kernels sample.
pub(super) fn sample(rgb: &ffmpeg::frame::Video, x: f32, y: f32) -> [f32; 3] {
    let (width, height) = (rgb.width() as i64, rgb.height() as i64);
    let stride = rgb.stride(0);
    let data = rgb.data(0);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let mut sum = [0.0f32; 3];
    for (dx, dy, weight) in [
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ] {
        let (px, py) = (x0 + dx, y0 + dy);
        if px < 0 || py < 0 || px >= width || py >= height {
            continue;
        }
        let pixel = &data[py as usize * stride + px as usize * 3..][..3];
        for (sum, value) in sum.iter_mut().zip(pixel) {
            *sum += f32::from(*value) * weight;
        }
    }
    sum.map(|value| value / 255.0)
}

/// The input `warp` reads from the RGB24 picture `rgb`, in the model's
/// `values`, into input `slot` of `tensor`, `size` in pixels.
fn warp_into(
    rgb: &ffmpeg::frame::Video,
    warp: Warp,
    size: (u32, u32),
    values: ModelInput,
    tensor: &mut Array4<f32>,
    slot: usize,
) {
    let planes = values.planes();
    for v in 0..size.1 {
        for u in 0..size.0 {
            let (x, y) = warp.source(u as f32, v as f32);
            for (channel, value) in sample(rgb, x, y).into_iter().enumerate() {
                let plane = planes[channel];
                tensor[[slot, plane, v as usize, u as usize]] =
                    value * values.scale[plane] + values.bias[plane];
            }
        }
    }
}

impl Element for Embedder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwOrtEmbedder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Embedder {
    /// Decoded video in system memory, any pixel layout.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ))
    }

    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let refused = |got| OrtError::UnsupportedBuffer {
            detector: "SwOrtEmbedder",
            wanted: "video frames in system memory",
            got,
        };
        let MediaBuffer::Video(frame) = &buf else {
            return Err(refused(buf.kind()).into());
        };
        if is_hardware(frame.format()) {
            return Err(refused("a hardware video frame").into());
        }
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
    use super::*;

    /// A picture whose every pixel is its own column in red and its own
    /// row in green.
    fn ramp() -> ffmpeg::frame::Video {
        let mut rgb = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, 8, 8);
        let stride = rgb.stride(0);
        for y in 0..8 {
            for x in 0..8 {
                rgb.data_mut(0)[y * stride + x * 3..][..3].copy_from_slice(&[
                    (x * 10) as u8,
                    (y * 10) as u8,
                    7,
                ]);
            }
        }
        rgb
    }

    /// Between pixels a sample is the mix of the four around it; outside
    /// the picture it is black.
    #[test]
    fn a_sample_mixes_the_four_pixels_around_it() {
        let rgb = ramp();
        let [r, g, b] = sample(&rgb, 2.25, 3.5);
        assert!((r * 255.0 - 22.5).abs() < 1e-3 && (g * 255.0 - 35.0).abs() < 1e-3);
        assert!((b * 255.0 - 7.0).abs() < 1e-3);
        assert_eq!(sample(&rgb, -5.0, 2.0), [0.0; 3]);
        // Half outside: half black.
        let [_, _, b] = sample(&rgb, -0.5, 2.0);
        assert!((b * 255.0 - 3.5).abs() < 1e-3, "{b}");
    }

    /// Values go into the model's planes in its order and scale: BGR, and
    /// from -1 to 1.
    #[test]
    fn values_go_in_the_models_order_and_scale() {
        let rgb = ramp();
        let mut tensor = Array4::<f32>::zeros((1, 3, 2, 2));
        let identity = Warp([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let values = ModelInput {
            order: super::super::ChannelOrder::Bgr,
            scale: [2.0; 3],
            bias: [-1.0; 3],
        };
        warp_into(&rgb, identity, (2, 2), values, &mut tensor, 0);
        let blue = 7.0 / 255.0 * 2.0 - 1.0;
        let red = 10.0 / 255.0 * 2.0 - 1.0;
        assert!((tensor[[0, 0, 0, 0]] - blue).abs() < 1e-6, "B first");
        assert!((tensor[[0, 2, 0, 1]] - red).abs() < 1e-6, "R last");
    }
}
