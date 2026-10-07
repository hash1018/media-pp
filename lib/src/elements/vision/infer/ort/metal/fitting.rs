//! What the Metal detector, classifier and embedder share: the kernels that
//! fit a VideoToolbox picture — or a rectangle of it, or an object through
//! an affine map — into a model's input, and the input they fit it into, in
//! memory the CPU and GPU share.

use objc2_metal::{MTLBuffer, MTLPixelFormat, MTLTextureUsage};

use crate::ffmpeg;
use crate::{
    color::ColorDescription,
    elements::{VideoToolboxDevice, VideoToolboxFrameFormat, VideoToolboxFramePool},
    platform::macos::{
        metal::{Buffer, Kernel, MetalGpu, Texture},
        pixel_buffer::PixelBuffer,
        videotoolbox::{NotVideoToolbox, sw_format_of},
    },
    pool::UnboundObjectPoolRef,
};

use super::super::{ChannelOrder, ModelInput, OrtError};
use crate::orientation::Orientation;

const SHADER: &str = include_str!("../../../../../shaders/metal/fit.metal");
const TONE_MAP: &str = include_str!("../../../../../shaders/metal/tone_map.metal");

/// A VideoToolbox picture a [`Fitting`] can read: NV12 or BGRA, its pixel
/// buffer held, no smaller than the picture says it is.
pub(super) struct Picture {
    buffer: PixelBuffer,
    nv12: bool,
    /// Its width and height.
    pub(super) size: (u32, u32),
    /// NV12 only: the rows its own colour description makes RGB with.
    rows: [[f32; 4]; 3],
}

impl Picture {
    /// `frame`, where it is a picture a fitting reads.
    pub(super) fn of(frame: &ffmpeg::frame::Video) -> Result<Self, OrtError> {
        let nv12 = match sw_format_of(frame) {
            Ok(ffmpeg::format::Pixel::NV12) => true,
            Ok(ffmpeg::format::Pixel::BGRA) => false,
            Ok(other) | Err(NotVideoToolbox::Format(other)) => {
                return Err(OrtError::UnsupportedPicture(other));
            }
            Err(NotVideoToolbox::NoFramesContext) => {
                return Err(OrtError::MissingPixelBuffer("frames context"));
            }
        };
        let buffer = PixelBuffer::of_frame(frame).ok_or(OrtError::MissingPixelBuffer("pixels"))?;
        let size = (frame.width(), frame.height());
        let surface = buffer.size();
        if size.0 > surface.0 || size.1 > surface.1 {
            return Err(OrtError::PictureOutsideSurface {
                picture: size,
                surface,
            });
        }
        let rows = if nv12 {
            ColorDescription::of(frame).yuv_to_rgb_rows(frame.height())
        } else {
            [[0.0; 4]; 3]
        };
        Ok(Self {
            buffer,
            nv12,
            size,
            rows,
        })
    }
}

/// One rectangle of a picture fitted into one input: `crop` — left, top,
/// width, height, inside the picture as stored — turned as `orientation`
/// says, scaled to `scaled` and placed at `offset` in input `slot`, the
/// rest grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Cut {
    pub(super) crop: (u32, u32, u32, u32),
    pub(super) orientation: Orientation,
    pub(super) offset: (u32, u32),
    pub(super) scaled: (u32, u32),
    pub(super) slot: usize,
}

/// What brings an HDR picture to SDR for a model: the kernel, and the BGRA
/// pictures it writes into, of one size at a time.
struct SdrCopy {
    kernel: Kernel,
    /// Its own context: a pixel buffer belongs to none, so the pictures it
    /// makes are read on whichever the rest of the pipeline shares.
    device: VideoToolboxDevice,
    pool: Option<VideoToolboxFramePool>,
}

impl SdrCopy {
    fn new(gpu: &MetalGpu) -> Result<Self, OrtError> {
        let [kernel] = gpu.kernel_array(TONE_MAP, ["hdr_to_bgra"])?;
        let device = VideoToolboxDevice::new()
            .map_err(|error| OrtError::UnsupportedModel(format!("no VideoToolbox: {error}")))?;
        Ok(Self {
            kernel,
            device,
            pool: None,
        })
    }

    /// `frame`, a P010 picture, brought to SDR BGRA as `tone_map` says, in
    /// a picture of its own pool, written and waited for.
    fn of(
        &mut self,
        gpu: &MetalGpu,
        frame: &ffmpeg::frame::Video,
        tone_map: &crate::tone_map::ToneMap,
    ) -> Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, OrtError> {
        let size = (frame.width(), frame.height());
        if self.pool.as_ref().map(|pool| (pool.width(), pool.height())) != Some(size) {
            self.pool = Some(
                VideoToolboxFramePool::new(
                    &self.device,
                    VideoToolboxFrameFormat::Bgra,
                    size.0,
                    size.1,
                )
                .map_err(|error| OrtError::UnsupportedModel(format!("no SDR copy: {error}")))?,
            );
        }
        let copy = self
            .pool
            .as_ref()
            .expect("made above")
            .get()
            .map_err(|error| OrtError::UnsupportedModel(format!("no SDR copy: {error}")))?;
        let from = PixelBuffer::of_frame(frame).ok_or(OrtError::MissingPixelBuffer("pixels"))?;
        let to = PixelBuffer::of_frame(&copy).ok_or(OrtError::MissingPixelBuffer("pixels"))?;
        let surface = from.size();
        if size.0 > surface.0 || size.1 > surface.1 {
            return Err(OrtError::PictureOutsideSurface {
                picture: size,
                surface,
            });
        }
        let read = MTLTextureUsage::ShaderRead;
        let textures = [
            gpu.plane(&from, 0, MTLPixelFormat::R16Unorm, read)?,
            gpu.plane(&from, 1, MTLPixelFormat::RG16Unorm, read)?,
            gpu.plane(
                &to,
                0,
                MTLPixelFormat::BGRA8Unorm,
                MTLTextureUsage::ShaderWrite,
            )?,
        ];
        // As the shader's `ToneMap`: three rows, the transfer and the EETF's
        // three numbers, three rows of the gamut, then the size.
        let mut bytes: Vec<u8> = tone_map
            .rows
            .iter()
            .flatten()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        bytes.extend(tone_map.transfer.to_ne_bytes());
        for value in [tone_map.source_peak_pq, tone_map.target_peak, tone_map.knee] {
            bytes.extend(value.to_ne_bytes());
        }
        bytes.extend(
            tone_map
                .gamut
                .iter()
                .flatten()
                .flat_map(|value| value.to_ne_bytes()),
        );
        for word in [size.0, size.1, 0, 0] {
            bytes.extend(word.to_ne_bytes());
        }
        let bound: Vec<&Texture> = textures.iter().collect();
        let mut pass = gpu.pass()?;
        pass.dispatch(&self.kernel, &bound, Some(&bytes), size);
        pass.finish()?;
        let mut copy = copy;
        // What it now holds: SDR, as the copy's colour says.
        // SAFETY: two live, distinct frames.
        unsafe { ffmpeg::ffi::av_frame_copy_props(copy.as_mut_ptr(), frame.as_ptr()) };
        crate::color::ColorDescription {
            space: ffmpeg::color::Space::RGB,
            range: ffmpeg::color::Range::JPEG,
            ..crate::color::ColorDescription::BT709_LIMITED
        }
        .describe(&mut copy);
        Ok(copy)
    }
}

/// The kernels, and `capacity` inputs of a model's size in shared memory —
/// apart from any session, so a test can check them without a model.
pub(super) struct Fitting {
    pub(super) model: (u32, u32),
    capacity: usize,
    gpu: MetalGpu,
    nv12: Kernel,
    bgra: Kernel,
    warp_nv12: Kernel,
    warp_bgra: Kernel,
    /// Where an HDR picture is brought to SDR for the model: made with the
    /// first such picture.
    sdr: Option<SdrCopy>,
    /// The inputs, each three planes of `f32`, which the kernels write and
    /// a session reads.
    tensor: Buffer,
}

// SAFETY: the kernels and buffer are used by the one thread transforming at
// a time, through `&mut self`; Metal's objects are thread-safe, and the
// buffer is written only by a pass this waits for before it is read — the
// reasoning `MetalPass` gives for its own.
unsafe impl Send for Fitting {}

impl Fitting {
    /// The kernels, and `capacity` inputs of `model`'s size.
    pub(super) fn new(model: (u32, u32), capacity: usize) -> Result<Self, OrtError> {
        let gpu = MetalGpu::new()?;
        let [nv12, bgra, warp_nv12, warp_bgra] =
            gpu.kernel_array(SHADER, ["fit_nv12", "fit_bgra", "warp_nv12", "warp_bgra"])?;
        let capacity = capacity.max(1);
        let tensor = gpu.shared_buffer(capacity * Self::floats(model) * size_of::<f32>())?;
        Ok(Self {
            model,
            capacity,
            gpu,
            nv12,
            bgra,
            warp_nv12,
            warp_bgra,
            sdr: None,
            tensor,
        })
    }

    /// `frame` as a picture to fit: as it is, or — an HDR P010 picture,
    /// HLG or PQ — an SDR BGRA copy of it, brought down by
    /// `core/tone_map.rs`'s definition, as the CUDA detectors look at HDR;
    /// a model was trained on SDR pictures. What is found goes on `frame`
    /// itself.
    ///
    /// # Errors
    ///
    /// [`OrtError::UnsupportedPicture`] for a P010 picture tagged neither
    /// HLG nor PQ, which has no tone map, and what [`Picture::of`] refuses.
    pub(super) fn picture(&mut self, frame: &ffmpeg::frame::Video) -> Result<Picture, OrtError> {
        if sw_format_of(frame) != Ok(ffmpeg::format::Pixel::P010LE) {
            return Picture::of(frame);
        }
        let tone_map = crate::tone_map::ToneMap::of_frame(frame)
            .ok_or(OrtError::UnsupportedPicture(ffmpeg::format::Pixel::P010LE))?;
        let sdr = match &mut self.sdr {
            Some(sdr) => sdr,
            empty => empty.insert(SdrCopy::new(&self.gpu)?),
        };
        let copy = sdr.of(&self.gpu, frame, &tone_map)?;
        Picture::of(&copy)
    }

    /// How many `f32`s one input holds.
    fn floats(model: (u32, u32)) -> usize {
        3 * model.0 as usize * model.1 as usize
    }

    /// How many inputs it holds.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Fits each of `cuts` of `picture` into its input, each channel then
    /// times `scale` plus `bias`, in one pass on the GPU, and waits for it.
    ///
    /// # Panics
    ///
    /// If a cut's slot is past the capacity, or its crop is not inside the
    /// picture: the callers' own arithmetic.
    pub(super) fn fit(
        &mut self,
        picture: &Picture,
        cuts: &[Cut],
        (scale, bias): ([f32; 3], [f32; 3]),
    ) -> Result<(), OrtError> {
        let values = ModelInput {
            scale,
            bias,
            ..ModelInput::default()
        };
        self.fit_each(&[(picture, cuts)], values)
    }

    /// [`Self::fit`] for several pictures, each with its own cuts, in one
    /// pass, into the values a model wants — its planes in its own order,
    /// each then times its scale plus its bias.
    ///
    /// # Panics
    ///
    /// As [`Self::fit`].
    pub(super) fn fit_each(
        &mut self,
        pictures: &[(&Picture, &[Cut])],
        values: ModelInput,
    ) -> Result<(), OrtError> {
        let ModelInput { scale, bias, order } = values;
        let read = MTLTextureUsage::ShaderRead;
        // Every picture's planes, held until the pass has finished reading
        // them.
        let mut planes: Vec<(&Kernel, Vec<Texture>)> = Vec::with_capacity(pictures.len());
        for &(picture, _) in pictures {
            planes.push(if picture.nv12 {
                (
                    &self.nv12,
                    vec![
                        self.gpu
                            .plane(&picture.buffer, 0, MTLPixelFormat::R8Unorm, read)?,
                        self.gpu
                            .plane(&picture.buffer, 1, MTLPixelFormat::RG8Unorm, read)?,
                    ],
                )
            } else {
                (
                    &self.bgra,
                    vec![
                        self.gpu
                            .plane(&picture.buffer, 0, MTLPixelFormat::BGRA8Unorm, read)?,
                    ],
                )
            });
        }
        let mut pass = self.gpu.pass()?;
        pass.bind_buffer(&self.tensor, 1);
        for (&(picture, cuts), (kernel, textures)) in pictures.iter().zip(&planes) {
            let bound: Vec<&Texture> = textures.iter().collect();
            for cut in cuts {
                let (left, top, width, height) = cut.crop;
                assert!(
                    cut.slot < self.capacity,
                    "input {} of {}",
                    cut.slot,
                    self.capacity
                );
                assert!(
                    width > 0
                        && height > 0
                        && left + width <= picture.size.0
                        && top + height <= picture.size.1,
                    "{:?} inside a {:?} picture",
                    cut.crop,
                    picture.size
                );
                let shown = cut.orientation.display_size(width, height);
                let [xx, xy, x0, yx, yy, y0] = cut.orientation.sampling(width, height);
                let mut parameters: Vec<u8> = [
                    self.model.0,
                    self.model.1,
                    cut.offset.0,
                    cut.offset.1,
                    cut.scaled.0,
                    cut.scaled.1,
                    shown.0,
                    shown.1,
                    left,
                    top,
                    cut.slot as u32,
                    u32::from(order == ChannelOrder::Bgr),
                ]
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect();
                // `x_of` and `y_of`, each an `int4`, after the twelve words
                // — 48 bytes, where an `int4` may start.
                parameters.extend(
                    [xx, xy, x0, 0, yx, yy, y0, 0]
                        .iter()
                        .flat_map(|word| word.to_ne_bytes()),
                );
                let affine = [
                    [scale[0], scale[1], scale[2], 1.0],
                    [bias[0], bias[1], bias[2], 0.0],
                ];
                parameters.extend(
                    picture
                        .rows
                        .iter()
                        .chain(&affine)
                        .flatten()
                        .flat_map(|value| value.to_ne_bytes()),
                );
                pass.dispatch(kernel, &bound, Some(&parameters), self.model);
            }
        }
        pass.finish()?;
        Ok(())
    }

    /// Reads `picture` through each of `maps` into the input of its index,
    /// in the model's `values`, in one pass on the GPU, and waits for it:
    /// input pixel `(u, v)` from `(m[0] u + m[1] v + m[2], m[3] u + m[4] v +
    /// m[5])` of the picture, between the four pixels around it, black
    /// outside it — what the CUDA embedder's `warp_nv12` and `warp_bgra`
    /// read.
    ///
    /// # Panics
    ///
    /// If there are more maps than inputs: the caller's own arithmetic.
    pub(super) fn warp(
        &mut self,
        picture: &Picture,
        maps: &[[f32; 6]],
        values: ModelInput,
    ) -> Result<(), OrtError> {
        assert!(
            maps.len() <= self.capacity,
            "{} maps for {} inputs",
            maps.len(),
            self.capacity
        );
        let ModelInput { scale, bias, order } = values;
        let read = MTLTextureUsage::ShaderRead;
        let (kernel, textures) = if picture.nv12 {
            (
                &self.warp_nv12,
                vec![
                    self.gpu
                        .plane(&picture.buffer, 0, MTLPixelFormat::R8Unorm, read)?,
                    self.gpu
                        .plane(&picture.buffer, 1, MTLPixelFormat::RG8Unorm, read)?,
                ],
            )
        } else {
            (
                &self.warp_bgra,
                vec![
                    self.gpu
                        .plane(&picture.buffer, 0, MTLPixelFormat::BGRA8Unorm, read)?,
                ],
            )
        };
        let bound: Vec<&Texture> = textures.iter().collect();
        let mut pass = self.gpu.pass()?;
        pass.bind_buffer(&self.tensor, 1);
        for (slot, map) in maps.iter().enumerate() {
            let mut parameters: Vec<u8> = [
                self.model.0,
                self.model.1,
                picture.size.0,
                picture.size.1,
                slot as u32,
                u32::from(order == ChannelOrder::Bgr),
                0,
                0,
            ]
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .collect();
            // From 32 bytes, where a `float4` may start: the map's two rows,
            // the colour rows, then the model's scale and bias.
            let floats = [
                [map[0], map[1], map[2], 0.0],
                [map[3], map[4], map[5], 0.0],
                picture.rows[0],
                picture.rows[1],
                picture.rows[2],
                [scale[0], scale[1], scale[2], 1.0],
                [bias[0], bias[1], bias[2], 0.0],
            ];
            parameters.extend(
                floats
                    .iter()
                    .flatten()
                    .flat_map(|value| value.to_ne_bytes()),
            );
            pass.dispatch(kernel, &bound, Some(&parameters), self.model);
        }
        pass.finish()?;
        Ok(())
    }

    /// The first `inputs` inputs, as the last [`Self::fit`] left them: each
    /// R, G and B planes of the model's width by height.
    ///
    /// # Panics
    ///
    /// If `inputs` is past the capacity.
    pub(super) fn input(&self, inputs: usize) -> &[f32] {
        assert!(inputs <= self.capacity, "{inputs} of {}", self.capacity);
        // SAFETY: the buffer is this fitting's own, `capacity` inputs of
        // `f32` in shared memory, of which `inputs` are asked for, and the
        // pass that wrote it has finished; nothing writes it again until the
        // next `fit`, which takes `&mut self` and so waits for this borrow to
        // end.
        unsafe {
            std::slice::from_raw_parts(
                self.tensor.contents().as_ptr().cast::<f32>(),
                inputs * Self::floats(self.model),
            )
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::buffer::MediaBuffer;
    use crate::element::{RawSink, SrcPads};
    use crate::elements::{AppSink, InputScale, VideoToolboxDevice, VideoToolboxUpload};
    use crate::test_support::try_videotoolbox_device;

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
    pub(crate) fn upload(device: &VideoToolboxDevice, frame: ffmpeg::frame::Video) -> MediaBuffer {
        let mut upload = VideoToolboxUpload::new("upload", device);
        let uploaded = capture(&mut upload);
        upload.consume(MediaBuffer::video(frame)).expect("uploads");
        uploaded.lock().unwrap().remove(0)
    }

    /// A `format` picture in system memory in which no sample is its
    /// neighbour's, so a kernel reading the wrong one is caught.
    pub(crate) fn pattern(
        format: ffmpeg::format::Pixel,
        width: u32,
        height: u32,
    ) -> ffmpeg::frame::Video {
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

    /// The source pixels output pixel `d` of `scaled` covers, of `source`:
    /// `[lo, hi)`, as the kernels find them.
    pub(crate) fn covered(d: u32, source: u32, scaled: u32) -> (usize, usize) {
        let lo = d * source / scaled;
        let hi = ((d + 1) * source).div_ceil(scaled).min(source).max(lo + 1);
        (lo as usize, hi as usize)
    }

    /// The RGB the kernels are to make of the pixels `xs` by `ys` of
    /// `frame`, 0 to 1: their mean — of Y', Cb and Cr, made RGB by the
    /// frame's own rows, for NV12.
    pub(crate) fn mean_rgb(
        frame: &ffmpeg::frame::Video,
        xs: std::ops::Range<usize>,
        ys: std::ops::Range<usize>,
    ) -> [f32; 3] {
        let count = (xs.len() * ys.len()) as f32;
        let mut sum = [0.0f32; 3];
        for y in ys {
            for x in xs.clone() {
                let sample = if frame.format() == ffmpeg::format::Pixel::NV12 {
                    let chroma = &frame.data(1)[(y / 2) * frame.stride(1) + (x / 2) * 2..];
                    [frame.data(0)[y * frame.stride(0) + x], chroma[0], chroma[1]]
                } else {
                    let bgra = &frame.data(0)[y * frame.stride(0) + x * 4..];
                    [bgra[2], bgra[1], bgra[0]]
                };
                for (sum, value) in sum.iter_mut().zip(sample) {
                    *sum += f32::from(value) / 255.0;
                }
            }
        }
        let mean = sum.map(|sum| sum / count);
        if frame.format() == ffmpeg::format::Pixel::NV12 {
            let rows = ColorDescription::of(frame).yuv_to_rgb_rows(frame.height());
            let yuv = [mean[0], mean[1], mean[2], 1.0];
            rows.map(|row| {
                row.iter()
                    .zip(yuv)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    .clamp(0.0, 1.0)
            })
        } else {
            mean
        }
    }

    /// Boxes of a picture, each stretched into its own input of a batch and
    /// put in ImageNet's scale, are what the CPU reads of them: the mean of
    /// the pixels each input pixel covers, from the box's own corner — an
    /// odd one on NV12 too — in the input its slot says, the other inputs
    /// untouched.
    #[test]
    fn boxes_are_cut_into_their_inputs_and_scaled() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let model = (16, 16);
        let mut fitting = Fitting::new(model, 3).expect("the kernels compile");
        let (scale, bias) = InputScale::ImageNet.affine();
        for format in [ffmpeg::format::Pixel::NV12, ffmpeg::format::Pixel::BGRA] {
            let source = pattern(format, 96, 64);
            // Twice the input's size each way, so each input pixel covers
            // two by two; the second starts on odd pixels.
            let crops = [(4, 6, 32, 32), (33, 17, 32, 32)];
            let MediaBuffer::Video(frame) = upload(&device, source.clone()) else {
                panic!("a picture is uploaded");
            };
            let picture = Picture::of(&frame).expect("readable");
            let cuts: Vec<Cut> = crops
                .iter()
                .enumerate()
                .map(|(slot, &crop)| Cut {
                    crop,
                    orientation: Orientation::UPRIGHT,
                    offset: (0, 0),
                    scaled: model,
                    slot: slot * 2,
                })
                .collect();
            fitting.fit(&picture, &cuts, (scale, bias)).expect("fits");
            let input = fitting.input(3);
            let plane = (model.0 * model.1) as usize;
            for (cut, (left, top, width, height)) in cuts.iter().zip(crops) {
                for y in 0..model.1 as usize {
                    for x in 0..model.0 as usize {
                        let (x0, x1) = covered(x as u32, width, model.0);
                        let (y0, y1) = covered(y as u32, height, model.1);
                        let (left, top) = (left as usize, top as usize);
                        let wanted = mean_rgb(&source, left + x0..left + x1, top + y0..top + y1);
                        for channel in 0..3 {
                            let at =
                                cut.slot * 3 * plane + channel * plane + y * model.0 as usize + x;
                            let wanted = wanted[channel] * scale[channel] + bias[channel];
                            assert!(
                                (input[at] - wanted).abs() < 5e-3,
                                "{format:?} slot {}: ({x}, {y}) channel {channel} is {}, not {wanted}",
                                cut.slot,
                                input[at]
                            );
                        }
                    }
                }
            }
        }
    }

    /// A picture shrunk to a third is read from every pixel, not one in
    /// nine: of every third column white, each input pixel is a third
    /// white, where the sample at its centre read it all white. The
    /// regression the CUDA kernels have too.
    #[test]
    fn a_shrunk_picture_is_the_mean_of_what_it_covers() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let model = (32, 16);
        let mut fitting = Fitting::new(model, 1).expect("the kernels compile");
        let mut source = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 96, 48);
        let stride = source.stride(0);
        for row in source.data_mut(0).chunks_mut(stride).take(48) {
            for (x, pixel) in row[..96 * 4].chunks_mut(4).enumerate() {
                let value = if x % 3 == 1 { 255 } else { 0 };
                pixel.copy_from_slice(&[value, value, value, 255]);
            }
        }
        let MediaBuffer::Video(frame) = upload(&device, source) else {
            panic!("a picture is uploaded");
        };
        let picture = Picture::of(&frame).expect("readable");
        let cut = Cut {
            crop: (0, 0, 96, 48),
            orientation: Orientation::UPRIGHT,
            offset: (0, 0),
            scaled: model,
            slot: 0,
        };
        fitting
            .fit(&picture, &[cut], ([1.0; 3], [0.0; 3]))
            .expect("fits");
        for (index, value) in fitting.input(1).iter().enumerate() {
            assert!(
                (value - 1.0 / 3.0).abs() < 1e-3,
                "float {index} is {value}, not a third"
            );
        }
    }

    /// A picture stored turned is fitted the way it is shown: whichever of
    /// the eight ways it is stored, the model is handed what the upright
    /// picture gives it — as the CUDA kernels are held to.
    #[test]
    fn a_turned_picture_is_fitted_the_way_it_is_shown() {
        use crate::orientation::Rotation;
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let model = (16, 16);
        let mut fitting = Fitting::new(model, 1).expect("the kernels compile");
        // Wider than tall, shrunk by three and by two, no two pixels alike.
        let size = (48u32, 32u32);
        let shown = |x: u32, y: u32| ((x * 5 + y * 11) % 251) as u8;
        let mut fitted = |orientation: Orientation, bgra: bool| {
            let stored = orientation.display_size(size.0, size.1);
            let [xx, xy, x0, yx, yy, y0] = orientation.sampling(stored.0, stored.1);
            let format = if bgra {
                ffmpeg::format::Pixel::BGRA
            } else {
                ffmpeg::format::Pixel::NV12
            };
            let mut frame = ffmpeg::frame::Video::new(format, stored.0, stored.1);
            if !bgra {
                frame.data_mut(1).fill(128);
            }
            let stride = frame.stride(0);
            for y in 0..size.1 as i32 {
                for x in 0..size.0 as i32 {
                    let (sx, sy) = (
                        (xx * x + xy * y + x0) as usize,
                        (yx * x + yy * y + y0) as usize,
                    );
                    let value = shown(x as u32, y as u32);
                    if bgra {
                        frame.data_mut(0)[sy * stride + sx * 4..][..4]
                            .copy_from_slice(&[value, value, value, 255]);
                    } else {
                        frame.data_mut(0)[sy * stride + sx] = value;
                    }
                }
            }
            let MediaBuffer::Video(frame) = upload(&device, frame) else {
                panic!("a picture is uploaded");
            };
            let picture = Picture::of(&frame).expect("readable");
            let cut = Cut {
                crop: (0, 0, stored.0, stored.1),
                orientation,
                offset: (0, 0),
                scaled: model,
                slot: 0,
            };
            fitting
                .fit(&picture, &[cut], ([1.0; 3], [0.0; 3]))
                .expect("fits");
            fitting.input(1).to_vec()
        };
        for bgra in [false, true] {
            let upright = fitted(Orientation::UPRIGHT, bgra);
            for rotation in [
                Rotation::None,
                Rotation::Clockwise90,
                Rotation::Half,
                Rotation::Clockwise270,
            ] {
                for mirrored in [false, true] {
                    let orientation = Orientation { rotation, mirrored };
                    let turned = fitted(orientation, bgra);
                    for (index, (turned, upright)) in turned.iter().zip(&upright).enumerate() {
                        assert!(
                            (turned - upright).abs() < 1e-6,
                            "bgra={bgra}, {orientation:?}: float {index} is {turned}, not {upright}"
                        );
                    }
                }
            }
        }
    }

    fn warp_pattern(x: usize, y: usize) -> u8 {
        ((x * 5 + y * 3) % 200 + 20) as u8
    }

    /// The warp kernels read each input pixel from where the map says,
    /// between the four pixels around it, black outside the picture — what
    /// the CPU embedder samples — through a turn and a scale, from NV12 and
    /// BGRA, into the input of the map's place; and in the model's order and
    /// values.
    #[test]
    fn the_warp_kernels_read_through_the_map() {
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let model = (12, 10);
        let mut fitting = Fitting::new(model, 2).expect("the kernels compile");
        // A third of a quarter turn, scaled by 2.5, moved so that part of
        // the input falls off the picture's left and top.
        let (c, d) = (2.5 * 0.5f32.cos(), 2.5 * 0.5f32.sin());
        let map = [c, -d, 3.0, d, c, -4.0];
        let expected = |u: usize, v: usize| -> f32 {
            let (u, v) = (u as f32, v as f32);
            let (x, y) = (
                map[0] * u + map[1] * v + map[2],
                map[3] * u + map[4] * v + map[5],
            );
            let (x0, y0) = (x.floor(), y.floor());
            let (fx, fy) = (x - x0, y - y0);
            let mut sum = 0.0;
            for (dx, dy, w) in [
                (0.0, 0.0, (1.0 - fx) * (1.0 - fy)),
                (1.0, 0.0, fx * (1.0 - fy)),
                (0.0, 1.0, (1.0 - fx) * fy),
                (1.0, 1.0, fx * fy),
            ] {
                let (px, py) = (x0 + dx, y0 + dy);
                if px >= 0.0 && py >= 0.0 && px < 40.0 && py < 30.0 {
                    sum += w * f32::from(warp_pattern(px as usize, py as usize));
                }
            }
            sum / 255.0
        };
        let plane = model.0 * model.1;
        for format in [ffmpeg::format::Pixel::BGRA, ffmpeg::format::Pixel::NV12] {
            let mut frame = ffmpeg::frame::Video::new(format, 40, 30);
            let stride = frame.stride(0);
            for y in 0..30 {
                for x in 0..40 {
                    let v = warp_pattern(x, y);
                    if format == ffmpeg::format::Pixel::BGRA {
                        frame.data_mut(0)[y * stride + x * 4..][..4]
                            .copy_from_slice(&[v, v, v, 255]);
                    } else {
                        frame.data_mut(0)[y * stride + x] = v;
                    }
                }
            }
            if format == ffmpeg::format::Pixel::NV12 {
                frame.data_mut(1).fill(128);
                // Full range, so that a neutral chroma leaves R = G = B = Y'.
                frame.set_color_range(ffmpeg::color::Range::JPEG);
                frame.set_color_space(ffmpeg::color::Space::BT709);
            }
            let MediaBuffer::Video(frame) = upload(&device, frame) else {
                panic!("a picture is uploaded");
            };
            let picture = Picture::of(&frame).expect("readable");
            // The first input left grey by a fit; the map into the second.
            let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            fitting
                .warp(&picture, &[identity, map], ModelInput::default())
                .expect("warps");
            let input = fitting.input(2);
            let mut outside = 0;
            for v in 0..model.1 as usize {
                for u in 0..model.0 as usize {
                    let want = expected(u, v);
                    outside += usize::from(want == 0.0);
                    for channel in 0..3 {
                        let got = input[(3 + channel) * plane as usize + v * model.0 as usize + u];
                        assert!(
                            (got - want).abs() < 2.0 / 255.0,
                            "{format:?} channel {channel}, ({u}, {v}): {got} against {want}"
                        );
                    }
                }
            }
            assert!(outside > 0, "some of the input is off the picture");
            // The identity reads the picture's own top-left pixels.
            assert!((input[0] - f32::from(warp_pattern(0, 0)) / 255.0).abs() < 1e-3);

            // BGR, times two less one: the planes swapped, then each scaled.
            let values = ModelInput {
                order: ChannelOrder::Bgr,
                scale: [2.0, 3.0, 4.0],
                bias: [-1.0, -2.0, -3.0],
            };
            fitting.warp(&picture, &[identity], values).expect("warps");
            let input = fitting.input(1);
            let want = f32::from(warp_pattern(0, 0)) / 255.0;
            for (channel, (scale, bias)) in [(2.0, -1.0), (3.0, -2.0), (4.0, -3.0)]
                .into_iter()
                .enumerate()
            {
                let got = input[channel * plane as usize];
                assert!(
                    (got - (want * scale + bias)).abs() < 1e-2,
                    "{format:?} plane {channel}: {got}"
                );
            }
        }
    }

    /// An HDR picture, HLG or PQ, is looked at through an SDR copy that is
    /// `core/tone_map.rs`'s definition, pixel for pixel to within the
    /// rounding of a GPU's powers; a P010 picture tagged neither is refused,
    /// having no tone map.
    #[test]
    fn an_hdr_picture_is_looked_at_in_sdr() {
        use crate::tone_map::ToneMap;
        let Some(device) = try_videotoolbox_device() else {
            return;
        };
        let mut fitting = Fitting::new((16, 16), 1).expect("the kernels compile");
        let (width, height) = (64usize, 32usize);
        // Ten-bit luma across the range, two chroma pairs off grey.
        let luma = |x: usize, y: usize| (64 + (x * 13 + y * 7) % 877) as u16;
        let chroma = |x: usize, y: usize| -> (u16, u16) {
            match (x + y) % 3 {
                0 => (512, 512),
                1 => (400, 620),
                _ => (600, 450),
            }
        };
        for transfer in [
            ffmpeg::color::TransferCharacteristic::ARIB_STD_B67,
            ffmpeg::color::TransferCharacteristic::SMPTE2084,
            ffmpeg::color::TransferCharacteristic::BT709,
        ] {
            let mut frame = ffmpeg::frame::Video::new(
                ffmpeg::format::Pixel::P010LE,
                width as u32,
                height as u32,
            );
            let (stride, cstride) = (frame.stride(0), frame.stride(1));
            for y in 0..height {
                for x in 0..width {
                    frame.data_mut(0)[y * stride + x * 2..][..2]
                        .copy_from_slice(&(luma(x, y) << 6).to_le_bytes());
                }
            }
            for y in 0..height / 2 {
                for x in 0..width / 2 {
                    let (cb, cr) = chroma(x, y);
                    let at = y * cstride + x * 4;
                    frame.data_mut(1)[at..at + 2].copy_from_slice(&(cb << 6).to_le_bytes());
                    frame.data_mut(1)[at + 2..at + 4].copy_from_slice(&(cr << 6).to_le_bytes());
                }
            }
            crate::color::ColorDescription {
                space: ffmpeg::color::Space::BT2020NCL,
                range: ffmpeg::color::Range::MPEG,
                primaries: ffmpeg::color::Primaries::BT2020,
                transfer,
            }
            .describe(&mut frame);
            let MediaBuffer::Video(uploaded) = upload(&device, frame.clone()) else {
                panic!("a picture is uploaded");
            };
            let Some(map) = ToneMap::of_frame(&frame) else {
                assert!(
                    matches!(
                        fitting.picture(&uploaded),
                        Err(OrtError::UnsupportedPicture(ffmpeg::format::Pixel::P010LE))
                    ),
                    "SDR P010 has no tone map"
                );
                continue;
            };
            let picture = fitting.picture(&uploaded).expect("an SDR copy");
            assert!(!picture.nv12, "the copy is BGRA");
            let sdr = SdrCopy::new(&fitting.gpu)
                .expect("the kernel compiles")
                .of(&fitting.gpu, &uploaded, &map)
                .expect("an SDR copy");
            let mut download = crate::elements::VideoToolboxDownload::new("download");
            let downloaded = capture(&mut download);
            download
                .consume(MediaBuffer::Video(Arc::new(sdr).into()))
                .expect("downloads");
            let MediaBuffer::Video(copy) = downloaded.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            let mut worst = 0;
            for y in 0..height {
                for x in 0..width {
                    let (cb, cr) = chroma(x / 2, y / 2);
                    let unit = |code: u16| f32::from(code << 6) / 65535.0;
                    let want = map.apply(unit(luma(x, y)), unit(cb), unit(cr));
                    let got = &copy.data(0)[y * copy.stride(0) + x * 4..][..4];
                    for (got, want) in [got[2], got[1], got[0]].iter().zip(want) {
                        worst = worst.max(got.abs_diff(want));
                    }
                }
            }
            assert!(worst <= 2, "{transfer:?}: off by {worst} of 255");
        }
    }
}
