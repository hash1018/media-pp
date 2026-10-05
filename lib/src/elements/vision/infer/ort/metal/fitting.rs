//! What the Metal detector and classifier share: the kernels that fit a
//! VideoToolbox picture — or a rectangle of it — into a model's input, and
//! the input they fit it into, in memory the CPU and GPU share.

use objc2_metal::{MTLBuffer, MTLPixelFormat, MTLTextureUsage};

use crate::ffmpeg;
use crate::{
    color::ColorDescription,
    platform::macos::{
        metal::{Buffer, Kernel, MetalGpu, Texture},
        pixel_buffer::PixelBuffer,
        videotoolbox::{NotVideoToolbox, sw_format_of},
    },
};

use super::super::OrtError;

const SHADER: &str = include_str!("../../../../../shaders/metal/fit.metal");

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
/// width, height, inside the picture — scaled to `scaled` and placed at
/// `offset` in input `slot`, the rest grey.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Cut {
    pub(super) crop: (u32, u32, u32, u32),
    pub(super) offset: (u32, u32),
    pub(super) scaled: (u32, u32),
    pub(super) slot: usize,
}

/// The kernels, and `capacity` inputs of a model's size in shared memory —
/// apart from any session, so a test can check them without a model.
pub(super) struct Fitting {
    pub(super) model: (u32, u32),
    capacity: usize,
    gpu: MetalGpu,
    nv12: Kernel,
    bgra: Kernel,
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
        let [nv12, bgra] = <[Kernel; 2]>::try_from(gpu.kernels(SHADER, &["fit_nv12", "fit_bgra"])?)
            .unwrap_or_else(|_| unreachable!("two kernels for two names"));
        let capacity = capacity.max(1);
        let tensor = gpu.shared_buffer(capacity * Self::floats(model) * size_of::<f32>())?;
        Ok(Self {
            model,
            capacity,
            gpu,
            nv12,
            bgra,
            tensor,
        })
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
        affine: ([f32; 3], [f32; 3]),
    ) -> Result<(), OrtError> {
        self.fit_each(&[(picture, cuts)], affine)
    }

    /// [`Self::fit`] for several pictures, each with its own cuts, in one
    /// pass.
    ///
    /// # Panics
    ///
    /// As [`Self::fit`].
    pub(super) fn fit_each(
        &mut self,
        pictures: &[(&Picture, &[Cut])],
        (scale, bias): ([f32; 3], [f32; 3]),
    ) -> Result<(), OrtError> {
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
                let mut parameters: Vec<u8> = [
                    self.model.0,
                    self.model.1,
                    cut.offset.0,
                    cut.offset.1,
                    cut.scaled.0,
                    cut.scaled.1,
                    width,
                    height,
                    left,
                    top,
                    cut.slot as u32,
                    0,
                ]
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect();
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

    /// The RGB the CPU reads at `(x, y)` of `frame`, 0 to 1.
    fn rgb_at(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> [f32; 3] {
        if frame.format() == ffmpeg::format::Pixel::NV12 {
            let rows = ColorDescription::of(frame).yuv_to_rgb_rows(frame.height());
            let luma = frame.data(0)[y * frame.stride(0) + x];
            let chroma = &frame.data(1)[(y / 2) * frame.stride(1) + (x / 2) * 2..];
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
            let bgra = &frame.data(0)[y * frame.stride(0) + x * 4..];
            [bgra[2], bgra[1], bgra[0]].map(|value| f32::from(value) / 255.0)
        }
    }

    /// Boxes of a picture, each stretched into its own input of a batch and
    /// put in ImageNet's scale, are what the CPU reads of them: the sample
    /// under each pixel's centre, from the box's own corner — an odd one on
    /// NV12 too — in the input its slot says, the other inputs untouched.
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
            // Twice the input's size each way, so every centre is an exact
            // sample; the second starts on odd pixels.
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
                        let sx =
                            left as usize + (x * 2 + 1) * width as usize / (2 * model.0 as usize);
                        let sy =
                            top as usize + (y * 2 + 1) * height as usize / (2 * model.1 as usize);
                        let wanted = rgb_at(&source, sx, sy);
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
}
