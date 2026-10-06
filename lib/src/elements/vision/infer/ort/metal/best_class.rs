//! Each box's best class of a YOLOv8 or YOLO11 output, found on the GPU.

use objc2_metal::MTLBuffer;

use crate::platform::macos::metal::{Buffer, Kernel, MetalGpu};

use super::super::OrtError;

const SHADER: &str = include_str!("../../../../../shaders/metal/best_class.metal");

/// The kernel, and the memory the CPU and GPU share that it reads an
/// output from and writes six floats a box to — each grown to the largest
/// output seen.
///
/// A `[batch, 4 + classes, boxes]` output read on the CPU had every box's
/// every class compared there — about 3 ms of a 26 ms run for eight
/// pictures of YOLO11n on an M5. ONNX Runtime hands Core ML's output over
/// in its own memory, so it is copied to where the GPU reads it, which on
/// Apple silicon is the same memory and a fraction of the comparing.
pub(super) struct BestClass {
    gpu: MetalGpu,
    kernel: Kernel,
    output: Option<Buffer>,
    best: Option<Buffer>,
}

// SAFETY: the kernel and buffers are used by the one thread transforming at
// a time, through `&mut self`; Metal's objects are thread-safe, and the
// buffers are written only by a pass this waits for before they are read —
// the reasoning `Fitting` gives for its own.
unsafe impl Send for BestClass {}

impl BestClass {
    pub(super) fn new() -> Result<Self, OrtError> {
        let gpu = MetalGpu::new()?;
        let [kernel] = gpu.kernel_array(SHADER, ["best_class"])?;
        Ok(Self {
            gpu,
            kernel,
            output: None,
            best: None,
        })
    }

    /// `buffer`, made again where it holds fewer than `floats` floats.
    fn room(
        gpu: &MetalGpu,
        buffer: &mut Option<Buffer>,
        floats: usize,
    ) -> Result<Buffer, OrtError> {
        let bytes = floats * size_of::<f32>();
        match buffer {
            Some(held) if held.length() >= bytes => Ok(held.clone()),
            slot => Ok(slot.insert(gpu.shared_buffer(bytes)?).clone()),
        }
    }

    /// Each box of `output` — `pictures` outputs of `rows` (four, then a
    /// score per class) by `boxes` floats — as six floats: its centre,
    /// width and height, and its best class's score and number, one picture
    /// after another.
    ///
    /// # Panics
    ///
    /// If `output` holds fewer than `pictures · rows · boxes` floats, or
    /// `rows` is not more than four: the caller's own reading of its shape.
    pub(super) fn find(
        &mut self,
        output: &[f32],
        pictures: usize,
        rows: usize,
        boxes: usize,
    ) -> Result<&[f32], OrtError> {
        let read = pictures * rows * boxes;
        assert!(
            rows > 4 && output.len() >= read,
            "{pictures}x{rows}x{boxes}"
        );
        let source = Self::room(&self.gpu, &mut self.output, read)?;
        let best = Self::room(&self.gpu, &mut self.best, pictures * boxes * 6)?;
        // SAFETY: the buffer is this reader's own, of at least `read` floats
        // in shared memory, and no pass is using it: the last one was
        // waited for.
        unsafe {
            std::ptr::copy_nonoverlapping(
                output.as_ptr(),
                source.contents().as_ptr().cast::<f32>(),
                read,
            );
        }
        let size: Vec<u8> = [rows as u32, boxes as u32, pictures as u32, 0]
            .iter()
            .flat_map(|word| word.to_ne_bytes())
            .collect();
        let mut pass = self.gpu.pass()?;
        pass.dispatch_groups(
            &self.kernel,
            &[],
            &[(&source, 0), (&best, 0)],
            Some(&size),
            (boxes.div_ceil(256), pictures, 1),
            (256, 1, 1),
        );
        pass.finish()?;
        // SAFETY: the buffer holds at least `pictures · boxes · 6` floats,
        // which the pass, finished, has written; nothing writes it again
        // until the next `find`, which takes `&mut self` and so waits for
        // this borrow to end.
        Ok(unsafe {
            std::slice::from_raw_parts(best.contents().as_ptr().cast::<f32>(), pictures * boxes * 6)
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array3;

    use super::super::super::{Letterbox, OrtDetectorOptions, decode_batch, decode_best};
    use super::*;

    /// What the kernel writes, read, is what the whole output reads to on
    /// the CPU — ties to the first class and all — and each box's six
    /// floats are its centre, size, best score and class. Skipped without
    /// a GPU.
    #[test]
    fn best_classes_found_on_the_gpu_read_as_the_whole_output_does() {
        let Ok(mut reader) = BestClass::new() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let (pictures, classes, boxes) = (3, 7, 300);
        let rows = 4 + classes;
        // Scores that tie, peak in every class and fall under the threshold
        // in turn; boxes spread over the input so few overlap.
        let output =
            Array3::from_shape_fn((pictures, rows, boxes), |(picture, row, index)| match row {
                0 => (index % 20) as f32 * 32.0 + 16.0,
                1 => (index / 20) as f32 * 40.0 + 20.0,
                2 | 3 => 12.0 + (picture * 3 + index % 5) as f32,
                _ => {
                    let class = row - 4;
                    ((index * 7 + class * 13 + picture * 5) % 11) as f32 / 10.0
                }
            });
        let floats = output.as_slice().expect("standard layout");
        let best = reader
            .find(floats, pictures, rows, boxes)
            .expect("runs")
            .to_vec();
        for picture in 0..pictures {
            for index in 0..boxes {
                let scores: Vec<f32> = (4..rows).map(|row| output[[picture, row, index]]).collect();
                let (class, score) =
                    scores
                        .iter()
                        .enumerate()
                        .fold((0, scores[0]), |kept, (class, &score)| {
                            if score > kept.1 { (class, score) } else { kept }
                        });
                let at = (picture * boxes + index) * 6;
                assert_eq!(
                    best[at..at + 6],
                    [
                        output[[picture, 0, index]],
                        output[[picture, 1, index]],
                        output[[picture, 2, index]],
                        output[[picture, 3, index]],
                        score,
                        class as f32
                    ],
                    "picture {picture}, box {index}"
                );
            }
        }
        let letterboxes = [
            Letterbox::new((1920, 1080), (640, 640)),
            Letterbox::new((640, 640), (640, 640)),
            Letterbox::new((480, 640), (640, 640)),
        ];
        let options = OrtDetectorOptions::default();
        let whole = decode_batch(output.view().into_dyn(), &letterboxes, &options).unwrap();
        assert!(whole.iter().all(|found| !found.is_empty()));
        assert_eq!(decode_best(&best, boxes, &letterboxes, &options), whole);
    }
}
