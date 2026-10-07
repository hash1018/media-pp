//! [`SwCutDetector`]: where one shot ends and the next begins, in pictures
//! in system memory.

use std::sync::Arc;

use ffmpeg_next::{self as ffmpeg, format::Pixel};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, element_pp_log},
    elements::vision::batch::PerStream,
    error::Result,
    transform::{Filter, FilterStage, Output, filter_stage},
};

use super::{CutDetectorOptions, CutDetectorOptionsError, Judge, Thumbnail, cell_means};

/// Errors from an [`SwCutDetector`].
#[derive(Debug, ThisError)]
pub enum SwCutDetectorError {
    /// The options were refused.
    #[error(transparent)]
    Options(#[from] CutDetectorOptionsError),
    /// It was handed something other than a picture in system memory.
    #[error("SwCutDetector takes pictures in system memory, got {0}")]
    UnsupportedBuffer(&'static str),
    /// A picture in a layout it reads through a conversion could not be
    /// converted.
    #[error("converting a {0:?} picture to look at failed: {1}")]
    Convert(Pixel, ffmpeg::Error),
}

/// Finds where one shot of an edited video ends and the next begins, in
/// pictures in system memory, and hands each picture on — the first of
/// each shot after the stream's first carrying a
/// [`SceneCut`](crate::elements::SceneCut) — as
/// [`CutDetectorOptions`] says; see it, and the module, for how.
///
/// It reads 8-bit NV12 and 4:2:0 pictures, and P010, in place; any other
/// layout through a conversion to 4:2:0 at its own size. It holds
/// [`CutDetectorOptions::lookahead`] pictures of each stream, handing each
/// on once the ones after it are seen, and every one still held at the end
/// of the stream; a seek or a flush lets go of them. After a
/// [`StreamMux`](crate::elements::StreamMux) each stream is judged on its
/// own.
pub struct SwCutDetector(FilterStage<Detecting>);

filter_stage!(SwCutDetector);

/// A picture layout's conversion to the 4:2:0 it is read in, at its size.
struct Converting {
    from: (Pixel, u32, u32),
    context: ffmpeg::software::scaling::Context,
    picture: ffmpeg::frame::Video,
}

// SAFETY: the scaling context and frame are owned by this conversion alone,
// used by the one thread transforming at a time, as the detectors' are.
unsafe impl Send for Converting {}

/// What an [`SwCutDetector`] does with each picture.
struct Detecting {
    name: Arc<str>,
    pp_log: PpLog,
    options: CutDetectorOptions,
    judges: PerStream<Judge>,
    converting: Option<Converting>,
}

impl SwCutDetector {
    /// A cut detector judging as `options` say.
    ///
    /// # Errors
    ///
    /// Options it cannot judge with.
    pub fn new(
        name: impl Into<String>,
        options: CutDetectorOptions,
    ) -> std::result::Result<Self, SwCutDetectorError> {
        options.check()?;
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::SwCutDetector, &name, None);
        pp_info!(pp_log: &pp_log, "opened: {options:?}");
        Ok(Self(FilterStage::new(Detecting {
            name,
            pp_log,
            options,
            judges: PerStream::default(),
            converting: None,
        })))
    }
}

/// `frame`'s thumbnail, read from its planes where it is a layout read in
/// place.
pub(super) fn thumbnail_of(frame: &ffmpeg::frame::Video) -> Option<Thumbnail> {
    let (width, height) = (frame.width(), frame.height());
    let (cells, chroma_cells) = Thumbnail::cells_of(width, height);
    let chroma = (width.div_ceil(2), height.div_ceil(2));
    let mut luma = Vec::with_capacity((cells.0 * cells.1) as usize);
    let mut both = Vec::with_capacity((chroma_cells.0 * chroma_cells.1 * 2) as usize);
    // Each layout's luma channel, and its two chroma planes and channels.
    let (luma_bytes, luma_channel, chroma_planes): (u32, u32, [(usize, u32, u32); 2]) =
        match frame.format() {
            Pixel::NV12 => (1, 0, [(1, 2, 0), (1, 2, 1)]),
            // The high byte of each little-endian sample.
            Pixel::P010LE => (2, 1, [(1, 4, 1), (1, 4, 3)]),
            Pixel::YUV420P | Pixel::YUVJ420P => (1, 0, [(1, 1, 0), (2, 1, 0)]),
            _ => return None,
        };
    cell_means(
        frame.data(0),
        frame.stride(0),
        width,
        height,
        luma_bytes,
        luma_channel,
        cells,
        &mut luma,
    );
    let mut planes = [Vec::new(), Vec::new()];
    for ((plane, channels, channel), means) in chroma_planes.into_iter().zip(&mut planes) {
        cell_means(
            frame.data(plane),
            frame.stride(plane),
            chroma.0,
            chroma.1,
            channels,
            channel,
            chroma_cells,
            means,
        );
    }
    for (cb, cr) in planes[0].iter().zip(&planes[1]) {
        both.extend([*cb, *cr]);
    }
    Some(Thumbnail {
        cells,
        luma,
        chroma_cells,
        chroma: both,
    })
}

impl Detecting {
    /// `frame`'s thumbnail, through a conversion where it is not a layout
    /// read in place.
    fn thumbnail(
        &mut self,
        frame: &ffmpeg::frame::Video,
    ) -> std::result::Result<Thumbnail, SwCutDetectorError> {
        if let Some(thumbnail) = thumbnail_of(frame) {
            return Ok(thumbnail);
        }
        let from = (frame.format(), frame.width(), frame.height());
        let converting = match &mut self.converting {
            Some(converting) if converting.from == from => converting,
            slot => {
                let context = ffmpeg::software::scaling::Context::get(
                    from.0,
                    from.1,
                    from.2,
                    Pixel::YUV420P,
                    from.1,
                    from.2,
                    ffmpeg::software::scaling::Flags::POINT,
                )
                .map_err(|error| SwCutDetectorError::Convert(from.0, error))?;
                slot.insert(Converting {
                    from,
                    context,
                    picture: ffmpeg::frame::Video::new(Pixel::YUV420P, from.1, from.2),
                })
            }
        };
        converting
            .context
            .run(frame, &mut converting.picture)
            .map_err(|error| SwCutDetectorError::Convert(from.0, error))?;
        Ok(thumbnail_of(&converting.picture).expect("4:2:0 is read in place"))
    }
}

impl Element for Detecting {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::SwCutDetector
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Filter for Detecting {
    /// Pictures in system memory, any layout.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::VideoFrame,
            MemoryDomain::System,
        ))
    }

    /// The pictures it was handed.
    fn output_contract(&self) -> OutputContract {
        OutputContract::Passthrough
    }

    fn transform(&mut self, buf: MediaBuffer, out: &mut Output) -> Result<()> {
        let MediaBuffer::Video(frame) = &buf else {
            let kind = buf.kind();
            pp_error!(self, "unsupported buffer: {kind}");
            return Err(SwCutDetectorError::UnsupportedBuffer(kind).into());
        };
        if crate::elements::vision::is_hardware(frame.format()) {
            return Err(SwCutDetectorError::UnsupportedBuffer("a hardware video frame").into());
        }
        let thumbnail = self
            .thumbnail(frame)
            .inspect_err(|error| pp_error!(self, "{error}"))?;
        let (name, options) = (&self.name, self.options);
        let (judge, moved) = self
            .judges
            .get(&buf, |_| Judge::new(Arc::clone(name), options));
        if moved {
            // That stream was sought: what it held is from before.
            judge.clear();
        }
        judge.push(buf, thumbnail, out);
        Ok(())
    }

    /// Every picture still held, judged by those after it there are.
    fn drain(&mut self, out: &mut Output) -> Result<()> {
        for judge in self.judges.values_mut() {
            judge.finish(out);
        }
        Ok(())
    }

    /// A seek or a flush: what was held is let go of.
    fn reset(&mut self) {
        self.judges.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::buffer::set_time_base;
    use crate::element::{RawSink, RawSinkExt, SrcPads};
    use crate::elements::{AppSink, SceneCut};
    use crate::stream::StreamEvent;

    fn capture(stage: &mut dyn SrcPads) -> Arc<Mutex<Vec<MediaBuffer>>> {
        let kept = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&kept);
        stage.src_pads()[0].link(Box::new(AppSink::new("kept", move |buf| {
            sink.lock().unwrap().push(buf);
            Ok(())
        })));
        kept
    }

    /// A `format` picture of 128 by 72 whose left half is `left` and right
    /// half `right`, as luma, its chroma grey.
    fn picture(format: Pixel, index: i64, left: u8, right: u8) -> MediaBuffer {
        let mut frame = ffmpeg::frame::Video::new(format, 128, 72);
        let stride = frame.stride(0);
        for row in frame.data_mut(0).chunks_mut(stride).take(72) {
            for (x, byte) in row[..128].iter_mut().enumerate() {
                *byte = if x < 64 { left } else { right };
            }
        }
        for plane in 1..frame.planes() {
            frame.data_mut(plane).fill(128);
        }
        frame.set_pts(Some(index));
        set_time_base(&mut frame, ffmpeg::Rational(1, 30));
        MediaBuffer::video(frame)
    }

    fn cuts(kept: &[MediaBuffer]) -> Vec<usize> {
        kept.iter()
            .enumerate()
            .filter(|(_, buf)| {
                buf.metadata()
                    .is_some_and(|metadata| metadata.get::<SceneCut>().is_some())
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// Two shots, in each layout it reads in place and one it converts:
    /// every picture comes out, in order, the last two at the end of the
    /// stream, and the first of the second shot carries the cut.
    #[test]
    fn the_first_picture_of_a_new_shot_carries_the_cut() {
        for format in [Pixel::NV12, Pixel::YUV420P, Pixel::RGB24] {
            let mut detector = SwCutDetector::new(
                "cuts",
                CutDetectorOptions {
                    min_scene: std::time::Duration::ZERO,
                    ..CutDetectorOptions::default()
                },
            )
            .unwrap();
            let kept = capture(&mut detector);
            for index in 0..20 {
                let (left, right) = if index < 12 { (40, 200) } else { (200, 40) };
                // RGB24 is read through a conversion, as grey.
                let buf = if format == Pixel::RGB24 {
                    let mut frame = ffmpeg::frame::Video::new(Pixel::RGB24, 128, 72);
                    let stride = frame.stride(0);
                    for row in frame.data_mut(0).chunks_mut(stride).take(72) {
                        for (x, pixel) in row[..384].chunks_mut(3).enumerate() {
                            pixel.fill(if x < 64 { left } else { right });
                        }
                    }
                    frame.set_pts(Some(index));
                    set_time_base(&mut frame, ffmpeg::Rational(1, 30));
                    MediaBuffer::video(frame)
                } else {
                    picture(format, index, left, right)
                };
                detector.consume(buf).expect("judged");
            }
            assert_eq!(kept.lock().unwrap().len(), 18, "{format:?}: two held back");
            detector.stream_event(&StreamEvent::Eos).expect("drained");
            let kept = kept.lock().unwrap();
            assert_eq!(kept.len(), 20, "{format:?}");
            assert_eq!(cuts(&kept), vec![12], "{format:?}");
        }
    }

    /// A flush lets go of what was held: none of it comes out after.
    #[test]
    fn a_flush_lets_go_of_what_was_held() {
        let mut detector = SwCutDetector::new("cuts", CutDetectorOptions::default()).unwrap();
        let kept = capture(&mut detector);
        for index in 0..5 {
            detector
                .consume(picture(Pixel::NV12, index, 50, 50))
                .expect("judged");
        }
        detector
            .control(&crate::control::ControlMsg::Flush)
            .expect("flushed");
        detector.stream_event(&StreamEvent::Eos).expect("drained");
        assert_eq!(kept.lock().unwrap().len(), 3, "the two held are gone");
    }

    #[test]
    fn options_it_cannot_judge_with_are_refused() {
        let refused = SwCutDetector::new(
            "cuts",
            CutDetectorOptions {
                threshold: 0.0,
                ..CutDetectorOptions::default()
            },
        );
        assert!(matches!(
            refused,
            Err(SwCutDetectorError::Options(
                CutDetectorOptionsError::Threshold(_)
            ))
        ));
    }
}
