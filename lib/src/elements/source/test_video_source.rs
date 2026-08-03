use std::sync::Arc;

use ffmpeg_next as ffmpeg;
use rust_hlog::hinfo;
use thiserror::Error as ThisError;

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_hlog},
    pad::SrcPad,
    pool::UnboundObjectPool,
};

/// Errors specific to `TestVideoSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum TestVideoSourceError {
    #[error("TestVideoSource doesn't support seeking a generated stream")]
    SeekUnsupported,
}

/// Construction-time options for [`TestVideoSource::new`].
#[derive(Debug, Clone, Copy)]
pub struct TestVideoOptions {
    pub width: u32,
    pub height: u32,
    /// How fast `pts` advances per generated frame — not a real-time pace
    /// on its own (see [`TestVideoSource`]'s own docs); only meaningful
    /// once a downstream [`crate::elements::Pacer`] is built against
    /// [`TestVideoSource::time_base`].
    pub framerate: ffmpeg::Rational,
}

impl Default for TestVideoOptions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            framerate: ffmpeg::Rational::new(30, 1),
        }
    }
}

/// Generates a synthetic, moving-diagonal-gradient video stream —
/// GStreamer's `videotestsrc` equivalent. No real decode/demux involved:
/// `run()` fabricates one `Pixel::YUV420P` frame per tick, stamps it with
/// an increasing `pts` (one tick per frame, in [`TestVideoSource::time_base`]'s
/// units), and pushes it straight downstream — useful for exercising
/// `Scaler`/`Pacer`/`Dx12Renderer`/etc. without a real file or camera.
/// `Dx12Renderer` in particular already handles `Pixel::YUV420P` on its
/// CPU-upload path, so this can feed a renderer directly, no decoder
/// needed.
///
/// Doesn't pace itself — generates as fast as whatever's downstream can
/// absorb, same convention as `FileDemuxer`/`RtspSource`; put a `Pacer`
/// after it for real-time playback.
///
/// Runs until `Stop` — never reaches `Eos` on its own (no frame-count
/// limit is exposed, deliberately, mirroring a live camera source more
/// than a file).
#[rust_hlog::hlog]
pub struct TestVideoSource {
    name: Arc<str>,
    options: TestVideoOptions,
    pad: SrcPad,
    frame_index: i64,
    /// Reused across every generated frame — see [`UnboundObjectPool`]'s
    /// docs. `init` builds a fresh `Pixel::YUV420P` frame at this
    /// element's fixed size; the next `generate_frame` call overwrites
    /// every pixel anyway, so `release` has nothing to reset.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

impl TestVideoSource {
    pub fn new(name: impl Into<String>, options: TestVideoOptions) -> Self {
        let name: Arc<str> = name.into().into();
        let hlog = element_hlog(ElementType::TestVideoSource, &name, None);
        let pad = SrcPad::new(format!("{name}_src"));
        hinfo!(
            hlog: &hlog,
            "created: {}x{}, framerate={}",
            options.width,
            options.height,
            options.framerate
        );
        let (width, height) = (options.width, options.height);
        let pool = UnboundObjectPool::new(
            0,
            move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::YUV420P, width, height),
            |_| {},
        );
        Self {
            name,
            hlog,
            options,
            pad,
            frame_index: 0,
            pool,
        }
    }

    /// The unit each generated frame's `pts` is expressed in — what you
    /// need to construct a matching [`crate::elements::Pacer`].
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(
            self.options.framerate.denominator(),
            self.options.framerate.numerator(),
        )
    }

    /// Fabricates the next frame: a diagonal gradient on the Y plane that
    /// shifts by one step per frame (so it visibly moves once played
    /// back), flat neutral chroma (grayscale — color isn't the point,
    /// motion/format correctness is).
    fn generate_frame(&mut self) -> crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video> {
        let mut frame = self.pool.get();

        let offset = self.frame_index;
        let width = self.options.width as usize;
        let y_stride = frame.stride(0);
        let y_height = frame.plane_height(0) as usize;
        {
            let y_plane = frame.data_mut(0);
            for row in 0..y_height {
                for col in 0..width {
                    y_plane[row * y_stride + col] =
                        ((col as i64 + row as i64 + offset) % 256) as u8;
                }
            }
        }
        for plane in [1usize, 2usize] {
            frame.data_mut(plane).fill(128);
        }

        frame.set_pts(Some(self.frame_index));
        self.frame_index += 1;
        frame
    }
}

impl Element for TestVideoSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::TestVideoSource
    }

    fn hlog(&self) -> &rust_hlog::HLog {
        &self.hlog
    }

    fn hlog_mut(&mut self) -> &mut rust_hlog::HLog {
        &mut self.hlog
    }
}

impl Source for TestVideoSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for TestVideoSource {
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> crate::error::Result<()> {
        hinfo!(self, "run: starting");
        loop {
            if drain_control(control, self, bus)? {
                hinfo!(self, "run: stopped");
                return Ok(());
            }
            let frame = self.generate_frame();
            // A downstream failure drops just this one frame — same
            // "report, don't die" contract `Queue`'s worker gives a
            // failing `Sink` — rather than ending this whole source
            // thread over it.
            if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(frame))) {
                bus.post(
                    &self.hlog,
                    BusEvent::Error {
                        element_type: ElementType::TestVideoSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
        }
    }

    fn seek(&mut self, _target: std::time::Duration) -> crate::error::Result<std::time::Duration> {
        Err(TestVideoSourceError::SeekUnsupported.into())
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, thread, time::Duration};

    use rust_hlog::HLog;

    use super::*;
    use crate::{
        control::ControlMsg,
        element::Sink,
        pipeline::{ChainBuilder, Pipeline},
    };

    /// Captures every frame's `(format, width, height, pts)` it sees, in
    /// order — enough to check both pixel format/size and that `pts`
    /// actually advances frame over frame.
    #[rust_hlog::hlog]
    struct RecordingSink {
        seen: Arc<Mutex<Vec<(ffmpeg::format::Pixel, u32, u32, Option<i64>)>>>,
    }

    impl Element for RecordingSink {
        fn name(&self) -> Arc<str> {
            "recorder".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn hlog(&self) -> &HLog {
            &self.hlog
        }
        fn hlog_mut(&mut self) -> &mut HLog {
            &mut self.hlog
        }
    }

    impl Sink for RecordingSink {
        fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
            if let MediaBuffer::Video(frame) = buf {
                self.seen.lock().unwrap().push((
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    frame.pts(),
                ));
            }
            Ok(())
        }
        fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn generates_correctly_sized_yuv420p_frames_with_increasing_pts() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingSink {
            seen: seen.clone(),
            hlog: element_hlog(ElementType::Other, "recorder", None),
        };
        let source = TestVideoSource::new(
            "test-video",
            TestVideoOptions {
                width: 16,
                height: 16,
                framerate: ffmpeg::Rational::new(30, 1),
            },
        );

        let pipeline = Pipeline::new("test", source, |source, ctx| {
            let branch = ChainBuilder::new(ctx.clone()).build(Box::new(sink));
            source.src_pads()[0].link(branch);
        });

        pipeline.run();
        thread::sleep(Duration::from_millis(50));
        pipeline.stop();

        // Blocks until every `Bus` handle has dropped — i.e. the source
        // thread has actually exited, not just acked `Stop`.
        pipeline.bus().log_events();

        let frames = seen.lock().unwrap();
        assert!(!frames.is_empty(), "expected at least one generated frame");
        for window in frames.windows(2) {
            let (format, width, height, pts) = window[0];
            assert_eq!(format, ffmpeg::format::Pixel::YUV420P);
            assert_eq!((width, height), (16, 16));
            assert!(
                window[1].3 > pts,
                "expected pts to strictly increase frame over frame, got {:?} then {:?}",
                pts,
                window[1].3
            );
        }
    }

    #[test]
    fn seek_is_explicitly_unsupported() {
        let mut source = TestVideoSource::new("test-video", TestVideoOptions::default());
        assert!(source.seek(Duration::from_secs(1)).is_err());
    }
}
