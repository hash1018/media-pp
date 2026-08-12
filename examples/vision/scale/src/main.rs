use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ffmpeg_next::{format::Pixel, media, software::scaling::Flags};
use media_pp::{
    Error, Result,
    buffer::MediaBuffer,
    bus::BusEvent,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_hlog},
    elements::{FileDemuxer, Scaler, SwDecoder},
    pipeline::Pipeline,
};
use rust_hlog::HLog;

const DST_FORMAT: Pixel = Pixel::RGB24;
const DST_WIDTH: u32 = 640;
const DST_HEIGHT: u32 = 640;

/// Demux -> SwDecoder -> Scaler -> (prints the first scaled frame's actual
/// format/size, then counts the rest). Proves `Scaler` really converts
/// pixel format (whatever the decoder produces -> RGB24) and resizes
/// (source resolution -> a fixed 640x640, the kind of input an ONNX
/// object-detection model would want), not just that it compiles.
///
///     cargo run -p scale -- path/to/video.mp4
fn main() -> media_pp::Result<()> {
    media_pp::init()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test-video/h265.mp4".into());

    let (source, streams) = FileDemuxer::open("demux", &path)?;
    let video = streams
        .iter()
        .find(|s| s.kind == media::Type::Video)
        .ok_or_else(|| Error::Other("no video stream in file".into()))?;
    let params = source
        .stream_parameters(video.index)
        .ok_or_else(|| Error::Other("stream disappeared".into()))?;

    let count = Arc::new(AtomicUsize::new(0));
    let sink = VerifyingSink {
        count: count.clone(),
        hlog: element_hlog(ElementType::Other, "verify", None),
    };

    let pipeline = Pipeline::new("scale", source, |source, ctx| {
        let decoder = SwDecoder::new("decoder", params).expect("failed to open decoder");
        let scaler = Scaler::new("scaler", DST_FORMAT, DST_WIDTH, DST_HEIGHT, Flags::BILINEAR);
        let branch = ctx
            .branch()
            .pipe(decoder) // same thread as the demux — cheap enough not to need a queue
            .queue("frames", 8) // scaler/sink run on their own thread
            .pipe(scaler)
            .to(Box::new(sink))?;
        ctx.attach(source, video.index, branch)?;
        Ok(())
    })?;

    pipeline.run();

    // Watch for `Eos`/`Error` and `stop()` on either — errors no longer
    // end the pipeline on their own, so this is what makes the loop
    // below actually finish instead of running forever after a failure.
    for event in pipeline.bus().iter() {
        match &event {
            BusEvent::Eos { name, .. } => println!("[{name}] eos"),
            BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
            BusEvent::Dropped { name, .. } => eprintln!("[{name}] dropped a buffer (queue full)"),
            BusEvent::Seeked {
                name,
                requested,
                landed,
                ..
            } => println!("[{name}] seeked: requested {requested:.2?}, landed {landed:.2?}"),
        }
        if matches!(event, BusEvent::Eos { .. } | BusEvent::Error { .. }) {
            pipeline.stop();
        }
    }

    println!("scaled frames: {}", count.load(Ordering::Relaxed));
    Ok(())
}

/// Terminal sink that prints the *first* scaled frame's actual
/// format/dimensions — proof `Scaler` really produced `DST_FORMAT` at
/// `DST_WIDTH`x`DST_HEIGHT`, not just that the pipeline ran to
/// completion — then counts every frame after that the same way
/// `FrameCounter` would.
#[rust_hlog::hlog]
struct VerifyingSink {
    count: Arc<AtomicUsize>,
}

impl Element for VerifyingSink {
    fn name(&self) -> Arc<str> {
        "verify".into()
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

impl Sink for VerifyingSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if let MediaBuffer::Video(frame) = &buf
            && self.count.fetch_add(1, Ordering::Relaxed) == 0
        {
            println!(
                "first scaled frame: {:?} {}x{} (expected {DST_FORMAT:?} {DST_WIDTH}x{DST_HEIGHT})",
                frame.format(),
                frame.width(),
                frame.height(),
            );
            assert_eq!(
                frame.format(),
                DST_FORMAT,
                "Scaler didn't convert pixel format"
            );
            assert_eq!(frame.width(), DST_WIDTH, "Scaler didn't resize width");
            assert_eq!(frame.height(), DST_HEIGHT, "Scaler didn't resize height");
        }
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}
