//! A camera on Linux, through V4L2.
//!
//! # What this owns and what FFmpeg owns
//!
//! The buffer negotiation, the mmap ring and the dequeue loop are FFmpeg's
//! `video4linux2` demuxer's — it has done them for twenty years and there is
//! nothing this crate would do differently. What is written here is the part
//! that demuxer has no call for: asking a camera what it offers, choosing
//! among it, and turning what
//! arrives into the one thing a compositor takes.
//!
//! # Why it hands over NV12
//!
//! A camera speaks whatever its firmware speaks — YUYV at low resolutions,
//! Motion JPEG at high ones, and those two cover nearly every USB camera.
//! Neither is something the compositors upload, so the conversion has to
//! happen somewhere; doing it here means the element's output is the same
//! shape its Windows counterpart's is, and a caller's pipeline is the same on
//! both. `MfCaptureSource` hands over NV12 for the same reason.
//!
//! MJPEG is decoded rather than refused, because a camera that offers 720p
//! usually offers it *only* compressed: dropping to the raw format would drop
//! to 640x480, which is not the mode the user picked.

use std::sync::Arc;
use std::time::Duration;

use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::VideoFormat,
    error::Result,
    pad::SrcPad,
    platform::linux::v4l2::{V4l2CaptureFormat, V4l2Device},
    pool::UnboundObjectPool,
};

/// Errors specific to [`V4l2CaptureSource`]. Converts into the crate-wide
/// [`crate::error::Error`] via `?`.
#[derive(Debug, ThisError)]
pub enum V4l2CaptureSourceError {
    /// FFmpeg refused the device, the format, or a frame from it.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    /// The `video4linux2` demuxer is not in this build of FFmpeg, so there is
    /// no camera backend at all.
    #[error("this FFmpeg has no video4linux2 input device")]
    NoV4l2Demuxer,

    /// The device opened but announced no video stream, which a camera that
    /// is being held by something else can do.
    #[error("{0:?} offers no video stream")]
    NoVideoStream(String),

    /// The camera negotiated a frame size NV12 cannot describe: its chroma is
    /// half-sized in both axes, so an odd dimension has no whole pixel to
    /// carry.
    #[error("negotiated an odd {width}x{height} frame, which NV12 cannot carry")]
    OddFrameSize {
        /// Negotiated frame width in pixels.
        width: u32,
        /// Negotiated frame height in pixels.
        height: u32,
    },
}

/// What to open, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V4l2CaptureOptions {
    /// The camera, as [`V4l2CaptureSource::list_devices`] listed it.
    pub device: V4l2Device,
    /// The mode to ask for, or `None` to take whatever the driver offers
    /// first — which is what a caller with no preference should send, since a
    /// mode a camera does not have is refused rather than approximated.
    pub format: Option<V4l2CaptureFormat>,
}

/// One camera, delivering NV12 frames in system memory.
///
/// The Linux counterpart of Windows' `MfCaptureSource`, with the same
/// shape: `open` answers with the source and the geometry it
/// negotiated, `run` pushes frames until it is stopped or the device goes
/// away, and a device that disappears mid-capture ends the source with an
/// error rather than reconnecting — a pipeline is one-shot, so coming back
/// is whoever watches the bus building a new one.
pub struct V4l2CaptureSource {
    name: Arc<str>,
    pp_log: PpLog,
    input: ffmpeg::format::context::Input,
    stream: usize,
    decoder: ffmpeg::decoder::Video,
    /// Built on the first frame rather than at open: what the decoder
    /// actually produces is not knowable until it has produced one, and a
    /// camera can be asked for a format it answers in a different pixel
    /// layout.
    scaler: Option<ffmpeg::software::scaling::Context>,
    format: VideoFormat,
    pads: Vec<SrcPad>,
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: the only member without a `Send` of its own is the scaling
// context, which wraps a heap-allocated `SwsContext` with no thread affinity
// — the same reasoning `SwScaler` states, and for the same missing impl in
// ffmpeg-next. Every method that touches it takes `&mut self`, so no two
// threads reach it at once, and a source is driven by one worker thread.
unsafe impl Send for V4l2CaptureSource {}

impl V4l2CaptureSource {
    /// Opens the camera and negotiates a mode.
    ///
    /// The geometry that comes back is what the device actually gave, which
    /// is not always what was asked for: a driver may answer a request with
    /// the nearest thing it has.
    pub fn open(
        name: impl Into<String>,
        options: V4l2CaptureOptions,
    ) -> std::result::Result<(Self, VideoFormat), V4l2CaptureSourceError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::V4l2CaptureSource, &name, None);

        // SAFETY: both calls are FFmpeg's own registration and lookup, which
        // take no arguments this side owns; the returned descriptor is
        // FFmpeg's static one and is only handed back to it.
        let demuxer = unsafe {
            ffi::avdevice_register_all();
            ffi::av_find_input_format(c"video4linux2".as_ptr())
        };
        if demuxer.is_null() {
            return Err(V4l2CaptureSourceError::NoV4l2Demuxer);
        }

        let mut settings = ffmpeg::Dictionary::new();
        if let Some(format) = options.format {
            settings.set("video_size", &format!("{}x{}", format.width, format.height));
            settings.set(
                "framerate",
                &format!(
                    "{}/{}",
                    format.framerate.numerator(),
                    format.framerate.denominator()
                ),
            );
            // And which of the camera's formats carries that mode: one
            // geometry is not one mode, and leaving the choice open lets the
            // demuxer pick a format that does not have the size, after which
            // the driver answers with whatever it does have — see
            // `format_name_for`.
            if let Some(name) = crate::platform::linux::v4l2::format_name_for(
                &options.device.id,
                format.width,
                format.height,
                format.framerate,
            ) {
                settings.set("input_format", name);
            }
        }

        let input = open_input(demuxer, &options.device.id, settings)?;
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| V4l2CaptureSourceError::NoVideoStream(options.device.id.clone()))?;
        let index = stream.index();
        let time_base = stream.time_base();
        let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
            .decoder()
            .video()?;

        let (width, height) = (decoder.width(), decoder.height());
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(V4l2CaptureSourceError::OddFrameSize { width, height });
        }
        let format = VideoFormat {
            width,
            height,
            time_base,
        };

        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?}, {}x{} {:?} -> NV12",
            options.device.id,
            width,
            height,
            decoder.format()
        );
        let pads = vec![SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::System,
            )),
        )];
        Ok((
            Self {
                name,
                pp_log,
                input,
                stream: index,
                decoder,
                scaler: None,
                format,
                pads,
                pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {}),
            },
            format,
        ))
    }

    /// Every camera this machine has, in node order.
    pub fn list_devices() -> std::io::Result<Vec<V4l2Device>> {
        crate::platform::linux::v4l2::list_devices()
    }

    /// Every mode one camera offers, best first.
    pub fn list_formats(device: &str) -> std::io::Result<Vec<V4l2CaptureFormat>> {
        crate::platform::linux::v4l2::list_formats(device)
    }

    /// Decodes one packet and pushes whatever pictures it made.
    fn deliver(&mut self, packet: &ffmpeg::Packet, bus: &Bus) -> Result<()> {
        if self.decoder.send_packet(packet).is_err() {
            // One packet the decoder would not take — a truncated JPEG from a
            // camera that was unplugged mid-frame, most often. The next one
            // usually decodes, and ending the source over a dropped frame
            // would be worse than the frame.
            pp_warn!(self, "a frame did not decode; dropping it");
            return Ok(());
        }
        let mut decoded = ffmpeg::frame::Video::empty();
        while self.decoder.receive_frame(&mut decoded).is_ok() {
            let mut converted = self.pool.get();
            self.convert(&decoded, &mut converted)?;
            if let Err(error) = self.pads[0].push(MediaBuffer::Video(Arc::new(converted))) {
                // Reported rather than fatal, the same contract every other
                // source here keeps: a sink that failed is not a camera that
                // stopped.
                bus.post(
                    &self.pp_log,
                    BusEvent::Error {
                        element_type: ElementType::V4l2CaptureSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
        }
        Ok(())
    }

    /// Into NV12, building the converter on the first frame that needs it.
    fn convert(
        &mut self,
        decoded: &ffmpeg::frame::Video,
        converted: &mut ffmpeg::frame::Video,
    ) -> Result<()> {
        let scaler = match &mut self.scaler {
            Some(scaler) => scaler,
            none => none.insert(
                ffmpeg::software::scaling::Context::get(
                    decoded.format(),
                    decoded.width(),
                    decoded.height(),
                    ffmpeg::format::Pixel::NV12,
                    self.format.width,
                    self.format.height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                )
                .map_err(V4l2CaptureSourceError::Ffmpeg)?,
            ),
        };
        if converted.format() != ffmpeg::format::Pixel::NV12
            || converted.width() != self.format.width
            || converted.height() != self.format.height
        {
            *converted = ffmpeg::frame::Video::new(
                ffmpeg::format::Pixel::NV12,
                self.format.width,
                self.format.height,
            );
        }
        scaler
            .run(decoded, converted)
            .map_err(V4l2CaptureSourceError::Ffmpeg)?;
        // The camera's own timestamp, in the demuxer's time base — the same
        // reasoning the Windows source gives for counting the device's clock
        // rather than a nominal frame grid: exposure lengthens an interval,
        // and a mode nominally at 30 fps is not delivered on a 1/30 one.
        converted.set_pts(decoded.pts().or_else(|| decoded.timestamp()));
        Ok(())
    }
}

/// `avformat_open_input` with the device's own options.
///
/// Written out rather than taken from `ffmpeg::format::open_with`, which has
/// no way to pass an input format *and* a dictionary in this binding.
fn open_input(
    demuxer: *const ffi::AVInputFormat,
    path: &str,
    settings: ffmpeg::Dictionary,
) -> std::result::Result<ffmpeg::format::context::Input, V4l2CaptureSourceError> {
    let path = std::ffi::CString::new(path).map_err(|_| ffmpeg::Error::InvalidData)?;
    let mut context = std::ptr::null_mut();
    // SAFETY: `disown` hands over the dictionary's own pointer, which is
    // exactly what `avformat_open_input` takes and what the free below
    // reclaims — the wrapper must not also free it.
    let mut options = unsafe { settings.disown() };
    // SAFETY: `context` is a null out-parameter FFmpeg fills or leaves null,
    // `path` is a live `CString` for the whole call, `demuxer` is FFmpeg's own
    // static descriptor, and `options` is the dictionary this call takes
    // ownership of the contents of — freed below either way.
    let code =
        unsafe { ffi::avformat_open_input(&mut context, path.as_ptr(), demuxer, &mut options) };
    // SAFETY: `options` is what the call left of the dictionary: either the
    // options it did not recognise, or the whole of it on failure.
    unsafe { ffi::av_dict_free(&mut options) };
    if code < 0 {
        return Err(V4l2CaptureSourceError::Ffmpeg(ffmpeg::Error::from(code)));
    }
    // SAFETY: the call above filled `context` with an opened input, and this
    // hands ownership of it to the wrapper that closes it on drop.
    let mut input = unsafe { ffmpeg::format::context::Input::wrap(context) };
    // SAFETY: `input` owns the context this reads streams from.
    let code = unsafe { ffi::avformat_find_stream_info(input.as_mut_ptr(), std::ptr::null_mut()) };
    if code < 0 {
        return Err(V4l2CaptureSourceError::Ffmpeg(ffmpeg::Error::from(code)));
    }
    Ok(input)
}

impl Element for V4l2CaptureSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::V4l2CaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for V4l2CaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        &mut self.pads
    }
}

impl SourceElement for V4l2CaptureSource {
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        loop {
            if drain_control(control, self, bus)?.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }

            let mut packet = ffmpeg::Packet::empty();
            match packet.read(&mut self.input) {
                Ok(()) => {
                    if packet.stream() == self.stream {
                        self.deliver(&packet, bus)?;
                    }
                }
                // A camera does not end its own stream; if it says so, it has
                // been unplugged.
                Err(ffmpeg::Error::Eof) => break,
                Err(error) => {
                    pp_error!(self, "read failed: {error}");
                    return Err(V4l2CaptureSourceError::Ffmpeg(error).into());
                }
            }
        }
        for pad in self.pads.iter_mut() {
            pad.push_eos(&self.pp_log)?;
        }
        pp_info!(self, "event=eos phase=source_completed outcome=ok");
        Ok(())
    }

    fn seek(&mut self, _target: Duration) -> std::result::Result<Duration, crate::error::Error> {
        Err(ffmpeg::Error::from(ffi::AVERROR(libc::ENOSYS)).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;
    use crate::control::{ControlMsg, channel};
    use crate::element::Sink;
    use std::sync::Mutex as StdMutex;

    struct CountingSink {
        pp_log: PpLog,
        frames: Arc<StdMutex<Vec<(u32, u32, ffmpeg::format::Pixel)>>>,
    }

    impl Element for CountingSink {
        fn name(&self) -> Arc<str> {
            "counter".into()
        }
        fn element_type(&self) -> ElementType {
            ElementType::Other
        }
        fn pp_log(&self) -> &PpLog {
            &self.pp_log
        }
        fn pp_log_mut(&mut self) -> &mut PpLog {
            &mut self.pp_log
        }
    }

    impl Sink for CountingSink {
        fn consume(&mut self, buffer: MediaBuffer) -> Result<()> {
            if let MediaBuffer::Video(frame) = buffer {
                self.frames
                    .lock()
                    .unwrap()
                    .push((frame.width(), frame.height(), frame.format()));
            }
            Ok(())
        }
        fn control(&mut self, _message: ControlMsg) -> Result<()> {
            Ok(())
        }
    }

    /// The whole path on real hardware: open the first camera this machine
    /// has, take a few frames, and check they are the shape a compositor
    /// takes. Skipped where there is no camera, which is every CI runner.
    ///
    /// NV12 is the assertion that matters. A camera speaks YUYV or Motion
    /// JPEG and neither is uploadable, so a source that handed either of them
    /// on would compile, run, and produce a picture nothing could draw.
    #[test]
    fn a_camera_delivers_nv12_frames_a_compositor_can_take() {
        let Ok(devices) = V4l2CaptureSource::list_devices() else {
            return;
        };
        let Some(device) = devices.into_iter().next() else {
            eprintln!("skipping: this machine has no camera");
            return;
        };
        // The best mode rather than any: on this machine that is 1280x720,
        // which only Motion JPEG carries — so the decode path is the one
        // under test rather than the raw passthrough.
        let mode = V4l2CaptureSource::list_formats(&device.id)
            .ok()
            .and_then(|modes| modes.into_iter().next());
        let opened = V4l2CaptureSource::open(
            "camera",
            V4l2CaptureOptions {
                device: device.clone(),
                format: mode,
            },
        );
        let (mut source, format) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: {:?} would not open: {error}", device.id);
                return;
            }
        };
        assert!(format.width > 0 && format.height > 0);

        let frames = Arc::new(StdMutex::new(Vec::new()));
        source.src_pads()[0].link(Box::new(CountingSink {
            pp_log: element_pp_log(ElementType::Other, "counter", None),
            frames: Arc::clone(&frames),
        }));

        // Stopped from another thread once enough frames have arrived: a
        // camera never ends its own stream, so the run loop only returns
        // when it is told to.
        let (control_tx, control_rx) = channel();
        let watched = Arc::clone(&frames);
        let stopper = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while watched.lock().unwrap().len() < 3 && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(20));
            }
            control_tx.send(ControlMsg::Stop);
        });

        let (bus, _receiver) = Bus::new();
        source.run(&control_rx, &bus).expect("the camera ran");
        stopper.join().expect("the stopper thread");

        let frames = frames.lock().unwrap();
        assert!(
            frames.len() >= 3,
            "a camera that opened must deliver pictures: got {}",
            frames.len()
        );
        for (width, height, pixel) in frames.iter() {
            assert_eq!(*pixel, ffmpeg::format::Pixel::NV12);
            assert_eq!((*width, *height), (format.width, format.height));
        }
    }
}
