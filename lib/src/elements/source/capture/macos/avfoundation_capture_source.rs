//! A camera on macOS, through AVFoundation.
//!
//! A capture session hands each picture to a delegate on a dispatch queue
//! of its own, as a Core Video pixel buffer: NV12, which this asks for and
//! Apple's cameras speak natively. The delegate passes it to the source's
//! thread, which copies it into system memory — or, opened onto a
//! [`VideoToolboxDevice`](crate::elements::VideoToolboxDevice), hands the
//! buffer itself on as a VideoToolbox frame, nothing copied.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use ffmpeg_next as ffmpeg;
use objc2::{
    AnyThread, DefinedClass, define_class, msg_send,
    rc::Retained,
    runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject},
};
use objc2_av_foundation::{
    AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceInput, AVCaptureOutput, AVCaptureSession,
    AVCaptureVideoDataOutput, AVCaptureVideoDataOutputSampleBufferDelegate,
};
use objc2_core_media::{CMSampleBuffer, CMTime};
use objc2_core_video::{
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract},
    element::{Element, ElementType, Produced, Source, SourceStage, Wait, element_pp_log},
    elements::{AvFoundationCaptureFormat, AvFoundationDevice, VideoFormat},
    error::Result,
    platform::macos::{
        avfoundation::{self, Authorization, SetFormatError},
        pixel_buffer::{PixelBuffer, PixelBufferError},
    },
    pool::UnboundObjectPool,
    produce::{Received, source_stage},
};

/// The unit every emitted `pts` counts in: a microsecond of the host clock
/// AVFoundation stamps pictures on.
const TIME_BASE_DENOMINATOR: i32 = 1_000_000;
/// How many pictures may wait between the capture queue and the source's
/// thread before the next is dropped rather than queued: a camera's own
/// rate is the pace, and a picture a tenth of a second late is one to drop.
const QUEUED: usize = 3;
/// How long the camera may send nothing before it is asked whether it is
/// still there. One unplugged sends nothing and says nothing.
const SILENT_DEVICE_CHECK: Duration = Duration::from_secs(2);

/// Errors specific to [`AvFoundationCaptureSource`]. Converts into the
/// crate-wide [`crate::error::Error`] via `?`.
#[derive(Debug, ThisError)]
pub enum AvFoundationCaptureSourceError {
    /// The user, or a policy, has not let this program use the camera.
    /// Allowed again in System Settings, Privacy & Security, Camera.
    #[error("this program may not use the camera")]
    PermissionDenied,

    /// The camera asked for is not there — unplugged since it was listed.
    #[error("{0:?} is not there")]
    DeviceNotFound(String),

    /// The camera would not become a capture session's input — another
    /// program holds it, or it failed to open.
    #[error("{name:?} would not open: {reason}")]
    Open {
        /// The camera's name.
        name: String,
        /// What AVFoundation said.
        reason: String,
    },

    /// The camera does not offer the mode asked for.
    #[error("{name:?} offers no {width}x{height} at {frame_rate} fps")]
    FormatNotOffered {
        /// The camera's name.
        name: String,
        /// Width asked for.
        width: u32,
        /// Height asked for.
        height: u32,
        /// Rate asked for.
        frame_rate: ffmpeg::Rational,
    },

    /// The camera's mode could not be set: another program is configuring
    /// it.
    #[error("{name:?} would not take a mode: {reason}")]
    Locked {
        /// The camera's name.
        name: String,
        /// What AVFoundation said.
        reason: String,
    },

    /// A picture came in a layout other than the NV12 asked for — its
    /// four-character code.
    #[error("the camera sent a {0:#x} picture where NV12 was asked for")]
    NotNv12(u32),

    /// A picture could not be read.
    #[error("a picture from the camera could not be read")]
    Unreadable,

    /// The camera went away while it was capturing — unplugged, or taken by
    /// a program with a higher claim. This source does not reopen it.
    #[error("{0:?} is no longer there to capture")]
    DeviceGone(String),

    /// FFmpeg could not make the frames context VideoToolbox frames are
    /// stamped with.
    #[cfg(feature = "videotoolbox")]
    #[error("failed to build the VideoToolbox frames context: {0}")]
    HwFrames(String),
}

impl From<PixelBufferError> for AvFoundationCaptureSourceError {
    fn from(error: PixelBufferError) -> Self {
        match error {
            PixelBufferError::Unexpected(code) => Self::NotNv12(code),
            PixelBufferError::Lock(_) | PixelBufferError::Truncated => Self::Unreadable,
        }
    }
}

/// What to open, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvFoundationCaptureOptions {
    /// The camera, as [`AvFoundationCaptureSource::list_devices`] listed it.
    pub device: AvFoundationDevice,
    /// Which of [`AvFoundationCaptureSource::list_formats`] to ask for, or
    /// `None` for the mode the camera is in when opened, at that format's
    /// highest rate. A camera this process has not put in a mode is in the
    /// system's default for it — 1080p on a MacBook's camera, rather than
    /// the largest, which on a Mac can be a 4:3 or portrait one; one it has
    /// stays in the mode it was last opened in.
    pub format: Option<AvFoundationCaptureFormat>,
}

/// One camera on macOS, delivering NV12 frames — the macOS counterpart of
/// `MfCaptureSource` and `V4l2CaptureSource`, with the same shape: `open`
/// answers with the source and the geometry it negotiated, and a camera
/// that goes away ends the source with an error rather than reconnecting.
///
/// [`Self::open`] delivers them in system memory; with `videotoolbox`,
/// `open_videotoolbox` delivers the pixel buffers the camera filled
/// themselves, as VideoToolbox frames, for a `VideoToolboxEncoder` to take
/// with nothing copied.
///
/// # Timestamps
///
/// `pts` counts the host clock AVFoundation stamps each picture with, in
/// microseconds (see [`Self::time_base`]), rebased so the first frame of a
/// run leaves at zero. A pause does not advance it: the whole frozen
/// interval collapses to one nominal frame interval, so resuming continues
/// the timeline rather than leaving a hole the length of the pause — as
/// `MfCaptureSource`'s does.
///
/// # Mode
///
/// The camera delivers the mode `open` reports and nothing else, at that
/// mode's rate throughout: its frame duration is fixed, so it does not slow
/// down to expose longer in dim light. A capture session puts a camera in
/// its preset's format each time it starts, whatever format was set — on
/// macOS it has no preset that leaves one alone — so the source puts the
/// camera in its mode and holds it locked for configuration across every
/// start, including a resume. Another program holding the camera's
/// configuration then is [`AvFoundationCaptureSourceError::Locked`].
///
/// # Permission
///
/// macOS asks the user whether a program may use the camera, the first time
/// one does, and `open` waits for the answer. It asks only on behalf of a
/// program whose `Info.plist` says why it wants to, under
/// `NSCameraUsageDescription` — the application's, or for a command-line
/// program the terminal's — and ends one that asks without it. A program
/// that is refused gets [`AvFoundationCaptureSourceError::PermissionDenied`].
///
/// Pictures the source's thread has not taken yet are few: past that, the
/// next is dropped and reported as [`crate::bus::BusEvent::Dropped`].
pub struct AvFoundationCaptureSource(SourceStage<Capturing>);

source_stage!(AvFoundationCaptureSource);

/// One picture from the capture queue, and when the camera took it.
struct Captured {
    buffer: PixelBuffer,
    time: CMTime,
}

/// What the capture delegate holds.
struct DelegateIvars {
    pictures: Sender<Captured>,
    dropped: Arc<AtomicU64>,
}

define_class!(
    /// The capture session's sample buffer delegate: passes each picture on
    /// to the source's thread, or counts it dropped where that has not
    /// taken the last few.
    #[unsafe(super(NSObject))]
    #[name = "MediaPpAvFoundationCaptureDelegate"]
    #[ivars = DelegateIvars]
    struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for Delegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn did_output(
            &self,
            _output: &AVCaptureOutput,
            sample: &CMSampleBuffer,
            _connection: &AVCaptureConnection,
        ) {
            // SAFETY: AVFoundation hands over a live sample buffer for the
            // length of this call; its pixel buffer is retained to outlive it.
            let (image, time) =
                unsafe { (sample.image_buffer(), sample.presentation_time_stamp()) };
            let Some(image) = image else {
                return;
            };
            let captured = Captured {
                buffer: PixelBuffer::new(image),
                time,
            };
            if let Err(TrySendError::Full(_)) = self.ivars().pictures.try_send(captured) {
                self.ivars().dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
);

impl Delegate {
    fn new(ivars: DelegateIvars) -> Retained<Self> {
        let this = Self::alloc().set_ivars(ivars);
        // SAFETY: `NSObject`'s own initializer, on the freshly allocated
        // object whose ivars are set.
        unsafe { msg_send![super(this), init] }
    }
}

/// Where the pictures go.
enum Output {
    /// Copied into NV12 frames in system memory, from this pool.
    System(UnboundObjectPool<ffmpeg::frame::Video>),
    /// Handed on themselves, stamped with a frames context of their size.
    #[cfg(feature = "videotoolbox")]
    VideoToolbox {
        device: crate::platform::ffmpeg::AvBufferRef,
        frames: Option<(u32, u32, crate::platform::ffmpeg::AvBufferRef)>,
    },
}

/// What an [`AvFoundationCaptureSource`] hands on, a picture at a time as
/// the camera delivers them: all of its work, which the framework makes the
/// source.
struct Capturing {
    name: Arc<str>,
    pp_log: PpLog,
    /// Its pipeline's, for saying pictures were dropped — `None` driven by
    /// hand.
    bus: Option<Bus>,
    session: Retained<AVCaptureSession>,
    /// Kept for as long as the session delivers through them.
    _output: Retained<AVCaptureVideoDataOutput>,
    _delegate: Retained<Delegate>,
    _queue: dispatch2::DispatchRetained<dispatch2::DispatchQueue>,
    device: Retained<AVCaptureDevice>,
    /// The mode it delivers, put in again each time the session starts.
    mode: AvFoundationCaptureFormat,
    device_name: String,
    pictures: Receiver<Captured>,
    dropped: Arc<AtomicU64>,
    output: Output,
    format: VideoFormat,
    frame_rate: ffmpeg::Rational,
    timeline: Timeline,
    running: bool,
    /// When a picture last came, for noticing the camera has gone.
    heard_from: Option<Instant>,
}

// SAFETY: AVFoundation's session, device and output objects may be used
// from any thread — starting and stopping a session off the main thread is
// what Apple asks for — and the delegate and queue are only ever held here,
// never called. Every method that touches them takes `&mut self`, and a
// source is driven by one worker thread.
unsafe impl Send for Capturing {}

impl AvFoundationCaptureSource {
    /// Every camera this Mac has, the system's default marked. Asks the
    /// user nothing.
    pub fn list_devices() -> Vec<AvFoundationDevice> {
        avfoundation::list_devices()
    }

    /// Every mode one camera offers, largest first — empty for a camera
    /// that is not there.
    pub fn list_formats(device: &AvFoundationDevice) -> Vec<AvFoundationCaptureFormat> {
        avfoundation::device(&device.id)
            .map(|device| avfoundation::list_formats(&device))
            .unwrap_or_default()
    }

    /// Opens the camera, delivering NV12 frames in system memory, and
    /// answers with the geometry it negotiated.
    ///
    /// Asks the user for the camera where nobody has yet, and waits for the
    /// answer — see the type's docs on what that needs.
    pub fn open(
        name: impl Into<String>,
        options: AvFoundationCaptureOptions,
    ) -> std::result::Result<(Self, VideoFormat), AvFoundationCaptureSourceError> {
        Self::assemble(
            name.into(),
            options,
            Output::System(UnboundObjectPool::new(
                0,
                ffmpeg::frame::Video::empty,
                |_| {},
            )),
        )
    }

    /// As [`Self::open`], delivering the camera's pixel buffers themselves
    /// as VideoToolbox frames on `device` — NV12, for a `VideoToolboxEncoder`
    /// to take with nothing copied.
    #[cfg(feature = "videotoolbox")]
    pub fn open_videotoolbox(
        name: impl Into<String>,
        options: AvFoundationCaptureOptions,
        device: &crate::elements::VideoToolboxDevice,
    ) -> std::result::Result<(Self, VideoFormat), AvFoundationCaptureSourceError> {
        let device = device.retain().try_clone().ok_or_else(|| {
            AvFoundationCaptureSourceError::HwFrames("could not reference the device".into())
        })?;
        Self::assemble(
            name.into(),
            options,
            Output::VideoToolbox {
                device,
                frames: None,
            },
        )
    }

    fn assemble(
        name: String,
        options: AvFoundationCaptureOptions,
        output: Output,
    ) -> std::result::Result<(Self, VideoFormat), AvFoundationCaptureSourceError> {
        crate::ensure_ffmpeg();
        let name: Arc<str> = name.into();
        let pp_log = element_pp_log(ElementType::AvFoundationCaptureSource, &name, None);
        let camera = options.device;
        match avfoundation::authorization() {
            Authorization::Authorized => {}
            Authorization::NotDetermined if avfoundation::request_access() => {}
            _ => return Err(AvFoundationCaptureSourceError::PermissionDenied),
        }
        let device = avfoundation::device(&camera.id)
            .ok_or_else(|| AvFoundationCaptureSourceError::DeviceNotFound(camera.name.clone()))?;

        // SAFETY: a fresh session and the device's input made for it; the
        // input is only added where the session says it can be.
        let session = unsafe {
            let session = AVCaptureSession::new();
            let input =
                AVCaptureDeviceInput::deviceInputWithDevice_error(&device).map_err(|error| {
                    AvFoundationCaptureSourceError::Open {
                        name: camera.name.clone(),
                        reason: error.localizedDescription().to_string(),
                    }
                })?;
            if !session.canAddInput(&input) {
                return Err(AvFoundationCaptureSourceError::Open {
                    name: camera.name.clone(),
                    reason: "the capture session will not take it".into(),
                });
            }
            session.addInput(&input);
            session
        };
        let (pictures_tx, pictures) = crossbeam_channel::bounded(QUEUED);
        let dropped = Arc::new(AtomicU64::new(0));
        let delegate = Delegate::new(DelegateIvars {
            pictures: pictures_tx,
            dropped: Arc::clone(&dropped),
        });
        let queue = dispatch2::DispatchQueue::new("media-pp.avfoundation-capture", None);
        // SAFETY: the key is Core Video's own constant, toll-free bridged to
        // the NSString the dictionary is keyed by; the delegate and queue are
        // kept for as long as the output delivers through them; the output is
        // only added where the session says it can be.
        let output_object = unsafe {
            let output = AVCaptureVideoDataOutput::new();
            let key = &*(kCVPixelBufferPixelFormatTypeKey as *const _ as *const NSString);
            let settings: Retained<NSDictionary<NSString, AnyObject>> =
                NSDictionary::from_retained_objects(
                    &[key],
                    &[Retained::into_super(Retained::into_super(NSNumber::new_u32(
                        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                    )))
                    .into()],
                );
            output.setVideoSettings(Some(&settings));
            output.setAlwaysDiscardsLateVideoFrames(true);
            output.setSampleBufferDelegate_queue(
                Some(ProtocolObject::from_ref(&*delegate)),
                Some(&queue),
            );
            if !session.canAddOutput(&output) {
                return Err(AvFoundationCaptureSourceError::Open {
                    name: camera.name.clone(),
                    reason: "the capture session will not deliver pictures".into(),
                });
            }
            session.addOutput(&output);
            output
        };

        // Whatever the mode, it is held, as it would otherwise be what the
        // session's preset makes it when it starts; it is put in now for what
        // is reported to be what the camera delivers, and again at each start.
        let mode = options
            .format
            .unwrap_or_else(|| avfoundation::active_format(&device));
        drop(
            avfoundation::hold_format(&device, mode)
                .map_err(|error| format_error(&camera.name, mode, error))?,
        );
        let format = VideoFormat {
            width: mode.width,
            height: mode.height,
            time_base: ffmpeg::Rational::new(1, TIME_BASE_DENOMINATOR),
        };
        let frame_interval = (i64::from(TIME_BASE_DENOMINATOR)
            * i64::from(mode.frame_rate.denominator()))
            / i64::from(mode.frame_rate.numerator().max(1));
        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?}, {}x{} at {} fps, NV12{}",
            camera.name,
            mode.width,
            mode.height,
            mode.frame_rate,
            if matches!(output, Output::System(_)) { "" } else { " in VideoToolbox frames" }
        );
        Ok((
            Self(SourceStage::new(Capturing {
                name,
                pp_log,
                bus: None,
                session,
                _output: output_object,
                _delegate: delegate,
                _queue: queue,
                device,
                mode,
                device_name: camera.name,
                pictures,
                dropped,
                output,
                format,
                frame_rate: mode.frame_rate,
                timeline: Timeline::new(frame_interval),
                running: false,
                heard_from: None,
            })),
            format,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in: a microsecond of
    /// the host clock — see the type's docs.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.0.inner().format.time_base
    }

    /// The rate the camera's mode was negotiated at — what an encoder after
    /// it is opened with. Nominal: frames are stamped as the camera delivers
    /// them.
    pub fn frame_rate(&self) -> ffmpeg::Rational {
        self.0.inner().frame_rate
    }
}

/// A host time in microseconds.
fn microseconds(time: CMTime) -> i64 {
    if time.timescale <= 0 {
        return 0;
    }
    (i128::from(time.value) * i128::from(TIME_BASE_DENOMINATOR) / i128::from(time.timescale)) as i64
}

/// Host times made into `pts`: from zero at the first picture, and across a
/// pause one nominal frame on from the last — `MfCaptureSource::stamp`'s
/// bookkeeping.
fn format_error(
    name: &str,
    mode: AvFoundationCaptureFormat,
    error: SetFormatError,
) -> AvFoundationCaptureSourceError {
    match error {
        SetFormatError::NotOffered => AvFoundationCaptureSourceError::FormatNotOffered {
            name: name.to_owned(),
            width: mode.width,
            height: mode.height,
            frame_rate: mode.frame_rate,
        },
        SetFormatError::Locked(reason) => AvFoundationCaptureSourceError::Locked {
            name: name.to_owned(),
            reason,
        },
    }
}

struct Timeline {
    /// One nominal frame in microseconds — what a pause costs.
    interval: i64,
    /// Host time the current run counts from, moved on by each pause.
    origin: Option<i64>,
    /// Host time of the last picture stamped.
    last: Option<i64>,
    /// Set by a resume; taken by the next picture.
    rebase: bool,
}

impl Timeline {
    fn new(interval: i64) -> Self {
        Self {
            interval,
            origin: None,
            last: None,
            rebase: false,
        }
    }

    /// The next picture, taken after a pause, follows the last by one frame.
    fn resumed(&mut self) {
        self.rebase = true;
    }

    fn stamp(&mut self, time: i64) -> i64 {
        match (self.origin, self.rebase.then_some(self.last).flatten()) {
            (Some(origin), Some(last)) => {
                self.origin = Some(origin + (time - last) - self.interval);
            }
            (None, _) => self.origin = Some(time),
            _ => {}
        }
        self.rebase = false;
        self.last = Some(time);
        time - self
            .origin
            .expect("the origin was just set if it was missing")
    }
}

impl Capturing {
    fn start(&mut self) -> std::result::Result<(), AvFoundationCaptureSourceError> {
        if !self.running {
            // Held across the start, which would otherwise put the camera
            // back in the session preset's format.
            let _hold = avfoundation::hold_format(&self.device, self.mode)
                .map_err(|error| format_error(&self.device_name, self.mode, error))?;
            // SAFETY: a configured session, started off the main thread as
            // Apple asks; this blocks until the camera runs.
            unsafe { self.session.startRunning() };
            self.running = true;
        }
        Ok(())
    }

    fn stop(&mut self) {
        if self.running {
            // SAFETY: as for `start`.
            unsafe { self.session.stopRunning() };
            self.running = false;
        }
    }

    /// Says, once, how many pictures were dropped since last time.
    fn report_dropped(&self) {
        let dropped = self.dropped.swap(0, Ordering::Relaxed);
        if dropped == 0 {
            return;
        }
        pp_warn!(
            self,
            "dropped {dropped} picture(s): downstream is not keeping up"
        );
        if let Some(bus) = &self.bus {
            bus.post(
                &self.pp_log,
                BusEvent::Dropped {
                    element_type: ElementType::AvFoundationCaptureSource,
                    name: self.name.clone(),
                },
            );
        }
    }

    /// `captured` as the frame this source hands on.
    fn frame(&mut self, captured: Captured) -> Result<MediaBuffer> {
        let pts = self.timeline.stamp(microseconds(captured.time));
        match &mut self.output {
            Output::System(pool) => {
                let mut frame = pool.get();
                captured
                    .buffer
                    .copy_to(&mut frame, ffmpeg::format::Pixel::NV12)
                    .map_err(AvFoundationCaptureSourceError::from)?;
                frame.set_pts(Some(pts));
                crate::buffer::set_time_base(&mut frame, self.format.time_base);
                Ok(MediaBuffer::Video(Arc::new(frame).into()))
            }
            #[cfg(feature = "videotoolbox")]
            Output::VideoToolbox { device, frames } => {
                let (width, height) = captured.buffer.size();
                if !frames
                    .as_ref()
                    .is_some_and(|&(w, h, _)| (w, h) == (width, height))
                {
                    // SAFETY: `device` is this source's own reference to a
                    // live VideoToolbox device context.
                    let made = unsafe {
                        crate::platform::macos::videotoolbox::create_frames_ctx(
                            device,
                            ffmpeg::format::Pixel::NV12,
                            width,
                            height,
                        )
                    }
                    .map_err(|error| AvFoundationCaptureSourceError::HwFrames(error.to_string()))?;
                    *frames = Some((width, height, made));
                }
                let (_, _, frames_ctx) = frames.as_ref().expect("made above");
                let mut frame = captured
                    .buffer
                    .into_videotoolbox_frame(frames_ctx)
                    .map_err(AvFoundationCaptureSourceError::from)?;
                frame.set_pts(Some(pts));
                crate::buffer::set_time_base(&mut frame, self.format.time_base);
                Ok(MediaBuffer::video(frame))
            }
        }
    }
}

impl Drop for Capturing {
    fn drop(&mut self) {
        self.stop();
    }
}

impl Element for Capturing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::AvFoundationCaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        self.bus = Some(context.bus.for_element(context.source_id));
    }
}

impl Source for Capturing {
    fn is_live(&self) -> bool {
        true
    }

    fn output_contract(&self) -> OutputContract {
        let domain = match self.output {
            Output::System(_) => MemoryDomain::System,
            #[cfg(feature = "videotoolbox")]
            Output::VideoToolbox { .. } => MemoryDomain::VideoToolbox,
        };
        OutputContract::Fixed(
            PortContract::frame(MediaKind::VideoFrame, domain).with_layouts(PixelLayoutSet::NV12),
        )
    }

    /// The next picture the camera delivered; with none for a while, a look
    /// at whether the camera is still there.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        self.report_dropped();
        let deadline = wait.now() + SILENT_DEVICE_CHECK;
        match wait.recv_until(&self.pictures, deadline) {
            Some(Received::Got(captured)) => {
                self.heard_from = Some(Instant::now());
                Ok(Produced::Buffer(self.frame(captured)?))
            }
            Some(Received::LetGo) => Ok(Produced::Nothing),
            Some(Received::Gone) => {
                pp_error!(self, "the capture delegate went away");
                Err(AvFoundationCaptureSourceError::DeviceGone(self.device_name.clone()).into())
            }
            None => {
                // SAFETY: a plain query of the device.
                if !unsafe { self.device.isConnected() } {
                    pp_error!(self, "the camera is gone");
                    return Err(AvFoundationCaptureSourceError::DeviceGone(
                        self.device_name.clone(),
                    )
                    .into());
                }
                Ok(Produced::Nothing)
            }
        }
    }

    /// Stops the camera for the pause.
    fn pausing(&mut self) -> Result<()> {
        self.stop();
        Ok(())
    }

    /// Lets go of what the pause left queued, so playing on starts from live
    /// pictures, and starts the camera again.
    fn resuming(&mut self) -> Result<()> {
        while self.pictures.try_recv().is_ok() {}
        self.timeline.resumed();
        Ok(self.start()?)
    }

    /// Starts the camera on the source's own thread.
    fn starting(&mut self) -> Result<()> {
        Ok(self.start()?)
    }

    fn stopping(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default camera, where this program may already use it — never
    /// asking, since a test runner that asks without a camera usage
    /// description is ended by macOS — or `None` after saying why not.
    fn try_camera() -> Option<AvFoundationDevice> {
        if avfoundation::authorization() != Authorization::Authorized {
            eprintln!("skipping: this program has not been allowed the camera");
            return None;
        }
        let camera = AvFoundationCaptureSource::list_devices()
            .into_iter()
            .find(|camera| camera.is_default);
        if camera.is_none() {
            eprintln!("skipping: this Mac has no camera");
        }
        camera
    }

    /// A pause collapses to one frame interval: the next picture follows the
    /// last by one frame whatever time has passed.
    #[test]
    fn a_pause_costs_one_frame_interval() {
        let mut timeline = Timeline::new(33_333);
        assert_eq!(timeline.stamp(1_000_000), 0);
        assert_eq!(timeline.stamp(1_033_333), 33_333);
        timeline.resumed();
        assert_eq!(
            timeline.stamp(9_000_000),
            66_666,
            "one interval after the last"
        );
        assert_eq!(timeline.stamp(9_033_333), 99_999, "and on from there");
    }

    type Picture = (ffmpeg::format::Pixel, u32, u32, Option<i64>);

    /// What `camera` in `mode` delivers over a couple of seconds, and what it
    /// said it would.
    fn capture(
        camera: &AvFoundationDevice,
        mode: Option<AvFoundationCaptureFormat>,
    ) -> (VideoFormat, Vec<Picture>) {
        let (source, format) = AvFoundationCaptureSource::open(
            "camera",
            AvFoundationCaptureOptions {
                device: camera.clone(),
                format: mode,
            },
        )
        .expect("the camera opens");
        let frames = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = crate::elements::AppSink::new("camera-sink", {
            let frames = Arc::clone(&frames);
            move |buffer| {
                if let MediaBuffer::Video(frame) = &buffer {
                    frames.lock().unwrap().push((
                        frame.format(),
                        frame.width(),
                        frame.height(),
                        frame.pts(),
                    ));
                }
                Ok(())
            }
        });
        let (pipeline, ()) = crate::pipeline::Pipeline::new("camera", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        pipeline.run().expect("the camera runs");
        std::thread::sleep(Duration::from_millis(2500));
        pipeline.stop();
        let frames = std::mem::take(&mut *frames.lock().unwrap());
        (format, frames)
    }

    fn assert_delivers(format: VideoFormat, frames: &[Picture]) {
        assert!(frames.len() >= 10, "{} pictures", frames.len());
        assert_eq!(frames[0].3, Some(0), "the first leaves at zero");
        for &(pixel, width, height, _) in frames {
            assert_eq!(pixel, ffmpeg::format::Pixel::NV12);
            assert_eq!(
                (width, height),
                (format.width, format.height),
                "what it said at open"
            );
        }
        assert!(frames.windows(2).all(|pair| pair[0].3 < pair[1].3));
    }

    /// A camera this program may use delivers NV12 pictures of the size it
    /// opened at, timed from zero on the host clock.
    #[test]
    fn a_camera_delivers_nv12_timed_from_zero() {
        let Some(camera) = try_camera() else {
            return;
        };
        let (format, frames) = capture(&camera, None);
        assert_delivers(format, &frames);
    }

    /// A mode asked for is the mode delivered — a session left on its preset
    /// put the camera back in 1080p when it started — and a camera opened
    /// afterwards without one delivers the size it reports, whichever that is.
    #[test]
    fn a_camera_delivers_the_mode_asked_for() {
        let Some(camera) = try_camera() else {
            return;
        };
        let modes = AvFoundationCaptureSource::list_formats(&camera);
        let (default, _) = capture(&camera, None);
        let Some(&mode) = modes
            .iter()
            .find(|mode| (mode.width, mode.height) != (default.width, default.height))
        else {
            eprintln!("skipping: {} offers one size only", camera.name);
            return;
        };
        let (format, frames) = capture(&camera, Some(mode));
        assert_eq!((format.width, format.height), (mode.width, mode.height));
        assert_delivers(format, &frames);
        let (format, frames) = capture(&camera, None);
        assert_delivers(format, &frames);
    }
}
