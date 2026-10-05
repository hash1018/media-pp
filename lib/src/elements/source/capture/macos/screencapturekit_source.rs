//! A display or a window on macOS, through ScreenCaptureKit.
//!
//! A stream hands each picture to a delegate on a dispatch queue of its
//! own, as a Core Video pixel buffer: BGRA, which this asks for. The
//! delegate keeps the latest, and the source's thread emits it at a fixed
//! rate — copied into system memory, or, opened onto a
//! [`VideoToolboxDevice`](crate::elements::VideoToolboxDevice), handed on
//! itself as a VideoToolbox frame, nothing copied.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ffmpeg_next as ffmpeg;
use ffmpeg_next::ffi;
use objc2::{
    AnyThread, DefinedClass, define_class, msg_send,
    rc::Retained,
    runtime::{AnyObject, NSObject, NSObjectProtocol, ProtocolObject},
};
use objc2_core_graphics::kCGColorSpaceSRGB;
use objc2_core_media::{CMSampleBuffer, CMTime, CMTimeFlags};
use objc2_core_video::kCVPixelFormatType_32BGRA;
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString};
use objc2_screen_capture_kit::{
    SCContentFilter, SCFrameStatus, SCStream, SCStreamConfiguration, SCStreamDelegate,
    SCStreamFrameInfoStatus, SCStreamOutput, SCStreamOutputType,
};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::rate::{FrameRate, FrameRateHandle};
use crate::{
    buffer::{MediaBuffer, picture_is_referenced, release_picture},
    contract::{MediaKind, MemoryDomain, OutputContract, PixelLayoutSet, PortContract},
    element::{Element, ElementType, Produced, Source, SourceStage, Wait, element_pp_log},
    elements::{ScreenCaptureKitDisplay, ScreenCaptureKitWindow, VideoFormat},
    error::Result,
    platform::macos::{
        pixel_buffer::{PixelBuffer, PixelBufferError},
        screencapturekit::{self, ContentError},
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
    produce::source_stage,
    schedule::PeriodicSchedule,
};

/// How long the source's thread waits at most before looking at what the
/// stream has said, whatever the rate.
const POLL_GRANULARITY: Duration = Duration::from_millis(100);
/// How many pixel buffers ScreenCaptureKit keeps for the stream — the most
/// it allows is 8. This holds one as the picture it offers, and a queue
/// downstream a few more before they are released; with too few the stream
/// has nowhere to draw the next and skips it.
const QUEUE_DEPTH: isize = 6;

/// Errors specific to [`ScreenCaptureKitSource`]. Converts into the
/// crate-wide [`crate::error::Error`] via `?`.
#[derive(Debug, ThisError)]
pub enum ScreenCaptureKitSourceError {
    /// This program may not record the screen. macOS asks the user the
    /// first time and sends them to System Settings, Privacy & Security,
    /// Screen & System Audio Recording, where it is allowed — after which
    /// the program has to be started again.
    #[error("this program may not record the screen")]
    PermissionDenied,

    /// ScreenCaptureKit would not say what there is to capture.
    #[error("ScreenCaptureKit would not list what there is to capture: {0}")]
    Content(String),

    /// The display asked for is not there — disconnected since it was
    /// listed.
    #[error("display {0} is not there")]
    DisplayNotFound(u32),

    /// The window asked for is not there — closed, or off screen, since it
    /// was listed.
    #[error("window {0} is not there")]
    WindowNotFound(u32),

    /// [`ScreenCaptureKitOptions::frame_rate`]'s numerator or denominator
    /// is not positive. Refused before anything is asked of the system.
    #[error("invalid frame rate {0}; numerator and denominator must both be positive")]
    InvalidFrameRate(ffmpeg::Rational),

    /// The stream could not be made, started or stopped.
    #[error("the ScreenCaptureKit stream failed: {0}")]
    Stream(String),

    /// The stream stopped by itself: what it captured is gone — a window
    /// closed, a display disconnected — or the user stopped it. Terminal,
    /// as `PipeWireScreenCaptureSourceError::SourceGone` is; reopening is
    /// the caller's decision.
    #[error("the captured display or window is gone: {0}")]
    SourceGone(String),

    /// The stream delivered a layout other than the BGRA it was asked for —
    /// its four-character code.
    #[error("the stream delivered pixel format {0:#010x}, not BGRA")]
    NotBgra(u32),

    /// A picture could not be read.
    #[error("a picture could not be read")]
    Unreadable,

    /// A VideoToolbox frames context for the pictures could not be made.
    #[cfg(feature = "videotoolbox")]
    #[error("the VideoToolbox frames context could not be made: {0}")]
    HwFrames(String),
}

impl From<PixelBufferError> for ScreenCaptureKitSourceError {
    fn from(error: PixelBufferError) -> Self {
        match error {
            PixelBufferError::Unexpected(code) => Self::NotBgra(code),
            PixelBufferError::Lock(_) | PixelBufferError::Truncated => Self::Unreadable,
        }
    }
}

impl From<ContentError> for ScreenCaptureKitSourceError {
    fn from(error: ContentError) -> Self {
        match error {
            ContentError::PermissionDenied => Self::PermissionDenied,
            ContentError::Failed(reason) => Self::Content(reason),
        }
    }
}

/// What to capture — see [`ScreenCaptureKitSource::list_displays`] and
/// [`ScreenCaptureKitSource::list_windows`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreenCaptureKitTarget {
    /// A whole display, by [`ScreenCaptureKitDisplay::id`]: the desktop,
    /// every window on it, the menu bar and the Dock.
    Display(u32),
    /// One window, by [`ScreenCaptureKitWindow::id`]: its own content only,
    /// wherever it is and whatever covers it.
    Window(u32),
}

/// What to open, and how.
#[derive(Debug, Clone)]
pub struct ScreenCaptureKitOptions {
    /// What to capture.
    pub target: ScreenCaptureKitTarget,
    /// The constant rate frames are emitted at — a fixed *output* rate, as
    /// `PipeWireScreenCaptureOptions::frame_rate` and
    /// `DxgiCaptureOptions::frame_rate` are, not a cap on the stream's own
    /// rate, which follows what changes on screen. A fraction, so
    /// `30000/1001` is expressible. One that is not positive is refused
    /// with [`ScreenCaptureKitSourceError::InvalidFrameRate`].
    pub frame_rate: ffmpeg::Rational,
    /// Whether the system draws the pointer into the captured pixels.
    pub include_cursor: bool,
}

impl ScreenCaptureKitOptions {
    /// `target` at `30/1`, without the pointer — the defaults every other
    /// screen capture in this crate has.
    pub fn new(target: ScreenCaptureKitTarget) -> Self {
        Self {
            target,
            frame_rate: ffmpeg::Rational::new(30, 1),
            include_cursor: false,
        }
    }
}

/// A display or one window on macOS, delivering BGRA frames at a fixed rate
/// — the macOS counterpart of `DxgiCaptureSource`, `WgcCaptureSource` and
/// `PipeWireScreenCaptureSource`, in their shape: `open` answers with the
/// source and the size it captures at, and what it captures going away ends
/// the source with an error rather than waiting for it to come back.
///
/// [`Self::open`] delivers them in system memory; with `videotoolbox`,
/// `open_videotoolbox` delivers the pixel buffers ScreenCaptureKit drew
/// themselves, as VideoToolbox frames, for a `VideoToolboxEncoder` to take
/// with nothing copied.
///
/// # A fixed output rate
///
/// ScreenCaptureKit delivers a picture when something on screen changes,
/// at most as often as the rate this was opened at, and nothing while
/// nothing does. This emits at the rate throughout, the latest picture
/// each tick: a tick that finds nothing new offers the picture it already
/// has under its own `pts`, pointing a new frame at the same pixels rather
/// than copying them again — what lets `SwScaler`, a compositor and
/// `ChangeGate` downstream recognise the repeat. `pts` counts ticks, so
/// [`Self::time_base`] is the reciprocal of the rate. [`Self::frame_rate`]
/// changes how often this emits, not how often the stream draws.
///
/// # Size
///
/// A display is captured at its size in pixels — twice its size in points
/// each way on a Retina display — and a window at its own, on the display
/// it is mostly on, as it was when opened. That size holds for the whole
/// capture: a window resized afterwards is scaled to fit it, its aspect
/// kept, rather than changing what everything downstream was built for.
///
/// # Frame format
///
/// `Pixel::BGRA`, in sRGB, which the stream is asked to convert the
/// display's own colour space to, tagged `color_space = RGB` /
/// `color_range = JPEG` — the full-range RGB contract every screen capture
/// in this crate has.
///
/// # Permission
///
/// Recording the screen needs the user's permission, given in System
/// Settings to the application responsible for this program — its own,
/// bundled, or the terminal's that started it. `open` asks the first time,
/// which macOS does by showing a prompt and sending the user to System
/// Settings, and returns [`ScreenCaptureKitSourceError::PermissionDenied`]
/// at once; a program allowed there has to be started again before it may.
pub struct ScreenCaptureKitSource(SourceStage<Capturing>);

source_stage!(ScreenCaptureKitSource);

/// What the stream has said, written on its queue and read on the source's
/// thread.
#[derive(Default)]
struct Latest {
    /// The last complete picture.
    picture: Option<PixelBuffer>,
    /// Bumped with each picture, so a tick can tell a new one from the one
    /// it already offers.
    captures: u64,
    /// Why the stream stopped, where it stopped by itself.
    stopped: Option<String>,
}

/// What the stream's delegate holds.
struct DelegateIvars {
    latest: Arc<Mutex<Latest>>,
}

define_class!(
    /// The stream's output and delegate: keeps the latest complete picture,
    /// and why the stream stopped, where it stopped by itself.
    #[unsafe(super(NSObject))]
    #[name = "MediaPpScreenCaptureKitDelegate"]
    #[ivars = DelegateIvars]
    struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl SCStreamOutput for Delegate {
        #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
        fn did_output(
            &self,
            _stream: &SCStream,
            sample: &CMSampleBuffer,
            kind: SCStreamOutputType,
        ) {
            if kind != SCStreamOutputType::Screen || !is_complete(sample) {
                return;
            }
            // SAFETY: ScreenCaptureKit hands over a live sample buffer for
            // the length of this call; its pixel buffer is retained to
            // outlive it.
            let Some(image) = (unsafe { sample.image_buffer() }) else {
                return;
            };
            if let Ok(mut latest) = self.ivars().latest.lock() {
                latest.picture = Some(PixelBuffer::new(image));
                latest.captures += 1;
            }
        }
    }

    unsafe impl SCStreamDelegate for Delegate {
        #[unsafe(method(stream:didStopWithError:))]
        fn did_stop(&self, _stream: &SCStream, error: &NSError) {
            if let Ok(mut latest) = self.ivars().latest.lock() {
                latest
                    .stopped
                    .get_or_insert_with(|| error.localizedDescription().to_string());
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

/// Whether `sample` holds a new picture. ScreenCaptureKit also sends
/// samples that say nothing changed, or that the stream started or
/// stopped, some with a pixel buffer of what was there before — only a
/// complete one is a picture to show.
fn is_complete(sample: &CMSampleBuffer) -> bool {
    // SAFETY: a live sample buffer; its attachments, where it has them, are
    // an array of dictionaries keyed by strings, toll-free bridged to
    // Foundation's, and read for the length of this call.
    unsafe {
        let Some(attachments) = sample.sample_attachments_array(false) else {
            return false;
        };
        let attachments = &*(objc2_core_foundation::CFRetained::as_ptr(&attachments).as_ptr()
            as *const NSArray<NSDictionary<NSString, AnyObject>>);
        let Some(first) = attachments.firstObject() else {
            return false;
        };
        first
            .objectForKey(SCStreamFrameInfoStatus)
            .and_then(|status| status.downcast::<NSNumber>().ok())
            .is_some_and(|status| status.integerValue() == SCFrameStatus::Complete.0)
    }
}

/// Where the pictures go.
enum Output {
    /// Copied into BGRA frames in system memory, from this pool.
    System(UnboundObjectPool<ffmpeg::frame::Video>),
    /// Handed on themselves, stamped with a frames context of their size,
    /// each held in a slot of this pool.
    #[cfg(feature = "videotoolbox")]
    VideoToolbox {
        device: Arc<crate::platform::ffmpeg::AvBufferRef>,
        frames: Option<(u32, u32, crate::platform::ffmpeg::AvBufferRef)>,
        pool: UnboundObjectPool<ffmpeg::frame::Video>,
    },
}

/// What a [`ScreenCaptureKitSource`] hands on, a picture a tick: all of its
/// work, which the framework makes the source.
struct Capturing {
    name: Arc<str>,
    pp_log: PpLog,
    stream: Retained<SCStream>,
    /// Kept for as long as the stream delivers through them.
    _delegate: Retained<Delegate>,
    _queue: dispatch2::DispatchRetained<dispatch2::DispatchQueue>,
    running: bool,
    latest: Arc<Mutex<Latest>>,
    output: Output,
    /// When each frame is due, on the clock a pause does not move — made as
    /// the first is asked for.
    schedule: Option<PeriodicSchedule>,
    /// Whether a tick's work was done since the schedule last moved.
    ticked: bool,
    frame_rate: Arc<FrameRate>,
    /// The emitted `pts`: ticks since the first.
    frame_index: i64,
    /// Empty frames pointed at `last_picture`, each carrying one tick's
    /// timestamp.
    wrapper_pool: UnboundObjectPool<ffmpeg::frame::Video>,
    /// The picture offered, kept so a tick with nothing new can point a
    /// wrapper at it rather than make it again.
    last_picture: Option<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
    /// The `Latest::captures` count behind `last_picture`.
    picture_captures: u64,
    /// Pictures a wrapper already pushed downstream may still be pointing
    /// at: a replaced picture waits here until nothing but itself references
    /// its pixels, so its pool slot is not written over while they show.
    retired: Vec<Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>>,
}

// SAFETY: ScreenCaptureKit's stream may be started and stopped from any
// thread, and the delegate and queue are only ever held here, never called.
// Every method that touches them takes `&mut self`, and a source is driven
// by one worker thread.
unsafe impl Send for Capturing {}

impl ScreenCaptureKitSource {
    /// Every display there is, the main one marked.
    ///
    /// Needs the permission to record the screen, as opening does:
    /// [`ScreenCaptureKitSourceError::PermissionDenied`] without it.
    pub fn list_displays()
    -> std::result::Result<Vec<ScreenCaptureKitDisplay>, ScreenCaptureKitSourceError> {
        let content = shareable_content()?;
        // SAFETY: plain queries of live display objects.
        let displays = unsafe { content.displays() };
        Ok(displays
            .iter()
            .map(|display| screencapturekit::describe_display(&display))
            .collect())
    }

    /// Every window worth capturing, on this Space or another: an
    /// application's, at the normal level — not the menu bar, the Dock or a
    /// status item — front to back. One not on screen is listed where it
    /// has a title; a minimized one captures nothing until it is restored.
    ///
    /// Needs the permission to record the screen, as opening does.
    pub fn list_windows()
    -> std::result::Result<Vec<ScreenCaptureKitWindow>, ScreenCaptureKitSourceError> {
        let content = shareable_content()?;
        Ok(screencapturekit::capturable_windows(&content)
            .iter()
            .map(|window| screencapturekit::describe_window(&content, window))
            .collect())
    }

    /// Opens a stream of `options.target`, delivering BGRA frames in system
    /// memory. The stream starts as the source does, on its own thread.
    pub fn open(
        name: impl Into<String>,
        options: ScreenCaptureKitOptions,
    ) -> std::result::Result<(Self, VideoFormat), ScreenCaptureKitSourceError> {
        Self::assemble(name.into(), options, |width, height| {
            Output::System(UnboundObjectPool::new(
                0,
                move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height),
                |_| {},
            ))
        })
    }

    /// Opens the same stream, but hands on the pixel buffers ScreenCaptureKit
    /// drew as VideoToolbox frames holding BGRA, on `device` — the one every
    /// VideoToolbox element in the pipeline shares.
    #[cfg(feature = "videotoolbox")]
    pub fn open_videotoolbox(
        name: impl Into<String>,
        options: ScreenCaptureKitOptions,
        device: &crate::elements::VideoToolboxDevice,
    ) -> std::result::Result<(Self, VideoFormat), ScreenCaptureKitSourceError> {
        let device = device.retain();
        Self::assemble(name.into(), options, move |_, _| Output::VideoToolbox {
            device,
            frames: None,
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, release_picture),
        })
    }

    fn assemble(
        name: String,
        options: ScreenCaptureKitOptions,
        output: impl FnOnce(u32, u32) -> Output,
    ) -> std::result::Result<(Self, VideoFormat), ScreenCaptureKitSourceError> {
        crate::ensure_ffmpeg();
        let rate = options.frame_rate;
        if rate.numerator() <= 0 || rate.denominator() <= 0 {
            return Err(ScreenCaptureKitSourceError::InvalidFrameRate(rate));
        }
        let name: Arc<str> = name.into();
        let pp_log = element_pp_log(ElementType::ScreenCaptureKitSource, &name, None);
        let content = shareable_content()?;

        // SAFETY: fresh content filters made of the content's own live
        // display or window objects.
        let (filter, (width, height)) = unsafe {
            match options.target {
                ScreenCaptureKitTarget::Display(id) => {
                    let display = screencapturekit::find_display(&content, id)
                        .ok_or(ScreenCaptureKitSourceError::DisplayNotFound(id))?;
                    let described = screencapturekit::describe_display(&display);
                    (
                        SCContentFilter::initWithDisplay_excludingWindows(
                            SCContentFilter::alloc(),
                            &display,
                            &NSArray::new(),
                        ),
                        (described.width, described.height),
                    )
                }
                ScreenCaptureKitTarget::Window(id) => {
                    let window = screencapturekit::find_window(&content, id)
                        .ok_or(ScreenCaptureKitSourceError::WindowNotFound(id))?;
                    (
                        SCContentFilter::initWithDesktopIndependentWindow(
                            SCContentFilter::alloc(),
                            &window,
                        ),
                        screencapturekit::window_pixels(&content, &window),
                    )
                }
            }
        };

        // SAFETY: plain setters on a fresh configuration; the colour space
        // name is Core Graphics' own constant.
        let configuration = unsafe {
            let configuration = SCStreamConfiguration::new();
            configuration.setWidth(width as usize);
            configuration.setHeight(height as usize);
            configuration.setPixelFormat(kCVPixelFormatType_32BGRA);
            configuration.setColorSpaceName(kCGColorSpaceSRGB);
            configuration.setMinimumFrameInterval(CMTime {
                value: i64::from(rate.denominator()),
                timescale: rate.numerator(),
                flags: CMTimeFlags::Valid,
                epoch: 0,
            });
            configuration.setShowsCursor(options.include_cursor);
            configuration.setScalesToFit(true);
            configuration.setQueueDepth(QUEUE_DEPTH);
            configuration
        };

        let latest = Arc::new(Mutex::new(Latest::default()));
        let delegate = Delegate::new(DelegateIvars {
            latest: Arc::clone(&latest),
        });
        let queue = dispatch2::DispatchQueue::new("media-pp.screencapturekit", None);
        // SAFETY: the filter and configuration are live and copied by the
        // stream; the delegate and queue are kept for as long as the stream
        // delivers through them.
        let stream = unsafe {
            let stream = SCStream::initWithFilter_configuration_delegate(
                SCStream::alloc(),
                &filter,
                &configuration,
                Some(ProtocolObject::from_ref(&*delegate)),
            );
            stream
                .addStreamOutput_type_sampleHandlerQueue_error(
                    ProtocolObject::from_ref(&*delegate),
                    SCStreamOutputType::Screen,
                    Some(&queue),
                )
                .map_err(|error| {
                    ScreenCaptureKitSourceError::Stream(error.localizedDescription().to_string())
                })?;
            stream
        };

        let format = VideoFormat {
            width,
            height,
            time_base: rate.invert(),
        };
        let output = output(width, height);
        pp_info!(
            pp_log: &pp_log,
            "opened: {:?}, {}x{} at {} fps, BGRA{}, include_cursor={}",
            options.target,
            width,
            height,
            rate,
            if matches!(output, Output::System(_)) { "" } else { " in VideoToolbox frames" },
            options.include_cursor
        );
        Ok((
            Self(SourceStage::new(Capturing {
                name,
                pp_log,
                stream,
                _delegate: delegate,
                _queue: queue,
                running: false,
                latest,
                output,
                schedule: None,
                ticked: false,
                frame_rate: FrameRate::new(rate),
                frame_index: 0,
                wrapper_pool: UnboundObjectPool::new(
                    0,
                    ffmpeg::frame::Video::empty,
                    release_picture,
                ),
                last_picture: None,
                picture_captures: 0,
                retired: Vec::new(),
            })),
            format,
        ))
    }

    /// The unit each emitted frame's `pts` counts in: the reciprocal of the
    /// rate, since `pts` counts ticks.
    pub fn time_base(&self) -> ffmpeg::Rational {
        self.0.inner().frame_rate.get().invert()
    }

    /// Runtime control for the rate this emits at.
    ///
    /// Taken before this is moved into a `Pipeline`, which is the only
    /// chance to. Changing the rate re-means [`Self::time_base`] and every
    /// timestamp after the change — see [`crate::rate`]. The stream keeps
    /// drawing at most as often as the rate it was opened at; a tick that
    /// finds nothing new answers with the picture it already has.
    pub fn frame_rate(&self) -> FrameRateHandle {
        self.0.inner().frame_rate.handle()
    }
}

/// What there is to capture, the permission asked for where it was not
/// given yet.
fn shareable_content() -> std::result::Result<
    Retained<objc2_screen_capture_kit::SCShareableContent>,
    ScreenCaptureKitSourceError,
> {
    if !screencapturekit::may_record() {
        return Err(ScreenCaptureKitSourceError::PermissionDenied);
    }
    Ok(screencapturekit::shareable_content()?)
}

impl Capturing {
    fn start(&mut self) -> std::result::Result<(), ScreenCaptureKitSourceError> {
        if !self.running {
            screencapturekit::complete("start the stream", |handler| {
                // SAFETY: a configured stream; the handler is copied and
                // called once.
                unsafe { self.stream.startCaptureWithCompletionHandler(Some(handler)) }
            })
            .map_err(ScreenCaptureKitSourceError::Stream)?;
            self.running = true;
        }
        Ok(())
    }

    fn stop(&mut self) {
        if self.running {
            self.running = false;
            if let Err(reason) = screencapturekit::complete("stop the stream", |handler| {
                // SAFETY: as for `start`.
                unsafe { self.stream.stopCaptureWithCompletionHandler(Some(handler)) }
            }) {
                pp_warn!(self, "the stream did not stop cleanly: {reason}");
            }
        }
    }

    /// The latest picture, made the one offered where it is new, under this
    /// tick's own `pts`; `None` before the first.
    fn emit_frame(&mut self) -> Result<Option<UnboundObjectPoolRef<ffmpeg::frame::Video>>> {
        let (picture, captures) = {
            let latest = self.latest.lock().map_err(|_| {
                ScreenCaptureKitSourceError::Stream("the stream's delegate panicked".into())
            })?;
            match &latest.picture {
                Some(picture) if latest.captures != self.picture_captures => {
                    (Some(picture.clone()), latest.captures)
                }
                _ => (None, latest.captures),
            }
        };
        if let Some(picture) = picture {
            self.make_picture(picture)?;
            self.picture_captures = captures;
        }
        let Some(picture) = self.last_picture.as_ref() else {
            return Ok(None);
        };
        let mut wrapper = self.wrapper_pool.get();
        // SAFETY: `ptr` is the pooled wrapper's own `AVFrame`, unreferenced
        // before it is given a new one, and the source is the picture this
        // holds the pooled reference to — both live, and distinct.
        let referenced = unsafe {
            let ptr = wrapper.as_mut_ptr();
            ffi::av_frame_unref(ptr);
            ffi::av_frame_ref(ptr, picture.as_ptr())
        };
        if referenced < 0 {
            return Err(ScreenCaptureKitSourceError::Unreadable.into());
        }
        wrapper.set_pts(Some(self.frame_index));
        crate::buffer::set_time_base(&mut wrapper, self.frame_rate.get().invert());
        self.frame_index += 1;
        Ok(Some(wrapper))
    }

    /// Makes `picture` the one offered from here on: copied into a frame
    /// the pool considers free, or itself as a VideoToolbox frame. Never
    /// over the previous one, which earlier wrappers may still be showing.
    fn make_picture(&mut self, picture: PixelBuffer) -> Result<()> {
        if let Some(previous) = self.last_picture.take() {
            self.retired.push(previous);
        }
        self.retired
            .retain(|picture| picture_is_referenced(picture));
        let frame = match &mut self.output {
            Output::System(pool) => {
                let mut frame = pool.get();
                picture
                    .copy_to(&mut frame, ffmpeg::format::Pixel::BGRA)
                    .map_err(ScreenCaptureKitSourceError::from)?;
                frame
            }
            #[cfg(feature = "videotoolbox")]
            Output::VideoToolbox {
                device,
                frames,
                pool,
            } => {
                let (width, height) = picture.size();
                if !frames
                    .as_ref()
                    .is_some_and(|&(w, h, _)| (w, h) == (width, height))
                {
                    // SAFETY: `device` is this source's own reference to a
                    // live VideoToolbox device context.
                    let made = unsafe {
                        crate::platform::macos::videotoolbox::create_frames_ctx(
                            device,
                            ffmpeg::format::Pixel::BGRA,
                            width,
                            height,
                        )
                    }
                    .map_err(|error| ScreenCaptureKitSourceError::HwFrames(error.to_string()))?;
                    *frames = Some((width, height, made));
                }
                let (_, _, frames_ctx) = frames.as_ref().expect("made above");
                let mut slot = pool.get();
                *slot = picture
                    .into_videotoolbox_frame(frames_ctx)
                    .map_err(ScreenCaptureKitSourceError::from)?;
                slot
            }
        };
        self.last_picture = Some(Arc::new(frame));
        Ok(())
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
        ElementType::ScreenCaptureKitSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
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
            PortContract::frame(MediaKind::VideoFrame, domain).with_layouts(PixelLayoutSet::BGRA),
        )
    }

    /// The latest picture, once its tick is due on the schedule — kept on
    /// the clock a pause does not move, so playing on after one is not a
    /// burst of the ticks it would have owed.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        let now = wait.now();
        let interval = self.frame_rate.interval();
        let schedule = self
            .schedule
            .get_or_insert_with(|| PeriodicSchedule::new(interval, now));
        if std::mem::take(&mut self.ticked) {
            schedule.advance_after_tick(now);
        }
        let rate_changed = schedule.interval() != interval;
        if rate_changed {
            schedule.set_interval(interval, now);
        }
        let due = now + schedule.remaining(now);
        if rate_changed {
            pp_info!(self, "frame rate is now {}", self.frame_rate.get());
        }

        let stopped = self
            .latest
            .lock()
            .ok()
            .and_then(|mut latest| latest.stopped.take());
        if let Some(reason) = stopped {
            // The stream is stopped already; nothing is left to stop.
            self.running = false;
            pp_error!(self, "the stream stopped: {reason}");
            return Err(ScreenCaptureKitSourceError::SourceGone(reason).into());
        }

        if !wait.until(due.min(now + POLL_GRANULARITY)) || wait.now() < due {
            return Ok(Produced::Nothing);
        }
        self.ticked = true;
        Ok(self.emit_frame()?.map_or(Produced::Nothing, |frame| {
            Produced::Buffer(MediaBuffer::Video(Arc::new(frame).into()))
        }))
    }

    /// Starts the stream on the source's own thread.
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

    /// A rate that is not positive is refused before the system is asked
    /// anything — no permission prompt for a capture that could not run.
    #[test]
    fn a_rate_that_is_not_positive_is_refused_first() {
        for rate in [ffmpeg::Rational::new(0, 1), ffmpeg::Rational::new(30, 0)] {
            let options = ScreenCaptureKitOptions {
                frame_rate: rate,
                ..ScreenCaptureKitOptions::new(ScreenCaptureKitTarget::Display(0))
            };
            assert!(matches!(
                ScreenCaptureKitSource::open("screen", options),
                Err(ScreenCaptureKitSourceError::InvalidFrameRate(_))
            ));
        }
    }

    /// The main display, where this program may already record the screen;
    /// never asks, since asking shows the user a prompt.
    fn try_main_display() -> Option<ScreenCaptureKitDisplay> {
        if !objc2_core_graphics::CGPreflightScreenCaptureAccess() {
            eprintln!("skipping: this program has not been allowed to record the screen");
            return None;
        }
        ScreenCaptureKitSource::list_displays()
            .expect("the displays are listed")
            .into_iter()
            .find(|display| display.is_main)
    }

    /// A display delivers full-range RGB BGRA of the size it was opened at,
    /// a frame a tick counted from zero, and never faster than the rate asked
    /// for — whether anything on it changes or not.
    ///
    /// Not how many in a given time: a hosted Mac's timers fire late, and
    /// its stream is slow to start, so a count per second is the machine's.
    /// Late ticks only make fewer frames, so the rate still bounds them from
    /// above.
    #[test]
    fn a_display_delivers_bgra_a_frame_a_tick() {
        let Some(display) = try_main_display() else {
            return;
        };
        let options = ScreenCaptureKitOptions {
            frame_rate: ffmpeg::Rational::new(20, 1),
            ..ScreenCaptureKitOptions::new(ScreenCaptureKitTarget::Display(display.id))
        };
        let (source, format) = ScreenCaptureKitSource::open("screen", options).expect("opens");
        assert_eq!(
            (format.width, format.height),
            (display.width, display.height)
        );
        assert_eq!(format.time_base, ffmpeg::Rational::new(1, 20));
        let frames = Arc::new(Mutex::new(Vec::new()));
        let sink = crate::elements::AppSink::new("screen-sink", {
            let frames = Arc::clone(&frames);
            move |buffer| {
                if let MediaBuffer::Video(frame) = &buffer {
                    frames.lock().unwrap().push((
                        frame.format(),
                        frame.color_space(),
                        frame.color_range(),
                        frame.width(),
                        frame.height(),
                        frame.pts(),
                    ));
                }
                Ok(())
            }
        });
        let (pipeline, ()) = crate::pipeline::Pipeline::new("screen", source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring");
        let started = std::time::Instant::now();
        pipeline.run().expect("the stream starts");
        let deadline = started + Duration::from_secs(10);
        while frames.lock().unwrap().len() < 20 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        pipeline.stop();
        let took = started.elapsed().as_secs_f64();
        let frames = frames.lock().unwrap();
        assert!(frames.len() >= 20, "{} frames in {took:.1} s", frames.len());
        assert!(
            frames.len() as f64 <= 20.0 * took + 1.0,
            "{} frames in {took:.1} s is faster than 20 a second",
            frames.len()
        );
        for (index, &(pixel, space, range, width, height, pts)) in frames.iter().enumerate() {
            assert_eq!(pixel, ffmpeg::format::Pixel::BGRA);
            assert_eq!(
                (space, range),
                (ffmpeg::color::Space::RGB, ffmpeg::color::Range::JPEG)
            );
            assert_eq!((width, height), (format.width, format.height));
            assert_eq!(pts, Some(index as i64), "a tick each");
        }
    }
}
