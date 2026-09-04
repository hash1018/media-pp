use std::{ptr, sync::Arc, time::Duration};

use ffmpeg_next as ffmpeg;
use thiserror::Error as ThisError;
use windows::Win32::Media::MediaFoundation::{
    IMFMediaSource, IMFMediaType, IMFSample, IMFSourceReader, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE,
    MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED,
    MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR, MFCreateMediaType, MFMediaType_Video,
    MFVideoFormat_NV12,
};

use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{self, ControlMsg, ControlReceiver, RequestKind},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::VideoFormat,
    error::Result,
    pad::SrcPad,
    platform::windows::{
        com::ComApartment,
        mf::{
            MfCaptureFormat, MfDevice, MfRuntime, frame_rate, frame_size, list_devices,
            list_formats, open_device_source, open_reader,
        },
    },
    pool::{UnboundObjectPool, UnboundObjectPoolRef},
};

/// The unit every emitted `pts` counts in: Media Foundation's own 100ns
/// tick. See [`MfCaptureSource::time_base`] for why the device's clock
/// rather than `1 / fps`.
const TIME_BASE_DENOMINATOR: i32 = 10_000_000;

/// Errors specific to `MfCaptureSource`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum MfCaptureSourceError {
    /// A COM or Media Foundation operation failed.
    #[error("windows error: {0}")]
    Windows(#[from] windows::core::Error),

    /// [`MfCaptureOptions::device`] names a device that is not attached.
    #[error("no video capture device with symbolic link {0:?}")]
    DeviceNotFound(String),

    /// [`MfCaptureOptions::format`] names a mode this camera does not offer.
    #[error("this camera offers no {width}x{height} mode at {framerate} fps")]
    FormatNotOffered {
        /// Requested frame width in pixels.
        width: u32,
        /// Requested frame height in pixels.
        height: u32,
        /// Requested frames per second.
        framerate: ffmpeg::Rational,
    },

    /// The camera negotiated a frame size NV12 cannot describe. Its chroma
    /// planes are half-sized in both axes, so an odd dimension has no whole
    /// pixel to carry.
    #[error("negotiated an odd {width}x{height} frame, which NV12 cannot carry")]
    OddFrameSize {
        /// Negotiated frame width in pixels.
        width: u32,
        /// Negotiated frame height in pixels.
        height: u32,
    },

    /// The device stopped mid-capture — unplugged, taken by a driver reset,
    /// or handed to something with a higher claim on it. Broken out of
    /// [`MfCaptureSourceError::Windows`] for the same reason
    /// `DxgiCaptureSourceError::AccessLost` is: it is the one common
    /// *recoverable* failure, and the contract is the same "fail fast, the
    /// caller builds a fresh one" every live source in this crate has.
    #[error("the capture device is gone — it needs to be reopened")]
    DeviceLost,

    /// The camera changed its own output format mid-capture. Every frame
    /// this element emits is allocated at the size negotiated in `open`, so
    /// a new one is not something it can carry — the caller reopens.
    #[error("the capture device changed format mid-capture — it needs to be reopened")]
    FormatChanged,

    /// One sample carried less data than its negotiated frame needs.
    #[error("a {width}x{height} NV12 frame needs {needed} bytes, and the sample carried {got}")]
    ShortSample {
        /// Negotiated frame width in pixels.
        width: u32,
        /// Negotiated frame height in pixels.
        height: u32,
        /// Bytes one whole NV12 frame at that size occupies.
        needed: usize,
        /// Bytes the sample actually carried.
        got: usize,
    },

    /// Seeking was requested on a live camera.
    #[error("MfCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,
}

/// Construction-time options for [`MfCaptureSource::open`].
#[derive(Debug, Clone)]
pub struct MfCaptureOptions {
    /// Which camera to capture — one entry out of
    /// [`MfCaptureSource::list_devices`], or a hand-built one if the caller
    /// already knows a device's symbolic link some other way.
    pub device: MfDevice,
    /// Which of [`MfCaptureSource::list_formats`] to ask that camera for.
    ///
    /// `None` takes the device's own first offered mode, which is the only
    /// preference a camera states. It is frequently *not* the largest one:
    /// a webcam usually lists an uncompressed mode first and keeps its
    /// higher resolutions for MJPEG, so a caller that wants the best
    /// picture should pick from `list_formats` rather than leave this
    /// alone.
    pub format: Option<MfCaptureFormat>,
}

/// Captures a camera through Media Foundation's synchronous source reader —
/// GStreamer's `mfvideosrc` equivalent. One src pad, pushing CPU-resident
/// `Pixel::NV12` [`MediaBuffer::Video`] frames at the negotiated size, ready
/// for [`D3d11Upload`](crate::elements::D3d11Upload) (which takes NV12
/// directly) or a [`SwScaler`](crate::elements::SwScaler).
///
/// # NV12 whatever the camera speaks
///
/// A webcam offers the same picture in several subtypes — commonly YUY2 or
/// NV12 at its lower resolutions and MJPEG at its higher ones, because that
/// is what fits down a USB 2.0 pipe. This element asks for NV12 and lets
/// Media Foundation insert its own MJPEG decoder and colour converter where
/// one is needed (see `MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING`), so a
/// caller picks a *shape* — width, height, rate — and never a subtype.
/// [`MfCaptureFormat`] is that choice and nothing more.
///
/// What it costs is that MJPEG modes are decoded on the CPU. That is the
/// same trade `DxgiCaptureSource` makes emitting BGRA, and the honest one
/// here: a camera's own hardware decoder is not reachable through the
/// synchronous reader.
///
/// # Timestamps
///
/// `pts` counts the device's own clock in 100ns ticks (see
/// [`MfCaptureSource::time_base`]), rebased so the first frame of a run
/// leaves at zero. Pause does not advance it: the whole frozen interval
/// collapses to one nominal frame interval, so Resume continues the
/// timeline rather than leaving a hole the length of the pause in it.
///
/// # Ending is a disconnection
///
/// This element does not reconnect. A camera that is unplugged, reset, or
/// taken by an application with a higher claim on it ends the source with
/// [`MfCaptureSourceError::DeviceLost`], the pipeline finishes, and — since
/// a pipeline is one-shot — coming back means building a new one. The same
/// contract [`RtspSource`](crate::elements::RtspSource) and
/// `DxgiCaptureSource` document for their own losses.
///
/// # Stop latency
///
/// `IMFSourceReader::ReadSample` is synchronous and has no timeout, so
/// control is looked at once per delivered frame — a bound of one frame
/// interval at the negotiated rate, and the reason a mode's rate is worth
/// picking deliberately. A camera that stops delivering without failing
/// would leave that call parked; in practice a device that goes away fails
/// the read rather than going quiet, which is what makes this bounded.
pub struct MfCaptureSource {
    pp_log: PpLog,
    name: Arc<str>,
    /// Held for this element's whole life: the reader and the source below
    /// are only valid while Media Foundation is up.
    _runtime: MfRuntime,
    /// Kept alongside the reader so teardown can `Shutdown` it, which is
    /// what actually releases the camera. Dropping the last reference
    /// without that leaves the device held against the next open.
    source: IMFMediaSource,
    reader: IMFSourceReader,
    width: u32,
    height: u32,
    /// What the camera says its samples mean, which the D3D11 compositor
    /// reads off each frame to pick an NV12 conversion matrix. Leaving them
    /// unset makes it guess instead of using what this source produces.
    color_space: ffmpeg::color::Space,
    color_range: ffmpeg::color::Range,
    /// One nominal frame in [`TIME_BASE_DENOMINATOR`] ticks — what a Pause
    /// costs the timeline instead of its real duration.
    frame_interval: i64,
    /// Device time the current run counts from. Moved forward by each
    /// pause so `pts` stays continuous across one.
    origin: Option<i64>,
    /// Device time of the last frame pushed, which is what a resume rebases
    /// against.
    last_sample_time: Option<i64>,
    /// Set by Resume; consumed by the next sample that arrives.
    rebase: bool,
    /// Reused across every emitted frame — see [`UnboundObjectPool`]'s own
    /// docs. Each frame is written whole before it is pushed, so `release`
    /// has nothing to reset.
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
    pad: SrcPad,
}

// SAFETY: every handle here is a `windows-rs` COM interface wrapper.
// `IMFMediaSource` and the synchronous `IMFSourceReader` are documented
// free-threaded, so handing them to the source thread is sound, and `&mut
// self` on every method that touches them already rules out concurrent
// access — the same reasoning `DxgiCaptureSource` and `WasapiCaptureSource`
// document for their own.
unsafe impl Send for MfCaptureSource {}

impl Drop for MfCaptureSource {
    fn drop(&mut self) {
        // SAFETY: `source` is live and this is the last thing done with it;
        // `Shutdown` is idempotent and is what releases the camera.
        if let Err(error) = unsafe { self.source.Shutdown() } {
            pp_error!(self, "shutting the capture device down failed: {error}");
        }
    }
}

/// The stream index every call here reads, as the reader's own constant.
fn video_stream() -> u32 {
    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32
}

impl MfCaptureSource {
    /// Every video capture device currently attached, as an [`MfDevice`]
    /// list a caller can show in a picker and hand straight to
    /// [`MfCaptureOptions::device`].
    pub fn list_devices() -> std::result::Result<Vec<MfDevice>, MfCaptureSourceError> {
        Ok(list_devices()?)
    }

    /// Every picture shape `device` offers, deduplicated across the
    /// subtypes it would deliver each in — see [`MfCaptureFormat`].
    ///
    /// Opens the camera to ask and closes it again, which on most hardware
    /// lights its indicator for the duration. There is no way to enumerate
    /// a device's modes without opening it.
    pub fn list_formats(
        device: &MfDevice,
    ) -> std::result::Result<Vec<MfCaptureFormat>, MfCaptureSourceError> {
        let _apartment = ComApartment::new()?;
        let _runtime = MfRuntime::new()?;
        let source = open_device_source(&device.id)?;
        let formats = open_reader(&source).and_then(|reader| list_formats(&reader));
        // SAFETY: `source` is the live source opened above, and shutting it
        // down releases the camera whether or not enumeration succeeded.
        let _ = unsafe { source.Shutdown() };
        Ok(formats?)
    }

    /// Opens `options.device` at `options.format` and configures it to
    /// deliver NV12. Returns the element alongside the stream's actual
    /// [`VideoFormat`] — what a caller needs to build a matching downstream
    /// upload, scaler or encoder, the same pattern
    /// `DxgiCaptureSource::open` follows.
    ///
    /// The returned format is what the camera negotiated, not what was
    /// asked for. A mode is selected out of the device's own list, so the
    /// two normally agree, but the reader is free to answer with a
    /// neighbouring rate and this reports what it actually did.
    pub fn open(
        name: impl Into<String>,
        options: MfCaptureOptions,
    ) -> std::result::Result<(Self, VideoFormat), MfCaptureSourceError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::MfCaptureSource, &name, None);
        let _apartment = ComApartment::new()?;
        let runtime = MfRuntime::new()?;

        // Resolved against what is attached right now rather than trusted:
        // a stored symbolic link outlives the camera it named, and
        // `MFCreateDeviceSource` answers a stale one with a bare E_INVALIDARG
        // that says nothing about which device went missing.
        if !list_devices()?
            .iter()
            .any(|found| found.id == options.device.id)
        {
            return Err(MfCaptureSourceError::DeviceNotFound(
                options.device.id.clone(),
            ));
        }

        let source = open_device_source(&options.device.id)?;
        match Self::configure(&name, pp_log, runtime, &source, &options) {
            Ok(opened) => Ok(opened),
            Err(error) => {
                // The camera is held from `open_device_source` onward, so
                // every failure past it has to give it back. Without this a
                // rejected format would leave the device unopenable until
                // the process exits.
                // SAFETY: `source` is the live source opened above and
                // nothing else holds a reference to it.
                let _ = unsafe { source.Shutdown() };
                Err(error)
            }
        }
    }

    /// The half of `open` that runs with the camera already held, split out
    /// so one `Shutdown` covers every way it can fail.
    fn configure(
        name: &Arc<str>,
        pp_log: PpLog,
        runtime: MfRuntime,
        source: &IMFMediaSource,
        options: &MfCaptureOptions,
    ) -> std::result::Result<(Self, VideoFormat), MfCaptureSourceError> {
        let reader = open_reader(source)?;
        let native = Self::select_native_type(&reader, options)?;
        // Ask for the mode first, so the camera runs at the shape that was
        // picked, and only then for NV12 — asking for NV12 alone would let
        // the reader choose which of the device's modes to convert from.
        // SAFETY: `native` is one of this reader's own native types.
        unsafe { reader.SetCurrentMediaType(video_stream(), None, &native) }?;

        // SAFETY: the new type is filled in below before it is handed over,
        // and every key set on it is documented for a video media type.
        let nv12 = unsafe {
            let nv12 = MFCreateMediaType()?;
            nv12.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            nv12.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
            reader.SetCurrentMediaType(video_stream(), None, &nv12)?;
            nv12
        };
        drop(nv12);

        // What the reader settled on, which is the only shape the frames
        // below may be allocated at.
        // SAFETY: `reader` is live and the stream was just configured.
        let current = unsafe { reader.GetCurrentMediaType(video_stream()) }?;
        let (width, height) = frame_size(&current)?;
        let framerate = frame_rate(&current)?;
        if width % 2 != 0 || height % 2 != 0 {
            return Err(MfCaptureSourceError::OddFrameSize { width, height });
        }

        let (color_space, color_range) = describe_color(&current, height);

        let time_base = ffmpeg::Rational::new(1, TIME_BASE_DENOMINATOR);
        let frame_interval = if framerate.numerator() > 0 {
            i64::from(TIME_BASE_DENOMINATOR) * i64::from(framerate.denominator())
                / i64::from(framerate.numerator())
        } else {
            0
        };

        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?}, {}x{} NV12 at {} fps",
            options.device.name,
            width,
            height,
            framerate
        );

        let pad = SrcPad::with_contract(
            format!("{name}_src"),
            OutputContract::Fixed(PortContract::frame(
                MediaKind::VideoFrame,
                MemoryDomain::System,
            )),
        );

        Ok((
            Self {
                pp_log,
                name: name.clone(),
                _runtime: runtime,
                source: source.clone(),
                reader,
                width,
                height,
                color_space,
                color_range,
                frame_interval,
                origin: None,
                last_sample_time: None,
                rebase: false,
                pool: UnboundObjectPool::new(
                    0,
                    move || ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height),
                    |_| {},
                ),
                pad,
            },
            VideoFormat {
                width,
                height,
                time_base,
            },
        ))
    }

    /// The device's own media type for the requested mode, or its first one
    /// where no mode was requested.
    fn select_native_type(
        reader: &IMFSourceReader,
        options: &MfCaptureOptions,
    ) -> std::result::Result<
        windows::Win32::Media::MediaFoundation::IMFMediaType,
        MfCaptureSourceError,
    > {
        let Some(wanted) = options.format else {
            // SAFETY: `reader` is live; index zero is the device's own first
            // offered type, which every capture device has.
            return Ok(unsafe { reader.GetNativeMediaType(video_stream(), 0) }?);
        };
        for index in 0.. {
            // SAFETY: `reader` is live. Walking the index until it runs out
            // is how this list is enumerated; `list_formats` bounds it the
            // same way and this stops at the first failure.
            let Ok(media_type) = (unsafe { reader.GetNativeMediaType(video_stream(), index) })
            else {
                break;
            };
            let (Ok((width, height)), Ok(framerate)) =
                (frame_size(&media_type), frame_rate(&media_type))
            else {
                continue;
            };
            if width == wanted.width && height == wanted.height && framerate == wanted.framerate {
                return Ok(media_type);
            }
        }
        Err(MfCaptureSourceError::FormatNotOffered {
            width: wanted.width,
            height: wanted.height,
            framerate: wanted.framerate,
        })
    }

    /// The unit each emitted frame's `pts` is expressed in: Media
    /// Foundation's own 100ns tick, so `1 / 10_000_000`.
    ///
    /// The device's clock rather than `1 / fps`, unlike
    /// [`DxgiCaptureSource`](crate::elements::DxgiCaptureSource), whose
    /// `1 / fps` is honest because it decides itself when a frame happens. A
    /// camera does not: exposure lengthens an interval, and a mode nominally
    /// at 30 fps is not delivered on a 1/30 grid. Counting the device's own
    /// clock keeps every one of those intervals as it actually was, rather
    /// than rounding it onto a grid the camera never promised.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, TIME_BASE_DENOMINATOR)
    }

    fn classify_error(&self, error: windows::core::Error) -> MfCaptureSourceError {
        use windows::Win32::Media::MediaFoundation::{
            MF_E_HW_MFT_FAILED_START_STREAMING, MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED,
            MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED,
        };
        if matches!(
            error.code(),
            MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED
                | MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED
                | MF_E_HW_MFT_FAILED_START_STREAMING
        ) {
            MfCaptureSourceError::DeviceLost
        } else {
            MfCaptureSourceError::Windows(error)
        }
    }

    /// Reads one sample, or `None` where the reader answered without one —
    /// a stream tick, which is Media Foundation reporting a gap rather than
    /// a picture and is not an error.
    fn read_sample(&mut self) -> std::result::Result<Option<IMFSample>, MfCaptureSourceError> {
        let mut flags = 0u32;
        let mut sample = None;
        // SAFETY: every output is a live local; the reader is live and this
        // thread is the only one reading it.
        unsafe {
            self.reader.ReadSample(
                video_stream(),
                0,
                None,
                Some(&mut flags),
                None,
                Some(&mut sample),
            )
        }
        .map_err(|error| self.classify_error(error))?;

        if flags & MF_SOURCE_READERF_ERROR.0 as u32 != 0
            || flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0
        {
            // A live camera has no end. Being told the stream ended means
            // the device went away under the reader.
            return Err(MfCaptureSourceError::DeviceLost);
        }
        if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
            return Err(MfCaptureSourceError::FormatChanged);
        }
        Ok(sample)
    }

    /// This sample's `pts` in [`Self::time_base`] units, and the bookkeeping
    /// that keeps the timeline continuous across a Pause.
    fn stamp(&mut self, sample_time: i64) -> i64 {
        match (
            self.origin,
            self.rebase.then_some(self.last_sample_time).flatten(),
        ) {
            (Some(origin), Some(last)) => {
                // Collapse the frozen interval to one nominal frame, so
                // Resume continues the timeline instead of leaving a hole
                // in it the length of the pause.
                self.origin = Some(origin + (sample_time - last) - self.frame_interval);
            }
            (None, _) => self.origin = Some(sample_time),
            _ => {}
        }
        self.rebase = false;
        self.last_sample_time = Some(sample_time);
        sample_time
            - self
                .origin
                .expect("the origin was just set if it was missing")
    }

    /// Copies one sample's NV12 planes into a pooled frame.
    ///
    /// `ConvertToContiguousBuffer` is what makes the bounds knowable: it
    /// hands back the frame packed at NV12's default stride, which is the
    /// width, so one length check covers both planes. A strided buffer
    /// would need `IMF2DBuffer` and its pitch, and the reader's own video
    /// processor does not produce one.
    fn build_frame(
        &mut self,
        sample: &IMFSample,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MfCaptureSourceError> {
        // SAFETY: `sample` is the live sample just read.
        let buffer = unsafe { sample.ConvertToContiguousBuffer() }?;

        let mut data: *mut u8 = ptr::null_mut();
        let mut length = 0u32;
        // SAFETY: both outputs are live locals, and no earlier lock on this
        // buffer is outstanding — it was created by the call above.
        unsafe { buffer.Lock(&mut data, None, Some(&mut length)) }?;

        let result = self.copy_planes(data, length as usize);

        // SAFETY: balances the successful `Lock` above, on every path out of
        // the copy.
        if let Err(error) = unsafe { buffer.Unlock() } {
            return Err(error.into());
        }
        result
    }

    /// The copy itself, with `data` locked for `length` bytes.
    fn copy_planes(
        &mut self,
        data: *mut u8,
        length: usize,
    ) -> std::result::Result<UnboundObjectPoolRef<ffmpeg::frame::Video>, MfCaptureSourceError> {
        let (width, height) = (self.width as usize, self.height as usize);
        let luma = width * height;
        let needed = luma + luma / 2;
        if data.is_null() || length < needed {
            return Err(MfCaptureSourceError::ShortSample {
                width: self.width,
                height: self.height,
                needed,
                got: if data.is_null() { 0 } else { length },
            });
        }

        let mut frame = self.pool.get();
        // Pooled frames are reused, so this is re-stated per frame rather
        // than once at construction: `release` resets nothing, and a frame
        // that reached the compositor without it would be converted by a
        // guess instead of by what the camera said.
        frame.set_color_space(self.color_space);
        frame.set_color_range(self.color_range);
        for (plane, rows) in [(0usize, height), (1usize, height / 2)] {
            let stride = frame.stride(plane);
            let offset = if plane == 0 { 0 } else { luma };
            let destination = frame.data_mut(plane);
            for row in 0..rows {
                // SAFETY: the source is readable for `needed` bytes, and
                // `offset + row * width + width` is at most `needed` for
                // both planes. The destination row is inside `destination`,
                // whose length is FFmpeg's own `stride * rows`.
                unsafe {
                    ptr::copy_nonoverlapping(
                        data.add(offset + row * width),
                        destination.as_mut_ptr().add(row * stride),
                        width,
                    );
                }
            }
        }
        Ok(frame)
    }

    /// Pushes `frame` downstream, reporting — rather than dying on — a
    /// failing `Sink`, the same "drop this one buffer, keep going" contract
    /// every other capture source in this crate gives its own pushes.
    fn push_frame(&mut self, frame: UnboundObjectPoolRef<ffmpeg::frame::Video>, bus: &Bus) {
        if let Err(error) = self.pad.push(MediaBuffer::Video(Arc::new(frame))) {
            bus.post(
                &self.pp_log,
                BusEvent::Error {
                    element_type: ElementType::MfCaptureSource,
                    name: self.name.clone(),
                    error,
                },
            );
        }
    }

    /// Like [`crate::control::drain_control`], but drives the raw control
    /// receiver directly so Resume can flush the reader before reading
    /// again. Samples the camera queued while this source was frozen are
    /// stale by definition, and delivering them would put a burst of
    /// backdated pictures downstream at the moment of Resume.
    ///
    /// Returns whether the loop should stop.
    fn handle_control(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<bool> {
        while let Some((request, ack)) = control.try_recv() {
            let RequestKind::Control(msg) = request else {
                control::apply_finish(self, bus, &ack);
                return Ok(true);
            };
            if msg != ControlMsg::Pause {
                if control::apply_one(self, bus, &msg, &ack)? {
                    return Ok(true);
                }
                continue;
            }
            control::apply_one(self, bus, &msg, &ack)?;

            loop {
                let Some((paused_msg, paused_ack)) = control.recv() else {
                    return Ok(true);
                };
                let RequestKind::Control(paused_msg) = paused_msg else {
                    control::apply_finish(self, bus, &paused_ack);
                    return Ok(true);
                };
                if paused_msg == ControlMsg::Resume {
                    // Resume every downstream stage first, then discard what
                    // the camera queued while nothing was reading, so the
                    // caller cannot observe a half-resumed source.
                    control::apply_one_unacked(self, bus, &paused_msg)?;
                    // SAFETY: `reader` is live and this thread is the only
                    // one reading it; flushing between reads is exactly
                    // what this call is for.
                    if let Err(error) = unsafe { self.reader.Flush(video_stream()) } {
                        return Err(self.classify_error(error).into());
                    }
                    self.rebase = true;
                    let _ = paused_ack.send(());
                    break;
                }
                if control::apply_one(self, bus, &paused_msg, &paused_ack)? {
                    return Ok(true);
                }
                // A redundant Pause (or another one-shot control) was
                // forwarded and acknowledged; remain frozen until Resume.
            }
        }
        Ok(false)
    }
}

impl Element for MfCaptureSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::MfCaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for MfCaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for MfCaptureSource {
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        let _apartment = ComApartment::new().map_err(MfCaptureSourceError::from)?;

        loop {
            if self.handle_control(control, bus)? {
                pp_info!(self, "stopped");
                return Ok(());
            }

            let Some(sample) = self.read_sample()? else {
                // A stream tick: the camera reported a gap rather than a
                // picture. Nothing to push, and nothing wrong.
                pp_debug!(self, "the camera reported a gap with no sample");
                continue;
            };
            // SAFETY: `sample` is the live sample just read.
            let sample_time =
                unsafe { sample.GetSampleTime() }.map_err(|error| self.classify_error(error))?;
            let mut frame = self.build_frame(&sample)?;
            frame.set_pts(Some(self.stamp(sample_time)));
            self.push_frame(frame, bus);
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(MfCaptureSourceError::SeekUnsupported.into())
    }
}

/// What a negotiated media type says its YUV samples mean.
///
/// Both attributes are optional, and a webcam routinely states neither. The
/// fallback is Media Foundation's own documented default rather than a
/// guess of this crate's: BT.601 below 720 lines and BT.709 at or above it,
/// studio range either way, which is what its video processor produces when
/// nothing asks it for something else.
fn describe_color(
    media_type: &IMFMediaType,
    height: u32,
) -> (ffmpeg::color::Space, ffmpeg::color::Range) {
    use windows::Win32::Media::MediaFoundation::{
        MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_YUV_MATRIX, MFNominalRange_0_255,
        MFVideoTransferMatrix_BT709,
    };

    // SAFETY: `media_type` is live and both keys are documented UINT32
    // attributes; a type that carries neither answers with an error, which
    // is the ordinary case rather than a failure.
    let (matrix, range) = unsafe {
        (
            media_type.GetUINT32(&MF_MT_YUV_MATRIX),
            media_type.GetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE),
        )
    };

    let space = match matrix {
        Ok(value) if value == MFVideoTransferMatrix_BT709.0 as u32 => ffmpeg::color::Space::BT709,
        Ok(_) => ffmpeg::color::Space::SMPTE170M,
        Err(_) if height >= 720 => ffmpeg::color::Space::BT709,
        Err(_) => ffmpeg::color::Space::SMPTE170M,
    };
    let range = match range {
        Ok(value) if value == MFNominalRange_0_255.0 as u32 => ffmpeg::color::Range::JPEG,
        _ => ffmpeg::color::Range::MPEG,
    };
    (space, range)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame as it arrived, which is what the contract below is checked
    /// against.
    #[derive(Debug, Clone, Copy)]
    struct Delivered {
        width: u32,
        height: u32,
        pixel: ffmpeg::format::Pixel,
        pts: i64,
    }

    /// A camera to test against, or `None` on a machine with none attached —
    /// which is an ordinary environment (a CI runner, a desktop without a
    /// webcam) rather than a failure.
    fn try_camera() -> Option<MfDevice> {
        match MfCaptureSource::list_devices() {
            Ok(devices) => devices.into_iter().next(),
            Err(error) => {
                eprintln!("skipping: video capture devices could not be enumerated: {error}");
                None
            }
        }
    }

    /// Media Foundation states no matrix or range for most webcam modes, and
    /// the fallback has to be its own documented default rather than a guess
    /// — the D3D11 compositor converts NV12 by exactly these two fields.
    #[test]
    fn an_unstated_matrix_falls_back_to_the_one_media_foundation_would_use() {
        let _apartment = ComApartment::new().expect("COM must initialize");
        let _runtime = MfRuntime::new().expect("Media Foundation must start");
        // SAFETY: creating an empty media type takes no pointers, and this
        // one deliberately carries neither colour attribute.
        let bare = unsafe { MFCreateMediaType() }.expect("an empty media type must be creatable");

        assert_eq!(
            describe_color(&bare, 480),
            (ffmpeg::color::Space::SMPTE170M, ffmpeg::color::Range::MPEG),
            "a standard-definition mode that states nothing is BT.601 studio range"
        );
        assert_eq!(
            describe_color(&bare, 720),
            (ffmpeg::color::Space::BT709, ffmpeg::color::Range::MPEG),
            "720 lines and above is BT.709 studio range"
        );
    }

    /// What the matrix and range attributes say wins over the size-based
    /// fallback, since a camera that states them means them.
    #[test]
    fn a_stated_matrix_and_range_are_used_instead_of_the_fallback() {
        use windows::Win32::Media::MediaFoundation::{
            MF_MT_VIDEO_NOMINAL_RANGE, MF_MT_YUV_MATRIX, MFNominalRange_0_255,
            MFVideoTransferMatrix_BT709,
        };

        let _apartment = ComApartment::new().expect("COM must initialize");
        let _runtime = MfRuntime::new().expect("Media Foundation must start");
        // SAFETY: the type is created here and both keys are documented
        // UINT32 attributes for a video media type.
        let stated = unsafe {
            let stated = MFCreateMediaType().expect("an empty media type must be creatable");
            stated
                .SetUINT32(&MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT709.0 as u32)
                .expect("setting a UINT32 attribute must succeed");
            stated
                .SetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_0_255.0 as u32)
                .expect("setting a UINT32 attribute must succeed");
            stated
        };

        assert_eq!(
            describe_color(&stated, 480),
            (ffmpeg::color::Space::BT709, ffmpeg::color::Range::JPEG),
            "a 480-line mode that says BT.709 full range means it"
        );
    }

    /// A symbolic link is the only identity that outlives a session, so a
    /// device without one would be unopenable and must not be offered.
    #[test]
    fn every_listed_device_states_a_symbolic_link_and_a_name() {
        let Ok(devices) = MfCaptureSource::list_devices() else {
            eprintln!("skipping: video capture devices could not be enumerated");
            return;
        };
        for device in devices {
            assert!(
                !device.id.is_empty(),
                "a listed device must carry the symbolic link an open resolves"
            );
            assert!(
                !device.name.is_empty(),
                "a listed device must carry something to show in a picker"
            );
        }
    }

    /// A camera with no offered mode could not be opened at all, and the list
    /// must not repeat one shape just because the device delivers it in
    /// several subtypes — see `MfCaptureFormat`.
    #[test]
    fn a_camera_offers_at_least_one_mode_and_states_none_of_them_twice() {
        let Some(device) = try_camera() else {
            eprintln!("skipping: no video capture device attached");
            return;
        };
        let formats = MfCaptureSource::list_formats(&device)
            .expect("an attached camera must state its modes");
        assert!(
            !formats.is_empty(),
            "{:?} offered no mode at all",
            device.name
        );
        for (index, format) in formats.iter().enumerate() {
            assert!(
                !formats[..index].contains(format),
                "{format:?} was stated twice"
            );
            assert!(
                format.width > 0 && format.height > 0,
                "{format:?} has no picture in it"
            );
        }
    }

    /// A stored symbolic link outlives the camera it named, so opening a
    /// stale one has to say which device is missing rather than pass on the
    /// bare `E_INVALIDARG` Media Foundation answers with.
    #[test]
    fn a_device_that_is_not_attached_is_named_in_the_error() {
        let options = MfCaptureOptions {
            device: MfDevice {
                id: r"\\?\usb#vid_0000&pid_0000#nothing-is-attached-here".to_owned(),
                name: "a camera that was unplugged".to_owned(),
            },
            format: None,
        };
        match MfCaptureSource::open("missing-camera", options) {
            Err(MfCaptureSourceError::DeviceNotFound(id)) => {
                assert!(
                    id.contains("nothing-is-attached-here"),
                    "the error must name the link that could not be resolved, got {id:?}"
                );
            }
            Err(other) => panic!("expected DeviceNotFound, got {other}"),
            Ok(_) => panic!("a symbolic link naming nothing must not open"),
        }
    }

    /// Asking for a shape the camera does not have must be refused rather
    /// than quietly answered with a neighbouring one, since a caller sizes
    /// its downstream on what it asked for.
    #[test]
    fn a_mode_the_camera_does_not_offer_is_refused() {
        let Some(device) = try_camera() else {
            eprintln!("skipping: no video capture device attached");
            return;
        };
        let options = MfCaptureOptions {
            device,
            format: Some(MfCaptureFormat {
                width: 12,
                height: 8,
                framerate: ffmpeg::Rational::new(1, 1),
            }),
        };
        match MfCaptureSource::open("impossible-mode", options) {
            Err(MfCaptureSourceError::FormatNotOffered { width, height, .. }) => {
                assert_eq!((width, height), (12, 8));
            }
            Err(other) => panic!("expected FormatNotOffered, got {other}"),
            Ok(_) => panic!("no camera offers a 12x8 mode"),
        }
    }

    /// The whole contract downstream depends on: NV12 frames at exactly the
    /// size `open` reported, with `pts` starting at zero and rising.
    #[test]
    fn capture_delivers_rising_nv12_frames_at_the_negotiated_size() {
        use crate::{elements::AppSink, pipeline::Pipeline};
        use std::sync::{Arc, Mutex};

        let Some(device) = try_camera() else {
            eprintln!("skipping: no video capture device attached");
            return;
        };
        let name = device.name.clone();
        let options = MfCaptureOptions {
            device,
            format: None,
        };
        let (source, format) = match MfCaptureSource::open("camera", options) {
            Ok(opened) => opened,
            Err(error) => {
                // A camera another application already holds is a real
                // environment, not a failure of this element.
                eprintln!("skipping: {name:?} could not be opened: {error}");
                return;
            }
        };

        let seen: Arc<Mutex<Vec<Delivered>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let sink = AppSink::new("camera-sink", move |buf| {
            if let MediaBuffer::Video(frame) = buf {
                recorded
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(Delivered {
                        width: frame.width(),
                        height: frame.height(),
                        pixel: frame.format(),
                        pts: frame.pts().unwrap_or(-1),
                    });
            }
            Ok(())
        });

        let pipeline = Pipeline::new("camera-capture", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring a camera to an AppSink should succeed");
        pipeline.run().expect("the pipeline must start");
        std::thread::sleep(Duration::from_millis(1500));
        pipeline.stop();

        let frames = seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let Some(&first) = frames.first() else {
            eprintln!("skipping: {name:?} delivered no frame within the window");
            return;
        };
        assert_eq!(
            (first.width, first.height),
            (format.width, format.height),
            "every frame must be the size open reported"
        );
        assert_eq!(first.pixel, ffmpeg::format::Pixel::NV12);
        assert_eq!(first.pts, 0, "the first frame of a run leaves at zero");
        assert!(
            frames.windows(2).all(|pair| pair[1].pts > pair[0].pts),
            "pts must rise frame to frame, got {:?}",
            frames.iter().map(|frame| frame.pts).collect::<Vec<_>>()
        );
    }

    /// Pause must not leave a hole the length of the pause in the timeline: a
    /// downstream muxer would take that as the camera having genuinely
    /// stopped for that long.
    #[test]
    fn a_pause_costs_the_timeline_one_frame_rather_than_its_own_length() {
        use crate::{elements::AppSink, pipeline::Pipeline};
        use std::sync::{Arc, Mutex};

        let Some(device) = try_camera() else {
            eprintln!("skipping: no video capture device attached");
            return;
        };
        let name = device.name.clone();
        let options = MfCaptureOptions {
            device,
            format: None,
        };
        let Ok((source, _format)) = MfCaptureSource::open("camera-pause", options) else {
            eprintln!("skipping: {name:?} could not be opened");
            return;
        };

        let seen: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let sink = AppSink::new("camera-pause-sink", move |buf| {
            if let MediaBuffer::Video(frame) = buf {
                recorded
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(frame.pts().unwrap_or(-1));
            }
            Ok(())
        });

        let pipeline = Pipeline::new("camera-pause", source, |source, ctx| {
            let branch = ctx.branch().to(Box::new(sink))?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("wiring a camera to an AppSink should succeed");
        pipeline.run().expect("the pipeline must start");

        std::thread::sleep(Duration::from_millis(700));
        let before = seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        pipeline.pause();
        let paused = Duration::from_millis(1000);
        std::thread::sleep(paused);
        pipeline.resume();
        std::thread::sleep(Duration::from_millis(700));
        pipeline.stop();

        let stamps = seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if before == 0 || stamps.len() <= before {
            eprintln!("skipping: {name:?} delivered too few frames around the pause");
            return;
        }
        let across = stamps[before] - stamps[before - 1];
        let paused_ticks =
            paused.as_nanos() as i64 * i64::from(TIME_BASE_DENOMINATOR) / 1_000_000_000;
        assert!(
            across < paused_ticks / 2,
            "the timeline advanced {across} ticks across a {paused_ticks}-tick pause, \
             which is the hole this rebase exists to close"
        );
    }
}
