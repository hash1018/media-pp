//! The camera's one video stream.
//!
//! Frame Server asks for each picture with `RequestSample` and takes it as a
//! `MEMediaSample` event. Requests are answered on a thread of the stream's
//! own, one each `1 / FRAME_RATE`: answering at once, as fast as they come,
//! would let a reader spin the camera as fast as it can ask, where a camera
//! has a rate. Each picture is the producer's last one from the shared
//! section, or a plain dark one while there is none.

use std::sync::{
    Arc, Mutex,
    mpsc::{self, Receiver, Sender},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::{
    Foundation::{E_POINTER, ERROR_SET_NOT_FOUND, S_OK},
    Media::{
        KernelStreaming::{IKsControl, IKsControl_Impl, KSIDENTIFIER, PINNAME_VIDEO_CAPTURE},
        MediaFoundation::*,
    },
    System::{Com::StructuredStorage::PROPVARIANT, SystemInformation::GetTickCount64},
};
use windows_core::{GUID, HRESULT, IUnknown, Interface, Ref, Result, implement};

use crate::{
    Agile,
    attributes::{delegate_attributes, store},
    protocol,
    section::Section,
};

/// The luma of the picture shown while nothing is being produced: dark, but
/// not black, so that a live camera with no producer can be told from a
/// dead one.
const PLACEHOLDER_LUMA: u8 = 32;

/// The stream, as Media Foundation sees it.
#[implement(IMFMediaStream2, IKsControl, IMFAttributes)]
pub(crate) struct Stream {
    attributes: IMFAttributes,
    descriptor: IMFStreamDescriptor,
    queue: IMFMediaEventQueue,
    section_name: Arc<str>,
    state: Mutex<State>,
}

struct State {
    /// The source this stream belongs to, until it shuts down — the one
    /// reference that makes a cycle, broken there.
    source: Option<IMFMediaSource>,
    allocator: Option<IMFVideoSampleAllocatorEx>,
    stream_state: MF_STREAM_STATE,
    delivery: Option<Delivery>,
    shut_down: bool,
}

/// The thread answering requests, and the way to hand it one.
struct Delivery {
    requests: Sender<Option<Agile<IUnknown>>>,
    thread: JoinHandle<()>,
}

impl Stream {
    /// A stream offering every size of [`protocol::SIZES`] in NV12, at
    /// [`protocol::FRAME_RATE`], the largest current.
    pub(crate) fn new(index: u32, section_name: Arc<str>) -> Result<Self> {
        let types = protocol::SIZES
            .iter()
            .map(|&(width, height)| media_type(width, height).map(Some))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: a fresh descriptor over the types just made.
        let descriptor = unsafe { MFCreateStreamDescriptor(index, &types)? };
        // SAFETY: the descriptor's own handler, and a type it offers.
        unsafe {
            let handler = descriptor.GetMediaTypeHandler()?;
            handler.SetCurrentMediaType(types[0].as_ref())?;
        }
        let attributes = store()?;
        for target in [&attributes, &descriptor.cast::<IMFAttributes>()?] {
            // SAFETY: plain values set on a store this stream owns.
            unsafe {
                target.SetGUID(&MF_DEVICESTREAM_STREAM_CATEGORY, &PINNAME_VIDEO_CAPTURE)?;
                target.SetUINT32(&MF_DEVICESTREAM_STREAM_ID, index)?;
                target.SetUINT32(&MF_DEVICESTREAM_FRAMESERVER_SHARED, 1)?;
                target.SetUINT32(
                    &MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES,
                    MFFrameSourceTypes_Color.0 as u32,
                )?;
            }
        }
        // SAFETY: creates a new, empty event queue.
        let queue = unsafe { MFCreateEventQueue()? };
        Ok(Self {
            attributes,
            descriptor,
            queue,
            section_name,
            state: Mutex::new(State {
                source: None,
                allocator: None,
                stream_state: MF_STREAM_STATE_STOPPED,
                delivery: None,
                shut_down: false,
            }),
        })
    }

    pub(crate) fn descriptor(&self) -> &IMFStreamDescriptor {
        &self.descriptor
    }

    pub(crate) fn set_source(&self, source: IMFMediaSource) {
        self.lock().source = Some(source);
    }

    pub(crate) fn set_allocator(&self, allocator: Option<IMFVideoSampleAllocatorEx>) {
        self.lock().allocator = allocator;
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Starts handing out pictures in `media_type`, the one the reader
    /// chose — or the current one where it chose none.
    pub(crate) fn start(&self, media_type: Option<IMFMediaType>) -> Result<()> {
        let media_type = match media_type {
            Some(media_type) => media_type,
            // SAFETY: the descriptor's own handler.
            None => unsafe {
                self.descriptor
                    .GetMediaTypeHandler()?
                    .GetCurrentMediaType()?
            },
        };
        let (width, height) = frame_size(&media_type)?;
        let mut state = self.lock();
        if state.shut_down {
            return Err(MF_E_SHUTDOWN.into());
        }
        if state.delivery.is_some() {
            stop_delivery(&mut state);
        }
        let section = Arc::new(Section::create(&self.section_name)?);
        let header = section.header();
        header
            .width
            .store(width, std::sync::atomic::Ordering::Release);
        header
            .height
            .store(height, std::sync::atomic::Ordering::Release);
        if let Some(allocator) = &state.allocator {
            // SAFETY: an allocator Frame Server handed over, set up for the
            // type the stream now runs in.
            unsafe { allocator.InitializeSampleAllocatorEx(4, 10, None, &media_type)? };
        }
        let (requests, waiting) = mpsc::channel();
        let deliverer = Deliverer {
            queue: Agile(self.queue.clone()),
            allocator: state.allocator.clone().map(Agile),
            section,
            width,
            height,
            frame: vec![0; protocol::nv12_bytes(width, height)],
        };
        let thread = std::thread::Builder::new()
            .name("media-pp-vcam".into())
            .spawn(move || deliverer.run(waiting))
            .map_err(|_| windows_core::Error::from(E_POINTER))?;
        state.delivery = Some(Delivery { requests, thread });
        state.stream_state = MF_STREAM_STATE_RUNNING;
        drop(state);
        // SAFETY: an event this stream's own queue carries.
        unsafe {
            self.queue.QueueEventParamVar(
                MEStreamStarted.0 as u32,
                &GUID::zeroed(),
                S_OK,
                std::ptr::null(),
            )
        }
    }

    pub(crate) fn stop(&self) -> Result<()> {
        let mut state = self.lock();
        if state.shut_down {
            return Err(MF_E_SHUTDOWN.into());
        }
        stop_delivery(&mut state);
        if let Some(allocator) = &state.allocator {
            // SAFETY: undoes `start`'s initialization; harmless where there
            // was none.
            let _ = unsafe { allocator.UninitializeSampleAllocator() };
        }
        state.stream_state = MF_STREAM_STATE_STOPPED;
        drop(state);
        // SAFETY: as in `start`.
        unsafe {
            self.queue.QueueEventParamVar(
                MEStreamStopped.0 as u32,
                &GUID::zeroed(),
                S_OK,
                std::ptr::null(),
            )
        }
    }

    pub(crate) fn shutdown(&self) {
        let mut state = self.lock();
        stop_delivery(&mut state);
        state.source = None;
        state.allocator = None;
        state.shut_down = true;
        // SAFETY: shuts the queue down once; later calls fail with
        // `MF_E_SHUTDOWN`, as they should.
        let _ = unsafe { self.queue.Shutdown() };
    }
}

/// Ends the delivery thread, if there is one, and waits for it. The size it
/// told the producer goes back to none, so the producer stops writing.
fn stop_delivery(state: &mut State) {
    if let Some(delivery) = state.delivery.take() {
        drop(delivery.requests);
        let _ = delivery.thread.join();
    }
}

/// What the delivery thread owns.
struct Deliverer {
    queue: Agile<IMFMediaEventQueue>,
    allocator: Option<Agile<IMFVideoSampleAllocatorEx>>,
    section: Arc<Section>,
    width: u32,
    height: u32,
    frame: Vec<u8>,
}

impl Deliverer {
    fn run(mut self, requests: Receiver<Option<Agile<IUnknown>>>) {
        let interval = Duration::from_nanos(1_000_000_000 / u64::from(protocol::FRAME_RATE));
        let mut due = Instant::now();
        while let Ok(token) = requests.recv() {
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
            due = due.max(now) + interval;
            // A request that cannot be answered is dropped, not fatal: the
            // reader asks again, and the next one may well work.
            let _ = self.deliver(token);
        }
        let header = self.section.header();
        header.width.store(0, std::sync::atomic::Ordering::Release);
        header.height.store(0, std::sync::atomic::Ordering::Release);
    }

    fn deliver(&mut self, token: Option<Agile<IUnknown>>) -> Result<()> {
        // SAFETY: plain call; both processes read the same tick count.
        let now_ms = unsafe { GetTickCount64() } as i64;
        let stale_before = now_ms - protocol::STALE_AFTER_MS as i64;
        // SAFETY: the section is mapped for as long as `self` holds it.
        let fresh = unsafe {
            protocol::read_frame(
                self.section.view(),
                self.width,
                self.height,
                stale_before,
                &mut self.frame,
            )
        };
        if !fresh {
            placeholder(&mut self.frame, self.width, self.height);
        }
        let sample = self.sample()?;
        // SAFETY: stamps the sample this thread just made.
        unsafe {
            sample.SetSampleTime(MFGetSystemTime())?;
            sample.SetSampleDuration(10_000_000 / i64::from(protocol::FRAME_RATE))?;
            if let Some(token) = &token {
                sample.SetUnknown(&MFSampleExtension_Token, &token.0)?;
            }
            self.queue
                .0
                .QueueEventParamUnk(MEMediaSample.0 as u32, &GUID::zeroed(), S_OK, &sample)
        }
    }

    /// A sample holding `self.frame`: from Frame Server's allocator where it
    /// gave one, so that the buffer is one it can hand across processes,
    /// and in plain memory otherwise.
    fn sample(&self) -> Result<IMFSample> {
        let Some(allocator) = &self.allocator else {
            // SAFETY: a new buffer exactly the frame's size, filled before
            // it is handed on.
            unsafe {
                let buffer = MFCreateMemoryBuffer(self.frame.len() as u32)?;
                let mut data = std::ptr::null_mut();
                buffer.Lock(&mut data, None, None)?;
                std::ptr::copy_nonoverlapping(self.frame.as_ptr(), data, self.frame.len());
                buffer.Unlock()?;
                buffer.SetCurrentLength(self.frame.len() as u32)?;
                let sample = MFCreateSample()?;
                sample.AddBuffer(&buffer)?;
                return Ok(sample);
            }
        };
        // SAFETY: a sample from an allocator initialized for this stream's
        // type; its buffer is locked, written within its pitch and its
        // planes, and unlocked.
        unsafe {
            let sample = allocator.0.AllocateSample()?;
            let buffer = sample.GetBufferByIndex(0)?;
            let buffer2d: IMF2DBuffer2 = buffer.cast()?;
            let (mut line0, mut pitch, mut start, mut length) =
                (std::ptr::null_mut(), 0i32, std::ptr::null_mut(), 0u32);
            buffer2d.Lock2DSize(
                MF2DBuffer_LockFlags_Write,
                &mut line0,
                &mut pitch,
                &mut start,
                &mut length,
            )?;
            copy_nv12(&self.frame, self.width, self.height, line0, pitch);
            buffer2d.Unlock2D()?;
            Ok(sample)
        }
    }
}

/// Copies a packed NV12 frame into a buffer whose rows are `pitch` apart,
/// the chroma plane following the luma plane's `height` rows.
///
/// # Safety
///
/// `line0` points at a locked buffer of at least `pitch * height * 3 / 2`
/// bytes, and `pitch` is at least `width`.
unsafe fn copy_nv12(frame: &[u8], width: u32, height: u32, line0: *mut u8, pitch: i32) {
    let (width, height, pitch) = (width as usize, height as usize, pitch as isize);
    for row in 0..height + height / 2 {
        // SAFETY: as the caller guarantees; each row is `width` bytes of
        // the packed frame into its own row of the buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.as_ptr().add(row * width),
                line0.offset(row as isize * pitch),
                width,
            );
        }
    }
}

/// A flat dark picture, neutral in colour.
fn placeholder(frame: &mut [u8], width: u32, height: u32) {
    let luma = width as usize * height as usize;
    frame[..luma].fill(PLACEHOLDER_LUMA);
    frame[luma..protocol::nv12_bytes(width, height)].fill(128);
}

/// An NV12 type `width` by `height` at the camera's rate.
pub(crate) fn media_type(width: u32, height: u32) -> Result<IMFMediaType> {
    let pack = |high: u32, low: u32| (u64::from(high) << 32) | u64::from(low);
    // SAFETY: plain values set on a type this function owns.
    unsafe {
        let media_type = MFCreateMediaType()?;
        media_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        media_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
        media_type.SetUINT64(&MF_MT_FRAME_SIZE, pack(width, height))?;
        media_type.SetUINT64(&MF_MT_FRAME_RATE, pack(protocol::FRAME_RATE, 1))?;
        media_type.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, pack(1, 1))?;
        media_type.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
        media_type.SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1)?;
        // What the producing element converts every picture to.
        media_type.SetUINT32(&MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT709.0 as u32)?;
        media_type.SetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_16_235.0 as u32)?;
        media_type.SetUINT32(&MF_MT_VIDEO_PRIMARIES, MFVideoPrimaries_BT709.0 as u32)?;
        media_type.SetUINT32(&MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_709.0 as u32)?;
        media_type.SetUINT32(&MF_MT_FIXED_SIZE_SAMPLES, 1)?;
        media_type.SetUINT32(&MF_MT_DEFAULT_STRIDE, width)?;
        let bytes = protocol::nv12_bytes(width, height) as u32;
        media_type.SetUINT32(&MF_MT_SAMPLE_SIZE, bytes)?;
        media_type.SetUINT32(&MF_MT_AVG_BITRATE, bytes * 8 * protocol::FRAME_RATE)?;
        Ok(media_type)
    }
}

/// The size a type says its frames are.
pub(crate) fn frame_size(media_type: &IMFMediaType) -> Result<(u32, u32)> {
    // SAFETY: reads one value from the type.
    let packed = unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE)? };
    Ok(((packed >> 32) as u32, packed as u32))
}

impl IMFMediaEventGenerator_Impl for Stream_Impl {
    fn GetEvent(&self, flags: MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS) -> Result<IMFMediaEvent> {
        // SAFETY: forwarded to the stream's own queue.
        unsafe { self.queue.GetEvent(flags.0) }
    }

    fn BeginGetEvent(&self, callback: Ref<IMFAsyncCallback>, state: Ref<IUnknown>) -> Result<()> {
        // SAFETY: as above.
        unsafe { self.queue.BeginGetEvent(callback.as_ref(), state.as_ref()) }
    }

    fn EndGetEvent(&self, result: Ref<IMFAsyncResult>) -> Result<IMFMediaEvent> {
        // SAFETY: as above.
        unsafe { self.queue.EndGetEvent(result.as_ref()) }
    }

    fn QueueEvent(
        &self,
        event: u32,
        extended: *const GUID,
        status: HRESULT,
        value: *const PROPVARIANT,
    ) -> Result<()> {
        // SAFETY: as above.
        unsafe {
            self.queue
                .QueueEventParamVar(event, extended, status, value)
        }
    }
}

impl IMFMediaStream_Impl for Stream_Impl {
    fn GetMediaSource(&self) -> Result<IMFMediaSource> {
        let state = self.lock();
        if state.shut_down {
            return Err(MF_E_SHUTDOWN.into());
        }
        state.source.clone().ok_or_else(|| MF_E_SHUTDOWN.into())
    }

    fn GetStreamDescriptor(&self) -> Result<IMFStreamDescriptor> {
        if self.lock().shut_down {
            return Err(MF_E_SHUTDOWN.into());
        }
        Ok(self.descriptor.clone())
    }

    fn RequestSample(&self, token: Ref<IUnknown>) -> Result<()> {
        let state = self.lock();
        if state.shut_down {
            return Err(MF_E_SHUTDOWN.into());
        }
        let Some(delivery) = &state.delivery else {
            return Err(MF_E_INVALIDREQUEST.into());
        };
        delivery
            .requests
            .send(token.cloned().map(Agile))
            .map_err(|_| MF_E_SHUTDOWN.into())
    }
}

impl IMFMediaStream2_Impl for Stream_Impl {
    fn SetStreamState(&self, value: MF_STREAM_STATE) -> Result<()> {
        let current = self.lock().stream_state;
        if current == value {
            return Ok(());
        }
        match value {
            MF_STREAM_STATE_RUNNING => self.start(None),
            MF_STREAM_STATE_STOPPED => self.stop(),
            _ => Err(MF_E_INVALID_STATE_TRANSITION.into()),
        }
    }

    fn GetStreamState(&self) -> Result<MF_STREAM_STATE> {
        Ok(self.lock().stream_state)
    }
}

impl IKsControl_Impl for Stream_Impl {
    fn KsProperty(
        &self,
        _: *const KSIDENTIFIER,
        _: u32,
        _: *mut core::ffi::c_void,
        _: u32,
        _: *mut u32,
    ) -> Result<()> {
        Err(ERROR_SET_NOT_FOUND.to_hresult().into())
    }

    fn KsMethod(
        &self,
        _: *const KSIDENTIFIER,
        _: u32,
        _: *mut core::ffi::c_void,
        _: u32,
        _: *mut u32,
    ) -> Result<()> {
        Err(ERROR_SET_NOT_FOUND.to_hresult().into())
    }

    fn KsEvent(
        &self,
        _: *const KSIDENTIFIER,
        _: u32,
        _: *mut core::ffi::c_void,
        _: u32,
        _: *mut u32,
    ) -> Result<()> {
        Err(ERROR_SET_NOT_FOUND.to_hresult().into())
    }
}

delegate_attributes!(Stream_Impl);
