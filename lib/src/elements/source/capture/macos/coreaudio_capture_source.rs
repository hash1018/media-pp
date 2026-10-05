use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicI32, AtomicPtr, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use crossbeam_queue::ArrayQueue;
use ffmpeg_next as ffmpeg;
use objc2_audio_toolbox::{
    AURenderCallbackStruct, AudioUnitRender, AudioUnitRenderActionFlags,
    kAudioOutputUnitProperty_CurrentDevice, kAudioOutputUnitProperty_EnableIO,
    kAudioOutputUnitProperty_SetInputCallback, kAudioUnitProperty_MaximumFramesPerSlice,
    kAudioUnitProperty_StreamFormat, kAudioUnitScope_Global, kAudioUnitScope_Input,
    kAudioUnitScope_Output,
};
use objc2_core_audio_types::{
    AudioBuffer, AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp,
    kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked, kAudioFormatLinearPCM,
};
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};
use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    element::{Element, ElementType, Produced, Source, SourceStage, Wait, element_pp_log},
    elements::{AudioFormat, CoreAudioDevice, CoreAudioDeviceKind, CoreAudioProcess},
    error::Result,
    platform::macos::coreaudio::{self, HalUnit, OsStatusError, ProcessTap, Refcon, TapError},
    produce::source_stage,
};

/// How many packets the IO thread can hand over before the source's thread
/// takes any: 64 of a device's usual 512 frames, about 0.7 s at 48 kHz.
/// Past that a stalled downstream costs packets, not the device's thread.
const PACKETS: usize = 64;
/// How long the source waits, once it has handed on every packet there was,
/// before it looks again — about one IO buffer.
const POLL_INTERVAL: Duration = Duration::from_millis(10);
/// How long a running device may hand over nothing before it is asked
/// whether it is still there. An unplugged device stops calling back
/// without a word, so this is how its end is noticed.
const SILENT_DEVICE_CHECK: Duration = Duration::from_secs(1);
/// AUHAL's input side is element 1, its output side element 0.
const INPUT_ELEMENT: u32 = 1;
const OUTPUT_ELEMENT: u32 = 0;
/// The largest callback AUHAL makes when it does not say: its default
/// `kAudioUnitProperty_MaximumFramesPerSlice`.
const DEFAULT_MAX_FRAMES: u32 = 4096;
/// How long a tap may hand over nothing before its silence is made up. A tap
/// is clocked by an output device, which runs only while something plays;
/// shorter than this, a gap is a device's ordinary jitter, not its silence.
const IDLE: Duration = Duration::from_millis(100);

/// Which device a [`CoreAudioCaptureSource`] records from — an input, or an
/// output whose playback it captures.
#[derive(Debug, Clone)]
pub struct CoreAudioCaptureOptions {
    /// One entry out of [`CoreAudioCaptureSource::list_devices`].
    pub device: CoreAudioDevice,
}

/// Why a [`CoreAudioCaptureSource`] could not open its device or goes on no
/// longer.
#[derive(Debug, ThisError)]
pub enum CoreAudioCaptureSourceError {
    /// A Core Audio call failed — a device unplugged since it was listed
    /// fails this way, `kAudioHardwareBadObjectError`.
    #[error("Core Audio could not {operation}: OSStatus {status}{}", coreaudio::four_char_code(.status))]
    CoreAudio {
        /// What was being done.
        operation: &'static str,
        /// The `OSStatus` it returned.
        status: i32,
    },
    /// The device has nothing to record from.
    #[error("{0:?} has no input channels")]
    NoInputChannels(String),
    /// The device runs at a rate, or with a number of channels, an
    /// [`AudioFormat`] cannot describe.
    #[error("{name:?} records {channels} channel(s) at {rate}Hz, which cannot be captured")]
    UnsupportedFormat {
        /// The device's name.
        name: String,
        /// Its nominal sample rate.
        rate: f64,
        /// Its input channels.
        channels: u32,
    },
    /// The system has no AUHAL unit to record through.
    #[error("the system has no AUHAL audio unit")]
    NoHalUnit,
    /// The system has no process taps, which capturing an output device or a
    /// process needs: it is older than macOS 14.2.
    #[error("capturing what is played needs macOS 14.2 or newer, for Core Audio process taps")]
    ProcessTapsUnsupported,
    /// The process asked for is not a Core Audio client — gone, or never
    /// having played or recorded anything.
    #[error("process {0} is not playing or recording through Core Audio")]
    NoSuchProcess(u32),
    /// The device went away while it was recording — unplugged, for one.
    /// This source does not reopen it: the caller, which knows what was
    /// picked, lists the devices and opens again.
    #[error("{0:?} is no longer there to record from")]
    DeviceGone(String),
}

impl From<OsStatusError> for CoreAudioCaptureSourceError {
    fn from(error: OsStatusError) -> Self {
        Self::CoreAudio {
            operation: error.operation,
            status: error.status,
        }
    }
}

impl From<TapError> for CoreAudioCaptureSourceError {
    fn from(error: TapError) -> Self {
        match error {
            TapError::Unsupported => Self::ProcessTapsUnsupported,
            TapError::NoSuchProcess(pid) => Self::NoSuchProcess(pid),
            TapError::Status(error) => error.into(),
        }
    }
}

/// Captures audio through Core Audio: what an input device records — a
/// microphone, a line input — what the system plays to an output device, or
/// what one application plays. One src pad, pushing `MediaBuffer::Audio`
/// frames as 32-bit float interleaved, in the format
/// [`CoreAudioCaptureSource::open`] or [`CoreAudioCaptureSource::open_process`]
/// reports. If something downstream needs another format, an
/// [`crate::elements::AudioResampler`] converts — this element does not, as
/// `WasapiCaptureSource` and `PipeWireAudioCaptureSource`, its Windows and
/// Linux counterparts, do not; they are also where the device-or-process
/// shape of this API comes from.
///
/// What is played is captured through a Core Audio process tap, macOS 14.2
/// and newer: a tap of the processes asked for, and a private aggregate
/// device that records it, both this source's own and gone with it.
///
/// # Timeline
///
/// `pts` counts samples, in [`Self::time_base`]. A running input device
/// calls back at its own rate whatever it hears, so its timeline is
/// continuous by counting what it captured. A tap is clocked by an output
/// device, which runs only while something plays, and hands over nothing at
/// all while nothing does — as WASAPI's loopback does. So where a tap has
/// handed over nothing for a moment, the silence it stands for is made up,
/// keeping `pts` with the clock; while it hands over sound, nothing is ever
/// inserted into it. A pause stops the device, and what it would have
/// captured meanwhile is not counted — playing on takes up where it left
/// off.
///
/// # Threading
///
/// Core Audio hands over what the device captured on its real-time IO
/// thread, which must never wait. So each capture goes into one of a fixed
/// set of packets, made once in `open`, that the source's thread takes and
/// gives back. When a stalled downstream has left none free, what the device
/// captured is dropped, counted, and reported as
/// [`crate::bus::BusEvent::Dropped`]; every packet carries where the device
/// captured it, so the gap stays where it happened.
///
/// # Permission
///
/// macOS asks the user whether a program may use the microphone, the first
/// time one records — the program, for a command-line one, being the
/// terminal or application that started it. While that question is open,
/// [`CoreAudioCaptureSource::open`] waits for its answer inside Core Audio,
/// however long that takes. Denied, the device records silence rather than
/// failing, so a capture that comes back silent is the first thing to check
/// against System Settings, Privacy & Security, Microphone.
///
/// Capturing what is played needs the "System Audio Recording" permission
/// in the same place, and asks for it only of an application whose
/// `Info.plist` says why it wants it, under `NSAudioCaptureUsageDescription`
/// — one that does not is not asked, and its taps record silence. A program
/// that captures what is played has to be bundled as an application that
/// says so, and so has whatever terminal runs a command-line one.
///
/// Runs until `Stop` — never reaches `Eos` on its own, as no live source in
/// this crate does. An unplugged device ends it with
/// [`CoreAudioCaptureSourceError::DeviceGone`].
pub struct CoreAudioCaptureSource(SourceStage<Capturing>);

source_stage!(CoreAudioCaptureSource);

/// One capture from the device, in a buffer made once and used again.
struct Packet {
    /// As long as the largest capture the device makes; `frames` of it hold
    /// this one.
    bytes: Vec<u8>,
    frames: usize,
    /// Where in what the device has captured this one begins.
    position: u64,
}

/// What the input callback shares with the source.
struct Shared {
    /// The unit the callback renders the device's input from — set once,
    /// before the callback is.
    unit: AtomicPtr<c_void>,
    bytes_per_frame: usize,
    channels: u32,
    /// Packets free to capture into, and captured ones waiting for the
    /// source. Every packet is in one of the two, or with whichever side is
    /// using it, so neither ever fills.
    free: ArrayQueue<Packet>,
    captured: ArrayQueue<Packet>,
    /// Frames the device has captured, dropped ones included: where the next
    /// capture begins.
    captured_frames: AtomicU64,
    /// Frames dropped since the source last said so.
    dropped_frames: AtomicU64,
    /// The last `AudioUnitRender` failure, 0 where there has been none.
    render_error: AtomicI32,
}

/// The unit's input callback, on Core Audio's real-time IO thread.
///
/// Waits on nothing and allocates nothing: it takes a free packet, renders
/// the device's input into it, and hands it over — or, with none free,
/// counts what it could not keep.
unsafe extern "C-unwind" fn input_callback(
    refcon: NonNull<c_void>,
    flags: NonNull<AudioUnitRenderActionFlags>,
    timestamp: NonNull<AudioTimeStamp>,
    bus: u32,
    frames: u32,
    _data: *mut AudioBufferList,
) -> i32 {
    // SAFETY: `refcon` is the `Shared` `InputUnit::open` registered with the
    // callback, whose `Refcon` keeps a strong reference to it until the unit
    // that calls this is disposed.
    let shared = unsafe { refcon.cast::<Shared>().as_ref() };
    let position = shared
        .captured_frames
        .fetch_add(u64::from(frames), Ordering::AcqRel);
    let bytes = frames as usize * shared.bytes_per_frame;
    let Some(mut packet) = shared.free.pop() else {
        shared
            .dropped_frames
            .fetch_add(u64::from(frames), Ordering::Relaxed);
        return 0;
    };
    if bytes > packet.bytes.len() {
        shared
            .dropped_frames
            .fetch_add(u64::from(frames), Ordering::Relaxed);
        let _ = shared.free.push(packet);
        return 0;
    }
    let mut list = AudioBufferList {
        mNumberBuffers: 1,
        mBuffers: [AudioBuffer {
            mNumberChannels: shared.channels,
            mDataByteSize: bytes as u32,
            mData: packet.bytes.as_mut_ptr().cast::<c_void>(),
        }],
    };
    // SAFETY: the unit is live while it calls this; the flags and timestamp
    // are Core Audio's own for this call, and `list` points at `bytes`
    // writable bytes of the packet, which this thread holds.
    let status = unsafe {
        AudioUnitRender(
            shared.unit.load(Ordering::Acquire).cast(),
            flags.as_ptr(),
            timestamp,
            bus,
            frames,
            NonNull::from(&mut list),
        )
    };
    if status != 0 {
        shared.render_error.store(status, Ordering::Relaxed);
        shared
            .dropped_frames
            .fetch_add(u64::from(frames), Ordering::Relaxed);
        let _ = shared.free.push(packet);
        return 0;
    }
    packet.frames = frames as usize;
    packet.position = position;
    // Never full: there are as many places in it as packets.
    let _ = shared.captured.push(packet);
    0
}

/// An AUHAL unit recording one device, and the callback's reference to what
/// it shares — declared after the unit, so the unit is disposed of first.
struct InputUnit {
    unit: HalUnit,
    _shared: Refcon<Shared>,
}

impl InputUnit {
    /// An AUHAL unit recording `device` in `format` into `shared`'s packets,
    /// initialized and stopped. `shared` is made here, once the unit has said
    /// how large a capture can be.
    fn open(
        device: u32,
        format: AudioFormat,
    ) -> std::result::Result<(Self, Arc<Shared>), CoreAudioCaptureSourceError> {
        let mut unit = HalUnit::new()?.ok_or(CoreAudioCaptureSourceError::NoHalUnit)?;
        // Recording only: the input side on, the output side off. Set before
        // the device, which the unit then opens for input.
        unit.set(
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            INPUT_ELEMENT,
            &1u32,
            "enable the unit's input",
        )?;
        unit.set(
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output,
            OUTPUT_ELEMENT,
            &0u32,
            "disable the unit's output",
        )?;
        unit.set(
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &device,
            "point the input unit at the device",
        )?;
        let bytes_per_frame = 4 * u32::from(format.channels);
        let stream = AudioStreamBasicDescription {
            mSampleRate: f64::from(format.sample_rate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: bytes_per_frame,
            mFramesPerPacket: 1,
            mBytesPerFrame: bytes_per_frame,
            mChannelsPerFrame: u32::from(format.channels),
            mBitsPerChannel: 32,
            mReserved: 0,
        };
        // The output scope of the input element is what the unit hands this
        // side: the device's own rate, so the unit converts nothing.
        unit.set(
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            INPUT_ELEMENT,
            &stream,
            "give the input unit its format",
        )?;
        let max_frames = unit
            .get::<u32>(
                kAudioUnitProperty_MaximumFramesPerSlice,
                kAudioUnitScope_Global,
                0,
                "read the input unit's largest capture",
            )
            .ok()
            .filter(|&frames| frames > 0)
            .unwrap_or(DEFAULT_MAX_FRAMES);
        let packet_bytes = max_frames as usize * bytes_per_frame as usize;
        let free = ArrayQueue::new(PACKETS);
        for _ in 0..PACKETS {
            let _ = free.push(Packet {
                bytes: vec![0; packet_bytes],
                frames: 0,
                position: 0,
            });
        }
        let shared = Arc::new(Shared {
            unit: AtomicPtr::new(unit.raw().cast()),
            bytes_per_frame: bytes_per_frame as usize,
            channels: u32::from(format.channels),
            free,
            captured: ArrayQueue::new(PACKETS),
            captured_frames: AtomicU64::new(0),
            dropped_frames: AtomicU64::new(0),
            render_error: AtomicI32::new(0),
        });
        let refcon = Refcon::new(Arc::clone(&shared));
        let callback = AURenderCallbackStruct {
            inputProc: Some(input_callback),
            inputProcRefCon: refcon.as_ptr(),
        };
        unit.set(
            kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global,
            0,
            &callback,
            "set the input unit's callback",
        )?;
        unit.initialize()?;
        Ok((
            Self {
                unit,
                _shared: refcon,
            },
            shared,
        ))
    }
}

/// What a [`CoreAudioCaptureSource`] hands on, a packet at a time as the IO
/// thread captures it: all of its work, which the framework makes the
/// source.
struct Capturing {
    name: Arc<str>,
    pp_log: PpLog,
    format: AudioFormat,
    /// What the unit records — the device, or a tap's aggregate device —
    /// and the name to say it by.
    device_id: u32,
    device_name: String,
    /// Its pipeline's, for saying captured frames were dropped — `None`
    /// driven by hand.
    bus: Option<Bus>,
    unit: InputUnit,
    /// The tap the unit records, where it records one — declared after the
    /// unit, so that is gone before the tap is.
    tap: Option<ProcessTap>,
    shared: Arc<Shared>,
    running: bool,
    /// When a packet last came, or the device was last found alive; `None`
    /// until the first, and again after a pause.
    heard_from: Option<Instant>,
    /// Where the timeline began, on the clock a pause does not move — what
    /// a tap's silence is made up against. Set as it is first asked.
    started: Option<Instant>,
    /// When a tap last handed anything over, on the same clock.
    last_packet: Option<Instant>,
    /// Frames of silence made up so far, which put a tap's packets that far
    /// on from where the device captured them.
    silence: u64,
    /// Where the next frame goes.
    next_pts: u64,
}

impl CoreAudioCaptureSource {
    /// Every device there is to capture: the inputs, then the outputs whose
    /// playback can be captured — each with the system's default for its
    /// [`kind`](CoreAudioDevice::kind) marked, as
    /// `WasapiCaptureSource::list_devices` lists both kinds of endpoint.
    pub fn list_devices() -> std::result::Result<Vec<CoreAudioDevice>, CoreAudioCaptureSourceError>
    {
        let mut devices = coreaudio::list_input_devices()?;
        devices.extend(coreaudio::list_output_devices()?);
        Ok(devices)
    }

    /// Opens `options.device` — an input to record, or an output whose
    /// playback to capture — and returns the format it captures in: the
    /// device's nominal rate and channels, 32-bit float interleaved, what a
    /// caller needs to build what follows, as `WasapiCaptureSource::open`
    /// returns it.
    ///
    /// Nothing records until the source runs.
    pub fn open(
        name: impl Into<String>,
        options: CoreAudioCaptureOptions,
    ) -> std::result::Result<(Self, AudioFormat), CoreAudioCaptureSourceError> {
        let device = options.device;
        match device.kind {
            CoreAudioDeviceKind::Input => Self::assemble(name.into(), device.id, device.name, None),
            CoreAudioDeviceKind::Output => {
                let tap = ProcessTap::device(&device.uid)?;
                Self::assemble(name.into(), tap.device_id(), device.name, Some(tap))
            }
        }
    }

    /// Every process Core Audio has as a client — the applications a capture
    /// of what one plays can be pointed at, whose `id` goes to
    /// [`CoreAudioCaptureSource::open_process`].
    pub fn list_processes()
    -> std::result::Result<Vec<CoreAudioProcess>, CoreAudioCaptureSourceError> {
        Ok(coreaudio::list_processes()?)
    }

    /// Opens a capture of what one process plays, mixed to stereo at the
    /// default output's rate, wherever it plays it — what lets a caller
    /// record a game without the chat program beside it.
    ///
    /// `process_id` is a live process; the capture covers it and the
    /// processes it had started by now, since a browser plays from helpers
    /// it starts. A helper started later, or one the system starts for it
    /// rather than the application itself — an XPC service — is not in it.
    /// It is bound to that process: when it exits, this goes quiet rather
    /// than following the next instance of the same application.
    pub fn open_process(
        name: impl Into<String>,
        process_id: u32,
    ) -> std::result::Result<(Self, AudioFormat), CoreAudioCaptureSourceError> {
        let tap = ProcessTap::processes(&coreaudio::process_tree(process_id))?;
        Self::assemble(
            name.into(),
            tap.device_id(),
            format!("process {process_id}"),
            Some(tap),
        )
    }

    /// What every open ends with: a unit recording `device_id`, which is
    /// `tap`'s aggregate device where there is one.
    fn assemble(
        name: String,
        device_id: u32,
        device_name: String,
        tap: Option<ProcessTap>,
    ) -> std::result::Result<(Self, AudioFormat), CoreAudioCaptureSourceError> {
        let name: Arc<str> = name.into();
        let pp_log = element_pp_log(ElementType::CoreAudioCaptureSource, &name, None);
        let channels = coreaudio::input_channels(device_id)?;
        if channels == 0 {
            return Err(CoreAudioCaptureSourceError::NoInputChannels(device_name));
        }
        let rate = coreaudio::nominal_sample_rate(device_id)?;
        let (Ok(channels_u16), true) = (
            u16::try_from(channels),
            rate.is_finite() && rate >= 1.0 && rate <= f64::from(i32::MAX),
        ) else {
            return Err(CoreAudioCaptureSourceError::UnsupportedFormat {
                name: device_name,
                rate,
                channels,
            });
        };
        let format = AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            rate.round() as u32,
            channels_u16,
        );
        let (unit, shared) = InputUnit::open(device_id, format)?;
        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?} (id {}{}), {}Hz, {} channel(s), format={:?}",
            device_name,
            device_id,
            if tap.is_some() { ", tapped" } else { "" },
            format.sample_rate,
            format.channels,
            format.sample_format
        );
        Ok((
            Self(SourceStage::new(Capturing {
                name,
                pp_log,
                format,
                device_id,
                device_name,
                bus: None,
                unit,
                tap,
                shared,
                running: false,
                heard_from: None,
                started: None,
                last_packet: None,
                silence: 0,
                next_pts: 0,
            })),
            format,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in: one sample.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.0.inner().format.sample_rate as i32)
    }
}

/// Wraps one packet as an `ffmpeg` frame at `pts` — where the device
/// captured it, and any silence made up before it — so a packet dropped
/// before it leaves its gap.
fn build_frame(format: &AudioFormat, packet: &Packet, pts: u64) -> ffmpeg::frame::Audio {
    let mut frame =
        ffmpeg::frame::Audio::new(format.sample_format, packet.frames, format.channel_layout());
    frame.set_rate(format.sample_rate);
    // Only the tight bytes: the frame's own plane may be padded longer.
    let tight_bytes = (packet.frames * format.channels as usize * format.sample_format.bytes())
        .min(packet.bytes.len())
        .min(frame.data_mut(0).len());
    frame.data_mut(0)[..tight_bytes].copy_from_slice(&packet.bytes[..tight_bytes]);
    frame.set_pts(Some(pts as i64));
    crate::buffer::set_time_base(
        &mut frame,
        ffmpeg::Rational::new(1, format.sample_rate as i32),
    );
    frame
}

/// `frames` of silence at `pts`.
fn silence_frame(format: &AudioFormat, frames: usize, pts: u64) -> ffmpeg::frame::Audio {
    let mut frame =
        ffmpeg::frame::Audio::new(format.sample_format, frames, format.channel_layout());
    frame.set_rate(format.sample_rate);
    frame.data_mut(0).fill(0);
    frame.set_pts(Some(pts as i64));
    crate::buffer::set_time_base(
        &mut frame,
        ffmpeg::Rational::new(1, format.sample_rate as i32),
    );
    frame
}

/// How many frames of silence a tap owes: none while it has handed
/// something over within [`IDLE`], and otherwise what takes `pts` from
/// `next_pts` up to `elapsed`'s worth at `rate` — at most a tenth of a
/// second's at once.
fn silence_owed(elapsed: Duration, since_packet: Duration, next_pts: u64, rate: u32) -> u64 {
    if since_packet < IDLE {
        return 0;
    }
    let due = (elapsed.as_secs_f64() * f64::from(rate)) as u64;
    due.saturating_sub(next_pts).min(u64::from(rate / 10))
}

impl Capturing {
    fn start(&mut self) -> Result<()> {
        if !self.running {
            self.unit
                .unit
                .start()
                .map_err(CoreAudioCaptureSourceError::from)?;
            self.running = true;
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        if self.running {
            self.unit
                .unit
                .stop()
                .map_err(CoreAudioCaptureSourceError::from)?;
            self.running = false;
        }
        Ok(())
    }

    /// Gives every captured packet back unread — what a pause left, which
    /// playing on must not hand over as a burst of the past.
    fn discard_captured(&self) {
        while let Some(packet) = self.shared.captured.pop() {
            let _ = self.shared.free.push(packet);
        }
    }

    /// Says, once, how much the IO thread has had to drop since last time.
    fn report_dropped(&self) {
        let dropped = self.shared.dropped_frames.swap(0, Ordering::Relaxed);
        if dropped == 0 {
            return;
        }
        pp_warn!(
            self,
            "dropped {dropped} captured frame(s): downstream is not keeping up"
        );
        if let Some(bus) = &self.bus {
            bus.post(
                &self.pp_log,
                BusEvent::Dropped {
                    element_type: ElementType::CoreAudioCaptureSource,
                    name: self.name.clone(),
                },
            );
        }
    }

    /// The next captured packet as a frame, or `None` where there is none
    /// yet. An input device that has gone quiet for long is asked whether
    /// it is still there, and one that is not ends the source; a tap going
    /// quiet is only nothing playing.
    fn next_frame(&mut self, now: Instant) -> Result<Option<ffmpeg::frame::Audio>> {
        let status = self.shared.render_error.swap(0, Ordering::Relaxed);
        if status != 0 {
            return Err(CoreAudioCaptureSourceError::CoreAudio {
                operation: "render the device's input",
                status,
            }
            .into());
        }
        self.report_dropped();
        let wall = Instant::now();
        let Some(packet) = self.shared.captured.pop() else {
            let heard_from = *self.heard_from.get_or_insert(wall);
            if self.tap.is_none()
                && self.running
                && wall.duration_since(heard_from) >= SILENT_DEVICE_CHECK
            {
                if !coreaudio::is_alive(self.device_id) {
                    return Err(
                        CoreAudioCaptureSourceError::DeviceGone(self.device_name.clone()).into(),
                    );
                }
                self.heard_from = Some(wall);
            }
            return Ok(None);
        };
        let pts = packet.position + self.silence;
        let frame = build_frame(&self.format, &packet, pts);
        self.next_pts = pts + packet.frames as u64;
        let _ = self.shared.free.push(packet);
        self.heard_from = Some(wall);
        self.last_packet = Some(now);
        Ok(Some(frame))
    }

    /// The silence a quiet tap owes by `now`, as a frame — see
    /// [`silence_owed`].
    fn owed_silence(&mut self, now: Instant) -> Option<ffmpeg::frame::Audio> {
        let started = self.started?;
        let since_packet = now.saturating_duration_since(self.last_packet.unwrap_or(started));
        let frames = silence_owed(
            now.saturating_duration_since(started),
            since_packet,
            self.next_pts,
            self.format.sample_rate,
        );
        if frames == 0 {
            return None;
        }
        let frame = silence_frame(&self.format, frames as usize, self.next_pts);
        self.silence += frames;
        self.next_pts += frames;
        Some(frame)
    }
}

impl Element for Capturing {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CoreAudioCaptureSource
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
        OutputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    /// The next packet the IO thread captured, as a frame stamped with where
    /// the device captured it; with none yet, a look again
    /// [`POLL_INTERVAL`] later — and for a quiet tap, the silence it owes.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        let audio = |frame| Produced::Buffer(MediaBuffer::Audio(Arc::new(frame)));
        self.started.get_or_insert(wait.now());
        if let Some(frame) = self.next_frame(wait.now())? {
            return Ok(audio(frame));
        }
        if !wait.until(wait.now() + POLL_INTERVAL) {
            return Ok(Produced::Nothing);
        }
        if let Some(frame) = self.next_frame(wait.now())? {
            return Ok(audio(frame));
        }
        Ok(match self.tap {
            Some(_) => self
                .owed_silence(wait.now())
                .map_or(Produced::Nothing, audio),
            None => Produced::Nothing,
        })
    }

    /// Stops the device for the pause. Left running, it would fill every
    /// packet with audio nobody reads and then drop the rest.
    fn pausing(&mut self) -> Result<()> {
        self.stop()
    }

    /// Lets go of what the pause left captured, so playing on starts from
    /// live audio, and starts the device again.
    fn resuming(&mut self) -> Result<()> {
        self.discard_captured();
        self.heard_from = None;
        self.start()
    }

    /// Starts the device on the source's own thread.
    fn starting(&mut self) -> Result<()> {
        self.start()
    }

    /// Stops the device, on the thread that started it.
    fn stopping(&mut self) {
        if let Err(error) = self.stop() {
            pp_error!(self, "stopping the device failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::thread;

    use super::*;
    use crate::elements::AppSink;

    fn format() -> AudioFormat {
        AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            48_000,
            2,
        )
    }

    /// A packet becomes a frame of its own samples, stamped with where the
    /// device captured it and in samples.
    #[test]
    fn a_packet_is_a_frame_stamped_where_it_was_captured() {
        let format = format();
        let mut bytes = vec![0u8; 4096 * 8];
        for (index, byte) in bytes.iter_mut().take(3 * 8).enumerate() {
            *byte = index as u8;
        }
        let packet = Packet {
            bytes,
            frames: 3,
            position: 1_920,
        };
        let frame = build_frame(&format, &packet, packet.position);
        assert_eq!(frame.samples(), 3);
        assert_eq!(frame.pts(), Some(1_920));
        assert_eq!(frame.rate(), 48_000);
        assert_eq!(
            &frame.data(0)[..3 * 8],
            &(0..24).map(|byte| byte as u8).collect::<Vec<_>>()[..]
        );
    }

    /// A quiet tap owes nothing within [`IDLE`] of its last packet — a
    /// device's jitter is not silence — and past it, what takes `pts` up to
    /// the clock, a tenth of a second at a time.
    #[test]
    fn a_tap_owes_silence_only_once_it_has_gone_quiet() {
        let second = Duration::from_secs(1);
        assert_eq!(
            silence_owed(second, IDLE / 2, 0, 48_000),
            0,
            "within its idle margin"
        );
        assert_eq!(silence_owed(second, IDLE, 0, 48_000), 4_800, "a tenth");
        assert_eq!(
            silence_owed(second, IDLE, 46_000, 48_000),
            2_000,
            "the rest"
        );
        assert_eq!(
            silence_owed(second, IDLE, 48_000, 48_000),
            0,
            "nothing once caught up"
        );
    }

    /// The first default input, opened, or `None` after saying why not.
    fn try_capture(name: &str) -> Option<(CoreAudioCaptureSource, AudioFormat)> {
        let device = match CoreAudioCaptureSource::list_devices() {
            Ok(devices) => devices
                .into_iter()
                .find(|device| device.kind == CoreAudioDeviceKind::Input && device.is_default),
            Err(error) => {
                eprintln!("skipping: unable to list Core Audio inputs: {error}");
                return None;
            }
        };
        let Some(device) = device else {
            eprintln!("skipping: no default Core Audio input");
            return None;
        };
        match CoreAudioCaptureSource::open(name, CoreAudioCaptureOptions { device }) {
            Ok(opened) => Some(opened),
            Err(error) => {
                eprintln!("skipping: unable to open the default input: {error}");
                None
            }
        }
    }

    /// A device that is not there fails `open` with the Core Audio error
    /// that says so.
    #[test]
    fn a_device_that_is_not_there_is_refused_at_open() {
        let device = CoreAudioDevice {
            id: u32::MAX,
            uid: "gone".into(),
            name: "Gone".into(),
            kind: CoreAudioDeviceKind::Input,
            is_default: false,
        };
        let Err(error) = CoreAudioCaptureSource::open("in", CoreAudioCaptureOptions { device })
        else {
            panic!("a device that is not there cannot be opened");
        };
        assert!(
            matches!(error, CoreAudioCaptureSourceError::CoreAudio { status, .. } if status != 0),
            "{error}"
        );
    }

    /// What the device records arrives in the format `open` said, stamped
    /// one frame after another at the device's rate — and a pause costs
    /// nothing: playing on neither owes the pause's worth at once nor hands
    /// over what it held. What it held is let go of, which is the one gap
    /// there may be, where it played on.
    ///
    /// Microphone permission is not needed for this: denied, the device
    /// records silence, still at its rate.
    #[test]
    fn a_capture_is_continuous_at_the_device_rate_and_a_pause_costs_nothing() {
        let Some((source, format)) = try_capture("test-capture") else {
            return;
        };
        let frames = Arc::new(Mutex::new(Vec::<(i64, usize)>::new()));
        let sink = AppSink::new("test-capture-sink", {
            let frames = Arc::clone(&frames);
            move |buffer| {
                if let MediaBuffer::Audio(frame) = &buffer {
                    assert_eq!(frame.rate(), format.sample_rate);
                    assert_eq!(frame.channels(), format.channels);
                    frames
                        .lock()
                        .expect("frames poisoned")
                        .push((frame.pts().expect("a pts"), frame.samples()));
                }
                Ok(())
            }
        });
        let (pipeline, ()) =
            crate::pipeline::Pipeline::new("test-capture", source, |source, ctx| {
                let branch = ctx.branch().to(sink)?;
                ctx.attach(source, 0, branch)?;
                Ok(())
            })
            .expect("test pipeline wiring must succeed");

        let playing = Duration::from_millis(400);
        pipeline.run().expect("the capture must start");
        thread::sleep(playing);
        pipeline.pause();
        thread::sleep(Duration::from_millis(1000));
        pipeline.resume();
        thread::sleep(playing);
        pipeline.stop();

        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. } | BusEvent::Dropped { .. }))
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
        let frames = frames.lock().expect("frames poisoned");
        let (mut next, mut gaps) = (0, 0);
        for &(pts, samples) in frames.iter() {
            assert!(
                pts >= next,
                "no frame overlaps the one before: {pts} < {next}"
            );
            gaps += usize::from(pts > next);
            next = pts + samples as i64;
        }
        assert!(gaps <= 1, "{gaps} gaps, where only the pause may leave one");
        let played = ((playing * 2).as_secs_f64() * f64::from(format.sample_rate)) as i64;
        // Half, not all: starting the device and the last poll's worth still
        // in flight when this stops come off the window. The pause's own
        // second, owed at once, would put it well past the upper bound.
        assert!(
            next > played / 2 && next < played * 5 / 4,
            "{next} samples where {played} were recorded"
        );
    }

    /// Samples an opened capture hands over in a pipeline that plays for
    /// `playing`, pauses a second, and plays as long again — with every
    /// frame following the one before, and no error or drop on the bus.
    fn captured_across_a_pause(
        name: &str,
        source: CoreAudioCaptureSource,
        playing: Duration,
    ) -> i64 {
        let frames = Arc::new(Mutex::new(Vec::<(i64, usize)>::new()));
        let sink = AppSink::new(format!("{name}-sink"), {
            let frames = Arc::clone(&frames);
            move |buffer| {
                if let MediaBuffer::Audio(frame) = &buffer {
                    frames
                        .lock()
                        .expect("frames poisoned")
                        .push((frame.pts().expect("a pts"), frame.samples()));
                }
                Ok(())
            }
        });
        let (pipeline, ()) = crate::pipeline::Pipeline::new(name, source, |source, ctx| {
            let branch = ctx.branch().to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .expect("test pipeline wiring must succeed");
        pipeline.run().expect("the capture must start");
        thread::sleep(playing);
        pipeline.pause();
        thread::sleep(Duration::from_millis(1000));
        pipeline.resume();
        thread::sleep(playing);
        pipeline.stop();

        let errors: Vec<_> = pipeline
            .bus()
            .iter()
            .filter(|event| matches!(event, BusEvent::Error { .. } | BusEvent::Dropped { .. }))
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
        let frames = frames.lock().expect("frames poisoned");
        let mut next = 0;
        for &(pts, samples) in frames.iter() {
            assert!(
                pts >= next,
                "no frame overlaps the one before: {pts} < {next}"
            );
            next = pts + samples as i64;
        }
        next
    }

    /// What an output device plays keeps a timeline with the clock whether
    /// or not anything plays — a tap hands over nothing while nothing does,
    /// and its silence is made up — and a pause costs it nothing.
    ///
    /// Needs no permission to hold: without it, the tap records silence, at
    /// the same rate. Skips where the system has no taps.
    #[test]
    fn an_output_capture_keeps_up_with_the_clock_and_a_pause_costs_nothing() {
        let Some(device) = CoreAudioCaptureSource::list_devices()
            .ok()
            .and_then(|devices| {
                devices
                    .into_iter()
                    .find(|device| device.kind == CoreAudioDeviceKind::Output && device.is_default)
            })
        else {
            eprintln!("skipping: no default Core Audio output to capture");
            return;
        };
        let (source, format) = match CoreAudioCaptureSource::open(
            "test-output-capture",
            CoreAudioCaptureOptions { device },
        ) {
            Ok(opened) => opened,
            Err(CoreAudioCaptureSourceError::ProcessTapsUnsupported) => {
                eprintln!("skipping: this system has no process taps");
                return;
            }
            Err(error) => panic!("the default output's playback must open: {error}"),
        };
        let playing = Duration::from_millis(400);
        let samples = captured_across_a_pause("test-output-capture", source, playing);
        let played = ((playing * 2).as_secs_f64() * f64::from(format.sample_rate)) as i64;
        // Half, not all: starting the device and the idle margin come off
        // the window. The pause's own second, owed at once, would put it
        // well past the upper bound.
        assert!(
            samples > played / 2 && samples < played * 5 / 4,
            "{samples} samples where {played} were due"
        );
    }

    /// A process that is not a Core Audio client is refused by name, before
    /// anything is made.
    #[test]
    fn a_process_that_plays_nothing_is_refused() {
        match CoreAudioCaptureSource::open_process("test-no-process", u32::MAX) {
            Err(CoreAudioCaptureSourceError::NoSuchProcess(pid)) => assert_eq!(pid, u32::MAX),
            Err(CoreAudioCaptureSourceError::ProcessTapsUnsupported) => {
                eprintln!("skipping: this system has no process taps");
            }
            Err(error) => panic!("expected NoSuchProcess, got {error}"),
            Ok(_) => panic!("a process that is not there cannot be captured"),
        }
    }

    /// What a listed process plays is captured in stereo, on a timeline that
    /// keeps up with the clock whether or not it is making a sound.
    #[test]
    fn a_process_capture_keeps_a_timeline_of_its_own() {
        let processes = CoreAudioCaptureSource::list_processes().unwrap_or_default();
        let Some(process) = processes.first() else {
            eprintln!("skipping: no other process is a Core Audio client");
            return;
        };
        let (source, format) =
            match CoreAudioCaptureSource::open_process("test-process-capture", process.id) {
                Ok(opened) => opened,
                Err(CoreAudioCaptureSourceError::ProcessTapsUnsupported) => {
                    eprintln!("skipping: this system has no process taps");
                    return;
                }
                Err(CoreAudioCaptureSourceError::NoSuchProcess(_)) => {
                    eprintln!("skipping: {} went away", process.executable);
                    return;
                }
                Err(error) => panic!("{} must open: {error}", process.executable),
            };
        assert_eq!(format.channels, 2, "a process is mixed to stereo");
        let playing = Duration::from_millis(400);
        let samples = captured_across_a_pause("test-process-capture", source, playing);
        let played = ((playing * 2).as_secs_f64() * f64::from(format.sample_rate)) as i64;
        assert!(
            samples > played / 2 && samples < played * 5 / 4,
            "{} gave {samples} samples where {played} were due",
            process.executable
        );
    }

    /// Every listed process is another one than this, listed once.
    #[test]
    fn listed_processes_are_others_and_each_once() {
        let processes = CoreAudioCaptureSource::list_processes().expect("processes list");
        let own = std::process::id();
        assert!(
            processes
                .iter()
                .all(|process| process.id != own && process.id != 0)
        );
        let mut ids: Vec<u32> = processes.iter().map(|process| process.id).collect();
        ids.dedup();
        assert_eq!(ids.len(), processes.len());
    }
}
