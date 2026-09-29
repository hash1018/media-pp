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
    element::{Element, ElementType, Produce, Produced, ProducingSource, Wait, element_pp_log},
    elements::{AudioFormat, CoreAudioDevice},
    error::Result,
    platform::macos::coreaudio::{self, HalUnit, OsStatusError, Refcon},
    produce::produce_source,
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

/// Which device a [`CoreAudioCaptureSource`] records from.
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

/// Captures audio from a Core Audio input device — a microphone, a line
/// input, an interface. One src pad, pushing `MediaBuffer::Audio` frames in
/// the device's own rate and channel count as 32-bit float interleaved,
/// which [`CoreAudioCaptureSource::open`] reports. If something downstream
/// needs another format, an [`crate::elements::AudioResampler`] converts —
/// this element does not, as `WasapiCaptureSource` and
/// `PipeWireAudioCaptureSource`, its Windows and Linux counterparts, do not.
///
/// # Timeline
///
/// A running input device calls back at its own rate whatever it hears, so
/// the timeline is continuous without anything filled in: `pts` counts the
/// samples the device has captured, in [`Self::time_base`]. A pause stops
/// the device, and what it would have captured meanwhile is not counted —
/// playing on takes up where it left off.
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
/// Runs until `Stop` — never reaches `Eos` on its own, as no live source in
/// this crate does. An unplugged device ends it with
/// [`CoreAudioCaptureSourceError::DeviceGone`].
pub struct CoreAudioCaptureSource(ProducingSource<Capturing>);

produce_source!(CoreAudioCaptureSource);

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
    device: CoreAudioDevice,
    /// Its pipeline's, for saying captured frames were dropped — `None`
    /// driven by hand.
    bus: Option<Bus>,
    unit: InputUnit,
    shared: Arc<Shared>,
    running: bool,
    /// When a packet last came, or the device was last found alive; `None`
    /// until the first, and again after a pause.
    heard_from: Option<Instant>,
}

impl CoreAudioCaptureSource {
    /// Every device with input channels, the system's default input marked.
    pub fn list_devices() -> std::result::Result<Vec<CoreAudioDevice>, CoreAudioCaptureSourceError>
    {
        Ok(coreaudio::list_input_devices()?)
    }

    /// Opens `options.device` for recording, and returns the format it
    /// captures in: the device's nominal rate and input channels, 32-bit
    /// float interleaved — what a caller needs to build what follows, as
    /// `WasapiCaptureSource::open` returns it.
    ///
    /// The device does not record until the source runs.
    pub fn open(
        name: impl Into<String>,
        options: CoreAudioCaptureOptions,
    ) -> std::result::Result<(Self, AudioFormat), CoreAudioCaptureSourceError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CoreAudioCaptureSource, &name, None);
        let device = options.device;
        let channels = coreaudio::input_channels(device.id)?;
        if channels == 0 {
            return Err(CoreAudioCaptureSourceError::NoInputChannels(device.name));
        }
        let rate = coreaudio::nominal_sample_rate(device.id)?;
        let (Ok(channels_u16), true) = (
            u16::try_from(channels),
            rate.is_finite() && rate >= 1.0 && rate <= f64::from(i32::MAX),
        ) else {
            return Err(CoreAudioCaptureSourceError::UnsupportedFormat {
                name: device.name,
                rate,
                channels,
            });
        };
        let format = AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            rate.round() as u32,
            channels_u16,
        );
        let (unit, shared) = InputUnit::open(device.id, format)?;
        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?} (id {}), {}Hz, {} channel(s), format={:?}",
            device.name,
            device.id,
            format.sample_rate,
            format.channels,
            format.sample_format
        );
        Ok((
            Self(ProducingSource::new(Capturing {
                name,
                pp_log,
                format,
                device,
                bus: None,
                unit,
                shared,
                running: false,
                heard_from: None,
            })),
            format,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in: one sample.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.0.inner().format.sample_rate as i32)
    }
}

/// Wraps one packet as an `ffmpeg` frame, stamped with where the device
/// captured it — so a packet dropped before it leaves its gap in `pts`.
fn build_frame(format: &AudioFormat, packet: &Packet) -> ffmpeg::frame::Audio {
    let mut frame =
        ffmpeg::frame::Audio::new(format.sample_format, packet.frames, format.channel_layout());
    frame.set_rate(format.sample_rate);
    // Only the tight bytes: the frame's own plane may be padded longer.
    let tight_bytes = (packet.frames * format.channels as usize * format.sample_format.bytes())
        .min(packet.bytes.len())
        .min(frame.data_mut(0).len());
    frame.data_mut(0)[..tight_bytes].copy_from_slice(&packet.bytes[..tight_bytes]);
    frame.set_pts(Some(packet.position as i64));
    crate::buffer::set_time_base(
        &mut frame,
        ffmpeg::Rational::new(1, format.sample_rate as i32),
    );
    frame
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
    /// yet. A device that has gone quiet for long is asked whether it is
    /// still there, and one that is not ends the source.
    fn next_frame(&mut self) -> Result<Option<ffmpeg::frame::Audio>> {
        let status = self.shared.render_error.swap(0, Ordering::Relaxed);
        if status != 0 {
            return Err(CoreAudioCaptureSourceError::CoreAudio {
                operation: "render the device's input",
                status,
            }
            .into());
        }
        self.report_dropped();
        let now = Instant::now();
        let Some(packet) = self.shared.captured.pop() else {
            let heard_from = *self.heard_from.get_or_insert(now);
            if self.running && now.duration_since(heard_from) >= SILENT_DEVICE_CHECK {
                if !coreaudio::is_alive(self.device.id) {
                    return Err(
                        CoreAudioCaptureSourceError::DeviceGone(self.device.name.clone()).into(),
                    );
                }
                self.heard_from = Some(now);
            }
            return Ok(None);
        };
        let frame = build_frame(&self.format, &packet);
        let _ = self.shared.free.push(packet);
        self.heard_from = Some(now);
        Ok(Some(frame))
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

impl Produce for Capturing {
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
    /// [`POLL_INTERVAL`] later.
    fn produce(&mut self, wait: &mut Wait<'_>) -> Result<Produced> {
        if let Some(frame) = self.next_frame()? {
            return Ok(Produced::Buffer(MediaBuffer::Audio(Arc::new(frame))));
        }
        if !wait.until(wait.now() + POLL_INTERVAL) {
            return Ok(Produced::Nothing);
        }
        Ok(match self.next_frame()? {
            Some(frame) => Produced::Buffer(MediaBuffer::Audio(Arc::new(frame))),
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
        let frame = build_frame(&format, &packet);
        assert_eq!(frame.samples(), 3);
        assert_eq!(frame.pts(), Some(1_920));
        assert_eq!(frame.rate(), 48_000);
        assert_eq!(
            &frame.data(0)[..3 * 8],
            &(0..24).map(|byte| byte as u8).collect::<Vec<_>>()[..]
        );
    }

    /// The first default input, opened, or `None` after saying why not.
    fn try_capture(name: &str) -> Option<(CoreAudioCaptureSource, AudioFormat)> {
        let device = match CoreAudioCaptureSource::list_devices() {
            Ok(devices) => devices.into_iter().find(|device| device.is_default),
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
}
