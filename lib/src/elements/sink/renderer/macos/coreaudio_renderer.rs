use std::{
    ffi::c_void,
    ptr::{self, NonNull},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use ffmpeg_next::{self as ffmpeg, Rescale, Rounding};
use objc2_audio_toolbox::{
    AURenderCallbackStruct, AudioUnitRenderActionFlags, kAudioOutputUnitProperty_CurrentDevice,
    kAudioUnitProperty_Latency, kAudioUnitProperty_SetRenderCallback,
    kAudioUnitProperty_StreamFormat, kAudioUnitScope_Global, kAudioUnitScope_Input,
};
use objc2_core_audio::{AudioConvertHostTimeToNanos, AudioGetCurrentHostTime};
use objc2_core_audio_types::{
    AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp, AudioTimeStampFlags,
    kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked, kAudioFormatLinearPCM,
};
use thiserror::Error as ThisError;

use super::sample_ring::SampleRing;
use crate::pp_log::{PpLog, pp_debug, pp_error, pp_info, pp_trace, pp_warn};
use crate::{
    buffer::MediaBuffer,
    contract::{InputContract, MediaKind, MemoryDomain, PortContract},
    element::{Element, ElementType, Render, element_pp_log},
    elements::filter::audio::stretcher::{Piece, Stretcher},
    elements::sink::renderer::audio_rate::PlayedMedia,
    elements::{AudioFormat, CoreAudioDevice},
    error::Result,
    platform::macos::coreaudio::{self, HalUnit, OsStatusError, Refcon},
    playback_clock::{AudioMasterRegistration, PlaybackClock, PlaybackClockError},
    render::{RenderStage, render_sink},
    time::{MediaTimestamp, TimeBase},
};

/// How much sound waits between `render` and the device: 100 ms, the size of
/// `WasapiRenderer`'s endpoint buffer.
const RING_DIVISOR: u32 = 10;
/// How long `render` waits before looking again at a full ring.
const POLL_INTERVAL: Duration = Duration::from_millis(2);
/// How much longer than the sound itself a drain waits for the device to
/// take it. A device that has stopped taking sound never finishes, and the
/// end of a stream must not become a hang.
const DRAIN_SLACK: Duration = Duration::from_secs(1);
/// How long a running device may take no sound at all, with the ring full,
/// before `render` gives up on it. It stops calling back when it goes away —
/// unplugged, or taken by another program in exclusive mode — and a waiting
/// `render` would otherwise wait for ever.
const STALL_TIMEOUT: Duration = Duration::from_secs(2);

/// Which device a [`CoreAudioRenderer`] opens.
///
/// The device's own rate and channel count are what it takes — see
/// [`CoreAudioRenderer::open`] — so nothing else is configured here.
#[derive(Debug, Clone)]
pub struct CoreAudioRendererOptions {
    /// One entry out of [`CoreAudioRenderer::list_devices`].
    pub device: CoreAudioDevice,
}

/// Why a [`CoreAudioRenderer`] could not open its device or play a frame.
///
/// [`CoreAudioRendererError::FormatMismatch`] is the one a pipeline hits most
/// often, and it is a wiring problem rather than a device problem: this
/// renderer does not convert, so an
/// [`AudioResampler`](crate::elements::AudioResampler) belongs in front of it.
#[derive(Debug, ThisError)]
pub enum CoreAudioRendererError {
    /// A Core Audio call failed — a device unplugged since it was listed
    /// fails this way, `kAudioHardwareBadObjectError`.
    #[error("Core Audio could not {operation}: OSStatus {status}{}", coreaudio::four_char_code(.status))]
    CoreAudio {
        /// What was being done.
        operation: &'static str,
        /// The `OSStatus` it returned.
        status: i32,
    },
    /// The device has nothing to play to.
    #[error("{0:?} has no output channels")]
    NoOutputChannels(String),
    /// The device runs at a rate, or with a number of channels, an
    /// [`AudioFormat`] cannot describe.
    #[error("{name:?} plays {channels} channel(s) at {rate}Hz, which cannot be rendered to")]
    UnsupportedFormat {
        /// The device's name.
        name: String,
        /// Its nominal sample rate.
        rate: f64,
        /// Its output channels.
        channels: u32,
    },
    /// The device stopped taking sound while it was playing — unplugged,
    /// for one.
    #[error("the device stopped taking audio for {0:?}")]
    Stalled(Duration),
    /// The system has no AUHAL unit to play through.
    #[error("the system has no AUHAL audio unit")]
    NoHalUnit,
    /// The input audio does not exactly match the device's format.
    #[error(
        "audio format mismatch: expected {expected:?}, got {actual:?}; insert AudioResampler before CoreAudioRenderer"
    )]
    FormatMismatch {
        /// What the device takes.
        expected: AudioFormat,
        /// What the frame carried.
        actual: AudioFormat,
    },
    /// The frame's data is shorter than its sample count says.
    #[error("audio frame buffer is shorter than its declared sample count")]
    TruncatedFrame,
    /// It was handed something other than decoded audio.
    #[error("CoreAudioRenderer only renders decoded Audio frames, got a {0}")]
    UnsupportedBuffer(&'static str),
    /// A frame played as the pipeline's audio master has no timestamp.
    #[error("audio frames need a PTS when CoreAudioRenderer is the playback-clock master")]
    MissingPts,
    /// Taking or updating the pipeline's playback clock failed.
    #[error(transparent)]
    PlaybackClock(#[from] PlaybackClockError),
    /// Stretching the sound to the playback rate failed in FFmpeg.
    #[error("stretching the sound to the playback rate failed: {0}")]
    Stretch(ffmpeg::Error),
}

impl From<OsStatusError> for CoreAudioRendererError {
    fn from(error: OsStatusError) -> Self {
        Self::CoreAudio {
            operation: error.operation,
            status: error.status,
        }
    }
}

/// Terminal audio sink playing to a Core Audio output device.
///
/// The device's format — its nominal rate, its output channels, as 32-bit
/// float interleaved — is returned by [`CoreAudioRenderer::open`] so a caller
/// can place an [`crate::elements::AudioResampler`] immediately before this
/// sink. This element intentionally performs no hidden format conversion,
/// and rejects a mismatched frame with
/// [`CoreAudioRendererError::FormatMismatch`] rather than guessing.
///
/// It publishes where the device's listener has actually got to as its
/// pipeline's audio master, and takes the clock to do that from the
/// pipeline rather than from the caller — see `Element::attach_context`. The
/// master slot is claimed on the first frame rather than when the element
/// is wired: claiming it at wiring time would move the clock into audio
/// priming there and then, and video scheduled against the same clock would
/// wait for audio that has not started — which for a branch attached to a
/// running [`crate::elements::Tee`] is a deadlock. The same contract as
/// `WasapiRenderer` and `PipeWireAudioRenderer`, its Windows and Linux
/// counterparts.
///
/// # Pacing
///
/// Core Audio pulls: the output unit calls back on its own real-time thread
/// whenever the device wants sound, and that thread must never wait. So
/// `consume` hands each frame to a ring of 100 ms that the callback reads,
/// and waits while the ring is full — which is what paces a pipeline to the
/// device. Put a [`crate::queue::Queue`] immediately before this sink when
/// that waiting must not hold up another branch. An empty ring is covered
/// with silence rather than stalling the device, and silence does not move
/// where playback is.
///
/// Stopping the device for a pause lets go of what it was playing at that
/// moment, one IO buffer at most (about 10 ms); the position it masters
/// steps over it, so the picture keeps to the sound.
pub struct CoreAudioRenderer(RenderStage<Rendering>);

render_sink!(CoreAudioRenderer);

/// What the output unit's callback shares with the element.
struct Shared {
    ring: SampleRing,
    bytes_per_frame: usize,
    sample_rate: u32,
    /// From the moment the IO thread hands the device a sample to the moment
    /// it is heard. Written once, after the unit is initialized.
    latency_ns: AtomicU64,
    /// Frames of sound the callback has handed the device. Silence written
    /// over an empty ring is not counted, so a gap in delivery does not move
    /// the media on.
    played_frames: AtomicU64,
    /// When what has been handed over is all heard — see [`Heard`].
    heard: Mutex<Heard>,
}

/// `frames` of sound handed to the device are all heard at host time
/// `at_ns`, and each before it one sample's length earlier. The callback
/// updates it as it hands over more, and a pause fixes it where it stopped.
#[derive(Clone, Copy, Default)]
struct Heard {
    frames: u64,
    at_ns: u64,
}

impl Shared {
    fn frames_ns(&self, frames: u64) -> u64 {
        ((u128::from(frames) * 1_000_000_000) / u128::from(self.sample_rate.max(1)))
            .min(u128::from(u64::MAX)) as u64
    }

    /// How many of the frames handed to the device have been heard at host
    /// time `now_ns`.
    fn heard_frames(&self, now_ns: u64) -> u64 {
        let heard = *self
            .heard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ahead_ns = heard.at_ns.saturating_sub(now_ns);
        let ahead = (u128::from(ahead_ns) * u128::from(self.sample_rate) / 1_000_000_000) as u64;
        heard.frames.saturating_sub(ahead)
    }

    /// Everything handed over counts as heard: the device has stopped, and
    /// what it had not played by then it never will.
    fn settle(&self) {
        let frames = self.played_frames.load(Ordering::Acquire);
        *self
            .heard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Heard { frames, at_ns: 0 };
    }

    /// The IO thread's half: fills `out` with what the ring holds, whole
    /// frames of it, and silence after. Returns how many bytes were sound.
    fn fill(&self, out: &mut [u8], output_ns: u64) -> usize {
        let whole = out.len() - out.len() % self.bytes_per_frame.max(1);
        let copied = self.ring.read_into(&mut out[..whole]);
        out[copied..].fill(0);
        let frames = (copied / self.bytes_per_frame.max(1)) as u64;
        if frames > 0 {
            let played = self.played_frames.fetch_add(frames, Ordering::AcqRel) + frames;
            let at_ns = output_ns
                .saturating_add(self.frames_ns(frames))
                .saturating_add(self.latency_ns.load(Ordering::Relaxed));
            // Never waits: a reader holding the lock only makes this update
            // miss, and the next callback makes it.
            if let Ok(mut heard) = self.heard.try_lock() {
                *heard = Heard {
                    frames: played,
                    at_ns,
                };
            }
        }
        copied
    }
}

fn host_now_ns() -> u64 {
    // SAFETY: both are pure reads of the host clock, with no preconditions.
    unsafe { AudioConvertHostTimeToNanos(AudioGetCurrentHostTime()) }
}

/// The output unit's render callback, on Core Audio's real-time IO thread.
///
/// Waits on nothing and allocates nothing: it copies out of the ring, covers
/// the rest with silence, and notes when what it handed over will be heard.
unsafe extern "C-unwind" fn render_callback(
    refcon: NonNull<c_void>,
    flags: NonNull<AudioUnitRenderActionFlags>,
    timestamp: NonNull<AudioTimeStamp>,
    _bus: u32,
    _frames: u32,
    data: *mut AudioBufferList,
) -> i32 {
    // SAFETY: `refcon` is the `Shared` `OutputUnit::open` registered with the
    // callback, whose `Refcon` keeps a strong reference to it until the unit
    // that calls this is disposed.
    let shared = unsafe { refcon.cast::<Shared>().as_ref() };
    let Some(data) = NonNull::new(data) else {
        return 0;
    };
    // SAFETY: Core Audio passes a timestamp valid for this call.
    let timestamp = unsafe { timestamp.read() };
    let output_ns = if timestamp
        .mFlags
        .contains(AudioTimeStampFlags::HostTimeValid)
    {
        // SAFETY: a pure conversion of the host time Core Audio gave.
        unsafe { AudioConvertHostTimeToNanos(timestamp.mHostTime) }
    } else {
        host_now_ns()
    };
    let list = data.as_ptr();
    // SAFETY: Core Audio passes a buffer list, of the format set on the unit's
    // input — interleaved, so one buffer — that it owns for this call; the
    // count is its first field.
    let buffers = unsafe { (*list).mNumberBuffers } as usize;
    // SAFETY: as above; only the field's address is taken.
    let first = unsafe { ptr::addr_of_mut!((*list).mBuffers) }
        .cast::<objc2_core_audio_types::AudioBuffer>();
    let mut sound = 0;
    for index in 0..buffers {
        // SAFETY: the list holds `buffers` buffers from `mBuffers` on.
        let buffer = unsafe { &*first.add(index) };
        let Some(bytes) = NonNull::new(buffer.mData.cast::<u8>()) else {
            continue;
        };
        // SAFETY: Core Audio's buffer is writable for `mDataByteSize` bytes
        // for the length of this call, and nothing else touches it.
        let out = unsafe {
            std::slice::from_raw_parts_mut(bytes.as_ptr(), buffer.mDataByteSize as usize)
        };
        if index == 0 {
            sound = shared.fill(out, output_ns);
        } else {
            out.fill(0);
        }
    }
    if sound == 0 {
        // SAFETY: the flags are Core Audio's, writable for this call.
        unsafe {
            (*flags.as_ptr()).0 |= AudioUnitRenderActionFlags::UnitRenderAction_OutputIsSilence.0
        };
    }
    0
}

/// An AUHAL unit playing one device, and the callback's reference to what it
/// reads — declared after the unit, so the unit is disposed of first.
struct OutputUnit {
    unit: HalUnit,
    _shared: Refcon<Shared>,
}

impl OutputUnit {
    /// An AUHAL unit playing `device` in `format`, calling back into
    /// `shared`, initialized and stopped.
    fn open(
        device: u32,
        format: AudioFormat,
        shared: Arc<Shared>,
    ) -> std::result::Result<Self, CoreAudioRendererError> {
        let mut unit = HalUnit::new()?.ok_or(CoreAudioRendererError::NoHalUnit)?;
        let shared = Refcon::new(shared);
        unit.set(
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &device,
            "point the output unit at the device",
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
        unit.set(
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            0,
            &stream,
            "give the output unit its input format",
        )?;
        let callback = AURenderCallbackStruct {
            inputProc: Some(render_callback),
            inputProcRefCon: shared.as_ptr(),
        };
        unit.set(
            kAudioUnitProperty_SetRenderCallback,
            kAudioUnitScope_Input,
            0,
            &callback,
            "set the output unit's render callback",
        )?;
        unit.initialize()?;
        Ok(Self {
            unit,
            _shared: shared,
        })
    }

    /// The unit's own processing latency, which AUHAL reports as none on
    /// every device seen so far; asked rather than assumed.
    fn latency_ns(&self) -> u64 {
        match self.unit.get::<f64>(
            kAudioUnitProperty_Latency,
            kAudioUnitScope_Global,
            0,
            "read the output unit's latency",
        ) {
            Ok(seconds) if seconds.is_finite() && seconds > 0.0 => (seconds * 1e9) as u64,
            _ => 0,
        }
    }
}

/// What a [`CoreAudioRenderer`] does with each frame: plays it on the device.
/// All of its work, which the framework makes the terminal.
struct Rendering {
    name: Arc<str>,
    pp_log: PpLog,
    format: AudioFormat,
    unit: OutputUnit,
    shared: Arc<Shared>,
    /// How many bytes the ring holds before the device is started: half of
    /// it, so the first callbacks find sound rather than splice silence in.
    prime_bytes: usize,
    running: bool,
    /// Whether the device is stopped for a pause — set and cleared by the
    /// framework's `pausing` and `resuming`, and nothing else: a seek's reset
    /// leaves it, so a paused seek's one sample waits in the ring.
    paused: bool,
    clock_binding: PlaybackClockBinding,
    timeline: Option<Timeline>,
    /// The sound at the playback rate — see [`Pipeline::set_rate`](crate::pipeline::Pipeline::set_rate).
    stretcher: Stretcher,
}

/// Mirrors the other renderers' binding states, so a dynamically attached
/// branch can defer claiming the exclusive audio-master slot.
enum PlaybackClockBinding {
    Unbound,
    Deferred(Arc<PlaybackClock>),
    Registered(AudioMasterRegistration),
}

impl PlaybackClockBinding {
    fn is_bound(&self) -> bool {
        !matches!(self, Self::Unbound)
    }

    fn registration(&self) -> Option<&AudioMasterRegistration> {
        match self {
            Self::Registered(master) => Some(master),
            Self::Unbound | Self::Deferred(_) => None,
        }
    }

    /// The rate playback goes at, 1.0 for a renderer no pipeline wired.
    fn rate(&self) -> f64 {
        match self {
            Self::Unbound => 1.0,
            Self::Deferred(clock) => clock.rate(),
            Self::Registered(master) => master.rate(),
        }
    }

    fn ensure_registered(&mut self) -> std::result::Result<(), PlaybackClockError> {
        if let Self::Deferred(playback_clock) = self {
            let registration = playback_clock.register_audio_master()?;
            *self = Self::Registered(registration);
        }
        Ok(())
    }
}

struct Timeline {
    /// `played_frames` when this timeline's first sample was handed over.
    played_origin: u64,
    /// Which media the frames handed over since then stand for.
    media: PlayedMedia,
}

impl CoreAudioRenderer {
    /// Every device with output channels, the system's default marked.
    pub fn list_devices() -> std::result::Result<Vec<CoreAudioDevice>, CoreAudioRendererError> {
        Ok(coreaudio::list_output_devices()?)
    }

    /// Opens `options.device` for playback, and returns the format it takes:
    /// the device's nominal rate and output channels, 32-bit float
    /// interleaved.
    ///
    /// The device stays stopped until there is sound to play.
    pub fn open(
        name: impl Into<String>,
        options: CoreAudioRendererOptions,
    ) -> std::result::Result<(Self, AudioFormat), CoreAudioRendererError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::CoreAudioRenderer, &name, None);
        let device = options.device;
        let channels = coreaudio::output_channels(device.id)?;
        if channels == 0 {
            return Err(CoreAudioRendererError::NoOutputChannels(device.name));
        }
        let rate = coreaudio::nominal_sample_rate(device.id)?;
        let (Ok(channels_u16), true) = (
            u16::try_from(channels),
            rate.is_finite() && rate >= 1.0 && rate <= f64::from(i32::MAX),
        ) else {
            return Err(CoreAudioRendererError::UnsupportedFormat {
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
        let bytes_per_frame = format.sample_format.bytes() * usize::from(format.channels);
        let ring_frames = (format.sample_rate / RING_DIVISOR).max(1) as usize;
        let shared = Arc::new(Shared {
            ring: SampleRing::new(ring_frames * bytes_per_frame),
            bytes_per_frame,
            sample_rate: format.sample_rate,
            latency_ns: AtomicU64::new(0),
            played_frames: AtomicU64::new(0),
            heard: Mutex::new(Heard::default()),
        });
        let unit = OutputUnit::open(device.id, format, Arc::clone(&shared))?;
        let latency_ns = shared
            .frames_ns(u64::from(coreaudio::output_latency_frames(device.id)))
            .saturating_add(unit.latency_ns());
        shared.latency_ns.store(latency_ns, Ordering::Relaxed);
        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?} (id {}), {}Hz, {} channel(s), format={:?}, ring_frames={ring_frames}, latency={:?}",
            device.name,
            device.id,
            format.sample_rate,
            format.channels,
            format.sample_format,
            Duration::from_nanos(latency_ns)
        );

        Ok((
            Self(RenderStage::new(Rendering {
                name,
                pp_log,
                format,
                unit,
                prime_bytes: (ring_frames / 2).max(1) * bytes_per_frame,
                shared,
                running: false,
                paused: false,
                clock_binding: PlaybackClockBinding::Unbound,
                timeline: None,
                stretcher: Stretcher::new(format),
            })),
            format,
        ))
    }

    /// The format [`Sink::consume`](crate::element::Sink::consume) accepts, unchanged since `open`.
    pub fn format(&self) -> AudioFormat {
        self.0.inner.format
    }
}

impl Rendering {
    fn start(&mut self) -> Result<()> {
        if !self.running {
            self.unit
                .unit
                .start()
                .map_err(CoreAudioRendererError::from)?;
            self.running = true;
        }
        Ok(())
    }

    /// Stops the device, and counts what it had not played as heard.
    fn stop(&mut self) -> Result<()> {
        if self.running {
            self.unit
                .unit
                .stop()
                .map_err(CoreAudioRendererError::from)?;
            self.running = false;
        }
        self.shared.settle();
        Ok(())
    }

    /// Starts the device once the ring holds enough to survive its first
    /// callbacks — see `prime_bytes`.
    fn start_once_primed(&mut self) -> Result<()> {
        if !self.paused && self.shared.ring.len() >= self.prime_bytes {
            self.start()?;
        }
        Ok(())
    }

    fn stop_and_reset(&mut self) -> Result<()> {
        self.stop()?;
        self.publish_position(false)?;
        // The callback no longer runs, so the writer may let go of what the
        // reader has not taken.
        self.shared.ring.clear();
        self.timeline = None;
        self.stretcher.reset();
        Ok(())
    }

    /// Publishes where the listener has got to, if this element is the
    /// audio master.
    fn publish_position(&self, running: bool) -> Result<()> {
        let (Some(master), Some(timeline)) = (self.clock_binding.registration(), &self.timeline)
        else {
            return Ok(());
        };
        let heard = self
            .shared
            .heard_frames(host_now_ns())
            .saturating_sub(timeline.played_origin);
        let position_ns = timeline.media.played(heard);
        master
            .publish(position_ns, timeline.media.handed_until(), running)
            .map_err(CoreAudioRendererError::from)?;
        Ok(())
    }

    fn audio_pts_ns(&self, frame: &ffmpeg::frame::Audio) -> Result<i64> {
        let pts = frame.pts().ok_or(CoreAudioRendererError::MissingPts)?;
        let source =
            TimeBase::new_unchecked(ffmpeg::Rational::new(1, self.format.sample_rate as i32));
        let nanos = TimeBase::new_unchecked(ffmpeg::Rational::new(1, 1_000_000_000));
        Ok(MediaTimestamp::new_unchecked(pts, source).rescale(nanos))
    }

    fn sample_offset_ns(&self, samples: usize) -> i64 {
        (samples as i64).rescale(
            ffmpeg::Rational::new(1, self.format.sample_rate as i32),
            ffmpeg::Rational::new(1, 1_000_000_000),
        )
    }

    /// Hands `frame` to the device. Paused — a paused seek's one sample — it
    /// waits in the ring, silent, until `Resume` starts the device.
    fn play(&mut self, frame: &ffmpeg::frame::Audio) -> Result<()> {
        validate_frame(self.format, frame)?;
        if frame.samples() == 0 {
            return Ok(());
        }
        // Deferred to the first real frame, so a dynamically attached branch
        // cannot stall video before it primes.
        self.clock_binding
            .ensure_registered()
            .map_err(CoreAudioRendererError::from)?;

        let frame_pts_ns = if self.clock_binding.registration().is_some() {
            Some(self.audio_pts_ns(frame)?)
        } else {
            None
        };
        // Taking over from the wall clock mid-stream, what is before where
        // it had got to is not played: the sound starts where the picture is.
        let mut frame_offset = 0usize;
        if let (Some(master), Some(frame_pts_ns)) =
            (self.clock_binding.registration(), frame_pts_ns)
            && let Some(target_ns) = master
                .priming_target_ns()
                .map_err(CoreAudioRendererError::from)?
            && target_ns > frame_pts_ns
        {
            frame_offset = priming_trim_samples(frame_pts_ns, target_ns, self.format.sample_rate);
            if frame_offset >= frame.samples() {
                return Ok(());
            }
        }

        // At the file's own speed the frame goes as it came. At a rate, what
        // is left of it after the trim is stretched to it first.
        let rate = self.clock_binding.rate();
        if self.stretcher.passes(rate) {
            return self.submit(frame, frame_offset, frame_pts_ns, 1.0);
        }
        let trimmed;
        let input = if frame_offset > 0 {
            trimmed = trim(self.format, frame, frame_offset);
            &trimmed
        } else {
            frame
        };
        let media_ns =
            frame_pts_ns.map(|pts| pts.saturating_add(self.sample_offset_ns(frame_offset)));
        let pieces = self
            .stretcher
            .stretch(input, media_ns.unwrap_or(0), rate)
            .map_err(CoreAudioRendererError::Stretch)?;
        self.submit_pieces(pieces, Some(input), media_ns)
    }

    /// Hands the device what the stretcher answered, in order: `input` for
    /// [`Piece::AsIs`], at `media_ns`.
    fn submit_pieces(
        &mut self,
        pieces: Vec<Piece>,
        input: Option<&ffmpeg::frame::Audio>,
        media_ns: Option<i64>,
    ) -> Result<()> {
        let timed = self.clock_binding.registration().is_some();
        for piece in pieces {
            match piece {
                Piece::AsIs => {
                    if let Some(input) = input {
                        self.submit(input, 0, media_ns, 1.0)?;
                    }
                }
                Piece::Stretched {
                    frame,
                    media_ns: stretched_ns,
                    rate,
                } => self.submit(&frame, 0, timed.then_some(stretched_ns), rate)?,
            }
        }
        Ok(())
    }

    /// Writes `frame` from `frame_offset` on into the ring, waiting for room
    /// as it goes. Its first sample is at `frame_pts_ns` in the media, and
    /// each sample of it stands for `rate` samples' worth.
    fn submit(
        &mut self,
        frame: &ffmpeg::frame::Audio,
        mut frame_offset: usize,
        frame_pts_ns: Option<i64>,
        rate: f64,
    ) -> Result<()> {
        let bytes = validate_frame(self.format, frame)?;
        let bytes_per_frame = self.shared.bytes_per_frame;
        let mut progress = (
            self.shared.played_frames.load(Ordering::Acquire),
            Instant::now(),
        );
        while frame_offset < frame.samples() {
            let free = (self.shared.ring.capacity() - self.shared.ring.len()) / bytes_per_frame;
            if free == 0 {
                if self.paused {
                    // Stopped, the device takes nothing until it is resumed,
                    // and what does not fit waits no longer than the pause:
                    // dropped, as `WasapiRenderer` drops it.
                    pp_debug!(
                        self,
                        "paused with the ring full; dropping what does not fit"
                    );
                    return Ok(());
                }
                // A full ring is primed whatever its size.
                self.start()?;
                let played = self.shared.played_frames.load(Ordering::Acquire);
                if played != progress.0 {
                    progress = (played, Instant::now());
                } else if progress.1.elapsed() >= STALL_TIMEOUT {
                    return Err(CoreAudioRendererError::Stalled(progress.1.elapsed()).into());
                }
                self.publish_position(true)?;
                thread::sleep(POLL_INTERVAL);
                continue;
            }
            let take = free.min(frame.samples() - frame_offset);
            // The ring holds nothing of an earlier timeline when this one
            // begins — a seek or a stop cleared it — so what the device has
            // played so far is exactly where this one's first sample falls.
            let played_origin = self.shared.played_frames.load(Ordering::Acquire)
                + (self.shared.ring.len() / bytes_per_frame) as u64;
            let from = frame_offset * bytes_per_frame;
            let written = self
                .shared
                .ring
                .write(&bytes[from..from + take * bytes_per_frame]);
            debug_assert_eq!(written, take * bytes_per_frame, "only the writer fills");
            if let Some(frame_pts_ns) = frame_pts_ns {
                let sample_rate = self.format.sample_rate;
                let offset_ns = (self.sample_offset_ns(frame_offset) as f64 * rate) as i64;
                self.timeline
                    .get_or_insert_with(|| Timeline {
                        played_origin,
                        media: PlayedMedia::new(
                            sample_rate,
                            frame_pts_ns.saturating_add(offset_ns),
                        ),
                    })
                    .media
                    .push(take as u64, rate);
            }
            frame_offset += take;
            // Paused, it waits in the ring: `Resume` starts the device.
            if !self.paused {
                self.start_once_primed()?;
                self.publish_position(true)?;
            }
        }
        Ok(())
    }

    /// Plays out what the ring and the device still hold, then reports the
    /// final position and stops.
    ///
    /// Paused, nothing is played out: starting the device to drain it is the
    /// blip a paused seek must not make, and what a paused seek to the end
    /// left in the ring goes with it. With nothing handed over there is
    /// nothing to wait for either, and it returns at once.
    fn play_out(&mut self) -> Result<()> {
        // What the stretcher still holds is the end of the sound too.
        let pieces = self
            .stretcher
            .finish()
            .map_err(CoreAudioRendererError::Stretch)?;
        if !self.paused {
            self.submit_pieces(pieces, None, None)?;
            if !self.shared.ring.is_empty() {
                self.start()?;
            }
            let outstanding = (self.shared.ring.len() / self.shared.bytes_per_frame) as u64;
            let deadline = Instant::now()
                + Duration::from_nanos(self.shared.frames_ns(outstanding))
                + DRAIN_SLACK;
            while !self.shared.ring.is_empty() {
                if Instant::now() >= deadline {
                    pp_warn!(
                        self,
                        "the device stopped taking audio during drain: {} frame(s) never played",
                        self.shared.ring.len() / self.shared.bytes_per_frame
                    );
                    break;
                }
                self.publish_position(true)?;
                thread::sleep(POLL_INTERVAL);
            }
            // The ring is empty; what the device took is heard once its
            // latency has passed.
            let heard_at = self
                .shared
                .heard
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .at_ns;
            let tail = Duration::from_nanos(heard_at.saturating_sub(host_now_ns()));
            if self.running && tail > Duration::ZERO {
                let until = Instant::now() + tail.min(DRAIN_SLACK);
                while Instant::now() < until {
                    self.publish_position(true)?;
                    thread::sleep(
                        POLL_INTERVAL.min(until.saturating_duration_since(Instant::now())),
                    );
                }
            }
        }
        self.publish_position(false)?;
        let final_position = self
            .timeline
            .as_ref()
            .map(|timeline| timeline.media.handed_until());
        self.stop_and_reset()?;
        if let (Some(master), Some(final_position)) =
            (self.clock_binding.registration(), final_position)
        {
            master
                .finish(final_position)
                .map_err(CoreAudioRendererError::from)?;
        }
        Ok(())
    }
}

fn validate_frame(
    expected: AudioFormat,
    frame: &ffmpeg::frame::Audio,
) -> std::result::Result<&[u8], CoreAudioRendererError> {
    let actual = AudioFormat::new(frame.format(), frame.rate(), frame.channels());
    if actual != expected {
        return Err(CoreAudioRendererError::FormatMismatch { expected, actual });
    }
    let tight_bytes = frame
        .samples()
        .saturating_mul(expected.channels as usize)
        .saturating_mul(expected.sample_format.bytes());
    frame
        .data(0)
        .get(..tight_bytes)
        .ok_or(CoreAudioRendererError::TruncatedFrame)
}

/// `frame` from its `offset`th sample on, as a frame of its own.
fn trim(format: AudioFormat, frame: &ffmpeg::frame::Audio, offset: usize) -> ffmpeg::frame::Audio {
    let samples = frame.samples() - offset;
    let mut trimmed =
        ffmpeg::frame::Audio::new(format.sample_format, samples, frame.channel_layout());
    trimmed.set_rate(format.sample_rate);
    let bytes_per_frame = format.sample_format.bytes() * format.channels as usize;
    let from = offset * bytes_per_frame;
    let len = samples * bytes_per_frame;
    trimmed.data_mut(0)[..len].copy_from_slice(&frame.data(0)[from..from + len]);
    trimmed
}

fn priming_trim_samples(frame_pts_ns: i64, target_ns: i64, sample_rate: u32) -> usize {
    target_ns
        .saturating_sub(frame_pts_ns)
        .max(0)
        .rescale_with(
            ffmpeg::Rational::new(1, 1_000_000_000),
            ffmpeg::Rational::new(1, sample_rate as i32),
            Rounding::Up,
        )
        .max(0) as usize
}

impl Element for Rendering {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::CoreAudioRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }

    /// Takes the pipeline's playback clock, and claims its audio-master slot
    /// on the first frame rather than here.
    ///
    /// Deferred deliberately. Claiming it now would move the clock into
    /// `AudioPriming` the moment this is wired, and video scheduled against
    /// the same clock would then hold for audio that has not started — a
    /// branch attached to a running `Tee` can sit behind a demuxer blocked on
    /// a full video queue, which is exactly the deadlock this ordering
    /// avoids. An exclusive-master conflict therefore surfaces from the first
    /// `consume` rather than from wiring.
    fn attach_context(&mut self, context: &Arc<crate::element::Context>) {
        if !self.clock_binding.is_bound() {
            self.clock_binding =
                PlaybackClockBinding::Deferred(Arc::clone(&context.playback_clock));
        }
    }
}

impl Render for Rendering {
    /// Writes samples into the ring the output unit plays from.
    fn input_contract(&self) -> InputContract {
        InputContract::Fixed(PortContract::frame(
            MediaKind::AudioFrame,
            MemoryDomain::System,
        ))
    }

    fn render(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => self
                .play(&frame)
                .inspect_err(|error| pp_error!(self, "render failed: {error}")),
            other => Err(CoreAudioRendererError::UnsupportedBuffer(other.kind()).into()),
        }
    }

    /// The end of the stream is played out before it is taken.
    fn drain(&mut self) -> Result<()> {
        let outcome = self.play_out();
        pp_trace!(
            self,
            "event=eos phase=drained outcome={}",
            if outcome.is_ok() { "ok" } else { "error" }
        );
        outcome
    }

    /// Stops the device for the pause; what the ring holds stays there.
    fn pausing(&mut self) -> Result<()> {
        self.stop()?;
        self.publish_position(false)?;
        self.paused = true;
        Ok(())
    }

    /// Starts the device again where the ring holds anything to play.
    fn resuming(&mut self) -> Result<()> {
        self.paused = false;
        if !self.shared.ring.is_empty() {
            self.start()?;
            self.publish_position(true)?;
        }
        Ok(())
    }

    /// Discards what the ring holds, for a seek or a stop alike. Whether it
    /// is paused is left alone: that is `pausing` and `resuming`'s, and a
    /// reset that cleared it would let a *paused* seek's one preroll sample
    /// reach the device — a blip and a click as it started and stopped
    /// around it.
    fn reset(&mut self) -> Result<()> {
        self.stop_and_reset()
    }
}

impl Drop for Rendering {
    fn drop(&mut self) {
        // A dynamically detached renderer hands the position it last played
        // back to the playback clock before its registration drops, or video
        // resumes wall-clock pacing from an older update, behind what was
        // heard. The unit itself is disposed of by `OutputUnit`'s own drop.
        let _ = self.stop();
        let _ = self.publish_position(false);
    }
}

#[cfg(test)]
mod tests {
    use ffmpeg::format::sample::Type;

    use super::*;
    use crate::element::{Sink, SinkExt};
    use crate::{
        clock::Clock, control::ControlMsg, playback_clock::PlaybackMaster, stream::StreamEvent,
    };

    fn frame(format: AudioFormat, samples: usize) -> ffmpeg::frame::Audio {
        let mut frame =
            ffmpeg::frame::Audio::new(format.sample_format, samples, format.channel_layout());
        frame.set_rate(format.sample_rate);
        frame.data_mut(0).fill(0);
        frame
    }

    fn shared(sample_rate: u32, latency_ns: u64) -> Shared {
        Shared {
            ring: SampleRing::new(64 * 8),
            bytes_per_frame: 8,
            sample_rate,
            latency_ns: AtomicU64::new(latency_ns),
            played_frames: AtomicU64::new(0),
            heard: Mutex::new(Heard::default()),
        }
    }

    /// The default device, opened, or `None` after saying why not.
    fn try_renderer(name: &str) -> Option<CoreAudioRenderer> {
        let device = match CoreAudioRenderer::list_devices() {
            Ok(devices) => devices.into_iter().find(|device| device.is_default)?,
            Err(error) => {
                eprintln!("skipping: unable to list Core Audio devices: {error}");
                return None;
            }
        };
        match CoreAudioRenderer::open(name, CoreAudioRendererOptions { device }) {
            Ok((renderer, _)) => Some(renderer),
            Err(error) => {
                eprintln!("skipping: unable to open the default output: {error}");
                None
            }
        }
        .or_else(|| {
            eprintln!("skipping: no default Core Audio output");
            None
        })
    }

    /// What the callback takes from the ring is sound, the rest of the
    /// device's buffer silence, and only the sound moves playback on.
    #[test]
    fn an_empty_ring_is_covered_with_silence_that_does_not_count() {
        let shared = shared(48_000, 0);
        shared.ring.write(&[7; 3 * 8]);
        let mut out = [0xFF; 5 * 8];

        assert_eq!(shared.fill(&mut out, 1_000), 3 * 8);
        assert!(out[..3 * 8].iter().all(|&byte| byte == 7), "the sound");
        assert!(out[3 * 8..].iter().all(|&byte| byte == 0), "then silence");
        assert_eq!(shared.played_frames.load(Ordering::Acquire), 3);

        assert_eq!(shared.fill(&mut out, 2_000), 0, "nothing left");
        assert_eq!(
            shared.played_frames.load(Ordering::Acquire),
            3,
            "silence is not media time"
        );
    }

    /// What has been handed to the device is heard its latency later, one
    /// sample after another — so before then less of it counts, and a
    /// stopped device's all of it.
    #[test]
    fn what_is_heard_follows_the_device_timestamps_and_its_latency() {
        // 1000 Hz: a frame is a millisecond. 10 ms of latency.
        let shared = shared(1_000, 10_000_000);
        shared.ring.write(&[0; 20 * 8]);
        shared.fill(&mut [0; 20 * 8], 100_000_000);
        // The 20 frames are all heard at 100 + 20 + 10 = 130 ms.
        assert_eq!(shared.heard_frames(130_000_000), 20);
        assert_eq!(shared.heard_frames(125_000_000), 15);
        assert_eq!(shared.heard_frames(100_000_000), 0);
        assert_eq!(
            shared.heard_frames(500_000_000),
            20,
            "no further than handed"
        );

        shared.settle();
        assert_eq!(shared.heard_frames(0), 20, "stopped, all of it counts");
    }

    #[test]
    fn validates_the_exact_device_format() {
        let expected = AudioFormat::new(ffmpeg::format::Sample::F32(Type::Packed), 48_000, 2);
        let frame = frame(expected, 480);
        assert_eq!(validate_frame(expected, &frame).unwrap().len(), 480 * 2 * 4);
    }

    #[test]
    fn rejects_audio_that_skipped_the_required_resampler() {
        let expected = AudioFormat::new(ffmpeg::format::Sample::F32(Type::Packed), 48_000, 2);
        let actual = AudioFormat::new(ffmpeg::format::Sample::I16(Type::Packed), 44_100, 1);
        let error = validate_frame(expected, &frame(actual, 441)).unwrap_err();
        assert!(matches!(
            error,
            CoreAudioRendererError::FormatMismatch {
                expected: error_expected,
                actual: error_actual,
            } if error_expected == expected && error_actual == actual
        ));
    }

    #[test]
    fn an_os_status_names_its_four_character_code() {
        let error: CoreAudioRendererError = OsStatusError {
            operation: "read a device's sample rate",
            status: i32::from_be_bytes(*b"!obj"),
        }
        .into();
        assert_eq!(
            error.to_string(),
            "Core Audio could not read a device's sample rate: OSStatus 560947818 ('!obj')"
        );
    }

    #[test]
    fn binding_does_not_claim_the_clock_until_audio_can_prime_it() {
        let playback = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        playback.ensure_wall_origin(1_000);
        let mut binding = PlaybackClockBinding::Deferred(playback.clone());

        assert_eq!(playback.master(), PlaybackMaster::Wall);
        binding.ensure_registered().unwrap();
        assert!(matches!(binding, PlaybackClockBinding::Registered(_)));
        assert_eq!(playback.master(), PlaybackMaster::AudioPriming);
    }

    /// A device that is not there fails `open` with the Core Audio error
    /// that says so, before anything is created.
    #[test]
    fn a_device_that_is_not_there_is_refused_at_open() {
        let device = CoreAudioDevice {
            id: u32::MAX,
            uid: "gone".into(),
            name: "Gone".into(),
            is_default: false,
        };
        let Err(error) = CoreAudioRenderer::open("out", CoreAudioRendererOptions { device }) else {
            panic!("a device that is not there cannot be opened");
        };
        assert!(
            matches!(error, CoreAudioRendererError::CoreAudio { status, .. } if status != 0),
            "{error}"
        );
    }

    /// Sound starts the device once the ring is primed, and a pause stops it
    /// with what is left still in the ring for `Resume`.
    #[test]
    fn sound_starts_the_device_and_a_pause_keeps_what_is_left() {
        let Some(mut renderer) = try_renderer("start-test") else {
            return;
        };
        let format = renderer.format();
        let samples = format.sample_rate as usize / 10;
        renderer
            .consume(MediaBuffer::Audio(Arc::new(frame(format, samples))))
            .expect("silence plays");
        assert!(renderer.0.inner.running, "a full ring starts the device");

        renderer.control(&ControlMsg::Pause).expect("pause");
        assert!(!renderer.0.inner.running, "stopped for the pause");
        assert!(
            !renderer.0.inner.shared.ring.is_empty(),
            "and still holding sound"
        );

        renderer.control(&ControlMsg::Resume).expect("resume");
        assert!(renderer.0.inner.running, "playing again");
    }

    /// A flush lets go of the ring and leaves a pause as it was: only
    /// `Resume` lifts it.
    #[test]
    fn a_flush_empties_the_ring_and_keeps_the_pause() {
        let Some(mut renderer) = try_renderer("flush-test") else {
            return;
        };
        let format = renderer.format();
        renderer.control(&ControlMsg::Pause).expect("pause");
        renderer
            .consume(MediaBuffer::Audio(Arc::new(frame(format, 480))))
            .expect("taken while paused");
        assert!(!renderer.0.inner.shared.ring.is_empty(), "held in the ring");
        assert!(!renderer.0.inner.running, "and not played while paused");

        renderer.control(&ControlMsg::Flush).expect("flush");
        assert!(renderer.0.inner.shared.ring.is_empty(), "flushed");
        assert!(renderer.0.inner.paused, "a flush must not resume");

        renderer.control(&ControlMsg::Resume).expect("resume");
        assert!(!renderer.0.inner.paused, "only Resume lifts the pause");
        assert!(
            !renderer.0.inner.running,
            "with nothing to play, it stays stopped"
        );
    }

    /// The end of the stream waits for what was handed over to be played —
    /// not less, however soon the ring empties — and then stops the device.
    #[test]
    fn the_end_of_the_stream_waits_for_the_sound_to_play() {
        let Some(mut renderer) = try_renderer("drain-test") else {
            return;
        };
        let format = renderer.format();
        let samples = format.sample_rate as usize * 3 / 10;
        let started = Instant::now();
        renderer
            .consume(MediaBuffer::Audio(Arc::new(frame(format, samples))))
            .expect("silence plays");
        renderer
            .stream_event(&StreamEvent::Eos)
            .expect("the end is taken");
        let elapsed = started.elapsed();

        assert!(
            elapsed >= Duration::from_millis(250),
            "300 ms of sound took {elapsed:?} to play"
        );
        assert!(elapsed < Duration::from_secs(2), "and no hang: {elapsed:?}");
        assert!(!renderer.0.inner.running, "stopped once played");
        assert!(renderer.0.inner.shared.ring.is_empty());
    }

    /// An end with nothing handed over since a pause and a flush — a
    /// picture turned round to play backwards reading back to the start of
    /// its file — returns at once.
    #[test]
    fn an_eos_with_nothing_handed_since_a_flush_returns_at_once() {
        let Some(mut renderer) = try_renderer("empty-eos-test") else {
            return;
        };
        let format = renderer.format();
        renderer
            .consume(MediaBuffer::Audio(Arc::new(frame(
                format,
                format.sample_rate as usize / 10,
            ))))
            .expect("silence plays");
        renderer.control(&ControlMsg::Pause).unwrap();
        renderer.control(&ControlMsg::Flush).unwrap();

        let started = Instant::now();
        renderer
            .stream_event(&StreamEvent::Eos)
            .expect("the end is taken");
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "an end with nothing to play waited {:?}",
            started.elapsed()
        );
    }

    /// As the pipeline's audio master it moves the playback clock with what
    /// is heard, from the first frame's timestamp on.
    #[test]
    fn it_masters_the_playback_clock_with_what_is_heard() {
        let Some(mut renderer) = try_renderer("master-test") else {
            return;
        };
        let clock = Arc::new(PlaybackClock::new(Arc::new(Clock::new())));
        renderer.0.inner.clock_binding = PlaybackClockBinding::Deferred(clock.clone());
        let format = renderer.format();
        let samples = format.sample_rate as usize / 10;
        for index in 0..5 {
            let mut frame = frame(format, samples);
            frame.set_pts(Some((index * samples) as i64));
            renderer
                .consume(MediaBuffer::Audio(Arc::new(frame)))
                .expect("silence plays");
        }

        assert_eq!(clock.master(), PlaybackMaster::Audio);
        let position = clock
            .position_ns()
            .expect("the audio master has a position");
        assert!(
            (0..500_000_000).contains(&position),
            "half a second handed over, {position} ns heard"
        );
        renderer.control(&ControlMsg::Stop).expect("stop");
    }
}
