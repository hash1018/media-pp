use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    thread::JoinHandle,
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender};
use ffmpeg_next as ffmpeg;
use pipewire as pw;
use pw::{properties::properties, spa};
use spa::pod::Pod;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_info, pp_trace, pp_warn};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, element_pp_log},
    elements::AudioFormat,
    error::Result,
    platform::linux::pipewire::{
        PipeWireAudioDevice, PipeWireAudioDeviceKind, PipeWireDeviceError,
    },
    playback_clock::{AudioMasterRegistration, PlaybackClock, PlaybackClockError},
};

/// How long [`PipeWireAudioRenderer::open`] waits for the stream to negotiate a
/// format. Nothing here is interactive, so this only has to outlast normal
/// graph scheduling and exists to turn an unresponsive daemon into an error
/// rather than a permanent hang.
const NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a blocked `consume` waits for queue space before re-checking that
/// the stream is still alive. Bounds how long a torn-down stream can hold up
/// the pipeline thread.
const SEND_GRANULARITY: Duration = Duration::from_millis(100);

/// How many submitted frames may sit between `consume` and the PipeWire thread.
///
/// This is the element's entire playback pacing: `consume` blocks once the
/// queue is full, so a pipeline pushes audio no faster than the device drains
/// it — the same "device-buffer backpressure is the playback clock" contract
/// `WasapiRenderer` documents. Deep enough to ride out scheduling jitter,
/// shallow enough that the block starts promptly rather than buffering
/// seconds of audio ahead of the speakers.
const QUEUE_CAPACITY: usize = 8;

/// How many frames must be queued before the stream is allowed to start.
///
/// PipeWire begins pulling the moment a stream goes active, so activating with
/// an empty queue makes the first few graph cycles underrun: each one splices
/// silence into the output and is audible as a click. Priming first costs a
/// short startup latency and buys a clean start. Half the queue leaves room for
/// `consume` to keep filling while the device drains the rest.
const PRIME_FRAMES: usize = QUEUE_CAPACITY / 2;

/// Errors specific to `PipeWireAudioRenderer`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum PipeWireAudioRendererError {
    #[error("pipewire error: {0}")]
    PipeWire(String),

    #[error(transparent)]
    Device(#[from] PipeWireDeviceError),

    /// [`PipeWireAudioRendererOptions::device`] named a capture node. Playback
    /// needs a [`PipeWireAudioDeviceKind::Sink`]; rejecting here keeps the
    /// mistake a construction-time error instead of a stream that silently
    /// never plays.
    #[error("PipeWireAudioRenderer needs a Sink device, but {0:?} is a Source")]
    NotAPlaybackDevice(String),

    #[error("timed out waiting for the PipeWire audio stream to negotiate a format")]
    NegotiationTimeout,

    #[error("unsupported PipeWire audio sample format {0:?}")]
    UnsupportedFormat(u32),

    #[error("PipeWire negotiated an empty audio format ({rate}Hz, {channels} channel(s))")]
    EmptyFormat { rate: u32, channels: u32 },

    /// The incoming frame's layout does not match what `open` negotiated. This
    /// element performs no hidden conversion — see its own docs.
    #[error(
        "expected {expected:?} audio at {expected_rate}Hz x{expected_channels}, \
         got {actual:?} at {actual_rate}Hz x{actual_channels}"
    )]
    FormatMismatch {
        expected: ffmpeg::format::Sample,
        expected_rate: u32,
        expected_channels: u16,
        actual: ffmpeg::format::Sample,
        actual_rate: u32,
        actual_channels: u16,
    },

    #[error("PipeWireAudioRenderer only accepts MediaBuffer::Audio, got {0}")]
    UnexpectedBuffer(&'static str),

    #[error("audio frames need a PTS when PipeWireAudioRenderer is the playback-clock master")]
    MissingPts,

    #[error("the PipeWire playback stream ended")]
    StreamEnded,

    #[error(transparent)]
    PlaybackClock(#[from] PlaybackClockError),

    #[error("this renderer is already bound to a playback clock")]
    PlaybackClockAlreadyBound,

    #[error("a playback clock must be bound before the first frame is rendered")]
    PlaybackClockBoundAfterStart,
}

/// Construction-time options for [`PipeWireAudioRenderer::open`].
#[derive(Debug, Clone)]
pub struct PipeWireAudioRendererOptions {
    /// Which node to play to — one entry out of
    /// [`PipeWireAudioRenderer::list_devices`]. Must be a
    /// [`PipeWireAudioDeviceKind::Sink`].
    pub device: PipeWireAudioDevice,
}

/// Mirrors `WasapiRenderer`'s own binding states so a dynamically attached
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

    fn ensure_registered(&mut self) -> std::result::Result<(), PlaybackClockError> {
        if let Self::Deferred(playback_clock) = self {
            let registration = playback_clock.register_audio_master()?;
            *self = Self::Registered(registration);
        }
        Ok(())
    }
}

/// State the PipeWire thread publishes for `consume` to read.
///
/// The stream itself is not `Send` and lives entirely on that thread, so the
/// timing values `publish_position` needs are mirrored here as plain atomics
/// rather than queried across the boundary.
struct Playback {
    /// Frames of real media handed to the device so far. Silence written to
    /// cover an underrun is deliberately *not* counted, so a gap in delivery
    /// does not advance media time — the same correction `WasapiRenderer`
    /// makes by rebasing its timeline when the device drains.
    played_frames: AtomicU64,
    /// The stream's most recently reported delay to the device, in frames.
    /// Frames already written but not yet audible.
    delay_frames: AtomicI64,
    /// Set once the PipeWire thread has stopped for any reason, so a blocked
    /// `consume` can fail instead of waiting forever.
    ended: AtomicBool,
}

/// What the PipeWire thread reports back to `open` once, at startup.
enum Startup {
    Ready(AudioFormat),
    Failed(PipeWireAudioRendererError),
}

/// One submitted frame being handed to the device, with a cursor over how much
/// of it has already been copied.
///
/// A cursor rather than draining from the front: a frame typically spans
/// several graph cycles, and `Vec::drain` would memmove the remainder on every
/// one of them.
#[derive(Default)]
struct Pending {
    bytes: Vec<u8>,
    offset: usize,
}

impl Pending {
    fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset >= self.bytes.len()
    }

    /// Copies as much as fits into `out`, returning how many bytes moved.
    fn copy_into(&mut self, out: &mut [u8]) -> usize {
        let take = out.len().min(self.bytes.len() - self.offset);
        out[..take].copy_from_slice(&self.bytes[self.offset..self.offset + take]);
        self.offset += take;
        take
    }
}

/// Sent into the PipeWire thread's own main loop.
enum Command {
    SetActive(bool),
    /// Discard whatever is still queued, for `Stop`/`Flush` semantics.
    Flush,
    Terminate,
}

/// Terminal audio sink backed by a PipeWire playback node.
///
/// The node's negotiated format is returned by [`PipeWireAudioRenderer::open`]
/// so a caller can place an [`crate::elements::AudioResampler`] immediately
/// before this sink. This element intentionally performs no hidden format
/// conversion, and rejects a mismatched frame with
/// [`PipeWireAudioRendererError::FormatMismatch`] rather than guessing.
///
/// Call [`PipeWireAudioRenderer::bind_playback_clock`] while wiring a fixed A/V
/// pipeline to publish this node's actual played-sample position as that
/// pipeline's audio master; a branch attached to a running dynamic
/// [`crate::elements::Tee`] uses
/// [`PipeWireAudioRenderer::bind_playback_clock_deferred`] instead, so it
/// cannot stall video before the first audio frame arrives. Same contract as
/// `WasapiRenderer`, which is the Windows counterpart of this element.
///
/// # Pacing
///
/// Queue backpressure is the playback clock: `consume` blocks once
/// `QUEUE_CAPACITY` frames are outstanding, so upstream can run no faster
/// than the device drains. Put a [`crate::queue::Queue`] immediately before
/// this sink when that blocking must not hold up another branch.
///
/// Unlike the WASAPI path, PipeWire pulls: the daemon calls back on its own
/// realtime thread whenever it needs samples, and that thread must never block.
/// `consume` therefore hands frames to a bounded queue instead of writing to a
/// device buffer directly, and an empty queue is covered with silence rather
/// than stalling the graph.
pub struct PipeWireAudioRenderer {
    name: Arc<str>,
    pp_log: PpLog,
    format: AudioFormat,
    frames: Sender<Vec<u8>>,
    playback: Arc<Playback>,
    clock_binding: PlaybackClockBinding,
    /// Whether the stream has been started yet. It stays inactive until
    /// `PRIME_FRAMES` are queued — see that constant's own docs.
    primed: bool,
    /// Media timestamp the first submitted frame carried, and how far the
    /// submitted range now extends. `None` until the first frame with a `pts`.
    timeline: Option<Timeline>,
    /// Ends the PipeWire thread's main loop. `Option` only so `Drop` can take
    /// it; always `Some` while this element is alive.
    commands: Option<pw::channel::Sender<Command>>,
    /// Joined by `Drop` — this element owns the thread it spawned.
    worker: Option<JoinHandle<()>>,
}

struct Timeline {
    /// `pts` of the first frame, in nanoseconds.
    media_origin_ns: i64,
    /// How far the *submitted* media extends, whether or not it has played.
    submitted_until_ns: i64,
    /// `played_frames` when this timeline began, so a rebase after an underrun
    /// does not count the intervening silence as media time.
    played_origin: u64,
}

impl PipeWireAudioRenderer {
    /// Every currently-published playback node, ready to hand to
    /// [`PipeWireAudioRendererOptions::device`].
    ///
    /// Filtered to [`PipeWireAudioDeviceKind::Sink`]: a capture node cannot be
    /// played to, so listing one here would only invite an error from `open`.
    pub fn list_devices()
    -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireAudioRendererError> {
        let mut devices = crate::platform::linux::pipewire::list_devices()?;
        devices.retain(|device| device.kind == PipeWireAudioDeviceKind::Sink);
        Ok(devices)
    }

    /// Opens `options.device` for playback.
    ///
    /// Returns the element alongside the stream's negotiated [`AudioFormat`] —
    /// what a caller needs to configure the [`crate::elements::AudioResampler`]
    /// feeding it, the same shape `WasapiRenderer::open` returns.
    pub fn open(
        name: impl Into<String>,
        options: PipeWireAudioRendererOptions,
    ) -> std::result::Result<(Self, AudioFormat), PipeWireAudioRendererError> {
        if options.device.kind != PipeWireAudioDeviceKind::Sink {
            return Err(PipeWireAudioRendererError::NotAPlaybackDevice(
                options.device.name.clone(),
            ));
        }
        let name = name.into();
        let pp_log = element_pp_log(ElementType::PipeWireAudioRenderer, &name, None);

        let (frame_tx, frame_rx) = crossbeam_channel::bounded(QUEUE_CAPACITY);
        let (startup_tx, startup_rx) = mpsc::channel::<Startup>();
        let (command_tx, command_rx) = pw::channel::channel::<Command>();
        let playback = Arc::new(Playback {
            played_frames: AtomicU64::new(0),
            delay_frames: AtomicI64::new(0),
            ended: AtomicBool::new(false),
        });

        let device = options.device.clone();
        let worker = std::thread::Builder::new()
            .name(format!("{name}-pipewire-render"))
            .spawn({
                let startup_tx = startup_tx.clone();
                let playback = playback.clone();
                move || {
                    if let Err(error) =
                        run_pipewire(device, frame_rx, playback.clone(), &startup_tx, command_rx)
                    {
                        let _ = startup_tx.send(Startup::Failed(error));
                    }
                    // Whatever ended the loop, a blocked `consume` has to learn
                    // that nothing will drain the queue again.
                    playback.ended.store(true, Ordering::Release);
                }
            })
            .map_err(|e| PipeWireAudioRendererError::PipeWire(e.to_string()))?;

        // From here on the thread is running, so every early return has to tear
        // it down rather than leaking it.
        let startup = match startup_rx.recv_timeout(NEGOTIATION_TIMEOUT) {
            Ok(startup) => startup,
            Err(RecvTimeoutError::Timeout) => {
                let _ = command_tx.send(Command::Terminate);
                let _ = worker.join();
                return Err(PipeWireAudioRendererError::NegotiationTimeout);
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                return Err(PipeWireAudioRendererError::PipeWire(
                    "the PipeWire thread exited before negotiating a format".into(),
                ));
            }
        };
        let format = match startup {
            Startup::Ready(format) => format,
            Startup::Failed(error) => {
                let _ = command_tx.send(Command::Terminate);
                let _ = worker.join();
                return Err(error);
            }
        };

        // Park the stream until `consume` has primed the queue. Negotiation
        // needed an active stream, but pulling with nothing queued makes the
        // first graph cycles underrun and splice audible clicks; the silence
        // between here and the first primed frame is inaudible by definition.
        let _ = command_tx.send(Command::SetActive(false));

        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?} (node {}), {}Hz, {} channel(s), format={:?}",
            options.device.description,
            options.device.id,
            format.sample_rate,
            format.channels,
            format.sample_format
        );

        Ok((
            Self {
                name: name.into(),
                pp_log,
                format,
                frames: frame_tx,
                playback,
                clock_binding: PlaybackClockBinding::Unbound,
                primed: false,
                timeline: None,
                commands: Some(command_tx),
                worker: Some(worker),
            },
            format,
        ))
    }

    /// The format [`Sink::consume`] accepts, unchanged since `open`.
    pub fn format(&self) -> AudioFormat {
        self.format
    }

    /// Makes this node the pipeline's exclusive audio playback master. Call
    /// during the wiring closure, before boxing the renderer into its terminal
    /// branch.
    pub fn bind_playback_clock(
        &mut self,
        playback_clock: Arc<PlaybackClock>,
    ) -> std::result::Result<(), PipeWireAudioRendererError> {
        self.check_bindable()?;
        let master = playback_clock.register_audio_master()?;
        self.clock_binding = PlaybackClockBinding::Registered(master);
        Ok(())
    }

    /// Binds a dynamically attached node without claiming the audio-master slot
    /// until its first audio frame arrives.
    ///
    /// This avoids a priming deadlock when an upstream demuxer can block on a
    /// full video queue before reaching the first packet for the newly attached
    /// audio branch. Unlike [`Self::bind_playback_clock`], an exclusive-master
    /// conflict is therefore returned from that first [`Sink::consume`] call.
    pub fn bind_playback_clock_deferred(
        &mut self,
        playback_clock: Arc<PlaybackClock>,
    ) -> std::result::Result<(), PipeWireAudioRendererError> {
        self.check_bindable()?;
        self.clock_binding = PlaybackClockBinding::Deferred(playback_clock);
        Ok(())
    }

    fn check_bindable(&self) -> std::result::Result<(), PipeWireAudioRendererError> {
        if self.clock_binding.is_bound() {
            return Err(PipeWireAudioRendererError::PlaybackClockAlreadyBound);
        }
        if self.timeline.is_some() {
            return Err(PipeWireAudioRendererError::PlaybackClockBoundAfterStart);
        }
        Ok(())
    }

    /// Nanoseconds `frames` samples occupy at the negotiated rate.
    fn frames_ns(&self, frames: u64) -> i64 {
        ((u128::from(frames) * 1_000_000_000u128) / u128::from(self.format.sample_rate.max(1)))
            .min(i64::MAX as u128) as i64
    }

    /// Publishes how far playback has actually reached, if this element is the
    /// audio master.
    ///
    /// The position is derived from frames the device has consumed minus the
    /// stream's reported delay, so it tracks what the listener has heard rather
    /// than what has merely been handed over.
    fn publish_position(&self, running: bool) -> Result<()> {
        let (Some(master), Some(timeline)) = (self.clock_binding.registration(), &self.timeline)
        else {
            return Ok(());
        };
        let played = self
            .playback
            .played_frames
            .load(Ordering::Acquire)
            .saturating_sub(timeline.played_origin);
        let delay = self.playback.delay_frames.load(Ordering::Acquire).max(0) as u64;
        let audible = played.saturating_sub(delay);
        let position_ns = timeline
            .media_origin_ns
            .saturating_add(self.frames_ns(audible));
        master
            .publish(position_ns, timeline.submitted_until_ns, running)
            .map_err(PipeWireAudioRendererError::from)?;
        Ok(())
    }

    fn audio_pts_ns(&self, frame: &ffmpeg::frame::Audio) -> Result<i64> {
        let pts = frame.pts().ok_or(PipeWireAudioRendererError::MissingPts)?;
        // Frames carry `pts` in their own sample-rate time base, which is the
        // rate this stream negotiated.
        Ok(self.frames_ns(pts.max(0) as u64))
    }

    fn render(&mut self, frame: &ffmpeg::frame::Audio) -> Result<()> {
        let actual_rate = frame.rate();
        let actual_channels = frame.channel_layout().channels() as u16;
        if frame.format() != self.format.sample_format
            || actual_rate != self.format.sample_rate
            || actual_channels != self.format.channels
        {
            return Err(PipeWireAudioRendererError::FormatMismatch {
                expected: self.format.sample_format,
                expected_rate: self.format.sample_rate,
                expected_channels: self.format.channels,
                actual: frame.format(),
                actual_rate,
                actual_channels,
            }
            .into());
        }
        if frame.samples() == 0 {
            return Ok(());
        }

        // Claiming the master slot is deferred to the first real frame so a
        // dynamically attached branch cannot stall video before it primes.
        self.clock_binding
            .ensure_registered()
            .map_err(PipeWireAudioRendererError::from)?;

        let bytes_per_frame = self.format.channels as usize * self.format.sample_format.bytes();
        let tight = frame.samples() * bytes_per_frame;
        let plane = frame.data(0);
        let payload = plane[..tight.min(plane.len())].to_vec();

        let pts_ns = if self.clock_binding.registration().is_some() {
            Some(self.audio_pts_ns(frame)?)
        } else {
            frame.pts().map(|pts| self.frames_ns(pts.max(0) as u64))
        };
        if let Some(pts_ns) = pts_ns {
            let end_ns = pts_ns.saturating_add(self.frames_ns(frame.samples() as u64));
            match &mut self.timeline {
                Some(timeline) => timeline.submitted_until_ns = end_ns,
                None => {
                    self.timeline = Some(Timeline {
                        media_origin_ns: pts_ns,
                        submitted_until_ns: end_ns,
                        played_origin: self.playback.played_frames.load(Ordering::Acquire),
                    })
                }
            }
        }

        // Blocking here is the pacing: once the queue is full, upstream waits
        // for the device to drain rather than running ahead.
        let mut pending = payload;
        loop {
            match self.frames.send_timeout(pending, SEND_GRANULARITY) {
                Ok(()) => break,
                Err(crossbeam_channel::SendTimeoutError::Timeout(returned)) => {
                    if self.playback.ended.load(Ordering::Acquire) {
                        return Err(PipeWireAudioRendererError::StreamEnded.into());
                    }
                    // A full queue is exactly the primed condition, and nothing
                    // will drain it until the stream starts — so check here too,
                    // or a queue that fills before the threshold check deadlocks
                    // against a stream that never starts.
                    self.start_once_primed();
                    self.publish_position(true)?;
                    pending = returned;
                }
                Err(crossbeam_channel::SendTimeoutError::Disconnected(_)) => {
                    return Err(PipeWireAudioRendererError::StreamEnded.into());
                }
            }
        }
        self.start_once_primed();
        self.publish_position(true)?;
        Ok(())
    }

    /// Starts the stream once enough audio is queued to survive the first few
    /// graph cycles — see `PRIME_FRAMES`.
    fn start_once_primed(&mut self) {
        if !self.primed && self.frames.len() >= PRIME_FRAMES {
            self.send_command(Command::SetActive(true));
            self.primed = true;
        }
    }

    /// Waits for everything already queued to reach the device, then reports
    /// the final position to the playback clock.
    fn drain(&mut self) -> Result<()> {
        // A stream shorter than `PRIME_FRAMES` never reached the threshold;
        // start it now or its audio would never play at all.
        if !self.primed {
            self.send_command(Command::SetActive(true));
            self.primed = true;
        }
        while !self.frames.is_empty() && !self.playback.ended.load(Ordering::Acquire) {
            self.publish_position(true)?;
            std::thread::sleep(SEND_GRANULARITY);
        }
        self.publish_position(false)?;
        let final_position = self
            .timeline
            .as_ref()
            .map(|timeline| timeline.submitted_until_ns);
        if let (Some(master), Some(final_position)) =
            (self.clock_binding.registration(), final_position)
        {
            master
                .finish(final_position)
                .map_err(PipeWireAudioRendererError::from)?;
        }
        Ok(())
    }

    fn send_command(&self, command: Command) {
        if let Some(commands) = &self.commands
            && commands.send(command).is_err()
        {
            pp_warn!(
                self,
                "the PipeWire playback thread is no longer accepting commands"
            );
        }
    }
}

impl Drop for PipeWireAudioRenderer {
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            let _ = commands.send(Command::Terminate);
        }
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            pp_warn!(self, "the PipeWire playback thread panicked");
        }
    }
}

impl Element for PipeWireAudioRenderer {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::PipeWireAudioRenderer
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for PipeWireAudioRenderer {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        match buf {
            MediaBuffer::Audio(frame) => self.render(&frame),
            MediaBuffer::Eos => {
                pp_trace!(self, "event=eos phase=received");
                let outcome = self.drain();
                pp_trace!(
                    self,
                    "event=eos phase=drained outcome={}",
                    if outcome.is_ok() { "ok" } else { "error" }
                );
                outcome
            }
            other => Err(PipeWireAudioRendererError::UnexpectedBuffer(other.kind()).into()),
        }
    }

    fn control(&mut self, msg: ControlMsg) -> Result<()> {
        match msg {
            ControlMsg::Pause => {
                self.send_command(Command::SetActive(false));
                self.publish_position(false)?;
            }
            ControlMsg::Resume => {
                self.send_command(Command::SetActive(true));
                self.publish_position(true)?;
            }
            ControlMsg::Stop => {
                // `Stop` means abandon, not natural EOS: discard queued audio
                // instead of draining it.
                self.send_command(Command::Flush);
                self.send_command(Command::SetActive(false));
                self.publish_position(false)?;
                self.timeline = None;
                self.primed = false;
            }
            ControlMsg::Seek(_) => {
                // A seek invalidates everything queued and restarts the media
                // timeline at whatever arrives next.
                self.send_command(Command::Flush);
                self.timeline = None;
            }
        }
        Ok(())
    }
}

/// Maps a negotiated SPA audio format onto the FFmpeg sample format that
/// describes the same interleaved bytes.
fn ffmpeg_sample_format(format: spa::param::audio::AudioFormat) -> Option<ffmpeg::format::Sample> {
    use ffmpeg::format::sample::Type::Packed;
    use spa::param::audio::AudioFormat as Spa;

    Some(match format {
        Spa::F32LE => ffmpeg::format::Sample::F32(Packed),
        Spa::S16LE => ffmpeg::format::Sample::I16(Packed),
        Spa::S32LE => ffmpeg::format::Sample::I32(Packed),
        _ => return None,
    })
}

/// The PipeWire thread body: owns the main loop, context, core, and stream,
/// none of which are `Send`, and so all of which must be created and dropped
/// here rather than handed across from `open`.
fn run_pipewire(
    device: PipeWireAudioDevice,
    frames: Receiver<Vec<u8>>,
    playback: Arc<Playback>,
    startup: &mpsc::Sender<Startup>,
    commands: pw::channel::Receiver<Command>,
) -> std::result::Result<(), PipeWireAudioRendererError> {
    fn pw_err(error: impl std::fmt::Display) -> PipeWireAudioRendererError {
        PipeWireAudioRendererError::PipeWire(error.to_string())
    }

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(pw_err)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(pw_err)?;
    let core = context.connect_rc(None).map_err(pw_err)?;

    let stream = pw::stream::StreamRc::new(
        core.clone(),
        "media-pp-audio-playback",
        properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Production",
        },
    )
    .map_err(pw_err)?;

    // Whatever is left of the frame the previous callback did not fully
    // consume. Only ever touched from this thread's own loop.
    let leftover = Arc::new(Mutex::new(Pending::default()));
    {
        // Distinct bindings: the command handler owns its own clones for the
        // whole life of the loop, while `stream`/`leftover` stay usable below.
        let command_stream = stream.clone();
        let command_leftover = leftover.clone();
        let quit = mainloop.clone();
        let _commands = commands.attach(mainloop.loop_(), move |command| match command {
            Command::SetActive(active) => {
                let _ = command_stream.set_active(active);
            }
            Command::Flush => {
                if let Ok(mut leftover) = command_leftover.lock() {
                    *leftover = Pending::default();
                }
            }
            Command::Terminate => quit.quit(),
        });

        let format = Arc::new(Mutex::new(None::<AudioFormat>));
        let _listener = stream
            .add_local_listener_with_user_data(())
            .param_changed({
                let format = format.clone();
                let startup = startup.clone();
                move |_, (), id, param| {
                    let Some(param) = param else { return };
                    if id != spa::param::ParamType::Format.as_raw() {
                        return;
                    }
                    let Ok((media_type, media_subtype)) =
                        spa::param::format_utils::parse_format(param)
                    else {
                        return;
                    };
                    if media_type != spa::param::format::MediaType::Audio
                        || media_subtype != spa::param::format::MediaSubtype::Raw
                    {
                        return;
                    }
                    let mut info = spa::param::audio::AudioInfoRaw::new();
                    if info.parse(param).is_err() {
                        return;
                    }
                    let Some(sample_format) = ffmpeg_sample_format(info.format()) else {
                        let _ = startup.send(Startup::Failed(
                            PipeWireAudioRendererError::UnsupportedFormat(info.format().as_raw()),
                        ));
                        return;
                    };
                    let (rate, channels) = (info.rate(), info.channels());
                    if rate == 0 || channels == 0 {
                        let _ = startup.send(Startup::Failed(
                            PipeWireAudioRendererError::EmptyFormat { rate, channels },
                        ));
                        return;
                    }
                    let negotiated = AudioFormat::new(sample_format, rate, channels as u16);
                    if let Ok(mut format) = format.lock() {
                        *format = Some(negotiated);
                    }
                    let _ = startup.send(Startup::Ready(negotiated));
                }
            })
            .process({
                let format = format.clone();
                let leftover = leftover.clone();
                let playback = playback.clone();
                let frames = frames.clone();
                move |stream, ()| {
                    let Some(mut buffer) = stream.dequeue_buffer() else {
                        return;
                    };
                    let Some(negotiated) = format.lock().ok().and_then(|f| *f) else {
                        return;
                    };
                    let bytes_per_frame =
                        negotiated.channels as usize * negotiated.sample_format.bytes();
                    if bytes_per_frame == 0 {
                        return;
                    }

                    // Report the delay before filling, so `consume` sees a value
                    // that matches what it is about to be told has played.
                    if let Ok(time) = stream.time() {
                        playback.delay_frames.store(time.delay(), Ordering::Release);
                    }

                    // How many frames the graph wants *this cycle*. Filling the
                    // whole buffer instead would hand over far more than one
                    // quantum, draining the queue in a burst and leaving the
                    // following cycles with nothing but silence — audibly
                    // choppy playback rather than a continuous stream.
                    let requested_frames = buffer.requested() as usize;

                    let written_bytes = {
                        let datas = buffer.datas_mut();
                        let Some(data) = datas.first_mut() else {
                            return;
                        };
                        let capacity_frames =
                            data.data().map(|d| d.len()).unwrap_or(0) / bytes_per_frame;
                        // `requested` is 0 when the graph gives no hint; fall
                        // back to the buffer's own capacity in that case only.
                        let want_frames = if requested_frames > 0 {
                            requested_frames.min(capacity_frames)
                        } else {
                            capacity_frames
                        };
                        let want = want_frames * bytes_per_frame;
                        if want == 0 {
                            return;
                        }

                        let mut written = 0usize;
                        {
                            let Ok(mut pending) = leftover.lock() else {
                                return;
                            };
                            let Some(out) = data.data() else { return };
                            while written < want {
                                if pending.is_empty() {
                                    match frames.try_recv() {
                                        Ok(next) => *pending = Pending::new(next),
                                        // Underrun: cover the rest with silence
                                        // rather than blocking the graph.
                                        Err(_) => break,
                                    }
                                }
                                written += pending.copy_into(&mut out[written..want]);
                            }
                            if written < want {
                                out[written..want].fill(0);
                            }
                        }
                        // Only real media advances the played position; the
                        // silence above is a gap, not media time.
                        playback
                            .played_frames
                            .fetch_add((written / bytes_per_frame) as u64, Ordering::AcqRel);
                        want
                    };

                    let datas = buffer.datas_mut();
                    let data = &mut datas[0];
                    let chunk = data.chunk_mut();
                    *chunk.offset_mut() = 0;
                    *chunk.stride_mut() = bytes_per_frame as i32;
                    *chunk.size_mut() = written_bytes as u32;
                }
            })
            .register()
            .map_err(pw_err)?;

        // Leave rate and channels unset so the graph's own native values are
        // accepted, rather than forcing a conversion this element would then
        // have to describe.
        let mut audio_info = spa::param::audio::AudioInfoRaw::new();
        audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
        let obj = spa::pod::Object {
            type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
            id: spa::param::ParamType::EnumFormat.as_raw(),
            properties: audio_info.into(),
        };
        let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &spa::pod::Value::Object(obj),
        )
        .map_err(pw_err)?
        .0
        .into_inner();
        let mut params = [Pod::from_bytes(&values).ok_or_else(|| {
            PipeWireAudioRendererError::PipeWire("failed to build format pod".into())
        })?];

        stream
            .connect(
                spa::utils::Direction::Output,
                Some(device.id),
                // Not `INACTIVE`: that would also defer format negotiation,
                // which `open` has to wait for. The stream is parked right
                // after negotiation instead — see `open`.
                //
                // No `RT_PROCESS`: this callback receives from a channel and
                // takes a lock, so it is not realtime-safe and must not claim
                // to be. PipeWire then drives it from the main loop instead.
                pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
                &mut params,
            )
            .map_err(pw_err)?;

        mainloop.run();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_interleaved_formats_are_accepted() {
        use spa::param::audio::AudioFormat as Spa;
        assert_eq!(
            ffmpeg_sample_format(Spa::F32LE),
            Some(ffmpeg::format::Sample::F32(
                ffmpeg::format::sample::Type::Packed
            ))
        );
        assert!(ffmpeg_sample_format(Spa::F32P).is_none());
    }

    #[test]
    fn a_capture_node_is_rejected_before_any_stream_is_opened() {
        let device = PipeWireAudioDevice {
            id: 7,
            name: "some-mic".into(),
            description: "Some Mic".into(),
            kind: PipeWireAudioDeviceKind::Source,
            is_default: true,
        };
        let Err(error) =
            PipeWireAudioRenderer::open("out", PipeWireAudioRendererOptions { device })
        else {
            panic!("a Source device cannot be played to");
        };
        assert!(
            matches!(error, PipeWireAudioRendererError::NotAPlaybackDevice(name) if name == "some-mic"),
            "the mistake must surface as a typed construction error, \
             not as a stream that silently never plays"
        );
    }

    #[test]
    fn seek_is_rejected_as_a_typed_error() {
        let error: crate::Error = PipeWireAudioRendererError::StreamEnded.into();
        assert!(matches!(
            error,
            crate::Error::PipeWireAudioRendererError(PipeWireAudioRendererError::StreamEnded)
        ));
    }
}
