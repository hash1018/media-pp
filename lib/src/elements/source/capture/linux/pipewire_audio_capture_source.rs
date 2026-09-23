use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError, SyncSender},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use ffmpeg_next as ffmpeg;
use pipewire as pw;
use pw::{properties::properties, spa};
use spa::pod::Pod;
use thiserror::Error as ThisError;

use crate::pp_log::{PpLog, pp_error, pp_info, pp_warn};

use crate::{
    buffer::MediaBuffer,
    bus::{Bus, BusEvent},
    contract::{MediaKind, MemoryDomain, OutputContract, PortContract},
    control::{self, ControlMsg, ControlOutcome, ControlReceiver, RequestKind},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::AudioFormat,
    error::Result,
    pad::SrcPad,
    platform::linux::pipewire::{
        PipeWireAudioApplication, PipeWireAudioDevice, PipeWireAudioDeviceKind, PipeWireDeviceError,
    },
};

/// How long [`PipeWireAudioCaptureSource::open`] waits for the stream to
/// negotiate a format. Nothing here is interactive — unlike the screen-capture
/// path there is no dialog to wait on — so this only has to outlast normal
/// graph scheduling, and exists to turn an unresponsive daemon into an error
/// rather than a permanent hang.
const NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(5);

/// Bounds `Stop` latency while `run` waits for the next captured packet.
const RECV_GRANULARITY: Duration = Duration::from_millis(100);

/// How many captured packets may be queued between the PipeWire thread and
/// [`SourceElement::run`] before the oldest are dropped.
///
/// PipeWire delivers on its own realtime thread and must never be blocked by a
/// slow downstream, so the hand-off is a bounded queue rather than a lock the
/// producer could wait on. At the ~21ms packets this element sees in practice,
/// this is roughly a second of slack — enough to ride out a scheduling hiccup,
/// short enough that a genuinely stalled consumer surfaces as reported drops
/// instead of unbounded memory growth.
const PACKET_QUEUE_CAPACITY: usize = 48;

/// Errors specific to `PipeWireAudioCaptureSource`. Converts into the
/// crate-wide `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum PipeWireAudioCaptureSourceError {
    #[error("pipewire error: {0}")]
    PipeWire(String),

    /// The stream connected but never produced a `Format` param within
    /// `NEGOTIATION_TIMEOUT`.
    #[error("timed out waiting for the PipeWire audio stream to negotiate a format")]
    NegotiationTimeout,

    #[error(transparent)]
    Device(#[from] PipeWireDeviceError),

    /// The daemon negotiated a sample format this element cannot describe as
    /// an FFmpeg sample format.
    #[error("unsupported PipeWire audio sample format {0:?}")]
    UnsupportedFormat(u32),

    /// The negotiated format carried a zero rate or channel count, leaving
    /// nothing coherent to describe downstream.
    #[error("PipeWire negotiated an empty audio format ({rate}Hz, {channels} channel(s))")]
    EmptyFormat { rate: u32, channels: u32 },

    #[error("PipeWireAudioCaptureSource doesn't support seeking a live capture")]
    SeekUnsupported,

    #[error("the PipeWire audio capture stream ended")]
    StreamEnded,
}

/// Construction-time options for [`PipeWireAudioCaptureSource::open`].
#[derive(Debug, Clone)]
pub struct PipeWireAudioCaptureOptions {
    /// Which node to capture from — one entry out of
    /// [`PipeWireAudioCaptureSource::list_devices`]. Its own
    /// [`PipeWireAudioDeviceKind`] is what decides whether this captures
    /// monitor ports or the node directly; there is no separate mode to keep
    /// consistent with it.
    pub device: PipeWireAudioDevice,
}

/// What one application's capture asks the graph for.
///
/// Fixed, unlike a device capture, which takes the graph's own values: an
/// application's stream is linked to port by port, and ports exist only once
/// a format says how many there are. The same reasoning as
/// `WasapiCaptureSource`'s process loopback, which asks Windows for one
/// format and lets it convert.
const APPLICATION_RATE: u32 = 48_000;
const APPLICATION_CHANNELS: u32 = 2;

/// What a capture is pointed at.
#[derive(Debug, Clone)]
enum Target {
    /// A device node: a sink captured through its monitor, or a source
    /// captured directly. The session manager makes the links.
    Device(PipeWireAudioDevice),
    /// One application's playback stream, by node id. Nothing links a
    /// capture to one of these on its own — see `link_application`.
    Application(u32),
}

/// One captured packet handed from the PipeWire thread to `run`.
struct Packet {
    /// Interleaved samples in the negotiated format, exactly
    /// `frames * channels * bytes_per_sample` long.
    bytes: Vec<u8>,
    frames: usize,
    /// How many frames the device had captured before this packet, counting
    /// the ones this element went on to discard.
    ///
    /// Stamped where the capture happens, because that is the only place
    /// that knows where a packet sits in the captured stream. Counting on the
    /// consuming side instead put the gap left by a discarded packet in front
    /// of whatever came out of the queue next — audio captured *before* the
    /// drop — so the timeline said the loss happened up to a full queue
    /// earlier than it did.
    position: u64,
}

/// What the PipeWire thread reports back to `open` once, at startup.
enum Startup {
    Ready(AudioFormat),
    Failed(PipeWireAudioCaptureSourceError),
}

// Result of a state mutation performed on the PipeWire thread.
type CommandResult = std::result::Result<(), String>;
type CommandReply = SyncSender<CommandResult>;

/// Sent into the PipeWire thread's own main loop.
enum Command {
    /// Starts or stops the capture stream itself. Stopping is what makes a
    /// paused source stop asking the daemon for audio nobody will read — see
    /// [`PipeWireAudioCaptureSource::handle_control`].
    SetActive {
        active: bool,
        reply: CommandReply,
    },
    Terminate,
}

fn queue_set_active(
    commands: &pw::channel::Sender<Command>,
    active: bool,
) -> std::result::Result<mpsc::Receiver<CommandResult>, PipeWireAudioCaptureSourceError> {
    // Capacity one prevents a late PipeWire reply from blocking its own loop
    // after a timed-out caller has gone away.
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    commands
        .send(Command::SetActive {
            active,
            reply: reply_tx,
        })
        .map_err(|_| PipeWireAudioCaptureSourceError::StreamEnded)?;
    Ok(reply_rx)
}

fn wait_set_active(
    reply: mpsc::Receiver<CommandResult>,
    active: bool,
) -> std::result::Result<(), PipeWireAudioCaptureSourceError> {
    match reply.recv_timeout(NEGOTIATION_TIMEOUT) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(PipeWireAudioCaptureSourceError::PipeWire(error)),
        Err(RecvTimeoutError::Timeout) => Err(PipeWireAudioCaptureSourceError::PipeWire(format!(
            "timed out waiting for PipeWire to {} the capture stream",
            if active { "activate" } else { "deactivate" }
        ))),
        Err(RecvTimeoutError::Disconnected) => Err(PipeWireAudioCaptureSourceError::StreamEnded),
    }
}

/// Captures system audio or microphone input through PipeWire, emitting
/// `MediaBuffer::Audio` frames in the graph's own native rate, channel count,
/// and sample format — no resampling.
///
/// # What this captures
///
/// Whatever [`PipeWireAudioCaptureOptions::device`] names, chosen
/// programmatically. This is the substantive difference from
/// [`crate::elements::PipeWireScreenCaptureSource`], which shares the same
/// backend but cannot select anything: audio capture goes straight to a
/// PipeWire node with no xdg-desktop-portal involved, so there is no dialog,
/// no user interaction, and no restore token. `list_devices` plus a `device`
/// field is the whole selection story, the same shape
/// `WasapiCaptureSource` uses.
///
/// A [`PipeWireAudioDeviceKind::Sink`] device is captured through its monitor
/// ports (what the system is playing); a [`PipeWireAudioDeviceKind::Source`]
/// is captured directly (a microphone).
///
/// # Timeline
///
/// PipeWire keeps a capture stream running at the graph's rate even when
/// nothing is playing, delivering real silence rather than stopping. That is
/// why this element has no silence-synthesis path: unlike
/// `WasapiCaptureSource`, which must manufacture silence because WASAPI
/// delivers literally nothing while a render endpoint is idle, the timeline
/// here is already continuous and `pts` stays in lockstep by simply counting
/// the samples that arrive.
///
/// # Threading
///
/// PipeWire delivers on its own realtime thread, which must never block on a
/// slow downstream. Captured packets therefore cross into
/// [`SourceElement::run`] through a bounded queue; when a stalled consumer
/// fills it, the packet that would not fit is dropped and reported as
/// [`crate::bus::BusEvent::Dropped`] rather than stalling the daemon or
/// growing without bound. The gap that leaves stays where it happened: every
/// packet carries the position the device captured it at, so the frames after
/// a drop are stamped past it.
///
/// # Format
///
/// Whatever the graph negotiates, reported by `open` and unchanged on the way
/// downstream — the same division of labor `WasapiCaptureSource` documents. If
/// something downstream needs a fixed rate or layout, use
/// [`crate::elements::AudioResampler`] rather than hiding conversion here.
pub struct PipeWireAudioCaptureSource {
    name: Arc<str>,
    pp_log: PpLog,
    pad: SrcPad,
    format: AudioFormat,
    packets: Receiver<Packet>,
    /// Frames the PipeWire thread had to discard because `packets` was full.
    /// Reporting is independent of timestamps: each packet already carries
    /// its capture position, including gaps left by these discarded frames.
    dropped_frames: Arc<AtomicU64>,
    /// Controls the PipeWire stream and ends its main loop. `Option` only so
    /// `Drop` can take it; always `Some` while this element is alive.
    commands: Option<pw::channel::Sender<Command>>,
    /// Joined by `Drop` — this element owns the thread it spawned.
    worker: Option<JoinHandle<()>>,
}

impl PipeWireAudioCaptureSource {
    /// Enumerates every currently-published audio node — both
    /// [`PipeWireAudioDeviceKind::Sink`] (capturable as system audio) and
    /// [`PipeWireAudioDeviceKind::Source`] (microphones) — as a list a caller
    /// can show in a picker, search, or filter, then hand straight to
    /// [`PipeWireAudioCaptureOptions::device`].
    ///
    /// Needs no portal and shows no dialog. See
    /// [`crate::elements::PipeWireAudioRenderer::list_devices`] for the same
    /// list filtered to playback nodes.
    pub fn list_devices()
    -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireAudioCaptureSourceError> {
        Ok(crate::platform::linux::pipewire::list_devices()?)
    }

    /// Opens `options.device` and starts capturing.
    ///
    /// Returns the element alongside the stream's negotiated [`AudioFormat`] —
    /// what a caller needs to build a matching downstream encoder or muxer, the
    /// same shape `WasapiCaptureSource::open` returns. It carries no
    /// `time_base` (unlike [`crate::elements::VideoFormat`]) because every
    /// audio element here derives it as `1 / sample_rate`; see
    /// [`PipeWireAudioCaptureSource::time_base`].
    pub fn open(
        name: impl Into<String>,
        options: PipeWireAudioCaptureOptions,
    ) -> std::result::Result<(Self, AudioFormat), PipeWireAudioCaptureSourceError> {
        let described = format!(
            "device={:?} ({:?}, node {})",
            options.device.description, options.device.kind, options.device.id
        );
        Self::open_target(name.into(), Target::Device(options.device), described)
    }

    /// Every application the graph currently publishes a playback stream for
    /// — what a per-application picker shows, and where the node id
    /// [`Self::open_application`] takes comes from.
    ///
    /// Not a process list: a program that has never played anything has no
    /// stream, the same distinction `WasapiCaptureSource::list_processes`
    /// draws between a process and an audio session.
    pub fn list_applications()
    -> std::result::Result<Vec<PipeWireAudioApplication>, PipeWireAudioCaptureSourceError> {
        Ok(crate::platform::linux::pipewire::list_applications()?)
    }

    /// Opens a capture of what one application is playing, rather than of a
    /// device — the PipeWire counterpart of
    /// `WasapiCaptureSource::open_process`.
    ///
    /// `node` is one run of one program, from [`Self::list_applications`]:
    /// the stream is linked to that node's output ports directly, because
    /// nothing else will. A session manager links a capture stream to a
    /// microphone or a monitor, which is what it means by "an input", and
    /// asking one for an application's own stream gets the default
    /// microphone instead — measured, and the reason this connects with no
    /// autoconnect and makes the links itself.
    ///
    /// What it captures is that node and nothing else: a program playing
    /// through several streams publishes several nodes, and a program that
    /// stops playing withdraws its own. When the node goes, the links go
    /// with it and this captures silence rather than ending — the caller
    /// decides what a program that has gone quiet means, by asking
    /// [`Self::list_applications`] again.
    ///
    /// The format is fixed at 48 kHz stereo `f32`: ports have to exist
    /// before anything can be linked to them, and a stream with no format
    /// yet has none.
    pub fn open_application(
        name: impl Into<String>,
        node: u32,
    ) -> std::result::Result<(Self, AudioFormat), PipeWireAudioCaptureSourceError> {
        Self::open_target(
            name.into(),
            Target::Application(node),
            format!("application=node {node}"),
        )
    }

    fn open_target(
        name: String,
        target: Target,
        described: String,
    ) -> std::result::Result<(Self, AudioFormat), PipeWireAudioCaptureSourceError> {
        let pp_log = element_pp_log(ElementType::PipeWireAudioCaptureSource, &name, None);

        let (packet_tx, packet_rx) = crossbeam_channel::bounded(PACKET_QUEUE_CAPACITY);
        let dropped_frames = Arc::new(AtomicU64::new(0));
        let (startup_tx, startup_rx) = mpsc::channel::<Startup>();
        let (command_tx, command_rx) = pw::channel::channel::<Command>();

        let opened = target.clone();
        let worker = std::thread::Builder::new()
            .name(format!("{name}-pipewire-audio"))
            .spawn({
                let startup_tx = startup_tx.clone();
                let dropped_frames = dropped_frames.clone();
                move || {
                    if let Err(error) =
                        run_pipewire(opened, packet_tx, dropped_frames, &startup_tx, command_rx)
                    {
                        let _ = startup_tx.send(Startup::Failed(error));
                    }
                }
            })
            .map_err(|e| PipeWireAudioCaptureSourceError::PipeWire(e.to_string()))?;

        // From here on the thread is running, so every early return has to
        // tear it down rather than leaking it.
        let startup = match startup_rx.recv_timeout(NEGOTIATION_TIMEOUT) {
            Ok(startup) => startup,
            Err(RecvTimeoutError::Timeout) => {
                let _ = command_tx.send(Command::Terminate);
                let _ = worker.join();
                return Err(PipeWireAudioCaptureSourceError::NegotiationTimeout);
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = worker.join();
                return Err(PipeWireAudioCaptureSourceError::PipeWire(
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

        pp_info!(
            pp_log: &pp_log,
            "opened: {described}, {}Hz, {} channel(s), format={:?}",
            format.sample_rate,
            format.channels,
            format.sample_format
        );

        Ok((
            Self {
                pad: SrcPad::with_contract(
                    format!("{name}_src"),
                    OutputContract::Fixed(PortContract::frame(
                        MediaKind::AudioFrame,
                        MemoryDomain::System,
                    )),
                ),
                name: name.into(),
                pp_log,
                format,
                packets: packet_rx,
                dropped_frames,
                commands: Some(command_tx),
                worker: Some(worker),
            },
            format,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.format.sample_rate as i32)
    }

    /// Applies one stream activation change on the PipeWire thread and waits
    /// for its result. A control request is synchronous, so merely enqueueing
    /// this command would let Pause/Resume acknowledge a state the stream had
    /// not reached yet.
    fn set_active(&self, active: bool) -> Result<()> {
        let commands = self
            .commands
            .as_ref()
            .ok_or(PipeWireAudioCaptureSourceError::StreamEnded)?;
        let reply = queue_set_active(commands, active)?;
        wait_set_active(reply, active)?;
        Ok(())
    }

    /// Like [`crate::control::drain_control`], but drives the control receiver
    /// directly (the same reason `WasapiCaptureSource` does
    /// — see `drain_control`'s own docs) so it can bracket the blocking
    /// `Pause` wait with the stream's own `set_active`.
    ///
    /// Without this the daemon keeps capturing for the whole pause with
    /// nothing draining the queue: it fills, every later packet is discarded
    /// as an overload, and `Resume` emits the queue's stale pre-pause audio as
    /// a burst before any live audio. Deactivating the stream is the PipeWire
    /// equivalent of the `IAudioClient::Stop` that source performs, and
    /// discarding what is already queued before reactivating is its `Reset`.
    fn handle_control(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<ControlOutcome> {
        let mut paused_for = Duration::ZERO;
        while let Some((request, ack)) = control.try_recv() {
            let RequestKind::Control(msg) = request else {
                control::apply_finish(self, bus, &ack);
                return Ok(ControlOutcome {
                    stopped: true,
                    paused_for,
                });
            };
            if msg != ControlMsg::Pause {
                if control::apply_one(self, bus, &msg, &ack)? {
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                }
                continue;
            }

            // Include the stream transition and the downstream cascade in the
            // frozen interval: this source produces no media during either.
            let pause_start = Instant::now();
            self.set_active(false)?;
            control::apply_one(self, bus, &msg, &ack)?;

            loop {
                let Some((paused_msg, paused_ack)) = control.recv() else {
                    paused_for += pause_start.elapsed();
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                };
                let RequestKind::Control(paused_msg) = paused_msg else {
                    control::apply_finish(self, bus, &paused_ack);
                    paused_for += pause_start.elapsed();
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                };

                if paused_msg == ControlMsg::Resume {
                    // Keep the stream stopped until every downstream element
                    // has resumed, then discard whatever the pause boundary
                    // left queued so `Resume` starts from live audio, and only
                    // then acknowledge.
                    control::apply_one_unacked(self, bus, &paused_msg)?;
                    discard_captured(&self.packets);
                    self.set_active(true)?;
                    let _ = paused_ack.send(());
                    paused_for += pause_start.elapsed();
                    break;
                }

                if control::apply_one(self, bus, &paused_msg, &paused_ack)? {
                    paused_for += pause_start.elapsed();
                    return Ok(ControlOutcome {
                        stopped: true,
                        paused_for,
                    });
                }
                // A redundant Pause (or another one-shot control) was
                // forwarded and acknowledged; remain frozen until Resume.
            }
        }
        Ok(ControlOutcome {
            stopped: false,
            paused_for,
        })
    }
}

/// The samples inside one mapped buffer.
///
/// SPA describes the valid region of a buffer with its chunk's own `offset`
/// and `size`, not with the bounds of the mapping: a node is free to hand
/// over a mapping whose data starts partway in. Reading from the start of the
/// mapping instead would return whatever precedes the samples on such a node
/// -- silence or noise, never an error.
///
/// Both bounds are clamped to the mapping, and the result is truncated to
/// whole frames so no packet ever ends mid-frame. `None` means the chunk
/// describes nothing this element can read.
fn chunk_samples(bytes: &[u8], offset: usize, size: usize, frame_bytes: usize) -> Option<&[u8]> {
    let start = offset.min(bytes.len());
    let end = offset.saturating_add(size).min(bytes.len());
    let usable = end.saturating_sub(start) / frame_bytes * frame_bytes;
    (usable > 0).then(|| &bytes[start..start + usable])
}

/// Drops everything captured before a pause, so `Resume` emits live audio
/// rather than the queue's stale contents.
fn discard_captured(packets: &Receiver<Packet>) {
    while packets.try_recv().is_ok() {}
}

/// Wraps one captured packet as an `ffmpeg` frame, stamped with where the
/// device captured it.
///
/// A free function taking the format rather than a method: the `pts` is the
/// packet's own `position`, so nothing about this depends on element state,
/// and the property that matters — a discarded packet leaves its gap where it
/// happened — is then testable without a capture device.
fn build_frame(format: &AudioFormat, packet: &Packet) -> ffmpeg::frame::Audio {
    // Derived rather than stored: `ffmpeg::ChannelLayout` is not `Send`, and
    // keeping it as a field would force an `unsafe impl Send` on the whole
    // element to satisfy `Element: Send` — for a value `AudioFormat` can
    // already reproduce exactly.
    let mut frame =
        ffmpeg::frame::Audio::new(format.sample_format, packet.frames, format.channel_layout());
    frame.set_rate(format.sample_rate);
    // `frame.data_mut(0)`'s length is FFmpeg's own padded linesize, not
    // necessarily the tight sample bytes — copy only what the packet actually
    // holds, the same bound `WasapiCaptureSource::build_frame` documents for
    // the mirror-image reason.
    let tight_bytes = packet.bytes.len().min(frame.data_mut(0).len());
    frame.data_mut(0)[..tight_bytes].copy_from_slice(&packet.bytes[..tight_bytes]);
    frame.set_pts(Some(packet.position as i64));
    // In samples, as `PipeWireAudioCaptureSource::time_base` says.
    crate::buffer::set_time_base(
        &mut frame,
        ffmpeg::Rational::new(1, format.sample_rate as i32),
    );
    frame
}

impl Drop for PipeWireAudioCaptureSource {
    fn drop(&mut self) {
        // Dropping the sender alone would not wake a blocked main loop, so
        // signal first, then join the thread this element owns.
        if let Some(commands) = self.commands.take() {
            let _ = commands.send(Command::Terminate);
        }
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            pp_warn!(self, "the PipeWire audio capture thread panicked");
        }
    }
}

impl Element for PipeWireAudioCaptureSource {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::PipeWireAudioCaptureSource
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for PipeWireAudioCaptureSource {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl SourceElement for PipeWireAudioCaptureSource {
    fn is_live(&self) -> bool {
        true
    }

    fn is_seekable(&self) -> bool {
        false
    }

    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        loop {
            let outcome = self.handle_control(control, bus)?;
            if outcome.stopped {
                pp_info!(self, "stopped");
                return Ok(());
            }

            // Blocking with a short timeout rather than on the channel alone
            // keeps `Stop`/`Pause` responsive between packets, the same
            // reasoning behind every other source's poll granularity.
            let packet = match self.packets.recv_timeout(RECV_GRANULARITY) {
                Ok(packet) => packet,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // The PipeWire thread ended on its own — the daemon went
                    // away or the node disappeared. This source cannot
                    // continue, unlike a single failed push.
                    pp_error!(self, "the PipeWire audio stream ended");
                    return Err(PipeWireAudioCaptureSourceError::PipeWire(
                        "the PipeWire audio stream ended".into(),
                    )
                    .into());
                }
            };

            // The gap a drop leaves is already in the timeline: every packet
            // carries where it was captured, so the frames stamped after a
            // drop skip past it on their own. Reported here, not accounted
            // for here.
            let dropped = self.dropped_frames.swap(0, Ordering::Relaxed);
            if dropped > 0 {
                pp_warn!(
                    self,
                    "dropped {dropped} captured frame(s): downstream is not keeping up"
                );
                bus.post(
                    &self.pp_log,
                    BusEvent::Dropped {
                        element_type: ElementType::PipeWireAudioCaptureSource,
                        name: self.name.clone(),
                    },
                );
            }

            let frame = build_frame(&self.format, &packet);
            if let Err(error) = self.pad.push(MediaBuffer::Audio(Arc::new(frame))) {
                bus.post(
                    &self.pp_log,
                    BusEvent::Error {
                        element_type: ElementType::PipeWireAudioCaptureSource,
                        name: self.name.clone(),
                        error,
                    },
                );
            }
        }
    }

    fn seek(&mut self, _target: Duration) -> Result<Duration> {
        Err(PipeWireAudioCaptureSourceError::SeekUnsupported.into())
    }
}

/// Maps a negotiated SPA audio format onto the FFmpeg sample format that
/// describes the same interleaved bytes.
///
/// Only the interleaved variants PipeWire actually negotiates for capture are
/// accepted; a planar or otherwise unexpected format is rejected rather than
/// reinterpreted, since a wrong guess here would silently produce noise.
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
    target: Target,
    packets: Sender<Packet>,
    dropped_frames: Arc<AtomicU64>,
    startup: &mpsc::Sender<Startup>,
    commands: pw::channel::Receiver<Command>,
) -> std::result::Result<(), PipeWireAudioCaptureSourceError> {
    fn pw_err(error: impl std::fmt::Display) -> PipeWireAudioCaptureSourceError {
        PipeWireAudioCaptureSourceError::PipeWire(error.to_string())
    }

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(pw_err)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(pw_err)?;
    let core = context.connect_rc(None).map_err(pw_err)?;

    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Production",
    };
    if matches!(&target, Target::Device(device) if device.kind == PipeWireAudioDeviceKind::Sink) {
        // Turns "connect to this sink" into "capture what this sink is
        // playing" by binding to its monitor ports instead of its inputs.
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }

    // `StreamRc` rather than a borrowed stream: the command handler below
    // needs a handle of its own to start and stop the capture with.
    let stream =
        pw::stream::StreamRc::new(core.clone(), "media-pp-audio-capture", props).map_err(pw_err)?;

    // Attach to the outer loop, but let the callback own its own clones: the
    // `AttachedReceiver` borrows the `Loop` for as long as it lives, so the
    // handles it is attached to have to outlive it.
    let quit_loop = mainloop.clone();
    let command_stream = stream.clone();
    let _commands = commands.attach(mainloop.loop_(), move |command| match command {
        // Both directions are the daemon's own stream state: an inactive
        // capture stream stops being scheduled at all, rather than filling
        // buffers this element would immediately discard.
        Command::SetActive { active, reply } => {
            let result = command_stream
                .set_active(active)
                .map_err(|error| error.to_string());
            let _ = reply.send(result);
        }
        Command::Terminate => quit_loop.quit(),
    });

    // Negotiated format, shared with the process callback. Both callbacks run
    // on this thread's own loop, so a `Cell` is enough — no lock needed.
    let format = Rc::new(Cell::new(None::<AudioFormat>));

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
                let Ok((media_type, media_subtype)) = spa::param::format_utils::parse_format(param)
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
                        PipeWireAudioCaptureSourceError::UnsupportedFormat(info.format().as_raw()),
                    ));
                    return;
                };
                let (rate, channels) = (info.rate(), info.channels());
                if rate == 0 || channels == 0 {
                    let _ = startup.send(Startup::Failed(
                        PipeWireAudioCaptureSourceError::EmptyFormat { rate, channels },
                    ));
                    return;
                }
                let negotiated = AudioFormat::new(sample_format, rate, channels as u16);
                format.set(Some(negotiated));
                let _ = startup.send(Startup::Ready(negotiated));
            }
        })
        .process({
            let format = format.clone();
            // Frames the device has captured so far, dropped ones included:
            // what stamps each packet's place in the captured stream.
            let mut captured_frames = 0u64;
            move |stream, ()| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    return;
                }
                let Some(format) = format.get() else {
                    return; // data before the format param — nothing to describe it with
                };
                let data = &mut datas[0];
                let (offset, size) = (data.chunk().offset() as usize, data.chunk().size() as usize);
                if size == 0 {
                    return; // a tick with no new content
                }
                let frame_bytes = format.channels as usize * format.sample_format.bytes();
                if frame_bytes == 0 {
                    return;
                }
                let Some(bytes) = data.data() else { return };
                let Some(samples) = chunk_samples(bytes, offset, size, frame_bytes) else {
                    return;
                };
                let frames = samples.len() / frame_bytes;
                let packet = Packet {
                    bytes: samples.to_vec(),
                    frames,
                    position: captured_frames,
                };
                captured_frames += frames as u64;
                // Never block PipeWire's realtime thread on a slow consumer.
                // A discarded packet is counted, not forgotten: `run` folds
                // the frame count into `pts` so the timeline keeps matching
                // wall-clock time.
                if let Err(TrySendError::Full(packet)) = packets.try_send(packet) {
                    dropped_frames.fetch_add(packet.frames as u64, Ordering::Relaxed);
                }
            }
        })
        .register()
        .map_err(pw_err)?;

    // Leave rate and channels unset so the graph's own native values are
    // accepted, rather than forcing a conversion this element would then have
    // to describe. Only the sample format is constrained, to the interleaved
    // layouts `ffmpeg_sample_format` can describe.
    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    if matches!(target, Target::Application(_)) {
        // An application's stream is linked to port by port, and a stream
        // whose format is still open has no ports to link — so this one says
        // what it wants and lets the graph convert.
        audio_info.set_rate(APPLICATION_RATE);
        audio_info.set_channels(APPLICATION_CHANNELS);
        let mut position = [0; spa::param::audio::MAX_CHANNELS];
        position[0] = spa::sys::SPA_AUDIO_CHANNEL_FL;
        position[1] = spa::sys::SPA_AUDIO_CHANNEL_FR;
        audio_info.set_position(position);
    }
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
        PipeWireAudioCaptureSourceError::PipeWire("failed to build format pod".into())
    })?];

    let (connect_to, flags) = match &target {
        // A device is the session manager's business: it knows what "capture
        // this" means for a microphone and for a monitor.
        Target::Device(device) => (
            Some(device.id),
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        ),
        // An application's is not. Asking for one with `AUTOCONNECT` gets the
        // default microphone instead — measured on WirePlumber, which reads
        // "an input" as a device — so this connects to nothing and links
        // itself below.
        Target::Application(_) => (None, pw::stream::StreamFlags::MAP_BUFFERS),
    };
    stream
        .connect(spa::utils::Direction::Input, connect_to, flags, &mut params)
        .map_err(pw_err)?;

    // Kept for the life of the loop: dropping a link proxy takes the link
    // with it, and dropping the listener stops the ports that arrive later
    // from being linked at all.
    let _links = match target {
        Target::Application(node) => Some(link_application(&core, &stream, node)?),
        Target::Device(_) => None,
    };

    mainloop.run();
    Ok(())
}

/// One port of a node, as the registry describes it.
struct Port {
    id: u32,
    /// `audio.channel` — `FL`, `FR`, `MONO`. A port that names its channel
    /// can be paired with the one carrying the same sound rather than with
    /// whichever arrived in the same order.
    channel: Option<String>,
}

/// One link this capture made, with the two ports it joins.
///
/// Kept together rather than in a list of links beside a list of pairs: a
/// port that goes has to take its own link with it, and two vectors that
/// have to stay in step are two vectors that can stop being in step.
struct Made {
    output: u32,
    input: u32,
    /// Held for what dropping it does, which is why nothing reads it: a link
    /// proxy let go of takes its link with it, since it was made with
    /// `OBJECT_LINGER` false.
    #[allow(dead_code)]
    link: pw::link::Link,
}

/// Everything an application capture needs kept alive: the registry it
/// watches for ports, the listener that does the watching, and the links it
/// has made.
struct ApplicationLinks {
    made: Rc<RefCell<Vec<Made>>>,
    listener: Option<pw::registry::Listener>,
    registry: pw::registry::RegistryRc,
}

impl Drop for ApplicationLinks {
    fn drop(&mut self) {
        // Listener first: it borrows nothing, but a callback firing while the
        // links are being dropped would be making links to a stream that is
        // going away.
        self.listener.take();
        self.made.borrow_mut().clear();
        let _ = &self.registry;
    }
}

/// Links one application's playback to this capture, and keeps doing it as
/// ports appear.
///
/// Ports on both sides arrive through the registry, in no fixed order and
/// not necessarily before this returns: the application's are already there,
/// this stream's appear as it negotiates. So every port is collected as it
/// comes and the pairing is redone each time, which also covers an
/// application that grows a channel while it plays.
///
/// And as ports go. A program that stops playing withdraws its node, and a
/// stream that renegotiates replaces its own ports; a port left in these
/// lists after it has gone is one the pairing keeps offering, which is a
/// link that cannot be made and — where ports do not name their channels,
/// so the pairing is by position — one that lands on the wrong port.
fn link_application(
    core: &pw::core::CoreRc,
    stream: &pw::stream::StreamRc,
    node: u32,
) -> std::result::Result<ApplicationLinks, PipeWireAudioCaptureSourceError> {
    let registry = core
        .get_registry_rc()
        .map_err(|error| PipeWireAudioCaptureSourceError::PipeWire(error.to_string()))?;
    let made = Rc::new(RefCell::new(Vec::<Made>::new()));
    let outputs = Rc::new(RefCell::new(Vec::<Port>::new()));
    let inputs = Rc::new(RefCell::new(Vec::<Port>::new()));

    let listener = registry
        .add_listener_local()
        .global({
            let core = core.clone();
            let stream = stream.clone();
            let outputs = outputs.clone();
            let inputs = inputs.clone();
            let made = made.clone();
            move |global| {
                if global.type_ != pw::types::ObjectType::Port {
                    return;
                }
                let Some(props) = global.props else { return };
                let Some(owner) = props.get("node.id").and_then(|id| id.parse::<u32>().ok()) else {
                    return;
                };
                let port = Port {
                    id: global.id,
                    channel: props.get("audio.channel").map(str::to_owned),
                };
                // The stream's node id is only known once the daemon has
                // given it one, which is why this is read here rather than
                // when the listener was made.
                let own = stream.node_id();
                match props.get("port.direction") {
                    Some("out") if owner == node => outputs.borrow_mut().push(port),
                    Some("in") if owner == own => inputs.borrow_mut().push(port),
                    _ => return,
                }
                let (outputs, inputs) = (outputs.borrow(), inputs.borrow());
                let mut made = made.borrow_mut();
                for (output, input) in pair_ports(&outputs, &inputs) {
                    if made
                        .iter()
                        .any(|link| link.output == output && link.input == input)
                    {
                        continue;
                    }
                    match make_link(&core, node, output, own, input) {
                        Ok(link) => made.push(Made {
                            output,
                            input,
                            link,
                        }),
                        // Reported and carried on: one port pair that would
                        // not link leaves a capture missing a channel, which
                        // is worth saying and is not worth ending a capture
                        // that has the others.
                        Err(error) => {
                            pp_warn!(
                                pp_log: &element_pp_log(
                                    ElementType::PipeWireAudioCaptureSource,
                                    "pipewire-audio-capture",
                                    None,
                                ),
                                "could not link port {output} of node {node} to this capture: \
                                 {error}"
                            );
                        }
                    }
                }
            }
        })
        .global_remove({
            let outputs = outputs.clone();
            let inputs = inputs.clone();
            let made = made.clone();
            // Whatever went: an id that belonged to neither side's ports
            // matches nothing here, and one that did leaves with the link it
            // was carrying. Dropping that link proxy is what lets the same
            // pair be linked again if the port comes back — a program that
            // starts playing again publishes new ports, and the pairing
            // above is then free to reach them.
            move |id| {
                outputs.borrow_mut().retain(|port| port.id != id);
                inputs.borrow_mut().retain(|port| port.id != id);
                made.borrow_mut()
                    .retain(|link| link.output != id && link.input != id);
            }
        })
        .register();

    Ok(ApplicationLinks {
        made,
        listener: Some(listener),
        registry,
    })
}

/// Which of the application's ports feeds which of this capture's.
///
/// By channel where both sides say which they are, so a stereo application
/// reaches a stereo capture the right way round. Otherwise by position, and
/// with the shorter side repeated: an application playing in stereo into a
/// mono capture should be captured, not half-captured.
fn pair_ports(outputs: &[Port], inputs: &[Port]) -> Vec<(u32, u32)> {
    if outputs.is_empty() || inputs.is_empty() {
        return Vec::new();
    }
    let mut pairs = Vec::new();
    for index in 0..outputs.len().max(inputs.len()) {
        let output = &outputs[index % outputs.len()];
        let input = output
            .channel
            .as_deref()
            .and_then(|channel| {
                inputs
                    .iter()
                    .find(|input| input.channel.as_deref() == Some(channel))
            })
            .unwrap_or(&inputs[index % inputs.len()]);
        let pair = (output.id, input.id);
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    pairs
}

fn make_link(
    core: &pw::core::CoreRc,
    output_node: u32,
    output_port: u32,
    input_node: u32,
    input_port: u32,
) -> std::result::Result<pw::link::Link, pw::Error> {
    core.create_object::<pw::link::Link>(
        "link-factory",
        &properties! {
            *pw::keys::LINK_OUTPUT_NODE => output_node.to_string(),
            *pw::keys::LINK_OUTPUT_PORT => output_port.to_string(),
            *pw::keys::LINK_INPUT_NODE => input_node.to_string(),
            *pw::keys::LINK_INPUT_PORT => input_port.to_string(),
            // The link belongs to this capture: it goes when the stream does,
            // rather than outliving the element that made it.
            *pw::keys::OBJECT_LINGER => "false",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node is free to put its samples partway into the mapping, and SPA's
    /// chunk is what says where. Reading from the mapping's start instead
    /// returns whatever precedes them.
    #[test]
    fn samples_are_read_from_the_chunk_the_node_described() {
        let mut buffer = vec![0xFFu8; 64];
        buffer[16..48].fill(0x11);

        let samples = chunk_samples(&buffer, 16, 32, 4).expect("the chunk holds whole frames");
        assert_eq!(samples.len(), 32);
        assert!(
            samples.iter().all(|&byte| byte == 0x11),
            "an ignored offset reads the bytes before the samples"
        );
    }

    #[test]
    fn a_chunk_is_clamped_to_the_mapping_and_to_whole_frames() {
        let buffer = vec![0x22u8; 64];

        // A size reaching past the mapping is clamped rather than trusted.
        assert_eq!(chunk_samples(&buffer, 48, 64, 4).map(<[u8]>::len), Some(16));
        // A region that does not hold a whole frame is not half a frame.
        assert_eq!(chunk_samples(&buffer, 60, 3, 4), None);
        // An offset past the mapping describes nothing at all.
        assert_eq!(chunk_samples(&buffer, 128, 16, 4), None);
    }

    /// Opens a capture on whatever this machine publishes, or skips with a
    /// reason. Sinks are preferred: a monitor stream captures whatever is
    /// playing and needs no microphone permission.
    fn try_capture_source() -> Option<PipeWireAudioCaptureSource> {
        let devices = match PipeWireAudioCaptureSource::list_devices() {
            Ok(devices) if !devices.is_empty() => devices,
            Ok(_) => {
                eprintln!("skipping: this machine publishes no audio nodes");
                return None;
            }
            Err(error) => {
                eprintln!("skipping: no usable PipeWire session ({error})");
                return None;
            }
        };
        let device = devices
            .iter()
            .find(|device| device.kind == PipeWireAudioDeviceKind::Sink)
            .or_else(|| devices.first())
            .expect("the list is not empty")
            .clone();
        match PipeWireAudioCaptureSource::open(
            "capture-test",
            PipeWireAudioCaptureOptions { device },
        ) {
            Ok((source, _format)) => Some(source),
            Err(error) => {
                eprintln!("skipping: the node could not be opened ({error})");
                None
            }
        }
    }

    /// One application's capture, against an application of this test's own:
    /// this crate's renderer, whose playback stream is a node like any
    /// program's.
    ///
    /// What it proves is the linking. A capture connected with no autoconnect
    /// and nothing linked to it is never scheduled, so it produces no packets
    /// at all — packets arriving are the graph driving a link this element
    /// made itself. Silence would do: the samples are here so a link to the
    /// wrong node reads as one.
    #[cfg(feature = "pipewire-audio-renderer")]
    #[test]
    fn an_application_capture_takes_what_that_application_plays() {
        use crate::element::Sink;
        use crate::elements::{PipeWireAudioRenderer, PipeWireAudioRendererOptions};

        let Some(sink) = (match PipeWireAudioRenderer::list_devices() {
            Ok(devices) => devices
                .into_iter()
                .find(|device| device.kind == PipeWireAudioDeviceKind::Sink),
            Err(error) => {
                eprintln!("skipping: no usable PipeWire session ({error})");
                return;
            }
        }) else {
            eprintln!("skipping: this machine publishes no playback device");
            return;
        };
        // What is playing already, so the renderer's own stream can be told
        // from it: a native PipeWire client publishes neither a binary nor a
        // pid to recognize it by, and this crate's renderer is one.
        let before: Vec<u32> = PipeWireAudioCaptureSource::list_applications()
            .map(|applications| applications.iter().map(|one| one.node).collect())
            .unwrap_or_default();
        let (mut renderer, played) = match PipeWireAudioRenderer::open(
            "application-capture-test-playback",
            PipeWireAudioRendererOptions { device: sink },
        ) {
            Ok(opened) => opened,
            Err(error) => {
                eprintln!("skipping: the playback device could not be opened ({error})");
                return;
            }
        };

        // A tone quiet enough not to startle whoever is running the tests,
        // and loud enough to tell from the silence a capture linked to the
        // wrong thing returns.
        let mut phase = 0.0f32;
        let mut play = |frames: usize| {
            for _ in 0..frames {
                let samples = played.sample_rate as usize / 100;
                let mut frame = ffmpeg::frame::Audio::new(
                    played.sample_format,
                    samples,
                    played.channel_layout(),
                );
                frame.set_rate(played.sample_rate);
                let channels = played.channels as usize;
                // Written as bytes: interleaved `f32` is one plane whose
                // samples ffmpeg counts per channel, and this writes them all.
                let data = frame.data_mut(0);
                for sample in 0..samples {
                    phase += 440.0 * std::f32::consts::TAU / played.sample_rate as f32;
                    let value = (phase.sin() * 0.02).to_le_bytes();
                    for channel in 0..channels {
                        let at = (sample * channels + channel) * 4;
                        data[at..at + 4].copy_from_slice(&value);
                    }
                }
                renderer
                    .consume(MediaBuffer::Audio(Arc::new(frame)))
                    .expect("the renderer takes what it is given");
            }
        };

        // Playing is what publishes the node: a renderer with nothing to play
        // has no stream for anything to find.
        play(20);

        // The stream the renderer added, which is the node that was not
        // there a moment ago.
        let application = (0..30).find_map(|_| {
            let found = PipeWireAudioCaptureSource::list_applications()
                .ok()?
                .into_iter()
                .find(|application| !before.contains(&application.node));
            if found.is_none() {
                play(10);
                std::thread::sleep(Duration::from_millis(100));
            }
            found
        });
        let Some(application) = application else {
            eprintln!("skipping: this session does not publish a playing application as a node");
            return;
        };

        // Not a skip: the node is there and playing, so anything but an open
        // capture is this element failing to link itself — a stream with
        // nothing linked to it never negotiates a format, and that is what
        // an open times out on.
        let (capture, format) = PipeWireAudioCaptureSource::open_application(
            "application-capture-test",
            application.node,
        )
        .expect("an application that is playing can be captured");

        play(60);
        std::thread::sleep(Duration::from_millis(500));

        let mut captured = Vec::new();
        while let Ok(packet) = capture.packets.try_recv() {
            captured.extend_from_slice(&packet.bytes);
        }
        assert!(
            !captured.is_empty(),
            "an application capture that linked to nothing is never scheduled, \
             so no packets at all means no link was made"
        );
        let loudest = captured
            .as_chunks::<4>()
            .0
            .iter()
            .map(|sample| f32::from_le_bytes(*sample).abs())
            .fold(0.0f32, f32::max);
        assert!(
            loudest > 0.001,
            "the capture was linked to something other than the application: \
             it heard {loudest} where the application was playing a tone"
        );
        assert_eq!(format.sample_rate, APPLICATION_RATE);
    }

    /// A paused source must stop the capture itself, not just stop reading it.
    /// Left running, the daemon fills the queue during the pause, every later
    /// packet is discarded as an overload, and `Resume` emits the stale queue
    /// before any live audio.
    #[test]
    fn a_paused_capture_stops_asking_the_device_for_audio() {
        let Some(source) = try_capture_source() else {
            return;
        };
        // Capturing to begin with, or the rest of this proves nothing.
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            !source.packets.is_empty(),
            "the capture produced nothing at all, so this cannot tell a paused \
             stream from a silent one"
        );

        source
            .set_active(false)
            .expect("the capture stream can be paused");
        std::thread::sleep(Duration::from_millis(300));
        discard_captured(&source.packets);
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            source.packets.is_empty(),
            "a paused source must not keep capturing audio nobody will read"
        );

        source
            .set_active(true)
            .expect("the capture stream can be resumed");
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            !source.packets.is_empty(),
            "resuming must start the capture again"
        );
    }

    #[test]
    fn a_capture_control_completes_only_after_the_pipewire_reply() {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            reply_tx.send(Ok(())).expect("the caller is still waiting");
        });
        let started = Instant::now();

        wait_set_active(reply_rx, false).expect("the stream was deactivated");
        worker.join().expect("the reply thread finishes");

        assert!(
            started.elapsed() >= Duration::from_millis(40),
            "enqueueing alone must not acknowledge Pause"
        );
    }

    /// What `Resume` discards: the queue's contents, so playback starts from
    /// live audio rather than the pause boundary's leftovers.
    #[test]
    fn resuming_discards_what_the_pause_left_queued() {
        let (tx, rx) = crossbeam_channel::bounded::<Packet>(8);
        let format = stereo_f32();
        for position in 0..4 {
            tx.send(packet(&format, position * 480, 480))
                .expect("the queue has room");
        }

        discard_captured(&rx);

        assert!(rx.is_empty(), "stale audio must not outlive the pause");
        tx.send(packet(&format, 4 * 480, 480))
            .expect("the queue still accepts audio");
        assert_eq!(rx.try_recv().map(|p| p.position), Ok(1920));
    }

    fn stereo_f32() -> AudioFormat {
        AudioFormat::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            48_000,
            2,
        )
    }

    fn packet(format: &AudioFormat, position: u64, frames: usize) -> Packet {
        let bytes_per_frame = format.channels as usize * format.sample_format.bytes();
        Packet {
            bytes: vec![0xAB; frames * bytes_per_frame],
            frames,
            position,
        }
    }

    /// A discarded packet leaves a gap in the captured stream, and the gap
    /// belongs where the capture lost it. Stamping `pts` from a count kept on
    /// this side instead put it in front of whatever came out of the queue
    /// next — audio captured *before* the drop — moving the gap up to a full
    /// queue earlier than it happened.
    #[test]
    fn a_dropped_packet_leaves_its_gap_where_the_capture_lost_it() {
        let format = stereo_f32();
        // Three packets captured back to back; the middle one never made it
        // into the queue, so `run` never sees it.
        let first = build_frame(&format, &packet(&format, 0, 480));
        let third = build_frame(&format, &packet(&format, 960, 480));

        assert_eq!(first.pts(), Some(0));
        assert_eq!(
            third.pts(),
            Some(960),
            "the frame after a drop is stamped where it was captured, leaving \
             the gap over the packet that was actually lost"
        );
    }

    /// Nothing else about a packet changes with its position: the samples are
    /// copied whole, and the frame describes the negotiated format.
    #[test]
    fn a_captured_packet_keeps_its_samples_and_format() {
        let format = stereo_f32();
        let frame = build_frame(&format, &packet(&format, 4_800, 240));

        assert_eq!(frame.samples(), 240);
        assert_eq!(frame.rate(), 48_000);
        assert_eq!(frame.format(), format.sample_format);
        assert!(
            frame.data(0)[..240 * 8].iter().all(|&byte| byte == 0xAB),
            "every captured sample reaches the frame"
        );
    }

    fn device(kind: PipeWireAudioDeviceKind, name: &str) -> PipeWireAudioDevice {
        PipeWireAudioDevice {
            id: 1,
            name: name.into(),
            description: name.into(),
            kind,
            is_default: false,
        }
    }

    #[test]
    fn only_interleaved_formats_pipewire_negotiates_are_accepted() {
        use spa::param::audio::AudioFormat as Spa;
        assert_eq!(
            ffmpeg_sample_format(Spa::F32LE),
            Some(ffmpeg::format::Sample::F32(
                ffmpeg::format::sample::Type::Packed
            ))
        );
        assert_eq!(
            ffmpeg_sample_format(Spa::S16LE),
            Some(ffmpeg::format::Sample::I16(
                ffmpeg::format::sample::Type::Packed
            ))
        );
        assert_eq!(
            ffmpeg_sample_format(Spa::S32LE),
            Some(ffmpeg::format::Sample::I32(
                ffmpeg::format::sample::Type::Packed
            ))
        );
    }

    #[test]
    fn a_planar_format_is_rejected_rather_than_reinterpreted() {
        // Reading planar bytes as interleaved would produce noise, not an
        // error, so this must stay a hard rejection.
        assert!(ffmpeg_sample_format(spa::param::audio::AudioFormat::F32P).is_none());
    }

    #[test]
    fn defaults_are_matched_per_kind_not_across_kinds() {
        // A sink and a source can share a node.name; flagging by name alone
        // would mark both when only one is actually the default.
        let mut nodes = [
            device(PipeWireAudioDeviceKind::Sink, "shared"),
            device(PipeWireAudioDeviceKind::Source, "shared"),
        ];
        // Exercise the parsing directly with the metadata shape pw-metadata
        // prints, without depending on a running daemon.
        let line = "update: id:0 key:'default.audio.sink' \
                    value:'{\"name\":\"shared\"}' type:'Spa:String:JSON'";
        let name = line
            .split("\"name\":\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("the metadata line carries a name");
        assert_eq!(name, "shared");
        for node in nodes.iter_mut() {
            if node.kind == PipeWireAudioDeviceKind::Sink && node.name == name {
                node.is_default = true;
            }
        }
        assert!(nodes[0].is_default, "the sink is the default");
        assert!(
            !nodes[1].is_default,
            "a source sharing the name must not be flagged from the sink's default"
        );
    }

    #[test]
    fn seek_is_rejected_as_a_typed_error() {
        // `open` needs a live daemon, so the rejection is asserted against the
        // error itself rather than through a constructed element.
        let error: crate::Error = PipeWireAudioCaptureSourceError::SeekUnsupported.into();
        assert!(matches!(
            error,
            crate::Error::PipeWireAudioCaptureSourceError(
                PipeWireAudioCaptureSourceError::SeekUnsupported
            )
        ));
    }
}
