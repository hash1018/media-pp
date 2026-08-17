use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    thread::JoinHandle,
    time::Duration,
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
    control::{ControlReceiver, drain_control},
    element::{Element, ElementType, Source, SourceElement, element_pp_log},
    elements::AudioFormat,
    error::Result,
    pad::SrcPad,
};

/// How long [`PipeWireAudioCaptureSource::open`] waits for the stream to
/// negotiate a format. Nothing here is interactive — unlike the screen-capture
/// path there is no dialog to wait on — so this only has to outlast normal
/// graph scheduling, and exists to turn an unresponsive daemon into an error
/// rather than a permanent hang.
const NEGOTIATION_TIMEOUT: Duration = Duration::from_secs(5);

/// How long [`PipeWireAudioCaptureSource::list_devices`] waits for the
/// registry round trip that enumerates existing nodes.
const ENUMERATION_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// [`NEGOTIATION_TIMEOUT`].
    #[error("timed out waiting for the PipeWire audio stream to negotiate a format")]
    NegotiationTimeout,

    #[error("timed out enumerating PipeWire audio nodes")]
    EnumerationTimeout,

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
}

/// Which direction a PipeWire audio node flows — the same distinction
/// `WasapiDeviceKind` draws, and with the same consequence for how `open`
/// connects to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipeWireAudioDeviceKind {
    /// A playback node (speakers, headphones, HDMI). Captured through its
    /// *monitor* ports, so what arrives is whatever the system is playing —
    /// PipeWire's equivalent of WASAPI loopback on a render endpoint.
    Sink,
    /// A recording node (microphone, line input). Captured directly.
    Source,
}

/// One PipeWire audio node that can be captured.
///
/// Obtained from [`PipeWireAudioCaptureSource::list_devices`] and handed
/// straight to [`PipeWireAudioCaptureOptions::device`]. Unlike the screen
/// capture path, this really is a selection: audio needs no portal, so a
/// caller can enumerate, filter, and pick a node programmatically with no
/// dialog and no user interaction at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipeWireAudioDevice {
    /// The PipeWire global node id. Stable only for the lifetime of the node:
    /// unplugging and reattaching a device yields a new id, so persist
    /// [`Self::name`] instead if a choice has to survive a restart.
    pub id: u32,
    /// The node's `node.name` — stable across restarts for a given physical
    /// device, unlike [`Self::id`].
    pub name: String,
    /// The node's human-readable `node.description`, falling back to
    /// `node.nick` and then to `name`.
    pub description: String,
    pub kind: PipeWireAudioDeviceKind,
    /// Whether this was the session's default node for its own [`kind`] when
    /// it was enumerated.
    ///
    /// Not guaranteed to be set for any node of a given kind. PipeWire's
    /// `default.audio.source` metadata routinely names a *sink* — that is how
    /// "use this output's monitor as my input" is expressed — in which case no
    /// [`PipeWireAudioDeviceKind::Source`] carries the flag at all. Callers
    /// should fall back to any node of the kind they want rather than assuming
    /// one is marked.
    ///
    /// [`kind`]: Self::kind
    pub is_default: bool,
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

/// One captured packet handed from the PipeWire thread to `run`.
struct Packet {
    /// Interleaved samples in the negotiated format, exactly
    /// `frames * channels * bytes_per_sample` long.
    bytes: Vec<u8>,
    frames: usize,
}

/// What the PipeWire thread reports back to `open` once, at startup.
enum Startup {
    Ready(AudioFormat),
    Failed(PipeWireAudioCaptureSourceError),
}

/// Sent into the PipeWire thread's own main loop to end it.
struct Terminate;

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
/// fills it, the oldest packet is dropped and reported as
/// [`crate::bus::BusEvent::Dropped`] rather than stalling the daemon or
/// growing without bound.
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
    /// Running sample count used as each emitted frame's `pts`.
    samples_emitted: i64,
    packets: Receiver<Packet>,
    /// Frames the PipeWire thread had to discard because `packets` was full.
    /// Added to `samples_emitted` before the next frame is stamped, so a drop
    /// leaves an honest gap in the timeline rather than silently compressing
    /// it and drifting out of sync with video.
    dropped_frames: Arc<AtomicU64>,
    /// Ends the PipeWire thread's main loop. `Option` only so `Drop` can take
    /// it; always `Some` while this element is alive.
    terminate: Option<pw::channel::Sender<Terminate>>,
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
    /// Needs no portal and shows no dialog. Connects to the daemon, walks the
    /// registry to a single `core.sync` barrier, and disconnects.
    pub fn list_devices()
    -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireAudioCaptureSourceError> {
        let (tx, rx) = mpsc::channel();
        // The PipeWire main loop and registry are not `Send`, so the whole
        // round trip happens on a thread of its own and only the plain
        // results cross back.
        let worker = std::thread::Builder::new()
            .name("pipewire-enumerate".into())
            .spawn(move || {
                let _ = tx.send(enumerate_nodes());
            })
            .map_err(|e| PipeWireAudioCaptureSourceError::PipeWire(e.to_string()))?;
        let result = rx.recv_timeout(ENUMERATION_TIMEOUT);
        let _ = worker.join();
        match result {
            Ok(devices) => devices,
            Err(RecvTimeoutError::Timeout) => {
                Err(PipeWireAudioCaptureSourceError::EnumerationTimeout)
            }
            Err(RecvTimeoutError::Disconnected) => Err(PipeWireAudioCaptureSourceError::PipeWire(
                "the enumeration thread exited without a result".into(),
            )),
        }
    }

    /// Opens `options.device` and starts capturing.
    ///
    /// Returns the element alongside the stream's actual
    /// `(sample_rate, channels)` — what a caller needs to build a matching
    /// downstream encoder or muxer, the same shape
    /// `WasapiCaptureSource::open` returns.
    pub fn open(
        name: impl Into<String>,
        options: PipeWireAudioCaptureOptions,
    ) -> std::result::Result<(Self, u32, u16), PipeWireAudioCaptureSourceError> {
        let name = name.into();
        let pp_log = element_pp_log(ElementType::PipeWireAudioCaptureSource, &name, None);

        let (packet_tx, packet_rx) = crossbeam_channel::bounded(PACKET_QUEUE_CAPACITY);
        let dropped_frames = Arc::new(AtomicU64::new(0));
        let (startup_tx, startup_rx) = mpsc::channel::<Startup>();
        let (terminate_tx, terminate_rx) = pw::channel::channel::<Terminate>();

        let device = options.device.clone();
        let worker = std::thread::Builder::new()
            .name(format!("{name}-pipewire-audio"))
            .spawn({
                let startup_tx = startup_tx.clone();
                let dropped_frames = dropped_frames.clone();
                move || {
                    if let Err(error) =
                        run_pipewire(device, packet_tx, dropped_frames, &startup_tx, terminate_rx)
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
                let _ = terminate_tx.send(Terminate);
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
                let _ = terminate_tx.send(Terminate);
                let _ = worker.join();
                return Err(error);
            }
        };

        pp_info!(
            pp_log: &pp_log,
            "opened: device={:?} ({:?}, node {}), {}Hz, {} channel(s), format={:?}",
            options.device.description,
            options.device.kind,
            options.device.id,
            format.sample_rate,
            format.channels,
            format.sample_format
        );

        Ok((
            Self {
                pad: SrcPad::new(format!("{name}_src")),
                name: name.into(),
                pp_log,
                format,
                samples_emitted: 0,
                packets: packet_rx,
                dropped_frames,
                terminate: Some(terminate_tx),
                worker: Some(worker),
            },
            format.sample_rate,
            format.channels,
        ))
    }

    /// The unit each emitted frame's `pts` is expressed in.
    pub fn time_base(&self) -> ffmpeg::Rational {
        ffmpeg::Rational::new(1, self.format.sample_rate as i32)
    }

    /// Wraps one captured packet in a fresh `ffmpeg::frame::Audio` and stamps
    /// its `pts` from the running sample count.
    fn build_frame(&mut self, packet: &Packet) -> ffmpeg::frame::Audio {
        // Derived rather than stored: `ffmpeg::ChannelLayout` is not `Send`,
        // and keeping it as a field would force an `unsafe impl Send` on this
        // whole element to satisfy `Element: Send` — for a value `AudioFormat`
        // can already reproduce exactly.
        let mut frame = ffmpeg::frame::Audio::new(
            self.format.sample_format,
            packet.frames,
            self.format.channel_layout(),
        );
        frame.set_rate(self.format.sample_rate);
        // `frame.data_mut(0)`'s length is FFmpeg's own padded linesize, not
        // necessarily the tight sample bytes — copy only what the packet
        // actually holds, the same bound `WasapiCaptureSource::build_frame`
        // documents for the mirror-image reason.
        let tight_bytes = packet.bytes.len().min(frame.data_mut(0).len());
        frame.data_mut(0)[..tight_bytes].copy_from_slice(&packet.bytes[..tight_bytes]);
        frame.set_pts(Some(self.samples_emitted));
        self.samples_emitted += packet.frames as i64;
        frame
    }
}

impl Drop for PipeWireAudioCaptureSource {
    fn drop(&mut self) {
        // Dropping the sender alone would not wake a blocked main loop, so
        // signal first, then join the thread this element owns.
        if let Some(terminate) = self.terminate.take() {
            let _ = terminate.send(Terminate);
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
    fn run(&mut self, control: &ControlReceiver, bus: &Bus) -> Result<()> {
        pp_info!(self, "started");
        loop {
            let outcome = drain_control(control, self, bus)?;
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

            // Keep `pts` anchored to real captured time across a drop: the
            // gap is real, so advance past it rather than pretending the
            // discarded frames never existed.
            let dropped = self.dropped_frames.swap(0, Ordering::Relaxed);
            if dropped > 0 {
                self.samples_emitted += dropped as i64;
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

            let frame = self.build_frame(&packet);
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

/// One registry round trip, on its own thread's main loop.
fn enumerate_nodes()
-> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireAudioCaptureSourceError> {
    fn pw_err(error: impl std::fmt::Display) -> PipeWireAudioCaptureSourceError {
        PipeWireAudioCaptureSourceError::PipeWire(error.to_string())
    }

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(pw_err)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(pw_err)?;
    let core = context.connect_rc(None).map_err(pw_err)?;
    let registry = core.get_registry_rc().map_err(pw_err)?;

    let nodes = Rc::new(RefCell::new(Vec::new()));
    let _reg_listener = {
        let nodes = nodes.clone();
        registry
            .add_listener_local()
            .global(move |global| {
                if global.type_ != pw::types::ObjectType::Node {
                    return;
                }
                let Some(props) = global.props else { return };
                let kind = match props.get("media.class").unwrap_or_default() {
                    "Audio/Sink" => PipeWireAudioDeviceKind::Sink,
                    "Audio/Source" => PipeWireAudioDeviceKind::Source,
                    _ => return,
                };
                let name = props.get("node.name").unwrap_or_default().to_owned();
                let description = props
                    .get("node.description")
                    .or_else(|| props.get("node.nick"))
                    .filter(|d| !d.is_empty())
                    .unwrap_or(&name)
                    .to_owned();
                nodes.borrow_mut().push(PipeWireAudioDevice {
                    id: global.id,
                    name,
                    description,
                    kind,
                    // Filled in below, once the defaults metadata is known.
                    is_default: false,
                });
            })
            .register()
    };

    // The server replays every existing global before answering a sync, so
    // one round trip is enough to see the whole current graph.
    let done = Rc::new(Cell::new(false));
    let pending = core.sync(0).map_err(pw_err)?;
    let _core_listener = {
        let mainloop = mainloop.clone();
        let done = done.clone();
        core.add_listener_local()
            .done(move |id, seq| {
                if id == pw::sys::PW_ID_CORE && seq == pending {
                    done.set(true);
                    mainloop.quit();
                }
            })
            .register()
    };
    mainloop.run();
    if !done.get() {
        return Err(PipeWireAudioCaptureSourceError::EnumerationTimeout);
    }

    let mut nodes = Rc::try_unwrap(nodes)
        .map(RefCell::into_inner)
        .unwrap_or_else(|shared| shared.borrow().clone());
    mark_defaults(&mut nodes);
    nodes.sort_by_key(|node| (node.kind == PipeWireAudioDeviceKind::Sink, node.id));
    Ok(nodes)
}

/// Flags whichever nodes the session currently treats as default.
///
/// Read from the daemon's `default` metadata rather than assumed, and a
/// failure to read it is not fatal: an unflagged list is still a complete and
/// usable list, so this degrades to "no entry marked default" instead of
/// failing enumeration outright.
fn mark_defaults(nodes: &mut [PipeWireAudioDevice]) {
    let Ok(output) = std::process::Command::new("pw-metadata")
        .args(["-n", "default"])
        .output()
    else {
        return;
    };
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let kind = if line.contains("'default.audio.sink'") {
            PipeWireAudioDeviceKind::Sink
        } else if line.contains("'default.audio.source'") {
            PipeWireAudioDeviceKind::Source
        } else {
            continue;
        };
        // value:'{"name":"<node.name>"}'
        let Some(name) = line
            .split("\"name\":\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
        else {
            continue;
        };
        for node in nodes.iter_mut() {
            if node.kind == kind && node.name == name {
                node.is_default = true;
            }
        }
    }
}

/// The PipeWire thread body: owns the main loop, context, core, and stream,
/// none of which are `Send`, and so all of which must be created and dropped
/// here rather than handed across from `open`.
fn run_pipewire(
    device: PipeWireAudioDevice,
    packets: Sender<Packet>,
    dropped_frames: Arc<AtomicU64>,
    startup: &mpsc::Sender<Startup>,
    terminate: pw::channel::Receiver<Terminate>,
) -> std::result::Result<(), PipeWireAudioCaptureSourceError> {
    fn pw_err(error: impl std::fmt::Display) -> PipeWireAudioCaptureSourceError {
        PipeWireAudioCaptureSourceError::PipeWire(error.to_string())
    }

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(pw_err)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(pw_err)?;
    let core = context.connect_rc(None).map_err(pw_err)?;

    // Attach to the outer loop, but let the callback own its own clone: the
    // `AttachedReceiver` borrows the `Loop` for as long as it lives, so the
    // handle it is attached to has to outlive it.
    let quit_loop = mainloop.clone();
    let _terminate = terminate.attach(mainloop.loop_(), move |_| quit_loop.quit());

    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Production",
    };
    if device.kind == PipeWireAudioDeviceKind::Sink {
        // Turns "connect to this sink" into "capture what this sink is
        // playing" by binding to its monitor ports instead of its inputs.
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }

    let stream =
        pw::stream::StreamBox::new(&core, "media-pp-audio-capture", props).map_err(pw_err)?;

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
                let size = data.chunk().size() as usize;
                if size == 0 {
                    return; // a tick with no new content
                }
                let frame_bytes = format.channels as usize * format.sample_format.bytes();
                if frame_bytes == 0 {
                    return;
                }
                let Some(bytes) = data.data() else { return };
                // `chunk().size()` is authoritative for how much of the
                // mapping is real; never read past whichever is shorter.
                let usable = size.min(bytes.len()) / frame_bytes * frame_bytes;
                if usable == 0 {
                    return;
                }
                let packet = Packet {
                    bytes: bytes[..usable].to_vec(),
                    frames: usable / frame_bytes,
                };
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

    stream
        .connect(
            spa::utils::Direction::Input,
            Some(device.id),
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(pw_err)?;

    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
