//! Shared PipeWire audio node enumeration.
//!
//! Both the capture source and the renderer pick from the same set of graph
//! nodes, so the device type and the registry round trip that produces it live
//! here rather than in either element — the same split
//! `platform::windows::wasapi` uses for `WasapiDevice`.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::mpsc::{self, RecvTimeoutError},
    time::Duration,
};

use pipewire as pw;
use thiserror::Error as ThisError;

/// How long [`list_devices`] waits for the registry round trip.
const ENUMERATION_TIMEOUT: Duration = Duration::from_secs(5);

/// Errors raised while enumerating PipeWire audio nodes. Each element wraps
/// this into its own error type rather than exposing it directly.
#[derive(Debug, ThisError)]
pub enum PipeWireDeviceError {
    #[error("pipewire error: {0}")]
    PipeWire(String),

    #[error("timed out enumerating PipeWire audio nodes")]
    EnumerationTimeout,
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

/// One PipeWire audio node that can be captured from or played to.
///
/// Obtained from `PipeWireAudioCaptureSource::list_devices` or
/// `PipeWireAudioRenderer::list_devices` and handed straight to that
/// element's own `device` option. Unlike the screen-capture path, this really
/// is a selection: audio needs no portal, so a caller can enumerate, filter,
/// and pick a node programmatically with no dialog and no user interaction.
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

/// Enumerates every currently-published audio node.
///
/// Needs no portal and shows no dialog: connects to the daemon, walks the
/// registry to a single `core.sync` barrier, and disconnects. The PipeWire main
/// loop and registry are not `Send`, so the whole round trip happens on a
/// thread of its own and only the plain results cross back.
pub(crate) fn list_devices() -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError> {
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name("pipewire-enumerate".into())
        .spawn(move || {
            let _ = tx.send(enumerate_nodes());
        })
        .map_err(|e| PipeWireDeviceError::PipeWire(e.to_string()))?;
    let result = rx.recv_timeout(ENUMERATION_TIMEOUT);
    let _ = worker.join();
    match result {
        Ok(devices) => devices,
        Err(RecvTimeoutError::Timeout) => Err(PipeWireDeviceError::EnumerationTimeout),
        Err(RecvTimeoutError::Disconnected) => Err(PipeWireDeviceError::PipeWire(
            "the enumeration thread exited without a result".into(),
        )),
    }
}

/// One registry round trip, on its own thread's main loop.
pub(crate) fn enumerate_nodes() -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError>
{
    fn pw_err(error: impl std::fmt::Display) -> PipeWireDeviceError {
        PipeWireDeviceError::PipeWire(error.to_string())
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
        return Err(PipeWireDeviceError::EnumerationTimeout);
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
