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
    let (quit_tx, quit_rx) = pw::channel::channel::<Quit>();
    let worker = std::thread::Builder::new()
        .name("pipewire-enumerate".into())
        .spawn(move || {
            let _ = tx.send(enumerate_nodes(quit_rx));
        })
        .map_err(|e| PipeWireDeviceError::PipeWire(e.to_string()))?;

    await_enumeration(rx, worker, ENUMERATION_TIMEOUT, move || {
        let _ = quit_tx.send(Quit);
    })
}

/// Waits out one enumeration, giving up when `timeout` passes.
///
/// The thread is joined only once it has something to report. Joining
/// unconditionally is what made the timeout no timeout at all: a `core.sync`
/// that never comes back, or a daemon that stops answering, left the caller
/// blocked in `join` long after the deadline it asked for. On a timeout the
/// loop is asked to stop through `stop` and the thread is left to end on its
/// own -- it owns everything it touches, so nothing here has to outlive it.
fn await_enumeration(
    rx: mpsc::Receiver<std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError>>,
    worker: std::thread::JoinHandle<()>,
    timeout: Duration,
    stop: impl FnOnce(),
) -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError> {
    match rx.recv_timeout(timeout) {
        Ok(devices) => {
            let _ = worker.join();
            devices
        }
        Err(RecvTimeoutError::Timeout) => {
            stop();
            Err(PipeWireDeviceError::EnumerationTimeout)
        }
        Err(RecvTimeoutError::Disconnected) => {
            let _ = worker.join();
            Err(PipeWireDeviceError::PipeWire(
                "the enumeration thread exited without a result".into(),
            ))
        }
    }
}

/// Sent into [`enumerate_nodes`]'s own main loop when the caller has stopped
/// waiting for it.
pub(crate) struct Quit;

/// One registry round trip, on its own thread's main loop.
///
/// `quit` ends the loop early: [`list_devices`] sends it when its timeout has
/// passed, so a wedged daemon leaves this thread finishing on its own rather
/// than holding a connection open for the life of the process.
pub(crate) fn enumerate_nodes(
    quit: pw::channel::Receiver<Quit>,
) -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError> {
    fn pw_err(error: impl std::fmt::Display) -> PipeWireDeviceError {
        PipeWireDeviceError::PipeWire(error.to_string())
    }

    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(pw_err)?;
    let context = pw::context::ContextRc::new(&mainloop, None).map_err(pw_err)?;
    let core = context.connect_rc(None).map_err(pw_err)?;
    let registry = core.get_registry_rc().map_err(pw_err)?;

    let quit_loop = mainloop.clone();
    let _quit = quit.attach(mainloop.loop_(), move |_| quit_loop.quit());

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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// The timeout has to bound what the *caller* waits, not just when the
    /// result is read: a daemon that never answers must not hold the caller
    /// past it.
    #[test]
    fn enumeration_gives_up_at_its_timeout_rather_than_on_the_thread() {
        let (tx, rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        // A thread that answers long after the deadline is what a wedged
        // registry round trip looks like from here.
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(3));
            let _ = tx.send(Ok(Vec::new()));
        });

        let started = Instant::now();
        let result = await_enumeration(rx, worker, Duration::from_millis(150), move || {
            let _ = stopped_tx.send(());
        });

        assert!(matches!(
            result,
            Err(PipeWireDeviceError::EnumerationTimeout)
        ));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the caller waited {:?}, past the timeout it asked for",
            started.elapsed()
        );
        assert!(
            stopped_rx.try_recv().is_ok(),
            "the loop must be told to stop, or it holds its connection open \
             for the life of the process"
        );
    }

    /// The ordinary path still joins, so a finished thread is never left
    /// behind.
    #[test]
    fn a_finished_enumeration_is_collected_and_joined() {
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = tx.send(Ok(vec![PipeWireAudioDevice {
                id: 1,
                name: "node".into(),
                description: "Node".into(),
                kind: PipeWireAudioDeviceKind::Sink,
                is_default: true,
            }]));
        });

        let devices = await_enumeration(rx, worker, Duration::from_secs(5), || {
            panic!("a result that arrived in time must not stop the loop early")
        })
        .expect("the enumeration succeeded");

        assert_eq!(devices.len(), 1);
    }
}
