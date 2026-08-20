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

/// Gives a loop woken by the timeout a bounded chance to release its
/// PipeWire objects before the caller returns. This is cleanup time, not a
/// second enumeration timeout.
const ENUMERATION_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(500);

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

#[derive(Default)]
struct DefaultNames {
    sink: Option<String>,
    source: Option<String>,
}

/// Enumerates every currently-published audio node.
///
/// Needs no portal and shows no dialog: connects to the daemon, walks the
/// registry to a single `core.sync` barrier, and disconnects. The PipeWire main
/// loop and registry are not `Send`, so the whole round trip happens on a
/// thread of its own and only the plain results cross back.
pub(crate) fn list_devices() -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError> {
    let (tx, rx) = mpsc::channel();
    let (exit_tx, exit_rx) = mpsc::channel();
    let (quit_tx, quit_rx) = pw::channel::channel::<Quit>();
    let worker = std::thread::Builder::new()
        .name("pipewire-enumerate".into())
        .spawn(move || {
            let _ = tx.send(enumerate_nodes(quit_rx));
            let _ = exit_tx.send(());
        })
        .map_err(|e| PipeWireDeviceError::PipeWire(e.to_string()))?;

    await_enumeration(
        rx,
        exit_rx,
        worker,
        ENUMERATION_TIMEOUT,
        ENUMERATION_SHUTDOWN_TIMEOUT,
        move || {
            let _ = quit_tx.send(Quit);
        },
    )
}

/// Waits out one enumeration, giving up when `timeout` passes.
///
/// The thread is joined once it has something to report. On timeout the loop
/// is first asked to stop, then given a short cleanup deadline; a genuinely
/// wedged PipeWire call still cannot hold the caller indefinitely.
fn await_enumeration(
    rx: mpsc::Receiver<std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError>>,
    exit: mpsc::Receiver<()>,
    worker: std::thread::JoinHandle<()>,
    timeout: Duration,
    shutdown_timeout: Duration,
    stop: impl FnOnce(),
) -> std::result::Result<Vec<PipeWireAudioDevice>, PipeWireDeviceError> {
    match rx.recv_timeout(timeout) {
        Ok(devices) => {
            let _ = worker.join();
            devices
        }
        Err(RecvTimeoutError::Timeout) => {
            stop();
            if exit.recv_timeout(shutdown_timeout).is_ok() {
                let _ = worker.join();
            }
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
    let defaults = Rc::new(RefCell::new(DefaultNames::default()));
    // A metadata listener must not outlive its proxy. Keeping both here also
    // keeps the daemon's replay of the `default` metadata active through the
    // sync barrier below.
    let metadata = Rc::new(RefCell::new(Vec::<(
        pw::metadata::MetadataListener,
        pw::metadata::Metadata,
    )>::new()));
    let _reg_listener = {
        let nodes = nodes.clone();
        let defaults = defaults.clone();
        let metadata = metadata.clone();
        let registry_for_bind = registry.clone();
        registry
            .add_listener_local()
            .global(move |global| {
                let Some(props) = global.props else { return };
                match global.type_ {
                    pw::types::ObjectType::Node => {
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
                    }
                    pw::types::ObjectType::Metadata
                        if props.get("metadata.name") == Some("default") =>
                    {
                        let Ok(proxy) = registry_for_bind.bind::<pw::metadata::Metadata, _>(global)
                        else {
                            return;
                        };
                        let listener = proxy
                            .add_listener_local()
                            .property({
                                let defaults = defaults.clone();
                                move |_, key, _, value| {
                                    let mut defaults = defaults.borrow_mut();
                                    match key {
                                        Some("default.audio.sink") => {
                                            defaults.sink = value.and_then(default_node_name)
                                        }
                                        Some("default.audio.source") => {
                                            defaults.source = value.and_then(default_node_name)
                                        }
                                        _ => {}
                                    }
                                    0
                                }
                            })
                            .register();
                        // Tuple fields drop in order, so unregister the
                        // listener before releasing the proxy it belongs to.
                        metadata.borrow_mut().push((listener, proxy));
                    }
                    _ => {}
                }
            })
            .register()
    };

    // The first sync waits for the registry replay. Metadata is bound from a
    // registry callback, so a second sync must be issued only after that first
    // barrier: its property replay is ordered before the second reply.
    let done = Rc::new(Cell::new(false));
    let first_pending = core.sync(0).map_err(pw_err)?;
    let second_pending = Rc::new(RefCell::new(None));
    let sync_error = Rc::new(RefCell::new(None));
    let _core_listener = {
        let mainloop = mainloop.clone();
        let done = done.clone();
        let core_for_sync = core.clone();
        let second_pending = second_pending.clone();
        let sync_error = sync_error.clone();
        core.add_listener_local()
            .done(move |id, seq| {
                if id != pw::core::PW_ID_CORE {
                    return;
                }
                if seq == first_pending {
                    match core_for_sync.sync(0) {
                        Ok(seq) => *second_pending.borrow_mut() = Some(seq),
                        Err(error) => {
                            *sync_error.borrow_mut() = Some(error.to_string());
                            mainloop.quit();
                        }
                    }
                } else if second_pending.borrow().as_ref() == Some(&seq) {
                    done.set(true);
                    mainloop.quit();
                }
            })
            .register()
    };
    mainloop.run();
    if let Some(error) = sync_error.borrow_mut().take() {
        return Err(PipeWireDeviceError::PipeWire(error));
    }
    if !done.get() {
        return Err(PipeWireDeviceError::EnumerationTimeout);
    }

    let mut nodes = Rc::try_unwrap(nodes)
        .map(RefCell::into_inner)
        .unwrap_or_else(|shared| shared.borrow().clone());
    mark_defaults(&mut nodes, &defaults.borrow());
    nodes.sort_by_key(|node| (node.kind == PipeWireAudioDeviceKind::Sink, node.id));
    Ok(nodes)
}

/// Flags whichever nodes the session currently treats as default.
///
/// Read from the daemon's `default` metadata rather than assumed, and a
/// failure to read it is not fatal: an unflagged list is still a complete and
/// usable list, so this degrades to "no entry marked default" instead of
/// failing enumeration outright.
fn default_node_name(value: &str) -> Option<String> {
    let (_, value) = value.split_once("\"name\"")?;
    let value = value.trim_start().strip_prefix(':')?.trim_start();
    let value = value.strip_prefix('"')?;
    let end = value.find('"')?;
    // PipeWire node names do not need JSON escapes in practice. Reject them
    // instead of silently comparing an encoded string with `node.name`.
    (!value[..end].contains('\\')).then(|| value[..end].to_owned())
}

fn mark_defaults(nodes: &mut [PipeWireAudioDevice], defaults: &DefaultNames) {
    for node in nodes {
        node.is_default = match node.kind {
            PipeWireAudioDeviceKind::Sink => defaults.sink.as_deref() == Some(&node.name),
            PipeWireAudioDeviceKind::Source => defaults.source.as_deref() == Some(&node.name),
        };
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
        let (_tx, rx) = mpsc::channel();
        let (exit_tx, exit_rx) = mpsc::channel();
        let (quit_tx, quit_rx) = mpsc::channel();
        let (stopped_tx, stopped_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = quit_rx.recv();
            let _ = exit_tx.send(());
        });

        let started = Instant::now();
        let result = await_enumeration(
            rx,
            exit_rx,
            worker,
            Duration::from_millis(150),
            Duration::from_millis(150),
            move || {
                let _ = stopped_tx.send(());
                let _ = quit_tx.send(());
            },
        );

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

    #[test]
    fn a_worker_that_ignores_quit_does_not_extend_the_cleanup_deadline() {
        let (result_tx, result_rx) = mpsc::channel();
        let (exit_tx, exit_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            // Keep the result channel connected while simulating a call that
            // cannot react to the stop request.
            let _result_tx = result_tx;
            let _ = release_rx.recv();
            let _ = exit_tx.send(());
            let _ = finished_tx.send(());
        });
        let started = Instant::now();

        let result = await_enumeration(
            result_rx,
            exit_rx,
            worker,
            Duration::from_millis(50),
            Duration::from_millis(50),
            || {},
        );

        assert!(matches!(
            result,
            Err(PipeWireDeviceError::EnumerationTimeout)
        ));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "an unresponsive worker must be detached at the cleanup deadline"
        );
        // Do not leave the deliberately detached test worker around.
        release_tx.send(()).expect("the worker is still waiting");
        finished_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("the detached worker can be released after the assertion");
    }

    /// The ordinary path still joins, so a finished thread is never left
    /// behind.
    #[test]
    fn a_finished_enumeration_is_collected_and_joined() {
        let (tx, rx) = mpsc::channel();
        let (exit_tx, exit_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = tx.send(Ok(vec![PipeWireAudioDevice {
                id: 1,
                name: "node".into(),
                description: "Node".into(),
                kind: PipeWireAudioDeviceKind::Sink,
                is_default: true,
            }]));
            let _ = exit_tx.send(());
        });

        let devices = await_enumeration(
            rx,
            exit_rx,
            worker,
            Duration::from_secs(5),
            Duration::from_millis(150),
            || panic!("a result that arrived in time must not stop the loop early"),
        )
        .expect("the enumeration succeeded");

        assert_eq!(devices.len(), 1);
    }

    #[test]
    fn parses_default_metadata_without_requiring_compact_json() {
        assert_eq!(
            default_node_name(r#"{ "name" : "alsa_output.pci" }"#).as_deref(),
            Some("alsa_output.pci")
        );
        assert_eq!(default_node_name(r#"{"id": 12}"#), None);
        assert_eq!(default_node_name(r#"{"name":"bad\\\"name"}"#), None);
    }

    #[test]
    fn marks_defaults_only_in_the_matching_direction() {
        let mut nodes = vec![
            PipeWireAudioDevice {
                id: 1,
                name: "same-name".into(),
                description: "Sink".into(),
                kind: PipeWireAudioDeviceKind::Sink,
                is_default: false,
            },
            PipeWireAudioDevice {
                id: 2,
                name: "same-name".into(),
                description: "Source".into(),
                kind: PipeWireAudioDeviceKind::Source,
                is_default: true,
            },
        ];

        mark_defaults(
            &mut nodes,
            &DefaultNames {
                sink: Some("same-name".into()),
                source: None,
            },
        );

        assert!(nodes[0].is_default);
        assert!(!nodes[1].is_default);
    }
}
