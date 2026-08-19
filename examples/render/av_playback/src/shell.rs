//! Everything about this example that is not the pipeline: the window, the
//! terminal command loop, and the timestamp parsing. Both backends drive the
//! identical shell, so the only thing that differs between platforms is how
//! the pipeline itself is built.

use std::{
    io::{self, BufRead},
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use media_pp::{elements::TeeHandle, graph::BranchId, pipeline::Pipeline};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle},
    window::{Window, WindowId},
};

/// Where the backend should present, handed to the playback thread.
///
/// # Safety of the `Send` impl
///
/// On Wayland these handles are raw pointers into the window the main thread
/// owns, so they are only valid while it lives. [`App`] is what makes that
/// hold: it stops the pipeline on `CloseRequested` and joins the playback
/// thread in `Drop`, both before the `Window` is dropped.
pub struct WindowTarget {
    // Only `linux_example::play` (main.rs) reads this, for `VulkanGpuContext`;
    // the Windows backend presents through `window` alone. Compiling for one
    // platform at a time makes that a dead-code warning on the other, not a
    // real defect — see this type's own doc comment for why the field still
    // belongs on the shared shell struct instead of behind a `cfg`.
    #[allow(dead_code)]
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
    pub width: u32,
    pub height: u32,
}

// SAFETY: see the type's own docs — the window outlives the playback thread.
unsafe impl Send for WindowTarget {}

pub struct PlaybackDone;

/// Runs the window and hands `play` a target to render into. `play` reports
/// its `Pipeline` back through `started` so the window can stop playback when
/// the user closes it.
pub fn run(
    title: &str,
    play: fn(String, WindowTarget, mpsc::Sender<Arc<Pipeline>>) -> media_pp::Result<()>,
) {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: av_playback <video-with-audio.mp4>");
        std::process::exit(2);
    };

    let event_loop = EventLoop::<PlaybackDone>::with_user_event()
        .build()
        .expect("failed to create event loop");
    let proxy = event_loop.create_proxy();
    let mut app = App {
        title: title.to_string(),
        path,
        play,
        proxy,
        window: None,
        playback: None,
        pipeline: None,
        pipeline_rx: None,
    };
    event_loop.run_app(&mut app).expect("event loop failed");
}

struct App {
    title: String,
    path: String,
    play: fn(String, WindowTarget, mpsc::Sender<Arc<Pipeline>>) -> media_pp::Result<()>,
    proxy: EventLoopProxy<PlaybackDone>,
    window: Option<Window>,
    playback: Option<thread::JoinHandle<()>>,
    pipeline: Option<Arc<Pipeline>>,
    pipeline_rx: Option<mpsc::Receiver<Arc<Pipeline>>>,
}

impl App {
    /// The playback thread reports its pipeline once it exists; pick it up
    /// lazily so the window never blocks waiting for startup.
    fn pipeline(&mut self) -> Option<&Arc<Pipeline>> {
        if self.pipeline.is_none()
            && let Some(rx) = &self.pipeline_rx
            && let Ok(pipeline) = rx.try_recv()
        {
            self.pipeline = Some(pipeline);
        }
        self.pipeline.as_ref()
    }
}

impl ApplicationHandler<PlaybackDone> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(self.title.clone())
                    .with_inner_size(LogicalSize::new(1280, 720))
                    .with_resizable(false),
            )
            .expect("failed to create window");

        let size = window.inner_size();
        let target = WindowTarget {
            display: window
                .display_handle()
                .expect("failed to get display handle")
                .as_raw(),
            window: window
                .window_handle()
                .expect("failed to get window handle")
                .as_raw(),
            width: size.width,
            height: size.height,
        };

        let (tx, rx) = mpsc::channel();
        self.pipeline_rx = Some(rx);
        let path = self.path.clone();
        let proxy = self.proxy.clone();
        let play = self.play;
        self.playback = Some(thread::spawn(move || {
            if let Err(error) = play(path, target, tx) {
                eprintln!("playback failed: {error}");
            }
            let _ = proxy.send_event(PlaybackDone);
        }));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.window.as_ref().map(Window::id) != Some(window_id) {
            return;
        }
        if matches!(event, WindowEvent::CloseRequested) {
            // Stop first: the playback thread is still rendering into this
            // window's surface, and on Wayland tearing that down underneath
            // it is a use-after-free, not just a lost frame.
            if let Some(pipeline) = self.pipeline() {
                pipeline.stop();
            }
            event_loop.exit();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: PlaybackDone) {
        event_loop.exit();
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(pipeline) = self.pipeline() {
            pipeline.stop();
        }
        if let Some(playback) = self.playback.take() {
            let _ = playback.join();
        }
        // Only now is it safe to drop the window.
        self.window = None;
    }
}

/// The terminal command loop. `attach_audio` is the one backend-specific
/// piece — everything else here is identical on every platform.
pub fn read_commands(
    pipeline: Arc<Pipeline>,
    audio_tee: TeeHandle,
    audio_output: &str,
    attach_audio: impl Fn() -> media_pp::Result<BranchId>,
) {
    let mut audio_branch = None;
    print_help(audio_output);

    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let command = line.trim().to_ascii_lowercase();
        let words = command.split_whitespace().collect::<Vec<_>>();

        match words.as_slice() {
            [] => {}
            ["audio", "on"] => {
                if audio_branch.is_some() {
                    println!("audio is already on");
                    continue;
                }
                match attach_audio() {
                    Ok(branch_id) => audio_branch = Some(branch_id),
                    Err(error) => eprintln!("could not enable audio: {error}"),
                }
            }
            ["audio", "off"] => {
                let Some(branch_id) = audio_branch else {
                    println!("audio is already off");
                    continue;
                };
                match audio_tee.detach(branch_id) {
                    Ok(()) => {
                        audio_branch = None;
                        println!("audio off; video returned to wall-clock pacing");
                    }
                    Err(error) => eprintln!("could not disable audio: {error}"),
                }
            }
            ["pause"] => {
                pipeline.pause();
                println!("paused");
            }
            ["resume"] => {
                pipeline.resume();
                println!("resumed");
            }
            ["seek", target] => match parse_timestamp(target) {
                Some(target) => {
                    let clock = if audio_branch.is_some() {
                        "audio master"
                    } else {
                        "wall clock"
                    };
                    println!("seeking to {target:.2?} ({clock})...");
                    pipeline.seek(target);
                }
                None => eprintln!(
                    "could not parse {target:?}; use seconds (`seek 30`) or mm:ss (`seek 1:15`)"
                ),
            },
            ["help"] => print_help(audio_output),
            ["q"] | ["quit"] => {
                pipeline.stop();
                break;
            }
            _ => eprintln!("unknown command; type `help` for the command list"),
        }
    }
}

fn print_help(audio_output: &str) {
    println!("commands:");
    println!("  audio on          attach the default {audio_output} output");
    println!("  audio off         detach audio and keep video playing");
    println!("  pause             pause playback");
    println!("  resume            resume playback");
    println!("  seek <seconds>    seek, for example `seek 30` or `seek 1:15`");
    println!("  help              print this help");
    println!("  q                 stop playback");
}

/// `"90"` (plain seconds) or `"1:30"` (mm:ss) -> `Duration`.
fn parse_timestamp(value: &str) -> Option<Duration> {
    let seconds = match value.split_once(':') {
        Some((minutes, seconds)) => {
            minutes.parse::<f64>().ok()? * 60.0 + seconds.parse::<f64>().ok()?
        }
        None => value.parse::<f64>().ok()?,
    };
    if seconds.is_finite() && seconds >= 0.0 {
        Some(Duration::from_secs_f64(seconds))
    } else {
        None
    }
}
