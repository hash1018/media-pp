//! A file played in a window with its sound — the GStreamer `playbin` of
//! this crate.
//!
//! Everything here is built from the crate's own elements, and a program
//! that outgrows it builds the same graph itself: `FileDemuxer`, a
//! `SwDecoder` per stream, a `VideoSynchronizer` in front of a
//! [`VideoWindow`], and an `AudioResampler` in front of the platform's
//! audio renderer. What it saves is the wiring, the choice of devices, and
//! the event loop every player writes.

use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use crossbeam_channel::select;
use thiserror::Error as ThisError;

#[cfg(target_os = "linux")]
use crate::elements::{
    PipeWireAudioRenderer as AudioOut, PipeWireAudioRendererOptions as AudioOutOptions,
};
#[cfg(target_os = "windows")]
use crate::elements::{WasapiRenderer as AudioOut, WasapiRendererOptions as AudioOutOptions};
use crate::{
    bus::BusEvent,
    contract::{
        MediaKind, MemoryDomain, OutputContract, PixelLayout, PixelLayoutSet, PortContract,
        check_link,
    },
    element::Sink,
    elements::{
        AudioFormat, AudioResampler, FileDemuxer, FileDemuxerError, Key, MouseButton, SwDecoder,
        SwScaler, VideoSynchronizer, VideoWindow, VideoWindowError, WindowControl, WindowEvent,
        WindowEvents, WindowOptions,
    },
    ffmpeg,
    pipeline::{Pipeline, SeekMode},
};

/// How far [`Player::respond_to`] moves on an arrow key.
const ARROW_STEP: Duration = Duration::from_secs(5);

/// How a [`Player`] opens its window and its sound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerOptions {
    /// The window the picture is shown in.
    pub window: WindowOptions,
    /// Whether to play the file's sound, on the default output device. A
    /// file without sound plays its picture either way; with this set and
    /// no output device, [`Player::open`] fails rather than playing silently.
    pub audio: bool,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        Self {
            window: WindowOptions::default(),
            audio: true,
        }
    }
}

/// Why a [`Player`] could not open a file or do what it was asked.
#[derive(Debug, ThisError)]
pub enum PlayerError {
    /// The file could not be opened, or has no picture.
    #[error(transparent)]
    Open(#[from] FileDemuxerError),
    /// The window could not be opened.
    #[error(transparent)]
    Window(#[from] VideoWindowError),
    /// The sound was asked for and there is no output to play it into.
    #[error("no sound output: {0}")]
    Audio(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The pipeline could not be built, started or moved: the element's
    /// or the pipeline's own error.
    #[error(transparent)]
    Pipeline(Box<crate::Error>),
}

impl From<crate::Error> for PlayerError {
    fn from(error: crate::Error) -> Self {
        Self::Pipeline(Box::new(error))
    }
}

/// What a [`Player`] reports: what happened to its window, the end of the
/// file, and an element's failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum PlayerEvent {
    /// Something happened to the window — see [`Player::respond_to`] for
    /// what a player usually does about it.
    Window(WindowEvent),
    /// The file has played to its end: every stream has. The last picture
    /// stays in the window until the player is stopped or dropped.
    Ended,
    /// An element failed. Playback goes on where it can — an error does not
    /// end a pipeline — and whether this one means stopping is the
    /// application's to decide.
    Error {
        /// The element that failed.
        name: Arc<str>,
        /// Why.
        error: crate::Error,
    },
}

/// A file played in a window of its own, with its sound — open, play, and
/// wait for [`PlayerEvent`]s:
///
/// ```no_run
/// use media_pp::player::{Player, PlayerEvent, PlayerOptions};
///
/// let player = Player::open("video.mp4", PlayerOptions::default())?;
/// player.play()?;
/// while let Some(event) = player.next_event() {
///     match event {
///         PlayerEvent::Window(event) if !player.respond_to(&event) => break,
///         PlayerEvent::Ended => break,
///         PlayerEvent::Error { name, error } => eprintln!("{name}: {error}"),
///         _ => {}
///     }
/// }
/// # Ok::<(), media_pp::player::PlayerError>(())
/// ```
///
/// The picture is decoded in software and drawn by a [`VideoWindow`]; the
/// sound, decoded and resampled to the output's format, plays on the
/// default output device — `WasapiRenderer` on Windows,
/// `PipeWireAudioRenderer` on Linux — and the picture follows it: the
/// synchronizer schedules frames by the samples actually played. Without
/// sound, the picture keeps wall-clock time.
///
/// Its methods take `&self` and may be called from any thread. Dropping it
/// stops playback, and closes the window.
pub struct Player {
    pipeline: Arc<Pipeline>,
    window: WindowControl,
    events: WindowEvents,
    duration: Option<Duration>,
    started: AtomicBool,
    paused: AtomicBool,
    stopped: AtomicBool,
    /// Where the last seek went — what [`Self::position`] says until the
    /// clock has a sample of the new position to read, which a seek while
    /// paused does not give it until playing resumes.
    sought: Mutex<Option<Duration>>,
}

impl Player {
    /// Opens `path`, its window and its sound output, and builds what plays
    /// them — without starting: see [`Self::play`].
    pub fn open(path: impl AsRef<Path>, options: PlayerOptions) -> Result<Self, PlayerError> {
        let (source, _) = FileDemuxer::open("file", path)?;
        let duration = source.duration();
        let video = source.best(ffmpeg::media::Type::Video)?;
        let audio = if options.audio {
            source.best(ffmpeg::media::Type::Audio).ok()
        } else {
            None
        };
        let output = audio.as_ref().map(|_| open_output()).transpose()?;
        let (screen, events) = VideoWindow::open("screen", options.window)?;
        let window = screen.window_control();
        let to_drawable = to_drawable(&video.parameters, &screen)?;

        let (pipeline, ()) = Pipeline::new("player", source, |source, ctx| {
            let mut picture = ctx
                .branch()
                .pipe(SwDecoder::new("video-decoder", video.parameters.clone())?)
                .queue("video-frames", 32)
                .pipe(VideoSynchronizer::new("video-sync"));
            // After the synchronizer: a frame it drops for being late never
            // pays for the conversion.
            if let Some(to_drawable) = to_drawable {
                picture = picture.pipe(to_drawable);
            }
            ctx.attach(source, video.index, picture.to(screen)?)?;
            if let (Some(audio), Some((speakers, format))) = (audio, output) {
                let sound = ctx
                    .branch()
                    .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
                    .pipe(AudioResampler::new("audio-resampler", format))
                    .queue("audio-output", 8)
                    .to(speakers)?;
                ctx.attach(source, audio.index, sound)?;
            }
            Ok(())
        })?;
        Ok(Self {
            pipeline,
            window,
            events,
            duration,
            started: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            sought: Mutex::new(None),
        })
    }

    /// Starts playback, or resumes it after [`Self::pause`].
    pub fn play(&self) -> Result<(), PlayerError> {
        if self.paused.swap(false, Ordering::AcqRel) {
            self.pipeline.resume();
        }
        if !self.started.swap(true, Ordering::AcqRel) {
            self.pipeline.run()?;
        }
        Ok(())
    }

    /// Holds the picture and the sound where they are. Before [`Self::play`],
    /// playback starts paused on the first picture.
    pub fn pause(&self) {
        if !self.paused.swap(true, Ordering::AcqRel) {
            self.pipeline.pause();
        }
    }

    /// Whether it was last paused rather than played.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// Moves playback to `position` from the start, clamped to the file's
    /// length where the file says it, and shows the picture there — paused
    /// or playing, as it was. Returns once that picture is on its way to the
    /// screen. Only once playing has started: see [`Self::play`].
    pub fn seek(&self, position: Duration) -> Result<(), PlayerError> {
        let position = self.duration.map_or(position, |end| position.min(end));
        self.pipeline.seek(position, SeekMode::Accurate)?;
        *self
            .sought
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(position);
        Ok(())
    }

    /// Where playback is — `None` before the first picture is due. After
    /// a seek, where it went, until playback has moved on from there.
    pub fn position(&self) -> Option<Duration> {
        let mut sought = self
            .sought
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match self.pipeline.position() {
            Some(position) => {
                *sought = None;
                Some(position)
            }
            None => *sought,
        }
    }

    /// How long the file plays, where it says.
    pub fn duration(&self) -> Option<Duration> {
        self.duration
    }

    /// What changes the window: its title, whether it fills the screen.
    pub fn window_control(&self) -> WindowControl {
        self.window.clone()
    }

    /// Stops playback for good. The window stays until the player is
    /// dropped; [`Self::next_event`] returns `None` from here on.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        self.pipeline.stop();
    }

    /// Waits for the next thing to report, from the window or the playback.
    /// `None` once [`Self::stop`] has been called, or the window and the
    /// playback are both gone.
    pub fn next_event(&self) -> Option<PlayerEvent> {
        self.next_event_before(None)
    }

    /// [`Self::next_event`], waiting at most `timeout` — for a loop that has
    /// something of its own to do, such as showing where playback is. `None`
    /// when the time is up, as well.
    pub fn next_event_timeout(&self, timeout: Duration) -> Option<PlayerEvent> {
        self.next_event_before(Some(timeout))
    }

    fn next_event_before(&self, timeout: Option<Duration>) -> Option<PlayerEvent> {
        let window = &self.events.events;
        let bus = self.pipeline.bus().receiver();
        let deadline = timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout));
        loop {
            if self.stopped.load(Ordering::Acquire) {
                return None;
            }
            let wait = deadline
                .map(|deadline| deadline.saturating_duration_since(std::time::Instant::now()));
            let (event, message) = match wait {
                Some(wait) => select! {
                    recv(window) -> event => (Some(event), None),
                    recv(bus) -> message => (None, Some(message)),
                    default(wait) => return None,
                },
                None => select! {
                    recv(window) -> event => (Some(event), None),
                    recv(bus) -> message => (None, Some(message)),
                },
            };
            let message = match (event, message) {
                (Some(Ok(event)), _) => return Some(PlayerEvent::Window(event)),
                // The window is gone: only the playback is left to wait on.
                (Some(Err(_)), _) => match wait {
                    Some(wait) => bus.recv_timeout(wait).ok()?,
                    None => bus.recv().ok()?,
                },
                (None, Some(message)) => message.ok()?,
                (None, None) => return None,
            };
            match message.event {
                BusEvent::Finished => return Some(PlayerEvent::Ended),
                BusEvent::Error { name, error, .. } => {
                    return Some(PlayerEvent::Error { name, error });
                }
                // Everything else is the pipeline's own bookkeeping.
                _ => {}
            }
        }
    }

    /// Does what a player usually does with a window event, and says whether
    /// to go on: Space pauses and plays, F or a double click fills the screen
    /// and puts it back, the left and right arrows move five seconds, and
    /// Escape or closing the window answer `false` — the caller's to act on,
    /// by stopping or dropping the player. Anything else is left alone.
    pub fn respond_to(&self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::Closed | WindowEvent::Key(Key::Escape) => return false,
            WindowEvent::Key(Key::Space) => {
                if self.is_paused() {
                    let _ = self.play();
                } else {
                    self.pause();
                }
            }
            WindowEvent::Key(Key::Char('f'))
            | WindowEvent::DoubleClick {
                button: MouseButton::Left,
                ..
            } => {
                let _ = self.window.set_fullscreen(!self.window.is_fullscreen());
            }
            WindowEvent::Key(Key::Right) => {
                let at = self.position().unwrap_or_default();
                let _ = self.seek(at + ARROW_STEP);
            }
            WindowEvent::Key(Key::Left) => {
                let at = self.position().unwrap_or_default();
                let _ = self.seek(at.saturating_sub(ARROW_STEP));
            }
            _ => {}
        }
        true
    }
}

/// The default output device, opened, with the format it plays.
fn open_output() -> Result<(AudioOut, AudioFormat), PlayerError> {
    let audio = |error| PlayerError::Audio(Box::new(error));
    let device = AudioOut::list_devices()
        .map_err(audio)?
        .into_iter()
        .find(|device| device.is_default)
        .ok_or_else(|| PlayerError::Audio("no default output device".into()))?;
    AudioOut::open("speakers", AudioOutOptions { device }).map_err(audio)
}

/// A scaler to YUV420P where the software decode of `params` is in a
/// layout the window does not draw — a 10-bit or a 4:4:4 one — asked of the
/// window's own input contract.
fn to_drawable(
    params: &ffmpeg::codec::Parameters,
    screen: &VideoWindow,
) -> crate::Result<Option<SwScaler>> {
    let decoder = ffmpeg::codec::context::Context::from_parameters(params.clone())?
        .decoder()
        .video()?;
    let decoded = OutputContract::Fixed(
        PortContract::frame(MediaKind::VideoFrame, MemoryDomain::System)
            .with_layouts(PixelLayoutSet::of(PixelLayout::of(decoder.format()))),
    );
    if !check_link(&decoded, &screen.input_contract()).is_refused() {
        return Ok(None);
    }
    Ok(Some(SwScaler::new(
        "to-yuv420p",
        ffmpeg::format::Pixel::YUV420P,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::Flags::BILINEAR,
    )))
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::test_support::try_test_video;

    fn wait_until(what: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !what() {
            if Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        true
    }

    fn open(audio: bool) -> Option<Player> {
        let path = try_test_video()?;
        let options = PlayerOptions {
            window: WindowOptions {
                title: "media-pp player test".into(),
                width: 320,
                height: 240,
            },
            audio,
        };
        match Player::open(path, options) {
            Ok(player) => Some(player),
            Err(error @ (PlayerError::Window(_) | PlayerError::Audio(_))) => {
                eprintln!("skipping: {error}");
                None
            }
            Err(error) => panic!("the fixture did not open: {error}"),
        }
    }

    /// Plays, holds still while paused, moves where it is sent, and ends at
    /// the end — with its sound, where there is an output to play it on.
    #[test]
    fn a_file_plays_pauses_seeks_and_ends() {
        let Some(player) = open(true).or_else(|| open(false)) else {
            return;
        };
        let length = player.duration().expect("the fixture says how long it is");
        player.play().unwrap();
        assert!(
            wait_until(|| player.position() > Some(Duration::from_millis(300))),
            "playback moves"
        );

        player.pause();
        assert!(player.is_paused());
        let held = player.position();
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(player.position(), held, "paused holds still");

        let near_end = length.saturating_sub(Duration::from_millis(700));
        player.seek(near_end).unwrap();
        let landed = player.position().expect("a position after seeking");
        assert!(
            landed.abs_diff(near_end) < Duration::from_millis(200),
            "sought to {near_end:?}, at {landed:?}"
        );
        player.play().unwrap();
        loop {
            match player.next_event_timeout(Duration::from_secs(10)) {
                Some(PlayerEvent::Ended) => break,
                Some(PlayerEvent::Error { name, error }) => panic!("{name}: {error}"),
                Some(_) => {}
                None => panic!("the file never ended"),
            }
        }
        player.stop();
        assert!(player.next_event().is_none(), "nothing after stopping");
    }

    /// The usual keys do what a player's do, and a close asks to stop.
    #[test]
    fn the_usual_keys_do_the_usual_things() {
        let Some(player) = open(false) else { return };
        player.play().unwrap();
        assert!(player.respond_to(&WindowEvent::Key(Key::Space)));
        assert!(player.is_paused());
        assert!(player.respond_to(&WindowEvent::Key(Key::Space)));
        assert!(!player.is_paused());
        assert!(player.respond_to(&WindowEvent::Key(Key::Char('f'))));
        assert!(player.window_control().is_fullscreen());
        assert!(player.respond_to(&WindowEvent::Key(Key::Char('f'))));
        assert!(!player.window_control().is_fullscreen());
        assert!(!player.respond_to(&WindowEvent::Key(Key::Escape)));
        assert!(!player.respond_to(&WindowEvent::Closed));
    }

    #[test]
    fn a_seek_before_playing_is_refused() {
        let Some(player) = open(false) else { return };
        assert!(matches!(
            player.seek(Duration::from_secs(1)),
            Err(PlayerError::Pipeline(_))
        ));
    }
}
