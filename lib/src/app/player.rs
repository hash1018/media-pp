//! A file played in a window with its sound — the GStreamer `playbin` of
//! this crate.
//!
//! Everything here is built from the crate's own elements, and a program
//! that outgrows it builds the same graph itself: `FileDemuxer`, a
//! `VideoDecodeBin` onto the window's GPU, a `VideoSynchronizer` in front of
//! a [`VideoWindow`], and a `SwDecoder` and an `AudioResampler` in front of
//! the platform's audio renderer. What it saves is the wiring, the choice of devices, and
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
        AudioFormat, AudioResampler, AudioVolume, AudioVolumeError, AudioVolumeHandle, DecodePath,
        DecodeTarget, DecodeThreadKind, DecodeThreading, FileDemuxer, FileDemuxerError,
        FileDemuxerHandle, Key, MouseButton, SoftwareReason, StreamInfo, SwDecoder, SwScaler,
        VideoDecodeBin, VideoDecodeBinHandle, VideoSynchronizer, VideoWindow, VideoWindowError,
        WindowControl, WindowEvent, WindowEvents, WindowOptions,
    },
    ffmpeg,
    pipeline::{Pipeline, SeekMode},
};

/// How far [`Player::respond_to`] moves on an arrow key.
const ARROW_STEP: Duration = Duration::from_secs(5);
/// How far [`Player::respond_to`] turns the volume on an arrow key.
const VOLUME_STEP: f32 = 0.1;

/// How many decoded pictures wait between the decoder and the synchronizer
/// when they are on the GPU: enough to ride out a slow packet, few enough
/// that a hardware decoder's fixed surface pool — NVDEC's is capped at 32 —
/// holds them with its own references.
const GPU_FRAMES_QUEUED: usize = 8;
/// The same, for pictures in system memory, whose pool grows.
const SYSTEM_FRAMES_QUEUED: usize = 32;

/// How a [`Player`] opens its window and its sound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerOptions {
    /// The window the picture is shown in.
    pub window: WindowOptions,
    /// Whether to play the file's sound, on the default output device. A
    /// file without sound plays its picture either way; with this set and
    /// no output device, [`Player::open`] fails rather than playing silently.
    pub audio: bool,
    /// Which of the file's sound tracks to play, by its stream index —
    /// [`StreamInfo::index`](crate::elements::StreamInfo::index), as
    /// `FileDemuxer::open` lists them — or `None` for the one the file
    /// marks as its main one. Only read where `audio` is set.
    pub audio_stream: Option<usize>,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        Self {
            window: WindowOptions::default(),
            audio: true,
            audio_stream: None,
        }
    }
}

/// Why a [`Player`] could not open a file or do what it was asked.
#[derive(Debug, ThisError)]
pub enum PlayerError {
    /// The file could not be opened, or has nothing to play: no picture,
    /// and no sound or none asked for.
    #[error(transparent)]
    Open(#[from] FileDemuxerError),
    /// [`PlayerOptions::audio_stream`] names no sound track of the file.
    #[error("the file has no sound track at stream {0}")]
    NoSuchAudioStream(usize),
    /// A volume that is not a finite, non-negative gain.
    #[error(transparent)]
    Volume(#[from] AudioVolumeError),
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
    /// stays in the window until the player is stopped or dropped, and a
    /// [`Player::seek`] from there plays on from where it lands.
    Ended,
    /// Playback is over for good: [`Player::stop`] was called, or the window
    /// and the playback are gone. Only [`Player::next_event_timeout`]
    /// returns it — every call from then on — where [`Player::next_event`]
    /// returns `None`.
    Stopped,
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

/// What a wait for the next event came to.
enum Wait {
    Event(PlayerEvent),
    Timeout,
    Gone,
}

/// A file played in a window of its own, with its sound — open, play, and
/// wait for [`PlayerEvent`]s:
///
/// ```no_run
/// use media_pp::player::{Player, PlayerEvent, PlayerOptions};
///
/// # fn main() -> media_pp::Result<()> {
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
/// # Ok(())
/// # }
/// ```
///
/// The picture is decoded on the GPU the [`VideoWindow`] draws with, and
/// stays there: D3D11VA on Windows (D3D12 in a build with only `d3d12`),
/// NVDEC on Linux where the build has `cuda` and the machine an NVIDIA GPU —
/// in software and uploaded where that GPU does not take the stream, and in
/// software straight into the window where there is no such GPU at all.
/// [`Self::decoding`] says which. The
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
    /// `None` for a file with no picture.
    decoder: Option<VideoDecodeBinHandle>,
    volume: AudioVolumeHandle,
    /// Where the file's timeline has been carried to by looping, which
    /// [`Self::position`] takes back off.
    laps: FileDemuxerHandle,
    /// The window of a file with no picture, which no pipeline holds —
    /// kept here so it stays open as long as the player.
    _idle: Option<VideoWindow>,
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
        let (source, streams) = FileDemuxer::open("file", path)?;
        let duration = source.duration();
        let laps = source.looping_handle();
        let video = source.best(ffmpeg::media::Type::Video);
        let audio = match (options.audio, options.audio_stream) {
            (false, _) => None,
            (true, None) => source.best(ffmpeg::media::Type::Audio).ok(),
            (true, Some(index)) => Some(
                streams
                    .into_iter()
                    .find(|stream| {
                        stream.index == index && stream.kind == ffmpeg::media::Type::Audio
                    })
                    .ok_or(PlayerError::NoSuchAudioStream(index))?,
            ),
        };
        // A file with nothing to play is refused as one with no picture, as
        // it always was; a file with only sound plays it.
        let video = match (video, &audio) {
            (Ok(video), _) => Some(video),
            (Err(_), Some(_)) => None,
            (Err(error), None) => return Err(error.into()),
        };
        let output = audio.as_ref().map(|_| open_output()).transpose()?;
        let (volume, volume_handle) = AudioVolume::new("volume");
        // What a hardware decoder's pool has to cover after the queue: the
        // picture the synchronizer waits on, the one on screen, and one the
        // queue may park while a pause or seek goes by.
        let budget = GPU_FRAMES_QUEUED as i32 + 3;
        let Some(video) = video else {
            // Sound alone: still a window, since it is where the keys arrive
            // and what closing means stop. It shows black.
            let (mut screen, events) = VideoWindow::open("screen", options.window)?;
            screen.consume(black())?;
            let window = screen.window_control();
            let (pipeline, ()) = Pipeline::new("player", source, |source, ctx| {
                if let (Some(audio), Some((speakers, format))) = (audio, output) {
                    ctx.attach(
                        source,
                        audio.index,
                        sound(ctx, &audio, format, volume, speakers)?,
                    )?;
                }
                Ok(())
            })?;
            return Ok(Self {
                pipeline,
                decoder: None,
                volume: volume_handle,
                laps,
                _idle: Some(screen),
                window,
                events,
                duration,
                started: AtomicBool::new(false),
                paused: AtomicBool::new(false),
                stopped: AtomicBool::new(false),
                sought: Mutex::new(None),
            });
        };
        let (screen, events, target) =
            VideoWindow::open_for_decoding("screen", options.window, budget)?;
        let window = screen.window_control();
        // A file has no deadline, so a software decode takes every thread,
        // several pictures at once: one thread holds 4K60 HEVC well short of
        // its rate, and once the picture falls behind the demuxer can read no
        // further and the sound runs dry with it.
        let threading = DecodeThreading {
            threads: None,
            kind: DecodeThreadKind::Frame,
        };
        let decode = |target| {
            VideoDecodeBin::open("video", video.parameters.clone(), target, Some(threading))
        };
        let decoder = match decode(target) {
            Ok(decoder) => decoder,
            // A GPU the window draws with but that cannot decode or take an
            // upload — a software adapter's — still draws what is decoded
            // in software.
            Err(_) => decode(DecodeTarget::System)?,
        };
        let on_gpu = decoder.path() != DecodePath::Software(SoftwareReason::SystemMemory);
        let queued = if on_gpu {
            GPU_FRAMES_QUEUED
        } else {
            SYSTEM_FRAMES_QUEUED
        };
        let to_drawable = if on_gpu {
            None
        } else {
            to_drawable(&video.parameters, &screen)?
        };
        let handle = decoder.handle();

        let (pipeline, ()) = Pipeline::new("player", source, |source, ctx| {
            let mut picture = ctx
                .branch()
                .pipe(decoder)
                .queue("video-frames", queued)
                .pipe(VideoSynchronizer::new("video-sync"));
            // After the synchronizer: a frame it drops for being late never
            // pays for the conversion.
            if let Some(to_drawable) = to_drawable {
                picture = picture.pipe(to_drawable);
            }
            ctx.attach(source, video.index, picture.to(screen)?)?;
            if let (Some(audio), Some((speakers, format))) = (audio, output) {
                ctx.attach(
                    source,
                    audio.index,
                    sound(ctx, &audio, format, volume, speakers)?,
                )?;
            }
            Ok(())
        })?;
        Ok(Self {
            pipeline,
            decoder: Some(handle),
            volume: volume_handle,
            laps,
            _idle: None,
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

    /// Where playback is in the file — `None` before the first picture is
    /// due. After a seek, where it went, until playback has moved on from
    /// there. While looping, where it is in the lap it is on.
    pub fn position(&self) -> Option<Duration> {
        let mut sought = self
            .sought
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match self.pipeline.position() {
            Some(position) => {
                *sought = None;
                Some(self.in_lap(position))
            }
            None => *sought,
        }
    }

    /// `position` on the pipeline's timeline, which looping carries on past
    /// the end of the file, as a position in the file. The demuxer moves
    /// on to the next lap a queue's depth before playback reaches it, so a
    /// position short of the latest lap's start is still in the lap before.
    fn in_lap(&self, position: Duration) -> Duration {
        let lap = self.laps.lap_offset();
        match position.checked_sub(lap) {
            Some(in_lap) => in_lap,
            None => {
                let previous = self
                    .duration
                    .map_or(Duration::ZERO, |length| lap.saturating_sub(length));
                position.saturating_sub(previous)
            }
        }
    }

    /// Where the picture is decoded: on the window's GPU, by its hardware
    /// or in software and uploaded, or in software into system memory — and
    /// why, where it is software. `None` for a file with no picture.
    pub fn decoding(&self) -> Option<DecodePath> {
        self.decoder.as_ref().map(VideoDecodeBinHandle::path)
    }

    /// Sets the volume of the sound, as a linear gain: `1.0` as the file has
    /// it, `0.5` half, `0.0` silent — above `1.0` louder, and clipping as it
    /// gets there. A change takes a moment rather than a click.
    pub fn set_volume(&self, volume: f32) -> Result<(), PlayerError> {
        self.volume.set_gain(volume)?;
        Ok(())
    }

    /// The volume last set, `1.0` to begin with — whether muted or not.
    pub fn volume(&self) -> f32 {
        self.volume.gain()
    }

    /// Silences the sound, or brings it back at the volume it had.
    pub fn set_muted(&self, muted: bool) {
        self.volume.set_muted(muted);
    }

    /// Whether the sound is muted.
    pub fn is_muted(&self) -> bool {
        self.volume.is_muted()
    }

    /// Plays the file again from the start each time it ends, instead of
    /// ending. Switched while it plays: turned off, the lap already under
    /// way plays out and then the file ends.
    pub fn set_looping(&self, looping: bool) {
        self.laps.set_looping(looping);
    }

    /// Whether it is set to loop.
    pub fn is_looping(&self) -> bool {
        self.laps.is_looping()
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
    /// playback are both gone — so `while let Some(event) = player.next_event()`
    /// ends when there is nothing left to report.
    pub fn next_event(&self) -> Option<PlayerEvent> {
        match self.wait(None) {
            Wait::Event(event) => Some(event),
            Wait::Timeout | Wait::Gone => None,
        }
    }

    /// [`Self::next_event`], waiting at most `timeout` — for a loop that has
    /// something of its own to do, such as showing where playback is.
    /// `None` means only that the time is up; once [`Self::stop`] has been
    /// called or the player's window and playback are gone, every call
    /// returns [`PlayerEvent::Stopped`] at once, so the loop can tell the two
    /// apart rather than spinning.
    pub fn next_event_timeout(&self, timeout: Duration) -> Option<PlayerEvent> {
        match self.wait(Some(timeout)) {
            Wait::Event(event) => Some(event),
            Wait::Timeout => None,
            Wait::Gone => Some(PlayerEvent::Stopped),
        }
    }

    fn wait(&self, timeout: Option<Duration>) -> Wait {
        let window = &self.events.events;
        let bus = self.pipeline.bus().receiver();
        let deadline = timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout));
        loop {
            if self.stopped.load(Ordering::Acquire) {
                return Wait::Gone;
            }
            let wait = deadline
                .map(|deadline| deadline.saturating_duration_since(std::time::Instant::now()));
            let (event, message) = match wait {
                Some(wait) => select! {
                    recv(window) -> event => (Some(event), None),
                    recv(bus) -> message => (None, Some(message)),
                    default(wait) => return Wait::Timeout,
                },
                None => select! {
                    recv(window) -> event => (Some(event), None),
                    recv(bus) -> message => (None, Some(message)),
                },
            };
            let message = match (event, message) {
                (Some(Ok(event)), _) => return Wait::Event(PlayerEvent::Window(event)),
                // The window is gone: only the playback is left to wait on.
                (Some(Err(_)), _) => match wait {
                    Some(wait) => match bus.recv_timeout(wait) {
                        Ok(message) => message,
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                            return Wait::Timeout;
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                            return Wait::Gone;
                        }
                    },
                    None => match bus.recv() {
                        Ok(message) => message,
                        Err(_) => return Wait::Gone,
                    },
                },
                (None, Some(Ok(message))) => message,
                (None, Some(Err(_))) | (None, None) => return Wait::Gone,
            };
            match message.event {
                BusEvent::Finished => return Wait::Event(PlayerEvent::Ended),
                BusEvent::Error { name, error, .. } => {
                    return Wait::Event(PlayerEvent::Error { name, error });
                }
                // Everything else is the pipeline's own bookkeeping.
                _ => {}
            }
        }
    }

    /// Does what a player usually does with a window event, and says whether
    /// to go on: Space pauses and plays, F or a double click fills the screen
    /// and puts it back, the left and right arrows move five seconds, the up
    /// and down arrows turn the volume up and down a tenth, M mutes and
    /// unmutes, and Escape or closing the window answer `false` — the
    /// caller's to act on, by stopping or dropping the player. Anything else
    /// is left alone.
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
            WindowEvent::Key(Key::Up) => {
                let _ = self.set_volume((self.volume() + VOLUME_STEP).min(1.0));
            }
            WindowEvent::Key(Key::Down) => {
                let _ = self.set_volume((self.volume() - VOLUME_STEP).max(0.0));
            }
            WindowEvent::Key(Key::Char('m')) => self.set_muted(!self.is_muted()),
            _ => {}
        }
        true
    }
}

/// The sound branch: decoded, brought to the output's format, turned to
/// the player's volume — after the resampler, which makes it the format the
/// output plays — and queued for the output.
fn sound(
    ctx: &Arc<crate::element::Context>,
    audio: &StreamInfo,
    format: AudioFormat,
    volume: AudioVolume,
    speakers: AudioOut,
) -> crate::Result<crate::pipeline::DetachedBranch> {
    ctx.branch()
        .pipe(SwDecoder::new("audio-decoder", audio.parameters.clone())?)
        .pipe(AudioResampler::new("audio-resampler", format))
        .pipe(volume)
        .queue("audio-output", 8)
        .to(speakers)
}

/// One black picture, for a window with nothing else to show.
fn black() -> crate::buffer::MediaBuffer {
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, 2, 2);
    for pixel in frame.data_mut(0).as_chunks_mut::<4>().0 {
        *pixel = [0, 0, 0, 255];
    }
    crate::buffer::MediaBuffer::video(frame)
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
            audio_stream: None,
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
        let played_to_the_end = || loop {
            match player.next_event_timeout(Duration::from_secs(10)) {
                Some(PlayerEvent::Ended) => break,
                Some(PlayerEvent::Error { name, error }) => panic!("{name}: {error}"),
                Some(PlayerEvent::Stopped) => {
                    panic!(
                        "playback stopped without ending, at {:?}",
                        player.position()
                    )
                }
                Some(_) => {}
                None => panic!("the file never ended"),
            }
        };
        played_to_the_end();
        // And from the end, back into the file: it plays on and ends again.
        player
            .seek(length.saturating_sub(Duration::from_millis(500)))
            .expect("a player that has ended can still be sought");
        played_to_the_end();
        player.stop();
        assert!(player.next_event().is_none(), "nothing after stopping");
        assert!(
            matches!(
                player.next_event_timeout(Duration::from_secs(5)),
                Some(PlayerEvent::Stopped)
            ),
            "a timed wait says it is over, at once, rather than timing out"
        );
    }

    /// Where the window's GPU takes the stream, the picture is decoded onto
    /// it rather than into system memory.
    #[cfg(feature = "d3d11")]
    #[test]
    fn the_picture_is_decoded_on_the_windows_gpu_where_it_can_be() {
        let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
            return;
        };
        let Some(path) = try_test_video() else { return };
        let (source, _) = FileDemuxer::open("file", &path).unwrap();
        let video = source.best(ffmpeg::media::Type::Video).unwrap();
        let target = DecodeTarget::D3d11 {
            gpu,
            downstream_hw_frames: GPU_FRAMES_QUEUED as i32 + 3,
        };
        if let Err(error) = VideoDecodeBin::open("probe", video.parameters, target, None) {
            eprintln!("skipping: this GPU does not take the stream ({error})");
            return;
        }
        let Some(player) = open(false) else { return };
        assert_ne!(
            player.decoding(),
            Some(DecodePath::Software(SoftwareReason::SystemMemory))
        );
        player.play().unwrap();
        assert!(
            wait_until(|| player.position() > Some(Duration::from_millis(300))),
            "playback moves"
        );
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
        assert_eq!(player.volume(), 1.0);
        assert!(player.respond_to(&WindowEvent::Key(Key::Down)));
        assert!((player.volume() - 0.9).abs() < 1e-6, "{}", player.volume());
        assert!(player.respond_to(&WindowEvent::Key(Key::Up)));
        assert!(player.respond_to(&WindowEvent::Key(Key::Up)));
        assert_eq!(player.volume(), 1.0, "the arrows stop at the file's own");
        assert!(player.respond_to(&WindowEvent::Key(Key::Char('m'))));
        assert!(player.is_muted());
        assert!(player.respond_to(&WindowEvent::Key(Key::Char('m'))));
        assert!(!player.is_muted());
        assert!(!player.respond_to(&WindowEvent::Key(Key::Escape)));
        assert!(!player.respond_to(&WindowEvent::Closed));
    }

    /// A volume that is no gain is refused, and leaves the one set before.
    #[test]
    fn a_volume_that_is_no_gain_is_refused() {
        let Some(player) = open(false) else { return };
        player.set_volume(0.25).unwrap();
        for wrong in [-0.5, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                player.set_volume(wrong),
                Err(PlayerError::Volume(_))
            ));
        }
        assert_eq!(player.volume(), 0.25);
    }

    /// A file with sound and no picture plays, in a window that shows
    /// black, to its end.
    #[test]
    fn a_file_with_only_sound_plays_to_its_end() {
        let Some(path) = crate::test_support::try_test_sound() else {
            return;
        };
        let options = PlayerOptions {
            window: WindowOptions {
                title: "media-pp player test, sound only".into(),
                width: 320,
                height: 240,
            },
            ..PlayerOptions::default()
        };
        let player = match Player::open(path, options) {
            Ok(player) => player,
            Err(error @ (PlayerError::Window(_) | PlayerError::Audio(_))) => {
                eprintln!("skipping: {error}");
                return;
            }
            Err(error) => panic!("a file with sound opens: {error}"),
        };
        assert_eq!(player.decoding(), None, "there is no picture to decode");
        player.play().unwrap();
        loop {
            match player.next_event_timeout(Duration::from_secs(10)) {
                Some(PlayerEvent::Ended) => break,
                Some(PlayerEvent::Error { name, error }) => panic!("{name}: {error}"),
                Some(PlayerEvent::Stopped) => panic!("stopped without ending"),
                Some(_) => {}
                None => panic!("the sound never ended"),
            }
        }
    }

    /// A sound track is chosen by its stream, and a stream that is not one
    /// is refused before anything is opened for it.
    #[test]
    fn a_sound_track_is_chosen_by_its_stream() {
        let Some(path) = try_test_video() else { return };
        let (_, streams) = FileDemuxer::open("probe", &path).unwrap();
        let index_of = |kind| {
            streams
                .iter()
                .find(|stream| stream.kind == kind)
                .map(|stream| stream.index)
                .unwrap()
        };
        let options = |audio_stream| PlayerOptions {
            window: WindowOptions {
                title: "media-pp player test, a chosen track".into(),
                width: 320,
                height: 240,
            },
            audio: true,
            audio_stream: Some(audio_stream),
        };
        let picture = index_of(ffmpeg::media::Type::Video);
        assert!(matches!(
            Player::open(&path, options(picture)),
            Err(PlayerError::NoSuchAudioStream(index)) if index == picture
        ));
        match Player::open(&path, options(index_of(ffmpeg::media::Type::Audio))) {
            Ok(_) => {}
            Err(error @ (PlayerError::Window(_) | PlayerError::Audio(_))) => {
                eprintln!("skipping the track that opens: {error}");
            }
            Err(error) => panic!("the file's own sound track opens: {error}"),
        }
    }

    /// Looping, the end starts the file again rather than ending it, and
    /// where playback is reads as a place in the file; turned off, the lap
    /// under way plays out and the file ends.
    #[test]
    fn looping_plays_the_file_again_instead_of_ending() {
        let Some(player) = open(false) else { return };
        let length = player.duration().expect("the fixture says how long it is");
        player.set_looping(true);
        assert!(player.is_looping());
        player.play().unwrap();
        player
            .seek(length.saturating_sub(Duration::from_millis(500)))
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(5);
        let mut wrapped = false;
        while Instant::now() < deadline && !wrapped {
            match player.next_event_timeout(Duration::from_millis(50)) {
                Some(PlayerEvent::Ended) => panic!("a looping file ended"),
                Some(PlayerEvent::Error { name, error }) => panic!("{name}: {error}"),
                _ => {}
            }
            let at = player.position().unwrap_or_default();
            assert!(
                at <= length + Duration::from_millis(100),
                "{at:?} is past the file"
            );
            wrapped = at < Duration::from_secs(2);
        }
        assert!(wrapped, "playback came round to the start");

        player.set_looping(false);
        loop {
            match player.next_event_timeout(Duration::from_secs(12)) {
                Some(PlayerEvent::Ended) => break,
                Some(PlayerEvent::Error { name, error }) => panic!("{name}: {error}"),
                Some(PlayerEvent::Stopped) => panic!("stopped without ending"),
                Some(_) => {}
                None => panic!("the lap under way never ended"),
            }
        }
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
