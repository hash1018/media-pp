//! A video file played in a window with its sound, through `Player` — the
//! whole of a player in one type: `FileDemuxer`, the picture decoded on the
//! window's GPU where it can be and the sound in software, the picture
//! synchronized to the sound played, a `VideoWindow` and the platform's
//! audio output, built by `Player::open`. Where the picture is decoded is
//! printed when it starts.
//!
//! A file with only sound plays too, its waveform in the window.
//!
//! Space pauses and plays, the left and right arrows move five seconds, the
//! up and down arrows turn the volume, M mutes, F or a double click fills
//! the screen, Escape or closing the window stops; the title shows where
//! playback is. One program for Windows and Linux.
//!
//!     cargo run -p player -- path/to/video.mp4

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("{} supports Windows and Linux only", env!("CARGO_PKG_NAME"));
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn main() -> impl std::process::Termination {
    example::run()
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod example {
    use std::time::Duration;

    use media_pp::{
        elements::WindowOptions,
        player::{Player, PlayerEvent, PlayerOptions},
    };

    const TITLE: &str = "media-pp player";

    pub(super) fn run() -> media_pp::Result<()> {
        let Some(path) = std::env::args().nth(1) else {
            eprintln!("usage: player <video>");
            std::process::exit(1);
        };
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let player = Player::open(
            &path,
            PlayerOptions {
                window: WindowOptions {
                    title: TITLE.into(),
                    ..WindowOptions::default()
                },
                ..PlayerOptions::default()
            },
        )?;
        let window = player.window_control();
        match player.decoding() {
            Some(path) => println!("decoding: {path:?}"),
            None => println!("sound only"),
        }
        player.play()?;

        let mut shown = None;
        loop {
            // A quarter of a second at most between looks at the position,
            // so the title keeps up with playback.
            match player.next_event_timeout(Duration::from_millis(250)) {
                Some(PlayerEvent::Window(event)) => {
                    if !player.respond_to(&event) {
                        break;
                    }
                }
                Some(PlayerEvent::Ended | PlayerEvent::Stopped) => break,
                Some(PlayerEvent::Error { name, error }) => eprintln!("[{name}] {error}"),
                _ => {}
            }
            let seconds = player.position().map(|position| position.as_secs());
            if seconds != shown {
                shown = seconds;
                let _ = window.set_title(&format!(
                    "{TITLE} — {} / {}",
                    clock(player.position()),
                    clock(player.duration())
                ));
            }
        }
        Ok(())
    }

    fn clock(time: Option<Duration>) -> String {
        match time {
            Some(time) => format!("{}:{:02}", time.as_secs() / 60, time.as_secs() % 60),
            None => "-:--".into(),
        }
    }
}
