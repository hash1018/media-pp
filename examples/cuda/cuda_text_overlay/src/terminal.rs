//! Arrow keys and `q`, delivered as [`TerminalCommand`]s over a channel.
//!
//! This is the one genuinely per-OS part of the example: a single keypress has
//! to arrive without Enter, which is `ReadConsoleInputW` on Windows and a
//! raw-mode termios terminal on Unix. A redirected or absent stdin simply
//! yields no commands, and the recording still follows its requested duration.

use std::sync::mpsc;

pub const MOVE_STEP: i32 = 10;

pub enum TerminalCommand {
    Move { dx: i32, dy: i32 },
    Quit,
}

#[cfg(windows)]
pub fn commands() -> mpsc::Receiver<TerminalCommand> {
    use std::thread;

    use windows::Win32::System::Console::{
        GetStdHandle, INPUT_RECORD, KEY_EVENT, ReadConsoleInputW, STD_INPUT_HANDLE,
    };

    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let input = match unsafe { GetStdHandle(STD_INPUT_HANDLE) } {
            Ok(input) => input,
            Err(error) => {
                eprintln!("terminal controls unavailable: {error}");
                return;
            }
        };

        loop {
            let mut record = INPUT_RECORD::default();
            let mut read = 0;
            if let Err(error) =
                unsafe { ReadConsoleInputW(input, std::slice::from_mut(&mut record), &mut read) }
            {
                // stdin may be redirected or detached in CI. Recording still
                // follows its requested duration; only live controls are absent.
                let _ = error;
                return;
            }
            if read == 0 || u32::from(record.EventType) != KEY_EVENT {
                continue;
            }

            // `EventType == KEY_EVENT` makes the matching union field active.
            let key = unsafe { record.Event.KeyEvent };
            if !key.bKeyDown.as_bool() {
                continue;
            }

            let command = match key.wVirtualKeyCode {
                0x25 => Some(TerminalCommand::Move {
                    dx: -MOVE_STEP,
                    dy: 0,
                }),
                0x26 => Some(TerminalCommand::Move {
                    dx: 0,
                    dy: -MOVE_STEP,
                }),
                0x27 => Some(TerminalCommand::Move {
                    dx: MOVE_STEP,
                    dy: 0,
                }),
                0x28 => Some(TerminalCommand::Move {
                    dx: 0,
                    dy: MOVE_STEP,
                }),
                0x51 => Some(TerminalCommand::Quit),
                _ => None,
            };

            if command.is_some_and(|command| sender.send(command).is_err()) {
                return;
            }
        }
    });
    receiver
}

#[cfg(unix)]
pub fn commands() -> mpsc::Receiver<TerminalCommand> {
    use std::{io::Read, thread};

    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let Some(restore) = raw_mode() else {
            eprintln!("terminal controls unavailable: stdin is not a terminal");
            return;
        };
        let mut stdin = std::io::stdin();
        let mut buffer = [0u8; 3];
        while let Ok(read) = stdin.read(&mut buffer[..1]) {
            if read == 0 {
                break;
            }
            let command = match buffer[0] {
                b'q' | b'Q' => Some(TerminalCommand::Quit),
                0x1b => {
                    // CSI sequence: ESC '[' followed by one final byte.
                    if stdin.read(&mut buffer[1..3]).unwrap_or(0) < 2 || buffer[1] != b'[' {
                        None
                    } else {
                        match buffer[2] {
                            b'A' => Some(TerminalCommand::Move {
                                dx: 0,
                                dy: -MOVE_STEP,
                            }),
                            b'B' => Some(TerminalCommand::Move {
                                dx: 0,
                                dy: MOVE_STEP,
                            }),
                            b'C' => Some(TerminalCommand::Move {
                                dx: MOVE_STEP,
                                dy: 0,
                            }),
                            b'D' => Some(TerminalCommand::Move {
                                dx: -MOVE_STEP,
                                dy: 0,
                            }),
                            _ => None,
                        }
                    }
                }
                _ => None,
            };
            if let Some(command) = command
                && sender.send(command).is_err()
            {
                break;
            }
        }
        restore();
    });
    receiver
}

/// Puts stdin in raw mode, returning what restores it. `None` when stdin is
/// not a terminal at all.
#[cfg(unix)]
fn raw_mode() -> Option<impl FnOnce()> {
    use std::os::fd::AsRawFd;

    let fd = std::io::stdin().as_raw_fd();
    let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
    if unsafe { libc::tcgetattr(fd, &mut original) } != 0 {
        return None;
    }
    let mut raw = original;
    unsafe { libc::cfmakeraw(&mut raw) };
    // Leave output processing alone: this example keeps printing lines, and
    // full raw mode would strip their carriage returns.
    raw.c_oflag = original.c_oflag;
    if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
        return None;
    }
    Some(move || {
        unsafe { libc::tcsetattr(fd, libc::TCSANOW, &original) };
    })
}
