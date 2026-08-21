//! The terminal command loop and the timestamp parsing this example's shell
//! adds on top of [`render_common::run_window`], which owns the window itself.

use std::{
    io::{self, BufRead},
    sync::Arc,
    time::Duration,
};

use media_pp::{elements::TeeHandle, graph::BranchId, pipeline::Pipeline};

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
