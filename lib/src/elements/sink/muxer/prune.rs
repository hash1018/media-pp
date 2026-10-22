//! A live DASH manifest's old segments deleted by this crate where the
//! FFmpeg linked cannot: [`SegmentPruner`].
//!
//! FFmpeg's DASH muxer deletes through its file protocol, which deletes only
//! where FFmpeg was built with `unistd.h`. One built with MSVC — as vcpkg
//! builds it for Windows — answers every delete with `ENOSYS`, so the
//! manifest moves on and its segments stay, until the disk is full. (Its
//! HLS muxer calls `unlink` itself, and deletes everywhere.) A pruner keeps
//! one segment more than FFmpeg would, so that where FFmpeg deletes, it
//! finds nothing to do.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How often a pruner looks: a segment is a second or more, and every sound
/// packet is a keyframe. Every so many keyframes as well, for an output
/// written faster than it plays, whose seconds go by in a few keyframes'
/// time.
const EVERY: Duration = Duration::from_millis(500);
const EVERY_KEYFRAMES: u32 = 50;

/// Deletes all but the newest segments of each sequence in a directory.
///
/// A segment is a file named `<prefix><key><number><suffix>`: the number
/// is the last run of digits before the suffix, and what comes between the
/// prefix and it names the sequence — a DASH representation's
/// `chunk-stream0-` and `chunk-stream1-` are two, an HLS playlist's
/// `segment_` one.
pub(super) struct SegmentPruner {
    dir: PathBuf,
    prefix: String,
    suffix: String,
    keep: usize,
    /// When it last looked, and how many keyframes it was asked at since.
    looked: Mutex<(Option<Instant>, u32)>,
}

impl SegmentPruner {
    pub(super) fn new(
        dir: impl Into<PathBuf>,
        prefix: impl Into<String>,
        suffix: impl Into<String>,
        keep: usize,
    ) -> Self {
        Self {
            dir: dir.into(),
            prefix: prefix.into(),
            suffix: suffix.into(),
            keep: keep.max(1),
            looked: Mutex::new((None, 0)),
        }
    }

    /// The directory a path's file is in: the working directory for a bare
    /// file name.
    pub(super) fn dir_of(path: &Path) -> PathBuf {
        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        }
    }

    /// Deletes what is older than the newest `keep` of each sequence, after
    /// a keyframe is written: at most twice a second, or once in
    /// [`EVERY_KEYFRAMES`] where they come faster. What cannot be read or
    /// deleted is left; a later call tries again.
    pub(super) fn prune(&self) {
        {
            let mut looked = self.looked.lock().unwrap_or_else(|p| p.into_inner());
            let (at, since) = &mut *looked;
            *since += 1;
            if at.is_some_and(|at| at.elapsed() < EVERY) && *since < EVERY_KEYFRAMES {
                return;
            }
            *looked = (Some(Instant::now()), 0);
        }
        self.prune_now();
    }

    /// Deletes what is older than the newest `keep` of each sequence now:
    /// as the output is finished, whatever the last look was.
    pub(super) fn prune_now(&self) {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        let mut sequences: std::collections::HashMap<String, Vec<(u64, PathBuf)>> =
            std::collections::HashMap::new();
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some((key, number)) = name.to_str().and_then(|name| self.parse(name)) else {
                continue;
            };
            sequences
                .entry(key)
                .or_default()
                .push((number, entry.path()));
        }
        for mut segments in sequences.into_values() {
            if segments.len() <= self.keep {
                continue;
            }
            segments.sort_unstable_by_key(|(number, _)| *number);
            let old = segments.len() - self.keep;
            for (_, path) in segments.into_iter().take(old) {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    /// A segment's sequence and number, where `name` is one of ours.
    fn parse(&self, name: &str) -> Option<(String, u64)> {
        let middle = name
            .strip_prefix(self.prefix.as_str())?
            .strip_suffix(self.suffix.as_str())?;
        let key = middle.trim_end_matches(|c: char| c.is_ascii_digit());
        let number = middle[key.len()..].parse().ok()?;
        Some((key.to_owned(), number))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_of_each_sequence_are_kept_and_nothing_else_is_touched() {
        let dir = std::env::temp_dir().join(format!(
            "media_pp_prune_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let names = [
            "chunk-stream0-00001.m4s",
            "chunk-stream0-00002.m4s",
            "chunk-stream0-00010.m4s",
            "chunk-stream0-00011.m4s",
            "chunk-stream1-00001.m4s",
            "chunk-stream1-00002.m4s",
            "chunk-stream0-00012.m4s.tmp",
            "init-stream0.m4s",
            "stream.mpd",
        ];
        for name in names {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        SegmentPruner::new(&dir, "chunk-stream", ".m4s", 2).prune();
        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            [
                "chunk-stream0-00010.m4s",
                "chunk-stream0-00011.m4s",
                "chunk-stream0-00012.m4s.tmp",
                "chunk-stream1-00001.m4s",
                "chunk-stream1-00002.m4s",
                "init-stream0.m4s",
                "stream.mpd",
            ],
            "by number, not by name: 00010 is newer than 00002"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "media_pp_prune_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn segments(dir: &Path) -> usize {
        std::fs::read_dir(dir).unwrap().count()
    }

    /// Keyframes written faster than they play — a stream re-muxed, a
    /// test — come within the half second a pruner waits between looks;
    /// it looks every so many of them as well, and once more as the output
    /// is finished, so that what is left is the window whatever the pace.
    #[test]
    fn a_fast_writer_is_pruned_by_keyframes_and_at_its_finish() {
        let dir = scratch("fast");
        let write = |number: u32| {
            std::fs::write(dir.join(format!("chunk-stream0-{number:05}.m4s")), b"x").unwrap()
        };
        let pruner = SegmentPruner::new(&dir, "chunk-stream", ".m4s", 2);
        for number in 1..=4 {
            write(number);
        }
        pruner.prune();
        assert_eq!(segments(&dir), 2, "the first keyframe looks");
        for number in 5..=8 {
            write(number);
        }
        pruner.prune();
        assert_eq!(segments(&dir), 6, "not again within the half second");
        for _ in 0..EVERY_KEYFRAMES {
            pruner.prune();
        }
        assert_eq!(segments(&dir), 2, "but within so many keyframes");
        for number in 9..=10 {
            write(number);
        }
        pruner.prune_now();
        assert_eq!(segments(&dir), 2, "and as the output is finished");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
