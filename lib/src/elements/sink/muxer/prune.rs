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
/// packet is a keyframe.
const EVERY: Duration = Duration::from_millis(500);

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
    looked: Mutex<Option<Instant>>,
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
            looked: Mutex::new(None),
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

    /// Deletes what is older than the newest `keep` of each sequence, at
    /// most twice a second. What cannot be read or deleted is left; a later
    /// call tries again.
    pub(super) fn prune(&self) {
        {
            let mut looked = self.looked.lock().unwrap_or_else(|p| p.into_inner());
            if looked.is_some_and(|at| at.elapsed() < EVERY) {
                return;
            }
            *looked = Some(Instant::now());
        }
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
}
