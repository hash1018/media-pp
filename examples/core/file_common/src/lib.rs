//! File-export examples' CLI, failure policy and staged output.
//!
//! This is example policy, not a change to `FileMuxer`: refuse input/output
//! aliases, opt in to replacing an existing destination, and publish only
//! after every track has finished and its file handle has closed.

use std::{
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use media_pp::{Error, Result, bus::BusEvent, pipeline::Pipeline};
use tempfile::TempPath;

/// The common positional paths and explicit overwrite flag.
pub struct FileArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    pub overwrite: bool,
}

impl FileArgs {
    /// Parses arguments without the executable name. `--` permits paths that
    /// begin with a dash; only the older examples have a default output name.
    pub fn parse(args: impl IntoIterator<Item = OsString>, default: Option<&str>) -> Result<Self> {
        let mut paths = Vec::new();
        let mut overwrite = false;
        let mut positional = false;
        for arg in args {
            if !positional && arg == "--" {
                positional = true;
            } else if !positional && arg == "--overwrite" {
                if overwrite {
                    return Err(message("--overwrite was supplied twice"));
                }
                overwrite = true;
            } else if !positional && arg.to_string_lossy().starts_with('-') {
                return Err(message(format!("unknown option {}", arg.display())));
            } else {
                paths.push(PathBuf::from(arg));
            }
        }
        if paths.is_empty() || paths.len() > 2 || (paths.len() == 1 && default.is_none()) {
            return Err(message("provide an input and an output path"));
        }
        let output = paths
            .get(1)
            .cloned()
            .unwrap_or_else(|| PathBuf::from(default.unwrap()));
        Ok(Self {
            input: paths.remove(0),
            output,
            overwrite,
        })
    }
}

/// A same-directory temporary output. Dropping without `commit` removes the
/// partial file; the destination is never opened by FFmpeg.
pub struct SafeOutput {
    temporary: TempPath,
    input: PathBuf,
    destination: PathBuf,
    overwrite: bool,
}

impl SafeOutput {
    pub fn new(args: &FileArgs) -> Result<Self> {
        let input = fs::canonicalize(&args.input).map_err(|e| at("open input", &args.input, e))?;
        let parent = args
            .output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent =
            fs::canonicalize(parent).map_err(|e| at("open output directory", parent, e))?;
        let name = args
            .output
            .file_name()
            .ok_or_else(|| message("output must name a file"))?;
        let destination = parent.join(name);
        validate_destination(&input, &destination, args.overwrite)?;
        // Keep the extension: FileMuxer chooses the container by it, not by
        // the destination's name after this temporary file has been moved.
        let mut suffix = OsString::from(".");
        if let Some(extension) = destination.extension() {
            suffix.push(extension);
        }
        let temporary = tempfile::Builder::new()
            .prefix(".media-pp-")
            .suffix(&suffix)
            .tempfile_in(parent)
            .map_err(|e| at("stage output", &destination, e))?
            .into_temp_path(); // close the reservation handle before FFmpeg opens it
        Ok(Self {
            temporary,
            input,
            destination,
            overwrite: args.overwrite,
        })
    }

    pub fn path(&self) -> &Path {
        &self.temporary
    }

    /// The caller must first drop the pipeline, closing FFmpeg's AVIO handle.
    /// Recheck aliases at publication, and use no-clobber publication unless
    /// replacement was explicitly authorized (including a newly appeared file).
    pub fn commit(self) -> Result<()> {
        validate_destination(&self.input, &self.destination, self.overwrite)?;
        let result = if self.overwrite {
            self.temporary.persist(&self.destination)
        } else {
            self.temporary.persist_noclobber(&self.destination)
        };
        result.map_err(|e| at("publish output", &self.destination, e.error))
    }
}

fn validate_destination(input: &Path, output: &Path, overwrite: bool) -> Result<()> {
    match fs::symlink_metadata(output) {
        Ok(metadata) => {
            if same_file::is_same_file(input, output)
                .map_err(|e| at("compare input and output", output, e))?
            {
                return Err(message(format!(
                    "input and output are the same file: {}",
                    output.display()
                )));
            }
            if !metadata.is_file() && !metadata.is_symlink() {
                return Err(message(format!(
                    "output is not a file: {}",
                    output.display()
                )));
            }
            if !overwrite {
                return Err(message(format!(
                    "output already exists: {}; use --overwrite to replace it",
                    output.display()
                )));
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(at("inspect output", output, e)),
    }
    Ok(())
}

/// Runs a file export to all tracks' EOS. An error fails this export, not the
/// library's recovery policy. Owns the pipeline so all file handles are closed
/// before the caller publishes or discards its staged output.
/// Pass the pipeline's only strong owner; do not retain another `Arc` clone.
pub fn run_to_end(pipeline: Arc<Pipeline>) -> Result<()> {
    pipeline.run()?;
    let mut result = Err(message("pipeline ended without Finished"));
    for event in pipeline.bus().iter() {
        match event {
            BusEvent::Finished => {
                result = Ok(());
                break;
            }
            BusEvent::Error { error, .. } => {
                result = Err(error);
                break;
            }
            _ => println!("{event}"),
        }
    }
    pipeline.stop();
    // Stop joins the source and queue workers. Do not lose a failure queued
    // just behind Finished, including one raised while stopping a track.
    while let Some(event) = pipeline.bus().try_recv() {
        if let BusEvent::Error { error, .. } = event
            && result.is_ok()
        {
            result = Err(error);
        }
    }
    result
}

pub fn message(text: impl Into<String>) -> Error {
    Error::Other(text.into())
}

fn at(action: &str, path: &Path, error: io::Error) -> Error {
    message(format!("could not {action} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(input: &Path, output: &Path, overwrite: bool) -> FileArgs {
        FileArgs {
            input: input.into(),
            output: output.into(),
            overwrite,
        }
    }

    #[test]
    fn missing_extra_and_unknown_arguments_are_errors() {
        assert!(FileArgs::parse([], None).is_err());
        assert!(FileArgs::parse(["in.mp4".into()], None).is_err());
        assert!(
            FileArgs::parse(["in.mp4".into(), "out.mp4".into(), "extra".into()], None).is_err()
        );
        assert!(FileArgs::parse(["in.mp4".into(), "--typo".into()], Some("out.mp4")).is_err());
        let parsed = FileArgs::parse(
            ["--overwrite".into(), "--".into(), "-in.mp4".into()],
            Some("out.mp4"),
        )
        .unwrap();
        assert!(parsed.overwrite);
        assert_eq!(parsed.input, Path::new("-in.mp4"));
    }

    #[test]
    fn same_file_relative_alias_and_hardlink_never_change_input() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.mp4");
        fs::write(&input, b"original input").unwrap();
        let alias = dir.path().join("./input.mp4");
        let hardlink = dir.path().join("hardlink.mp4");
        fs::hard_link(&input, &hardlink).unwrap();
        for output in [&input, &alias, &hardlink] {
            for overwrite in [false, true] {
                assert!(SafeOutput::new(&args(&input, output, overwrite)).is_err());
                assert_eq!(fs::read(&input).unwrap(), b"original input");
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn case_alias_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.mp4");
        fs::write(&input, b"original").unwrap();
        assert!(SafeOutput::new(&args(&input, &dir.path().join("INPUT.MP4"), true)).is_err());
        assert_eq!(fs::read(&input).unwrap(), b"original");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_alias_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.mp4");
        let alias = dir.path().join("link.mp4");
        fs::write(&input, b"original").unwrap();
        std::os::unix::fs::symlink(&input, &alias).unwrap();
        assert!(SafeOutput::new(&args(&input, &alias, true)).is_err());
        assert_eq!(fs::read(&input).unwrap(), b"original");
    }

    #[test]
    fn failed_export_preserves_destination_and_removes_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.mp4");
        let output = dir.path().join("out.mp4");
        fs::write(&input, b"input").unwrap();
        fs::write(&output, b"existing destination").unwrap();
        assert!(SafeOutput::new(&args(&input, &output, false)).is_err());
        let stage = SafeOutput::new(&args(&input, &output, true)).unwrap();
        let partial = stage.path().to_owned();
        assert_eq!(partial.extension(), output.extension());
        fs::write(&partial, b"partial export").unwrap();
        drop(stage);
        assert!(!partial.exists());
        assert_eq!(fs::read(output).unwrap(), b"existing destination");
    }

    #[test]
    fn publication_replaces_only_when_authorized_and_rechecks_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.mp4");
        let output = dir.path().join("out.mp4");
        fs::write(&input, b"input").unwrap();
        let stage = SafeOutput::new(&args(&input, &output, false)).unwrap();
        fs::write(stage.path(), b"finished").unwrap();
        fs::write(&output, b"appeared meanwhile").unwrap();
        assert!(stage.commit().is_err());
        assert_eq!(fs::read(&output).unwrap(), b"appeared meanwhile");
        let stage = SafeOutput::new(&args(&input, &output, true)).unwrap();
        fs::write(stage.path(), b"finished").unwrap();
        stage.commit().unwrap();
        assert_eq!(fs::read(&output).unwrap(), b"finished");
        let stage = SafeOutput::new(&args(&input, &output, true)).unwrap();
        fs::remove_file(&output).unwrap();
        fs::hard_link(&input, &output).unwrap();
        assert!(stage.commit().is_err());
        assert_eq!(fs::read(input).unwrap(), b"input");
    }

    #[test]
    fn publication_creates_a_finished_file_and_no_temporary_leftover() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.mp4");
        let output = dir.path().join("out.mp4");
        fs::write(&input, b"input").unwrap();
        let stage = SafeOutput::new(&args(&input, &output, false)).unwrap();
        let temporary = stage.path().to_owned();
        fs::write(&temporary, b"finished").unwrap();
        stage.commit().unwrap();
        assert_eq!(fs::read(output).unwrap(), b"finished");
        assert!(!temporary.exists());
    }

    #[test]
    fn a_queued_runtime_failure_is_an_error_even_if_the_stream_finishes() {
        use media_pp::{buffer::MediaBuffer, element::Sink, elements::AppSource, ffmpeg};
        struct Fail;
        impl Sink for Fail {
            fn render(&mut self, _: MediaBuffer) -> Result<()> {
                Err(message("injected runtime failure"))
            }
        }
        let (source, handle) = AppSource::new("source", 4);
        handle
            .push(MediaBuffer::Packet(Arc::new(ffmpeg::Packet::new(4)).into()))
            .unwrap();
        handle.finish().unwrap();
        let (pipeline, ()) = Pipeline::new("failing-export", source, |source, ctx| {
            ctx.attach(source, 0, ctx.branch().queue("worker", 2).to(Fail)?)?;
            Ok(())
        })
        .unwrap();
        assert!(
            run_to_end(pipeline)
                .unwrap_err()
                .to_string()
                .contains("injected runtime failure")
        );
    }
}
