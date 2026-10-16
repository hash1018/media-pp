# remux

`FileDemuxer -> FileMuxer`: copies every stream this crate has a kind for —
video, audio and subtitles — into a new `.mp4`, with no decode or re-encode.
Packets pass through byte for byte; only their timestamps are rescaled to the
time base the output gives each track.

Which streams those are is asked of `MediaKind::packet_for`; a stream with no
kind — a data stream, an attachment — is skipped with a line saying so. One
`FileDemuxer` has a pad per stream, so one `Pipeline` carries them all, and
the program stops once `BusEvent::Finished` says every track has reached the
file.

```sh
cargo run -p remux -- input.mp4 [output.mp4] [--overwrite]
```

The output defaults to `remuxed.mp4`.

Input/output aliases (including hard links) are always refused, even with
`--overwrite`. Existing outputs require that flag. Export goes to a temporary
file beside the destination, keeping its extension; only a successfully
finished, closed file is published. Setup and bus errors exit non-zero, leave
the old destination unchanged, and do not print `wrote ...`. Partial temporary
files are removed on ordinary failure, not guaranteed after a process crash.

The extension selects the output container; it must support every kept codec.
This example does not convert incompatible codecs. Use `--` before positional
paths that begin with a dash, and quote paths containing spaces.
