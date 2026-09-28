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
cargo run -p remux -- input.mp4 [output.mp4]
```

The output defaults to `remuxed.mp4`.
