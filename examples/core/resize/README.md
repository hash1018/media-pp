# resize

Save a resized H.264 video with its original sound, entirely in software:

```text
FileDemuxer ┬ video -> SwDecoder -> Queue -> SwScaler -> SwEncoder -> FileMuxer
            └ audio ---------------------------------------------> FileMuxer
```

```sh
cargo run -p resize -- input.mp4 output.mp4 --width 160
cargo run -p resize -- input.mp4 output.mp4 --width 160 --overwrite
```

No GPU features; the same program on Windows, Linux and macOS. Uses FFmpeg's
`libopenh264` encoder. Width is a positive even integer (at least 2), in
stored pixels; height preserves the stored picture's aspect ratio, rounded
to the nearest even pixel (at least 2) for YUV420P. A 320x240 input with
`--width 160` produces 160x120 video. Rotation is kept as display metadata,
not baked into the pixels. Frame timestamps and reported colour stay.

The example accepts SDR video with square pixels. HDR must be tone-mapped
and non-square pixels corrected before using it; neither is silently
converted into incorrectly described output. This is an 8-bit H.264 export,
not a general HDR conversion tool.

The best audio stream, if present, is copied without re-encoding or resetting
its timestamps, including a delayed start. Other audio tracks, subtitles and
attachments are not kept. The output extension selects the container; the
copied audio must be supported there. An incompatible codec fails the export
without touching the destination.

Input/output aliases are always refused, including with `--overwrite`.
Existing outputs require that flag. Work is written to a temporary file beside
the destination; only a completed, closed file is published. A bus error exits
non-zero, without `wrote ...` or a partial file at the destination. Temporary
files are removed on ordinary errors, not guaranteed after a process crash.

Unlike [`scale`](../../vision/scale), which verifies and counts resized
frames, this example writes a playable file. It does not cut a time range or
change the audio encoder's timeline policy.
