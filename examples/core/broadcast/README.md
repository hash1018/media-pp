# broadcast

One input written once into a `Broadcast` and read twice: recorded to a file
through a `BroadcastSource`, and counted by a `BroadcastReader` of the
program's own — while the input drops and comes back.

The input is a file played at its own speed, standing in for a camera. It
plays twice, a second apart, each time as a pipeline of its own that publishes
into the same broadcast — what an input opened again after it dropped does.
The recording and the reader go on across the drop: the reader is told one run
ended and another began, and the recording puts the second run after the
first, a second's gap between them, on one timeline from zero.

```text
FileDemuxer -> Pacer -> BroadcastSink        (one per track, each run)
                             |
                         Broadcast
                        /         \
BroadcastSource -> FileMuxer     BroadcastReader (counts, on a thread)
```

Each run plays at most five seconds of the input.

```sh
cargo run -p broadcast -- input.mp4 [output.mp4] [--overwrite]
```

The output defaults to `broadcast.mp4`. It is written beside the destination
and published only once the recording has finished; an existing destination
needs `--overwrite`.
