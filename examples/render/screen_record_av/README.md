# screen_record_av

Screen capture + system-audio capture (whatever the default playback device
is putting out) -> one `FileMuxer`: records the desktop and its system audio
together into a single playable `.mp4`. Two independent live sources sharing
one `Pipeline` via `PipelineBuilder` — each on its own thread, but one
`pipeline.finish()` reaches both.

Neither capture source ever reaches a natural `Eos` (same as `screen_record_software`),
so this runs until `q` + Enter in the same terminal, which `finish()`es the
pipeline: each source places an `Eos` behind its last buffer, both encoders
flush what they still hold, and the MP4's trailer is written once *every*
track, video and audio both, has ended — not on whichever finishes first.

Both platforms run the same shape: two independent live capture sources, one
`FileMuxer` with a video and an audio track, one `finish()` reaching both. On
Windows, video comes from `DxgiCaptureSource` and audio from
`WasapiCaptureSource` (loopback on the default render device). On Linux, video
comes from `PipeWireScreenCaptureSource` (through the portal, so the CLI takes
a restore token) and audio from `PipeWireAudioCaptureSource` (a sink's
monitor, selected programmatically — audio needs no portal on either
platform).

```sh
# Windows
cargo run -p screen_record_av -- [output.mp4]

# Linux
cargo run -p screen_record_av -- [output.mp4] [monitor|window] [restore-token]

# then in the same terminal: q + Enter to stop and finalize
```
