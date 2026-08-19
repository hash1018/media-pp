# audio_capture

`WasapiCaptureSource -> FrameCounter` (Windows) / `PipeWireAudioCaptureSource
-> FrameCounter` (Linux): lists every audio device, picks one, captures ~3
seconds from it and reports how many buffers came through — a smoke test for
each backend's list-devices-then-pick device API.

Both platforms run the identical CLI, pipeline, and `FrameCounter` terminus —
only the source element and its device type differ, which is the point of
showing them side by side. `WasapiCaptureSource` on Windows and
`PipeWireAudioCaptureSource` on Linux line up closely: audio capture needs no
portal on either platform, a device of kind `Render`/`Sink` is captured
through loopback/monitor (system audio), and `Capture`/`Source` is a
microphone. Screen capture is where the two platforms genuinely diverge — see
`PipeWireScreenCaptureSource`'s own documentation.

```sh
cargo run -p audio_capture              # default render device (system audio / loopback)
cargo run -p audio_capture -- mic       # default capture device (microphone)
cargo run -p audio_capture -- list      # just print every device and exit
cargo run -p audio_capture -- <name>    # first device whose name contains <name>
```
