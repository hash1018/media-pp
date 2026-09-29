# audio_capture

`WasapiCaptureSource -> FrameCounter` (Windows) / `PipeWireAudioCaptureSource
-> FrameCounter` (Linux) / `CoreAudioCaptureSource -> FrameCounter` (macOS):
lists every audio device, picks one, captures about three seconds from it and
reports how many buffers came through.

A device of kind `Render`/`Sink` is captured through loopback or monitor
(system audio), and `Capture`/`Source` is a microphone. On Linux, where no
node of the wanted kind is marked default, any node of that kind is taken.

On macOS only input devices can be captured so far, so the default — the
system's sound — says so and exits non-zero; use `mic` or a device's name.
macOS asks, the first time, whether the terminal running this may use the
microphone.

```sh
cargo run -p audio_capture              # default render device (system audio / loopback)
cargo run -p audio_capture -- mic       # default capture device (microphone)
cargo run -p audio_capture -- list      # just print every device and exit
cargo run -p audio_capture -- <name>    # first device whose name contains <name>
```
