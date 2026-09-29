# audio_capture

`WasapiCaptureSource -> FrameCounter` (Windows) / `PipeWireAudioCaptureSource
-> FrameCounter` (Linux) / `CoreAudioCaptureSource -> FrameCounter` (macOS):
lists every audio device, picks one, captures about three seconds from it and
reports how many buffers came through.

A device of kind `Render`/`Sink` is captured through loopback or monitor
(system audio), and `Capture`/`Source` is a microphone. On Linux, where no
node of the wanted kind is marked default, any node of that kind is taken.

On macOS an output device is captured through a Core Audio process tap,
which needs macOS 14.2 and the "System Audio Recording" permission for the
terminal running this — one whose `Info.plist` does not ask for it records
silence. A microphone asks for its own permission the first time.

```sh
cargo run -p audio_capture              # default render device (system audio / loopback)
cargo run -p audio_capture -- mic       # default capture device (microphone)
cargo run -p audio_capture -- list      # just print every device and exit
cargo run -p audio_capture -- <name>    # first device whose name contains <name>
```
