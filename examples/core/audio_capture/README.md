# audio_capture

`WasapiCaptureSource -> FrameCounter` (Windows) / `PipeWireAudioCaptureSource
-> FrameCounter` (Linux): lists every audio device, picks one, captures about
three seconds from it and reports how many buffers came through.

A device of kind `Render`/`Sink` is captured through loopback or monitor
(system audio), and `Capture`/`Source` is a microphone. On Linux, where no
node of the wanted kind is marked default, any node of that kind is taken.

```sh
cargo run -p audio_capture              # default render device (system audio / loopback)
cargo run -p audio_capture -- mic       # default capture device (microphone)
cargo run -p audio_capture -- list      # just print every device and exit
cargo run -p audio_capture -- <name>    # first device whose name contains <name>
```
