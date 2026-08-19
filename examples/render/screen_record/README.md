# screen_record

`CaptureSource -> Scaler -> SwEncoder -> Mp4Muxer`: captures the desktop live
and encodes it straight into a playable `.mp4` file — no window, no renderer,
just a headless recording (compare `screen_capture`, which renders instead of
encoding).

The capture source never reaches `Eos` on its own; this just captures for a
fixed duration and then `pipeline.stop()`s, which is also what finalizes the
MP4's trailer — `Mp4Muxer` writes it on `Stop` as well as `Eos`, since an MP4
file needs a valid trailer to be playable at all.

Both platforms run the same graph, codec, and terminus. On Windows,
`DxgiCaptureSource` captures the whole desktop via DXGI Desktop Duplication.
On Linux, `PipeWireScreenCaptureSource` captures through xdg-desktop-portal:
Wayland has no way to name a monitor, so the compositor prompts on the first
run and hands back a restore token that skips the prompt on later runs.

```sh
# Windows
cargo run -p screen_record -- [output.mp4] [seconds]

# Linux
cargo run -p screen_record -- [output.mp4] [seconds] [monitor|window] [restore-token]
```
