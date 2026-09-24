# screen_record_software

`CaptureSource -> SwScaler -> SwEncoder -> FileMuxer`: captures the desktop live
and encodes it straight into a playable `.mp4` file — no window, no renderer,
just a headless recording (compare `screen_preview_cpu`, which renders the
same CPU-frame path instead of encoding it on Windows and Linux).

The capture source never reaches `Eos` on its own; this just captures for a
fixed duration and then `pipeline.finish()`es: the capture places an `Eos`
behind its last frame, the encoder flushes what it still holds, and the muxer
writes the MP4's trailer after it. `stop()` would finalize a playable file
too — `FileMuxer` writes the trailer on `Stop` as well — but abandon the
frames still in the queue and the encoder, a few hundred milliseconds of the
end.

Both platforms run the same graph, codec, and terminus. On Windows,
`DxgiCaptureSource` captures the whole desktop via DXGI Desktop Duplication.
On Linux, `PipeWireScreenCaptureSource` captures through xdg-desktop-portal:
Wayland has no way to name a monitor, so the compositor prompts on the first
run and hands back a restore token that skips the prompt on later runs.

```sh
# Windows
cargo run -p screen_record_software -- [output.mp4] [seconds]

# Linux
cargo run -p screen_record_software -- [output.mp4] [seconds] [monitor|window] [restore-token]
```
