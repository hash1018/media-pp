# screen_record_software

`DxgiCaptureSource | PipeWireScreenCaptureSource | ScreenCaptureKitSource ->
Queue -> SwScaler -> Queue -> SwEncoder -> FileMuxer`: captures the desktop and encodes it into a
playable `.mp4` — a headless recording (compare `screen_preview_cpu`, which
shows the same CPU-frame path in a window instead).

A capture never ends by itself, so this records for a fixed time and then
calls `pipeline.finish()`: the capture ends its stream behind its last frame,
the encoder flushes what it holds, and the muxer writes the trailer after it.
`stop()` would also leave a playable file — `FileMuxer` writes the trailer on
a stop too — but abandon the last few hundred milliseconds still queued.

On Windows `DxgiCaptureSource` captures the whole desktop through DXGI Desktop
Duplication. On Linux `PipeWireScreenCaptureSource` captures through
xdg-desktop-portal: the compositor prompts on the first run and hands back a
restore token that skips the prompt later. On macOS `ScreenCaptureKitSource`
captures the main display, or with `window` the frontmost titled window;
macOS asks once for the permission to record the screen, given in System
Settings to the terminal it runs in, which then has to be started again.

```sh
# Windows
cargo run -p screen_record_software -- [output.mp4] [seconds]

# Linux
cargo run -p screen_record_software -- [output.mp4] [seconds] [monitor|window] [restore-token]

# macOS
cargo run -p screen_record_software -- [output.mp4] [seconds] [monitor|window]
```

It records to `screen_record_software.mp4` for 5 seconds by default.
