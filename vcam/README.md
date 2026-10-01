# media-pp-vcam

The camera side of media-pp's Windows virtual camera: a COM DLL holding a
Media Foundation media source, which Windows' Frame Server loads into its own
service process to serve every application that opens the camera. The
pictures come from an application running `MfVirtualCamera` (feature
`mf-virtual-camera`) through a section of shared memory; what the two share
is `lib/src/elements/sink/virtual_camera/protocol.rs`, compiled into both.

It is a crate of its own, not part of `media-pp`, because it is a different
program: a DLL loaded by a Windows service, which links nothing of FFmpeg —
the service would not find it — and is never published to crates.io.

## Install

Windows 11 (build 22000) or newer. Once per machine, from an elevated
PowerShell:

```powershell
cargo build --release -p media-pp-vcam
powershell -ExecutionPolicy Bypass -File vcam\install.ps1
```

The script copies the DLL to `Program Files\media-pp\vcam` — the service
cannot read it from a user's profile — and registers it for the machine.
`-Uninstall` removes it. Rebuilding needs the script again: Frame Server
keeps the loaded DLL open, and the script stops it first.

Then `cargo run -p virtual_camera` shows a test pattern as "media-pp Windows
Virtual Camera" in any application that lists cameras.

## What it offers

NV12 at 1920x1080, 1280x720 and 640x360, 30 frames a second, BT.709 limited
range. Each picture is the producer's newest; with no producer, or none for
over a second, a dark grey frame.

## Tests

`cargo test -p media-pp-vcam` drives the source in-process as Frame Server
does, with no registration, playing the producer's part on a section of its
own.
