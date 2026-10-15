# doctor

What this machine gives media-pp, and what it lacks. Run it after setting up
FFmpeg, or when something will not build or start:

```sh
cargo run -p doctor
cargo run -p doctor --features vulkan
```

It checks, and says what to do about each problem it finds:

- **FFmpeg**: the libraries the program loaded, against FFmpeg 8.0 — the
  release media-pp is built for — and against the headers it was compiled
  from; its licence.
- **Encoders** that FFmpeg carries: OpenH264, NVENC, Media Foundation,
  VideoToolbox, VA-API, Vulkan and D3D12 Video, for H.264 and HEVC.
- **Hardware decoding**: each kind of hardware FFmpeg knows, and whether it
  makes a device of it on this machine.
- **GPU backends** in this build: a device of each made and let go of —
  D3D11 and D3D12 on Windows, CUDA on Windows and Linux, VideoToolbox and
  Metal on macOS, Vulkan with `--features vulkan`. Vulkan is not on by
  default because building it needs the Vulkan headers, which a machine
  being diagnosed may not have.

No pipeline runs. It exits with a failure code when it found a problem.

A program that cannot find FFmpeg's libraries does not reach `main` at all:
Windows reports `STATUS_DLL_NOT_FOUND`, Linux names the missing `.so`. That
is the answer too — the libraries are not where the loader looks —
and [`scripts/setup-ffmpeg`](../../../scripts) sets that up.

Logs go to `./logs`, prefixed `doctor`.
