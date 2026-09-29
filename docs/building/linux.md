# Building on Linux

What the `linux` job in [`ci.yml`](../../.github/workflows/ci.yml) does on
Ubuntu, for a machine of your own. What every platform shares — which
FFmpeg, and why that one — is in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#setting-up-a-machine).

## Tools

- **Rust** stable, with `rustfmt` and `clippy`.
- What vcpkg builds FFmpeg's port with, and libclang for the bindgen
  `ffmpeg-sys-next` runs on every build:

  ```sh
  sudo apt-get install -y build-essential curl zip unzip tar pkg-config \
    ninja-build nasm autoconf automake libtool python3 libclang-dev
  ```

- **CMake**, for `whisper`'s whisper.cpp.

## FFmpeg

vcpkg at the tag CI pins, into a dynamic tree — the default `x64-linux`
triplet links statically, and would need every library FFmpeg's port pulls
in linked by hand:

```sh
git clone --branch 2026.01.16 https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
./vcpkg/vcpkg install "ffmpeg[openh264,vulkan]:x64-linux-dynamic"
export FFMPEG_DIR=$PWD/vcpkg/installed/x64-linux-dynamic
export LD_LIBRARY_PATH=$FFMPEG_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

`openh264` is the test fixture's encoder, and `vulkan` is only for the
crate's `vulkan` feature. The shared libraries are found at run time
through `LD_LIBRARY_PATH`; without it everything compiles, and every test
binary fails to start.

## Features that need more

- `pipewire-*`: PipeWire 0.3.50 or newer, `libpipewire-0.3-dev`.
- `vulkan`: the Vulkan headers, from `libvulkan-dev` or the Vulkan SDK
  (`VULKAN_SDK`); the loader is opened at run time. Without a GPU, Mesa's
  lavapipe (`mesa-vulkan-drivers libvulkan1`) runs what needs only a Vulkan
  device, as in CI; Vulkan Video decode and encode and presenting skip.
- `cuda`: only the NVIDIA driver; the kernels ship as PTX. A binary built
  with `cuda` does not start on a machine without it.
- `whisper-vulkan`: the Vulkan SDK's shader compiler at build time — see
  [`transcribe`](../../examples/core/transcribe/README.md).
- Screen capture goes through xdg-desktop-portal, whose picker opens on the
  first run. The restore token it prints can be passed on later runs, and
  the capture soak scenarios need one in `MEDIA_PP_SOAK_RESTORE_TOKEN`.

## Checks

As CI runs them:

```sh
cargo fmt --all -- --check
cargo clippy -p media-pp --all-targets --features pipewire-audio-capture,pipewire-audio-renderer,pipewire-screen-capture,v4l2-capture -- -D warnings
cargo clippy -p media-pp --all-targets --features vulkan,cuda -- -D warnings
cargo test -p media-pp --features pipewire-audio-capture,pipewire-audio-renderer,pipewire-screen-capture,v4l2-capture
cargo test -p media-pp --features vulkan
```

and the documentation with `RUSTDOCFLAGS="-D warnings"`, one feature set at
a time. The control conformance runs pinned to two cores are in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#control-sequences).
