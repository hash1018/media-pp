# Building on macOS

What the `macos` job in [`ci.yml`](../../.github/workflows/ci.yml) does on
Apple silicon, for a machine of your own. What every platform shares — which
FFmpeg, and why that one — is in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#setting-up-a-machine).

## Tools

- The **Xcode Command Line Tools** (`xcode-select --install`): the compiler,
  the SDK, and the libclang the bindgen `ffmpeg-sys-next` runs on every build
  needs. Full Xcode only for Instruments and signing.
- **Rust** stable, with `clippy`.
- **pkg-config**, which FFmpeg's `configure` finds OpenH264 with.
- **CMake**, for `whisper`'s whisper.cpp.

## FFmpeg

With VideoToolbox and AudioToolbox, and OpenH264 for the test fixture. The
quickest way is vcpkg at the tag CI pins, which enables VideoToolbox on
macOS by itself:

```sh
git clone --branch 2026.01.16 https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
./vcpkg/vcpkg install "ffmpeg[openh264]:arm64-osx-dynamic"
export FFMPEG_DIR=$PWD/vcpkg/installed/arm64-osx-dynamic
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$FFMPEG_DIR/lib"
export RUSTDOCFLAGS="$RUSTFLAGS"
```

vcpkg names its dylibs `@rpath/...`, so a binary finds them only through an
rpath of its own; `RUSTDOCFLAGS` carries it to the doctests, which rustdoc
links. `DYLD_LIBRARY_PATH` is no substitute: macOS strips it from the
environment of every protected binary — `/bin/sh` and `/bin/bash` among
them — and so from everything started through one.

Built from source instead, with `--prefix` where it will stay, the dylibs
carry their absolute paths and need no rpath:

```sh
prefix=$HOME/.local/ffmpeg-8.0
# OpenH264 v2.6.0 into the same prefix first:
#   make OS=darwin ARCH=arm64 PREFIX=$prefix install-shared
PKG_CONFIG_PATH=$prefix/lib/pkgconfig ./configure --prefix=$prefix \
  --enable-shared --disable-static --disable-gpl --disable-nonfree \
  --enable-videotoolbox --enable-audiotoolbox --enable-libopenh264 \
  --disable-doc --disable-ffplay --extra-ldflags="-Wl,-rpath,$prefix/lib"
make -j && make install
export FFMPEG_DIR=$prefix
```

Homebrew's `ffmpeg` follows FFmpeg's latest release, which is past 8.0.

## Permissions

The camera, the screen and another process's sound are each granted to an
application, by the user, once — to the terminal a test or an example runs
in. The camera and screen tests ask whether they have been allowed without
asking to be, and skip where not; a capture of an output device records
silence without the audio-capture permission rather than failing.

`tests/metal_window.rs` opens real windows and has a `main` of its own,
since AppKit serves windows only from the main thread's event loop, which
the test harness does not run. Built with `screencapturekit-capture` and run
with `MEDIA_PP_SCREEN_CHECK=1`, it also captures each window and checks its
colours — only then, since asking whether it may record the screen shows a
prompt.

## Checks

As CI runs them — the macOS backends together, and each alone, since the
code they share is gated on whichever of them uses it:

```sh
cargo clippy -p media-pp --all-targets --features videotoolbox,metal,avfoundation-capture,screencapturekit-capture,coreaudio-capture,coreaudio-renderer -- -D warnings
cargo clippy -p media-pp --all-targets --features metal -- -D warnings
cargo test -p media-pp --features videotoolbox,metal,avfoundation-capture,screencapturekit-capture,coreaudio-capture,coreaudio-renderer
```

and the documentation with `-D warnings` added to `RUSTDOCFLAGS`. macOS has
no way to pin a process to two cores, so the control conformance runs are
the other platforms'.
