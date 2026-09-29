# Contributing to media-pp

[`AGENTS.md`](AGENTS.md) holds the design and error-handling conventions every
change is held to; this file is how to build and test one.

## Tests

```sh
cargo test -p media-pp
```

Tests need no media: they synthesize their fixture from the crate's own
sources and encoders, so every machine tests the same file. A backend's tests
run under its feature (`--features d3d11,d3d12,cuda`, or the Linux or macOS
ones) and skip, saying why, on a machine without the hardware.

On a Mac, the camera and screen tests also skip unless the program running
them has been allowed the camera or screen recording, which a terminal asks
for once and keeps. `tests/metal_window.rs` opens real windows and has a
`main` of its own, since AppKit serves windows only from the main thread;
built with `screencapturekit-capture` and run with `MEDIA_PP_SCREEN_CHECK=1`
it also captures each window and checks its colours.

## Setting up a Mac

FFmpeg 8.0 is what the crate builds against — `lib/build.rs` stops on
anything older, and `ffmpeg-next` 8.1 does not build against FFmpeg 8.1's
headers — so it is 8.0.1, as CI pins, with VideoToolbox and AudioToolbox,
and OpenH264 for the test fixture. Beside it, the Xcode Command Line Tools
(the SDK, and the libclang `ffmpeg-sys-next`'s bindgen needs), pkg-config,
and CMake for `whisper`.

The quickest way to it is vcpkg at the tag CI pins, which enables
VideoToolbox on macOS by itself:

```sh
git clone --branch 2026.01.16 https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
./vcpkg/vcpkg install "ffmpeg[openh264]:arm64-osx-dynamic"
export FFMPEG_DIR=$PWD/vcpkg/installed/arm64-osx-dynamic
```

Its dylibs are named `@rpath/...`, so a test binary needs an rpath to find
them — `RUSTFLAGS` and `RUSTDOCFLAGS` set to
`-C link-arg=-Wl,-rpath,$FFMPEG_DIR/lib`, as CI sets them. Built from
source instead, with `--prefix` where it will stay, FFmpeg's dylibs carry
their absolute paths and need nothing:

```sh
prefix=$HOME/.local/ffmpeg-8.0
# OpenH264 v2.6.0 into the same prefix first:
#   make OS=darwin ARCH=arm64 PREFIX=$prefix install-shared
PKG_CONFIG_PATH=$prefix/lib/pkgconfig ./configure --prefix=$prefix \
  --enable-shared --disable-static --disable-gpl --disable-nonfree \
  --enable-videotoolbox --enable-audiotoolbox --enable-libopenh264 \
  --disable-doc --disable-ffplay --extra-ldflags="-Wl,-rpath,$prefix/lib"
make -j && make install
```

`FFMPEG_DIR` is read before any pkg-config discovery, so it wins over any
other FFmpeg on the machine. Homebrew's `ffmpeg` follows FFmpeg's latest
release, which is past 8.0.

## Control sequences

`core::pipeline::tests::conformance` runs random orders of pause, resume,
seek, frame step, rate, looping, finish and stop — and stop while another
call is under way — against the shapes of pipeline this crate is used in,
and checks what every terminal was handed: nothing new once a pause has
returned, nothing from before a seek after it, nothing paced running ahead
of the playback clock, an `Eos` after a finish, and no call that fails to
return. The file shapes are a matrix of six axes —
how the picture fans out, what decodes it, what filters it, what paces it,
what the sound goes through, how deep the queues are — covering every pair
of choices; `MEDIA_PP_CONTROL_FULL=1` runs every combination. What a known
bug breaks is listed in `KNOWN_BROKEN`, beside the ignored test that
reproduces it, and comes back into the sequences when the bug is fixed. An
ordinary test run plays a few fixed sequences. The races these are for show
when threads are short of cores, so CI also runs them pinned to two, seeded
from the clock, once on their own and once with two busy threads beside them
(`MEDIA_PP_CONTROL_LOAD`), since some bursts showed only with other work
around:

```sh
MEDIA_PP_CONTROL_ITERS=10 MEDIA_PP_CONTROL_RANDOM=1 taskset -c 0,1 cargo test -p media-pp --lib -- conformance
MEDIA_PP_CONTROL_ITERS=10 MEDIA_PP_CONTROL_RANDOM=1 MEDIA_PP_CONTROL_LOAD=2 taskset -c 0,1 cargo test -p media-pp --lib -- conformance
```

On Windows, set the shell's own affinity first —
`[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = 3` —
and cargo and the tests inherit it. A failure prints its shape, its seed and
the steps it took, and the line that replays it:
`MEDIA_PP_CONTROL_SHAPE=<shape> MEDIA_PP_CONTROL_SEED=<seed>` runs exactly
that sequence (`MEDIA_PP_CONTROL_STEPS` sets its length, 12 by default), and
`MEDIA_PP_CONTROL_TRACE=<directory>` writes the crate's log there at `Trace`,
every control message at every element, for reading what it did. A change to
how control travels — a new message, a new element that waits, a new source
loop — should pass a few hundred of these before it lands.

## Stress and leak scenarios

The scenarios in `lib/tests/soak.rs` run for tens of seconds and are
`#[ignore]`d. They read a real recording from `MEDIA_PP_TEST_VIDEO`:

```sh
cargo test -p media-pp --features d3d11,d3d12,cuda --test soak -- --ignored --nocapture
```

On Linux, `pipewire-screen-capture` takes the place of `d3d11`, and the
capture scenarios also need `MEDIA_PP_SOAK_RESTORE_TOKEN`, since the portal
would otherwise show its picker; any run of `screen_record_software` prints a
token to reuse.

## Documentation

[docs.rs] builds for Linux and so omits the Windows-only and macOS-only
API, which the API documentation workflow builds on runners of their own.
To build either locally, labelled by feature:

```powershell
$env:RUSTDOCFLAGS = "--cfg docsrs"
cargo +nightly doc -p media-pp --open --features d3d11,d3d12,dxgi-capture,wgc-capture,mf-capture,wasapi-capture,wasapi-renderer,webrtc
```

```sh
RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc -p media-pp --open --features videotoolbox,metal,avfoundation-capture,screencapturekit-capture,coreaudio-capture,coreaudio-renderer,webrtc
```

CI builds the documentation with `-D warnings` for every feature set, so a
public item's documentation may not link to a private one, or to one its
feature set does not have.

[docs.rs]: https://docs.rs/media-pp
