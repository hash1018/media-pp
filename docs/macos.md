# Porting media-pp to macOS

Where macOS stands, what a port has to add, and how to set a Mac up to do
it. First written from a read of the tree on a Linux machine; the build and
setup below were then checked on an Apple silicon Mac (macOS 26.5) at
`4de589d`. The backend half is still a reading of the code. Correct it as
you go.

obs-rs has a companion document, `docs/macos.md` in that repository, for the
application's half. The machine setup below serves both.

## Where macOS stands

The macOS backend so far is audio — `CoreAudioRenderer` behind
`coreaudio-renderer` and `CoreAudioCaptureSource` behind `coreaudio-capture`,
sharing their device listing and AUHAL unit in `platform/macos/coreaudio/` —
VideoToolbox frames behind `videotoolbox`: decode, encode, upload and
download, in `MemoryDomain::VideoToolbox`, and the bins' VideoToolbox arms —
and the camera, `AvFoundationCaptureSource` behind `avfoundation-capture`.
Beside them a macOS build gets everything that is not a platform backend:

- the pipeline core, `FileDemuxer`, `RtspSource`, `AppSource`/`AppSink`, the
  test sources, and every muxer (file, segmented, HLS, RTMP, RTSP, replay
  buffer);
- the audio mixer and audio filters, `SwAudioEncoder`;
- the software video elements — `SwDecoder`, `SwEncoder`, `SwScaler`,
  `SwVideoCompositor`, `SwChromaKey`, `SwVideoEffect` — and `VideoDecodeBin`
  / `VideoEncodeBin` with their `System` target only;
- the platform-neutral features: `rnnoise`, `webrtc`, `whisper`, `ort`.

It gets no screen capture, no window or `VideoWindow`,
no `Player`, and no GPU compositing: `cuda` compiles but finds no driver,
and `vulkan` is discussed below.

On the Mac, `cargo build -p media-pp` builds with no features and without a
warning, `cargo test -p media-pp` passes, and `cargo build --workspace`
builds every example — the ones with a backend say which platforms they
support and exit. `render_common` has only `Shutdown` there:
`stop_on_close` needs the library's window types, which exist only beside a
window renderer, so it opens to macOS with the Metal one. What remains:

- **The test fixture needs OpenH264.** `test_support::try_test_video`
  encodes its eight-second fixture with `VideoCodec::OpenH264`; a skip where
  it cannot, but `synthesize`, `synthesize_reordered`, `try_encoded_packets`
  and `try_tagged_packets` `expect()` it and panic. The FFmpeg the tests run
  against needs `libopenh264` — see the setup.
- **`lib/tests/common/mod.rs` `private_bytes`** reads `/proc/self/statm` off
  Windows and would panic on a Mac, and multiplies by a 4 KiB page where
  Apple silicon has 16 KiB. Only the `#[ignore]`d soak scenarios use it.

The text tests draw with `/System/Library/Fonts/Supplemental/Arial.ttf`,
the same face they use on Windows.

## Setting up the Mac

Everything below installs under `~/.local` and needs no `sudo`, so no
Homebrew either. The prefixes are baked into the libraries' install names
(below), so choose them once: renaming the home directory later leaves
every binary looking for its libraries under the old path, which
`install_name_tool -id`/`-change`/`-rpath` and an ad-hoc `codesign -f -s -`
can repair, and a rebuild is simpler.

### Tools

1. **Xcode Command Line Tools**: `xcode-select --install`. This is the
   compiler, the SDK, and the libclang that bindgen needs — `ffmpeg-sys-next`
   runs bindgen on every build. Full Xcode is only needed later, for
   Instruments and for signing.
2. **pkg-config**, which FFmpeg's `configure` finds OpenH264 with: pkgconf
   from its release tarball (2.5.1 was used), `./configure
   --prefix=$HOME/.local && make install`, and a `pkg-config` symlink to
   `pkgconf` beside it.
3. **CMake**, for `whisper`'s whisper.cpp — the `transcribe` example, and so
   `cargo build --workspace`: Kitware's `cmake-<version>-macos-universal`
   release unpacked under `~/.local/opt`, with `cmake` linked into
   `~/.local/bin` from `CMake.app/Contents/bin`.
4. **Rust**: `rustup` stable, plus the toolchain CI lints with. CI installs
   whatever stable is current, which has been newer than a local default
   before and failed lints nobody saw locally; check `rustup check` and
   install that version beside it, then lint with `cargo +<version> clippy …`.
5. The two repositories side by side, as the Linux and Windows machines have
   them.

### FFmpeg

media-pp needs FFmpeg 8.0 (libavcodec 62.8+); `lib/build.rs` stops on
anything older, and `ffmpeg-next` 8.1 does not build against 8.1's headers
(see `.github/actions/setup-ffmpeg`). So it is 8.0.1, the version CI pins,
built from source: LGPL, with VideoToolbox and AudioToolbox for hardware
coding, and OpenH264 for the test fixture — BSD-licensed, so the build stays
LGPL, and the one software video encoder in it. No x264 or x265, and no
Vulkan: MoltenVK has no Vulkan Video to give it.

OpenH264 first, into the prefix FFmpeg will have:

```sh
prefix=$HOME/.local/ffmpeg-8.0
# github.com/cisco/openh264, tag v2.6.0
make -j OS=darwin ARCH=arm64 PREFIX=$prefix
make OS=darwin ARCH=arm64 PREFIX=$prefix install-shared
```

Then FFmpeg, from `ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz`:

```sh
PKG_CONFIG_PATH=$prefix/lib/pkgconfig ./configure --prefix=$prefix \
  --enable-shared --disable-static --disable-gpl --disable-nonfree \
  --enable-videotoolbox --enable-audiotoolbox --enable-libopenh264 \
  --disable-doc --disable-ffplay --extra-ldflags="-Wl,-rpath,$prefix/lib"
make -j && make install
```

`configure` should report `License: LGPL version 2.1 or later`, and
`videotoolbox` and `audiotoolbox` under hardware acceleration. It also
picks up `avfoundation` (the camera input device), `appkit`, `coreimage`
and `securetransport` on its own. Check what it made:

```sh
$prefix/bin/ffmpeg -hide_banner -hwaccels      # videotoolbox
$prefix/bin/ffmpeg -hide_banner -encoders | grep -E 'videotoolbox|openh264'
```

FFmpeg's dylibs carry their absolute install names, so a binary linked
against them finds them without `DYLD_LIBRARY_PATH`. Point the build at the
prefix — in `~/.zshenv`, so every shell has it:

```sh
export PATH="$HOME/.local/bin:$HOME/.local/ffmpeg-8.0/bin:$PATH"
export FFMPEG_DIR="$HOME/.local/ffmpeg-8.0"
export PKG_CONFIG_PATH="$HOME/.local/ffmpeg-8.0/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
```

`FFMPEG_DIR` is read before any pkg-config discovery, so it wins over any
other FFmpeg on the machine.

Two other routes, neither tried: Homebrew's `ffmpeg` is GPL and follows
FFmpeg's latest release, which may be past 8.0; vcpkg's `ffmpeg[openh264]:arm64-osx-dynamic` at the tag CI pins
(`2026.01.16`) would match CI's tree, if its port enables VideoToolbox.

### First build

```sh
cargo build -p media-pp
cargo test -p media-pp
cargo build --workspace
```

Then the features that are platform-neutral: `--features rnnoise`,
`--features webrtc`, `whisper` and `ort`. The workspace build compiles the
examples that turn on `whisper`, `ort` and `webrtc`; their tests are not yet
run on a Mac.

### Before calling anything done

The repository's own checks, as `AGENTS.md` lists them, with CI's toolchain
for clippy. Until there is a macOS CI job, a change made on the Mac must
also leave the Linux and Windows builds as they were — the other two
machines verify that.

## The backend a port adds

The other platforms each have one GPU backend that owns its frames from
capture to encode: D3D11 on Windows, CUDA (or Vulkan without NVIDIA) on
Linux. `MemoryDomain` is how the link check keeps them apart, and a frame of
one domain cannot enter an element of another without an explicit upload or
download.

### Metal and VideoToolbox, or Vulkan on MoltenVK

Two ways to get a GPU backend on a Mac:

- **Native: VideoToolbox frames on Metal.** VideoToolbox decodes and encodes
  (FFmpeg's `videotoolbox` hwaccel, `h264_videotoolbox`,
  `hevc_videotoolbox`), its frames are `CVPixelBuffer`s backed by
  `IOSurface`s, Metal textures wrap those without a copy, and a Metal
  compositor draws them. ScreenCaptureKit also hands over `IOSurface`s. This
  is the shape of the D3D11 and CUDA backends, and the one to aim for.
- **Vulkan on MoltenVK.** The Vulkan compositor, uploads, downloads and
  effects exist, and MoltenVK runs Vulkan on Metal. What stands in the way:
  - MoltenVK has no Vulkan Video, so `VulkanDecoder` and `VulkanEncoder`
    would refuse and every decode would be software.
  - Getting a VideoToolbox or ScreenCaptureKit `IOSurface` into a Vulkan
    image means `VK_EXT_metal_objects`, which nothing here uses.
  - `platform::vulkan::device::choose_device` creates its instance without
    `VK_KHR_portability_enumeration` and its create flag, so through the
    Vulkan loader a MoltenVK device is not listed at all and `NoDevice` is
    the likely answer.
  - FFmpeg's Vulkan hardware context on MoltenVK is untested here.
  - The `vulkan` feature's build needs an FFmpeg with `--enable-vulkan` and
    the Vulkan SDK's headers (`VULKAN_SDK`).

  It may still be worth a day early on: with the portability flag added, a
  compositor that runs on MoltenVK would put a picture on screen before the
  Metal compositor exists.

### What a native backend has to supply

Each item names where the existing backends plug in. A new memory domain is
a breaking change: `MemoryDomain` and `ElementType` are not
`#[non_exhaustive]`.

- **A memory domain** — `core/contract.rs:199`. The variant, `bit()`,
  `ALL`, `Display`, and the cross-domain `remedy()` hints and its
  per-domain layout match. `MemoryDomainSet` has three bits left.
- **Errors and element types** — each backend element has a `cfg`'d variant
  in `error.rs` (`Error` is `#[non_exhaustive]`) and an `ElementType` in
  `core/element.rs:50`.
- **The device** — the counterpart of `D3d11Gpu`, `CudaDevice` and
  `VulkanDevice`: one per process, shared by every element on it, holding
  the `MTLDevice`, a command queue, and FFmpeg's `AV_HWDEVICE_TYPE_VIDEOTOOLBOX`
  context. `ffmpeg-sys-next` does not bind `hwcontext_videotoolbox.h`;
  generate it in `build.rs` the way the `vulkan` feature generates
  `hwcontext_vulkan.h`.
- **Decode** — a `DecodeTarget` variant in
  `filter/decoder/video_decode_bin.rs:83` and every method that matches on
  it: `domain`, `decodes`, `keeps_alpha`, `converts_10bit`, `makes_rgb`,
  `hardware_format`, `convert`, `software_format`, `hardware_decoder`,
  `upload`, `made_rgb`, and the `no_video_device` and `refused` arms. The
  decoder itself follows `CudaDecoder` or `VulkanDecoder` (preroll, QoS,
  playing backwards); `decoder/hw_decoder.rs` `capable_decoder` answers
  whether FFmpeg has a VideoToolbox decoder for a codec.
- **Encode** — an `EncodeInput` variant (`encoder/video/video_encode_bin.rs:78`),
  its `for_decoded` mapping, an `EncodePath` (`:201`, `is_hardware`), an arm
  in `open_passing_over`, and a path function beside `d3d11`, `cuda` and
  `vulkan`, falling back to software through a download.
- **Compositor** — a compositor implementing `VideoCompositorControl`,
  `VideoLayerControl` and `TextLayerControl`
  (`elements/source/compositor/control.rs`) through the `compositor_control!`
  macro, as the software, D3D11, CUDA and Vulkan compositors do, including
  `RenderMode::Offline` through `timed_inputs.rs`.
- **Upload, download, conversion, effects** — the same set the other
  backends have: upload from system memory (NV12, BGRA, YUV420P as NV12),
  download, NV12 to BGRA, chroma key, video effect. The Vulkan elements
  compile WGSL with naga at construction; naga also writes Metal Shading
  Language (its `msl-out` feature), so the existing shaders may carry over
  without a shader toolchain.
- **Capture** — `Produce`s that say they are live (`is_live`), with
  `time_base()` and a `FrameRateHandle`, setting the device up in `starting`
  and letting it go in `stopping`, on the source's own thread, and stopping
  it for a pause in `pausing`/`resuming`, as the Windows and Linux captures
  do:
  - screen and window: ScreenCaptureKit (`SCStream` with an
    `SCContentFilter` for a display or a window), in system memory and as
    `IOSurface` frames on the device, as `PipeWireScreenCaptureSource` has
    `open` and `open_gpu`;
  - camera: done, `AvFoundationCaptureSource` — AVFoundation directly, an
    `AVCaptureVideoDataOutput` asking for NV12 (`420v`), in system memory
    or, with `videotoolbox`, the camera's own pixel buffers as VideoToolbox
    frames. `platform/macos/pixel_buffer.rs` does both for any
    `CVPixelBuffer`, for ScreenCaptureKit to reuse. Two things only a real
    camera showed: a session puts its camera back in its preset's format at
    every `startRunning` — `AVCaptureSessionPresetInputPriority` is refused
    on macOS — so the source holds the camera locked for configuration
    across each start; and the permission, like a tap's, is asked only for
    an application whose `Info.plist` has `NSCameraUsageDescription`, which
    Claude Code's lacks — macOS ends a program asking without it, so the
    library's camera tests skip unless already allowed, and a check against
    the camera runs as a bundled application;
  - audio: done, `CoreAudioCaptureSource` — devices directly, and what the
    system or one application plays through a Core Audio process tap
    (macOS 14.2+) with a private aggregate device, recorded through the same
    AUHAL input path. A tap needs the "System Audio Recording" permission,
    which macOS asks for only on behalf of an application whose `Info.plist`
    has `NSAudioCaptureUsageDescription`: a command-line program inherits
    whatever its terminal declares, and Claude Code declares none, so its
    taps record silence and a check that sound arrives has to run as a
    bundled application of its own. Taps hand over nothing while nothing
    plays; the source makes that silence up.
  - Device listing is what obs-rs calls: `list_devices`, `list_formats`,
    `list_applications`/`list_processes`, each with an `is_default` where
    there is one.
- **Audio output** — done: `CoreAudioRenderer` keeps the renderer contract
  `Player` uses (`app/player.rs`), `list_devices`, `open(name, options) ->
  (Self, AudioFormat)`, `format()` and a `Render`, as `WasapiRenderer` and
  `PipeWireAudioRenderer` do. It plays through an AUHAL output unit whose
  real-time callback reads a ring the renderer writes, and says where
  playback is from the callback's host timestamps plus the device's and its
  stream's latency. `Player` needs a window as well before it opens here.
- **A frame renderer for a program's own presenter** — what obs-rs's
  Preview is built on: `CudaFrameRenderer` hands over device pointers,
  `D3d11FrameRenderer` textures. A Metal one hands over the `IOSurface` or
  `MTLTexture` of each composited frame.
- **A window** — `window.rs`'s `WindowOptions`, `WindowEvents`,
  `WindowControl` and a key mapper (`Key::from_virtual_key` on Windows,
  `from_keysym` on Linux), a window renderer (`open`, `for_window` through
  raw-window-handle's AppKit handle, `window_control`), presentation delay
  (`presentation_delay.rs`), and `VideoWindow`'s `Gpu`/`Backend` aliases and
  `open_for_decoding` arm. Then `Player` (`app/mod.rs`, `lib.rs`) can be
  opened for macOS, with an `AudioOut` for the Core Audio renderer.
- **Features and gating** — a feature per backend part, as Windows has
  `d3d11`, `dxgi-capture`, `wasapi-renderer`, with its dependencies under a
  `cfg(target_os = "macos")` table (`objc2` and its framework crates are the
  usual route). The helpers every GPU backend shares list themselves in
  `cfg`s: `decoder/preroll_gate.rs` `hw_surface_budget`, `upload/nv12.rs`,
  `core/frame_size.rs` `try_get`, `core/color.rs`, `core/tone_map`,
  `test_support::encoder_session`, and the renderer gates in
  `elements/sink/renderer/mod.rs` and `filter/audio/stretcher.rs`.
- **Tests** — a `try_<device>()` helper in `test_support.rs` that prints
  `skipping: …` and returns `None` where there is no device, as
  `try_vulkan_device` and `try_cuda_device` do; a fixture font for macOS.
- **Examples** — the existing pattern: `media-pp` under a target table in
  the example's `Cargo.toml`, and a `cfg`-selected `main` that says which
  platforms it supports. A backend-bearing example adds macOS to both.

## A suggested order

1. ~~Build `-p media-pp` and run its tests on the Mac; fix what the first
   build finds (and this document). Fix `render_common` so the workspace
   builds.~~ Done.
2. Audio first: a Core Audio renderer and capture source. They are the
   smallest backend pieces, need no GPU, and make `Player` possible. Done:
   devices, and what the system or one application plays through a Core
   Audio process tap (macOS 14.2+).
3. Capture in system memory: ScreenCaptureKit and the camera, feeding the
   software compositor. That is enough for obs-rs to show and record a
   screen, slowly. The camera is done, `AvFoundationCaptureSource`, and
   already delivers VideoToolbox frames as well.
4. The Metal device, VideoToolbox decode and encode, upload and download.
   Done without Metal, all through FFmpeg: `VideoToolboxDevice`,
   `VideoToolboxDecoder`, `VideoToolboxEncoder`, `VideoToolboxUpload`,
   `VideoToolboxDownload`, `MemoryDomain::VideoToolbox`, and the
   `DecodeTarget` / `EncodeInput` arms — `transcode` runs on the media
   engine end to end. The Metal device comes with the compositor.
5. The Metal compositor and effects, then capture straight onto the device.
6. The window renderer and `Player`.
7. A macOS job in CI (`macos-latest` runners are Apple silicon).
