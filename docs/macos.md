# Porting media-pp to macOS

Where macOS stands, what a port has to add, and how to set a Mac up to do
it. Written from a read of the tree at `f420607` on a Linux machine: nothing
here has been built on a Mac yet, so the first build is where this document
starts being checked against reality. Correct it as you go.

obs-rs has a companion document, `docs/macos.md` in that repository, for the
application's half. The machine setup below serves both.

## Where macOS stands

Nothing in `lib/src` names macOS, Apple, VideoToolbox, Metal, Core Audio or
ScreenCaptureKit. What a macOS build gets is everything that is not a
platform backend:

- the pipeline core, `FileDemuxer`, `RtspSource`, `AppSource`/`AppSink`, the
  test sources, and every muxer (file, segmented, HLS, RTMP, RTSP, replay
  buffer);
- the audio mixer and audio filters, `SwAudioEncoder`;
- the software video elements — `SwDecoder`, `SwEncoder`, `SwScaler`,
  `SwVideoCompositor`, `SwChromaKey`, `SwVideoEffect` — and `VideoDecodeBin`
  / `VideoEncodeBin` with their `System` target only;
- the platform-neutral features: `rnnoise`, `webrtc`, `whisper`, `ort` (the
  last two build native code; not yet tried on a Mac).

It gets no capture of any kind, no audio output, no window or `VideoWindow`,
no `Player`, and no GPU backend that works: `cuda` compiles but finds no
driver, and `vulkan` is discussed below.

By reading, `cargo build -p media-pp` with no features should build on a Mac
with FFmpeg 8 found through pkg-config: every ungated `cfg` has a non-Windows
branch. Three things are known not to:

- **`cargo build --workspace`.** `examples/render/render_common` uses
  `media_pp` unconditionally but only depends on it under Linux and Windows
  target tables, and `av_playback` and `gpu_video_compositor` depend on
  `render_common` unconditionally. Build `-p media-pp` and single examples
  until that is fixed.
- **The test fixture without OpenH264.** `test_support::try_test_video`
  encodes its eight-second fixture with `VideoCodec::OpenH264`; a skip where
  it cannot, but `synthesize`, `synthesize_reordered`, `try_encoded_packets`
  and `try_tagged_packets` `expect()` it and panic. The FFmpeg the tests run
  against needs `libopenh264` — see the setup.
- **Text tests skip.** The font lists in `sw_video_compositor.rs`
  (`system_font`) and the CUDA and Vulkan compositor tests (`try_font`) hold
  only Windows and Linux paths. `/System/Library/Fonts/Helvetica.ttc` or
  `/System/Library/Fonts/AppleSDGothicNeo.ttc` (which obs-rs already uses)
  would do.

`lib/tests/common/mod.rs` `private_bytes` reads `/proc/self/statm` off
Windows and would panic on a Mac, but only the `#[ignore]`d soak scenarios
use it.

## Setting up the Mac

### Tools

1. **Xcode Command Line Tools**: `xcode-select --install`. This is the
   compiler, the SDK, and the libclang that bindgen needs — `ffmpeg-sys-next`
   runs bindgen on every build. Full Xcode is only needed later, for
   Instruments and for signing.
2. **Homebrew** (<https://brew.sh>), then `brew install pkgconf`.
3. **Rust**: `rustup` stable, plus the toolchain CI lints with. CI installs
   whatever stable is current, which has been newer than a local default
   before and failed lints nobody saw locally; check `rustup check` and
   install that version beside it (`rustup toolchain install 1.98.1` at the
   time of writing), then lint with `cargo +1.98.1 clippy …`.
4. The two repositories side by side, as the Linux and Windows machines have
   them: `~/work/media-pp` and `~/work/obs-rs`.

### FFmpeg

media-pp needs FFmpeg 8.0 or newer (libavcodec 62.8+); `lib/build.rs` says
so and stops on anything older. Two ways to get one:

- **Homebrew, to start**: `brew install ffmpeg`, then check
  `ffmpeg -version` says 8.x. Homebrew's build is GPL (it enables x264 and
  others). That is fine for development and never for a release archive.
  Check what it has:

  ```sh
  ffmpeg -hide_banner -hwaccels                   # videotoolbox
  ffmpeg -hide_banner -encoders | grep -E 'videotoolbox|openh264|x264'
  ffmpeg -hide_banner -buildconf | grep -E 'vulkan|openh264'
  ```

  If `libopenh264` is missing, the test fixture is (see above); build the
  vcpkg one instead, or add OpenH264 some other way before running the
  suite.
- **vcpkg, as CI builds it**: the pinned tree CI and the release use, which
  is what makes a Mac build comparable with the other two. The
  `.github/actions/setup-ffmpeg` action pins vcpkg `2026.01.16` and installs
  `ffmpeg[openh264,vulkan]` for `x64-windows` and `x64-linux-dynamic`; the
  Mac's triplet is `arm64-osx-dynamic`:

  ```sh
  git clone https://github.com/microsoft/vcpkg ~/vcpkg
  git -C ~/vcpkg checkout 2026.01.16
  ~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
  ~/vcpkg/vcpkg install "ffmpeg[openh264]:arm64-osx-dynamic"
  export FFMPEG_DIR=~/vcpkg/installed/arm64-osx-dynamic
  export DYLD_LIBRARY_PATH=$FFMPEG_DIR/lib   # for the tests to load it
  ```

  `FFMPEG_DIR` is read before any pkg-config discovery, so it wins over a
  Homebrew FFmpeg that is also installed. Nothing on the Mac has been built
  this way yet; the triplet and the features are the first things to check.

### First build

```sh
cd ~/work/media-pp
cargo build -p media-pp
cargo test -p media-pp
cargo build -p app_sink -p decode -p transcode   # examples with no backend
```

Then the features that are platform-neutral: `--features rnnoise`,
`--features webrtc`, and later `whisper` and `ort`, which build native code.

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
  - camera: FFmpeg's `avfoundation` input device, the way
    `V4l2CaptureSource` wraps `video4linux2` — or AVFoundation directly;
  - audio: Core Audio for devices; ScreenCaptureKit audio or Core Audio
    process taps (macOS 14.2+) for one application's sound, as
    `list_applications` / `open_application` offer on Linux.
  - Device listing is what obs-rs calls: `list_devices`, `list_formats`,
    `list_applications`/`list_processes`, each with an `is_default` where
    there is one.
- **Audio output** — the renderer contract `Player` uses
  (`app/player.rs`): `list_devices`, `open(name, options) -> (Self,
  AudioFormat)`, `format()`, a `Render` taking system-memory audio frames —
  `drain` to play out what it holds at the end, `reset` for a seek's flush,
  `stopping`, `pausing`/`resuming` for the device, as `WasapiRenderer` and
  `PipeWireAudioRenderer` are — registering as the audio master with the
  playback clock, and rate through `Stretcher`.
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

1. Build `-p media-pp` and run its tests on the Mac; fix what the first
   build finds (and this document). Fix `render_common` so the workspace
   builds.
2. Audio first: a Core Audio renderer and capture source. They are the
   smallest backend pieces, need no GPU, and make `Player` possible.
3. Capture in system memory: ScreenCaptureKit and the camera, feeding the
   software compositor. That is enough for obs-rs to show and record a
   screen, slowly.
4. The Metal device, VideoToolbox decode and encode, upload and download.
5. The Metal compositor and effects, then capture straight onto the device.
6. The window renderer and `Player`.
7. A macOS job in CI (`macos-latest` runners are Apple silicon).
