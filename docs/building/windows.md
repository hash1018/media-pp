# Building on Windows

What the `windows` job in [`ci.yml`](../../.github/workflows/ci.yml) does on
`windows-2025`, for a machine of your own. What every platform shares — which
FFmpeg, and why that one — is in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#setting-up-a-machine).

## Tools

- **Rust** stable on the MSVC toolchain, with `rustfmt` and `clippy`, and
  the Visual Studio Build Tools' C++ workload it links with — which vcpkg
  and whisper.cpp build with too.
- **libclang**, for the bindgen `ffmpeg-sys-next` runs on every build: LLVM,
  with `LIBCLANG_PATH` set to its `bin` where bindgen does not find it.
- **CMake**, for `whisper`'s whisper.cpp.

## FFmpeg

vcpkg at the tag CI pins, into a dynamic tree:

```powershell
git clone --branch 2026.01.16 https://github.com/microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat -disableMetrics
.\vcpkg\vcpkg.exe install "ffmpeg[openh264,vulkan]:x64-windows"
$env:FFMPEG_DIR = "$PWD\vcpkg\installed\x64-windows"
$env:PATH = "$env:FFMPEG_DIR\bin;$env:PATH"
```

`openh264` is the test fixture's encoder, and `vulkan` is only for the
crate's `vulkan` feature: FFmpeg's Vulkan context, and the Vulkan headers it
includes, which vcpkg installs beside it. The DLLs are found at run time
through `PATH`; without it everything compiles, and every test binary fails
to start with `STATUS_DLL_NOT_FOUND`.

## Features that need more

- `ort`: building it downloads ONNX Runtime and links its DLLs into the
  target directory as symlinks, which need Developer Mode. A `cargo check`
  or `cargo doc` with `DOCS_RS=1` skips the download, which is how CI checks
  it.
- `cuda`: only the NVIDIA driver; the kernels ship as PTX. `nvcuda.dll` is
  linked by name, so a binary built with `cuda` does not start on a machine
  without the driver.
- `whisper-vulkan`: the Vulkan SDK's shader compiler at build time, long
  paths enabled, and a short target directory — see
  [`transcribe`](../../examples/core/transcribe/README.md).
- The soak scenarios' D3D11 live-object count needs the D3D11 debug layer,
  which the optional Windows feature *Graphics Tools* installs; without it
  those scenarios skip.

## Checks

As CI runs them:

```powershell
cargo fmt --all -- --check
cargo clippy -p media-pp --all-targets --features d3d11,d3d12,dxgi-capture,wgc-capture,mf-capture,wasapi-capture,wasapi-renderer,vulkan -- -D warnings
cargo test -p media-pp --features d3d11,d3d12,dxgi-capture,wgc-capture,mf-capture,wasapi-capture,wasapi-renderer,vulkan
```

and the documentation with `$env:RUSTDOCFLAGS = "-D warnings"`, one feature
set at a time. The control conformance runs pinned to two cores are in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#control-sequences).
