//! Rejects an FFmpeg older than 8.0 at build time, with the `vulkan`
//! feature generates the bindings to FFmpeg's Vulkan hardware context, and
//! with `ort-cuda` links the NVIDIA libraries ONNX Runtime's providers need.
//!
//! `ffmpeg-sys-next` (`links = "ffmpeg"`) probes the installed libraries and
//! republishes what it found as build metadata. Cargo forwards that metadata
//! only to the build script of a crate that depends on it *directly*, which is
//! the sole reason `Cargo.toml` names `ffmpeg-sys-next` — no code in this
//! crate uses it, the FFI comes through `ffmpeg::ffi`.
//!
//! The check reads what `ffmpeg-sys-next` detected on *its* last run. That
//! build script declares no `rerun-if-env-changed`, so repointing
//! `PKG_CONFIG_PATH` at a different FFmpeg leaves both its generated bindings
//! and this gate stale until `cargo clean -p ffmpeg-sys-next`.

/// Set to `"true"` when the detected libavcodec is 62.8 or newer, which is
/// FFmpeg 8.0. `ffmpeg-sys-next` publishes the key either way, so an empty
/// value means "detected, and older".
const DETECTED_FFMPEG_8_0: &str = "DEP_FFMPEG_FFMPEG_8_0";

fn main() {
    println!("cargo:rerun-if-env-changed={DETECTED_FFMPEG_8_0}");

    #[cfg(feature = "vulkan")]
    vulkan::bind();

    // docs.rs documents rather than links, and its image supplies whatever
    // FFmpeg it supplies; failing there would break the published docs without
    // protecting anyone's runtime.
    if std::env::var_os("DOCS_RS").is_some() {
        return;
    }

    #[cfg(any(feature = "ort-cuda", feature = "cuda-visual-tracking"))]
    nvidia::link();

    if std::env::var(DETECTED_FFMPEG_8_0).as_deref() != Ok("true") {
        println!(
            "cargo::error=media-pp requires FFmpeg 8.0 or newer (libavcodec 62.8+), but \
             ffmpeg-sys-next found an older installation. Point PKG_CONFIG_PATH at an FFmpeg 8 \
             build, then run `cargo clean -p ffmpeg-sys-next` so its bindings are regenerated."
        );
    }
}

/// Bindings to `libavutil/hwcontext_vulkan.h`, which `ffmpeg-sys-next` does
/// not generate.
///
/// Generated from the installed header rather than written out by hand: the
/// structs there change with FFmpeg's version and its deprecation switches
/// (`AVVulkanDeviceContext` loses five pairs of fields with
/// `FF_API_VULKAN_FIXED_QUEUES`), and a hand-mirrored FFmpeg hardware
/// context is what corrupted memory in this crate's D3D11VA history. bindgen
/// also emits layout assertions, so a header the bindings were not made from
/// fails the build rather than a frame.
///
/// FFmpeg installs that header only when it was built with Vulkan, and it
/// includes `vulkan/vulkan.h`; either missing is a build error that says so.
#[cfg(feature = "vulkan")]
mod vulkan {
    use std::path::{Path, PathBuf};

    const HEADER: &str = "libavutil/hwcontext_vulkan.h";

    pub(super) fn bind() {
        println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
        println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
        println!("cargo:rerun-if-env-changed=VULKAN_SDK");

        let ffmpeg = ffmpeg_includes();
        let Some(header) = ffmpeg
            .iter()
            .map(|dir| dir.join(HEADER))
            .find(|path| path.is_file())
        else {
            println!(
                "cargo::error=the `vulkan` feature needs an FFmpeg built with Vulkan \
                 (`--enable-vulkan`, or vcpkg's `ffmpeg[vulkan]`), and {HEADER} was not \
                 found under {ffmpeg:?}. Set FFMPEG_DIR, or PKG_CONFIG_PATH, to that FFmpeg."
            );
            return;
        };
        println!("cargo:rerun-if-changed={}", header.display());

        let mut includes = ffmpeg.clone();
        includes.extend(vulkan_includes());
        if !includes
            .iter()
            .any(|dir| dir.join("vulkan/vulkan.h").is_file())
            && !cfg!(target_os = "linux")
        {
            println!(
                "cargo::error=the `vulkan` feature needs the Vulkan headers, which \
                 {HEADER} includes: install the Vulkan SDK (and set VULKAN_SDK), or \
                 vcpkg's `vulkan-headers`, which `ffmpeg[vulkan]` installs beside FFmpeg's own."
            );
            return;
        }

        let bindings = bindgen::Builder::default()
            .header(header.to_string_lossy())
            .clang_args(includes.iter().map(|dir| format!("-I{}", dir.display())))
            .allowlist_type("AVVulkanDeviceContext|AVVulkanDeviceQueueFamily")
            .allowlist_type("AVVulkanFramesContext|AVVkFrame|AVVkFrameFlags")
            .allowlist_function("av_vk_frame_alloc|av_vkfmt_from_pixfmt")
            // FFmpeg's own types come from `ffmpeg-sys-next`, as one type
            // everywhere; the Vulkan ones are generated with the rest.
            .blocklist_type("AVHWDeviceContext|AVHWFramesContext|AVPixelFormat")
            .raw_line(
                "use ffmpeg_next::ffi::{AVHWDeviceContext, AVHWFramesContext, AVPixelFormat};",
            )
            .default_enum_style(bindgen::EnumVariation::Consts)
            .derive_default(true)
            .layout_tests(true)
            .generate_comments(false)
            .generate();
        let bindings = match bindings {
            Ok(bindings) => bindings,
            // On Linux the headers are looked for where the compiler looks,
            // so a machine without them is only found out here — said as
            // plainly as the check above says it elsewhere.
            Err(error) if error.to_string().contains("vulkan/vulkan.h") => {
                println!(
                    "cargo::error=the `vulkan` feature needs the Vulkan headers, which \
                     {HEADER} includes, and the compiler found none: install your \
                     distribution's (`libvulkan-dev` on Debian and Ubuntu, \
                     `vulkan-headers` on Fedora and Arch), or set VULKAN_SDK to a \
                     directory with them under `include`."
                );
                return;
            }
            Err(error) => panic!("generating bindings to {HEADER}: {error}"),
        };
        let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
        bindings
            .write_to_file(out.join("hwcontext_vulkan.rs"))
            .expect("write the Vulkan hardware context bindings");
    }

    /// Where FFmpeg's headers are, found as `ffmpeg-sys-next` finds them:
    /// `FFMPEG_DIR`, else pkg-config's `libavutil`.
    fn ffmpeg_includes() -> Vec<PathBuf> {
        if let Some(dir) = std::env::var_os("FFMPEG_DIR") {
            return vec![Path::new(&dir).join("include")];
        }
        pkg_config::Config::new()
            .cargo_metadata(false)
            .env_metadata(false)
            .probe("libavutil")
            .map(|library| library.include_paths)
            .unwrap_or_default()
    }

    /// Where the Vulkan headers may be besides FFmpeg's own directory, where
    /// vcpkg puts them: the Vulkan SDK's. On Linux the compiler's own
    /// search path is where a distribution's `libvulkan-dev` puts them.
    fn vulkan_includes() -> Vec<PathBuf> {
        let Some(sdk) = std::env::var_os("VULKAN_SDK") else {
            return Vec::new();
        };
        ["Include", "include"]
            .iter()
            .map(|dir| Path::new(&sdk).join(dir))
            .filter(|dir| dir.is_dir())
            .collect()
    }
}

/// Links the libraries ONNX Runtime's CUDA provider needs — CUDA 13's
/// runtime, cuBLAS and cuRAND, and cuDNN 9 — and with `ort-tensorrt` those
/// its TensorRT provider needs besides, TensorRT 10 and its ONNX parser;
/// and with `cuda-visual-tracking`, cuFFT 12, which `ObjectTracker`
/// transforms its correlation filters with on CUDA pictures.
///
/// Linked rather than left to the providers, which open them only when a
/// detector starts: an application built with the feature is one for
/// NVIDIA GPUs, and like a missing FFmpeg, a missing library stops it
/// before it starts, by name, rather than when a detector is asked for.
/// What linking cannot say, a version too old, `CudaOrtDetector::runtime`
/// asks each library at run time. The driver is not linked: a build
/// machine has none.
///
/// The link keeps a library only while the program calls into it — rustc
/// passes `--as-needed` — and `CudaOrtDetector::runtime` calls each for
/// its version, so an application that never makes a detector does not
/// need them either.
///
/// The libraries are looked for in `MEDIA_PP_NVIDIA_LIB_DIRS`, a path
/// list, then where the loader will look for them — `LD_LIBRARY_PATH` on
/// Linux, `LIB` and `PATH` on Windows — then the CUDA installation and the
/// system's directories. One not found is said in a warning and linked by
/// name anyway, which leaves the linker to look and to fail: a check, which
/// never links, still passes. Where the program finds them at run time is
/// the application's to arrange, as with FFmpeg.
#[cfg(any(feature = "ort-cuda", feature = "cuda-visual-tracking"))]
mod nvidia {
    use std::path::PathBuf;

    /// Each library as the linker looks for it on Linux: the versioned name,
    /// since NVIDIA's pip wheels ship no unversioned one.
    #[cfg(feature = "ort-cuda")]
    const CUDA: &[&str] = &[
        "libcudart.so.13",
        "libcublasLt.so.13",
        "libcublas.so.13",
        "libcurand.so.10",
        "libcudnn.so.9",
    ];
    #[cfg(feature = "ort-tensorrt")]
    const TENSORRT: &[&str] = &["libnvinfer.so.10", "libnvonnxparser.so.10"];
    #[cfg(feature = "cuda-visual-tracking")]
    const CUFFT: &[&str] = &["libcufft.so.12"];

    pub(super) fn link() {
        // On Windows the extern blocks name each DLL themselves
        // (`raw-dylib`): NVIDIA's wheels there carry the DLLs and no import
        // libraries, so there is nothing for a linker to find, and nothing
        // has to be there to build — only, as here, to run.
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
            return;
        }
        for variable in ["MEDIA_PP_NVIDIA_LIB_DIRS", "LD_LIBRARY_PATH", "CUDA_PATH"] {
            println!("cargo:rerun-if-env-changed={variable}");
        }

        #[allow(unused_mut)]
        let mut libraries: Vec<&str> = Vec::new();
        #[cfg(feature = "ort-cuda")]
        libraries.extend_from_slice(CUDA);
        #[cfg(feature = "ort-tensorrt")]
        libraries.extend_from_slice(TENSORRT);
        #[cfg(feature = "cuda-visual-tracking")]
        libraries.extend_from_slice(CUFFT);

        let places = places();
        let mut found: Vec<&PathBuf> = Vec::new();
        let mut missing = Vec::new();
        for library in libraries {
            match places.iter().find(|dir| dir.join(library).is_file()) {
                Some(dir) if !found.contains(&dir) => found.push(dir),
                Some(_) => {}
                None => missing.push(library),
            }
            println!("cargo:rustc-link-lib=dylib:+verbatim={library}");
        }
        for dir in found {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        if !missing.is_empty() {
            println!(
                "cargo::warning=media-pp links {missing:?} for its NVIDIA features, which were \
                 not found: linking will fail. Name their directories in MEDIA_PP_NVIDIA_LIB_DIRS."
            );
        }
    }

    /// Where to look, in order: what the builder named, where the loader
    /// will look, the CUDA installation, and the system's directories —
    /// the last only for a build for this machine.
    fn places() -> Vec<PathBuf> {
        let mut places = Vec::new();
        for variable in ["MEDIA_PP_NVIDIA_LIB_DIRS", "LD_LIBRARY_PATH"] {
            if let Some(value) = std::env::var_os(variable) {
                places.extend(std::env::split_paths(&value));
            }
        }
        if let Some(cuda) = std::env::var_os("CUDA_PATH") {
            places.push(PathBuf::from(cuda).join("lib64"));
        }
        if std::env::var("HOST").ok() == std::env::var("TARGET").ok() {
            places.extend(
                [
                    "/usr/local/cuda/lib64",
                    "/usr/lib/x86_64-linux-gnu",
                    "/usr/lib/aarch64-linux-gnu",
                    "/usr/lib64",
                    "/usr/lib",
                ]
                .map(PathBuf::from),
            );
        }
        places
    }
}
