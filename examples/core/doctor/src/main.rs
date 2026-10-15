//! What this machine gives media-pp, and what it lacks: the FFmpeg the
//! program found and whether it is the one media-pp is built for, the
//! encoders that FFmpeg carries, the hardware it can decode with, and
//! whether each GPU backend this build has opens a device here. Each line
//! says what it found; each problem says what to do about it.
//!
//! No pipeline runs. The checks are FFmpeg's own answers — its version, its
//! encoders, a hardware device context made and let go of — and a device of
//! each of media-pp's backends made and let go of: D3D11 and D3D12 on
//! Windows, CUDA on Windows and Linux, Metal on macOS, and Vulkan when built
//! with `--features vulkan`.
//!
//! A program that cannot find FFmpeg's libraries at all does not get as far
//! as `main` — Windows says `STATUS_DLL_NOT_FOUND`, Linux names the missing
//! `.so` — and that, too, is the answer: the libraries are not where the
//! loader looks. `scripts/setup-ffmpeg` sets that up.
//!
//!     cargo run -p doctor
//!     cargo run -p doctor --features vulkan

fn main() -> std::process::ExitCode {
    doctor::run()
}

mod doctor {
    use std::ffi::CStr;
    use std::process::ExitCode;

    use media_pp::ffmpeg::{self, ffi};

    /// The FFmpeg libraries' major versions media-pp is built against:
    /// FFmpeg 8.0's.
    const WANTED: [(&str, u32); 3] = [("libavcodec", 62), ("libavformat", 62), ("libavutil", 60)];

    /// What each check found.
    #[derive(Default)]
    struct Report {
        problems: Vec<String>,
    }

    impl Report {
        fn ok(&self, what: &str, detail: impl std::fmt::Display) {
            println!("  ok    {what}: {detail}");
        }

        fn note(&self, what: &str, detail: impl std::fmt::Display) {
            println!("  --    {what}: {detail}");
        }

        /// A problem, and what to do about it.
        fn problem(&mut self, what: &str, detail: impl std::fmt::Display, fix: &str) {
            println!("  FAIL  {what}: {detail}");
            println!("        -> {fix}");
            self.problems.push(what.to_owned());
        }
    }

    pub(super) fn run() -> ExitCode {
        let _log_guard = match media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        ) {
            Ok(guard) => Some(guard),
            Err(error) => {
                eprintln!("logging is off: {error}");
                None
            }
        };
        let mut report = Report::default();
        println!("media-pp doctor\n\nFFmpeg");
        ffmpeg_libraries(&mut report);
        encoders(&report);
        hardware_contexts(&report);
        println!("\nGPU backends in this build");
        devices(&mut report);

        println!();
        if report.problems.is_empty() {
            println!("Nothing wrong found.");
            ExitCode::SUCCESS
        } else {
            println!(
                "{} problem(s): {}. See docs/building/ for this platform.",
                report.problems.len(),
                report.problems.join(", ")
            );
            ExitCode::FAILURE
        }
    }

    /// `major.minor.micro` of a packed FFmpeg version number.
    fn version(packed: u32) -> (u32, u32, u32) {
        (packed >> 16, (packed >> 8) & 0xff, packed & 0xff)
    }

    /// The libraries the program loaded, against the headers it was built
    /// from and the release media-pp wants.
    fn ffmpeg_libraries(report: &mut Report) {
        let loaded = [
            ("libavcodec", ffmpeg::codec::version()),
            ("libavformat", ffmpeg::format::version()),
            ("libavutil", ffmpeg::util::version()),
        ];
        let headers = [
            ("libavcodec", ffi::LIBAVCODEC_VERSION_MAJOR as u32),
            ("libavformat", ffi::LIBAVFORMAT_VERSION_MAJOR as u32),
            ("libavutil", ffi::LIBAVUTIL_VERSION_MAJOR as u32),
        ];
        for ((name, packed), (_, built)) in loaded.iter().zip(headers) {
            let (major, minor, micro) = version(*packed);
            let wanted = WANTED.iter().find(|(n, _)| n == name).map(|(_, v)| *v);
            if Some(major) != wanted {
                report.problem(
                    name,
                    format!(
                        "{major}.{minor}.{micro} loaded, FFmpeg 8.0's is {}",
                        wanted.unwrap_or(0)
                    ),
                    "install FFmpeg 8.0 — scripts/setup-ffmpeg builds the one CI uses",
                );
            } else if major != built {
                report.problem(
                    name,
                    format!("{major}.{minor}.{micro} loaded, but built against headers of {built}"),
                    "point FFMPEG_DIR and the library path at the same FFmpeg tree, then rebuild",
                );
            } else {
                report.ok(name, format!("{major}.{minor}.{micro}"));
            }
        }
        report.note("licence", ffmpeg::util::license());
    }

    /// Which of the encoders media-pp can open this FFmpeg carries. Whether
    /// a hardware one opens depends on the GPU too, which the devices below
    /// say.
    fn encoders(report: &Report) {
        let named = |names: &[&str]| -> String {
            names
                .iter()
                .map(|name| {
                    let here = ffmpeg::encoder::find_by_name(name).is_some();
                    format!("{name} {}", if here { "yes" } else { "no" })
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        report.note(
            "H.264 encoders",
            named(&[
                "libopenh264",
                "h264_nvenc",
                "h264_mf",
                "h264_videotoolbox",
                "h264_vaapi",
                "h264_vulkan",
            ]),
        );
        report.note(
            "HEVC encoders",
            named(&["hevc_nvenc", "hevc_mf", "hevc_videotoolbox", "hevc_d3d12va"]),
        );
        report.note("AAC encoder", named(&["aac"]));
    }

    /// Each kind of hardware this FFmpeg can decode with, and whether it
    /// makes a device context of it on this machine — what a hardware
    /// decoder does first.
    fn hardware_contexts(report: &Report) {
        let mut kind = ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;
        let mut found = Vec::new();
        loop {
            // SAFETY: iterates FFmpeg's static table, starting from NONE.
            kind = unsafe { ffi::av_hwdevice_iterate_types(kind) };
            if kind == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                break;
            }
            // SAFETY: a type FFmpeg itself just handed out names a static
            // string.
            let name = unsafe { CStr::from_ptr(ffi::av_hwdevice_get_type_name(kind)) }
                .to_string_lossy()
                .into_owned();
            let mut context = std::ptr::null_mut();
            // SAFETY: a context made on the default device of this type, with
            // no options, and freed at once where made.
            let made = unsafe {
                let code = ffi::av_hwdevice_ctx_create(
                    &mut context,
                    kind,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                );
                if code >= 0 {
                    ffi::av_buffer_unref(&mut context);
                }
                code >= 0
            };
            found.push(format!("{name} {}", if made { "yes" } else { "no" }));
        }
        report.note("hardware decoding", found.join(", "));
    }

    /// A device of each backend this build has, made and let go of.
    fn devices(report: &mut Report) {
        #[cfg(target_os = "windows")]
        {
            match media_pp::elements::D3d11Gpu::new() {
                Ok(_) => report.ok("D3D11", "device made"),
                Err(error) => report.problem("D3D11", error, "update the GPU driver"),
            }
            match media_pp::elements::D3d12Gpu::new() {
                Ok(_) => report.ok("D3D12", "device made"),
                Err(error) => report.problem("D3D12", error, "update the GPU driver"),
            }
        }
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        {
            match media_pp::elements::CudaDevice::new() {
                Ok(_) => report.ok("CUDA", "device made"),
                // Not a problem on a machine without NVIDIA: everything else
                // still works, and the GPU elements of another backend take
                // its place.
                Err(error) => report.note("CUDA", format!("none here ({error})")),
            }
        }
        #[cfg(target_os = "macos")]
        {
            match media_pp::elements::VideoToolboxDevice::new() {
                Ok(_) => report.ok("VideoToolbox and Metal", "device made"),
                Err(error) => report.problem(
                    "VideoToolbox and Metal",
                    error,
                    "macOS 13 or newer on Apple silicon is what this is built for",
                ),
            }
        }
        #[cfg(feature = "vulkan")]
        {
            match media_pp::elements::VulkanDevice::new() {
                Ok(device) => report.ok("Vulkan", device.name()),
                Err(error) => report.problem(
                    "Vulkan",
                    error,
                    "install the GPU's Vulkan driver — Mesa's on AMD and Intel under Linux",
                ),
            }
        }
        #[cfg(not(feature = "vulkan"))]
        report.note(
            "Vulkan",
            "not in this build; run with --features vulkan to check it",
        );
        if !cfg!(any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos",
            feature = "vulkan"
        )) {
            report.note("GPU", "none on this platform: software elements only");
        }
    }
}
