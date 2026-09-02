//! `capture -> NVENC -> FileMuxer`: records the desktop into a playable
//! `.mp4` with no CPU color conversion anywhere in the graph.
//!
//! The contrast with `screen_record_software` is the whole point. That example runs
//! `capture -> SwScaler -> SwEncoder`: every frame is converted BGRA->YUV420P
//! by libswscale and encoded on the CPU. Here NVENC consumes the captured
//! BGRA directly — it does its own color conversion inside the encode block —
//! so there is no `SwScaler` in this graph at all, and only the compressed
//! packets ever come back.
//!
//! Both platforms run the identical graph and terminal sink; only the GPU
//! stack differs. The pixels start GPU-resident on both: DXGI hands over a
//! texture under `CaptureMode::Gpu`, and PipeWire hands over a DMA-BUF that
//! `open_gpu` imports into a CUDA surface. Nothing in either branch copies a
//! frame through system memory.
//!
//! Needs an NVIDIA GPU and an ffmpeg build with NVENC; both encoders report a
//! typed error rather than panicking on anything else.
//!
//!     cargo run -p screen_record_nvenc -- <output.mp4> [seconds]

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!(
        "{} supports Windows (DXGI) and Linux (PipeWire) only",
        env!("CARGO_PKG_NAME")
    );
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common;

#[cfg(target_os = "windows")]
mod windows_example {
    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            CaptureMode, D3d11VideoCodec, D3d11VideoEncoder, D3d11VideoEncoderOptions,
            D3d11VideoInputFormat, DxgiCaptureOptions, DxgiCaptureSource, FileMuxer,
        },
        pipeline::Pipeline,
    };
    use render_common::D3d11GpuContext;

    use crate::common;

    pub fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let recording = common::parse_args("")?;

        // Opened first: `CaptureMode::Gpu` resolves the capture adapter, builds
        // its own device and hands it back. The encoder has to be built from
        // that same device — a texture from one `ID3D11Device` is not valid on
        // another, which is the invariant this whole D3D11 stack rests on.
        let capture_options = DxgiCaptureOptions {
            fps: 30,
            capture_mode: CaptureMode::Gpu,
            ..DxgiCaptureOptions::default()
        };
        let (source, format, device) = DxgiCaptureSource::open("screen", capture_options)?;
        let device = device.expect("CaptureMode::Gpu always returns a device");
        let gpu = D3d11GpuContext::new(Some(device))
            .map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let encoder = D3d11VideoEncoder::new(
            "encoder",
            gpu.device(),
            gpu.context(),
            D3d11VideoEncoderOptions {
                codec: D3d11VideoCodec::H264Nvenc,
                // What removes the conversion step: DXGI desktop duplication
                // produces BGRA and NVENC accepts BGRA textures as-is.
                input_format: D3d11VideoInputFormat::Bgra,
                width: format.width,
                height: format.height,
                time_base: format.time_base,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 8_000_000,
                gop_size: 60, // ~2s @ 30fps
                max_b_frames: None,
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let mut muxer = FileMuxer::create(&recording.path)?;
        muxer.add_stream("video", encoder.parameters(), format.time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("screen-record-nvenc", source, |source, ctx| {
            let branch = ctx
                .branch()
                // Thread boundary so a slow encode cannot stall capture; the
                // desktop keeps being duplicated at its own fixed rate.
                .queue("captured", 4)
                .pipe(encoder)
                .to(muxer_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        println!(
            "recording {}s of the desktop at {}x{} (h264_nvenc) to {} ...",
            recording.seconds, format.width, format.height, recording.path
        );
        common::record(&pipeline, recording.seconds)?;
        println!("wrote {}", recording.path);
        Ok(())
    }
}

/// The Linux half of the same example, and the same graph as
/// `windows_example` down to the element count: capture -> Queue -> NVENC ->
/// FileMuxer, same codec, same terminus. `open_gpu` negotiates DMA-BUF and
/// imports each captured buffer straight into a CUDA BGRA surface, so there
/// is no upload element here and the BGRA stays BGRA all the way into the
/// encode block — which is what keeps libswscale out of this graph.
///
/// The CLI differences are the same ones `screen_record_software` documents: Wayland
/// has no way to name a monitor, so the compositor prompts on the first run
/// and hands back a restore token that skips the prompt next time.
#[cfg(target_os = "linux")]
mod linux_example {
    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            CaptureSourceKind, CudaCodec, CudaDevice, CudaEncoder, CudaEncoderOptions,
            CudaFrameFormat, FileMuxer, PipeWireScreenCaptureOptions, PipeWireScreenCaptureSource,
        },
        pipeline::Pipeline,
    };

    use crate::common;

    pub fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let recording = common::parse_args(" [monitor|window] [restore-token]")?;

        // Monitor by default, matching the Windows branch's whole-desktop
        // capture. `window` is worth reaching for when one application is the
        // subject: a monitor stream stalls while any client is fullscreen,
        // where a window stream does not — see `PipeWireScreenCaptureSource`.
        let source_kind = match std::env::args().nth(3).as_deref() {
            Some("window") => CaptureSourceKind::Window,
            _ => CaptureSourceKind::Monitor,
        };
        // Last so it can simply be left off: it is a long opaque string that
        // only a repeat run has.
        let restore_token = std::env::args().nth(4);
        if restore_token.is_none() {
            eprintln!("opening the portal — approve the screen-share dialog to continue...");
        }

        // One CUDA context for the capture and the encoder — the invariant
        // every CUDA element in this crate is built around, and what the
        // encoder validates every incoming frame against. Built before the
        // capture because `open_gpu` allocates its surfaces from it.
        let cuda = CudaDevice::new().map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let (source, format, restore_token) = PipeWireScreenCaptureSource::open_gpu(
            "screen",
            PipeWireScreenCaptureOptions {
                fps: 30,
                source_kind,
                include_cursor: true,
                restore_token,
            },
            &cuda,
        )?;

        // The capture's own size, not a rounded-down one: the encoder is
        // fixed-size, so anything else would reject every frame. A stream
        // whose dimensions are odd would need a `CudaScaler` in between, and
        // NVENC reports that as a typed error at open rather than at the
        // first frame.
        let encoder = CudaEncoder::new(
            "encoder",
            &cuda,
            CudaEncoderOptions {
                codec: CudaCodec::H264,
                // The same choice the Windows branch makes, for the same
                // reason: what the capture produces is what NVENC ingests.
                input_format: CudaFrameFormat::Bgra,
                width: format.width,
                height: format.height,
                time_base: format.time_base,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 8_000_000,
                gop_size: 60, // ~2s @ 30fps
                max_b_frames: None,
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let mut muxer = FileMuxer::create(&recording.path)?;
        muxer.add_stream("video", encoder.parameters(), format.time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let (width, height) = (format.width, format.height);
        let pipeline = Pipeline::new("screen-record-nvenc", source, |source, ctx| {
            let branch = ctx
                .branch()
                // Thread boundary so the encode cannot stall capture; the
                // compositor keeps producing at its own rate.
                .queue("captured", 4)
                .pipe(encoder)
                .to(muxer_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        println!(
            "recording {}s of the desktop at {width}x{height} (h264_nvenc) to {} ...",
            recording.seconds, recording.path
        );
        common::record(&pipeline, recording.seconds)?;
        println!("wrote {}", recording.path);
        match restore_token {
            Some(token) => println!(
                "re-run without a dialog:\n  ... {} {} {} {token}",
                recording.path,
                recording.seconds,
                if matches!(source_kind, CaptureSourceKind::Window) {
                    "window"
                } else {
                    "monitor"
                }
            ),
            None => println!("the compositor issued no restore token; the next run will prompt"),
        }
        Ok(())
    }
}
