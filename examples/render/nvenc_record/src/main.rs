//! `AppSource -> SwScaler(NV12) -> upload -> NVENC -> Mp4Muxer`: encodes
//! GPU-resident frames on the GPU's own NVENC block straight into a playable
//! `.mp4`, with no CPU readback anywhere after the upload.
//!
//! The contrast with a software tail is the point: `gpu_video_compositor`'s
//! recording branch has to run `Download -> SwScaler -> SwEncoder`, pulling
//! every frame back over PCIe and converting and encoding it on the CPU,
//! because `SwEncoder` has no GPU input path. Here the frame stays on the GPU
//! from the upload onward.
//!
//! Both platforms run the identical graph and CLI; only the GPU stack
//! differs — `D3d11Upload`/`D3d11NvencEncoder` on Windows,
//! `CudaUpload`/`CudaEncoder` on Linux. Needs an NVIDIA GPU and an ffmpeg
//! build with NVENC; both encoders report a typed error rather than panicking
//! on anything else. No window and no media file are involved, so this runs
//! headless.
//!
//!     cargo run -p nvenc_record -- [output.mp4] [seconds]

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod common;

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn main() {
    eprintln!("{} supports Windows and Linux only", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "linux")]
fn main() -> impl std::process::Termination {
    linux_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use ffmpeg_next as ffmpeg;
    use media_pp::{
        elements::{
            AppSource, D3d11NvencCodec, D3d11NvencEncoder, D3d11NvencEncoderOptions,
            D3d11NvencInputFormat, D3d11Upload, Mp4Muxer, SwScaler,
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
        let recording = common::parse_args()?;
        let (width, height) = (recording.width, recording.height);

        // One device and one shared immediate context for both D3D11 stages —
        // the invariant every D3D11 element in this crate is built around.
        let gpu =
            D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;
        let (source, source_handle) = AppSource::new("source", 8);

        let encoder = D3d11NvencEncoder::new(
            "encoder",
            gpu.device(),
            gpu.context(),
            D3d11NvencEncoderOptions {
                codec: D3d11NvencCodec::H264,
                // D3d11Upload produces NV12 textures. Feeding this element a
                // D3d11VideoCompositor or DxgiCaptureSource GPU-mode output
                // instead means `Bgra` here and no SwScaler at all — NVENC takes
                // BGRA textures directly.
                input_format: D3d11NvencInputFormat::Nv12,
                width,
                height,
                time_base: recording.time_base,
                frame_rate: recording.frame_rate,
                bit_rate: 4_000_000,
                gop_size: 60,
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let mut muxer = Mp4Muxer::create(&recording.path)?;
        muxer.add_stream("video", encoder.parameters(), recording.time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("nvenc-record", source, |source, ctx| {
            // AppSource emits YUV420P on the CPU, so this one SwScaler is the
            // only format conversion in the graph; the upload requires NV12
            // and everything downstream of it is GPU-resident.
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = D3d11Upload::new("upload", gpu.device(), width, height);
            let branch = ctx
                .branch()
                .pipe(scaler)
                .pipe(upload)
                .queue("encode-frames", 8)
                .pipe(encoder)
                .to(muxer_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        println!(
            "recording {}s of {width}x{height} h264_nvenc to {} ...",
            recording.frame_count / 30,
            recording.path
        );
        pipeline.run();
        let feeder = common::spawn_feeder(source_handle, width, height, recording.frame_count);
        common::finish(&pipeline, feeder, &recording.path)
    }
}

#[cfg(target_os = "linux")]
mod linux_example {
    use ffmpeg_next as ffmpeg;
    use media_pp::{
        elements::{
            AppSource, CudaCodec, CudaDevice, CudaEncoder, CudaEncoderOptions, CudaUpload,
            Mp4Muxer, SwScaler,
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
        let recording = common::parse_args()?;
        let (width, height) = (recording.width, recording.height);

        // One CUDA context for both stages — the invariant every CUDA element
        // in this crate is built around, and what the encoder validates every
        // incoming frame against.
        let cuda = CudaDevice::new().map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let (source, source_handle) = AppSource::new("source", 8);

        let encoder = CudaEncoder::new(
            "encoder",
            &cuda,
            CudaEncoderOptions {
                codec: CudaCodec::H264,
                width,
                height,
                time_base: recording.time_base,
                frame_rate: recording.frame_rate,
                bit_rate: 4_000_000,
                gop_size: 60,
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let mut muxer = Mp4Muxer::create(&recording.path)?;
        muxer.add_stream("video", encoder.parameters(), recording.time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("nvenc-record", source, |source, ctx| {
            // AppSource emits YUV420P on the CPU, so this one SwScaler is the
            // only format conversion in the graph; the upload requires NV12
            // and everything downstream of it is GPU-resident.
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new("upload", &cuda, width, height)
                .map_err(|e| media_pp::Error::Other(e.to_string()))?;
            let branch = ctx
                .branch()
                .pipe(scaler)
                .pipe(upload)
                .queue("encode-frames", 8)
                .pipe(encoder)
                .to(muxer_sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })?;

        println!(
            "recording {}s of {width}x{height} h264_nvenc to {} ...",
            recording.frame_count / 30,
            recording.path
        );
        pipeline.run();
        let feeder = common::spawn_feeder(source_handle, width, height, recording.frame_count);
        common::finish(&pipeline, feeder, &recording.path)
    }
}
