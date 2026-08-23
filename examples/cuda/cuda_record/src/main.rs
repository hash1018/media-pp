//! `AppSource -> SwScaler(NV12) -> CudaUpload -> CudaEncoder -> Mp4Muxer`:
//! encodes GPU-resident frames on the GPU's own NVENC block straight into a
//! playable `.mp4`, with no CPU readback anywhere after the upload.
//!
//! The contrast with a software tail is the point: a recording branch that
//! ends in `SwEncoder` has to run `CudaDownload -> SwScaler -> SwEncoder`,
//! pulling every frame back over PCIe and converting and encoding it on the
//! CPU. Here the frame stays on the GPU from the upload onward.
//!
//! Nothing here is platform-specific — CUDA is a vendor backend, not a Linux
//! one, so this runs unchanged on Windows and Linux. `nvenc_record` is the
//! D3D11 counterpart for the same graph. Needs an NVIDIA GPU and an ffmpeg
//! build with NVENC; `CudaEncoder` reports a typed error rather than panicking
//! on anything else. No window and no media file are involved, so this runs
//! headless.
//!
//!     cargo run -p cuda_record -- [output.mp4] [seconds]

fn main() -> impl std::process::Termination {
    example::run()
}

mod common;

mod example {
    use media_pp::ffmpeg;
    use media_pp::{
        elements::{
            AppSource, CudaCodec, CudaDevice, CudaEncoder, CudaEncoderOptions, CudaFrameFormat,
            CudaUpload, Mp4Muxer, SwScaler,
        },
        pipeline::Pipeline,
    };

    use crate::common;

    pub(super) fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let recording = common::parse_args()?;
        let (width, height) = (recording.width, recording.height);

        // One CUDA context for both stages — the invariant every CUDA element in
        // this crate is built around, and what the encoder validates every
        // incoming frame against.
        let cuda = CudaDevice::new().map_err(|e| media_pp::Error::Other(e.to_string()))?;
        let (source, source_handle) = AppSource::new("source", 8);

        let encoder = CudaEncoder::new(
            "encoder",
            &cuda,
            CudaEncoderOptions {
                codec: CudaCodec::H264,
                input_format: CudaFrameFormat::Nv12,
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

        let pipeline = Pipeline::new("cuda-record", source, |source, ctx| {
            // AppSource emits YUV420P on the CPU, so this one SwScaler is the only
            // format conversion in the graph; the upload requires NV12 and
            // everything downstream of it is GPU-resident.
            let scaler = SwScaler::new(
                "to-nv12",
                ffmpeg::format::Pixel::NV12,
                width,
                height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            );
            let upload = CudaUpload::new("upload", &cuda, CudaFrameFormat::Nv12, width, height)
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
        pipeline.run()?;
        let feeder = common::spawn_feeder(source_handle, width, height, recording.frame_count);
        common::finish(&pipeline, feeder, &recording.path)
    }
}
