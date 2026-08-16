#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("{} example only supports Windows", env!("CARGO_PKG_NAME"));
}

#[cfg(target_os = "windows")]
fn main() -> impl std::process::Termination {
    windows_example::run()
}

#[cfg(target_os = "windows")]
mod windows_example {
    use std::{thread, time::Duration};

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            D3d11NvencCodec, D3d11NvencEncoder, D3d11NvencEncoderOptions, D3d11NvencInputFormat,
            D3d11Upload, Mp4Muxer, Scaler, TestVideoOptions, TestVideoSource,
        },
        pipeline::Pipeline,
    };
    use render_common::D3d11GpuContext;

    /// TestVideoSource -> Scaler(NV12) -> D3d11Upload -> D3d11NvencEncoder ->
    /// Mp4Muxer: encodes GPU-resident textures on the GPU's own NVENC block
    /// straight into a playable `.mp4`, with no CPU readback anywhere after
    /// the upload.
    ///
    /// This is the hardware counterpart to what `gpu_video_compositor`'s
    /// recording branch does in software, and the contrast is the point: that
    /// branch has to run `D3d11Download -> Scaler -> SwEncoder`, pulling every
    /// frame back over PCIe and converting and encoding it on the CPU, because
    /// `SwEncoder` has no GPU input path. Here the frame stays on the GPU from
    /// `D3d11Upload` onward.
    ///
    /// Needs an NVIDIA GPU and an ffmpeg build with NVENC; `D3d11NvencEncoder`
    /// reports a typed error rather than panicking on anything else. No window
    /// and no media file are involved, so this runs headless.
    ///
    ///     cargo run -p nvenc_record -- [output.mp4] [seconds]
    pub fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "nvenc_record.mp4".into());
        let seconds: u64 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        let (width, height) = (1280u32, 720u32);
        let frame_rate = ffmpeg::Rational::new(30, 1);

        // One device and one shared immediate context for both D3D11 stages —
        // the invariant every D3D11 element in this crate is built around.
        let gpu =
            D3d11GpuContext::new(None).map_err(|e| media_pp::Error::Other(format!("{e:?}")))?;

        let source = TestVideoSource::new(
            "source",
            TestVideoOptions {
                width,
                height,
                framerate: frame_rate,
            },
        );
        let time_base = source.time_base();

        let encoder = D3d11NvencEncoder::new(
            "encoder",
            gpu.device(),
            gpu.context(),
            D3d11NvencEncoderOptions {
                codec: D3d11NvencCodec::H264,
                // D3d11Upload produces NV12 textures. Feeding this element a
                // D3d11VideoCompositor or DxgiCaptureSource GPU-mode output
                // instead means `Bgra` here and no Scaler at all — NVENC takes
                // BGRA textures directly.
                input_format: D3d11NvencInputFormat::Nv12,
                width,
                height,
                time_base,
                frame_rate,
                bit_rate: 4_000_000,
                gop_size: 60,
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let mut muxer = Mp4Muxer::create(&path)?;
        muxer.add_stream("video", encoder.parameters(), time_base)?;
        let muxer_sink = muxer.open()?.pop().expect("exactly one stream was added");

        let pipeline = Pipeline::new("nvenc-record", source, |source, ctx| {
            // TestVideoSource emits YUV420P on the CPU, so this one Scaler is
            // the only format conversion in the graph; D3d11Upload requires
            // NV12 and everything downstream of it is GPU-resident.
            let scaler = Scaler::new(
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

        println!("recording {seconds}s of {width}x{height} h264_nvenc to {path} ...");
        pipeline.run();

        thread::sleep(Duration::from_secs(seconds));
        pipeline.stop();

        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { name, error, .. } => eprintln!("[{name}] error: {error}"),
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                _ => {}
            }
        }

        println!("wrote {path}");
        Ok(())
    }
}
