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
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use ffmpeg_next as ffmpeg;
    use media_pp::{
        bus::BusEvent,
        elements::{
            CaptureMode, D3d11NvencCodec, D3d11NvencEncoder, D3d11NvencEncoderOptions,
            D3d11NvencInputFormat, DxgiCaptureOptions, DxgiCaptureSource, Mp4Muxer,
        },
        pipeline::Pipeline,
    };
    use render_common::D3d11GpuContext;

    /// DxgiCaptureSource (GPU mode) -> D3d11NvencEncoder -> Mp4Muxer: records
    /// the desktop into a playable `.mp4` without the pixels ever touching the
    /// CPU or going through a separate CPU color-conversion stage.
    ///
    /// The contrast with `screen_record` is the whole point. That example runs
    /// `DxgiCaptureSource (CPU mode) -> Scaler -> SwEncoder`: every frame is
    /// mapped back to system memory, converted BGRA->YUV420P by libswscale, and
    /// encoded on the CPU. Here capture writes a GPU-resident BGRA texture,
    /// NVENC consumes that texture directly — `D3d11NvencInputFormat::Bgra`,
    /// since NVENC does its own color conversion inside the encode block — and
    /// only the compressed packets ever reach the CPU. There is no `Scaler` and
    /// no `D3d11Download` in this graph at all.
    ///
    /// Needs an NVIDIA GPU and an ffmpeg build with NVENC. `Pipeline::finish`
    /// sends ordered EOS through the encoder and muxer so delayed frames are
    /// drained before the MP4 trailer is finalized.
    ///
    ///     cargo run -p screen_record_nvenc -- <output.mp4> [seconds]
    pub fn run() -> media_pp::Result<()> {
        media_pp::init()?;
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let mut args = std::env::args();
        let program = args.next().unwrap_or_else(|| env!("CARGO_PKG_NAME").into());
        let Some(path) = args.next() else {
            eprintln!("usage: {program} <output.mp4> [seconds]");
            return Err(media_pp::Error::Other("missing output path".into()));
        };
        let seconds = args
            .next()
            .map(|value| {
                value
                    .parse::<u64>()
                    .map_err(|_| media_pp::Error::Other(format!("invalid seconds: {value}")))
            })
            .transpose()?
            .unwrap_or(5);
        if seconds == 0 {
            return Err(media_pp::Error::Other(
                "recording duration must be greater than zero".into(),
            ));
        }
        if args.next().is_some() {
            eprintln!("usage: {program} <output.mp4> [seconds]");
            return Err(media_pp::Error::Other("too many arguments".into()));
        }

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
        let frame_rate = ffmpeg::Rational::new(30, 1);

        let encoder = D3d11NvencEncoder::new(
            "encoder",
            gpu.device(),
            gpu.context(),
            D3d11NvencEncoderOptions {
                codec: D3d11NvencCodec::H264,
                // What removes the conversion step: DXGI desktop duplication
                // produces BGRA and NVENC accepts BGRA textures as-is.
                input_format: D3d11NvencInputFormat::Bgra,
                width: format.width,
                height: format.height,
                time_base: format.time_base,
                frame_rate,
                bit_rate: 8_000_000,
                gop_size: 60, // ~2s @ 30fps
            },
        )
        .map_err(|e| media_pp::Error::Other(e.to_string()))?;

        let mut muxer = Mp4Muxer::create(&path)?;
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
            "recording {seconds}s of the desktop at {}x{} (h264_nvenc) to {path} ...",
            format.width, format.height
        );
        pipeline.run();

        let deadline = Instant::now() + Duration::from_secs(seconds);
        let mut pipeline_error = None;
        while Instant::now() < deadline && pipeline_error.is_none() {
            while let Some(event) = pipeline.bus().try_recv() {
                match event {
                    BusEvent::Error { name, error, .. } => {
                        eprintln!("[{name}] error: {error}");
                        pipeline_error = Some(error);
                        break;
                    }
                    BusEvent::Dropped { name, .. } => {
                        eprintln!("[{name}] dropped a buffer (queue full)")
                    }
                    BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                    // `BusEvent` is `#[non_exhaustive]`; this example only acts
                    // on the events above.
                    _ => {}
                }
            }
            thread::sleep(
                deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(20)),
            );
        }

        if pipeline_error.is_some() {
            pipeline.stop();
        } else {
            pipeline.finish();
        }

        for event in pipeline.bus().iter() {
            match event {
                BusEvent::Error { name, error, .. } => {
                    eprintln!("[{name}] error: {error}");
                    if pipeline_error.is_none() {
                        pipeline_error = Some(error);
                    }
                }
                BusEvent::Dropped { name, .. } => {
                    eprintln!("[{name}] dropped a buffer (queue full)")
                }
                BusEvent::Eos { name, .. } => println!("[{name}] eos"),
                // `BusEvent` is `#[non_exhaustive]`; this example only acts
                // on the events above.
                _ => {}
            }
        }

        if let Some(error) = pipeline_error {
            return Err(error);
        }

        println!("wrote {path}");
        Ok(())
    }
}
