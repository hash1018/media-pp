//! FileDemuxer -> VideoDecodeBin -> Queue -> VideoEncodeBin -> FileMuxer,
//! with the sound copied across as it is: a file's picture re-encoded to
//! H.264 into a new `.mp4`, decoded onto the GPU and encoded from it without
//! the pictures coming back to the CPU where the GPU takes both.
//!
//! The decode bin decodes onto a device — D3D11 on Windows, CUDA on Linux
//! where an NVIDIA GPU is there, system memory otherwise — and
//! `EncodeInput::for_decoded` turns where it put the pictures into what the
//! encode bin takes, so the two meet with nothing in between. The encode bin
//! opens NVENC where it can, Media Foundation next on Windows, and software
//! otherwise; both say which they chose, and this prints it. The encoder is
//! opened at the picture's own size and rate and told its colour, all read
//! off the input's `StreamInfo`, and the muxer takes its track from the
//! encode bin itself. The sound's packets go into the new file as they are.
//!
//! The file's source waits at its end rather than ending, so this stops the
//! pipeline on `Finished` — every packet has reached the file — or on an
//! error.
//!
//!     cargo run -p transcode -- input.mp4 [output.mp4]

fn main() -> impl std::process::Termination {
    example::run()
}

mod example {
    use media_pp::{
        bus::BusEvent,
        elements::{
            DecodeTarget, EncodeInput, FileDemuxer, FileMuxer, VideoDecodeBin, VideoEncodeBin,
            VideoEncodeOptions,
        },
        ffmpeg,
        pipeline::Pipeline,
    };

    /// Decoded pictures waiting between the decoder and the encoder.
    const QUEUED: usize = 4;

    pub(super) fn run() -> media_pp::Result<()> {
        let _log_guard = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;

        let Some(input_path) = std::env::args().nth(1) else {
            eprintln!("usage: transcode <input.mp4> [output.mp4]");
            std::process::exit(1);
        };
        let output_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "transcoded.mp4".into());

        let (source, _) = FileDemuxer::open("demux", &input_path)?;
        let video = source.best(ffmpeg::media::Type::Video)?;
        let audio = source.best(ffmpeg::media::Type::Audio).ok();
        let (width, height) = video.size().ok_or_else(|| {
            media_pp::Error::Other(format!("{input_path} does not say its picture size"))
        })?;
        let frame_rate = video
            .frame_rate
            .unwrap_or_else(|| ffmpeg::Rational::new(30, 1));

        let target = target()?;
        let decoder =
            VideoDecodeBin::open("decode", video.parameters.clone(), target.clone(), None)?;
        let input =
            EncodeInput::for_decoded(&target, decoder.output_format()).ok_or_else(|| {
                media_pp::Error::Other(format!(
                    "no encoder takes {:?} frames from this decoder",
                    decoder.output_format()
                ))
            })?;
        let encoder = VideoEncodeBin::open(
            "encode",
            input,
            VideoEncodeOptions {
                width,
                height,
                frame_rate,
                bit_rate: 4_000_000,
                // A keyframe every two seconds of pictures.
                gop_size: (2 * frame_rate.numerator() / frame_rate.denominator().max(1)).max(1)
                    as u32,
                max_b_frames: None,
                color: video.color(),
            },
        )?;
        println!(
            "decoding {:?}, encoding {:?}",
            decoder.path(),
            encoder.path()
        );

        let mut muxer = FileMuxer::create(&output_path)?;
        let video_track = muxer.add_stream("video", &encoder)?;
        let audio_track = audio
            .as_ref()
            .map(|audio| muxer.add_stream("audio", audio))
            .transpose()?;
        let mut sinks = muxer.open()?;
        let video_sink = sinks.take(video_track)?;
        let audio_sink = audio_track.map(|track| sinks.take(track)).transpose()?;

        let (pipeline, ()) = Pipeline::new("transcode", source, |source, ctx| {
            let picture = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", QUEUED)
                .pipe(encoder)
                .to(video_sink)?;
            ctx.attach(source, video.index, picture)?;
            if let (Some(audio), Some(sink)) = (audio, audio_sink) {
                ctx.attach(source, audio.index, ctx.branch().to(sink)?)?;
            }
            Ok(())
        })?;

        println!("transcoding {input_path} -> {output_path} ...");
        pipeline.run()?;
        for event in pipeline.bus().iter() {
            match &event {
                BusEvent::Error { .. } => eprintln!("{event}"),
                _ => println!("{event}"),
            }
            if matches!(event, BusEvent::Finished | BusEvent::Error { .. }) {
                pipeline.stop();
            }
        }
        println!("wrote {output_path}");
        Ok(())
    }

    /// What the decoder's pool has to cover after it: the queue, and a
    /// couple more for the encoder — nothing on D3D11, where it copies each
    /// picture in, but on CUDA the pictures it is still encoding, which it
    /// encodes in place. See `VideoEncodeBin`'s docs.
    const SURFACES: i32 = QUEUED as i32 + 2;

    /// Where the pictures are decoded to: D3D11 on Windows.
    #[cfg(target_os = "windows")]
    fn target() -> media_pp::Result<DecodeTarget> {
        Ok(DecodeTarget::D3d11 {
            gpu: media_pp::elements::D3d11Gpu::new()?,
            downstream_hw_frames: SURFACES,
        })
    }

    /// Where the pictures are decoded to: CUDA where there is an NVIDIA GPU,
    /// system memory otherwise.
    #[cfg(target_os = "linux")]
    fn target() -> media_pp::Result<DecodeTarget> {
        Ok(match media_pp::elements::CudaDevice::new() {
            Ok(device) => DecodeTarget::Cuda {
                device,
                downstream_hw_frames: SURFACES,
            },
            Err(error) => {
                println!("no CUDA device ({error}); decoding and encoding in software");
                DecodeTarget::System
            }
        })
    }

    /// Where the pictures are decoded to: system memory.
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    fn target() -> media_pp::Result<DecodeTarget> {
        let _ = SURFACES;
        Ok(DecodeTarget::System)
    }
}
