//! Saves a resized H.264 video with its original sound, entirely in software:
//!
//!     FileDemuxer -- video -> SwDecoder -> Queue -> SwScaler -> SwEncoder -> FileMuxer
//!                 \- audio ---------------------------------------------> FileMuxer
//!
//! No GPU features; the same graph on Windows, Linux and macOS. The requested
//! width is in stored pixels; height keeps their aspect ratio, rounded to an
//! even size for YUV420P. The display rotation and original timestamps stay.
//! Audio is copied, not re-encoded. This small example accepts SDR video with
//! square pixels; it does not tone-map HDR or correct anamorphic video.
//!
//! Output is staged, published only on success, and replaced only with
//! `--overwrite`. Input/output aliases are always refused.
//!
//!     cargo run -p resize -- input.mp4 output.mp4 --width 160 [--overwrite]

fn main() -> Result<(), media_pp::Report> {
    Ok(example::run()?)
}

mod example {
    use std::ffi::OsString;

    use file_common::{FileArgs, SafeOutput, message, run_to_end};
    use media_pp::{
        Result,
        elements::{
            FileDemuxer, FileMuxer, Mp4Layout, StreamInfo, SwDecoder, SwEncoder, SwEncoderOptions,
            SwScaler, VideoCodec,
        },
        ffmpeg::{self, format::Pixel, software::scaling::Flags},
        pipeline::Pipeline,
    };

    const USAGE: &str =
        "usage: resize <input> <output.mp4> --width <positive-even-pixels> [--overwrite]";

    pub(super) fn run() -> Result<()> {
        let _log = media_pp::log::init(
            env!("CARGO_PKG_NAME"),
            "logs",
            media_pp::log::Level::Trace,
            7,
        )?;
        let (args, width) =
            parse(std::env::args_os().skip(1)).map_err(|e| message(format!("{e}\n{USAGE}")))?;
        let output = SafeOutput::new(&args)?;
        let (source, _) = FileDemuxer::open("demux", &args.input)?;
        let video = source.best(ffmpeg::media::Type::Video)?;
        let audio = source.best(ffmpeg::media::Type::Audio).ok();
        let (source_width, source_height) = video
            .size()
            .ok_or_else(|| message("the input does not say its picture size"))?;
        let height = height_for_width(source_width, source_height, width)?;
        if video.color().is_some_and(|color| {
            matches!(
                color.transfer,
                ffmpeg::color::TransferCharacteristic::SMPTE2084
                    | ffmpeg::color::TransferCharacteristic::ARIB_STD_B67
            )
        }) {
            return Err(message(
                "resize writes SDR H.264; tone-map HDR before resizing with this example",
            ));
        }
        // SAFETY: a plain field of this stream's live, owned parameters.
        let aspect = unsafe { (*video.parameters().as_ptr()).sample_aspect_ratio };
        if aspect.num > 0 && aspect.den > 0 && aspect.num != aspect.den {
            return Err(message(
                "resize expects square pixels; correct the input's sample aspect ratio first",
            ));
        }
        let orientation = video.orientation()?;
        let frame_rate = video.frame_rate().unwrap_or(ffmpeg::Rational(30, 1));
        let decoder = SwDecoder::new("decode", video.parameters().clone())?;
        let scaler = SwScaler::new("scale", Pixel::YUV420P, width, height, Flags::BILINEAR);
        let encoder = SwEncoder::for_stream(
            "encode",
            SwEncoderOptions {
                codec: VideoCodec::OpenH264,
                width,
                height,
                pixel_format: Pixel::YUV420P,
                frame_rate,
                bit_rate: 2_000_000,
                gop_size: (2 * i64::from(frame_rate.numerator())
                    / i64::from(frame_rate.denominator()).max(1))
                .clamp(1, u32::MAX as i64) as u32,
                max_b_frames: None,
            },
            &video,
        )?;

        let mut muxer = FileMuxer::create(output.path(), Mp4Layout::Standard)?;
        let picture = muxer.add_stream(
            "video",
            StreamInfo::from(&encoder).with_orientation(orientation),
        )?;
        let sound = audio
            .as_ref()
            .map(|audio| muxer.add_stream("audio", audio))
            .transpose()?;
        let mut sinks = muxer.open()?;
        let video_sink = sinks.take(picture)?;
        let audio_sink = sound.map(|track| sinks.take(track)).transpose()?;
        let (pipeline, ()) = Pipeline::new("resize", source, |source, ctx| {
            let branch = ctx
                .branch()
                .pipe(decoder)
                .queue("frames", 4)
                .pipe(scaler)
                .pipe(encoder)
                .to(video_sink)?;
            ctx.attach(source, video.index(), branch)?;
            if let (Some(audio), Some(sink)) = (audio, audio_sink) {
                ctx.attach(source, audio.index(), ctx.branch().to(sink)?)?;
            }
            Ok(())
        })?;
        println!(
            "resizing {} -> {} ({width}x{height}) ...",
            args.input.display(),
            args.output.display()
        );
        run_to_end(pipeline)?;
        output.commit()?;
        println!("wrote {}", args.output.display());
        Ok(())
    }

    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<(FileArgs, u32)> {
        let mut args = args.into_iter();
        let mut paths = Vec::new();
        let mut width = None;
        while let Some(arg) = args.next() {
            if arg == "--" {
                paths.push(arg);
                paths.extend(args);
                break;
            }
            if arg == "--width" {
                if width.is_some() {
                    return Err(message("--width was supplied twice"));
                }
                width = Some(
                    args.next()
                        .and_then(|n| n.to_str().and_then(|n| n.parse::<u32>().ok()))
                        .filter(|n| *n >= 2 && *n % 2 == 0)
                        .ok_or_else(|| {
                            message("--width needs a positive even integer, at least 2")
                        })?,
                );
            } else {
                paths.push(arg);
            }
        }
        let width = width.ok_or_else(|| message("--width is required"))?;
        Ok((FileArgs::parse(paths, None)?, width))
    }

    fn height_for_width(source_width: u32, source_height: u32, width: u32) -> Result<u32> {
        if source_width == 0 || source_height == 0 {
            return Err(message("the input has an empty picture size"));
        }
        let height = ((u128::from(width) * u128::from(source_height) + u128::from(source_width))
            / (2 * u128::from(source_width))
            * 2)
        .max(2);
        u32::try_from(height).map_err(|_| message("the resized height exceeds the supported range"))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn required_paths_and_width_are_checked() {
            for args in [
                vec![],
                vec!["in.mp4", "out.mp4"],
                vec!["in.mp4", "--width", "160"],
                vec!["in.mp4", "out.mp4", "--width"],
                vec!["in.mp4", "out.mp4", "--width", "0"],
                vec!["in.mp4", "out.mp4", "--width", "159"],
                vec!["in.mp4", "out.mp4", "--width", "-2"],
                vec!["in.mp4", "out.mp4", "--width", "160", "--width", "160"],
            ] {
                assert!(parse(args.into_iter().map(OsString::from)).is_err());
            }
            let (args, width) = parse(
                [
                    "in file.mp4",
                    "out file.mp4",
                    "--width",
                    "160",
                    "--overwrite",
                ]
                .into_iter()
                .map(OsString::from),
            )
            .unwrap();
            assert_eq!(width, 160);
            assert!(args.overwrite);
            assert_eq!(args.input, std::path::Path::new("in file.mp4"));
        }

        #[test]
        fn height_keeps_aspect_to_the_nearest_even_pixel() {
            assert_eq!(height_for_width(320, 240, 160).unwrap(), 120);
            assert_eq!(height_for_width(1920, 1080, 640).unwrap(), 360);
            assert_eq!(height_for_width(1920, 1080, 160).unwrap(), 90);
            assert_eq!(height_for_width(1080, 1920, 160).unwrap(), 284);
            assert!(height_for_width(1, u32::MAX, u32::MAX).is_err());
        }
    }
}
