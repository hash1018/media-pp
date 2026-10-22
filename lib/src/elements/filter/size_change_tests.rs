//! A stream whose pictures change size partway — a camera reconfigured, a
//! WebRTC sender lowering its resolution — through every decoder and scaler
//! this machine has: a decoder goes on at the new size, and a scaler takes
//! any size in and goes on writing its own.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ffmpeg_next as ffmpeg;

use crate::buffer::MediaBuffer;
use crate::bus::BusEvent;
use crate::element::IntoFilter;
use crate::elements::{AppSink, AppSource};
use crate::pipeline::Pipeline;

const SMALL: (u32, u32) = (320, 240);
const LARGE: (u32, u32) = (640, 360);
/// What every scaler here is asked to write.
const OUT: (u32, u32) = (480, 270);

/// The parameters a decoder is opened with — the first size's — and the
/// stream: five pictures small, five large, five small again, each run
/// beginning at a keyframe carrying its own parameter sets.
fn changing() -> (ffmpeg::codec::Parameters, Vec<ffmpeg::Packet>) {
    changing_from("libopenh264").expect("OpenH264 is always present")
}

/// [`changing`], from the FFmpeg encoder called `encoder`, where it opens.
fn changing_from(encoder: &str) -> Option<(ffmpeg::codec::Parameters, Vec<ffmpeg::Packet>)> {
    let (parameters, small) = crate::test_support::try_configured(encoder, SMALL)?;
    let (_, large) = crate::test_support::try_configured(encoder, LARGE)?;
    let packets = small
        .iter()
        .chain(&large)
        .chain(&small)
        .enumerate()
        .map(|(n, packet)| {
            let mut packet = packet.clone();
            packet.set_pts(Some(n as i64));
            packet.set_dts(Some(n as i64));
            packet.set_duration(1);
            packet
        })
        .collect();
    Some((parameters, packets))
}

/// What `chain` handed on for `packets`.
#[derive(Debug)]
struct Ran {
    /// The size of each picture.
    sizes: Vec<(u32, u32)>,
    /// How many packets.
    packets: usize,
    /// Every error the pipeline reported.
    errors: Vec<String>,
}

/// The sizes of the pictures `chain` hands on for `packets`, and every
/// error the pipeline reported.
fn sizes_through(
    packets: Vec<ffmpeg::Packet>,
    chain: impl FnOnce(
        crate::pipeline::ChainBuilder,
    ) -> crate::error::Result<crate::pipeline::ChainBuilder>
    + Send
    + 'static,
) -> (Vec<(u32, u32)>, Vec<String>) {
    let ran = run(packets, chain);
    (ran.sizes, ran.errors)
}

/// Feeds `packets` through `chain` to its end.
fn run(
    packets: Vec<ffmpeg::Packet>,
    chain: impl FnOnce(
        crate::pipeline::ChainBuilder,
    ) -> crate::error::Result<crate::pipeline::ChainBuilder>
    + Send
    + 'static,
) -> Ran {
    let (source, handle) = AppSource::new("packets", 64);
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&sizes);
    let packets_out = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = Arc::clone(&packets_out);
    let (pipeline, ()) = Pipeline::new("size-change", source, move |source, ctx| {
        let branch = chain(ctx.branch())?.to(AppSink::new("pictures", move |buf| {
            match &buf {
                MediaBuffer::Video(frame) => {
                    seen.lock().unwrap().push((frame.width(), frame.height()));
                }
                MediaBuffer::Packet(_) => {
                    counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                MediaBuffer::Audio(_) => {}
            }
            Ok(())
        }))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("wire");
    pipeline.run().expect("run");
    for packet in packets {
        handle
            .push(MediaBuffer::Packet(Arc::new(packet).into()))
            .expect("push");
    }
    handle.finish().expect("finish");
    let mut errors = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        match pipeline.bus().recv_timeout(Duration::from_millis(100)) {
            Ok(event) => match event {
                BusEvent::Finished => break,
                BusEvent::Error { error, .. } => errors.push(error.to_string()),
                _ => {}
            },
            Err(_) if Instant::now() > deadline => {
                errors.push("never finished".into());
                break;
            }
            Err(_) => {}
        }
    }
    pipeline.stop();
    let sizes = sizes.lock().unwrap().clone();
    Ran {
        sizes,
        packets: packets_out.load(std::sync::atomic::Ordering::Relaxed),
        errors,
    }
}

/// What a decoder ought to hand on: every picture, at its own size.
fn assert_decoded(what: &str, (sizes, errors): (Vec<(u32, u32)>, Vec<String>)) {
    let expected: Vec<_> = [SMALL; 5]
        .into_iter()
        .chain([LARGE; 5])
        .chain([SMALL; 5])
        .collect();
    assert!(errors.is_empty(), "{what}: {errors:?}");
    assert_eq!(sizes, expected, "{what}");
}

/// What a scaler ought to hand on: every picture, at the size it writes.
fn assert_scaled(what: &str, (sizes, errors): (Vec<(u32, u32)>, Vec<String>)) {
    assert!(errors.is_empty(), "{what}: {errors:?}");
    assert_eq!(sizes, vec![OUT; 15], "{what}");
}

/// Whether `decoder` decodes the stream's first size here at all, saying
/// why where it does not: a CI runner's device opens, or even decodes,
/// no video. One that decodes the first size and then fails is a failure,
/// not this.
#[cfg(any(
    all(target_os = "windows", any(feature = "d3d11", feature = "d3d12")),
    feature = "cuda",
    feature = "vulkan"
))]
fn decodes_here<M>(
    what: &str,
    decoder: Option<impl IntoFilter<M> + Send + 'static>,
    packets: &[ffmpeg::Packet],
) -> bool {
    let Some(decoder) = decoder else {
        eprintln!("skipping {what}: it does not open here");
        return false;
    };
    let (sizes, errors) =
        sizes_through(packets[..5].to_vec(), move |branch| piped(branch, decoder));
    if errors.is_empty() && sizes == [SMALL; 5] {
        return true;
    }
    eprintln!("skipping {what}: it does not decode the stream here ({errors:?}, {sizes:?})");
    false
}

fn piped<M>(
    branch: crate::pipeline::ChainBuilder,
    element: impl IntoFilter<M>,
) -> crate::error::Result<crate::pipeline::ChainBuilder> {
    Ok(branch.pipe(element))
}

#[test]
fn software_decoding_and_scaling_follow_a_change_of_size() {
    use crate::elements::{SwDecoder, SwScaler};

    let (parameters, packets) = changing();
    let decoder = SwDecoder::new("decoder", parameters.clone()).unwrap();
    assert_decoded(
        "SwDecoder",
        sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
    );
    let decoder = SwDecoder::new("decoder", parameters).unwrap();
    let scaler = SwScaler::new(
        "scaler",
        ffmpeg::format::Pixel::YUV420P,
        OUT.0,
        OUT.1,
        ffmpeg::software::scaling::Flags::BILINEAR,
    );
    assert_scaled(
        "SwScaler",
        sizes_through(packets, move |branch| Ok(branch.pipe(decoder).pipe(scaler))),
    );
}

#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn d3d11_decoding_and_scaling_follow_a_change_of_size() {
    use crate::elements::{D3d11Decoder, D3d11Scaler, D3d11ScalerFormat};

    let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || D3d11Decoder::new("decoder", parameters.clone(), &gpu, 8).ok();
    if !decodes_here("D3d11Decoder", make(), &packets) {
        return;
    }
    let decoder = make().unwrap();
    assert_decoded(
        "D3d11Decoder",
        sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
    );
    let decoder = make().unwrap();
    let scaler = D3d11Scaler::new("scaler", &gpu, D3d11ScalerFormat::Nv12, OUT.0, OUT.1).unwrap();
    assert_scaled(
        "D3d11Scaler",
        sizes_through(packets, move |branch| Ok(branch.pipe(decoder).pipe(scaler))),
    );
}

#[cfg(all(target_os = "windows", feature = "d3d12"))]
#[test]
fn d3d12_decoding_and_scaling_follow_a_change_of_size() {
    use crate::elements::{D3d12Decoder, D3d12Scaler};

    let Some(gpu) = crate::test_support::try_d3d12_gpu() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || D3d12Decoder::new("decoder", parameters.clone(), &gpu).ok();
    if !decodes_here("D3d12Decoder", make(), &packets) {
        return;
    }
    let decoder = make().unwrap();
    assert_decoded(
        "D3d12Decoder",
        sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
    );
    let decoder = make().unwrap();
    let scaler = D3d12Scaler::new("scaler", &gpu, OUT.0, OUT.1).unwrap();
    assert_scaled(
        "D3d12Scaler",
        sizes_through(packets, move |branch| Ok(branch.pipe(decoder).pipe(scaler))),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_decoding_and_scaling_follow_a_change_of_size() {
    use crate::elements::{CudaDecoder, CudaScaler, CudaScalerInterp};

    let Some((device, _lock)) = crate::test_support::try_cuda_device() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || CudaDecoder::new("decoder", parameters.clone(), &device, 8).ok();
    if !decodes_here("CudaDecoder", make(), &packets) {
        return;
    }
    let decoder = make().unwrap();
    assert_decoded(
        "CudaDecoder",
        sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
    );
    let decoder = make().unwrap();
    let scaler = CudaScaler::new("scaler", &device, OUT.0, OUT.1, CudaScalerInterp::Bilinear);
    assert_scaled(
        "CudaScaler",
        sizes_through(packets, move |branch| Ok(branch.pipe(decoder).pipe(scaler))),
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_upload_and_scaling_follow_a_change_of_size() {
    use crate::elements::{SwDecoder, VulkanScaler, VulkanScalerInterp, VulkanUpload};

    let Some(device) = crate::test_support::try_vulkan_device() else {
        return;
    };
    let (parameters, packets) = changing();
    let decoder = SwDecoder::new("decoder", parameters.clone()).unwrap();
    let upload = VulkanUpload::new("upload", &device);
    assert_decoded(
        "VulkanUpload",
        sizes_through(packets.clone(), move |branch| {
            Ok(branch.pipe(decoder).pipe(upload))
        }),
    );
    let decoder = SwDecoder::new("decoder", parameters).unwrap();
    let upload = VulkanUpload::new("upload", &device);
    let scaler = VulkanScaler::new(
        "scaler",
        &device,
        OUT.0,
        OUT.1,
        VulkanScalerInterp::Bilinear,
    )
    .unwrap();
    assert_scaled(
        "VulkanScaler",
        sizes_through(packets, move |branch| {
            Ok(branch.pipe(decoder).pipe(upload).pipe(scaler))
        }),
    );
}

/// FFmpeg's Vulkan decoding crashes the process changing a context's size
/// in place; the decoder opens its context anew instead — see
/// `VulkanDecoder::reopen`.
#[cfg(feature = "vulkan")]
#[test]
fn vulkan_decoding_and_scaling_follow_a_change_of_size() {
    use crate::elements::{VulkanDecoder, VulkanScaler, VulkanScalerInterp};

    let Some(device) = crate::test_support::try_vulkan_device() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || VulkanDecoder::new("decoder", parameters.clone(), &device, 8).ok();
    if !decodes_here("VulkanDecoder", make(), &packets) {
        return;
    }
    let decoder = make().unwrap();
    assert_decoded(
        "VulkanDecoder",
        sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
    );
    let decoder = make().unwrap();
    let scaler = VulkanScaler::new(
        "scaler",
        &device,
        OUT.0,
        OUT.1,
        VulkanScalerInterp::Bilinear,
    )
    .unwrap();
    assert_scaled(
        "VulkanScaler after VulkanDecoder",
        sizes_through(packets, move |branch| Ok(branch.pipe(decoder).pipe(scaler))),
    );
}

/// What an encoder opened at the first size ought to do with the changing
/// stream: refuse each picture of the other size, as a typed error rather
/// than anything worse, and go on encoding once the first size is back —
/// both of its runs, every picture.
fn assert_refused_and_on(what: &str, ran: Ran) {
    assert_eq!(
        ran.packets, 10,
        "{what}: both runs of the first size encoded: {ran:?}"
    );
    assert_eq!(
        ran.errors.len(),
        5,
        "{what}: each picture of the other size refused: {ran:?}"
    );
    assert!(
        ran.errors
            .iter()
            .all(|error| error.contains("640") && error.contains("360")),
        "{what}: refused for its size: {:?}",
        ran.errors
    );
}

#[test]
fn a_software_encoder_refuses_another_size_and_goes_on() {
    use crate::elements::{SwDecoder, SwEncoder, SwEncoderOptions, VideoCodec};

    let (parameters, packets) = changing();
    let decoder = SwDecoder::new("decoder", parameters).unwrap();
    let encoder = SwEncoder::new(
        "encoder",
        SwEncoderOptions {
            codec: VideoCodec::OpenH264,
            width: SMALL.0,
            height: SMALL.1,
            pixel_format: ffmpeg::format::Pixel::YUV420P,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 300_000,
            gop_size: 30,
            max_b_frames: None,
        },
    )
    .unwrap();
    assert_refused_and_on(
        "SwEncoder",
        run(packets, move |branch| {
            Ok(branch.pipe(decoder).pipe(encoder))
        }),
    );
}

#[cfg(all(target_os = "windows", feature = "d3d11"))]
#[test]
fn a_d3d11_encoder_refuses_another_size_and_goes_on() {
    use crate::elements::{
        D3d11Decoder, D3d11VideoCodec, D3d11VideoEncoder, D3d11VideoEncoderOptions,
        D3d11VideoInputFormat,
    };

    let Some(gpu) = crate::test_support::try_d3d11_gpu() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || D3d11Decoder::new("decoder", parameters.clone(), &gpu, 8).ok();
    if !decodes_here("D3d11Decoder", make(), &packets) {
        return;
    }
    let encoder = [
        D3d11VideoCodec::H264Nvenc,
        D3d11VideoCodec::H264MediaFoundation,
    ]
    .into_iter()
    .find_map(|codec| {
        D3d11VideoEncoder::new(
            "encoder",
            &gpu,
            D3d11VideoEncoderOptions {
                codec,
                input_format: D3d11VideoInputFormat::Nv12,
                width: SMALL.0,
                height: SMALL.1,
                frame_rate: ffmpeg::Rational::new(30, 1),
                bit_rate: 1_000_000,
                gop_size: 30,
                max_b_frames: None,
            },
        )
        .ok()
    });
    let Some(encoder) = encoder else {
        eprintln!("skipping: no D3D11 H.264 encoder opens here");
        return;
    };
    let decoder = make().unwrap();
    assert_refused_and_on(
        "D3d11VideoEncoder",
        run(packets, move |branch| {
            Ok(branch.pipe(decoder).pipe(encoder))
        }),
    );
}

#[cfg(all(target_os = "windows", feature = "d3d12"))]
#[test]
fn a_d3d12_encoder_refuses_another_size_and_goes_on() {
    use crate::elements::{
        D3d12Decoder, D3d12VideoCodec, D3d12VideoEncoder, D3d12VideoEncoderOptions,
    };

    let Some(gpu) = crate::test_support::try_d3d12_gpu() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || D3d12Decoder::new("decoder", parameters.clone(), &gpu).ok();
    if !decodes_here("D3d12Decoder", make(), &packets) {
        return;
    }
    let Ok(encoder) = D3d12VideoEncoder::new(
        "encoder",
        &gpu,
        D3d12VideoEncoderOptions {
            codec: D3d12VideoCodec::H264,
            width: SMALL.0,
            height: SMALL.1,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 1_000_000,
            gop_size: 30,
            max_b_frames: None,
        },
    ) else {
        eprintln!("skipping: no D3D12 H.264 encoder opens here");
        return;
    };
    let decoder = make().unwrap();
    assert_refused_and_on(
        "D3d12VideoEncoder",
        run(packets, move |branch| {
            Ok(branch.pipe(decoder).pipe(encoder))
        }),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn a_cuda_encoder_refuses_another_size_and_goes_on() {
    use crate::elements::{
        CudaCodec, CudaDecoder, CudaEncoder, CudaEncoderOptions, CudaFrameFormat,
    };

    let Some((device, _lock)) = crate::test_support::try_cuda_device() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || CudaDecoder::new("decoder", parameters.clone(), &device, 8).ok();
    if !decodes_here("CudaDecoder", make(), &packets) {
        return;
    }
    let Ok(encoder) = CudaEncoder::new(
        "encoder",
        &device,
        CudaEncoderOptions {
            codec: CudaCodec::H264,
            input_format: CudaFrameFormat::Nv12,
            width: SMALL.0,
            height: SMALL.1,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 1_000_000,
            gop_size: 30,
            max_b_frames: None,
        },
    ) else {
        eprintln!("skipping: NVENC does not open here");
        return;
    };
    let decoder = make().unwrap();
    assert_refused_and_on(
        "CudaEncoder",
        run(packets, move |branch| {
            Ok(branch.pipe(decoder).pipe(encoder))
        }),
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn a_vulkan_encoder_refuses_another_size_and_goes_on() {
    use crate::elements::{VulkanCodec, VulkanDecoder, VulkanEncoder, VulkanEncoderOptions};

    let Some(device) = crate::test_support::try_vulkan_device() else {
        return;
    };
    let (parameters, packets) = changing();
    let make = || VulkanDecoder::new("decoder", parameters.clone(), &device, 8).ok();
    if !decodes_here("VulkanDecoder", make(), &packets) {
        return;
    }
    let Ok(encoder) = VulkanEncoder::new(
        "encoder",
        &device,
        VulkanEncoderOptions {
            codec: VulkanCodec::H264,
            width: SMALL.0,
            height: SMALL.1,
            frame_rate: ffmpeg::Rational::new(30, 1),
            bit_rate: 1_000_000,
            gop_size: 30,
            max_b_frames: None,
        },
    ) else {
        eprintln!("skipping: no Vulkan H.264 encoder opens here");
        return;
    };
    let decoder = make().unwrap();
    assert_refused_and_on(
        "VulkanEncoder",
        run(packets, move |branch| {
            Ok(branch.pipe(decoder).pipe(encoder))
        }),
    );
}

/// The same through every decoder for H.265, from whichever H.265 encoder
/// this machine has: `VulkanDecoder` watches H.265's parameter sets too.
#[test]
fn hevc_decoding_follows_a_change_of_size() {
    use crate::elements::SwDecoder;

    let Some((parameters, packets)) = ["hevc_nvenc", "hevc_mf", "hevc_qsv", "hevc_amf", "libx265"]
        .into_iter()
        .find_map(changing_from)
    else {
        eprintln!("skipping: no H.265 encoder opens here");
        return;
    };
    let decoder = SwDecoder::new("decoder", parameters.clone()).unwrap();
    assert_decoded(
        "SwDecoder, H.265",
        sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
    );

    #[cfg(all(target_os = "windows", feature = "d3d11"))]
    if let Some(gpu) = crate::test_support::try_d3d11_gpu() {
        use crate::elements::D3d11Decoder;
        let make = || D3d11Decoder::new("decoder", parameters.clone(), &gpu, 8).ok();
        if decodes_here("D3d11Decoder, H.265", make(), &packets) {
            let decoder = make().unwrap();
            assert_decoded(
                "D3d11Decoder, H.265",
                sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
            );
        }
    }
    #[cfg(all(target_os = "windows", feature = "d3d12"))]
    if let Some(gpu) = crate::test_support::try_d3d12_gpu() {
        use crate::elements::D3d12Decoder;
        let make = || D3d12Decoder::new("decoder", parameters.clone(), &gpu).ok();
        if decodes_here("D3d12Decoder, H.265", make(), &packets) {
            let decoder = make().unwrap();
            assert_decoded(
                "D3d12Decoder, H.265",
                sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
            );
        }
    }
    #[cfg(feature = "cuda")]
    if let Some((device, _lock)) = crate::test_support::try_cuda_device() {
        use crate::elements::CudaDecoder;
        let make = || CudaDecoder::new("decoder", parameters.clone(), &device, 8).ok();
        if decodes_here("CudaDecoder, H.265", make(), &packets) {
            let decoder = make().unwrap();
            assert_decoded(
                "CudaDecoder, H.265",
                sizes_through(packets.clone(), move |branch| piped(branch, decoder)),
            );
        }
    }
    #[cfg(feature = "vulkan")]
    if let Some(device) = crate::test_support::try_vulkan_device() {
        use crate::elements::VulkanDecoder;
        let make = || VulkanDecoder::new("decoder", parameters.clone(), &device, 8).ok();
        if decodes_here("VulkanDecoder, H.265", make(), &packets) {
            let decoder = make().unwrap();
            assert_decoded(
                "VulkanDecoder, H.265",
                sizes_through(packets, move |branch| piped(branch, decoder)),
            );
        }
    }
}
