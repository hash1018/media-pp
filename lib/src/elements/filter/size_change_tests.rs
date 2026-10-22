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
    let (parameters, small) = crate::test_support::h264_configured(SMALL);
    let (_, large) = crate::test_support::h264_configured(LARGE);
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
    (parameters, packets)
}

/// The sizes of the pictures `chain` hands on for the changing stream, and
/// every error the pipeline reported.
fn sizes_through(
    packets: Vec<ffmpeg::Packet>,
    chain: impl FnOnce(
        crate::pipeline::ChainBuilder,
    ) -> crate::error::Result<crate::pipeline::ChainBuilder>
    + Send
    + 'static,
) -> (Vec<(u32, u32)>, Vec<String>) {
    let (source, handle) = AppSource::new("packets", 64);
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&sizes);
    let (pipeline, ()) = Pipeline::new("size-change", source, move |source, ctx| {
        let branch = chain(ctx.branch())?.to(AppSink::new("pictures", move |buf| {
            if let MediaBuffer::Video(frame) = &buf {
                seen.lock().unwrap().push((frame.width(), frame.height()));
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
    (sizes, errors)
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
    eprintln!("skipping {what}: it does not decode H.264 here ({errors:?}, {sizes:?})");
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
