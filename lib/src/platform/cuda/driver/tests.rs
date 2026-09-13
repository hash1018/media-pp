use std::sync::{Arc, Mutex};

use ffmpeg_next::{self as ffmpeg};

use super::*;
use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    elements::{CudaDownload, CudaFrameFormat, CudaUpload},
    pool::UnboundObjectPool,
    pp_log::PpLog,
    test_support::try_cuda_device,
};

struct CapturingSink {
    pp_log: PpLog,
    received: Arc<Mutex<Vec<MediaBuffer>>>,
}

impl Element for CapturingSink {
    fn name(&self) -> Arc<str> {
        "capture".into()
    }
    fn element_type(&self) -> ElementType {
        ElementType::Other
    }
    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }
    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Sink for CapturingSink {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        self.received.lock().unwrap().push(buf);
        Ok(())
    }
    fn control(&mut self, _msg: ControlMsg) -> crate::error::Result<()> {
        Ok(())
    }
}

fn capture(element: &mut dyn Source) -> Arc<Mutex<Vec<MediaBuffer>>> {
    let received = Arc::new(Mutex::new(Vec::new()));
    element.src_pads()[0].link(Box::new(CapturingSink {
        received: received.clone(),
        pp_log: element_pp_log(ElementType::Other, "capture", None),
    }));
    received
}

/// Uploads one NV12 frame whose luma is `luma` everywhere and hands back
/// the CUDA-resident result, so a test has a real surface to operate on.
fn cuda_surface(
    device: &crate::elements::CudaDevice,
    width: u32,
    height: u32,
    luma: u8,
) -> Option<MediaBuffer> {
    let Ok(mut upload) = CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
    else {
        eprintln!("skipping: this machine has no usable CUDA frames context");
        return None;
    };
    let uploaded = capture(&mut upload);
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
    let y_stride = frame.stride(0);
    frame.data_mut(0)[..y_stride * height as usize].fill(luma);
    let uv_stride = frame.stride(1);
    frame.data_mut(1)[..uv_stride * (height / 2) as usize].fill(128);
    let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
    let mut slot = pool.get();
    *slot = frame;
    upload
        .consume(MediaBuffer::Video(Arc::new(slot)))
        .expect("upload");
    Some(uploaded.lock().unwrap().remove(0))
}

fn download(
    device: &crate::elements::CudaDevice,
    frame: MediaBuffer,
    width: u32,
    height: u32,
) -> Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
    let mut download = CudaDownload::new("download", device, CudaFrameFormat::Nv12, width, height);
    let received = capture(&mut download);
    download.consume(frame).expect("download");
    let buf = received.lock().unwrap().remove(0);
    match buf {
        MediaBuffer::Video(frame) => frame,
        other => panic!("expected a Video buffer, got {}", other.kind()),
    }
}

/// Reads a CUDA-resident BGRA frame back to the CPU.
fn download_bgra(
    device: &crate::elements::CudaDevice,
    frame: MediaBuffer,
    width: u32,
    height: u32,
) -> Arc<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>> {
    let mut download = CudaDownload::new("download", device, CudaFrameFormat::Bgra, width, height);
    let received = capture(&mut download);
    download.consume(frame).expect("download");
    let buf = received.lock().unwrap().remove(0);
    match buf {
        MediaBuffer::Video(frame) => frame,
        other => panic!("expected a Video buffer, got {}", other.kind()),
    }
}

/// The exact inverse of [`bt709_limited`], as the kernel computes it, so
/// a test can say what a colour should come back as instead of allowing
/// a tolerance and hoping.
fn bt709_limited_inverse(luma: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = (f32::from(luma) - 16.0) * (255.0 / 219.0);
    let u = (f32::from(u) - 128.0) * (255.0 / 224.0);
    let v = (f32::from(v) - 128.0) * (255.0 / 224.0);
    let b = y + 1.8556 * u;
    let r = y + 1.5748 * v;
    let g = (y - 0.2126 * r - 0.0722 * b) / 0.7152;
    let byte = |value: f32| (value.clamp(0.0, 255.0) + 0.5) as u8;
    (byte(r), byte(g), byte(b))
}

/// The conversion `CudaScaler` refuses, at the driver layer.
///
/// This is also what proves the PTX assembles: it is hand written, and
/// the driver JITs it when the module loads.
#[test]
fn nv12_to_bgra_undoes_the_conversion_that_made_it() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let driver = match CudaDriver::retain_primary() {
        Ok(driver) => driver,
        Err(error) => {
            eprintln!("skipping: no usable CUDA driver context ({error})");
            return;
        }
    };

    // Four flat colours, one per 2x2 block so chroma subsampling has
    // nothing to average across.
    let colours = [[0u8, 255, 0], [0, 0, 255], [255, 255, 255], [16, 32, 48]];
    let (width, height) = (8u32, 2u32);
    let block_of = |x: u32| colours[(x / 2) as usize];

    let Some(source) = cuda_bgra_surface(&device, width, height, |x, _| {
        let [b, g, r] = block_of(x);
        [b, g, r, 255]
    }) else {
        return;
    };
    let Some(nv12) = cuda_surface(&device, width, height, 0) else {
        return;
    };
    let Some(back) = cuda_bgra_surface(&device, width, height, |_, _| [7, 7, 7, 7]) else {
        return;
    };

    let (MediaBuffer::Video(source_frame), MediaBuffer::Video(nv12_frame)) = (&source, &nv12)
    else {
        panic!("expected Video buffers");
    };
    let MediaBuffer::Video(back_frame) = &back else {
        panic!("expected a Video buffer");
    };

    // There and back: the round trip is what makes the two kernels each
    // other's check rather than two separate guesses at BT.709.
    driver
        .bgra_to_nv12(
            BgraSurface::from_frame(source_frame).expect("a BGRA source"),
            Nv12Surface::from_frame(nv12_frame).expect("an NV12 destination"),
            width,
            height,
        )
        .expect("bgra_to_nv12");
    driver
        .nv12_to_bgra(
            Nv12Surface::from_frame(nv12_frame).expect("an NV12 source"),
            BgraSurface::from_frame(back_frame).expect("a BGRA destination"),
            width,
            height,
        )
        .expect("the conversion kernel must launch");
    driver.synchronize().expect("synchronize");

    let out = download_bgra(&device, back.clone(), width, height);
    let row = out.data(0);
    for x in 0..width {
        let [b, g, r] = block_of(x);
        let (expected_y, expected_u, expected_v) =
            bt709_limited(f32::from(r), f32::from(g), f32::from(b));
        let (want_r, want_g, want_b) = bt709_limited_inverse(expected_y, expected_u, expected_v);
        let at = x as usize * 4;
        let got = [row[at], row[at + 1], row[at + 2], row[at + 3]];
        assert_eq!(
            got,
            [want_b, want_g, want_r, 255],
            "pixel {x} came back wrong; it started as {:?}",
            [b, g, r]
        );
    }
}
/// The keying kernel, at the driver layer: the key colour goes fully
/// transparent, a clearly different colour stays fully opaque and keeps
/// its RGB, and something inside the feather band lands in between.
///
/// This is also what proves the PTX assembles at all — it is hand
/// written, and the driver JITs it when the module loads.
#[test]
fn key_bgra_writes_alpha_from_distance_to_the_key() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let driver = match CudaDriver::retain_primary() {
        Ok(driver) => driver,
        Err(error) => {
            eprintln!("skipping: no usable CUDA driver context ({error})");
            return;
        }
    };
    let (width, height) = (4u32, 1u32);
    // Pure green, pure red, an off-green landing in the middle of the
    // feather band, and white.
    //
    // The third is picked rather than guessed. Distance is Euclidean in
    // BGR over 0..1, divided by sqrt(3) so the cube's diagonal is 1, and
    // a threshold of 0.15 with 0.1 of smoothing opens a band from 0.10 to
    // 0.20. Green 189 gives (189/255 - 1) = -0.2588, so 0.2588 / sqrt(3)
    // = 0.1494 — just under halfway across, which is an alpha of 126.
    // Green 230 would have been 0.057, wholly below the band and keyed
    // out like the pure colour.
    let source_pixels = [
        [0u8, 255, 0, 255],
        [0, 0, 255, 255],
        [0, 189, 0, 255],
        [255, 255, 255, 255],
    ];
    let Some(source) = cuda_bgra_surface(&device, width, height, |x, _| source_pixels[x as usize])
    else {
        return;
    };
    let Some(destination) = cuda_bgra_surface(&device, width, height, |_, _| [7, 7, 7, 7]) else {
        return;
    };

    let (MediaBuffer::Video(source_frame), MediaBuffer::Video(destination_frame)) =
        (&source, &destination)
    else {
        panic!("expected Video buffers");
    };
    let (band_low, inv_band_width) = (0.15 - 0.05, 1.0 / 0.1);
    driver
        .key_bgra(
            BgraSurface::from_frame(source_frame).expect("a BGRA source"),
            BgraSurface::from_frame(destination_frame).expect("a BGRA destination"),
            width,
            height,
            Color::new(0, 255, 0),
            band_low,
            inv_band_width,
        )
        .expect("the keying kernel must launch");
    driver.synchronize().expect("synchronize");

    let out = download_bgra(&device, destination.clone(), width, height);
    let pixel = |x: usize| {
        let row = out.data(0);
        let at = x * 4;
        [row[at], row[at + 1], row[at + 2], row[at + 3]]
    };

    assert_eq!(pixel(0)[3], 0, "the key colour must key out completely");
    assert_eq!(
        pixel(1),
        [0, 0, 255, 255],
        "a clearly different colour keeps its alpha and its RGB"
    );
    let partial = pixel(2)[3];
    assert!(
        partial.abs_diff(126) <= 1,
        "a colour inside the feather band lands on the ramp, expected about 126, got {partial}"
    );
    assert_eq!(
        [pixel(2)[0], pixel(2)[1], pixel(2)[2]],
        [0, 189, 0],
        "only alpha is written; the colour passes through"
    );
    assert_eq!(pixel(3)[3], 255, "white is nowhere near green");
}

/// The driver layer's whole contract in one pass: a fill covers the
/// surface, and a blit moves exactly the requested rectangle to exactly
/// the requested place, leaving everything else as the fill left it.
#[test]
fn fill_then_blit_writes_the_expected_rectangles() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let driver = match CudaDriver::retain_primary() {
        Ok(driver) => driver,
        Err(error) => {
            eprintln!("skipping: no usable CUDA driver context ({error})");
            return;
        }
    };
    let (width, height) = (64u32, 64u32);
    let Some(canvas) = cuda_surface(&device, width, height, 0) else {
        return;
    };
    let Some(layer) = cuda_surface(&device, 32, 32, 200) else {
        return;
    };

    let (MediaBuffer::Video(canvas_frame), MediaBuffer::Video(layer_frame)) = (&canvas, &layer)
    else {
        panic!("expected Video buffers");
    };
    let canvas_surface = Nv12Surface::from_frame(canvas_frame).expect("canvas planes");
    let layer_surface = Nv12Surface::from_frame(layer_frame).expect("layer planes");

    driver
        .fill_nv12(canvas_surface, width, height, Color::WHITE)
        .expect("fill");
    driver
        .blit_nv12(
            layer_surface,
            canvas_surface,
            Nv12Region {
                source_x: 0,
                source_y: 0,
                destination_x: 16,
                destination_y: 8,
                width: 32,
                height: 32,
            },
        )
        .expect("blit");

    let out = download(&device, canvas.clone(), width, height);
    let stride = out.stride(0);
    let at = |x: usize, y: usize| out.data(0)[y * stride + x];
    assert_eq!(at(0, 0), 235, "the fill did not cover the top-left corner");
    assert_eq!(at(63, 63), 235, "the fill did not cover the bottom-right");
    assert_eq!(at(16, 8), 200, "the blit missed its top-left corner");
    assert_eq!(at(47, 39), 200, "the blit missed its bottom-right corner");
    assert_eq!(at(15, 8), 235, "the blit wrote left of its rectangle");
    assert_eq!(at(48, 8), 235, "the blit wrote right of its rectangle");
    assert_eq!(at(16, 7), 235, "the blit wrote above its rectangle");
    assert_eq!(at(16, 40), 235, "the blit wrote below its rectangle");

    let uv_stride = out.stride(1);
    assert_eq!(
        out.data(1)[uv_stride * 20 + 4],
        128,
        "chroma did not survive the fill"
    );
}

/// A CUDA surface whose luma is `f(x, y)`, so an indexing or pitch
/// mistake in the kernel shows up as a wrong *position*, not just a
/// wrong value.
/// Uploads one BGRA frame built by `pixel`, so a conversion test starts
/// from a CUDA-resident source with content it can predict.
fn cuda_bgra_surface(
    device: &crate::elements::CudaDevice,
    width: u32,
    height: u32,
    pixel: impl Fn(u32, u32) -> [u8; 4],
) -> Option<MediaBuffer> {
    let Ok(mut upload) = CudaUpload::new("upload", device, CudaFrameFormat::Bgra, width, height)
    else {
        eprintln!("skipping: this machine has no usable CUDA frames context");
        return None;
    };
    let uploaded = capture(&mut upload);
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
    let stride = frame.stride(0);
    for y in 0..height {
        let row = &mut frame.data_mut(0)[y as usize * stride..];
        for x in 0..width {
            row[x as usize * 4..x as usize * 4 + 4].copy_from_slice(&pixel(x, y));
        }
    }
    let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
    let mut slot = pool.get();
    *slot = frame;
    upload
        .consume(MediaBuffer::Video(Arc::new(slot)))
        .expect("upload");
    Some(uploaded.lock().unwrap().remove(0))
}

/// The blend that exists so an overlay can be transparent: where the
/// layer's own alpha is zero the canvas has to come through untouched,
/// and where it is full the layer has to replace it.
///
/// This is the property a blit cannot have. Converting a BGRA overlay to
/// NV12 and blitting it puts opaque black everywhere nothing was drawn,
/// which is what made a Drawing hide the capture beneath it.
#[test]
fn a_bgra_layer_blends_under_its_own_alpha_rather_than_covering() {
    let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
        return;
    };
    let Ok(driver) = CudaDriver::retain_primary() else {
        eprintln!("skipping: no usable CUDA driver on this machine");
        return;
    };
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;
    const CANVAS_LUMA: u8 = 90;

    // Opaque red on the left half, entirely transparent on the right —
    // and the transparent half is still *red*, so a kernel ignoring alpha
    // would fail rather than accidentally agree.
    let pixel = |x: u32, _y: u32| {
        if x < WIDTH / 2 {
            [0, 0, 255, 255]
        } else {
            [0, 0, 255, 0]
        }
    };
    let Some(layer) = cuda_bgra_surface(&device, WIDTH, HEIGHT, pixel) else {
        return;
    };
    let Some(canvas) = cuda_surface(&device, WIDTH, HEIGHT, CANVAS_LUMA) else {
        return;
    };
    let (MediaBuffer::Video(layer_frame), MediaBuffer::Video(canvas_frame)) = (&layer, &canvas)
    else {
        panic!("both uploads produce Video buffers");
    };
    let scratch = driver
        .overlay_scratch(WIDTH, HEIGHT)
        .expect("scratch for a layer this size");

    driver
        .blend_bgra_nv12(
            BgraSurface::from_frame(layer_frame).expect("a BGRA surface"),
            &scratch,
            Nv12Surface::from_frame(canvas_frame).expect("an NV12 surface"),
            Nv12Region {
                source_x: 0,
                source_y: 0,
                destination_x: 0,
                destination_y: 0,
                width: WIDTH,
                height: HEIGHT,
            },
            255,
        )
        .expect("blend");
    driver.synchronize().expect("synchronize");

    let blended = download(&device, canvas, WIDTH, HEIGHT);
    let stride = blended.stride(0);
    let luma = blended.data(0);
    let (expected_red, _, _) = bt709_limited(255.0, 0.0, 0.0);
    for y in 0..HEIGHT as usize {
        for x in 0..(WIDTH / 2) as usize {
            assert_eq!(
                luma[y * stride + x],
                expected_red,
                "the opaque half should be the layer's own colour at ({x}, {y})"
            );
        }
        for x in (WIDTH / 2) as usize..WIDTH as usize {
            assert_eq!(
                luma[y * stride + x],
                CANVAS_LUMA,
                "the transparent half should leave the canvas alone at ({x}, {y})"
            );
        }
    }
}

/// The conversion nothing else on the CUDA path can do, checked against
/// the definition the compositor fills backgrounds with rather than
/// against a tolerance: every luma byte is that pixel's own conversion,
/// and every chroma pair is the conversion of its 2x2 block's average.
#[test]
fn bgra_converts_to_nv12_exactly_as_the_shared_definition_says() {
    let Some((device, _cuda_lock)) = crate::test_support::try_cuda_device() else {
        return;
    };
    let Ok(driver) = CudaDriver::retain_primary() else {
        eprintln!("skipping: no usable CUDA driver on this machine");
        return;
    };
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;
    // A pattern with no symmetry between channels, so a kernel that mixed
    // two of them up could not still match.
    let pixel = |x: u32, y: u32| {
        [
            (x * 4 % 256) as u8,
            (y * 8 % 256) as u8,
            ((x + y) * 3 % 256) as u8,
            255,
        ]
    };
    let Some(source) = cuda_bgra_surface(&device, WIDTH, HEIGHT, pixel) else {
        return;
    };
    let Some(destination) = cuda_surface(&device, WIDTH, HEIGHT, 0) else {
        return;
    };

    let (MediaBuffer::Video(source_frame), MediaBuffer::Video(destination_frame)) =
        (&source, &destination)
    else {
        panic!("both uploads produce Video buffers");
    };
    driver
        .bgra_to_nv12(
            BgraSurface::from_frame(source_frame).expect("a BGRA surface"),
            Nv12Surface::from_frame(destination_frame).expect("an NV12 surface"),
            WIDTH,
            HEIGHT,
        )
        .expect("convert");
    driver.synchronize().expect("synchronize");

    let converted = download(&device, destination, WIDTH, HEIGHT);
    let luma_stride = converted.stride(0);
    let chroma_stride = converted.stride(1);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let [b, g, r, _] = pixel(x, y);
            let (expected, _, _) = bt709_limited(f32::from(r), f32::from(g), f32::from(b));
            assert_eq!(
                converted.data(0)[y as usize * luma_stride + x as usize],
                expected,
                "luma at {x},{y}"
            );
        }
    }
    for cy in 0..HEIGHT / 2 {
        for cx in 0..WIDTH / 2 {
            let mut sums = [0u32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let [b, g, r, _] = pixel(cx * 2 + dx, cy * 2 + dy);
                sums[0] += u32::from(b);
                sums[1] += u32::from(g);
                sums[2] += u32::from(r);
            }
            let (_, expected_u, expected_v) = bt709_limited(
                sums[2] as f32 / 4.0,
                sums[1] as f32 / 4.0,
                sums[0] as f32 / 4.0,
            );
            let at = cy as usize * chroma_stride + cx as usize * 2;
            assert_eq!(converted.data(1)[at], expected_u, "u at {cx},{cy}");
            assert_eq!(converted.data(1)[at + 1], expected_v, "v at {cx},{cy}");
        }
    }
}

fn cuda_surface_with(
    device: &crate::elements::CudaDevice,
    width: u32,
    height: u32,
    luma: impl Fn(u32, u32) -> u8,
    chroma: u8,
) -> Option<MediaBuffer> {
    let Ok(mut upload) = CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
    else {
        eprintln!("skipping: this machine has no usable CUDA frames context");
        return None;
    };
    let uploaded = capture(&mut upload);
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
    let y_stride = frame.stride(0);
    let plane = frame.data_mut(0);
    for y in 0..height {
        for x in 0..width {
            plane[y as usize * y_stride + x as usize] = luma(x, y);
        }
    }
    let uv_stride = frame.stride(1);
    frame.data_mut(1)[..uv_stride * (height / 2) as usize].fill(chroma);
    let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
    let mut slot = pool.get();
    *slot = frame;
    upload
        .consume(MediaBuffer::Video(Arc::new(slot)))
        .expect("upload");
    Some(uploaded.lock().unwrap().remove(0))
}

/// The kernel's whole contract: every blended byte matches the same
/// expression evaluated on the CPU. Hand-written PTX is only defensible
/// because this can be checked exactly rather than eyeballed.
#[test]
fn the_blend_kernel_matches_a_cpu_reference_byte_for_byte() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let driver = match CudaDriver::retain_primary() {
        Ok(driver) => driver,
        Err(error) => {
            eprintln!("skipping: no usable CUDA driver context ({error})");
            return;
        }
    };
    let (width, height) = (64u32, 64u32);
    // Ramps along different axes, so a swapped coordinate cannot pass.
    let Some(destination) = cuda_surface_with(&device, width, height, |_, y| (y * 3) as u8, 90)
    else {
        return;
    };
    let Some(source) = cuda_surface_with(&device, width, height, |x, _| (x * 4) as u8, 200) else {
        return;
    };
    let (MediaBuffer::Video(dst_frame), MediaBuffer::Video(src_frame)) = (&destination, &source)
    else {
        panic!("expected Video buffers");
    };
    let dst_surface = Nv12Surface::from_frame(dst_frame).expect("destination planes");
    let src_surface = Nv12Surface::from_frame(src_frame).expect("source planes");

    let alpha = 77u8;
    driver
        .blend_nv12(
            src_surface,
            dst_surface,
            Nv12Region {
                source_x: 0,
                source_y: 0,
                destination_x: 0,
                destination_y: 0,
                width,
                height,
            },
            alpha,
        )
        .expect("blend");
    driver.synchronize().expect("synchronize");

    let out = download(&device, destination.clone(), width, height);
    let stride = out.stride(0);
    let blend = |dst: u32, src: u32| {
        ((src * u32::from(alpha) + dst * (255 - u32::from(alpha)) + 127) / 255) as u8
    };
    for y in 0..height {
        for x in 0..width {
            let expected = blend(u32::from((y * 3) as u8), u32::from((x * 4) as u8));
            let actual = out.data(0)[y as usize * stride + x as usize];
            assert_eq!(
                actual, expected,
                "luma mismatch at ({x}, {y}): kernel {actual} != cpu {expected}"
            );
        }
    }
    let uv_stride = out.stride(1);
    let expected_chroma = blend(90, 200);
    for y in 0..height / 2 {
        for x in 0..width {
            let actual = out.data(1)[y as usize * uv_stride + x as usize];
            assert_eq!(
                actual, expected_chroma,
                "chroma mismatch at ({x}, {y}): kernel {actual} != cpu {expected_chroma}"
            );
        }
    }
}

/// The endpoints have to be exact, not merely close: a fully opaque
/// blend must equal the source, and a fully transparent one must leave
/// the destination untouched.
#[test]
fn alpha_endpoints_replace_and_preserve_exactly() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let Ok(driver) = CudaDriver::retain_primary() else {
        eprintln!("skipping: no usable CUDA driver context");
        return;
    };
    let (width, height) = (32u32, 32u32);
    for (alpha, expected) in [(255u8, 200u8), (0, 60)] {
        let Some(destination) = cuda_surface_with(&device, width, height, |_, _| 60, 128) else {
            return;
        };
        let Some(source) = cuda_surface_with(&device, width, height, |_, _| 200, 128) else {
            return;
        };
        let (MediaBuffer::Video(dst_frame), MediaBuffer::Video(src_frame)) =
            (&destination, &source)
        else {
            panic!("expected Video buffers");
        };
        driver
            .blend_nv12(
                Nv12Surface::from_frame(src_frame).expect("source planes"),
                Nv12Surface::from_frame(dst_frame).expect("destination planes"),
                Nv12Region {
                    source_x: 0,
                    source_y: 0,
                    destination_x: 0,
                    destination_y: 0,
                    width,
                    height,
                },
                alpha,
            )
            .expect("blend");
        driver.synchronize().expect("synchronize");

        let out = download(&device, destination.clone(), width, height);
        assert_eq!(
            out.data(0)[out.stride(0) * 5 + 5],
            expected,
            "alpha {alpha} must produce {expected}"
        );
    }
}

/// The two colors every background test in this crate uses, checked
/// against the BT.709 limited-range values they are defined to produce.
#[test]
fn black_and_white_map_to_limited_range_endpoints() {
    assert_eq!(rgb_to_bt709_limited(Color::BLACK), (16, 128, 128));
    let (y, u, v) = rgb_to_bt709_limited(Color::WHITE);
    assert_eq!(y, 235);
    assert!(
        u.abs_diff(128) <= 1 && v.abs_diff(128) <= 1,
        "white must be chroma-neutral, got ({u}, {v})"
    );
}
