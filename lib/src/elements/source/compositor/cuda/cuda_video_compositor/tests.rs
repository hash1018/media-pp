use std::sync::Mutex as StdMutex;

use super::super::super::text_layer::TextLayer;
use super::super::super::video_layer::VideoFit;
use super::*;
use crate::{
    color::Color,
    elements::{CudaDownload, CudaUpload},
    test_support::try_cuda_device,
};

struct CapturingSink {
    pp_log: PpLog,
    received: Arc<StdMutex<Vec<MediaBuffer>>>,
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
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.received.lock().unwrap().push(buf);
        Ok(())
    }
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

fn capture(element: &mut dyn Source) -> Arc<StdMutex<Vec<MediaBuffer>>> {
    let received = Arc::new(StdMutex::new(Vec::new()));
    element.src_pads()[0].link(Box::new(CapturingSink {
        received: received.clone(),
        pp_log: element_pp_log(ElementType::Other, "capture", None),
    }));
    received
}

fn options(width: u32, height: u32) -> VideoCompositorOptions {
    VideoCompositorOptions {
        width,
        height,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    }
}

/// A CUDA-resident NV12 frame of one flat luma value, which is what makes
/// a composed output checkable pixel by pixel.
fn cuda_frame(device: &CudaDevice, width: u32, height: u32, luma: u8) -> Option<MediaBuffer> {
    cuda_frame_with_pts(device, width, height, luma, 0)
}

/// A CUDA-resident BGRA frame, which is how an overlay arrives.
fn cuda_bgra_frame(
    device: &CudaDevice,
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
    frame.set_pts(Some(0));
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

fn cuda_frame_with_pts(
    device: &CudaDevice,
    width: u32,
    height: u32,
    luma: u8,
    pts: i64,
) -> Option<MediaBuffer> {
    let Ok(mut upload) = CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
    else {
        eprintln!("skipping: this machine has no usable CUDA frames context");
        return None;
    };
    let uploaded = capture(&mut upload);
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
    frame.set_pts(Some(pts));
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

/// Brings one composed CUDA frame back so its pixels can be asserted on.
fn download(
    device: &CudaDevice,
    frame: UnboundObjectPoolRef<ffmpeg::frame::Video>,
    width: u32,
    height: u32,
) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
    let mut download = CudaDownload::new("download", device, CudaFrameFormat::Nv12, width, height);
    let received = capture(&mut download);
    download
        .consume(MediaBuffer::Video(Arc::new(frame)))
        .expect("download");
    let buf = received.lock().unwrap().remove(0);
    match buf {
        MediaBuffer::Video(frame) => frame,
        other => panic!("expected a Video buffer, got {}", other.kind()),
    }
}

fn luma_at(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> u8 {
    frame.data(0)[y * frame.stride(0) + x]
}

/// A CUDA-resident NV12 frame whose luma says which quadrant a pixel is
/// in, so what was drawn is readable from the output alone.
fn cuda_quadrant_frame(device: &CudaDevice, width: u32, height: u32) -> Option<MediaBuffer> {
    let Ok(mut upload) = CudaUpload::new("upload", device, CudaFrameFormat::Nv12, width, height)
    else {
        eprintln!("skipping: this machine has no usable CUDA frames context");
        return None;
    };
    let uploaded = capture(&mut upload);
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
    frame.set_pts(Some(0));
    let stride = frame.stride(0);
    for y in 0..height as usize {
        for x in 0..width as usize {
            frame.data_mut(0)[y * stride + x] =
                match (x >= width as usize / 2, y >= height as usize / 2) {
                    (false, false) => 40,
                    (true, false) => 90,
                    (false, true) => 150,
                    (true, true) => 220,
                };
        }
    }
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

/// Cropping, all the way through the GPU path: the layer draws its source
/// region and nothing else, and the region is what the scaler is asked
/// for rather than something trimmed off a full-frame scale.
#[test]
fn a_layer_draws_only_its_source_region() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (128u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut input = handle
        .add_source(
            "layer",
            VideoLayer {
                fit: VideoFit::Stretch,
                // The bottom-right quadrant, which nothing else in the
                // frame shares a value with. Its offset is deliberately
                // not a multiple of CUDA's 512-byte texture alignment:
                // handing `scale_cuda` a view at such an offset fails
                // outright, which is what `CropScratch` exists for.
                source: Some(VideoSourceRect::new(130, 128, 126, 128)),
                ..VideoLayer::new(VideoRect::new(0, 0, width, height))
            },
        )
        .expect("add source");
    let Some(frame) = cuda_quadrant_frame(&device, 256, 256) else {
        return;
    };
    input.sink.consume(frame).expect("frame");

    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);

    for (x, y) in [(4, 4), (64, 64), (123, 123)] {
        let luma = luma_at(&out, x, y);
        assert!(
            luma.abs_diff(220) <= 2,
            "every pixel must come from the cropped quadrant; got {luma} at ({x},{y})"
        );
    }
}

/// What dragging a crop handle is: the region changes on every frame,
/// for as long as the pointer is down. Every one of those has to compose,
/// because a compositor that fails part way through a gesture leaves the
/// Scene blank and does not come back.
#[test]
fn a_region_that_changes_every_frame_keeps_composing() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (256u32, 256u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut input = handle
        .add_source(
            "layer",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, width, height))
            },
        )
        .expect("add source");

    for step in 0..60u32 {
        let Some(frame) = cuda_frame_with_pts(&device, 640, 480, 200, i64::from(step)) else {
            return;
        };
        input.sink.consume(frame).expect("frame");
        // Only the left edge moves, so the region's shape changes as it
        // shrinks and each axis passes the output's size on its own —
        // which is the case `scale_cuda` answers with a blank surface
        // and `scale_graph::detour` exists for. Odd amounts, so the
        // alignment the NV12 path applies is exercised too.
        let cut = step * 3;
        input
            .layer
            .set_source(Some(VideoSourceRect::new(cut, 0, 640 - cut, 480)))
            .expect("set source");

        let composed = compositor.compose_frame().expect("compose");
        let out = download(&device, composed, width, height);
        let luma = luma_at(&out, 128, 128);
        assert!(
            luma.abs_diff(200) <= 2,
            "the layer went blank at step {step} (cut {cut}): luma {luma}"
        );
    }
}

/// A layer scaled so that exactly one of its dimensions matches the
/// source's must still be drawn.
///
/// This is the shape `obs-rs` reported as a green flash roughly once in
/// six hundred frames while a layer was dragged: `scale_cuda` answers such
/// a size with an entirely zero surface, the blit copies that into the
/// composite, and zeroed NV12 resolves to green. Dragging sweeps the
/// scaled size continuously, so it passes through the source's own on the
/// way. See `scale_graph::detour`.
#[test]
fn a_layer_scaled_along_one_axis_is_not_blank() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (256u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    // The layer keeps the source's height and takes half its width, which
    // is what a horizontal drag passes through.
    let mut input = handle
        .add_source(
            "layer",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 64, height))
            },
        )
        .expect("add source");
    let Some(frame) = cuda_frame(&device, 128, height, 200) else {
        return;
    };
    input.sink.consume(frame).expect("frame");

    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);

    // Inside the layer: the source's own value, within the one level a
    // second resampling pass can round away.
    for (x, y) in [(2, 2), (32, 64), (61, 125)] {
        let luma = luma_at(&out, x, y);
        assert!(
            luma.abs_diff(200) <= 1,
            "the layer is blank at ({x},{y}): luma {luma}, not 200"
        );
    }
    // Outside it: still the background, so the layer has not been
    // stretched over the whole canvas to hide the defect.
    assert_eq!(
        luma_at(&out, 200, 64),
        16,
        "everything outside the layer must be the background"
    );
}

/// The composition contract: layers land where their rectangles say, the
/// higher `z_index` wins where they overlap, and everything else is the
/// background.
#[test]
fn composes_layers_in_z_order_at_their_rectangles() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (128u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };

    let mut back = handle
        .add_source(
            "back",
            VideoLayer {
                z_index: 0,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
            },
        )
        .expect("add back");
    let mut front = handle
        .add_source(
            "front",
            VideoLayer {
                z_index: 1,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(32, 32, 64, 64))
            },
        )
        .expect("add front");

    let Some(back_frame) = cuda_frame(&device, 32, 32, 100) else {
        return;
    };
    let Some(front_frame) = cuda_frame(&device, 32, 32, 200) else {
        return;
    };
    back.sink.consume(back_frame).expect("back frame");
    front.sink.consume(front_frame).expect("front frame");

    let composed = compositor.compose_frame().expect("compose");
    assert_eq!(composed.format(), ffmpeg::format::Pixel::CUDA);
    assert_eq!(composed.pts(), Some(0));
    let out = download(&device, composed, width, height);

    assert_eq!(luma_at(&out, 10, 10), 100, "the back layer is missing");
    assert_eq!(luma_at(&out, 80, 80), 200, "the front layer is missing");
    assert_eq!(
        luma_at(&out, 40, 40),
        200,
        "the higher z_index must win where the layers overlap"
    );
    assert_eq!(
        luma_at(&out, 120, 10),
        16,
        "everything outside a layer must be the background"
    );
}

/// An overlay is the layer that does not cover what is under it.
///
/// A capture is opaque and is placed with a copy; a Drawing, a caption,
/// anything drawn *over* the scene, is mostly nothing at all and has to
/// leave the picture beneath showing. That is why a layer may arrive as
/// BGRA — see [`LayerFormat`] — and this is the property that costs it.
#[test]
fn a_bgra_overlay_leaves_the_layer_under_it_showing() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (128u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut back = handle
        .add_source(
            "back",
            VideoLayer {
                z_index: 0,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 128, 128))
            },
        )
        .expect("add back");
    let mut overlay = handle
        .add_source(
            "overlay",
            VideoLayer {
                z_index: 1,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 128, 128))
            },
        )
        .expect("add overlay");

    let Some(back_frame) = cuda_frame(&device, 64, 64, 100) else {
        return;
    };
    // Opaque white down the left half, nothing at all down the right.
    let Some(overlay_frame) = cuda_bgra_frame(&device, 64, 64, |x, _| {
        if x < 32 {
            [255, 255, 255, 255]
        } else {
            [0, 0, 0, 0]
        }
    }) else {
        return;
    };
    back.sink.consume(back_frame).expect("back frame");
    overlay
        .sink
        .consume(overlay_frame)
        .expect("the compositor takes a BGRA overlay");

    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);

    assert_eq!(
        luma_at(&out, 100, 64),
        100,
        "where the overlay is transparent the layer under it has to show"
    );
    assert!(
        luma_at(&out, 20, 64) > 200,
        "and where it is opaque the overlay itself has to, got {}",
        luma_at(&out, 20, 64)
    );
}

/// A tick that finds nothing changed hands out the surface it composed
/// last rather than composing the same picture again — and a layer that
/// moves puts it straight back to work.
#[test]
fn an_unchanged_scene_is_composed_once() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (128u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut input = handle
        .add_source("only", VideoLayer::new(VideoRect::new(0, 0, 64, 64)))
        .expect("add source");
    let Some(frame) = cuda_frame(&device, 32, 32, 100) else {
        return;
    };
    input.sink.consume(frame).expect("frame");

    let composed = compositor.compose_frame().expect("compose");
    let surface = picture_id(&composed);
    assert_eq!(composed.pts(), Some(0));

    let repeated = compositor.compose_frame().expect("repeat");
    assert_eq!(
        picture_id(&repeated),
        surface,
        "nothing changed, so this is the picture already composed"
    );
    assert_eq!(
        repeated.pts(),
        Some(1),
        "a repeat carries this tick's timestamp, not the one it copies"
    );

    input
        .layer
        .set_layer(VideoLayer::new(VideoRect::new(16, 16, 64, 64)))
        .expect("move the layer");
    let moved = compositor.compose_frame().expect("compose again");
    assert_ne!(
        picture_id(&moved),
        surface,
        "a moved layer is a different picture and must be composed"
    );
}

/// The layer hands back the surface it will draw — the frame it was handed,
/// by reference — and nothing once its registration is gone.
#[test]
fn a_layer_hands_back_the_frame_it_will_draw_until_it_is_removed() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let Ok((_compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(64, 64))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut input = handle
        .add_source("still", VideoLayer::new(VideoRect::new(0, 0, 64, 64)))
        .expect("add source");
    assert!(
        input.layer.latest_frame().is_none(),
        "nothing has arrived yet"
    );

    let Some(MediaBuffer::Video(frame)) = cuda_frame(&device, 32, 32, 100) else {
        return;
    };
    input
        .sink
        .consume(MediaBuffer::Video(frame.clone()))
        .expect("frame");
    let latest = input.layer.latest_frame().expect("the frame handed over");
    assert!(Arc::ptr_eq(&latest, &frame));

    handle.remove_source("still");
    assert!(input.layer.latest_frame().is_none());
}

/// Runtime control: moving and hiding a layer changes the next frame, and
/// costs no rebuild of anything — the whole reason this backend places
/// with a copy rather than a filter.
#[test]
fn layer_handle_moves_and_hides_a_live_source() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (128u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut input = handle
        .add_source(
            "layer",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 32, 32))
            },
        )
        .expect("add");
    let Some(frame) = cuda_frame(&device, 32, 32, 180) else {
        return;
    };
    input.sink.consume(frame).expect("frame");

    let first = compositor.compose_frame().expect("compose");
    let out = download(&device, first, width, height);
    assert_eq!(luma_at(&out, 10, 10), 180);
    assert_eq!(luma_at(&out, 74, 74), 16);

    input
        .layer
        .set_rect(VideoRect::new(64, 64, 32, 32))
        .expect("move");
    let moved = compositor.compose_frame().expect("compose");
    let out = download(&device, moved, width, height);
    assert_eq!(luma_at(&out, 10, 10), 16, "the layer did not leave");
    assert_eq!(luma_at(&out, 74, 74), 180, "the layer did not arrive");

    input.layer.set_visible(false).expect("hide");
    let hidden = compositor.compose_frame().expect("compose");
    let out = download(&device, hidden, width, height);
    assert_eq!(luma_at(&out, 74, 74), 16, "a hidden layer was still drawn");
}

/// `Cover` is the fit no CUDA filter can express — it needs a crop, which
/// is exactly what a 2D copy does. It must fill its rectangle completely,
/// where `Contain` leaves background visible in the same rectangle.
#[test]
fn cover_fills_its_rectangle_where_contain_letterboxes() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (128u32, 128u32);

    // A wide source in a square rectangle: `Contain` letterboxes it
    // vertically, `Cover` crops the sides away instead.
    let composed_with = |fit: VideoFit| {
        let (mut compositor, handle) =
            CudaVideoCompositor::new("compositor", &device, options(width, height))
                .expect("compositor");
        let mut input = handle
            .add_source(
                "layer",
                VideoLayer {
                    fit,
                    ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
                },
            )
            .expect("add");
        let frame = cuda_frame(&device, 64, 16, 200).expect("frame");
        input.sink.consume(frame).expect("frame");
        let composed = compositor.compose_frame().expect("compose");
        download(&device, composed, width, height)
    };

    let contain = composed_with(VideoFit::Contain);
    assert_eq!(
        luma_at(&contain, 32, 32),
        200,
        "Contain must draw the image in the middle of its rectangle"
    );
    assert_eq!(
        luma_at(&contain, 32, 4),
        16,
        "Contain must leave background above the image"
    );

    let cover = composed_with(VideoFit::Cover);
    assert_eq!(
        luma_at(&cover, 32, 32),
        200,
        "Cover must draw the image in the middle of its rectangle"
    );
    assert_eq!(
        luma_at(&cover, 32, 4),
        200,
        "Cover must fill its rectangle, cropping the overflow"
    );
    assert_eq!(
        luma_at(&cover, 32, 80),
        16,
        "Cover must not draw outside its rectangle"
    );
}

/// A translucent layer is mixed with what is under it, by the kernel the
/// driver JIT-compiles from this crate's own PTX. The expected value is
/// the same expression evaluated here, so a wrong blend cannot pass.
#[test]
fn a_translucent_layer_is_blended_with_the_background() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (128u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut input = handle
        .add_source(
            "layer",
            VideoLayer {
                opacity: 0.5,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
            },
        )
        .expect("a translucent layer registers");
    let Some(frame) = cuda_frame(&device, 32, 32, 200) else {
        return;
    };
    input.sink.consume(frame).expect("frame");

    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);

    // Background is `Color::BLACK`, which is luma 16 in limited range.
    let alpha = (0.5f32 * 255.0).round() as u32;
    let expected = ((200 * alpha + 16 * (255 - alpha) + 127) / 255) as u8;
    assert_eq!(
        luma_at(&out, 10, 10),
        expected,
        "the layer was not blended with the background"
    );
    assert_eq!(
        luma_at(&out, 100, 100),
        16,
        "outside the layer must stay background"
    );

    // The endpoints still behave as before: fully opaque replaces,
    // fully transparent draws nothing.
    input.layer.set_opacity(1.0).expect("opaque");
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    assert_eq!(luma_at(&out, 10, 10), 200);

    input.layer.set_opacity(0.0).expect("transparent");
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    assert_eq!(luma_at(&out, 10, 10), 16);
}

/// A font every machine running these tests has. Skips rather than
/// fails when it is missing, the same way a hardware test skips without
/// a device.
fn try_font() -> Option<Vec<u8>> {
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ] {
        if let Ok(data) = std::fs::read(path) {
            return Some(data);
        }
    }
    eprintln!("skipping: no DejaVuSans on this machine to rasterize with");
    None
}

/// The text layer's contract: nothing is drawn until `set_text`, then the
/// glyphs land on the canvas, and clearing the text removes them again.
#[test]
fn a_text_layer_draws_after_set_text_and_clears_when_emptied() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let Some(font) = try_font() else {
        return;
    };
    let (width, height) = (256u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };

    let mut layer = TextLayer::new(font);
    layer.font_size = 48.0;
    layer.color = Color::WHITE;
    layer.x = 8;
    layer.y = 8;
    let text = handle
        .add_text_layer("clock", layer)
        .expect("add a text layer");

    // Nothing rasterized yet: the canvas is pure background.
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    let background = luma_at(&out, 10, 10);
    assert_eq!(background, 16, "an empty text layer drew something");

    text.set_text("HELLO").expect("set_text");
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    let lit = (0..64u32)
        .flat_map(|y| (0..200u32).map(move |x| (x, y)))
        .filter(|(x, y)| luma_at(&out, *x as usize, *y as usize) > background + 40)
        .count();
    assert!(
        lit > 100,
        "the text did not reach the canvas ({lit} bright pixels)"
    );

    // Text with no drawable glyphs clears the layer rather than erroring.
    text.set_text("   ").expect("blank set_text");
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    for y in 0..64usize {
        for x in 0..200usize {
            assert_eq!(
                luma_at(&out, x, y),
                background,
                "clearing the text left something at ({x}, {y})"
            );
        }
    }
}

/// Position, visibility, and opacity all act on the next composed frame,
/// and opacity is a real blend here rather than an on/off switch.
#[test]
fn text_position_visibility_and_opacity_take_effect() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let Some(font) = try_font() else {
        return;
    };
    let (width, height) = (256u32, 128u32);
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(width, height))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut layer = TextLayer::new(font);
    layer.font_size = 48.0;
    layer.color = Color::WHITE;
    let text = handle
        .add_text_layer("clock", layer)
        .expect("add a text layer");
    text.set_text("IIII").expect("set_text");

    let brightest = |frame: &ffmpeg::frame::Video, x0: usize, x1: usize| {
        (0..64usize)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .map(|(x, y)| luma_at(frame, x, y))
            .max()
            .unwrap_or(0)
    };

    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    assert!(brightest(&out, 0, 100) > 100, "text is not at the origin");
    assert_eq!(
        brightest(&out, 150, 250),
        16,
        "text is already on the right"
    );

    text.set_position(150, 0);
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    assert_eq!(brightest(&out, 0, 100), 16, "the text did not leave");
    assert!(brightest(&out, 150, 250) > 100, "the text did not arrive");

    // Half opacity over a luma-16 background must land near the midpoint
    // rather than at either end.
    text.set_opacity(0.5).expect("half opacity");
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    let half = brightest(&out, 150, 250);
    assert!(
        (100..=170).contains(&half),
        "half-opacity text should be mid-grey, got {half}"
    );

    text.set_visible(false);
    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed, width, height);
    assert_eq!(brightest(&out, 150, 250), 16, "a hidden text layer drew");
}

/// Input validation happens where the input is named, so a CPU frame or
/// one from another CUDA context never reaches a device pointer.
#[test]
fn a_cpu_frame_and_a_foreign_context_frame_are_typed_errors() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let Ok((_compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(64, 64))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    let mut input = handle
        .add_source("layer", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
        .expect("add");

    let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
    let mut slot = pool.get();
    *slot = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 32, 32);
    let error = input
        .sink
        .consume(MediaBuffer::Video(Arc::new(slot)))
        .expect_err("a CPU frame must not be composited");
    assert!(
        error.to_string().contains("only composites CUDA frames"),
        "expected UnsupportedFormat, got {error}"
    );

    // Directly, not `try_cuda_device` again: the lock it returns is
    // already held for this test and does not nest.
    let other_device = CudaDevice::new().expect("a second CUDA device");
    let Some(foreign) = cuda_frame(&other_device, 32, 32, 100) else {
        return;
    };
    let error = input
        .sink
        .consume(foreign)
        .expect_err("a frame from a foreign CUDA context must not be composited");
    assert!(
        error.to_string().contains("different CUDA context"),
        "expected ForeignContext, got {error}"
    );
}

/// Output timing: this element emits on its own clock, so PTS are
/// contiguous ticks of its own time base regardless of what the inputs
/// carried.
#[test]
fn output_pts_are_contiguous_ticks_of_its_own_time_base() {
    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let Ok((mut compositor, handle)) =
        CudaVideoCompositor::new("compositor", &device, options(64, 64))
    else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };
    assert_eq!(compositor.time_base(), ffmpeg::Rational::new(1, 30));
    let mut input = handle
        .add_source("layer", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
        .expect("add");
    let Some(mut frame) = cuda_frame(&device, 32, 32, 100) else {
        return;
    };
    if let MediaBuffer::Video(video) = &mut frame {
        // A wildly different input timeline, which must not leak through.
        // `get_mut` rather than a cast through `Arc::as_ptr`: the frame has
        // not been published yet, so this is the one moment it is uniquely
        // owned and can be written at all.
        Arc::get_mut(video)
            .expect("an unpublished frame is uniquely owned")
            .set_pts(Some(9_999));
    }
    input.sink.consume(frame).expect("frame");

    for expected in 0..3 {
        let composed = compositor.compose_frame().expect("compose");
        assert_eq!(composed.pts(), Some(expected));
    }
}

/// The canvas says what it is — BT.709, limited range, which is what every
/// fill and blend into it converts with — so what reads it downstream reads
/// it right. Held end to end over the chain a screenshot takes: composed,
/// downloaded, and scaled to RGB on the CPU. Untagged, that chain gave
/// (230, 20, 20) back as (211, 0, 22), because swscale took the canvas for
/// BT.601.
#[test]
fn the_canvas_says_it_is_bt709_and_reads_back_as_the_colour_it_was_filled_with() {
    use crate::elements::SwScaler;

    let Some((device, _cuda_lock)) = try_cuda_device() else {
        return;
    };
    let (width, height) = (64u32, 64u32);
    let Ok((mut compositor, _handle)) = CudaVideoCompositor::new(
        "compositor",
        &device,
        VideoCompositorOptions {
            background: Color::new(230, 20, 20),
            ..options(width, height)
        },
    ) else {
        eprintln!("skipping: this machine cannot open a CUDA compositor");
        return;
    };

    let composed = compositor.compose_frame().expect("compose");
    assert_eq!(composed.color_space(), ffmpeg::color::Space::BT709);
    assert_eq!(composed.color_range(), ffmpeg::color::Range::MPEG);
    assert_eq!(composed.color_primaries(), ffmpeg::color::Primaries::BT709);

    let downloaded = download(&device, composed, width, height);
    assert_eq!(
        downloaded.color_space(),
        ffmpeg::color::Space::BT709,
        "the download carries the description across"
    );

    let mut scaler = SwScaler::new(
        "to-rgb",
        ffmpeg::format::Pixel::RGB24,
        width,
        height,
        ffmpeg::software::scaling::Flags::BILINEAR,
    );
    let received = capture(&mut scaler);
    scaler
        .consume(MediaBuffer::Video(downloaded))
        .expect("scale");
    let MediaBuffer::Video(rgb) = received.lock().unwrap().remove(0) else {
        panic!("expected a Video buffer");
    };
    let at = 32 * rgb.stride(0) + 32 * 3;
    let got = [rgb.data(0)[at], rgb.data(0)[at + 1], rgb.data(0)[at + 2]];
    assert!(
        got.iter()
            .zip([230u8, 20, 20])
            .all(|(got, want)| got.abs_diff(want) <= 3),
        "the canvas read back as {got:?}"
    );
}
