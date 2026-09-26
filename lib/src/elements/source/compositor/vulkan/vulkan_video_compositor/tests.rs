use std::sync::{Arc, Mutex};

use super::super::super::text_layer::TextLayer;
use super::super::super::video_layer::{VideoFit, VideoSourceRect};
use super::*;
use crate::{
    color::Color,
    elements::{VulkanDownload, VulkanUpload},
    test_support::{CapturingSink, try_vulkan_device},
};

fn options(width: u32, height: u32) -> VideoCompositorOptions {
    VideoCompositorOptions {
        width,
        height,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background_alpha: 255,
        mode: crate::elements::RenderMode::Live,
        background: Color::BLACK,
    }
}

type Received = Arc<Mutex<Vec<MediaBuffer>>>;

fn capture(source: &mut dyn Source) -> Received {
    let received = Arc::new(Mutex::new(Vec::new()));
    source.src_pads()[0].link(Box::new(CapturingSink {
        received: received.clone(),
        pp_log: element_pp_log(ElementType::Other, "capture", None),
    }));
    received
}

/// `frame` uploaded to `device`.
fn upload(device: &VulkanDevice, frame: ffmpeg::frame::Video) -> MediaBuffer {
    let mut upload = VulkanUpload::new("upload", device);
    let uploaded = capture(&mut upload);
    upload.consume(MediaBuffer::video(frame)).expect("upload");
    uploaded.lock().unwrap().remove(0)
}

/// A Vulkan NV12 frame of one flat luma value, grey, which is what makes a
/// composed output checkable pixel by pixel.
fn nv12_frame(device: &VulkanDevice, width: u32, height: u32, luma: u8) -> MediaBuffer {
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
    frame.set_pts(Some(0));
    frame.data_mut(0).fill(luma);
    frame.data_mut(1).fill(128);
    upload(device, frame)
}

/// A Vulkan BGRA frame, which is how an overlay arrives.
fn bgra_frame(
    device: &VulkanDevice,
    width: u32,
    height: u32,
    pixel: impl Fn(u32, u32) -> [u8; 4],
) -> MediaBuffer {
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::BGRA, width, height);
    frame.set_pts(Some(0));
    let stride = frame.stride(0);
    for y in 0..height {
        let row = &mut frame.data_mut(0)[y as usize * stride..];
        for x in 0..width {
            row[x as usize * 4..x as usize * 4 + 4].copy_from_slice(&pixel(x, y));
        }
    }
    upload(device, frame)
}

/// One composed frame brought back to system memory.
fn download(
    device: &VulkanDevice,
    frame: UnboundObjectPoolRef<ffmpeg::frame::Video>,
) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
    let mut download = VulkanDownload::new("download", device);
    let received = capture(&mut download);
    download
        .consume(MediaBuffer::Video(Arc::new(frame)))
        .expect("download");
    match received.lock().unwrap().remove(0) {
        MediaBuffer::Video(frame) => frame,
        other => panic!("expected a Video buffer, got {}", other.kind()),
    }
}

fn luma_at(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> u8 {
    frame.data(0)[y * frame.stride(0) + x]
}

fn bgra_at(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> [u8; 4] {
    let at = y * frame.stride(0) + x * 4;
    frame.data(0)[at..at + 4].try_into().unwrap()
}

/// Within one step of `expected` — a sample that went through a canvas of
/// eight bits a channel, in RGB, and back.
fn near(got: u8, expected: u8, what: &str) {
    assert!(
        got.abs_diff(expected) <= 1,
        "{what}: got {got}, expected {expected}"
    );
}

fn nv12_compositor(
    device: &VulkanDevice,
    width: u32,
    height: u32,
) -> (VulkanVideoCompositor, VulkanVideoCompositorHandle) {
    VulkanVideoCompositor::with_format(
        "compositor",
        device,
        options(width, height),
        VulkanFrameFormat::Nv12,
    )
    .expect("an NV12 Vulkan compositor")
}

/// The composition contract: layers land where their rectangles say, the
/// higher `z_index` wins where they overlap, and everything else is the
/// background.
#[test]
fn composes_layers_in_z_order_at_their_rectangles() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let (mut compositor, handle) = nv12_compositor(&device, 128, 128);
    let mut back = handle
        .add_source(
            "back",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
            },
        )
        .unwrap();
    let mut front = handle
        .add_source(
            "front",
            VideoLayer {
                z_index: 1,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(32, 32, 64, 64))
            },
        )
        .unwrap();
    back.sink.consume(nv12_frame(&device, 32, 32, 100)).unwrap();
    front
        .sink
        .consume(nv12_frame(&device, 32, 32, 200))
        .unwrap();

    let composed = compositor.compose_frame().expect("compose");
    assert_eq!(composed.format(), ffmpeg::format::Pixel::VULKAN);
    assert_eq!(composed.pts(), Some(0));
    let out = download(&device, composed);
    assert_eq!(out.format(), ffmpeg::format::Pixel::NV12);
    near(luma_at(&out, 10, 10), 100, "the back layer");
    near(luma_at(&out, 80, 80), 200, "the front layer");
    near(
        luma_at(&out, 40, 40),
        200,
        "the higher z_index where they overlap",
    );
    near(
        luma_at(&out, 120, 10),
        16,
        "the background outside every layer",
    );
    // Grey in, grey out: the chroma is neutral.
    let chroma = &out.data(1)[..2];
    assert!(
        chroma.iter().all(|&sample| sample.abs_diff(128) <= 1),
        "{chroma:?}"
    );
}

/// On a BGRA canvas an overlay's alpha leaves what is under it showing, a
/// translucent background stays translucent where no layer drew, and a
/// grey NV12 layer is the grey its luma says.
#[test]
fn a_bgra_canvas_blends_by_alpha_and_keeps_its_transparency() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let (mut compositor, handle) = VulkanVideoCompositor::new(
        "compositor",
        &device,
        VideoCompositorOptions {
            background_alpha: 0,
            ..options(128, 64)
        },
    )
    .expect("a BGRA Vulkan compositor");
    let mut back = handle
        .add_source(
            "back",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
            },
        )
        .unwrap();
    let mut overlay = handle
        .add_source(
            "overlay",
            VideoLayer {
                z_index: 1,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
            },
        )
        .unwrap();
    // Luma 126 at limited range is grey 128.
    back.sink.consume(nv12_frame(&device, 32, 32, 126)).unwrap();
    // Opaque red at the top, half-transparent blue below.
    overlay
        .sink
        .consume(bgra_frame(&device, 32, 32, |_, y| {
            if y < 16 {
                [0, 0, 255, 255]
            } else {
                [255, 0, 0, 128]
            }
        }))
        .unwrap();

    let composed = compositor.compose_frame().expect("compose");
    let out = download(&device, composed);
    assert_eq!(out.format(), ffmpeg::format::Pixel::BGRA);
    assert_eq!(bgra_at(&out, 10, 10), [0, 0, 255, 255], "opaque red covers");
    let [b, g, r, a] = bgra_at(&out, 10, 50);
    near(b, 192, "half of blue over grey");
    near(g, 64, "half of grey's green");
    near(r, 64, "half of grey's red");
    assert_eq!(a, 255, "over an opaque layer");
    assert_eq!(
        bgra_at(&out, 100, 30)[3],
        0,
        "where nothing drew, the background's own alpha"
    );
}

/// A layer's picture fits its rectangle as its `fit` says: `Contain`
/// letterboxes a wide picture in a square, `Cover` fills it, cropping.
#[test]
fn cover_fills_its_rectangle_where_contain_letterboxes() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    for (fit, corner) in [(VideoFit::Contain, 16), (VideoFit::Cover, 200)] {
        let (mut compositor, handle) = nv12_compositor(&device, 64, 64);
        let mut input = handle
            .add_source(
                "wide",
                VideoLayer {
                    fit,
                    ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
                },
            )
            .unwrap();
        input
            .sink
            .consume(nv12_frame(&device, 64, 32, 200))
            .unwrap();
        let out = download(&device, compositor.compose_frame().unwrap());
        near(luma_at(&out, 32, 32), 200, "the middle is the picture");
        near(
            luma_at(&out, 32, 2),
            corner,
            &format!("the top edge, {fit:?}"),
        );
    }
}

/// A source region draws that part of the picture alone, stretched over
/// the rectangle.
#[test]
fn a_layer_draws_only_its_source_region() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let (mut compositor, handle) = nv12_compositor(&device, 64, 64);
    let mut input = handle
        .add_source(
            "quadrants",
            VideoLayer {
                fit: VideoFit::Stretch,
                source: Some(VideoSourceRect::new(32, 32, 32, 32)),
                ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
            },
        )
        .unwrap();
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 64);
    frame.set_pts(Some(0));
    let stride = frame.stride(0);
    for y in 0..64 {
        for x in 0..64 {
            frame.data_mut(0)[y * stride + x] = if x >= 32 && y >= 32 { 220 } else { 40 };
        }
    }
    frame.data_mut(1).fill(128);
    input.sink.consume(upload(&device, frame)).unwrap();
    let out = download(&device, compositor.compose_frame().unwrap());
    for (x, y) in [(4, 4), (60, 4), (4, 60), (32, 32)] {
        near(
            luma_at(&out, x, y),
            220,
            &format!("({x}, {y}) is the bottom-right quadrant"),
        );
    }
}

/// A translucent layer is mixed with what is under it by its opacity.
#[test]
fn a_translucent_layer_is_blended_with_the_background() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let (mut compositor, handle) = nv12_compositor(&device, 64, 64);
    let mut input = handle
        .add_source(
            "half",
            VideoLayer {
                opacity: 0.5,
                ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
            },
        )
        .unwrap();
    input
        .sink
        .consume(nv12_frame(&device, 32, 32, 235))
        .unwrap();
    let out = download(&device, compositor.compose_frame().unwrap());
    // Half of white over black is grey 128 in RGB, luma 126.
    near(luma_at(&out, 32, 32), 126, "half of white over black");
}

/// A tick that finds nothing changed hands out the frame it composed last,
/// under its own timestamp, and a moved layer puts it back to work.
#[test]
fn an_unchanged_scene_is_composed_once() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let (mut compositor, handle) = nv12_compositor(&device, 64, 64);
    let mut input = handle
        .add_source("only", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
        .unwrap();
    input
        .sink
        .consume(nv12_frame(&device, 32, 32, 100))
        .unwrap();
    let composed = compositor.compose_frame().unwrap();
    let picture = picture_id(&composed);
    let repeated = compositor.compose_frame().unwrap();
    assert_eq!(
        picture_id(&repeated),
        picture,
        "the picture already composed"
    );
    assert_eq!(repeated.pts(), Some(1), "under this tick's timestamp");
    input
        .layer
        .set_rect(VideoRect::new(16, 16, 32, 32))
        .unwrap();
    let moved = compositor.compose_frame().unwrap();
    assert_ne!(
        picture_id(&moved),
        picture,
        "a moved layer is composed again"
    );
}

fn try_font() -> Option<Vec<u8>> {
    [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    .iter()
    .find_map(|path| std::fs::read(path).ok())
    .or_else(|| {
        eprintln!("skipping: no system font to draw text with");
        None
    })
}

/// A text layer draws in its colour once it has text, stacks by its
/// `z_index` among the video layers, and clears when emptied.
#[test]
fn a_text_layer_draws_and_stacks_by_its_z_index() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let Some(font) = try_font() else {
        return;
    };
    let (mut compositor, handle) =
        VulkanVideoCompositor::new("compositor", &device, options(256, 128)).unwrap();
    let text = handle
        .add_text_layer(
            "caption",
            TextLayer {
                font_size: 96.0,
                color: Color::new(255, 0, 0),
                ..TextLayer::new(font)
            },
        )
        .unwrap();
    text.set_text("MMMM").unwrap();
    let red = |frame: &ffmpeg::frame::Video| {
        (0..frame.height() as usize)
            .flat_map(|y| (0..frame.width() as usize).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let [b, g, r, _] = bgra_at(frame, x, y);
                r > 200 && g < 40 && b < 40
            })
            .count()
    };
    let drawn = red(&download(&device, compositor.compose_frame().unwrap()));
    assert!(
        drawn > 500,
        "the text is drawn in its colour: {drawn} red pixels"
    );

    // A video layer over it at a higher z_index hides it.
    let mut cover = handle
        .add_source(
            "cover",
            VideoLayer {
                z_index: 1,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 256, 128))
            },
        )
        .unwrap();
    cover
        .sink
        .consume(nv12_frame(&device, 32, 32, 126))
        .unwrap();
    let covered = red(&download(&device, compositor.compose_frame().unwrap()));
    assert_eq!(covered, 0, "under a video layer of a higher z_index");
    text.set_z_index(2);
    let raised = red(&download(&device, compositor.compose_frame().unwrap()));
    assert!(raised > 500, "raised over it again: {raised}");

    text.set_text("").unwrap();
    let cleared = red(&download(&device, compositor.compose_frame().unwrap()));
    assert_eq!(cleared, 0, "empty text draws nothing");
}

/// A frame in system memory and one from another device are refused by
/// the sink, naming why, and the input keeps what it had.
#[test]
fn a_cpu_frame_and_a_foreign_device_frame_are_typed_errors() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let (_compositor, handle) = nv12_compositor(&device, 64, 64);
    let mut input = handle
        .add_source("strict", VideoLayer::new(VideoRect::new(0, 0, 64, 64)))
        .unwrap();
    let cpu = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 32, 32);
    let error = input.sink.consume(MediaBuffer::video(cpu)).unwrap_err();
    assert!(error.to_string().contains("upload it first"), "{error}");

    let other = VulkanDevice::new().expect("a second device");
    let error = input
        .sink
        .consume(nv12_frame(&other, 32, 32, 100))
        .unwrap_err();
    assert!(
        error.to_string().contains("another Vulkan device"),
        "{error}"
    );
    assert!(input.layer.latest_frame().is_none(), "nothing was taken");
}

/// An NV12 canvas refuses a translucent background, which it has nowhere
/// to keep.
#[test]
fn an_nv12_canvas_refuses_a_translucent_background() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let refused = VulkanVideoCompositor::with_format(
        "compositor",
        &device,
        VideoCompositorOptions {
            background_alpha: 128,
            ..options(64, 64)
        },
        VulkanFrameFormat::Nv12,
    );
    assert!(matches!(
        refused,
        Err(VulkanVideoCompositorError::TranslucentBackground(128))
    ));
}

/// Pictures a Vulkan decoder made — images of FFmpeg's own, written on its
/// decode queue — are drawn once the decoder has written them: what comes
/// out is the picture it was handed, not the one before it or a half-made
/// one.
#[test]
fn decoded_pictures_are_composited() {
    let Some(device) = try_vulkan_device() else {
        return;
    };
    let Some(path) = crate::test_support::try_test_video() else {
        return;
    };
    let mut input = ffmpeg::format::input(&path).expect("open the test video");
    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .expect("a picture");
    let (index, params) = (stream.index(), stream.parameters());
    if !crate::elements::VulkanDecoder::supports(params.id()) {
        return;
    }
    let mut decoder = crate::elements::VulkanDecoder::new("decoder", params, &device, 4).unwrap();
    let decoded = capture(&mut decoder);
    for (stream, packet) in input.packets() {
        if stream.index() != index {
            continue;
        }
        match decoder.consume(MediaBuffer::Packet(Arc::new(packet))) {
            Err(crate::error::Error::VulkanDecoderError(
                crate::elements::VulkanDecoderError::HwAccelUnavailable,
            )) => return,
            result => result.unwrap(),
        }
        if decoded.lock().unwrap().len() >= 30 {
            break;
        }
    }
    let pictures: Vec<_> = decoded
        .lock()
        .unwrap()
        .drain(..)
        .filter_map(|buffer| match buffer {
            MediaBuffer::Video(frame) => Some(frame),
            _ => None,
        })
        .collect();
    let (earlier, picture) = (&pictures[0], &pictures[pictures.len() - 1]);
    let (width, height) = (picture.width(), picture.height());
    let read_back = |frame: &Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>| {
        let mut download = VulkanDownload::new("download", &device);
        let received = capture(&mut download);
        download.consume(MediaBuffer::Video(frame.clone())).unwrap();
        match received.lock().unwrap().remove(0) {
            MediaBuffer::Video(frame) => frame,
            _ => panic!("a picture"),
        }
    };
    let (reference, before) = (read_back(picture), read_back(earlier));

    let (mut compositor, handle) = nv12_compositor(&device, width, height);
    let mut layer = handle
        .add_source(
            "decoded",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, width, height))
            },
        )
        .unwrap();
    layer
        .sink
        .consume(MediaBuffer::Video(picture.clone()))
        .unwrap();
    let out = download(&device, compositor.compose_frame().unwrap());
    let mean = |other: &ffmpeg::frame::Video| {
        let total: u64 = (0..height as usize)
            .flat_map(|y| (0..width as usize).map(move |x| (x, y)))
            .map(|(x, y)| u64::from(luma_at(&out, x, y).abs_diff(luma_at(other, x, y))))
            .sum();
        total as f64 / f64::from(width * height)
    };
    let (own, other) = (mean(&reference), mean(&before));
    // Drawn one to one, the picture comes back as decoded but for what the
    // round trip changes: an 8-bit RGB canvas, and a BT.601 picture made
    // into BT.709 NV12, which moves the luma of anything coloured.
    assert!(own < 2.0, "{own} from the picture it was handed");
    assert!(
        own < other,
        "{own} from its own picture, {other} from an earlier one"
    );
}

/// The backend-independent traits drive this compositor as its own
/// handles do.
#[test]
fn the_backend_independent_traits_drive_it() {
    use crate::elements::VideoLayerControl;

    let Some(device) = try_vulkan_device() else {
        return;
    };
    let (mut compositor, handle) = nv12_compositor(&device, 64, 64);
    let still = super::super::super::control::arrange(&handle, 64, 64);
    let MediaBuffer::Video(frame) = nv12_frame(&device, 32, 32, 200) else {
        panic!("a frame");
    };
    VideoLayerControl::set_frame(&still, frame).unwrap();
    let out = download(&device, compositor.compose_frame().unwrap());
    near(
        luma_at(&out, 32, 32),
        200,
        "the picture set through the trait",
    );
}

/// Offline, each output frame shows what each input says at that frame's
/// time — two inputs in different time bases, one starting late — and the
/// render ends once both have, as the other compositors' do.
#[test]
fn an_offline_render_shows_what_each_input_says_at_each_output_time() {
    use crate::pipeline::Pipeline;
    use std::time::{Duration, Instant};

    let Some(device) = try_vulkan_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        frame_rate: ffmpeg::Rational::new(10, 1),
        mode: RenderMode::Offline { end: None },
        ..options(64, 64)
    };
    let (compositor, handle) =
        VulkanVideoCompositor::with_format("offline", &device, options, VulkanFrameFormat::Nv12)
            .unwrap();

    let timed = |luma: u8, pts: i64, base: ffmpeg::Rational, duration: i64| {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 64, 64);
        frame.data_mut(0).fill(luma);
        frame.data_mut(1).fill(128);
        frame.set_pts(Some(pts));
        crate::buffer::set_time_base(&mut frame, base);
        // SAFETY: a plain field of a frame this test owns outright.
        unsafe { (*frame.as_mut_ptr()).duration = duration };
        upload(&device, frame)
    };
    let feed = |name: &str, z_index: i32, frames: Vec<MediaBuffer>| {
        let layer = VideoLayer {
            z_index,
            fit: VideoFit::Stretch,
            ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
        };
        let sink = handle.add_source(name, layer).unwrap().sink;
        let (source, pusher) = crate::elements::AppSource::new(name, 64);
        let (pipeline, ()) = Pipeline::new(format!("{name}-feed"), source, |source, ctx| {
            let branch = ctx.branch().queue(format!("{name}-queue"), 2).to(sink)?;
            ctx.attach(source, 0, branch)?;
            Ok(())
        })
        .unwrap();
        pipeline.run().unwrap();
        for frame in frames {
            pusher.push(frame).unwrap();
        }
        pipeline
    };
    let ms = ffmpeg::Rational::new(1, 1000);
    let khz90 = ffmpeg::Rational::new(1, 90_000);
    let low: Vec<_> = [0, 250, 500, 750]
        .into_iter()
        .map(|start| timed(100, start, ms, 250))
        .collect();
    let high: Vec<_> = [45_000, 63_000]
        .into_iter()
        .map(|start| timed(200, start, khz90, 18_000))
        .collect();

    let received = Arc::new(Mutex::new(Vec::new()));
    let sink = CapturingSink {
        received: received.clone(),
        pp_log: element_pp_log(ElementType::Other, "capture", None),
    };
    let (render, ()) = Pipeline::new("render", compositor, |source, ctx| {
        let branch = ctx.branch().to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .unwrap();
    render.run().unwrap();
    // Each feed's pusher is dropped as the closure returns, which ends it.
    let _low = feed("low", 0, low);
    let _high = feed("high", 1, high);

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut finished = false;
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        if let Ok(BusEvent::Finished) = render.bus().recv_timeout(left) {
            finished = true;
            break;
        }
    }
    render.stop();
    assert!(finished, "the render ends once its inputs have");

    let composed: Vec<_> = received
        .lock()
        .unwrap()
        .iter()
        .filter_map(|buffer| match buffer {
            MediaBuffer::Video(frame) => Some(Arc::clone(frame)),
            _ => None,
        })
        .collect();
    let shown: Vec<_> = composed
        .into_iter()
        .map(|frame| {
            let pts = frame.pts().unwrap();
            let mut download = VulkanDownload::new("download", &device);
            let back = capture(&mut download);
            download.consume(MediaBuffer::Video(frame)).unwrap();
            let MediaBuffer::Video(frame) = back.lock().unwrap().remove(0) else {
                panic!("a picture");
            };
            let luma = luma_at(&frame, 0, 0);
            // Within the canvas's rounding of the two values used.
            (
                pts,
                if luma.abs_diff(200) <= 1 {
                    200
                } else if luma.abs_diff(100) <= 1 {
                    100
                } else {
                    luma
                },
            )
        })
        .collect();
    let expected: Vec<_> = (0..10)
        .map(|index| (index, if (5..9).contains(&index) { 200 } else { 100 }))
        .collect();
    assert_eq!(shown, expected);
}
