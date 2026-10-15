use std::sync::Mutex;

use crate::test_support::capture;

use super::super::super::text_layer::TextLayer;
use super::super::super::video_layer::{VideoFit, VideoSourceRect};
use super::*;
use crate::{
    bus::BusEvent,
    color::Color,
    elements::{D3d12Download, D3d12Upload},
    test_support::try_d3d12_gpu,
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

/// The compositor on `gpu`, or `None`, said, where this GPU cannot make one.
fn compositor(
    gpu: &D3d12Gpu,
    options: VideoCompositorOptions,
) -> Option<(D3d12VideoCompositor, D3d12VideoCompositorHandle)> {
    match D3d12VideoCompositor::new("compositor", gpu, options) {
        Ok(made) => Some(made),
        Err(error) => {
            eprintln!("skipping: this machine cannot open a D3D12 compositor ({error})");
            None
        }
    }
}

/// `frame` on `gpu`, or `None` where FFmpeg cannot take the device — as on
/// a WARP adapter.
fn upload(gpu: &D3d12Gpu, frame: ffmpeg::frame::Video) -> Option<MediaBuffer> {
    let Ok(mut upload) = D3d12Upload::new("upload", gpu) else {
        eprintln!("skipping: FFmpeg could not create a D3D12VA frames context");
        return None;
    };
    let uploaded = capture(&mut upload);
    upload.consume(MediaBuffer::video(frame)).expect("upload");
    Some(uploaded.lock().unwrap().remove(0))
}

/// An NV12 picture whose luma is `luma(x, y)` and whose colour is neutral,
/// so what is composed is checkable by luma alone, whatever matrix reads it.
fn picture(width: u32, height: u32, luma: impl Fn(u32, u32) -> u8) -> ffmpeg::frame::Video {
    let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, width, height);
    frame.set_pts(Some(0));
    let stride = frame.stride(0);
    for y in 0..height {
        for x in 0..width {
            frame.data_mut(0)[y as usize * stride + x as usize] = luma(x, y);
        }
    }
    let uv_stride = frame.stride(1);
    frame.data_mut(1)[..uv_stride * (height / 2) as usize].fill(128);
    frame
}

/// A flat NV12 picture of `luma` on `gpu`.
fn flat(gpu: &D3d12Gpu, width: u32, height: u32, luma: u8) -> Option<MediaBuffer> {
    upload(gpu, picture(width, height, |_, _| luma))
}

/// One composed frame brought back to system memory.
fn download(
    frame: UnboundObjectPoolRef<ffmpeg::frame::Video>,
) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
    let mut download = D3d12Download::new("download");
    let received = capture(&mut download);
    download
        .consume(MediaBuffer::Video(Arc::new(frame).into()))
        .expect("download");
    let MediaBuffer::Video(frame) = received.lock().unwrap().remove(0) else {
        panic!("expected a video frame");
    };
    frame.into_payload()
}

fn luma_at(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> u8 {
    frame.data(0)[y * frame.stride(0) + x]
}

fn chroma_at(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> (u8, u8) {
    let at = (y / 2) * frame.stride(1) + (x / 2) * 2;
    (frame.data(1)[at], frame.data(1)[at + 1])
}

/// Composed at once and brought back.
fn composed(
    compositor: &mut D3d12VideoCompositor,
) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
    download(compositor.compositing().compose_frame().expect("compose"))
}

/// The composition contract: layers land where their rectangles say,
/// scaled to them, the higher `z_index` wins where they overlap, and
/// everything else is the background — luma 16 for black.
#[test]
fn composes_layers_in_z_order_at_their_rectangles() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(128, 128)) else {
        return;
    };
    let stretched = |z_index, rect| VideoLayer {
        z_index,
        fit: VideoFit::Stretch,
        ..VideoLayer::new(rect)
    };
    let mut back = handle
        .add_source("back", stretched(0, VideoRect::new(0, 0, 64, 64)))
        .expect("add back");
    let mut front = handle
        .add_source("front", stretched(1, VideoRect::new(32, 32, 64, 64)))
        .expect("add front");
    let (Some(back_frame), Some(front_frame)) = (flat(&gpu, 32, 32, 100), flat(&gpu, 32, 32, 200))
    else {
        return;
    };
    back.sink.consume(back_frame).expect("back frame");
    front.sink.consume(front_frame).expect("front frame");

    let frame = compositor.compositing().compose_frame().expect("compose");
    assert_eq!(frame.format(), ffmpeg::format::Pixel::D3D12);
    assert_eq!(frame.pts(), Some(0));
    let out = download(frame);
    assert_eq!(luma_at(&out, 10, 10), 100, "the back layer is missing");
    assert_eq!(luma_at(&out, 80, 80), 200, "the front layer is missing");
    assert_eq!(luma_at(&out, 40, 40), 200, "the higher z_index wins");
    assert_eq!(luma_at(&out, 120, 10), 16, "outside a layer is background");
    assert_eq!(chroma_at(&out, 80, 80), (128, 128), "grey stays grey");
}

/// Cropping on the GPU: the layer draws its source region, scaled to its
/// rectangle, and nothing of the rest — at an odd offset, which a sampler
/// minds no more than any other.
#[test]
fn a_layer_draws_only_its_source_region() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(128, 128)) else {
        return;
    };
    let mut input = handle
        .add_source(
            "layer",
            VideoLayer {
                fit: VideoFit::Stretch,
                source: Some(VideoSourceRect::new(131, 128, 125, 128)),
                ..VideoLayer::new(VideoRect::new(0, 0, 128, 128))
            },
        )
        .expect("add source");
    let quadrants = picture(256, 256, |x, y| match (x >= 128, y >= 128) {
        (false, false) => 40,
        (true, false) => 90,
        (false, true) => 150,
        (true, true) => 220,
    });
    let Some(frame) = upload(&gpu, quadrants) else {
        return;
    };
    input.sink.consume(frame).expect("frame");
    let out = composed(&mut compositor);
    for (x, y) in [(0, 0), (4, 4), (64, 64), (127, 127)] {
        assert_eq!(
            luma_at(&out, x, y),
            220,
            "every pixel comes from the cropped quadrant, at ({x},{y})"
        );
    }
}

/// Dragging a crop handle changes the region on every frame; every one of
/// them composes, and the layer never goes blank.
#[test]
fn a_region_that_changes_every_frame_keeps_composing() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(256, 256)) else {
        return;
    };
    let mut input = handle
        .add_source(
            "layer",
            VideoLayer {
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 256, 256))
            },
        )
        .expect("add source");
    let Some(frame) = flat(&gpu, 640, 480, 200) else {
        return;
    };
    input.sink.consume(frame).expect("frame");
    for step in 0..60u32 {
        let cut = step * 3;
        input
            .layer
            .set_source(Some(VideoSourceRect::new(cut, 0, 640 - cut, 480)))
            .expect("set source");
        let out = composed(&mut compositor);
        let luma = luma_at(&out, 128, 128);
        assert_eq!(luma, 200, "the layer went blank at step {step}");
    }
}

/// `Cover` fills its rectangle, cropping what overflows it, where `Contain`
/// leaves background in the same rectangle.
#[test]
fn cover_fills_its_rectangle_where_contain_letterboxes() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let composed_with = |fit: VideoFit| {
        let (mut compositor, handle) = compositor(&gpu, options(128, 128))?;
        let mut input = handle
            .add_source(
                "layer",
                VideoLayer {
                    fit,
                    ..VideoLayer::new(VideoRect::new(0, 0, 64, 64))
                },
            )
            .expect("add");
        input.sink.consume(flat(&gpu, 64, 16, 200)?).expect("frame");
        Some(composed(&mut compositor))
    };
    let (Some(contain), Some(cover)) = (
        composed_with(VideoFit::Contain),
        composed_with(VideoFit::Cover),
    ) else {
        return;
    };
    assert_eq!(luma_at(&contain, 32, 32), 200, "Contain draws the middle");
    assert_eq!(luma_at(&contain, 32, 4), 16, "and letterboxes above it");
    assert_eq!(luma_at(&cover, 32, 32), 200, "Cover draws the middle");
    assert_eq!(luma_at(&cover, 32, 4), 200, "and fills to the edge");
    assert_eq!(
        luma_at(&cover, 32, 80),
        16,
        "and nothing outside its rectangle"
    );
}

/// A translucent layer is mixed with what is under it, as the CUDA
/// compositor mixes it; opaque replaces, and transparent draws nothing.
#[test]
fn a_translucent_layer_is_blended_with_the_background() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(128, 128)) else {
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
    let Some(frame) = flat(&gpu, 32, 32, 200) else {
        return;
    };
    input.sink.consume(frame).expect("frame");
    let out = composed(&mut compositor);
    let mixed = luma_at(&out, 10, 10);
    assert!(mixed.abs_diff(108) <= 1, "half of 200 over 16, got {mixed}");
    assert_eq!(luma_at(&out, 100, 100), 16, "outside stays background");

    input.layer.set_opacity(1.0).expect("opaque");
    assert_eq!(luma_at(&composed(&mut compositor), 10, 10), 200);
    input.layer.set_opacity(0.0).expect("transparent");
    assert_eq!(luma_at(&composed(&mut compositor), 10, 10), 16);
}

/// Each picture is read in its own colour: a BT.601 one and a BT.709 one of
/// the same red each come out the canvas's BT.709 red, and the background
/// is filled in it — said so on the frame.
#[test]
fn every_picture_is_read_in_its_own_colour_and_the_canvas_says_its_own() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(
        &gpu,
        VideoCompositorOptions {
            background: Color::new(230, 20, 20),
            ..options(128, 64)
        },
    ) else {
        return;
    };
    // Red, (200, 30, 30), as each matrix writes it in limited range.
    let red = |space: ffmpeg::color::Space| {
        let (kr, kb) = match space {
            ffmpeg::color::Space::BT709 => (0.2126f32, 0.0722f32),
            _ => (0.299, 0.114),
        };
        let [r, g, b] = [200.0f32 / 255.0, 30.0 / 255.0, 30.0 / 255.0];
        let y = kr * r + (1.0 - kr - kb) * g + kb * b;
        let cb = (b - y) / (2.0 * (1.0 - kb));
        let cr = (r - y) / (2.0 * (1.0 - kr));
        let mut frame = picture(64, 64, |_, _| (16.0 + 219.0 * y).round() as u8);
        let uv_stride = frame.stride(1);
        for row in 0..32 {
            for column in 0..32 {
                let at = row * uv_stride + column * 2;
                frame.data_mut(1)[at] = (128.0 + 224.0 * cb).round() as u8;
                frame.data_mut(1)[at + 1] = (128.0 + 224.0 * cr).round() as u8;
            }
        }
        frame.set_color_space(space);
        frame.set_color_range(ffmpeg::color::Range::MPEG);
        frame
    };
    for (name, x, space) in [
        ("bt601", 0, ffmpeg::color::Space::BT470BG),
        ("bt709", 64, ffmpeg::color::Space::BT709),
    ] {
        let mut input = handle
            .add_source(
                name,
                VideoLayer {
                    fit: VideoFit::Stretch,
                    ..VideoLayer::new(VideoRect::new(x, 0, 64, 64))
                },
            )
            .expect("add");
        let Some(frame) = upload(&gpu, red(space)) else {
            return;
        };
        input.sink.consume(frame).expect("frame");
    }
    let frame = compositor.compositing().compose_frame().expect("compose");
    assert_eq!(frame.color_space(), ffmpeg::color::Space::BT709);
    assert_eq!(frame.color_range(), ffmpeg::color::Range::MPEG);
    let out = download(frame);
    let bt709 = red(ffmpeg::color::Space::BT709);
    let want = (luma_at(&bt709, 0, 0), chroma_at(&bt709, 0, 0));
    for x in [16, 96] {
        let got = (luma_at(&out, x, 32), chroma_at(&out, x, 32));
        assert!(
            got.0.abs_diff(want.0) <= 1
                && got.1.0.abs_diff(want.1.0) <= 1
                && got.1.1.abs_diff(want.1.1) <= 1,
            "at x={x}: {got:?} against BT.709's {want:?}"
        );
    }
}

/// A tick that finds nothing changed hands out the frame it composed last,
/// under its own timestamp; a layer that moves is composed anew.
#[test]
fn an_unchanged_scene_is_composed_once() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(128, 128)) else {
        return;
    };
    let mut input = handle
        .add_source("only", VideoLayer::new(VideoRect::new(0, 0, 64, 64)))
        .expect("add source");
    let Some(frame) = flat(&gpu, 32, 32, 100) else {
        return;
    };
    input.sink.consume(frame).expect("frame");
    let first = compositor.compositing().compose_frame().expect("compose");
    let surface = picture_id(&first);
    let repeated = compositor.compositing().compose_frame().expect("repeat");
    assert_eq!(picture_id(&repeated), surface, "the picture already made");
    assert_eq!(repeated.pts(), Some(1), "under this tick's timestamp");
    input
        .layer
        .set_layer(VideoLayer::new(VideoRect::new(16, 16, 64, 64)))
        .expect("move the layer");
    let moved = compositor.compositing().compose_frame().expect("compose");
    assert_ne!(picture_id(&moved), surface, "a moved layer is composed");
}

/// Moving and hiding a layer changes the next frame.
#[test]
fn layer_handle_moves_and_hides_a_live_source() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(128, 128)) else {
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
    let Some(frame) = flat(&gpu, 32, 32, 180) else {
        return;
    };
    input.sink.consume(frame).expect("frame");
    let out = composed(&mut compositor);
    assert_eq!((luma_at(&out, 10, 10), luma_at(&out, 74, 74)), (180, 16));
    input
        .layer
        .set_rect(VideoRect::new(64, 64, 32, 32))
        .expect("move");
    let out = composed(&mut compositor);
    assert_eq!((luma_at(&out, 10, 10), luma_at(&out, 74, 74)), (16, 180));
    input.layer.set_visible(false).expect("hide");
    assert_eq!(luma_at(&composed(&mut compositor), 74, 74), 16);
}

/// A font every machine running these tests has, or a skip, said.
fn try_font() -> Option<Vec<u8>> {
    let font = std::fs::read("C:/Windows/Fonts/arial.ttf").ok();
    if font.is_none() {
        eprintln!("skipping: no font on this machine to rasterize with");
    }
    font
}

/// Nothing is drawn until `set_text`, then the glyphs land on the canvas;
/// position, opacity and visibility act on the next frame, and text with
/// nothing to draw clears the layer.
#[test]
fn a_text_layer_draws_where_and_as_it_is_told() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some(font) = try_font() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(256, 128)) else {
        return;
    };
    let mut layer = TextLayer::new(font);
    layer.font_size = 48.0;
    layer.color = Color::WHITE;
    let text = handle.add_text_layer("clock", layer).expect("add text");
    let brightest = |frame: &ffmpeg::frame::Video, x0: usize, x1: usize| {
        (0..64usize)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .map(|(x, y)| luma_at(frame, x, y))
            .max()
            .unwrap_or(0)
    };
    assert_eq!(
        brightest(&composed(&mut compositor), 0, 256),
        16,
        "no text yet"
    );

    text.set_text("IIII").expect("set_text");
    let out = composed(&mut compositor);
    assert!(brightest(&out, 0, 100) > 200, "the text is at the origin");
    assert_eq!(brightest(&out, 150, 250), 16, "and not on the right");
    assert_eq!(
        chroma_at(&out, 4, 30),
        (128, 128),
        "white is grey in chroma"
    );

    text.set_position(150, 0);
    let out = composed(&mut compositor);
    assert_eq!(brightest(&out, 0, 100), 16, "the text left");
    assert!(brightest(&out, 150, 250) > 200, "the text arrived");

    text.set_opacity(0.5).expect("half opacity");
    let half = brightest(&composed(&mut compositor), 150, 250);
    assert!(
        (100..=170).contains(&half),
        "half-opacity text is mid-grey, got {half}"
    );

    text.set_visible(false);
    assert_eq!(
        brightest(&composed(&mut compositor), 150, 250),
        16,
        "hidden"
    );
    text.set_visible(true);
    text.set_text("   ").expect("blank text");
    assert_eq!(brightest(&composed(&mut compositor), 0, 256), 16, "cleared");
}

/// A text layer stacks among the video layers by its `z_index`.
#[test]
fn a_text_layer_stacks_among_video_layers_by_its_z_index() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some(font) = try_font() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(256, 128)) else {
        return;
    };
    let mut cover = handle
        .add_source(
            "cover",
            VideoLayer {
                z_index: 1,
                fit: VideoFit::Stretch,
                ..VideoLayer::new(VideoRect::new(0, 0, 256, 128))
            },
        )
        .expect("add the cover")
        .sink;
    let Some(frame) = flat(&gpu, 256, 128, 60) else {
        return;
    };
    cover.consume(frame).expect("cover frame");
    let mut layer = TextLayer::new(font);
    layer.font_size = 48.0;
    layer.color = Color::WHITE;
    layer.x = 8;
    layer.y = 8;
    let text = handle.add_text_layer("title", layer).expect("add text");
    text.set_text("HELLO").expect("set_text");
    let lit = |compositor: &mut D3d12VideoCompositor| {
        let out = composed(compositor);
        (0..64usize)
            .flat_map(|y| (0..200usize).map(move |x| (x, y)))
            .filter(|(x, y)| luma_at(&out, *x, *y) > 120)
            .count()
    };
    assert_eq!(lit(&mut compositor), 0, "under the cover, nothing shows");
    text.set_z_index(2);
    assert!(lit(&mut compositor) > 100, "raised over it, the text shows");
}

/// A picture in system memory, or of another device, is refused by the
/// sink that would take it, by name.
#[test]
fn a_picture_in_memory_or_of_another_device_is_refused() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((_compositor, handle)) = compositor(&gpu, options(64, 64)) else {
        return;
    };
    let mut input = handle
        .add_source("layer", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
        .expect("add");
    let error = input
        .sink
        .consume(MediaBuffer::video(picture(32, 32, |_, _| 100)))
        .expect_err("a picture in system memory");
    assert!(
        matches!(
            error,
            crate::error::Error::D3d12VideoCompositorError(
                D3d12VideoCompositorError::UnsupportedPicture(_)
            )
        ),
        "{error}"
    );
    assert!(input.layer.latest_frame().is_none(), "it was not taken");
    // D3D12 hands out one device an adapter in a process: another device is
    // another adapter, which this machine may not have.
    let Some(other) = try_d3d12_gpu() else {
        return;
    };
    if other.device().as_raw() == gpu.device().as_raw() {
        eprintln!("skipping the foreign half: one adapter, so one device");
        return;
    }
    let Some(foreign) = flat(&other, 32, 32, 100) else {
        return;
    };
    let error = input.sink.consume(foreign).expect_err("another device's");
    assert!(
        matches!(
            error,
            crate::error::Error::D3d12VideoCompositorError(
                D3d12VideoCompositorError::DeviceMismatch
            )
        ),
        "{error}"
    );
    assert!(input.layer.latest_frame().is_none(), "neither was taken");
}

/// Output timing is the compositor's own: contiguous ticks of its time
/// base, whatever the inputs carried.
#[test]
fn output_pts_are_contiguous_ticks_of_its_own_time_base() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(64, 64)) else {
        return;
    };
    assert_eq!(compositor.time_base(), ffmpeg::Rational::new(1, 30));
    let mut input = handle
        .add_source("layer", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
        .expect("add");
    let mut late = picture(32, 32, |_, _| 100);
    late.set_pts(Some(9_999));
    let Some(frame) = upload(&gpu, late) else {
        return;
    };
    input.sink.consume(frame).expect("frame");
    for expected in 0..3 {
        let frame = compositor.compositing().compose_frame().expect("compose");
        assert_eq!(frame.pts(), Some(expected));
    }
}

/// NV12 has no alpha: a background that is not opaque is refused, and odd
/// dimensions are too. No device: this is checked before one is touched.
#[test]
fn what_nv12_cannot_be_is_refused() {
    assert!(matches!(
        validate_output_options(VideoCompositorOptions {
            background_alpha: 128,
            ..options(64, 64)
        }),
        Err(D3d12VideoCompositorError::TranslucentBackground(128))
    ));
    assert!(matches!(
        validate_output_options(options(63, 64)),
        Err(D3d12VideoCompositorError::InvalidOutputDimensions { .. })
    ));
    assert!(validate_output_options(options(64, 64)).is_ok());
}

/// An offline render places each input's pictures by their timestamps —
/// the CPU compositor's contract, on the GPU: a luma-100 input over the
/// first second in quarter-second frames counted in milliseconds, a
/// luma-200 one over it from 0.5 s to 0.9 s counted at 90 kHz, at ten
/// frames a second; and the render ends by itself once both inputs have.
#[test]
fn an_offline_render_shows_what_each_input_says_at_each_output_time() {
    use crate::pipeline::Pipeline;

    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let options = VideoCompositorOptions {
        frame_rate: ffmpeg::Rational::new(10, 1),
        mode: RenderMode::Offline { end: None },
        ..options(64, 64)
    };
    let Some((compositor, handle)) = compositor(&gpu, options) else {
        return;
    };
    let timed = |luma: u8, pts: i64, base: ffmpeg::Rational, duration: i64| {
        let mut frame = picture(64, 64, |_, _| luma);
        frame.set_pts(Some(pts));
        crate::buffer::set_time_base(&mut frame, base);
        // SAFETY: a plain field of a frame this test owns outright.
        unsafe { (*frame.as_mut_ptr()).duration = duration };
        upload(&gpu, frame)
    };
    let ms = ffmpeg::Rational::new(1, 1000);
    let khz90 = ffmpeg::Rational::new(1, 90_000);
    let Some(low) = [0, 250, 500, 750]
        .into_iter()
        .map(|start| timed(100, start, ms, 250))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let Some(high) = [45_000, 63_000]
        .into_iter()
        .map(|start| timed(200, start, khz90, 18_000))
        .collect::<Option<Vec<_>>>()
    else {
        return;
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

    let received = Arc::new(Mutex::new(Vec::new()));
    let sink = crate::test_support::CapturingSink {
        received: received.clone(),
        pp_log: element_pp_log(ElementType::Other, "capture", None),
    };
    let (render, ()) = Pipeline::new("render", compositor, |source, ctx| {
        let branch = ctx.branch().to(sink)?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .unwrap();
    let _low = feed("low", 0, low);
    let _high = feed("high", 1, high);
    render.run().unwrap();

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

    let shown: Vec<_> = received
        .lock()
        .unwrap()
        .iter()
        .filter_map(|buffer| match buffer {
            MediaBuffer::Video(frame) => Some(Arc::clone(frame)),
            _ => None,
        })
        .map(|frame| {
            let pts = frame.pts().unwrap();
            let mut download = D3d12Download::new("download");
            let received = capture(&mut download);
            download
                .consume(MediaBuffer::Video(frame.into()))
                .expect("download");
            let MediaBuffer::Video(frame) = received.lock().unwrap().remove(0) else {
                panic!("expected a video frame");
            };
            (pts, luma_at(&frame, 0, 0))
        })
        .collect();
    let expected: Vec<_> = (0..10)
        .map(|index| (index, if (5..9).contains(&index) { 200 } else { 100 }))
        .collect();
    assert_eq!(shown, expected);
}

/// The backend-independent traits drive this compositor as its own
/// handles do, and a layer added with `add_layer` is drawn from the frame
/// set on it — which has to be of the compositor's device.
#[test]
fn the_backend_independent_traits_drive_it() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(64, 64)) else {
        return;
    };
    let still = crate::elements::source::compositor::control::arrange(&handle, 64, 64);
    let MediaBuffer::Video(in_memory) = MediaBuffer::video(picture(64, 64, |_, _| 0)) else {
        unreachable!()
    };
    assert!(
        crate::elements::VideoLayerControl::set_frame(&still, in_memory.into_payload()).is_err(),
        "a frame in system memory is refused"
    );
    let Some(MediaBuffer::Video(on_gpu)) = flat(&gpu, 64, 64, 200) else {
        return;
    };
    crate::elements::VideoLayerControl::set_frame(&still, on_gpu.into_payload()).unwrap();
    assert_eq!(luma_at(&composed(&mut compositor), 10, 10), 200);
}

#[test]
fn an_input_that_ends_is_shown_again() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((_compositor, handle)) = compositor(&gpu, options(64, 64)) else {
        return;
    };
    if flat(&gpu, 64, 64, 200).is_none() {
        return;
    }
    crate::elements::source::compositor::control::an_input_that_ends_is_shown_again(
        &handle,
        || flat(&gpu, 64, 64, 200).expect("a frame, as a moment ago"),
    );
}

/// A layer whose picture cannot be drawn is said on the bus under its own
/// name, and the rest of the frame is drawn — checked by hand, since only a
/// picture the sink was not shown can fail there: one set through the
/// handle of a layer that then had its picture replaced is refused at the
/// handle instead.
#[test]
fn a_layer_that_cannot_be_drawn_leaves_the_rest_drawn() {
    let Some(gpu) = try_d3d12_gpu() else {
        return;
    };
    let Some((mut compositor, handle)) = compositor(&gpu, options(64, 64)) else {
        return;
    };
    let (bus, events) = crate::bus::Bus::new();
    compositor.compositing().bus = Some(bus);
    let good = handle
        .add_layer("good", VideoLayer::new(VideoRect::new(0, 0, 32, 32)))
        .expect("add");
    let bad = handle
        .add_layer("bad", VideoLayer::new(VideoRect::new(32, 32, 32, 32)))
        .expect("add");
    let (Some(MediaBuffer::Video(picture_good)), Some(MediaBuffer::Video(picture_bad))) =
        (flat(&gpu, 32, 32, 200), flat(&gpu, 32, 32, 100))
    else {
        return;
    };
    good.set_frame(picture_good.into_payload()).expect("set");
    // Past the handle's check, as a picture of a removed device would be.
    let in_memory = MediaBuffer::video(picture(32, 32, |_, _| 0));
    let MediaBuffer::Video(in_memory) = in_memory else {
        unreachable!()
    };
    drop(picture_bad);
    bad.input
        .upgrade()
        .expect("registered")
        .latest_frame
        .store(Some(in_memory.into_payload()));
    let out = composed(&mut compositor);
    assert_eq!(luma_at(&out, 10, 10), 200, "the good layer is drawn");
    assert_eq!(luma_at(&out, 48, 48), 16, "the bad one is not");
    let said = events.try_recv().expect("said on the bus");
    assert!(
        matches!(&said, BusEvent::Error { name, .. } if &**name == "bad"),
        "{said:?}"
    );
}
