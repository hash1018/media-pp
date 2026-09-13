use std::collections::HashSet;

use super::*;
use crate::test_support::try_d3d11_device as try_device;
use crate::{
    color::Color,
    elements::{D3d11Download, VideoRect},
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
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        self.received.lock().unwrap().push(buf);
        Ok(())
    }

    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

fn bgra_texture(device: &ID3D11Device, width: u32, height: u32, bgra: [u8; 4]) -> ID3D11Texture2D {
    let pixels: Vec<u8> = (0..width * height).flat_map(|_| bgra).collect();
    bgra_texture_from_pixels(device, width, height, &pixels)
}

fn bgra_texture_from_pixels(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> ID3D11Texture2D {
    assert_eq!(pixels.len(), (width * height * 4) as usize);
    // SAFETY: `pixels` is live and exactly matches the BGRA description's
    // dimensions and pitch; the output interface slot is live.
    unsafe {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let initial = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr().cast::<c_void>(),
            SysMemPitch: width * 4,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        device
            .CreateTexture2D(&desc, Some(&initial), Some(&mut texture))
            .expect("CreateTexture2D failed");
        texture.expect("CreateTexture2D succeeded without producing a texture")
    }
}

fn nv12_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    y: u8,
    cb: u8,
    cr: u8,
) -> ID3D11Texture2D {
    let row_bytes = width as usize;
    let luma_size = row_bytes * height as usize;
    let mut pixels = vec![y; luma_size + row_bytes * height.div_ceil(2) as usize];
    for pair in pixels[luma_size..].as_chunks_mut::<2>().0 {
        *pair = [cb, cr];
    }
    // SAFETY: the contiguous NV12 buffer is live and sized for the padded
    // row pitch and both planes described below; the output slot is live.
    unsafe {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_NV12,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let initial = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr().cast::<c_void>(),
            SysMemPitch: width,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        device
            .CreateTexture2D(&desc, Some(&initial), Some(&mut texture))
            .expect("CreateTexture2D(NV12) failed");
        texture.expect("CreateTexture2D succeeded without producing a texture")
    }
}

fn texture_key(frame: &ffmpeg::frame::Video) -> usize {
    d3d11va_texture(frame).expect("expected a D3D11 frame").0 as usize
}

fn apply_color_rows(rows: [[f32; 4]; 3], y: f32, cb: f32, cr: f32) -> [f32; 3] {
    rows.map(|row| row[0] * y + row[1] * cb + row[2] * cr + row[3])
}

fn pooled_video(frame: ffmpeg::frame::Video) -> MediaBuffer {
    let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
    let mut pooled = pool.get();
    *pooled = frame;
    MediaBuffer::Video(Arc::new(pooled))
}

/// A `Bus` with its receiver immediately dropped, for tests that don't
/// care about per-layer error reporting — `Bus::post` no-ops once the
/// receiving end is gone.
fn test_bus() -> Bus {
    Bus::new().0
}

fn download_frame(
    device: &ID3D11Device,
    context: Arc<Mutex<ID3D11DeviceContext>>,
    composed: Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>>,
) -> Arc<UnboundObjectPoolRef<ffmpeg::frame::Video>> {
    let (width, height) = (composed.width(), composed.height());
    let mut download = D3d11Download::new("download", device, context, width, height)
        .expect("D3d11Download::new should succeed");
    let received = Arc::new(Mutex::new(Vec::new()));
    download.src_pads()[0].link(Box::new(CapturingSink {
        received: received.clone(),
        pp_log: element_pp_log(ElementType::Other, "capture", None),
    }));
    download
        .consume(MediaBuffer::Video(composed))
        .expect("download consume should succeed");
    let mut received = received.lock().unwrap();
    let MediaBuffer::Video(frame) = received.remove(0) else {
        panic!("expected a Video buffer");
    };
    frame
}

fn pixel(frame: &ffmpeg::frame::Video, x: usize, y: usize) -> [u8; 4] {
    let offset = y * frame.stride(0) + x * 4;
    frame.data(0)[offset..offset + 4].try_into().unwrap()
}

#[test]
fn invalid_text_layer_does_not_replace_an_existing_registration() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 4,
        height: 4,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (_compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context, options).unwrap();
    let existing = handle
        .add_layer("overlay", VideoLayer::new(VideoRect::new(0, 0, 1, 1)))
        .unwrap()
        .unwrap();

    let result = handle.add_text_layer("overlay", TextLayer::new(vec![0, 1, 2, 3]));

    assert!(matches!(result, Err(D3d11TextLayerError::InvalidFont(_))));
    assert_eq!(handle.source_count(), 1);
    assert!(existing.layer().is_some());
}

#[test]
fn composes_gpu_inputs_in_z_order_and_preserves_output_contract() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 4,
        height: 4,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");

    let mut background_layer = VideoLayer::new(VideoRect::new(0, 0, 4, 4));
    background_layer.fit = video_layer::VideoFit::Stretch;
    let mut red_sink = handle
        .add_source("red", background_layer)
        .unwrap()
        .unwrap()
        .sink;

    let mut overlay_layer = VideoLayer::new(VideoRect::new(1, 1, 2, 2));
    overlay_layer.z_index = 1;
    overlay_layer.fit = video_layer::VideoFit::Stretch;
    let mut blue_sink = handle
        .add_source("blue", overlay_layer)
        .unwrap()
        .unwrap()
        .sink;

    // BGRA byte order: [blue, green, red, alpha].
    let red_texture = bgra_texture(&device, 4, 4, [0, 0, 255, 255]);
    let blue_texture = bgra_texture(&device, 2, 2, [255, 0, 0, 255]);
    red_sink
        .consume(pooled_video(wrap_d3d11_texture(red_texture, 4, 4).unwrap()))
        .unwrap();
    blue_sink
        .consume(pooled_video(
            wrap_d3d11_texture(blue_texture, 2, 2).unwrap(),
        ))
        .unwrap();

    let composed = compositor
        .compose_frame(&test_bus())
        .expect("compose_frame failed");
    assert_eq!(composed.format(), ffmpeg::format::Pixel::D3D11);
    assert_eq!((composed.width(), composed.height()), (4, 4));
    assert_eq!(composed.pts(), Some(0));

    let downloaded = download_frame(&device, context, composed);
    assert_eq!(pixel(&downloaded, 0, 0), [0, 0, 255, 255], "red background");
    assert_eq!(pixel(&downloaded, 1, 1), [255, 0, 0, 255], "blue overlay");
}

/// The crop, per pixel, on the hardware that draws it.
///
/// `the_sampled_window_covers_the_frame_or_the_region_inside_it` checks
/// the arithmetic that produces `uv_offset`/`uv_scale` and needs no
/// device; what it cannot check is that the shader then samples where
/// those say. Software and CUDA each have this test; Direct3D did not,
/// because the machine the crop was written on could not run it.
///
/// Every texel carries its own coordinates, and the region is taken at
/// an odd offset scaled one-to-one, so the assertion is exact: output
/// (x, y) must be input (3 + x, 1 + y) and nothing else. A half-texel
/// error, an offset applied in the wrong direction, or a region measured
/// from the wrong corner all move at least one of them.
#[test]
fn a_layer_draws_only_its_source_region() {
    let Some((device, context)) = try_device() else {
        return;
    };
    const SOURCE: u32 = 8;
    const REGION: u32 = 4;
    const AT_X: u32 = 3;
    const AT_Y: u32 = 1;

    // BGRA, and blue/green carry the texel's own coordinates.
    let texel = |x: u32, y: u32| [(x * 16) as u8, (y * 16) as u8, 128, 255];
    let pixels: Vec<u8> = (0..SOURCE)
        .flat_map(|y| (0..SOURCE).flat_map(move |x| texel(x, y)))
        .collect();

    let options = VideoCompositorOptions {
        width: REGION,
        height: REGION,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");

    let mut layer = VideoLayer::new(VideoRect::new(0, 0, REGION, REGION));
    layer.fit = video_layer::VideoFit::Stretch;
    layer.source = Some(video_layer::VideoSourceRect::new(
        AT_X, AT_Y, REGION, REGION,
    ));
    let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;

    let texture = bgra_texture_from_pixels(&device, SOURCE, SOURCE, &pixels);
    sink.consume(pooled_video(
        wrap_d3d11_texture(texture, SOURCE, SOURCE).unwrap(),
    ))
    .unwrap();

    let composed = compositor
        .compose_frame(&test_bus())
        .expect("compose_frame failed");
    let downloaded = download_frame(&device, context, composed);

    for y in 0..REGION {
        for x in 0..REGION {
            assert_eq!(
                pixel(&downloaded, x as usize, y as usize),
                texel(AT_X + x, AT_Y + y),
                "output ({x}, {y}) must be input ({}, {})",
                AT_X + x,
                AT_Y + y
            );
        }
    }
}

#[test]
fn ignores_rows_outside_the_frame_visible_dimensions() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 4,
        height: 3,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");
    let mut layer = VideoLayer::new(VideoRect::new(0, 0, 4, 3));
    layer.fit = video_layer::VideoFit::Stretch;
    let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;

    // The frame exposes only the top three red rows of a four-row
    // texture. Sampling the full texture would blend the blue padding
    // row into the last visible output row.
    let mut pixels = Vec::with_capacity(4 * 4 * 4);
    for y in 0..4 {
        let color = if y < 3 {
            [0, 0, 255, 255]
        } else {
            [255, 0, 0, 255]
        };
        pixels.extend((0..4).flat_map(|_| color));
    }
    let texture = bgra_texture_from_pixels(&device, 4, 4, &pixels);
    sink.consume(pooled_video(wrap_d3d11_texture(texture, 4, 3).unwrap()))
        .unwrap();

    let composed = compositor
        .compose_frame(&test_bus())
        .expect("compose_frame failed");
    let downloaded = download_frame(&device, context, composed);
    for y in 0..3 {
        for x in 0..4 {
            assert_eq!(pixel(&downloaded, x, y), [0, 0, 255, 255]);
        }
    }
}

/// The window a layer samples, with and without a crop: an uncropped
/// layer takes the frame out of its (possibly taller) texture from the
/// origin, and a cropped one takes its region out of the same texture.
#[test]
fn the_sampled_window_covers_the_frame_or_the_region_inside_it() {
    let whole = visible_uv_window(
        VideoSourceRect::new(0, 0, 1920, 1080),
        1920,
        1080,
        1920,
        1088,
    )
    .expect("a frame inside its texture");
    assert_eq!(whole.offset, [0.0, 0.0]);
    assert!((whole.scale[1] - 1080.0 / 1088.0).abs() < 1e-6);

    let quadrant = visible_uv_window(
        VideoSourceRect::new(960, 540, 960, 540),
        1920,
        1080,
        1920,
        1088,
    )
    .expect("a region inside the frame");
    assert!((quadrant.offset[0] - 0.5).abs() < 1e-6);
    assert!((quadrant.scale[0] - 0.5).abs() < 1e-6);
    assert!(
        (quadrant.offset[1] - 540.0 / 1088.0).abs() < 1e-6,
        "the offset is measured against the texture, not the frame"
    );
}

#[test]
fn rejects_frame_dimensions_larger_than_the_backing_texture() {
    let error = visible_uv_window(
        VideoSourceRect::new(0, 0, 1920, 1088),
        1920,
        1088,
        1920,
        1080,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        D3d11VideoCompositorError::FrameExceedsTexture {
            frame_width: 1920,
            frame_height: 1088,
            texture_width: 1920,
            texture_height: 1080,
        }
    ));
}

#[test]
fn live_output_frames_keep_distinct_textures_until_the_last_arc_drops() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 1,
        height: 1,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");
    let mut layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
    layer.fit = video_layer::VideoFit::Stretch;
    let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;

    sink.consume(pooled_video(
        wrap_d3d11_texture(bgra_texture(&device, 1, 1, [0, 0, 255, 255]), 1, 1).unwrap(),
    ))
    .unwrap();
    let first = compositor
        .compose_frame(&test_bus())
        .expect("first compose failed");

    // A new input texture before each one: this is about frames that were
    // really composed, and a tick whose input has not changed hands out
    // the picture already composed rather than drawing a new one (see
    // `an_unchanged_scene_is_composed_once`).
    let mut later = Vec::new();
    for _ in 0..OUTPUT_POOL_SIZE {
        sink.consume(pooled_video(
            wrap_d3d11_texture(bgra_texture(&device, 1, 1, [255, 0, 0, 255]), 1, 1).unwrap(),
        ))
        .unwrap();
        later.push(
            compositor
                .compose_frame(&test_bus())
                .expect("later compose failed"),
        );
    }

    let mut keys = HashSet::new();
    keys.insert(texture_key(&first));
    keys.extend(later.iter().map(|frame| texture_key(frame)));
    assert_eq!(
        keys.len(),
        OUTPUT_POOL_SIZE + 1,
        "simultaneously-live output frames must never alias one texture"
    );

    let downloaded = download_frame(&device, context, first);
    assert_eq!(
        pixel(&downloaded, 0, 0),
        [0, 0, 255, 255],
        "later compositions must not overwrite a queued first frame"
    );
}

#[test]
fn nv12_conversion_uses_frame_color_space_and_range() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 2,
        height: 2,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");
    let mut layer = VideoLayer::new(VideoRect::new(0, 0, 2, 2));
    layer.fit = video_layer::VideoFit::Stretch;
    let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;
    let texture = nv12_texture(&device, 2, 2, 81, 90, 240);

    let mut bt601 = wrap_d3d11_texture(texture.clone(), 2, 2).unwrap();
    bt601.set_color_space(ffmpeg::color::Space::SMPTE170M);
    bt601.set_color_range(ffmpeg::color::Range::MPEG);
    sink.consume(pooled_video(bt601)).unwrap();
    let bt601 = compositor
        .compose_frame(&test_bus())
        .expect("BT.601 compose failed");
    let bt601 = download_frame(&device, context.clone(), bt601);

    let mut bt709 = wrap_d3d11_texture(texture, 2, 2).unwrap();
    bt709.set_color_space(ffmpeg::color::Space::BT709);
    bt709.set_color_range(ffmpeg::color::Range::MPEG);
    sink.consume(pooled_video(bt709)).unwrap();
    let bt709 = compositor
        .compose_frame(&test_bus())
        .expect("BT.709 compose failed");
    let bt709 = download_frame(&device, context, bt709);

    let pixel_601 = pixel(&bt601, 0, 0);
    let pixel_709 = pixel(&bt709, 0, 0);
    assert!(
        pixel_601[1].abs_diff(pixel_709[1]) >= 20,
        "the same NV12 sample should use different 601/709 matrices: {pixel_601:?} vs {pixel_709:?}"
    );
    assert_eq!(pixel_601[3], 255);
    assert_eq!(pixel_709[3], 255);
}

#[test]
fn nv12_conversion_distinguishes_limited_and_full_range() {
    let limited = yuv_to_rgb_rows(
        ffmpeg::color::Space::BT709,
        ffmpeg::color::Range::MPEG,
        1080,
    );
    let full = yuv_to_rgb_rows(
        ffmpeg::color::Space::BT709,
        ffmpeg::color::Range::JPEG,
        1080,
    );
    let neutral = 128.0 / 255.0;
    let limited_black = apply_color_rows(limited, 16.0 / 255.0, neutral, neutral);
    let limited_white = apply_color_rows(limited, 235.0 / 255.0, neutral, neutral);
    let full_black = apply_color_rows(full, 0.0, neutral, neutral);
    let full_white = apply_color_rows(full, 1.0, neutral, neutral);

    for channel in limited_black.into_iter().chain(full_black) {
        assert!(channel.abs() < 1e-5, "black mapped to {channel}");
    }
    for channel in limited_white.into_iter().chain(full_white) {
        assert!((channel - 1.0).abs() < 1e-5, "white mapped to {channel}");
    }
}

/// A tick that finds nothing changed hands out the picture it composed
/// last rather than drawing the same one again — and a layer that moves
/// puts it straight back to work.
#[test]
fn an_unchanged_scene_is_composed_once() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 2,
        height: 2,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");
    let mut layer = VideoLayer::new(VideoRect::new(0, 0, 2, 2));
    layer.fit = video_layer::VideoFit::Stretch;
    let input = handle.add_source("input", layer).unwrap().unwrap();
    let mut sink = input.sink;
    let layer_handle = input.layer;
    sink.consume(pooled_video(
        wrap_d3d11_texture(bgra_texture(&device, 2, 2, [0, 0, 255, 255]), 2, 2).unwrap(),
    ))
    .unwrap();

    let composed = compositor
        .compose_frame(&test_bus())
        .expect("first compose failed");
    let picture = texture_key(&composed);
    assert_eq!(composed.pts(), Some(0));

    let repeated = compositor
        .compose_frame(&test_bus())
        .expect("repeat failed");
    assert_eq!(
        texture_key(&repeated),
        picture,
        "nothing changed, so this is the picture already composed"
    );
    assert_eq!(
        repeated.pts(),
        Some(1),
        "a repeat carries this tick's timestamp, not the one it points at"
    );
    let downloaded = download_frame(&device, context.clone(), repeated);
    assert_eq!(
        pixel(&downloaded, 0, 0),
        [0, 0, 255, 255],
        "and it still shows what was composed"
    );

    layer_handle.set_rect(VideoRect::new(1, 1, 1, 1)).unwrap();
    let moved = compositor
        .compose_frame(&test_bus())
        .expect("compose after the move failed");
    assert_ne!(
        texture_key(&moved),
        picture,
        "a moved layer is a different picture and must be composed"
    );
    let downloaded = download_frame(&device, context, composed);
    assert_eq!(
        pixel(&downloaded, 0, 0),
        [0, 0, 255, 255],
        "composing again must not draw over the picture still held"
    );
}

/// A repeat holds the picture's buffer but not its pool slot, so the
/// slot must stay out of the pool while that repeat is still in flight —
/// otherwise a later composite draws into the texture it is showing.
#[test]
fn a_repeat_still_in_flight_is_never_composed_over() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 2,
        height: 2,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");
    let mut layer = VideoLayer::new(VideoRect::new(0, 0, 2, 2));
    layer.fit = video_layer::VideoFit::Stretch;
    let mut sink = handle.add_source("input", layer).unwrap().unwrap().sink;
    let blue = |device: &ID3D11Device| {
        pooled_video(
            wrap_d3d11_texture(bgra_texture(device, 2, 2, [255, 0, 0, 255]), 2, 2).unwrap(),
        )
    };

    sink.consume(blue(&device)).unwrap();
    // Composed, pushed, and consumed downstream: only the repeat below
    // still refers to this picture.
    drop(
        compositor
            .compose_frame(&test_bus())
            .expect("first compose failed"),
    );
    let in_flight = compositor
        .compose_frame(&test_bus())
        .expect("repeat failed");
    let showing = texture_key(&in_flight);

    // Every one of these changes the input, so every one really draws,
    // and each result is dropped immediately — the pool recycles as fast
    // as it can, which is exactly the case that would reuse the picture
    // the repeat above is still showing.
    for _ in 0..(OUTPUT_POOL_SIZE * 2 + 2) {
        sink.consume(blue(&device)).unwrap();
        let composed = compositor
            .compose_frame(&test_bus())
            .expect("later compose failed");
        assert_ne!(
            texture_key(&composed),
            showing,
            "composed into the texture a repeat still in flight is showing"
        );
    }

    let downloaded = download_frame(&device, context, in_flight);
    assert_eq!(
        pixel(&downloaded, 0, 0),
        [255, 0, 0, 255],
        "the repeat still shows the picture it was published with"
    );
}

#[test]
fn layer_handle_moves_blends_and_hides_a_live_source() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 3,
        height: 1,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device, context.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");

    let layer = VideoLayer::new(VideoRect::new(0, 0, 1, 1));
    let input = handle.add_source("white", layer).unwrap().unwrap();
    let mut sink = input.sink;
    let layer_handle = input.layer;

    let white_texture = bgra_texture(&device, 1, 1, [255, 255, 255, 255]);
    sink.consume(pooled_video(
        wrap_d3d11_texture(white_texture, 1, 1).unwrap(),
    ))
    .unwrap();

    layer_handle.set_rect(VideoRect::new(1, 0, 1, 1)).unwrap();
    layer_handle.set_opacity(0.5).unwrap();
    let blended = compositor
        .compose_frame(&test_bus())
        .expect("compose_frame failed");
    let downloaded = download_frame(&device, context.clone(), blended);
    assert_eq!(pixel(&downloaded, 0, 0), [0, 0, 0, 255], "background only");
    // 255 * 0.5 = 127.5 — the CPU SwVideoCompositor's software blend
    // rounds this to 128 (`f32::round`), but D3D11's fixed-function
    // blend hardware truncates instead, landing on 127. Both are
    // legitimate roundings of the same exact half-way value; this
    // one-off discrepancy is an inherent CPU-vs-GPU-blend-unit
    // difference, not a bug in either path.
    let blended_pixel = pixel(&downloaded, 1, 0);
    assert_eq!(
        blended_pixel[3], 255,
        "50% white over black: {blended_pixel:?}"
    );
    for channel in &blended_pixel[..3] {
        assert!(
            (127..=128).contains(channel),
            "50% white over black: {blended_pixel:?}"
        );
    }

    layer_handle.set_visible(false).unwrap();
    let hidden = compositor
        .compose_frame(&test_bus())
        .expect("compose_frame failed");
    assert_eq!(hidden.pts(), Some(1));
    let downloaded = download_frame(&device, context, hidden);
    assert_eq!(pixel(&downloaded, 1, 0), [0, 0, 0, 255], "hidden layer");
}

#[test]
fn skips_a_mismatched_device_texture_and_reports_it_on_the_bus() {
    let Some((device_a, context_a)) = try_device() else {
        return;
    };
    let Some((device_b, _context_b)) = try_device() else {
        return;
    };
    let options = VideoCompositorOptions {
        width: 1,
        height: 1,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (mut compositor, handle) =
        D3d11VideoCompositor::new("compositor", &device_a, context_a.clone(), options)
            .expect("D3d11VideoCompositor::new should succeed");
    let mut sink = handle
        .add_source("mismatched", VideoLayer::new(VideoRect::new(0, 0, 1, 1)))
        .unwrap()
        .unwrap()
        .sink;

    let foreign_texture = bgra_texture(&device_b, 1, 1, [255, 255, 255, 255]);
    sink.consume(pooled_video(
        wrap_d3d11_texture(foreign_texture, 1, 1).unwrap(),
    ))
    .unwrap();

    let (bus, bus_rx) = Bus::new();
    let composed = compositor
        .compose_frame(&bus)
        .expect("a mismatched-device layer must be skipped, not fail the whole frame");

    let error = match bus_rx
        .try_recv()
        .expect("the skipped layer should be reported on the bus")
    {
        BusEvent::Error { error, .. } => error,
        other => panic!("expected a BusEvent::Error, got {other:?}"),
    };
    assert!(matches!(
        error,
        crate::error::Error::D3d11VideoCompositorError(D3d11VideoCompositorError::DeviceMismatch)
    ));

    let downloaded = download_frame(&device_a, context_a, composed);
    assert_eq!(
        pixel(&downloaded, 0, 0),
        [0, 0, 0, 255],
        "mismatched-device layer must not be drawn — background only"
    );
}

struct TimestampSink {
    pp_log: PpLog,
    tx: crossbeam_channel::Sender<Instant>,
}

impl Element for TimestampSink {
    fn name(&self) -> Arc<str> {
        "timestamp-recorder".into()
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

impl Sink for TimestampSink {
    fn consume(&mut self, buf: MediaBuffer) -> Result<()> {
        if matches!(buf, MediaBuffer::Video(_)) {
            let _ = self.tx.send(Instant::now());
        }
        Ok(())
    }
    fn control(&mut self, _msg: ControlMsg) -> Result<()> {
        Ok(())
    }
}

/// Same regression as `SwVideoCompositor`'s
/// `resuming_after_a_pause_preserves_output_phase` — see that test's
/// docs for the full rationale. `D3d11VideoCompositor::run` had the
/// identical bug (never folding `paused_for` back into `next_due`).
#[test]
fn resuming_after_a_pause_preserves_output_phase() {
    use crate::pipeline::Pipeline;

    let Some((device, context)) = try_device() else {
        return;
    };
    let (tx, rx) = crossbeam_channel::unbounded();
    let sink = TimestampSink {
        tx,
        pp_log: element_pp_log(ElementType::Other, "timestamp-recorder", None),
    };
    let options = VideoCompositorOptions {
        width: 2,
        height: 2,
        frame_rate: ffmpeg::Rational::new(10, 1),
        background: Color::BLACK,
    };
    let (compositor, _handle) = D3d11VideoCompositor::new("compositor", &device, context, options)
        .expect("D3d11VideoCompositor::new should succeed");

    let pipeline = Pipeline::new("phase-test", compositor, |source, ctx| {
        let branch = ctx.branch().to(Box::new(sink))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    for _ in 0..2 {
        rx.recv_timeout(Duration::from_millis(500))
            .expect("expected steady frames before pausing");
    }
    pipeline.pause();
    thread::sleep(Duration::from_millis(500));

    let resumed_at = Instant::now();
    pipeline.resume();
    let first_after_resume = rx
        .recv_timeout(Duration::from_millis(500))
        .expect("expected a frame after resume");
    pipeline.stop();
    pipeline.bus().log_events();

    let gap = first_after_resume.saturating_duration_since(resumed_at);
    assert!(
        gap >= Duration::from_millis(50),
        "expected the post-pause frame to land close to a full 100ms \
         interval after resume (phase preserved from before the \
         pause), not almost immediately (phase reset to the resume \
         instant): got {gap:?}"
    );
}

/// This compositor reports its ticks through its pipeline, as
/// `SwVideoCompositor` does — see that one's
/// `a_compositor_reports_its_ticks_and_counts_a_held_push_as_missed` for
/// what the numbers mean. This checks the wiring on this backend: the
/// frames a Preview sees drawn are the ticks it reports.
#[test]
fn its_ticks_are_reported_through_its_pipeline() {
    use crate::pipeline::Pipeline;

    let Some((device, context)) = try_device() else {
        return;
    };
    let (tx, rx) = crossbeam_channel::unbounded();
    let sink = TimestampSink {
        tx,
        pp_log: element_pp_log(ElementType::Other, "timestamp-recorder", None),
    };
    let options = VideoCompositorOptions {
        width: 2,
        height: 2,
        frame_rate: ffmpeg::Rational::new(30, 1),
        background: Color::BLACK,
    };
    let (compositor, _handle) = D3d11VideoCompositor::new("ticking", &device, context, options)
        .expect("D3d11VideoCompositor::new should succeed");
    let pipeline = Pipeline::new("ticks", compositor, |source, ctx| {
        let branch = ctx.branch().to(Box::new(sink))?;
        ctx.attach(source, 0, branch)?;
        Ok(())
    })
    .expect("test pipeline wiring must succeed");

    pipeline.run().unwrap();
    for _ in 0..3 {
        rx.recv_timeout(Duration::from_millis(500))
            .expect("the compositor keeps drawing");
    }
    pipeline.stop();

    let stats = pipeline.stats();
    let compositor = stats
        .elements
        .iter()
        .find(|element| &*element.name == "ticking")
        .expect("the compositor is reported");
    let ticks = compositor.ticks.expect("a compositor reports its ticks");
    assert_eq!(
        ticks.made, compositor.pads[0].buffers,
        "every tick drawn is a frame pushed"
    );
    assert!(ticks.made >= 3, "{ticks:?}");
    assert!(ticks.work > Duration::ZERO, "{ticks:?}");
}

/// The whole point of the setter: the rate a running compositor emits at
/// can be changed, and reading it back says so.
#[test]
fn the_frame_rate_can_be_changed_while_it_is_running() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let (compositor, handle) = D3d11VideoCompositor::new(
        "rate",
        &device,
        context,
        VideoCompositorOptions {
            width: 64,
            height: 64,
            frame_rate: ffmpeg::Rational::new(60, 1),
            background: Color::BLACK,
        },
    )
    .expect("compositor");

    assert_eq!(compositor.frame_rate(), ffmpeg::Rational::new(60, 1));
    assert_eq!(compositor.time_base(), ffmpeg::Rational::new(1, 60));

    assert!(handle.set_frame_rate(ffmpeg::Rational::new(30, 1)));
    assert_eq!(handle.frame_rate(), Some(ffmpeg::Rational::new(30, 1)));
    // The element and the handle are reading one value, not two.
    assert_eq!(compositor.frame_rate(), ffmpeg::Rational::new(30, 1));
    // And the unit every output timestamp is in moves with it.
    assert_eq!(compositor.time_base(), ffmpeg::Rational::new(1, 30));
}

/// A rate that cannot be kept is refused, and refusing leaves the old one
/// running rather than a compositor ticking on a nonsense interval.
#[test]
fn an_impossible_frame_rate_is_refused_and_changes_nothing() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let (compositor, handle) = D3d11VideoCompositor::new(
        "rate-refused",
        &device,
        context,
        VideoCompositorOptions {
            width: 64,
            height: 64,
            frame_rate: ffmpeg::Rational::new(60, 1),
            background: Color::BLACK,
        },
    )
    .expect("compositor");

    for refused in [
        ffmpeg::Rational::new(0, 1),
        ffmpeg::Rational::new(-30, 1),
        ffmpeg::Rational::new(30, 0),
    ] {
        assert!(!handle.set_frame_rate(refused), "{refused} was accepted");
        assert_eq!(compositor.frame_rate(), ffmpeg::Rational::new(60, 1));
    }
}

/// Once the compositor is gone the handle answers rather than panicking,
/// the same way every other method on it does.
#[test]
fn the_setter_reports_a_compositor_that_is_gone() {
    let Some((device, context)) = try_device() else {
        return;
    };
    let (compositor, handle) = D3d11VideoCompositor::new(
        "rate-dropped",
        &device,
        context,
        VideoCompositorOptions {
            width: 64,
            height: 64,
            frame_rate: ffmpeg::Rational::new(60, 1),
            background: Color::BLACK,
        },
    )
    .expect("compositor");
    drop(compositor);

    assert!(!handle.set_frame_rate(ffmpeg::Rational::new(30, 1)));
    assert_eq!(handle.frame_rate(), None);
}
