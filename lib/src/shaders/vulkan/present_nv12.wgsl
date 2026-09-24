// Draws an NV12 frame — a luma plane and an interleaved chroma plane, each
// its own sampled image — as RGB, for `VulkanWindowRenderer`.
//
// The matrix and the range are not fixed here: they come in as push
// constants — WGSL's `immediate` address space — worked out from the frame's
// own colour description, so a BT.709 frame and a BT.601 one are each drawn
// with their own coefficients rather than one of them with the other's. See `Colour` in
// `vulkan_window_renderer.rs` for how the eight values are derived.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One oversized triangle that covers the viewport, with no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    var out: VsOut;
    // `v` runs against clip-space y: WGSL's clip space points +y up, while
    // this pipeline draws through an ordinary positive-height Vulkan
    // viewport, whose framebuffer y grows downward. Without the flip the
    // picture is upside down.
    out.uv = vec2<f32>(x, 1.0 - y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

struct Colour {
    // Y' = (Y - y_offset) * y_scale
    y_offset: f32,
    y_scale: f32,
    // C = (C - c_offset) * c_scale, for both Cb and Cr
    c_offset: f32,
    c_scale: f32,
    // R = Y' + cr_to_r Cr
    // G = Y' - cb_to_g Cb - cr_to_g Cr
    // B = Y' + cb_to_b Cb
    cr_to_r: f32,
    cb_to_g: f32,
    cr_to_g: f32,
    cb_to_b: f32,
};

var<immediate> colour: Colour;

@group(0) @binding(0) var frame_sampler: sampler;
@group(0) @binding(1) var luma: texture_2d<f32>;
@group(0) @binding(2) var chroma: texture_2d<f32>;

@fragment
fn fs_nv12(in: VsOut) -> @location(0) vec4<f32> {
    let y = (textureSample(luma, frame_sampler, in.uv).r - colour.y_offset) * colour.y_scale;
    let c = (textureSample(chroma, frame_sampler, in.uv).rg - vec2<f32>(colour.c_offset))
        * colour.c_scale;
    let rgb = vec3<f32>(
        y + colour.cr_to_r * c.y,
        y - colour.cb_to_g * c.x - colour.cr_to_g * c.y,
        y + colour.cb_to_b * c.x,
    );
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
