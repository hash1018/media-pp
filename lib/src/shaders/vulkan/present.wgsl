// Draws a video frame as RGB, for `VulkanWindowRenderer`: one vertex shader,
// and a fragment shader per layout the renderer takes. Each plane of a frame
// is its own sampled image, so an entry point samples whichever of bindings
// 1 to 3 its layout has; the renderer points the ones it does not have at
// the first plane, so every binding is always valid.
//
// For the YUV layouts the matrix and the range are not fixed here: they come
// in as push constants — WGSL's `immediate` address space — worked out from
// the frame's own colour description, so a BT.709 frame and a BT.601 one are
// each drawn with their own coefficients rather than one of them with the
// other's. They are three affine rows from `color::yuv_to_rgb_rows`, the
// same the D3D renderers use; see `Colour` in `vulkan_window_renderer.rs`.

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

// Each row turns (Y, Cb, Cr, 1), every sample normalized to 0..1, into one
// of R, G and B: range and matrix both, offsets in the fourth column.
struct Colour {
    r: vec4<f32>,
    g: vec4<f32>,
    b: vec4<f32>,
};

var<immediate> colour: Colour;

@group(0) @binding(0) var frame_sampler: sampler;
@group(0) @binding(1) var plane0: texture_2d<f32>;
@group(0) @binding(2) var plane1: texture_2d<f32>;
@group(0) @binding(3) var plane2: texture_2d<f32>;

fn to_rgb(luma: f32, cb: f32, cr: f32) -> vec4<f32> {
    let yuv = vec4<f32>(luma, cb, cr, 1.0);
    let rgb = vec3<f32>(dot(colour.r, yuv), dot(colour.g, yuv), dot(colour.b, yuv));
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

// NV12: luma, then Cb and Cr interleaved in one two-channel plane.
@fragment
fn fs_nv12(in: VsOut) -> @location(0) vec4<f32> {
    let chroma = textureSample(plane1, frame_sampler, in.uv).rg;
    return to_rgb(textureSample(plane0, frame_sampler, in.uv).r, chroma.x, chroma.y);
}

// YUV420P: luma, Cb and Cr, each a plane of its own.
@fragment
fn fs_yuv420p(in: VsOut) -> @location(0) vec4<f32> {
    return to_rgb(
        textureSample(plane0, frame_sampler, in.uv).r,
        textureSample(plane1, frame_sampler, in.uv).r,
        textureSample(plane2, frame_sampler, in.uv).r,
    );
}

// BGRA: already R'G'B'. The image is made B8G8R8A8, so sampling it reads the
// channels in RGBA order. Alpha is dropped — a window has nothing behind it
// to blend with.
@fragment
fn fs_bgra(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(plane0, frame_sampler, in.uv).rgb, 1.0);
}
