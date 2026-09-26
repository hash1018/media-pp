// `VulkanVideoCompositor`'s kernels: a canvas filled with the background,
// each layer blended onto it in turn — a video layer sampled from its
// picture, a text layer from its coverage mask — and the canvas turned into
// NV12 when that is what the compositor emits. One entry point per step,
// each compiled into a module of its own, so each pipeline's descriptors are
// exactly the bindings it reads.
//
// The canvas is an `rgba8unorm` image holding B, G, R, A in its four
// channels — the byte order of a BGRA frame — so a BGRA composition is
// handed on with a plain copy into its frame. `load` and `store` are the one
// place that order is known; everything else works in RGB.

// What one dispatch draws, as push constants — WGSL's `immediate` space.
struct Step {
    // The part of the canvas the dispatch writes: its top-left corner, then
    // its width and height, one invocation per pixel.
    region: vec4<i32>,
    // Where the whole scaled picture lies on the canvas and its size — for
    // a text layer, the mask's top-left corner.
    image: vec4<f32>,
    // The picture's texture coordinates: the part of the image a layer
    // draws, as a scale and then an offset of the picture's own 0..1.
    uv: vec4<f32>,
    // Three affine rows turning a normalized (Y, Cb, Cr, 1) sample into R,
    // G and B, by the layer's own colour description. The first holds the
    // colour instead for the background (with its alpha) and a text layer.
    r: vec4<f32>,
    g: vec4<f32>,
    b: vec4<f32>,
    // The layer's opacity, and 1 where its colour already has its alpha
    // multiplied in.
    blend: vec4<f32>,
};

var<immediate> step: Step;

@group(0) @binding(0) var canvas: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var picture_sampler: sampler;
@group(0) @binding(2) var plane0: texture_2d<f32>;
@group(0) @binding(3) var plane1: texture_2d<f32>;
@group(0) @binding(4) var luma: texture_storage_2d<r8unorm, write>;
@group(0) @binding(5) var chroma: texture_storage_2d<rg8unorm, write>;

fn load(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(canvas, p).bgra;
}

fn store(p: vec2<i32>, colour: vec4<f32>) {
    textureStore(canvas, p, colour.bgra);
}

// The canvas pixel invocation `id` writes, or none past the region's edge.
fn pixel(id: vec3<u32>) -> vec2<i32> {
    if (id.x >= u32(step.region.z) || id.y >= u32(step.region.w)) {
        return vec2<i32>(-1, -1);
    }
    return step.region.xy + vec2<i32>(id.xy);
}

// Where canvas pixel `p` samples the picture: its centre, as a fraction of
// the scaled picture, into the part of the image drawn.
fn source_uv(p: vec2<i32>) -> vec2<f32> {
    let local = (vec2<f32>(p) + vec2<f32>(0.5) - step.image.xy) / step.image.zw;
    return local * step.uv.xy + step.uv.zw;
}

// Lays `colour` over what is at `p`: `alpha` of it covers what is there,
// and the colour is scaled by `weight` — `alpha` for a plain colour, the
// opacity alone for one that already carries its alpha.
fn blend(p: vec2<i32>, colour: vec3<f32>, alpha: f32, weight: f32) {
    let under = load(p);
    let rgb = colour * weight + under.rgb * (1.0 - alpha);
    store(p, vec4<f32>(rgb, alpha + under.a * (1.0 - alpha)));
}

@compute @workgroup_size(8, 8)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = pixel(id);
    if (p.x < 0) {
        return;
    }
    store(p, step.r);
}

// NV12: luma, and Cb and Cr interleaved in a plane at half the size.
@compute @workgroup_size(8, 8)
fn layer_nv12(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = pixel(id);
    if (p.x < 0) {
        return;
    }
    let uv = source_uv(p);
    let yuv = vec4<f32>(
        textureSampleLevel(plane0, picture_sampler, uv, 0.0).r,
        textureSampleLevel(plane1, picture_sampler, uv, 0.0).rg,
        1.0,
    );
    let rgb = clamp(
        vec3<f32>(dot(step.r, yuv), dot(step.g, yuv), dot(step.b, yuv)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let opacity = step.blend.x;
    blend(p, rgb, opacity, opacity);
}

// BGRA: already R'G'B', with alpha. Read through a B8G8R8A8 view, so the
// sample comes back in RGBA order.
@compute @workgroup_size(8, 8)
fn layer_bgra(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = pixel(id);
    if (p.x < 0) {
        return;
    }
    let colour = textureSampleLevel(plane0, picture_sampler, source_uv(p), 0.0);
    let opacity = step.blend.x;
    let alpha = colour.a * opacity;
    blend(p, colour.rgb, alpha, select(alpha, opacity, step.blend.y > 0.5));
}

// A text layer: its colour, as much of it as the mask covers, never scaled.
@compute @workgroup_size(8, 8)
fn text(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = pixel(id);
    if (p.x < 0) {
        return;
    }
    let coverage = textureLoad(plane0, p - vec2<i32>(step.image.xy), 0).r;
    let alpha = coverage * step.blend.x;
    blend(p, step.r.rgb, alpha, alpha);
}

// BT.709 at limited range, what the canvas is said to be as NV12.
fn to_luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The canvas as NV12: each invocation one 2x2 block — four luma samples,
// and the chroma sample of their average.
@compute @workgroup_size(8, 8)
fn to_nv12(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(canvas));
    let base = vec2<i32>(id.xy) * 2;
    if (base.x >= size.x || base.y >= size.y) {
        return;
    }
    var sum = vec3<f32>(0.0);
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let p = base + vec2<i32>(dx, dy);
            let rgb = load(p).rgb;
            let y = 16.0 / 255.0 + 219.0 / 255.0 * to_luma(rgb);
            textureStore(luma, p, vec4<f32>(y, 0.0, 0.0, 1.0));
            sum += rgb;
        }
    }
    let rgb = sum / 4.0;
    let y = to_luma(rgb);
    let cb = (rgb.b - y) / 1.8556;
    let cr = (rgb.r - y) / 1.5748;
    textureStore(
        chroma,
        vec2<i32>(id.xy),
        vec4<f32>(128.0 / 255.0 + 224.0 / 255.0 * cb, 128.0 / 255.0 + 224.0 / 255.0 * cr, 0.0, 1.0),
    );
}
