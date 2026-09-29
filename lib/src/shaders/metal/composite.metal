// `MetalVideoCompositor`'s kernels, the Metal counterpart of
// `shaders/vulkan/composite.wgsl`: a canvas filled with the background, each
// layer blended onto it in turn — a video layer sampled from its picture, a
// text layer from its coverage mask — and the canvas written into the output
// frame, as BGRA or as NV12. Compiled from source by Metal when the
// compositor is made.
//
// The canvas is an `rgba8Unorm` texture in R, G, B, A order; a BGRA output
// is written through a `bgra8Unorm` view of its pixel buffer, which puts the
// bytes in that order itself.

#include <metal_stdlib>
using namespace metal;

// What one dispatch draws, bound at buffer 0.
struct Step {
    // The part of the canvas the dispatch writes: its top-left corner, then
    // its width and height, one thread per pixel.
    int4 region;
    // Where the whole scaled picture lies on the canvas and its size — for a
    // text layer, the mask's top-left corner.
    float4 image;
    // The picture's texture coordinates: the part of the image a layer
    // draws, as a scale and then an offset of the picture's own 0..1.
    float4 uv;
    // Three affine rows turning a normalized (Y, Cb, Cr, 1) sample into R, G
    // and B, by the layer's own colour description. The first holds the
    // colour instead for the background (with its alpha) and a text layer.
    float4 r;
    float4 g;
    float4 b;
    // The layer's opacity, and 1 where its colour already has its alpha
    // multiplied in.
    float4 blend;
};

constexpr sampler picture_sampler(filter::linear, address::clamp_to_edge);

// The canvas pixel thread `id` writes, or none past the region's edge.
static int2 pixel(constant Step &step, uint2 id) {
    if (id.x >= uint(step.region.z) || id.y >= uint(step.region.w)) {
        return int2(-1, -1);
    }
    return step.region.xy + int2(id);
}

// Where canvas pixel `p` samples the picture: its centre, as a fraction of
// the scaled picture, into the part of the image drawn.
static float2 source_uv(constant Step &step, int2 p) {
    float2 local = (float2(p) + float2(0.5) - step.image.xy) / step.image.zw;
    return local * step.uv.xy + step.uv.zw;
}

// Lays `colour` over what is at `p`: `alpha` of it covers what is there, and
// the colour is scaled by `weight` — `alpha` for a plain colour, the opacity
// alone for one that already carries its alpha.
static void blend(texture2d<float, access::read_write> canvas, int2 p, float3 colour,
                  float alpha, float weight) {
    float4 under = canvas.read(uint2(p));
    float3 rgb = colour * weight + under.rgb * (1.0 - alpha);
    canvas.write(float4(rgb, alpha + under.a * (1.0 - alpha)), uint2(p));
}

kernel void fill(texture2d<float, access::read_write> canvas [[texture(0)]],
                 constant Step &step [[buffer(0)]],
                 uint2 id [[thread_position_in_grid]]) {
    int2 p = pixel(step, id);
    if (p.x < 0) {
        return;
    }
    canvas.write(step.r, uint2(p));
}

// NV12: luma, and Cb and Cr interleaved in a plane at half the size.
kernel void layer_nv12(texture2d<float, access::read_write> canvas [[texture(0)]],
                       texture2d<float> plane0 [[texture(1)]],
                       texture2d<float> plane1 [[texture(2)]],
                       constant Step &step [[buffer(0)]],
                       uint2 id [[thread_position_in_grid]]) {
    int2 p = pixel(step, id);
    if (p.x < 0) {
        return;
    }
    float2 uv = source_uv(step, p);
    float4 yuv = float4(plane0.sample(picture_sampler, uv, level(0)).r,
                        plane1.sample(picture_sampler, uv, level(0)).rg, 1.0);
    float3 rgb = clamp(float3(dot(step.r, yuv), dot(step.g, yuv), dot(step.b, yuv)),
                       float3(0.0), float3(1.0));
    float opacity = step.blend.x;
    blend(canvas, p, rgb, opacity, opacity);
}

// BGRA: already R'G'B', with alpha. Read through a `bgra8Unorm` view, so
// the sample comes back in RGBA order.
kernel void layer_bgra(texture2d<float, access::read_write> canvas [[texture(0)]],
                       texture2d<float> plane0 [[texture(1)]],
                       constant Step &step [[buffer(0)]],
                       uint2 id [[thread_position_in_grid]]) {
    int2 p = pixel(step, id);
    if (p.x < 0) {
        return;
    }
    float4 colour = plane0.sample(picture_sampler, source_uv(step, p), level(0));
    float opacity = step.blend.x;
    float alpha = colour.a * opacity;
    blend(canvas, p, colour.rgb, alpha, step.blend.y > 0.5 ? opacity : alpha);
}

// A text layer: its colour, as much of it as the mask covers, never scaled.
kernel void text(texture2d<float, access::read_write> canvas [[texture(0)]],
                 texture2d<float> mask [[texture(1)]],
                 constant Step &step [[buffer(0)]],
                 uint2 id [[thread_position_in_grid]]) {
    int2 p = pixel(step, id);
    if (p.x < 0) {
        return;
    }
    float coverage = mask.read(uint2(p - int2(step.image.xy))).r;
    float alpha = coverage * step.blend.x;
    blend(canvas, p, step.r.rgb, alpha, alpha);
}

// The canvas into a BGRA frame.
kernel void to_bgra(texture2d<float, access::read_write> canvas [[texture(0)]],
                    texture2d<float, access::write> output [[texture(1)]],
                    uint2 id [[thread_position_in_grid]]) {
    if (id.x >= canvas.get_width() || id.y >= canvas.get_height()) {
        return;
    }
    output.write(canvas.read(id), id);
}

// BT.709 at limited range, what the canvas is said to be as NV12.
static float to_luma(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

// The canvas as NV12: each thread one 2x2 block — four luma samples, and the
// chroma sample of their average.
kernel void to_nv12(texture2d<float, access::read_write> canvas [[texture(0)]],
                    texture2d<float, access::write> luma [[texture(1)]],
                    texture2d<float, access::write> chroma [[texture(2)]],
                    uint2 id [[thread_position_in_grid]]) {
    uint2 base = id * 2;
    if (base.x >= canvas.get_width() || base.y >= canvas.get_height()) {
        return;
    }
    float3 sum = float3(0.0);
    for (uint dy = 0; dy < 2; dy++) {
        for (uint dx = 0; dx < 2; dx++) {
            uint2 p = base + uint2(dx, dy);
            float3 rgb = canvas.read(p).rgb;
            float y = 16.0 / 255.0 + 219.0 / 255.0 * to_luma(rgb);
            luma.write(float4(y, 0.0, 0.0, 1.0), p);
            sum += rgb;
        }
    }
    float3 rgb = sum / 4.0;
    float y = to_luma(rgb);
    float cb = (rgb.b - y) / 1.8556;
    float cr = (rgb.r - y) / 1.5748;
    chroma.write(float4(128.0 / 255.0 + 224.0 / 255.0 * cb, 128.0 / 255.0 + 224.0 / 255.0 * cr,
                        0.0, 1.0),
                 id);
}
