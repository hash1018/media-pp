// `MetalWindowRenderer`'s kernels: a picture drawn into a window's drawable,
// letterboxed — its aspect ratio kept, black bars around it — and converted
// from its own colour description. One kernel per layout it takes; each
// writes every pixel of the drawable, bars included.

#include <metal_stdlib>
using namespace metal;

// What one picture is drawn with, bound at buffer 0.
struct Present {
    // The picture's rectangle in the drawable: its top-left corner, then its
    // width and height.
    int4 picture;
    // The part of the textures the picture is: a scale of their own 0..1,
    // for a pixel buffer larger than the picture it holds.
    float4 uv;
    // Three affine rows turning a normalized (Y, Cb, Cr, 1) sample into R, G
    // and B, by the frame's own colour description.
    float4 r;
    float4 g;
    float4 b;
};

constexpr sampler picture_sampler(filter::linear, address::clamp_to_edge);

// Where drawable pixel `id` samples the picture, or false for a bar.
static bool picture_uv(constant Present &present, uint2 id, thread float2 &uv) {
    int2 local = int2(id) - present.picture.xy;
    if (local.x < 0 || local.y < 0 || local.x >= present.picture.z || local.y >= present.picture.w) {
        return false;
    }
    uv = (float2(local) + float2(0.5)) / float2(present.picture.zw) * present.uv.xy;
    return true;
}

static float4 from_yuv(constant Present &present, float3 yuv) {
    float4 sample = float4(yuv, 1.0);
    float3 rgb = float3(dot(present.r, sample), dot(present.g, sample), dot(present.b, sample));
    return float4(clamp(rgb, float3(0.0), float3(1.0)), 1.0);
}

static bool outside(texture2d<float, access::write> output, uint2 id) {
    return id.x >= output.get_width() || id.y >= output.get_height();
}

constant float4 black = float4(0.0, 0.0, 0.0, 1.0);

// NV12: luma, and Cb and Cr interleaved in a plane at half the size.
kernel void present_nv12(texture2d<float, access::write> output [[texture(0)]],
                         texture2d<float> luma [[texture(1)]],
                         texture2d<float> chroma [[texture(2)]],
                         constant Present &present [[buffer(0)]],
                         uint2 id [[thread_position_in_grid]]) {
    if (outside(output, id)) {
        return;
    }
    float2 uv;
    if (!picture_uv(present, id, uv)) {
        output.write(black, id);
        return;
    }
    float3 yuv = float3(luma.sample(picture_sampler, uv, level(0)).r,
                        chroma.sample(picture_sampler, uv, level(0)).rg);
    output.write(from_yuv(present, yuv), id);
}

// YUV420P: luma, Cb and Cr, each a plane of its own.
kernel void present_yuv420p(texture2d<float, access::write> output [[texture(0)]],
                            texture2d<float> luma [[texture(1)]],
                            texture2d<float> cb [[texture(2)]],
                            texture2d<float> cr [[texture(3)]],
                            constant Present &present [[buffer(0)]],
                            uint2 id [[thread_position_in_grid]]) {
    if (outside(output, id)) {
        return;
    }
    float2 uv;
    if (!picture_uv(present, id, uv)) {
        output.write(black, id);
        return;
    }
    float3 yuv = float3(luma.sample(picture_sampler, uv, level(0)).r,
                        cb.sample(picture_sampler, uv, level(0)).r,
                        cr.sample(picture_sampler, uv, level(0)).r);
    output.write(from_yuv(present, yuv), id);
}

// BGRA, read through a `bgra8Unorm` view so it comes back in RGBA order;
// drawn as it is, its alpha set aside — a window shows nothing behind it.
kernel void present_bgra(texture2d<float, access::write> output [[texture(0)]],
                         texture2d<float> picture [[texture(1)]],
                         constant Present &present [[buffer(0)]],
                         uint2 id [[thread_position_in_grid]]) {
    if (outside(output, id)) {
        return;
    }
    float2 uv;
    if (!picture_uv(present, id, uv)) {
        output.write(black, id);
        return;
    }
    output.write(float4(picture.sample(picture_sampler, uv, level(0)).rgb, 1.0), id);
}
