// `MetalVideoEffect`'s kernel, the Metal counterpart of
// `shaders/vulkan/video_effect.wgsl`: every effect resolved into one colour
// matrix, an exponent, an opacity and a luma mask — `EffectParams` in
// `video_effect/options.rs`, which the D3D11 shader, the CUDA kernel, the
// Vulkan kernel and the software loop evaluate too.
//
// Texture 0 is the output, written through a `bgra8Unorm` view — see
// `platform::macos::metal_pass`.

#include <metal_stdlib>
using namespace metal;

struct Effect {
    // The picture's width and height.
    uint4 size;
    // One row per output channel, red, green, blue: xyz weigh the input's
    // red, green and blue, w is the offset.
    float4 r;
    float4 g;
    float4 b;
    // exponent, opacity, luma_low, luma_low_inv
    float4 first;
    // luma_high, luma_high_inv
    float4 second;
};

kernel void effect(texture2d<float, access::write> output [[texture(0)]],
                   texture2d<float, access::read> source [[texture(1)]],
                   constant Effect &effect [[buffer(0)]],
                   uint2 id [[thread_position_in_grid]]) {
    if (id.x >= effect.size.x || id.y >= effect.size.y) {
        return;
    }
    float4 colour = source.read(id);
    float exponent = effect.first.x;
    float opacity = effect.first.y;
    float luma_low = effect.first.z;
    float luma_low_inv = effect.first.w;
    float luma_high = effect.second.x;
    float luma_high_inv = effect.second.y;

    float luma = clamp(dot(colour.rgb, float3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    float mask = clamp((luma - luma_low) * luma_low_inv + 1.0, 0.0, 1.0)
        * clamp((luma_high - luma) * luma_high_inv + 1.0, 0.0, 1.0);

    float3 rgb = colour.rgb;
    if (exponent != 1.0) {
        rgb = pow(rgb, float3(exponent));
    }
    float4 lifted = float4(rgb, 1.0);
    float3 corrected = clamp(float3(dot(effect.r, lifted), dot(effect.g, lifted), dot(effect.b, lifted)),
                             float3(0.0), float3(1.0));
    float alpha = clamp(colour.a * opacity * mask, 0.0, 1.0);
    output.write(float4(corrected, alpha), id);
}
