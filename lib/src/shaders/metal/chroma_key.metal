// `MetalChromaKey`'s kernel, the Metal counterpart of
// `shaders/vulkan/chroma_key.wgsl`: the key colour's distance out of alpha,
// the colour passed through — what the D3D11 shader and the CUDA kernel
// compute, from the band `options::feather_band` resolves.
//
// Texture 0 is the output, written through a `bgra8Unorm` view — see
// `platform::macos::metal_pass`.

#include <metal_stdlib>
using namespace metal;

struct Key {
    // The picture's width and height.
    uint4 size;
    // The key colour, and the band's low end.
    float4 key;
    // The band's inverse width.
    float4 band;
};

kernel void key(texture2d<float, access::write> output [[texture(0)]],
                texture2d<float, access::read> source [[texture(1)]],
                constant Key &key [[buffer(0)]],
                uint2 id [[thread_position_in_grid]]) {
    if (id.x >= key.size.x || id.y >= key.size.y) {
        return;
    }
    float4 colour = source.read(id);
    // Euclidean RGB distance to the key colour, scaled so opposite corners
    // of the colour cube are 1 — `SwChromaKey`'s `color_distance`.
    float distance = length(colour.rgb - key.key.rgb) / sqrt(3.0);
    float alpha = clamp((distance - key.key.w) * key.band.x, 0.0, 1.0);
    output.write(float4(colour.rgb, colour.a * alpha), id);
}
