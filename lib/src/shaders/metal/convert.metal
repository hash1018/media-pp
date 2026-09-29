// `MetalConverter`'s kernel, the Metal counterpart of
// `shaders/vulkan/convert.wgsl`: an NV12 picture as RGB, by the three affine
// rows its own colour description gives — `color::yuv_to_rgb_rows`, what
// the compositor and the renderers convert with. Each pixel reads its own
// luma and the chroma sample of the 2x2 block it is in.
//
// Texture 0 is the output, written through a `bgra8Unorm` view, which puts
// the bytes in a BGRA frame's order itself — see `platform::macos::metal_pass`.

#include <metal_stdlib>
using namespace metal;

struct Convert {
    // The picture's width and height.
    uint4 size;
    float4 r;
    float4 g;
    float4 b;
};

kernel void convert(texture2d<float, access::write> output [[texture(0)]],
                    texture2d<float, access::read> luma [[texture(1)]],
                    texture2d<float, access::read> chroma [[texture(2)]],
                    constant Convert &convert [[buffer(0)]],
                    uint2 id [[thread_position_in_grid]]) {
    if (id.x >= convert.size.x || id.y >= convert.size.y) {
        return;
    }
    float4 yuv = float4(luma.read(id).r, chroma.read(id / 2).rg, 1.0);
    float3 rgb = clamp(float3(dot(convert.r, yuv), dot(convert.g, yuv), dot(convert.b, yuv)),
                       float3(0.0), float3(1.0));
    output.write(float4(rgb, 1.0), id);
}
