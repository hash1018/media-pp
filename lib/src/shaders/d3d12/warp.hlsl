// The kernel `D3d12OrtEmbedder` reads an object into an embedding model's
// input with, the D3D12 counterpart of `warp_nv12` in
// `shaders/metal/fit.metal` and the CUDA kernel of that name: input pixel
// `(u, v)` is read from the picture at `(dot(x_of.xyz, (u, v, 1)),
// dot(y_of.xyz, (u, v, 1)))` — each pixel's centre at whole numbers —
// between the four pixels around it, as `cv2.warpAffine` reads a face onto
// the ArcFace template. Outside the picture, `size` in pixels, is black.
// The luma and chroma are each mixed over the pixels inside and made RGB by
// the picture's rows, then darkened by how much of the four was outside,
// which is what mixing the RGB of each comes to. Written to three float
// planes of input `slot.x` of `tensor` — R, G, B, or B, G, R where `slot.y`
// says so — each then times `scale` plus `bias`.
//
// Compiled apart from `fit.hlsl`, as its constants are at b0 too: `Warp` as
// 36 root constants, the luma and chroma planes' views at t0 and t1, and
// the tensor as a root UAV at u0.

cbuffer Warp : register(b0)
{
    uint2 model;
    uint2 size;
    uint2 slot;
    uint2 unused;
    float4 x_of;
    float4 y_of;
    float4 to_red;
    float4 to_green;
    float4 to_blue;
    float4 scale;
    float4 bias;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
RWStructuredBuffer<float> tensor : register(u0);

// The four pixels around `p` in a plane `extent` in pixels: where each is,
// held inside the plane, and its weight, naught for one outside it.
struct Four
{
    int2 at[4];
    float weight[4];
    float inside;
};

Four around(float2 p, uint2 extent)
{
    float2 corner = floor(p);
    float2 t = p - corner;
    float2 s = 1.0 - t;
    int2 first = int2(corner);
    const int2 step[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    const float weights[4] = { s.x * s.y, t.x * s.y, s.x * t.y, t.x * t.y };
    int2 last = int2(extent) - 1;
    Four four;
    four.inside = 0.0;
    [unroll]
    for (int i = 0; i < 4; i++) {
        int2 q = first + step[i];
        bool in_plane = q.x >= 0 && q.y >= 0 && q.x <= last.x && q.y <= last.y;
        four.at[i] = clamp(q, int2(0, 0), last);
        four.weight[i] = in_plane ? weights[i] : 0.0;
        four.inside += four.weight[i];
    }
    return four;
}

#define WARP_ROOT_SIGNATURE \
    "RootConstants(num32BitConstants = 36, b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 2)), " \
    "UAV(u0)"

[RootSignature(WARP_ROOT_SIGNATURE)]
[numthreads(16, 16, 1)]
void warp_nv12(uint3 thread_id : SV_DispatchThreadID)
{
    uint2 id = thread_id.xy;
    if (id.x >= model.x || id.y >= model.y) {
        return;
    }
    float3 at = float3(float(id.x), float(id.y), 1.0);
    float2 p = float2(dot(x_of.xyz, at), dot(y_of.xyz, at));
    Four four = around(p, size);
    float3 rgb = float3(0.0, 0.0, 0.0);
    if (four.inside > 0.0) {
        float y = 0.0;
        [unroll]
        for (int i = 0; i < 4; i++) {
            y += luma.Load(int3(four.at[i], 0)) * four.weight[i];
        }
        y /= four.inside;
        // The chroma plane, half each way, rounded up; a luma point's place
        // in it, centres to centres.
        Four cs = around((p + 0.5) * 0.5 - 0.5, (size + 1) / 2);
        float2 c = float2(0.5, 0.5);
        if (cs.inside > 0.0) {
            c = float2(0.0, 0.0);
            [unroll]
            for (int j = 0; j < 4; j++) {
                c += chroma.Load(int3(cs.at[j], 0)) * cs.weight[j];
            }
            c /= cs.inside;
        }
        float4 yuv = float4(y, c, 1.0);
        rgb = saturate(float3(dot(to_red, yuv), dot(to_green, yuv), dot(to_blue, yuv)))
            * four.inside;
    }
    uint plane = model.x * model.y;
    uint index = slot.x * 3 * plane + id.y * model.x + id.x;
    if (slot.y != 0) {
        rgb = rgb.bgr;
    }
    rgb = rgb * scale.rgb + bias.rgb;
    tensor[index] = rgb.r;
    tensor[plane + index] = rgb.g;
    tensor[2 * plane + index] = rgb.b;
}
