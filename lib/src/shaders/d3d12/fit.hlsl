// The kernel `D3d12OrtDetector` fits a picture into a model's input with,
// the D3D12 counterpart of `fit_nv12` in `shaders/metal/fit.metal` and in
// `platform::cuda::driver::ptx::FIT_PTX`: each thread one pixel of the
// model's `model.x` by `model.y` input, written to three float planes of
// input `slot.x` of `tensor` — R, then G, then B, or B, G, R where `slot.y`
// says so, each 0 to 1 and then times `scale` plus `bias`, as the model
// reads them.
//
// What is fitted is the rectangle of the picture at `origin`, `source` in
// size as shown, scaled inside the input at `offset` to `scaled` in size; a
// pixel outside it is the grey Ultralytics trains on, 114 of 255. Inside,
// each pixel is the mean of the source pixels it covers — three by three
// where a 1080p picture is shrunk to 640 — or, enlarging, of the one it
// falls in. An NV12 pixel is averaged as Y', Cb and Cr — each source pixel
// with the chroma sample it shares — and made RGB by the three rows its own
// colour description gives (`color::yuv_to_rgb_rows`).
//
// The rectangle is read the way the picture is shown: each pixel of it
// turned is read from the stored pixel `x_of` and `y_of` name — `origin +
// (dot(x_of.xyz, (x, y, 1)), dot(y_of.xyz, (x, y, 1)))` — which for a
// picture shown as stored is the pixel itself.
//
// The root signature, in `ort/d3d12/fitting.rs`: `Fit` as 40 root
// constants at b0, the luma and chroma planes' views at t0 and t1, and the
// tensor as a root UAV at u0.

cbuffer Fit : register(b0)
{
    uint2 model;
    uint2 offset;
    uint2 scaled;
    uint2 source;
    uint2 origin;
    uint2 slot;
    int4 x_of;
    int4 y_of;
    float4 to_red;
    float4 to_green;
    float4 to_blue;
    float4 scale;
    float4 bias;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
RWStructuredBuffer<float> tensor : register(u0);

static const float MARGIN = 114.0 / 255.0;

// The pixels of the rectangle as shown that input pixel `id` covers,
// `[lo, hi)`, or false where it is in the margin.
bool covered(uint2 id, out uint2 lo, out uint2 hi)
{
    lo = uint2(0, 0);
    hi = uint2(0, 0);
    if (id.x < offset.x || id.y < offset.y) {
        return false;
    }
    uint2 inside = id - offset;
    if (inside.x >= scaled.x || inside.y >= scaled.y) {
        return false;
    }
    lo = inside * source / scaled;
    hi = ((inside + 1) * source + scaled - 1) / scaled;
    hi = max(min(hi, source), lo + 1);
    return true;
}

// Where shown pixel (x, y) of the rectangle is in the picture as stored.
uint2 stored(uint x, uint y)
{
    int3 at = int3(int(x), int(y), 1);
    return origin + uint2(uint(dot(x_of.xyz, at)), uint(dot(y_of.xyz, at)));
}

#define FIT_ROOT_SIGNATURE \
    "RootConstants(num32BitConstants = 40, b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 2)), " \
    "UAV(u0)"

[RootSignature(FIT_ROOT_SIGNATURE)]
[numthreads(16, 16, 1)]
void fit_nv12(uint3 thread_id : SV_DispatchThreadID)
{
    uint2 id = thread_id.xy;
    if (id.x >= model.x || id.y >= model.y) {
        return;
    }
    float3 rgb = float3(MARGIN, MARGIN, MARGIN);
    uint2 lo, hi;
    if (covered(id, lo, hi)) {
        float3 sum = float3(0.0, 0.0, 0.0);
        for (uint y = lo.y; y < hi.y; y++) {
            for (uint x = lo.x; x < hi.x; x++) {
                uint2 at = stored(x, y);
                sum += float3(luma.Load(int3(at, 0)), chroma.Load(int3(at / 2, 0)));
            }
        }
        float4 yuv = float4(sum / float((hi.x - lo.x) * (hi.y - lo.y)), 1.0);
        rgb = saturate(float3(dot(to_red, yuv), dot(to_green, yuv), dot(to_blue, yuv)));
    }
    uint plane = model.x * model.y;
    uint at = slot.x * 3 * plane + id.y * model.x + id.x;
    if (slot.y != 0) {
        rgb = rgb.bgr;
    }
    rgb = rgb * scale.rgb + bias.rgb;
    tensor[at] = rgb.r;
    tensor[plane + at] = rgb.g;
    tensor[2 * plane + at] = rgb.b;
}
