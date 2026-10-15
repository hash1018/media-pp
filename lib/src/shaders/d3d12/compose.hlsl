// D3d12VideoCompositor's drawing: the canvas filled, each video layer
// sampled from its picture straight into its place — scaled, cropped,
// converted to the canvas's colour and mixed at its opacity in one pass a
// plane — and each text layer's coverage mixed in its colour.
//
// The canvas is NV12 in BT.709 limited range, two planes this crate owns:
// luma (u0) and chroma at half the size each way (u1), read back where a
// layer is mixed, so the GPU has to load R8 and R8G8 through a typed UAV.
//
// Every kernel shares one root signature, so a list changes pipeline
// keeping what is bound. Constants, in luma pixels of the canvas where they
// are places:
//   region  x, y, width, height of what is drawn, every one even
//   place   a layer: where its scaled image's top-left is, and the source
//           texels a canvas pixel spans, across and down;
//           a text: where in its mask the region starts (xy)
//   crop    a layer: where in its picture the part drawn starts, in texels,
//           and one over the picture's texture's width and height
//   rows    a layer: the picture's (Y, Cb, Cr, 1) to the canvas's Y, Cb, Cr
//   tint    a layer: the part drawn's width and height in texels, and its
//           opacity (w); a text or the fill: its colour's Y, Cb, Cr, and
//           its opacity (w)

#define ROOT "RootConstants(num32BitConstants = 28, b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 2)), " \
    "DescriptorTable(SRV(t0, numDescriptors = 2)), " \
    "StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_LINEAR, " \
    "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, " \
    "addressW = TEXTURE_ADDRESS_CLAMP)"

cbuffer Constants : register(b0) {
    uint4 region;
    float4 place;
    float4 crop;
    float4 row_y;
    float4 row_cb;
    float4 row_cr;
    float4 tint;
};

RWTexture2D<float> canvas_luma : register(u0);
RWTexture2D<float2> canvas_chroma : register(u1);
// A layer's picture's planes; a text's coverage in the first.
Texture2D<float> source_luma : register(t0);
Texture2D<float2> source_chroma : register(t1);
SamplerState linear_clamp : register(s0);

// The picture's colour at `at`, in texels of its luma plane, kept inside
// the part drawn by `inset` texels each side, so the bilinear sample never
// reaches what was cropped off.
float3 picture_at(float2 at, float inset) {
    float2 low = crop.xy + inset;
    float2 high = crop.xy + tint.xy - inset;
    float2 uv = clamp(at, low, high) * crop.zw;
    return float3(source_luma.SampleLevel(linear_clamp, uv, 0),
                  source_chroma.SampleLevel(linear_clamp, uv, 0));
}

[RootSignature(ROOT)]
[numthreads(8, 8, 1)]
void fill(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= region.zw)) {
        return;
    }
    uint2 p = region.xy + id.xy;
    canvas_luma[p] = tint.x;
    if (((p.x | p.y) & 1) == 0) {
        canvas_chroma[p / 2] = tint.yz;
    }
}

[RootSignature(ROOT)]
[numthreads(8, 8, 1)]
void layer_luma(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= region.zw)) {
        return;
    }
    uint2 p = region.xy + id.xy;
    float2 at = crop.xy + (float2(p) + 0.5 - place.xy) * place.zw;
    float y = saturate(dot(row_y, float4(picture_at(at, 0.5), 1.0)));
    canvas_luma[p] = lerp(canvas_luma[p], y, tint.w);
}

[RootSignature(ROOT)]
[numthreads(8, 8, 1)]
void layer_chroma(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= region.zw / 2)) {
        return;
    }
    uint2 q = region.xy / 2 + id.xy;
    // The middle of the 2x2 block of canvas pixels this sample covers.
    float2 at = crop.xy + (float2(q) * 2.0 + 1.0 - place.xy) * place.zw;
    float4 yuv = float4(picture_at(at, 1.0), 1.0);
    float2 cbcr = saturate(float2(dot(row_cb, yuv), dot(row_cr, yuv)));
    canvas_chroma[q] = lerp(canvas_chroma[q], cbcr, tint.w);
}

[RootSignature(ROOT)]
[numthreads(8, 8, 1)]
void text_luma(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= region.zw)) {
        return;
    }
    uint2 p = region.xy + id.xy;
    int2 at = int2(uint2(place.xy) + id.xy);
    float alpha = source_luma.Load(int3(at, 0)) * tint.w;
    canvas_luma[p] = lerp(canvas_luma[p], tint.x, alpha);
}

[RootSignature(ROOT)]
[numthreads(8, 8, 1)]
void text_chroma(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= region.zw / 2)) {
        return;
    }
    uint2 q = region.xy / 2 + id.xy;
    int2 at = int2(uint2(place.xy) + id.xy * 2);
    // The block's coverage, as its four pixels have it on average.
    float coverage = (source_luma.Load(int3(at, 0)) + source_luma.Load(int3(at + int2(1, 0), 0))
        + source_luma.Load(int3(at + int2(0, 1), 0)) + source_luma.Load(int3(at + int2(1, 1), 0)))
        * 0.25;
    canvas_chroma[q] = lerp(canvas_chroma[q], tint.yz, coverage * tint.w);
}
