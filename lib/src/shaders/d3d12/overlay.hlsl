// `D3d12DetectionOverlay`'s paint: each box line, label band and label text
// painted onto the copy of an NV12 picture — what `overlay.metal`'s
// `paint_nv12` paints on Metal, `SwDetectionOverlay` in system memory.
//
// The copy's planes are UNORM views of textures of their own, luma at u0
// and Cb, Cr at u1, read and written; the copy is made into them, and from
// them into the frame handed on, by copies around these dispatches.
//
// The root signature: `Paint` as 12 root constants at b0, the planes' table
// at u0 and u1, and the coverage mask's at t0.

cbuffer Paint : register(b0)
{
    // The rectangle painted: its top-left corner, then its width and height,
    // one thread per pixel, on whole 2x2 blocks.
    uint4 region;
    // Y', Cb, Cr, each 0 to 1.
    float4 colour;
    // x: 1 where the mask gives each pixel's coverage, 0 to paint solid.
    uint4 masked;
};

RWTexture2D<unorm float> luma : register(u0);
RWTexture2D<unorm float2> chroma : register(u1);
Texture2D<float> mask : register(t0);

// How much of pixel `id` of the region is painted: all of it, or what the
// mask says.
float coverage(uint2 id)
{
    return masked.x != 0 ? mask.Load(int3(id, 0)) : 1.0;
}

#define PAINT_ROOT_SIGNATURE \
    "RootConstants(num32BitConstants = 12, b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 2)), " \
    "DescriptorTable(SRV(t0))"

// The colour moved toward over the rectangle by its coverage; a chroma
// sample covers a 2x2 block and takes the block's average coverage, as the
// other overlays' do.
[RootSignature(PAINT_ROOT_SIGNATURE)]
[numthreads(8, 8, 1)]
void paint_nv12(uint3 thread_id : SV_DispatchThreadID)
{
    uint2 id = thread_id.xy;
    if (id.x >= region.z || id.y >= region.w) {
        return;
    }
    uint2 p = region.xy + id;
    float a = coverage(id);
    luma[p] = lerp(luma[p], colour.x, a);
    if (id.x % 2 == 0 && id.y % 2 == 0) {
        float block = (a + coverage(id + uint2(1, 0)) + coverage(id + uint2(0, 1)) +
                       coverage(id + uint2(1, 1))) / 4.0;
        chroma[p / 2] = lerp(chroma[p / 2], colour.yz, block);
    }
}
