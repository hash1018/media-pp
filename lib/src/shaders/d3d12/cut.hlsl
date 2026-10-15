// `D3d12CutDetector`'s thumbnail: each cell of a plane summed, a thread a
// cell, as `cut::cell_means` sums it on the CPU — cell `c` of `n` along a
// side `length` long spans `[c * length / n, (c + 1) * length / n)`. Only the
// sums are made here: each is of whole numbers below 2^24, so exact whatever
// the order, and the CPU divides each by its count as it divides its own,
// which is what makes the two thumbnails the same float for float.
//
// The planes are read as integers: NV12's through R8_UINT and R8G8_UINT
// views, P010's through R16_UINT and R16G16_UINT, `shift` 8 keeping each
// sample's high byte, as the CPU reads it.
//
// The root signature: `Cells` as 8 root constants at b0, the luma and chroma
// planes' views at t0 and t1, and the sums as a root UAV at u0 — luma's
// cells from `at`, or chroma's, Cb then Cr of each cell, from `at`.

cbuffer Cells : register(b0)
{
    uint2 size;
    uint2 cells;
    uint shift;
    uint at;
    uint2 unused;
};

Texture2D<uint> luma : register(t0);
Texture2D<uint2> chroma : register(t1);
RWStructuredBuffer<float> sums : register(u0);

// Where the `index`th of `count` cells along a side `length` long starts.
uint edge(uint index, uint count, uint length)
{
    return index * length / count;
}

#define CELLS_ROOT_SIGNATURE \
    "RootConstants(num32BitConstants = 8, b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 2)), " \
    "UAV(u0)"

[RootSignature(CELLS_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void luma_sums(uint3 thread_id : SV_DispatchThreadID)
{
    uint id = thread_id.x;
    if (id >= cells.x * cells.y) {
        return;
    }
    uint cx = id % cells.x;
    uint cy = id / cells.x;
    uint x0 = edge(cx, cells.x, size.x);
    uint x1 = edge(cx + 1, cells.x, size.x);
    uint y0 = edge(cy, cells.y, size.y);
    uint y1 = edge(cy + 1, cells.y, size.y);
    float sum = 0.0;
    for (uint y = y0; y < y1; y++) {
        for (uint x = x0; x < x1; x++) {
            sum += float(luma.Load(int3(x, y, 0)) >> shift);
        }
    }
    sums[at + id] = sum;
}

[RootSignature(CELLS_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void chroma_sums(uint3 thread_id : SV_DispatchThreadID)
{
    uint id = thread_id.x;
    if (id >= cells.x * cells.y) {
        return;
    }
    uint cx = id % cells.x;
    uint cy = id / cells.x;
    uint x0 = edge(cx, cells.x, size.x);
    uint x1 = edge(cx + 1, cells.x, size.x);
    uint y0 = edge(cy, cells.y, size.y);
    uint y1 = edge(cy + 1, cells.y, size.y);
    float2 sum = float2(0.0, 0.0);
    for (uint y = y0; y < y1; y++) {
        for (uint x = x0; x < x1; x++) {
            sum += float2(chroma.Load(int3(x, y, 0)) >> shift);
        }
    }
    sums[at + 2 * id] = sum.x;
    sums[at + 2 * id + 1] = sum.y;
}
