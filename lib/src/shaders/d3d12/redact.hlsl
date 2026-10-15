// `D3d12DetectionOverlay`'s hiding: a rectangle of a plane cut into cells,
// each painted the mean of the samples under it — a mosaic — or that mean
// blended into its neighbours' — a blur — or a fill's colour, over all of
// the rectangle or what its cut leaves, faded at the edge where asked. The
// D3D12 counterpart of `redact.metal` and of `SwDetectionOverlay`'s
// `hide_plane`, written to give the same bytes.
//
// What the CPU works out with a division or a square root — each cell's
// reciprocal, where each column and row of samples sits among the cells'
// centres, how much of the cover each sample takes — it works out itself
// and hands over in `table`, as HLSL's divisions and roots are not
// correctly rounded. What is left here is sums, products and differences,
// each rounded as written (`precise`), and the rounding half to even of
// `round`: the CPU's own operations in its own order. A cell's sum is one
// thread's, row by row, as the CPU sums it.
//
// The plane is read and written as integers, through an R8_UINT or
// R8G8_UINT view — or R16_UINT and R16G16_UINT for P010, `shift` 6 keeping
// the top ten bits of sixteen — its luma at u0 and chroma at u1; a kernel
// whose name ends in 1 reads the first, in 2 the second. `means` holds a
// `float4` a cell.
//
// The root signature: `Cells` as 16 root constants at b0, the planes'
// table at u0 and u1, `table` as a root SRV at t0 and `means` as a root
// UAV at u2.

cbuffer Cells : register(b0)
{
    // The rectangle's top-left corner in the plane, and its size, in
    // samples.
    uint2 origin;
    uint2 size;
    // How many cells across and down.
    uint2 cells;
    // 1 to blend each cell into its neighbours, 0 for a mosaic.
    uint smooth;
    // How far up a value sits in its sample: 6 for P010, 0 for bytes.
    uint shift;
    // Where in `means` this rectangle's cells start, a `float4` each.
    uint means_at;
    // Where in `table` each cell's reciprocal starts.
    uint reciprocals_at;
    // Where the columns' and the rows' places among the cells start, three
    // floats each — the cell before, the cell after, how far between — or
    // naught where the cells are not blended.
    uint columns_at;
    uint rows_at;
    // Where each sample's cover starts, row by row — below naught for one
    // left as it is — or 0xffffffff where every sample takes it whole.
    uint alpha_at;
    // Where a fill's colour is, a `float4` taken as every cell's mean, or
    // 0xffffffff for a mosaic or a blur.
    uint fill_at;
    uint2 unused;
};

RWTexture2D<uint> luma : register(u0);
RWTexture2D<uint2> chroma : register(u1);
StructuredBuffer<float> table : register(t0);
RWStructuredBuffer<float4> means : register(u2);

static const uint NONE = 0xffffffff;

// The `index`th of `count` cells along a side `length` long starts here.
uint cell_edge(uint index, uint count, uint length)
{
    return index * length / count;
}

// The bounds of cell `id`: left, right, top, bottom.
uint4 bounds(uint id)
{
    uint cx = id % cells.x;
    uint cy = id / cells.x;
    return uint4(cell_edge(cx, cells.x, size.x), cell_edge(cx + 1, cells.x, size.x),
                 cell_edge(cy, cells.y, size.y), cell_edge(cy + 1, cells.y, size.y));
}

// Sample `p` of the rectangle as its values: one channel or two.
float4 read1(uint2 p) { return float4(float(luma[origin + p] >> shift), 0.0, 0.0, 0.0); }
float4 read2(uint2 p) { return float4(float2(chroma[origin + p] >> shift), 0.0, 0.0); }

#define CELLS_ROOT_SIGNATURE \
    "RootConstants(num32BitConstants = 16, b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 2)), " \
    "SRV(t0), " \
    "UAV(u2)"

// Each cell's mean, a thread a cell: its samples summed row by row, times
// the reciprocal of their count.
[RootSignature(CELLS_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void cell_means1(uint3 thread_id : SV_DispatchThreadID)
{
    uint id = thread_id.x;
    if (id >= cells.x * cells.y) {
        return;
    }
    uint4 b = bounds(id);
    precise float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    for (uint y = b.z; y < b.w; y++) {
        for (uint x = b.x; x < b.y; x++) {
            sum += read1(uint2(x, y));
        }
    }
    precise float4 mean = sum * table[reciprocals_at + id];
    means[means_at + id] = mean;
}

[RootSignature(CELLS_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void cell_means2(uint3 thread_id : SV_DispatchThreadID)
{
    uint id = thread_id.x;
    if (id >= cells.x * cells.y) {
        return;
    }
    uint4 b = bounds(id);
    precise float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    for (uint y = b.z; y < b.w; y++) {
        for (uint x = b.x; x < b.y; x++) {
            sum += read2(uint2(x, y));
        }
    }
    precise float4 mean = sum * table[reciprocals_at + id];
    means[means_at + id] = mean;
}

// Cell `index`'s mean: its own, or the fill's colour.
float4 mean_of(uint index)
{
    [branch] if (fill_at != NONE) {
        return float4(table[fill_at], table[fill_at + 1], table[fill_at + 2],
                      table[fill_at + 3]);
    }
    return means[means_at + index];
}

// What sample `id` of the rectangle is painted, before it is faded: its
// cell's mean, or the four around it blended.
float4 cover_of(uint2 id)
{
    [branch] if (smooth != 0) {
        uint c0 = uint(table[columns_at + 3 * id.x]);
        uint c1 = uint(table[columns_at + 3 * id.x + 1]);
        float t = table[columns_at + 3 * id.x + 2];
        uint r0 = uint(table[rows_at + 3 * id.y]);
        uint r1 = uint(table[rows_at + 3 * id.y + 1]);
        float s = table[rows_at + 3 * id.y + 2];
        float4 m00 = mean_of(r0 * cells.x + c0);
        float4 m01 = mean_of(r0 * cells.x + c1);
        float4 m10 = mean_of(r1 * cells.x + c0);
        float4 m11 = mean_of(r1 * cells.x + c1);
        precise float4 top = m00 + (m01 - m00) * t;
        precise float4 bottom = m10 + (m11 - m10) * t;
        precise float4 value = top + (bottom - top) * s;
        return value;
    }
    uint cx = ((id.x + 1) * cells.x - 1) / size.x;
    uint cy = ((id.y + 1) * cells.y - 1) / size.y;
    return mean_of(cy * cells.x + cx);
}

// How much of the cover sample `id` takes: below naught for none. A branch,
// not `?:`, which reads both sides: `table` is a root view, unchecked, and
// read at `NONE` it faults the GPU.
float alpha_of(uint2 id)
{
    [branch] if (alpha_at == NONE) {
        return 1.0;
    }
    return table[alpha_at + id.y * size.x + id.x];
}

// Each sample of the rectangle painted, a thread a sample: where the edge
// fades, the cover mixed over what was there.
[RootSignature(CELLS_ROOT_SIGNATURE)]
[numthreads(8, 8, 1)]
void cell_paint1(uint3 thread_id : SV_DispatchThreadID)
{
    uint2 id = thread_id.xy;
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    float alpha = alpha_of(id);
    if (alpha < 0.0) {
        return;
    }
    precise float4 value = cover_of(id);
    if (alpha < 1.0) {
        float4 original = read1(id);
        value = original + (value - original) * alpha;
    }
    luma[origin + id] = uint(round(value.x)) << shift;
}

[RootSignature(CELLS_ROOT_SIGNATURE)]
[numthreads(8, 8, 1)]
void cell_paint2(uint3 thread_id : SV_DispatchThreadID)
{
    uint2 id = thread_id.xy;
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    float alpha = alpha_of(id);
    if (alpha < 0.0) {
        return;
    }
    precise float4 value = cover_of(id);
    if (alpha < 1.0) {
        float4 original = read2(id);
        value = original + (value - original) * alpha;
    }
    chroma[origin + id] = uint2(round(value.xy)) << shift;
}
