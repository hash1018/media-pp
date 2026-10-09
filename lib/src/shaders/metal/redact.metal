// `MetalDetectionOverlay`'s hiding: a rectangle of a plane cut into cells,
// each painted the mean of the samples under it — a mosaic — or that mean
// blended into its neighbours' — a blur — or, cut to the ellipse inside the
// rectangle, only the samples whose centres are inside it; with a feather,
// mixed over what each sample was less toward the edge. The Metal
// counterpart of
// `REDACT_PTX`'s `cell_means` and `cell_paint`, and of `SwDetectionOverlay`'s
// `hide_plane`, written to give the same bytes: the planes are read and
// written as integers through `*Uint` views, the sums made in the CPU's order,
// and every operation rounded as written — compiled without fast math, the
// fusing of a multiply into an add turned off, the one division taken as
// `precise::divide`, and the result rounded half to even by `rint`. A fill
// cut to an ellipse is one cell whose mean is the fill's colour, written into
// the means rather than taken.
//
// Texture 0 is the plane: `r8Uint` for luma, `rg8Uint` for NV12's Cb and Cr,
// `rgba8Uint` for BGRA, every byte of a sample treated alike. Buffer 0 is the
// `Cells`, buffer 1 the means, a `float4` a cell.

#pragma METAL fp contract(off)

#include <metal_stdlib>
using namespace metal;

struct Cells {
    // The rectangle's top-left corner in the plane, and its size, in samples.
    uint2 origin;
    uint2 size;
    // How many cells across and down.
    uint2 cells;
    // x: 1 to blend each cell into its neighbours, 0 for a mosaic; y: 1 to
    // paint only the ellipse inside the rectangle.
    uint2 smooth;
    // x: how far up a value sits in its sample — 6 for P010's ten bits at
    // the top of sixteen, read as `r16Uint` and `rg16Uint`, the six below
    // written naught; 0 for bytes.
    uint2 shift;
    // x: how far in from the edge the cover fades, a fraction of the way
    // to the middle; 0 for a hard edge.
    float2 feather;
};

// The `index`th of `cells` stretches a side `length` long starts here.
static uint cell_edge(uint index, uint cells, uint length) {
    return uint(ulong(index) * ulong(length) / ulong(cells));
}

// Each cell's mean, a thread a cell.
kernel void cell_means(texture2d<uint, access::read> plane [[texture(0)]],
                       constant Cells &c [[buffer(0)]],
                       device float4 *means [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= c.cells.x * c.cells.y) {
        return;
    }
    uint cx = id % c.cells.x;
    uint cy = id / c.cells.x;
    uint x0 = cell_edge(cx, c.cells.x, c.size.x);
    uint x1 = cell_edge(cx + 1, c.cells.x, c.size.x);
    uint y0 = cell_edge(cy, c.cells.y, c.size.y);
    uint y1 = cell_edge(cy + 1, c.cells.y, c.size.y);
    float4 sum = float4(0.0);
    for (uint y = y0; y < y1; y++) {
        for (uint x = x0; x < x1; x++) {
            sum += float4(plane.read(c.origin + uint2(x, y)) >> c.shift.x);
        }
    }
    float reciprocal = precise::divide(1.0f, float((x1 - x0) * (y1 - y0)));
    means[id] = sum * reciprocal;
}

// Where a sample sits among the cells' centres along a side: the cell before
// it, the one after, and how far it is from the first's centre to the
// second's.
static float between(uint sample, uint cells, uint length, thread uint &first,
                     thread uint &second) {
    float last = float(cells - 1);
    float scaled = (float(sample) + 0.5f) * float(cells);
    float at = precise::divide(scaled, float(length)) - 0.5f;
    at = min(max(at, 0.0f), last);
    first = uint(floor(at));
    second = min(first + 1, cells - 1);
    return at - float(first);
}

// Each sample of the rectangle painted, a thread a sample.
kernel void cell_paint(texture2d<uint, access::read_write> plane [[texture(0)]],
                       constant Cells &c [[buffer(0)]],
                       device const float4 *means [[buffer(1)]],
                       uint2 id [[thread_position_in_grid]]) {
    if (id.x >= c.size.x || id.y >= c.size.y) {
        return;
    }
    // How much of the cover the sample takes, as the CPU's `cover` computes
    // it, each step rounded as it rounds it: inside the ellipse touching the
    // rectangle's sides where asked, or the rectangle, wholly from `feather`
    // of the way in, by a smooth step toward the edge, the rectangle's
    // corners rounded as far; outside, left as it is.
    float feather = c.feather.x;
    float alpha = 1.0f;
    if (c.smooth.y != 0 || feather > 0.0f) {
        float dx = precise::divide((float(id.x) + 0.5f) * 2.0f, float(c.size.x)) - 1.0f;
        float dy = precise::divide((float(id.y) + 0.5f) * 2.0f, float(c.size.y)) - 1.0f;
        float t;
        bool soft = true;
        if (c.smooth.y != 0) {
            float r2 = dx * dx + dy * dy;
            if (r2 > 1.0f) {
                return;
            }
            soft = feather > 0.0f;
            t = soft ? precise::divide(1.0f - precise::sqrt(r2), feather) : 1.0f;
        } else {
            float edge = 1.0f - feather;
            float u = precise::divide(max(abs(dx) - edge, 0.0f), feather);
            float v = precise::divide(max(abs(dy) - edge, 0.0f), feather);
            t = 1.0f - precise::sqrt(u * u + v * v);
        }
        if (soft) {
            if (t <= 0.0f) {
                return;
            }
            t = min(t, 1.0f);
            alpha = t * t * (3.0f - 2.0f * t);
        }
    }
    uint across = c.cells.x;
    float4 value;
    if (c.smooth.x != 0) {
        uint c0, c1, r0, r1;
        float t = between(id.x, across, c.size.x, c0, c1);
        float s = between(id.y, c.cells.y, c.size.y, r0, r1);
        float4 m00 = means[r0 * across + c0];
        float4 m01 = means[r0 * across + c1];
        float4 m10 = means[r1 * across + c0];
        float4 m11 = means[r1 * across + c1];
        float4 top = m00 + (m01 - m00) * t;
        float4 bottom = m10 + (m11 - m10) * t;
        value = top + (bottom - top) * s;
    } else {
        uint cx = ((id.x + 1) * across - 1) / c.size.x;
        uint cy = ((id.y + 1) * c.cells.y - 1) / c.size.y;
        value = means[cy * across + cx];
    }
    // Where the edge fades, the cover mixed over what was there.
    if (alpha < 1.0f) {
        float4 original = float4(plane.read(c.origin + id) >> c.shift.x);
        value = original + (value - original) * alpha;
    }
    plane.write(uint4(rint(value)) << c.shift.x, c.origin + id);
}
