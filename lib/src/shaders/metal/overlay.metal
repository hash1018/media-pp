// `MetalDetectionOverlay`'s kernels: a copy of the picture, then each box
// line, label band and label text painted onto the copy — what
// `SwDetectionOverlay` paints in system memory and `CudaDetectionOverlay`
// with the CUDA driver.
//
// Textures 0 (and for NV12 1) are the copy's planes, read and written: luma
// as `r8Unorm` and Cb, Cr as `rg8Unorm`, or BGRA through a `bgra8Unorm` view
// that hands it over as RGBA. The copy kernels read the picture from the
// textures after them; the paint kernels read a coverage mask there.

#include <metal_stdlib>
using namespace metal;

// What one dispatch paints, bound at buffer 0.
struct Paint {
    // The rectangle painted: its top-left corner, then its width and
    // height, one thread per pixel. On whole 2x2 blocks for NV12.
    uint4 region;
    // The colour: R, G, B for BGRA; Y', Cb, Cr for NV12, each 0 to 1.
    float4 colour;
    // x: 1 where the mask gives each pixel's coverage, 0 to paint solid.
    uint4 masked;
};

// The whole of an NV12 picture copied: each thread a luma sample, and the
// thread at the top-left of each 2x2 block its chroma sample.
kernel void copy_nv12(texture2d<float, access::read_write> luma [[texture(0)]],
                      texture2d<float, access::read_write> chroma [[texture(1)]],
                      texture2d<float, access::read> from_luma [[texture(2)]],
                      texture2d<float, access::read> from_chroma [[texture(3)]],
                      constant Paint &paint [[buffer(0)]],
                      uint2 id [[thread_position_in_grid]]) {
    if (id.x >= paint.region.z || id.y >= paint.region.w) {
        return;
    }
    luma.write(from_luma.read(id), id);
    if (id.x % 2 == 0 && id.y % 2 == 0) {
        chroma.write(from_chroma.read(id / 2), id / 2);
    }
}

kernel void copy_bgra(texture2d<float, access::read_write> pixels [[texture(0)]],
                      texture2d<float, access::read> from [[texture(1)]],
                      constant Paint &paint [[buffer(0)]],
                      uint2 id [[thread_position_in_grid]]) {
    if (id.x >= paint.region.z || id.y >= paint.region.w) {
        return;
    }
    pixels.write(from.read(id), id);
}

// How much of pixel `id` of the region is painted: all of it, or what the
// mask says.
static float coverage(constant Paint &paint, texture2d<float, access::read> mask, uint2 id) {
    return paint.masked.x != 0 ? mask.read(id).r : 1.0;
}

// The colour moved toward over a rectangle of an NV12 picture by its
// coverage; a chroma sample covers a 2x2 block and takes the block's
// average coverage, as the other overlays' do.
kernel void paint_nv12(texture2d<float, access::read_write> luma [[texture(0)]],
                       texture2d<float, access::read_write> chroma [[texture(1)]],
                       texture2d<float, access::read> mask [[texture(2)]],
                       constant Paint &paint [[buffer(0)]],
                       uint2 id [[thread_position_in_grid]]) {
    if (id.x >= paint.region.z || id.y >= paint.region.w) {
        return;
    }
    uint2 p = paint.region.xy + id;
    float a = coverage(paint, mask, id);
    luma.write(float4(mix(luma.read(p).r, paint.colour.x, a)), p);
    if (id.x % 2 == 0 && id.y % 2 == 0) {
        float block = (a + coverage(paint, mask, id + uint2(1, 0)) +
                       coverage(paint, mask, id + uint2(0, 1)) +
                       coverage(paint, mask, id + uint2(1, 1))) / 4.0;
        float2 under = chroma.read(p / 2).rg;
        chroma.write(float4(mix(under, paint.colour.yz, block), 0.0, 0.0), p / 2);
    }
}

// The same over a BGRA picture, whose alpha becomes at least the coverage.
kernel void paint_bgra(texture2d<float, access::read_write> pixels [[texture(0)]],
                       texture2d<float, access::read> mask [[texture(1)]],
                       constant Paint &paint [[buffer(0)]],
                       uint2 id [[thread_position_in_grid]]) {
    if (id.x >= paint.region.z || id.y >= paint.region.w) {
        return;
    }
    uint2 p = paint.region.xy + id;
    float a = coverage(paint, mask, id);
    float4 under = pixels.read(p);
    pixels.write(float4(mix(under.rgb, paint.colour.rgb, a), max(under.a, a)), p);
}
