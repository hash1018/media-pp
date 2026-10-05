// The kernels `MetalOrtDetector` and `MetalOrtClassifier` fit a picture into
// a model's input with, the Metal counterpart of `fit_nv12` and `fit_bgra` in
// `platform::cuda::driver::ptx::FIT_PTX`: each thread one pixel of the
// model's `width` by `height` input, written to three float planes of input
// `slot` of buffer 1 — R, then G, then B, each 0 to 1 and then times `scale`
// plus `bias`, as the model reads them.
//
// What is fitted is the rectangle of the picture at `origin`, `source` in
// size — the whole of it for a detector, an object's box for a classifier —
// scaled inside the input at `offset` to `scaled` in size; a pixel outside it
// is the grey Ultralytics trains on, 114 of 255. Inside, each pixel is the
// mean of the source pixels it covers — three by three where a 1080p picture
// is shrunk to 640 — or, enlarging, of the one it falls in, as the CUDA
// kernels make it: the sample at its centre alone, one pixel in nine, lost
// thin objects there. An NV12 pixel is averaged as Y', Cb and Cr — each
// source pixel with the chroma sample it shares — and made RGB by the three
// rows its own colour description gives — `color::yuv_to_rgb_rows`, what
// `convert.metal` converts with; a BGRA one is averaged as it is, read
// through a `bgra8Unorm` view that hands it over as RGBA.

#include <metal_stdlib>
using namespace metal;

struct Fit {
    // The model's input size.
    uint2 model;
    // Where the scaled picture's top-left corner sits in the input.
    uint2 offset;
    // The size the picture is scaled to inside the input.
    uint2 scaled;
    // The size of the rectangle of the picture fitted.
    uint2 source;
    // Where that rectangle's top-left corner is in the picture.
    uint2 origin;
    // x: which input of the buffer, from 0.
    uint2 slot;
    // NV12 only: the rows that make a `(Y', Cb, Cr, 1)` sample RGB.
    float4 r;
    float4 g;
    float4 b;
    // What each channel, 0 to 1, is multiplied by and then added to.
    float4 scale;
    float4 bias;
};

constant float MARGIN = 114.0 / 255.0;

// The source pixels input pixel `id` covers, `[lo, hi)` in the picture —
// `d * source / scaled` to `(d + 1) * source / scaled` rounded up, at least
// one pixel and none past the rectangle — or false where it is in the
// margin.
static bool covered(constant Fit &fit, uint2 id, thread uint2 &lo, thread uint2 &hi) {
    if (id.x < fit.offset.x || id.y < fit.offset.y) {
        return false;
    }
    uint2 inside = id - fit.offset;
    if (inside.x >= fit.scaled.x || inside.y >= fit.scaled.y) {
        return false;
    }
    uint2 first = inside * fit.source / fit.scaled;
    uint2 last = ((inside + 1) * fit.source + fit.scaled - 1) / fit.scaled;
    last = max(min(last, fit.source), first + 1);
    lo = fit.origin + first;
    hi = fit.origin + last;
    return true;
}

static void store(device float *tensor, constant Fit &fit, uint2 id, float3 rgb) {
    uint plane = fit.model.x * fit.model.y;
    uint at = fit.slot.x * 3 * plane + id.y * fit.model.x + id.x;
    rgb = rgb * fit.scale.rgb + fit.bias.rgb;
    tensor[at] = rgb.r;
    tensor[plane + at] = rgb.g;
    tensor[2 * plane + at] = rgb.b;
}

kernel void fit_nv12(texture2d<float, access::read> luma [[texture(0)]],
                     texture2d<float, access::read> chroma [[texture(1)]],
                     constant Fit &fit [[buffer(0)]],
                     device float *tensor [[buffer(1)]],
                     uint2 id [[thread_position_in_grid]]) {
    if (id.x >= fit.model.x || id.y >= fit.model.y) {
        return;
    }
    float3 rgb = float3(MARGIN);
    uint2 lo, hi;
    if (covered(fit, id, lo, hi)) {
        float3 sum = float3(0.0);
        for (uint y = lo.y; y < hi.y; y++) {
            for (uint x = lo.x; x < hi.x; x++) {
                uint2 at = uint2(x, y);
                sum += float3(luma.read(at).r, chroma.read(at / 2).rg);
            }
        }
        float4 yuv = float4(sum / float((hi.x - lo.x) * (hi.y - lo.y)), 1.0);
        rgb = clamp(float3(dot(fit.r, yuv), dot(fit.g, yuv), dot(fit.b, yuv)),
                    float3(0.0), float3(1.0));
    }
    store(tensor, fit, id, rgb);
}

kernel void fit_bgra(texture2d<float, access::read> pixels [[texture(0)]],
                     constant Fit &fit [[buffer(0)]],
                     device float *tensor [[buffer(1)]],
                     uint2 id [[thread_position_in_grid]]) {
    if (id.x >= fit.model.x || id.y >= fit.model.y) {
        return;
    }
    float3 rgb = float3(MARGIN);
    uint2 lo, hi;
    if (covered(fit, id, lo, hi)) {
        float3 sum = float3(0.0);
        for (uint y = lo.y; y < hi.y; y++) {
            for (uint x = lo.x; x < hi.x; x++) {
                sum += pixels.read(uint2(x, y)).rgb;
            }
        }
        rgb = sum / float((hi.x - lo.x) * (hi.y - lo.y));
    }
    store(tensor, fit, id, rgb);
}
