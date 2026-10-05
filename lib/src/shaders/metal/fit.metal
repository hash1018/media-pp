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
// is the grey Ultralytics trains on, 114 of 255. Inside, each pixel takes the
// source sample at its centre — nearest rather than filtered, as the CUDA
// kernels do. An NV12 sample is made RGB by the three rows its own colour
// description gives — `color::yuv_to_rgb_rows`, what `convert.metal`
// converts with; a BGRA one is read as it is, through a `bgra8Unorm` view
// that hands it over as RGBA.

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

// Where in the picture input pixel `id` takes its sample, or false where it
// is in the margin.
static bool sample_of(constant Fit &fit, uint2 id, thread uint2 &at) {
    if (id.x < fit.offset.x || id.y < fit.offset.y) {
        return false;
    }
    uint2 inside = id - fit.offset;
    if (inside.x >= fit.scaled.x || inside.y >= fit.scaled.y) {
        return false;
    }
    float2 centre = (float2(inside) + 0.5) * float2(fit.source) / float2(fit.scaled);
    at = fit.origin + min(uint2(centre), fit.source - 1);
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
    uint2 at;
    if (sample_of(fit, id, at)) {
        float4 yuv = float4(luma.read(at).r, chroma.read(at / 2).rg, 1.0);
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
    uint2 at;
    if (sample_of(fit, id, at)) {
        rgb = pixels.read(at).rgb;
    }
    store(tensor, fit, id, rgb);
}
