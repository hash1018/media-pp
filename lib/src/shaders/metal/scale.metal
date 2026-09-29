// `MetalScaler`'s kernels, the Metal counterpart of
// `shaders/vulkan/scale.wgsl`: a picture resampled to another size one plane
// at a time, in two passes — across, from the source into `between`, then
// down, from `between` into the output — each output sample the normalised,
// weighted sum of the source samples under its kernel.
//
// Shrinking widens the kernel by the ratio, so that every source sample
// counts towards some output one. Without it a kernel samples the source
// between its samples and skips the rest, and fine detail — a line of
// text, a checkerboard — comes out as whichever samples it happened to
// land on, shimmering from frame to frame as anything moves.
//
// `between` is kept as 16-bit floats: the across pass's sums, Lanczos's
// negative lobes among them, go into the down pass as they came out. Each
// plane is read and written through a view of its own format — `r8Unorm`,
// `rg8Unorm` or `bgra8Unorm` — so one kernel serves every plane, the view
// putting the channels where the plane keeps them.

#include <metal_stdlib>
using namespace metal;

struct Scale {
    // The source plane's width and height, and the output plane's.
    uint4 size;
    // The kernel — `filter`, as `kernel` is a Metal keyword: 0 nearest,
    // 1 bilinear, 2 bicubic, 3 Lanczos.
    uint4 filter;
};

constant float PI = 3.14159265358979;

// How far either side of its centre the kernel reaches, in source samples,
// at a ratio of one.
static float support(constant Scale &scale) {
    switch (scale.filter.x) {
        case 0: return 0.5;
        case 1: return 1.0;
        case 2: return 2.0;
        default: return 3.0;
    }
}

static float weight(constant Scale &scale, float x) {
    float a = abs(x);
    switch (scale.filter.x) {
        // A box one sample wide: the nearest, or where shrinking widens it,
        // the average of what it covers.
        case 0: return a <= 0.5 ? 1.0 : 0.0;
        case 1: return max(1.0 - a, 0.0);
        // Catmull-Rom.
        case 2:
            if (a < 1.0) {
                return (1.5 * a - 2.5) * a * a + 1.0;
            }
            if (a < 2.0) {
                return ((-0.5 * a + 2.5) * a - 4.0) * a + 2.0;
            }
            return 0.0;
        // Lanczos, three lobes.
        default: {
            if (a < 1e-5) {
                return 1.0;
            }
            if (a >= 3.0) {
                return 0.0;
            }
            float p = PI * a;
            return 3.0 * sin(p) * sin(p / 3.0) / (p * p);
        }
    }
}

// Where output sample `at` of `out_len` sits among the `in_len` source
// samples, and how much wider than one sample the kernel is spread there.
struct Footprint {
    float centre;
    float stretch;
    int first;
    int last;
};

static Footprint footprint(constant Scale &scale, uint at, uint in_len, uint out_len) {
    float ratio = float(in_len) / float(out_len);
    float stretch = max(ratio, 1.0);
    float centre = (float(at) + 0.5) * ratio - 0.5;
    float reach = support(scale) * stretch;
    return Footprint{centre, stretch, int(ceil(centre - reach)), int(floor(centre + reach))};
}

kernel void across(texture2d<float, access::write> between [[texture(0)]],
                   texture2d<float, access::read> source [[texture(1)]],
                   constant Scale &scale [[buffer(0)]],
                   uint2 id [[thread_position_in_grid]]) {
    if (id.x >= scale.size.z || id.y >= scale.size.y) {
        return;
    }
    Footprint f = footprint(scale, id.x, scale.size.x, scale.size.z);
    int edge = int(scale.size.x) - 1;
    float4 sum = float4(0.0);
    float total = 0.0;
    for (int i = f.first; i <= f.last; i++) {
        float w = weight(scale, (float(i) - f.centre) / f.stretch);
        sum += w * source.read(uint2(clamp(i, 0, edge), id.y));
        total += w;
    }
    between.write(sum / total, id);
}

kernel void down(texture2d<float, access::read> between [[texture(0)]],
                 texture2d<float, access::write> output [[texture(1)]],
                 constant Scale &scale [[buffer(0)]],
                 uint2 id [[thread_position_in_grid]]) {
    if (id.x >= scale.size.z || id.y >= scale.size.w) {
        return;
    }
    Footprint f = footprint(scale, id.y, scale.size.y, scale.size.w);
    int edge = int(scale.size.y) - 1;
    float4 sum = float4(0.0);
    float total = 0.0;
    for (int i = f.first; i <= f.last; i++) {
        float w = weight(scale, (float(i) - f.centre) / f.stretch);
        sum += w * between.read(uint2(id.x, clamp(i, 0, edge)));
        total += w;
    }
    output.write(clamp(sum / total, float4(0.0), float4(1.0)), id);
}
