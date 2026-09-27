// `VulkanScaler`'s kernels: a picture resampled to another size one plane
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
// negative lobes among them, go into the down pass as they came out.

struct Scale {
    // The source plane's width and height, and the output plane's.
    size: vec4<u32>,
    // The kernel: 0 nearest, 1 bilinear, 2 bicubic, 3 Lanczos.
    kernel: vec4<u32>,
};

var<immediate> scale: Scale;

@group(0) @binding(0) var between_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var between_in: texture_storage_2d<rgba16float, read>;
@group(0) @binding(3) var out_r: texture_storage_2d<r8unorm, write>;
@group(0) @binding(4) var out_rg: texture_storage_2d<rg8unorm, write>;
@group(0) @binding(5) var out_rgba: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265358979;

// How far either side of its centre the kernel reaches, in source samples,
// at a ratio of one.
fn support() -> f32 {
    switch scale.kernel.x {
        case 0u: { return 0.5; }
        case 1u: { return 1.0; }
        case 2u: { return 2.0; }
        default: { return 3.0; }
    }
}

fn weight(x: f32) -> f32 {
    let a = abs(x);
    switch scale.kernel.x {
        // A box one sample wide: the nearest, or where shrinking widens it,
        // the average of what it covers.
        case 0u: { return select(0.0, 1.0, a <= 0.5); }
        case 1u: { return max(1.0 - a, 0.0); }
        // Catmull-Rom.
        case 2u: {
            if (a < 1.0) {
                return (1.5 * a - 2.5) * a * a + 1.0;
            }
            if (a < 2.0) {
                return ((-0.5 * a + 2.5) * a - 4.0) * a + 2.0;
            }
            return 0.0;
        }
        // Lanczos, three lobes.
        default: {
            if (a < 1e-5) {
                return 1.0;
            }
            if (a >= 3.0) {
                return 0.0;
            }
            let p = PI * a;
            return 3.0 * sin(p) * sin(p / 3.0) / (p * p);
        }
    }
}

// Where output sample `at` of `out_len` sits among the `in_len` source
// samples, and how much wider than one sample the kernel is spread there.
struct Footprint {
    centre: f32,
    stretch: f32,
    first: i32,
    last: i32,
};

fn footprint(at: u32, in_len: u32, out_len: u32) -> Footprint {
    let ratio = f32(in_len) / f32(out_len);
    let stretch = max(ratio, 1.0);
    let centre = (f32(at) + 0.5) * ratio - 0.5;
    let reach = support() * stretch;
    return Footprint(centre, stretch, i32(ceil(centre - reach)), i32(floor(centre + reach)));
}

@compute @workgroup_size(8, 8)
fn across(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= scale.size.z || id.y >= scale.size.y) {
        return;
    }
    let f = footprint(id.x, scale.size.x, scale.size.z);
    let edge = i32(scale.size.x) - 1;
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = f.first; i <= f.last; i++) {
        let w = weight((f32(i) - f.centre) / f.stretch);
        sum += w * textureLoad(source, vec2<i32>(clamp(i, 0, edge), i32(id.y)), 0);
        total += w;
    }
    textureStore(between_out, vec2<i32>(id.xy), sum / total);
}

fn down(id: vec2<u32>) -> vec4<f32> {
    let f = footprint(id.y, scale.size.y, scale.size.w);
    let edge = i32(scale.size.y) - 1;
    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = f.first; i <= f.last; i++) {
        let w = weight((f32(i) - f.centre) / f.stretch);
        sum += w * textureLoad(between_in, vec2<i32>(i32(id.x), clamp(i, 0, edge)));
        total += w;
    }
    return clamp(sum / total, vec4<f32>(0.0), vec4<f32>(1.0));
}

@compute @workgroup_size(8, 8)
fn down_r(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= scale.size.z || id.y >= scale.size.w) {
        return;
    }
    textureStore(out_r, vec2<i32>(id.xy), down(id.xy));
}

@compute @workgroup_size(8, 8)
fn down_rg(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= scale.size.z || id.y >= scale.size.w) {
        return;
    }
    textureStore(out_rg, vec2<i32>(id.xy), down(id.xy));
}

// Into an RGBA image in a BGRA frame's byte order — see
// `platform::vulkan::bgra_pass`.
@compute @workgroup_size(8, 8)
fn down_bgra(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= scale.size.z || id.y >= scale.size.w) {
        return;
    }
    textureStore(out_rgba, vec2<i32>(id.xy), down(id.xy).bgra);
}
