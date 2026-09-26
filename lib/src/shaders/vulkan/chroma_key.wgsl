// `VulkanChromaKey`'s kernel: the key colour's distance out of alpha, the
// colour passed through — what the D3D11 shader and the CUDA kernel compute,
// from the band `options::feather_band` resolves.
//
// Writes the output in a BGRA frame's byte order into an RGBA image — see
// `platform::vulkan::bgra_pass`.

struct Key {
    // The picture's width and height.
    size: vec4<u32>,
    // The key colour, and the band's low end.
    key: vec4<f32>,
    // The band's inverse width.
    band: vec4<f32>,
};

var<immediate> key: Key;

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var source: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= key.size.x || id.y >= key.size.y) {
        return;
    }
    let p = vec2<i32>(id.xy);
    let colour = textureLoad(source, p, 0);
    // Euclidean RGB distance to the key colour, scaled so opposite corners of
    // the colour cube are 1 — `SwChromaKey`'s `color_distance`.
    let distance = length(colour.rgb - key.key.rgb) / sqrt(3.0);
    let alpha = clamp((distance - key.key.w) * key.band.x, 0.0, 1.0);
    textureStore(output, p, vec4<f32>(colour.rgb, colour.a * alpha).bgra);
}
