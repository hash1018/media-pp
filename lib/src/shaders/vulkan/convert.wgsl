// `VulkanConverter`'s kernel: an NV12 picture as RGB, by the three affine
// rows its own colour description gives — `color::yuv_to_rgb_rows`, what the
// compositor and the renderers convert with. Each pixel reads its own luma
// and the chroma sample of the 2x2 block it is in.
//
// Writes the output in a BGRA frame's byte order into an RGBA image — see
// `platform::vulkan::bgra_pass`.

struct Convert {
    // The picture's width and height.
    size: vec4<u32>,
    r: vec4<f32>,
    g: vec4<f32>,
    b: vec4<f32>,
};

var<immediate> convert: Convert;

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var luma: texture_2d<f32>;
@group(0) @binding(2) var chroma: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= convert.size.x || id.y >= convert.size.y) {
        return;
    }
    let p = vec2<i32>(id.xy);
    let yuv = vec4<f32>(
        textureLoad(luma, p, 0).r,
        textureLoad(chroma, p / 2, 0).rg,
        1.0,
    );
    let rgb = clamp(
        vec3<f32>(dot(convert.r, yuv), dot(convert.g, yuv), dot(convert.b, yuv)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    textureStore(output, p, vec4<f32>(rgb, 1.0).bgra);
}
