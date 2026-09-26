// `VulkanVideoEffect`'s kernel: every effect resolved into one colour matrix,
// an exponent, an opacity and a luma mask — `EffectParams` in
// `video_effect/options.rs`, which the D3D11 shader, the CUDA kernel and the
// software loop evaluate too. This is that definition, one line to a step.
//
// Writes the output in a BGRA frame's byte order into an RGBA image — see
// `platform::vulkan::bgra_pass`.

struct Effect {
    // The picture's width and height.
    size: vec4<u32>,
    // One row per output channel, red, green, blue: xyz weigh the input's
    // red, green and blue, w is the offset.
    r: vec4<f32>,
    g: vec4<f32>,
    b: vec4<f32>,
    // exponent, opacity, luma_low, luma_low_inv
    first: vec4<f32>,
    // luma_high, luma_high_inv
    second: vec4<f32>,
};

var<immediate> effect: Effect;

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var source: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= effect.size.x || id.y >= effect.size.y) {
        return;
    }
    let p = vec2<i32>(id.xy);
    let colour = textureLoad(source, p, 0);
    let exponent = effect.first.x;
    let opacity = effect.first.y;
    let luma_low = effect.first.z;
    let luma_low_inv = effect.first.w;
    let luma_high = effect.second.x;
    let luma_high_inv = effect.second.y;

    let luma = clamp(dot(colour.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    let mask = clamp((luma - luma_low) * luma_low_inv + 1.0, 0.0, 1.0)
        * clamp((luma_high - luma) * luma_high_inv + 1.0, 0.0, 1.0);

    var rgb = colour.rgb;
    if (exponent != 1.0) {
        rgb = pow(rgb, vec3<f32>(exponent));
    }
    let lifted = vec4<f32>(rgb, 1.0);
    let corrected = clamp(
        vec3<f32>(dot(effect.r, lifted), dot(effect.g, lifted), dot(effect.b, lifted)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let alpha = clamp(colour.a * opacity * mask, 0.0, 1.0);
    textureStore(output, p, vec4<f32>(corrected, alpha).bgra);
}
