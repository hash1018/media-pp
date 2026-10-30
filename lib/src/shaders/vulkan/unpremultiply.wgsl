// `VulkanDmaBufSource`'s kernel: a buffer another process drew, as this
// pipeline's own BGRA — its colour divided by its alpha where the producer
// multiplied it in, its alpha made opaque where the producer left the byte
// meaningless, and as it is otherwise.
//
// Writes the output in a BGRA frame's byte order into an RGBA image — see
// `platform::vulkan::bgra_pass`.

struct Picture {
    // The picture's width and height, then what its fourth byte is: 0
    // straight alpha, 1 premultiplied, 2 nothing.
    size: vec4<u32>,
};

var<immediate> picture: Picture;

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var source: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= picture.size.x || id.y >= picture.size.y) {
        return;
    }
    let p = vec2<i32>(id.xy);
    var colour = textureLoad(source, p, 0);
    if (picture.size.z == 1u) {
        // A transparent pixel's colour is nothing; a colour above its own
        // alpha, which premultiplied colour cannot be, saturates.
        colour = vec4<f32>(
            select(vec3<f32>(0.0), min(colour.rgb / colour.a, vec3<f32>(1.0)), colour.a > 0.0),
            colour.a,
        );
    } else if (picture.size.z == 2u) {
        colour.a = 1.0;
    }
    textureStore(output, p, colour.bgra);
}
