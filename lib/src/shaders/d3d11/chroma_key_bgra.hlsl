// D3D11 chroma-key shader for D3d11ChromaKey — draws one BGRA input
// texture as a screen-covering triangle (the same vertex trick as
// composite_bgra.hlsl, and for the same reason: a full-target draw needs
// no vertex buffer) into an equally sized BGRA render target, passing RGB
// through untouched and replacing alpha with the keyed value.
//
// The band is handed over already resolved into `band_low`/
// `inv_band_width` rather than as the `threshold`/`smoothing` the caller
// set. That keeps the per-pixel work to one subtract, one multiply and a
// saturate, and — because a hard key arrives as an effectively infinite
// `inv_band_width` — reproduces SwChromaKey's un-feathered step exactly
// without a per-pixel branch or a division by zero. See
// `ChromaKeyConstants::new` for how the two are derived.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VertexOutput vs_main(uint vertex_id : SV_VertexID)
{
    VertexOutput output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(
        output.uv.x * 2.0 - 1.0,
        1.0 - output.uv.y * 2.0,
        0.0,
        1.0
    );
    return output;
}

cbuffer ChromaKeyBuffer : register(b0)
{
    float3 key_color;
    float band_low;
    float inv_band_width;
    float3 _padding;
    // The visible fraction of the input texture, for the same reason
    // composite_bgra.hlsl has one: a decoder pads its surfaces up to its
    // own alignment and that padding is not part of the picture.
    float2 uv_scale;
    float2 _uv_padding;
};

Texture2D<float4> bgra_texture : register(t0);
SamplerState key_sampler : register(s0);

float4 ps_chroma_key(VertexOutput input) : SV_Target
{
    float4 color = bgra_texture.Sample(key_sampler, input.uv * uv_scale);
    // Euclidean RGB distance to the key color, scaled so opposite corners
    // of the color cube are 1.0 — the same measure SwChromaKey's
    // `color_distance` computes on the CPU, so one ChromaKeyOptions means
    // the same thing to both backends.
    float distance = length(color.rgb - key_color) / sqrt(3.0);
    float alpha = saturate((distance - band_low) * inv_band_width);
    return float4(color.rgb, alpha);
}
