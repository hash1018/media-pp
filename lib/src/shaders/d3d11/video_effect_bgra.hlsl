// D3D11 video-effect shader for D3d11VideoEffect — draws one BGRA input
// texture as a screen-covering triangle (the vertex trick composite_bgra.hlsl
// and chroma_key_bgra.hlsl use) into an equally sized BGRA render target.
//
// Every effect arrives already resolved into the one block below — a colour
// matrix, an exponent, an opacity and a luma mask — which is what the CPU
// element and the CUDA kernel evaluate too. See `EffectParams` in
// video_effect/options.rs for the definition; this is that definition, one
// line to a step.

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

cbuffer EffectBuffer : register(b0)
{
    // One row per output channel, red, green, blue: xyz weigh the input's
    // red, green and blue, w is the offset.
    float4 row_r;
    float4 row_g;
    float4 row_b;
    float exponent;
    float opacity;
    // The visible fraction of the input texture — a decoder pads its
    // surfaces, and the padding is not part of the picture.
    float2 uv_scale;
    float luma_low;
    float luma_low_inv;
    float luma_high;
    float luma_high_inv;
};

Texture2D<float4> bgra_texture : register(t0);
SamplerState effect_sampler : register(s0);

float4 ps_effect(VertexOutput input) : SV_Target
{
    float4 color = bgra_texture.Sample(effect_sampler, input.uv * uv_scale);

    float luma = saturate(dot(color.rgb, float3(0.2126, 0.7152, 0.0722)));
    float mask = saturate((luma - luma_low) * luma_low_inv + 1.0)
               * saturate((luma_high - luma) * luma_high_inv + 1.0);

    float3 rgb = color.rgb;
    if (exponent != 1.0)
    {
        rgb = pow(rgb, exponent);
    }
    float4 lifted = float4(rgb, 1.0);
    float3 corrected = saturate(float3(
        dot(row_r, lifted),
        dot(row_g, lifted),
        dot(row_b, lifted)
    ));
    return float4(corrected, saturate(color.a * opacity * mask));
}
