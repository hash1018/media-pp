// D3D11 layer-compositing pixel shader for D3d11VideoCompositor's NV12
// input path (D3d11Decoder/an NV12-fed D3d11Upload) — reuses
// composite_bgra.hlsl's vs_main (a separate D3D11 shader object, not a
// shared translation unit, so no register collision to worry about).
// YUV->RGB coefficients arrive per frame so BT.601/BT.709/BT.2020 and
// limited/full-range inputs do not get interpreted as one fixed format.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

cbuffer LayerBuffer : register(b0)
{
    float4 yuv_to_red;
    float4 yuv_to_green;
    float4 yuv_to_blue;
    float opacity;
    float3 _padding;
    float2 uv_scale;
    float2 _uv_padding;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
SamplerState layer_sampler : register(s0);

// NV12 has no native alpha channel — treated as fully opaque before
// `opacity` is applied, same as the CPU VideoCompositor's InputScaler
// converting every input to opaque BGRA via libswscale.
float4 ps_nv12(VertexOutput input) : SV_Target
{
    float2 source_uv = input.uv * uv_scale;
    float4 yuv = float4(
        luma.Sample(layer_sampler, source_uv).r,
        chroma.Sample(layer_sampler, source_uv).rg,
        1.0
    );
    float3 rgb = float3(
        dot(yuv, yuv_to_red),
        dot(yuv, yuv_to_green),
        dot(yuv, yuv_to_blue)
    );

    return float4(saturate(rgb), opacity);
}
