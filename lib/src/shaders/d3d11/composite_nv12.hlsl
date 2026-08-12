// D3D11 layer-compositing pixel shader for D3d11VideoCompositor's NV12
// input path (D3d11Decoder/an NV12-fed D3d11Upload) — reuses
// composite_bgra.hlsl's vs_main (a separate D3D11 shader object, not a
// shared translation unit, so no register collision to worry about).
// YUV->RGB math is copied verbatim from render_common's own
// shaders/d3d11/nv12.hlsl (proven-correct BT.601 limited-range conversion,
// not reinvented here).

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

cbuffer LayerBuffer : register(b0)
{
    float opacity;
    float3 _padding;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
SamplerState layer_sampler : register(s0);

// NV12 has no native alpha channel — treated as fully opaque before
// `opacity` is applied, same as the CPU VideoCompositor's InputScaler
// converting every input to opaque BGRA via libswscale.
float4 ps_nv12(VertexOutput input) : SV_Target
{
    float y = luma.Sample(layer_sampler, input.uv).r;
    float2 uv = chroma.Sample(layer_sampler, input.uv).rg - 0.5;

    y = 1.16438356 * (y - 16.0 / 255.0);

    float3 rgb;
    rgb.r = y + 1.59602678 * uv.y;
    rgb.g = y - 0.39176229 * uv.x - 0.81296764 * uv.y;
    rgb.b = y + 2.01723214 * uv.x;

    return float4(saturate(rgb), opacity);
}
