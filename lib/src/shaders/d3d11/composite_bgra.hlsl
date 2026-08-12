// D3D11 layer-compositing shader for D3d11VideoCompositor's BGRA input
// path — draws one input texture as a screen-covering triangle (same
// vertex trick as render_common's shaders/d3d11/bgra.hlsl) into whatever
// viewport/scissor rect the caller set for this layer's on-canvas
// footprint; no vertex buffer needed, since a rectangle anywhere on the
// render target is just a different viewport over the same "fullscreen"
// triangle. `opacity` (per-draw constant buffer) is this layer's own
// runtime opacity. Kept as its own file (not sharing register declarations
// with composite_nv12.hlsl) for the same readability reasons
// render_common's bgra.hlsl/nv12.hlsl are split, even though D3D11 (unlike
// D3D12) has no single root signature that would force it.

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

cbuffer LayerBuffer : register(b0)
{
    float opacity;
    float3 _padding;
};

Texture2D<float4> bgra_texture : register(t0);
SamplerState layer_sampler : register(s0);

// A packed-BGRA layer (e.g. DxgiCaptureSource's GPU mode) is already RGB —
// no YUV conversion needed, same reasoning as render_common's own ps_bgra.
float4 ps_bgra(VertexOutput input) : SV_Target
{
    float4 color = bgra_texture.Sample(layer_sampler, input.uv);
    return float4(color.rgb, color.a * opacity);
}
