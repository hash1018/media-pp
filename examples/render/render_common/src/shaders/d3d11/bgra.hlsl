// D3D11 vertex shader + BGRA pixel shader. No [RootSignature] attribute
// here (that's a D3D12-only concept) — D3D11 shaders are independent
// objects (CreateVertexShader/CreatePixelShader) bound directly at draw
// time via VSSetShader/PSSetShader, with no root-signature layer to keep
// in sync between them.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Builds a screen-covering triangle without a vertex buffer.
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

Texture2D<float4> bgra_texture : register(t0);
SamplerState frame_sampler : register(s0);

// Straight passthrough — a packed-BGRA desktop-capture surface is already
// RGB, no YUV conversion needed (unlike ps_nv12 below).
float4 ps_bgra(VertexOutput input) : SV_Target
{
    return bgra_texture.Sample(frame_sampler, input.uv);
}
