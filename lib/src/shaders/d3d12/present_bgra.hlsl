// Pixel shader for a BGRA frame drawn straight from system memory — a screen
// capture's, say — uploaded into a B8G8R8A8 texture, so sampling it reads
// the channels in RGBA order. Alpha is dropped: a window has nothing behind
// it to blend with. The root signature is present_frame.hlsl's.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

Texture2D<float4> picture : register(t0);
SamplerState frame_sampler : register(s0);

float4 ps_bgra(VertexOutput input) : SV_Target
{
    return float4(picture.Sample(frame_sampler, input.uv).rgb, 1.0);
}
