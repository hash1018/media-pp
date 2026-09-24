// Pixel shader for a YUV420P frame drawn straight from system memory — what
// a software decode gives: luma, Cb and Cr each uploaded into a
// single-channel texture of its own, the chroma ones at half size. The root
// signature is present_frame.hlsl's; the table's three SRVs are the planes.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

Texture2D<float> luma : register(t0);
Texture2D<float> cb : register(t1);
Texture2D<float> cr : register(t2);
SamplerState frame_sampler : register(s0);

// The frame's own conversion rows — see present_nv12.hlsl.
cbuffer Colour : register(b0)
{
    float4 to_red;
    float4 to_green;
    float4 to_blue;
};

float4 ps_yuv420p(VertexOutput input) : SV_Target
{
    float4 ycbcr = float4(
        luma.Sample(frame_sampler, input.uv),
        cb.Sample(frame_sampler, input.uv),
        cr.Sample(frame_sampler, input.uv),
        1.0);
    float3 rgb = float3(dot(to_red, ycbcr), dot(to_green, ycbcr), dot(to_blue, ycbcr));
    return float4(saturate(rgb), 1.0);
}
