// D3D11 NV12 pixel shader — the format D3d11Decoder/an NV12-fed
// D3d11Upload produce. Reuses bgra.hlsl's vs_main (a separate D3D11
// shader object, not a shared translation unit — no register collision
// concern the way the D3D12 side's `d3d12/nv12.hlsl` documents, since
// D3D11 has no single root signature both shaders need to fit).
//
// A single DXGI_FORMAT_NV12 texture can have two different-format SRVs
// created on it directly — DXGI_FORMAT_R8_UNORM for the full-resolution
// luma plane, DXGI_FORMAT_R8G8_UNORM for the half-resolution interleaved
// chroma plane — no D3D12-style `PlaneSlice` needed; the SRV's own format
// alone tells the driver which plane to expose.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
SamplerState frame_sampler : register(s0);

// Converts an NV12 texture pair directly to RGB on the GPU. BT.601
// limited-range assumption, same as the D3D12 side's ps_nv12/ps_yuv420p.
float4 ps_nv12(VertexOutput input) : SV_Target
{
    float y = luma.Sample(frame_sampler, input.uv).r;
    float2 uv = chroma.Sample(frame_sampler, input.uv).rg - 0.5;

    y = 1.16438356 * (y - 16.0 / 255.0);

    float3 rgb;
    rgb.r = y + 1.59602678 * uv.y;
    rgb.g = y - 0.39176229 * uv.x - 0.81296764 * uv.y;
    rgb.b = y + 2.01723214 * uv.x;

    return float4(saturate(rgb), 1.0);
}
