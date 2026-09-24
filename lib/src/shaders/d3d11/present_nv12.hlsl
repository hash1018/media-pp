// D3d11WindowRenderer's NV12 pixel shader — the format D3d11Decoder/an NV12-fed
// D3d11Upload produce. Reuses present_bgra.hlsl's vs_main (a separate D3D11
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

// Three affine rows from the frame's own colour description — its matrix
// and range — written per frame by the renderer (`color::yuv_to_rgb_rows`).
// Each takes the samples as they are, (Y, Cb, Cr, 1) in 0..1, offsets
// included, so a BT.709 frame and a BT.601 one are each converted with
// their own coefficients rather than one of them with the other's.
cbuffer Colour : register(b0)
{
    float4 to_red;
    float4 to_green;
    float4 to_blue;
};

float4 ps_nv12(VertexOutput input) : SV_Target
{
    float4 ycbcr = float4(
        luma.Sample(frame_sampler, input.uv).r,
        chroma.Sample(frame_sampler, input.uv).rg,
        1.0);
    float3 rgb = float3(dot(to_red, ycbcr), dot(to_green, ycbcr), dot(to_blue, ycbcr));
    return float4(saturate(rgb), 1.0);
}
