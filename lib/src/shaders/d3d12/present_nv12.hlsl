// Pixel shader for zero-copy NV12 (semi-planar 4:2:0) submission — the
// format hardware video decode (e.g. D3D12VA) produces directly, and the
// one `D3d12Upload` writes for a CPU-decoded stream: a single resource
// with a full-resolution luma plane and a half-resolution
// interleaved-chroma plane.
//
// Compiled as its own translation unit (own `D3DCompile` call) so its
// texture registers are declared exactly once. The root signature is the
// one extracted from present_frame.hlsl's `vs_main` (`FRAME_ROOT_SIGNATURE`:
// the luma/chroma SRV pair at t0, the colour rows at b0, and a static
// sampler at s0), which this shader fits inside without needing its own copy
// of that attribute.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
SamplerState frame_sampler : register(s0);

// Three affine rows from the frame's own colour description — its matrix
// and range — set per frame by the renderer (`color::yuv_to_rgb_rows`).
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
