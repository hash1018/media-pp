// Pixel shader for zero-copy NV12 (semi-planar 4:2:0) submission — the
// format hardware video decode (e.g. D3D12VA) produces directly, as one
// resource with a full-resolution luma plane and a half-resolution
// interleaved-chroma plane, as opposed to the three separate planes
// `ps_yuv420p` (frame.hlsl) expects from a CPU-side YUV420P upload.
//
// Compiled as its own translation unit (own `D3DCompile` call, see
// `gpu_context.rs`) specifically so its `t0`/`t1` register declarations
// don't collide with frame.hlsl's `texture0`/`texture1`/`texture2` — the
// root signature is still the one extracted from frame.hlsl's `vs_main`
// (`FRAME_ROOT_SIGNATURE`: 3 contiguous SRVs at t0, static sampler s0),
// which this shader fits inside without needing its own copy of that
// attribute.

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
SamplerState frame_sampler : register(s0);

// Converts an NV12 texture pair directly to RGB on the GPU. Same BT.601
// limited-range assumption as `ps_yuv420p`.
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
