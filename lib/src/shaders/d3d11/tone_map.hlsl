// D3D11 tone-map shader for D3d11ToneMap — draws one P010 or NV12 HDR
// input, PQ or HLG, as SDR BT.709 into an equally sized BGRA render target,
// with the screen-covering triangle video_effect_bgra.hlsl uses.
//
// This is core/tone_map.rs's definition, one line to a step; `ToneMap` there
// fills the buffer below and says why each step is what it is.

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

cbuffer ToneMapBuffer : register(b0)
{
    float4 yuv_to_red;
    float4 yuv_to_green;
    float4 yuv_to_blue;
    uint transfer;
    float source_peak_pq;
    float target_peak;
    float knee;
    float4 gamut_red;
    float4 gamut_green;
    float4 gamut_blue;
    // The part of the texture the visible picture covers: a decoder pads
    // its surfaces.
    float2 uv_scale;
    float2 _padding;
};

Texture2D<float> luma : register(t0);
Texture2D<float2> chroma : register(t1);
SamplerState point_sampler : register(s0);

static const float PQ_M1 = 2610.0 / 16384.0;
static const float PQ_M2 = 2523.0 / 4096.0 * 128.0;
static const float PQ_C1 = 3424.0 / 4096.0;
static const float PQ_C2 = 2413.0 / 4096.0 * 32.0;
static const float PQ_C3 = 2392.0 / 4096.0 * 32.0;
static const float HLG_A = 0.17883277;
static const float HLG_B = 1.0 - 4.0 * HLG_A;
static const float HLG_C = 0.5599107;
static const float SDR_WHITE_NITS = 203.0;
static const float HLG_PEAK_NITS = 1000.0;

float3 pq_to_nits(float3 signal)
{
    float3 power = pow(max(signal, 0.0), 1.0 / PQ_M2);
    return 10000.0 * pow(max(power - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * power), 1.0 / PQ_M1);
}

float nits_to_pq(float nits)
{
    float y = pow(max(nits / 10000.0, 0.0), PQ_M1);
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), PQ_M2);
}

float3 hlg_to_nits(float3 signal)
{
    float3 low = signal * signal / 3.0;
    float3 high = (exp((signal - HLG_C) / HLG_A) + HLG_B) / 12.0;
    float3 scene = signal <= 0.5 ? low : high;
    float luminance = dot(scene, float3(0.2627, 0.6780, 0.0593));
    return HLG_PEAK_NITS * pow(max(luminance, 1e-6), 0.2) * scene;
}

float eetf(float normalised)
{
    if (normalised < knee)
    {
        return normalised;
    }
    float t = (normalised - knee) / (1.0 - knee);
    float t2 = t * t;
    float t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * knee
        + (t3 - 2.0 * t2 + t) * (1.0 - knee)
        + (-2.0 * t3 + 3.0 * t2) * target_peak;
}

float4 ps_tone_map(VertexOutput input) : SV_Target
{
    float2 uv = input.uv * uv_scale;
    float4 yuv = float4(
        luma.Sample(point_sampler, uv).r,
        chroma.Sample(point_sampler, uv).rg,
        1.0
    );
    float3 signal = saturate(float3(
        dot(yuv, yuv_to_red),
        dot(yuv, yuv_to_green),
        dot(yuv, yuv_to_blue)
    ));

    float3 nits = transfer == 1 ? pq_to_nits(signal) : hlg_to_nits(signal);

    float3 bt709 = max(float3(
        dot(nits, gamut_red.xyz),
        dot(nits, gamut_green.xyz),
        dot(nits, gamut_blue.xyz)
    ), 0.0);

    float largest = max(bt709.r, max(bt709.g, bt709.b));
    float scale = 1.0;
    if (largest > 0.0 && knee < 1.0)
    {
        float normalised = min(nits_to_pq(largest) / source_peak_pq, 1.0);
        float brought = pq_to_nits(eetf(normalised) * source_peak_pq).r;
        scale = brought / largest;
    }

    float3 sdr = saturate(bt709 * scale / SDR_WHITE_NITS);
    return float4(pow(sdr, 1.0 / 2.2), 1.0);
}
