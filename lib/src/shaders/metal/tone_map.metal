// HDR brought into SDR BT.709 BGRA on Metal — core/tone_map.rs's definition,
// one line to a step, as tone_map.hlsl draws it for D3D11 and `hdr_to_bgra`
// launches it on CUDA; `ToneMap` there fills the buffer below and says why
// each step is what it is.
//
// Two kernels, one step for each pixel (`to_sdr`), differing only in where
// the textures are bound. A P010 picture's luma is read as `r16Unorm` and
// its chroma as `rg16Unorm`, each sample as it is stored; the BGRA picture
// is written, a thread a pixel.
//
// - `hdr_to_bgra`, the detectors' SDR copy: luma at texture 0, chroma at 1,
//   the output at 2.
// - `tone_map`, `MetalConverter`'s through `platform::macos::metal_pass`:
//   the output at texture 0, luma at 1, chroma at 2.

#include <metal_stdlib>
using namespace metal;

struct ToneMap {
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
    // The picture's size: its pixel buffers may be larger.
    uint2 size;
};

constant float PQ_M1 = 2610.0 / 16384.0;
constant float PQ_M2 = 2523.0 / 4096.0 * 128.0;
constant float PQ_C1 = 3424.0 / 4096.0;
constant float PQ_C2 = 2413.0 / 4096.0 * 32.0;
constant float PQ_C3 = 2392.0 / 4096.0 * 32.0;
constant float HLG_A = 0.17883277;
constant float HLG_B = 1.0 - 4.0 * HLG_A;
constant float HLG_C = 0.5599107;
constant float SDR_WHITE_NITS = 203.0;
constant float HLG_PEAK_NITS = 1000.0;

static float3 pq_to_nits(float3 signal) {
    float3 power = pow(max(signal, 0.0), 1.0 / PQ_M2);
    return 10000.0 * pow(max(power - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * power), 1.0 / PQ_M1);
}

static float nits_to_pq(float nits) {
    float y = pow(max(nits / 10000.0, 0.0), PQ_M1);
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), PQ_M2);
}

static float3 hlg_to_nits(float3 signal) {
    float3 low = signal * signal / 3.0;
    float3 high = (exp((signal - HLG_C) / HLG_A) + HLG_B) / 12.0;
    float3 scene = select(high, low, signal <= 0.5);
    float luminance = dot(scene, float3(0.2627, 0.6780, 0.0593));
    return HLG_PEAK_NITS * pow(max(luminance, 1e-6), 0.2) * scene;
}

static float eetf(constant ToneMap &map, float normalised) {
    if (normalised < map.knee) {
        return normalised;
    }
    float t = (normalised - map.knee) / (1.0 - map.knee);
    float t2 = t * t;
    float t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * map.knee
        + (t3 - 2.0 * t2 + t) * (1.0 - map.knee)
        + (-2.0 * t3 + 3.0 * t2) * map.target_peak;
}

// One pixel's Y'CbCr, normalised as read, to gamma-encoded SDR BT.709.
static float3 to_sdr(constant ToneMap &map, float4 yuv) {
    float3 signal = saturate(float3(
        dot(yuv, map.yuv_to_red),
        dot(yuv, map.yuv_to_green),
        dot(yuv, map.yuv_to_blue)));

    float3 nits = map.transfer == 1 ? pq_to_nits(signal) : hlg_to_nits(signal);

    float3 bt709 = max(float3(
        dot(nits, map.gamut_red.xyz),
        dot(nits, map.gamut_green.xyz),
        dot(nits, map.gamut_blue.xyz)), 0.0);

    float largest = max(bt709.r, max(bt709.g, bt709.b));
    float scale = 1.0;
    if (largest > 0.0 && map.knee < 1.0) {
        float normalised = min(nits_to_pq(largest) / map.source_peak_pq, 1.0);
        float brought = pq_to_nits(float3(eetf(map, normalised) * map.source_peak_pq)).r;
        scale = brought / largest;
    }

    float3 out = saturate(bt709 * scale / SDR_WHITE_NITS);
    return pow(out, 1.0 / 2.2);
}

kernel void hdr_to_bgra(texture2d<float, access::read> luma [[texture(0)]],
                        texture2d<float, access::read> chroma [[texture(1)]],
                        texture2d<float, access::write> sdr [[texture(2)]],
                        constant ToneMap &map [[buffer(0)]],
                        uint2 id [[thread_position_in_grid]]) {
    if (id.x >= map.size.x || id.y >= map.size.y) {
        return;
    }
    float4 yuv = float4(luma.read(id).r, chroma.read(id / 2).rg, 1.0);
    sdr.write(float4(to_sdr(map, yuv), 1.0), id);
}

kernel void tone_map(texture2d<float, access::write> sdr [[texture(0)]],
                     texture2d<float, access::read> luma [[texture(1)]],
                     texture2d<float, access::read> chroma [[texture(2)]],
                     constant ToneMap &map [[buffer(0)]],
                     uint2 id [[thread_position_in_grid]]) {
    if (id.x >= map.size.x || id.y >= map.size.y) {
        return;
    }
    float4 yuv = float4(luma.read(id).r, chroma.read(id / 2).rg, 1.0);
    sdr.write(float4(to_sdr(map, yuv), 1.0), id);
}
