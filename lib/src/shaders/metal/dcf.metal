// What `ObjectTracker` runs, with `metal`, to follow objects by how they
// look on VideoToolbox pictures — the Metal counterpart of `DCF_PTX` in
// `platform::cuda::driver::ptx`, and the GPU half of `vision::track::dcf`,
// which the CPU runs for every other picture:
//
// - `dcf_sample_nv12` and `dcf_sample_bgra` sample each job's 64 by 64
//   neighbourhood of the picture's brightness, between pixels, edges held,
//   its logarithm;
// - `dcf_normalize` makes each zero in mean and one in energy and fades it
//   out by the Hann window, its sums made in threadgroup memory;
// - `dcf_fft` transforms the batch, a pass over the rows and one over the
//   columns, the inverse unscaled as the CPU's and cuFFT's is;
// - `dcf_correlate` multiplies each spectrum by its object's filter,
//   numerator over denominator plus the regulariser;
// - `dcf_peak` finds each response's peak, to a fraction of a sample, and
//   its peak-to-sidelobe ratio;
// - `dcf_learn` takes a spectrum into its object's filter at a rate.
//
// A neighbourhood is 4096 complex values, `CELLS`, its rows one after
// another; a batch of them sits one after another in buffer 1.

#include <metal_stdlib>
using namespace metal;

constant uint SIDE = 64;
constant uint CELLS = SIDE * SIDE;
/// The half-side of the area around the peak left out of the sidelobe.
constant int PEAK_EXCLUSION = 5;

struct Sample {
    // The picture's width and height.
    uint2 size;
    // How many jobs.
    uint count;
};

/// The brightness of pixel `at`, 0 to 255: a luma plane's, or a BGRA
/// pixel's by BT.709's weights, as the CPU's reading of BGRA takes it — a
/// `bgra8Unorm` view hands the pixel over as RGBA.
static float brightness(texture2d<float, access::read> picture, uint2 at, bool bgra) {
    float4 pixel = picture.read(at);
    return (bgra ? dot(pixel.rgb, float3(0.2126f, 0.7152f, 0.0722f)) : pixel.r) * 255.0f;
}

/// The picture between its pixels at `(x, y)`, its pixel centres at the
/// halves, edges held — as `Region::at` reads it.
static float between(texture2d<float, access::read> picture, bool bgra, uint2 size, float2 at) {
    float fx = clamp(at.x - 0.5f, 0.0f, float(size.x - 1));
    float fy = clamp(at.y - 0.5f, 0.0f, float(size.y - 1));
    uint x0 = uint(floor(fx));
    uint y0 = uint(floor(fy));
    uint x1 = min(x0 + 1, size.x - 1);
    uint y1 = min(y0 + 1, size.y - 1);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    float top = mix(brightness(picture, uint2(x0, y0), bgra),
                    brightness(picture, uint2(x1, y0), bgra), tx);
    float bottom = mix(brightness(picture, uint2(x0, y1), bgra),
                       brightness(picture, uint2(x1, y1), bgra), tx);
    return mix(top, bottom, ty);
}

/// Where sample `cell` of `job` is in the picture.
static float2 sample_at(float4 job, uint2 cell) {
    return float2(job.x + ((float(cell.x) + 0.5f) / float(SIDE) - 0.5f) * job.z,
                  job.y + ((float(cell.y) + 0.5f) / float(SIDE) - 0.5f) * job.w);
}

kernel void dcf_sample_nv12(texture2d<float, access::read> luma [[texture(0)]],
                            constant Sample &p [[buffer(0)]],
                            device const float4 *jobs [[buffer(1)]],
                            device float2 *out [[buffer(2)]],
                            uint3 id [[thread_position_in_grid]]) {
    if (id.x >= SIDE || id.y >= SIDE || id.z >= p.count) {
        return;
    }
    float value = between(luma, false, p.size, sample_at(jobs[id.z], id.xy));
    out[id.z * CELLS + id.y * SIDE + id.x] = float2(log(value + 1.0f), 0.0f);
}

kernel void dcf_sample_bgra(texture2d<float, access::read> pixels [[texture(0)]],
                            constant Sample &p [[buffer(0)]],
                            device const float4 *jobs [[buffer(1)]],
                            device float2 *out [[buffer(2)]],
                            uint3 id [[thread_position_in_grid]]) {
    if (id.x >= SIDE || id.y >= SIDE || id.z >= p.count) {
        return;
    }
    float value = between(pixels, true, p.size, sample_at(jobs[id.z], id.xy));
    out[id.z * CELLS + id.y * SIDE + id.x] = float2(log(value + 1.0f), 0.0f);
}

/// The sum of `value` over the threadgroup, which every thread is handed.
static float group_sum(float value, threadgroup float *scratch, uint lane, uint simd,
                       uint simds) {
    value = simd_sum(value);
    if (lane == 0) {
        scratch[simd] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < simds; s++) {
        total += scratch[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// A threadgroup per job.
kernel void dcf_normalize(device float2 *data [[buffer(1)]],
                          device const float *window [[buffer(2)]],
                          uint job [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    device float2 *patch = data + job * CELLS;
    float sum = 0.0f;
    for (uint i = t; i < CELLS; i += threads) {
        sum += patch[i].x;
    }
    float mean = group_sum(sum, scratch, lane, simd, simds) / float(CELLS);
    float squares = 0.0f;
    for (uint i = t; i < CELLS; i += threads) {
        float d = patch[i].x - mean;
        squares += d * d;
    }
    float energy = sqrt(group_sum(squares, scratch, lane, simd, simds));
    float scale = energy > 1e-6f ? 1.0f / energy : 0.0f;
    for (uint i = t; i < CELLS; i += threads) {
        patch[i] = float2((patch[i].x - mean) * scale * window[i], 0.0f);
    }
}

struct Transform {
    // How many jobs.
    uint count;
    // 1 to transform the columns, 0 the rows.
    uint columns;
    // 1 for the inverse transform, unscaled.
    uint inverse;
};

static float2 times(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// A thread per line — a row, or a column — of each job, transforming its 64
// values by radix-2 butterflies; `twiddles` are `exp(-2πik/64)` for k under
// 32, conjugated for the inverse.
kernel void dcf_fft(constant Transform &p [[buffer(0)]],
                    device float2 *data [[buffer(1)]],
                    device const float2 *twiddles [[buffer(2)]],
                    uint2 id [[thread_position_in_grid]]) {
    if (id.x >= SIDE || id.y >= p.count) {
        return;
    }
    device float2 *base = data + id.y * CELLS;
    uint start = p.columns != 0 ? id.x : id.x * SIDE;
    uint stride = p.columns != 0 ? SIDE : 1;
    float2 line[SIDE];
    // In bit-reversed order, so the butterflies combine neighbours.
    for (uint i = 0; i < SIDE; i++) {
        line[reverse_bits(i) >> 26] = base[start + i * stride];
    }
    for (uint length = 2; length <= SIDE; length <<= 1) {
        uint half_length = length / 2;
        uint step = SIDE / length;
        for (uint i = 0; i < SIDE; i += length) {
            for (uint k = 0; k < half_length; k++) {
                float2 w = twiddles[k * step];
                if (p.inverse != 0) {
                    w.y = -w.y;
                }
                float2 u = line[i + k];
                float2 v = times(line[i + k + half_length], w);
                line[i + k] = u + v;
                line[i + k + half_length] = u - v;
            }
        }
    }
    for (uint i = 0; i < SIDE; i++) {
        base[start + i * stride] = line[i];
    }
}

struct Correlate {
    // Values in the batch: jobs times `CELLS`.
    uint total;
    float regulariser;
};

// A thread per value.
kernel void dcf_correlate(constant Correlate &p [[buffer(0)]],
                          device float2 *spectra [[buffer(1)]],
                          device const uint *slots [[buffer(2)]],
                          device const float2 *numerator [[buffer(3)]],
                          device const float *denominator [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.total) {
        return;
    }
    uint at = slots[i / CELLS] * CELLS + i % CELLS;
    float2 filter = numerator[at] / (denominator[at] + p.regulariser);
    spectra[i] = times(filter, spectra[i]);
}

struct Learn {
    // Values in the batch: jobs times `CELLS`.
    uint total;
    float rate;
};

// A thread per value; no two jobs share a slot.
kernel void dcf_learn(constant Learn &p [[buffer(0)]],
                      device const float2 *spectra [[buffer(1)]],
                      device const uint *slots [[buffer(2)]],
                      device float2 *numerator [[buffer(3)]],
                      device float *denominator [[buffer(4)]],
                      device const float2 *target [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= p.total) {
        return;
    }
    uint cell = i % CELLS;
    uint at = slots[i / CELLS] * CELLS + cell;
    float2 f = spectra[i];
    float2 wanted = times(target[cell], float2(f.x, -f.y));
    numerator[at] = numerator[at] * (1.0f - p.rate) + wanted * p.rate;
    denominator[at] = denominator[at] * (1.0f - p.rate) + dot(f, f) * p.rate;
}

/// Whether `a` is within the exclusion of `b`, the response wrapping round.
static bool near(int a, int b) {
    int d = (a - b) % int(SIDE);
    if (d < 0) {
        d += int(SIDE);
    }
    return min(d, int(SIDE) - d) <= PEAK_EXCLUSION;
}

/// Where between `before` and `after` the top of a parabola through them and
/// `top` is, from -0.5 to 0.5.
static float parabola(float before, float top, float after) {
    float curve = before - 2.0f * top + after;
    return fabs(curve) < 1e-12f ? 0.0f : 0.5f * (before - after) / curve;
}

static float response_at(device const float2 *response, int x, int y) {
    uint wx = uint((x % int(SIDE) + int(SIDE)) % int(SIDE));
    uint wy = uint((y % int(SIDE) + int(SIDE)) % int(SIDE));
    return response[wy * SIDE + wx].x;
}

// A threadgroup per job: where its response peaks, to a fraction of a
// sample by a parabola through the peak and its neighbours either way, how
// high, and its peak-to-sidelobe ratio — `x, y, top, psr`.
kernel void dcf_peak(device const float2 *responses [[buffer(1)]],
                     device float4 *peaks [[buffer(2)]],
                     uint job [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float best_values[32];
    threadgroup uint best_cells[32];
    threadgroup float scratch[32];
    device const float2 *response = responses + job * CELLS;

    // The highest value, the first of equals.
    float best = -INFINITY;
    uint cell = 0;
    for (uint i = t; i < CELLS; i += threads) {
        float value = response[i].x;
        if (value > best) {
            best = value;
            cell = i;
        }
    }
    for (ushort offset = 16; offset > 0; offset /= 2) {
        float other = simd_shuffle_down(best, offset);
        uint other_cell = simd_shuffle_down(cell, offset);
        if (other > best || (other == best && other_cell < cell)) {
            best = other;
            cell = other_cell;
        }
    }
    if (lane == 0) {
        best_values[simd] = best;
        best_cells[simd] = cell;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top = best_values[0];
    uint at = best_cells[0];
    for (uint s = 1; s < simds; s++) {
        if (best_values[s] > top || (best_values[s] == top && best_cells[s] < at)) {
            top = best_values[s];
            at = best_cells[s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int px = int(at % SIDE);
    int py = int(at / SIDE);

    float sum = 0.0f;
    float squares = 0.0f;
    float count = 0.0f;
    for (uint i = t; i < CELLS; i += threads) {
        int x = int(i % SIDE);
        int y = int(i / SIDE);
        if (near(x, px) && near(y, py)) {
            continue;
        }
        float value = response[i].x;
        sum += value;
        squares += value * value;
        count += 1.0f;
    }
    sum = group_sum(sum, scratch, lane, simd, simds);
    squares = group_sum(squares, scratch, lane, simd, simds);
    count = group_sum(count, scratch, lane, simd, simds);
    if (t != 0) {
        return;
    }
    float fx = float(px) + parabola(response_at(response, px - 1, py), top,
                                    response_at(response, px + 1, py));
    float fy = float(py) + parabola(response_at(response, px, py - 1), top,
                                    response_at(response, px, py + 1));
    float mean = sum / count;
    float deviation = sqrt(max(squares / count - mean * mean, 1e-12f));
    peaks[job] = float4(fx, fy, top, (top - mean) / deviation);
}
