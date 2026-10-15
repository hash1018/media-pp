// What `ObjectTracker` runs to follow objects by how they look on D3D12
// pictures — the D3D12 counterpart of `dcf.metal` and of `DCF_PTX` in
// `platform::cuda::driver::ptx`, and the GPU half of `vision::track::dcf`,
// which the CPU runs for every other picture:
//
// - `dcf_sample` samples each job's 64 by 64 neighbourhood of the picture's
//   luma, between pixels, edges held, its logarithm;
// - `dcf_normalize` makes each zero in mean and one in energy and fades it
//   out by the Hann window, its sums made in group-shared memory;
// - `dcf_fft` transforms the batch, a pass over the rows and one over the
//   columns, the inverse unscaled as the CPU's and cuFFT's is;
// - `dcf_correlate` multiplies each spectrum by its object's filter,
//   numerator over denominator plus the regulariser;
// - `dcf_peak` finds each response's peak, to a fraction of a sample, and
//   its peak-to-sidelobe ratio;
// - `dcf_learn` takes a spectrum into its object's filter at a rate.
//
// A neighbourhood is 4096 complex values, `CELLS`, its rows one after
// another; a batch of them sits one after another in `spectra`.
//
// One root signature for all: four root constants at b0, whose meaning is
// each kernel's; the picture's luma view in a table at t0; the jobs, their
// slots, the window, the wanted response and the twiddles as root views at
// t1 to t5; and the spectra, the filters' numerators and denominators and
// the peaks as root views at u0 to u3.

static const uint SIDE = 64;
static const uint CELLS = SIDE * SIDE;
// The half-side of the area around the peak left out of the sidelobe.
static const int PEAK_EXCLUSION = 5;
// Threads in a group that reduces over one neighbourhood.
static const uint REDUCING = 256;

cbuffer Constants : register(b0)
{
    uint4 constants;
};

Texture2D<float> luma : register(t0);
StructuredBuffer<float4> jobs : register(t1);
StructuredBuffer<uint> slots : register(t2);
StructuredBuffer<float> window_values : register(t3);
StructuredBuffer<float2> target : register(t4);
StructuredBuffer<float2> twiddles : register(t5);
RWStructuredBuffer<float2> spectra : register(u0);
RWStructuredBuffer<float2> numerator : register(u1);
RWStructuredBuffer<float> denominator : register(u2);
RWStructuredBuffer<float4> peaks : register(u3);

#define DCF_ROOT_SIGNATURE \
    "RootConstants(num32BitConstants = 4, b0), " \
    "DescriptorTable(SRV(t0)), " \
    "SRV(t1), SRV(t2), SRV(t3), SRV(t4), SRV(t5), " \
    "UAV(u0), UAV(u1), UAV(u2), UAV(u3)"

// The brightness of pixel `at`, 0 to 255: the luma plane's, read through a
// UNORM view of eight or sixteen bits alike.
float brightness(uint2 at)
{
    return luma.Load(int3(at, 0)) * 255.0;
}

// The picture between its pixels at `at`, its pixel centres at the halves,
// edges held — as `Region::at` reads it.
float between(uint2 size, float2 at)
{
    float fx = clamp(at.x - 0.5, 0.0, float(size.x - 1));
    float fy = clamp(at.y - 0.5, 0.0, float(size.y - 1));
    uint x0 = uint(floor(fx));
    uint y0 = uint(floor(fy));
    uint x1 = min(x0 + 1, size.x - 1);
    uint y1 = min(y0 + 1, size.y - 1);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    float top = lerp(brightness(uint2(x0, y0)), brightness(uint2(x1, y0)), tx);
    float bottom = lerp(brightness(uint2(x0, y1)), brightness(uint2(x1, y1)), tx);
    return lerp(top, bottom, ty);
}

// constants: the picture's width and height, how many jobs, and where in
// the batch they start. A thread a sample, a group per 16 by 16 of a job.
[RootSignature(DCF_ROOT_SIGNATURE)]
[numthreads(16, 16, 1)]
void dcf_sample(uint3 id : SV_DispatchThreadID)
{
    uint count = constants.z;
    if (id.x >= SIDE || id.y >= SIDE || id.z >= count) {
        return;
    }
    float4 job = jobs[id.z];
    float2 at = float2(job.x + ((float(id.x) + 0.5) / float(SIDE) - 0.5) * job.z,
                       job.y + ((float(id.y) + 0.5) / float(SIDE) - 0.5) * job.w);
    float value = between(constants.xy, at);
    spectra[id.z * CELLS + id.y * SIDE + id.x] = float2(log(value + 1.0), 0.0);
}

groupshared float shared_values[REDUCING];
groupshared uint shared_cells[REDUCING];

// The sum of `value` over the group, which every thread is handed.
float group_sum(float value, uint t)
{
    shared_values[t] = value;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = REDUCING / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            shared_values[t] += shared_values[t + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float total = shared_values[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

// A group per job.
[RootSignature(DCF_ROOT_SIGNATURE)]
[numthreads(REDUCING, 1, 1)]
void dcf_normalize(uint3 group : SV_GroupID, uint t : SV_GroupIndex)
{
    uint base = group.x * CELLS;
    float sum = 0.0;
    for (uint i = t; i < CELLS; i += REDUCING) {
        sum += spectra[base + i].x;
    }
    float mean = group_sum(sum, t) / float(CELLS);
    float squares = 0.0;
    for (uint j = t; j < CELLS; j += REDUCING) {
        float d = spectra[base + j].x - mean;
        squares += d * d;
    }
    float energy = sqrt(group_sum(squares, t));
    float scale = energy > 1e-6 ? 1.0 / energy : 0.0;
    for (uint k = t; k < CELLS; k += REDUCING) {
        spectra[base + k] = float2((spectra[base + k].x - mean) * scale * window_values[k], 0.0);
    }
}

float2 times(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// constants: how many jobs, 1 to transform the columns or 0 the rows, and
// 1 for the inverse, unscaled. A thread per line — a row, or a column — of
// each job, transforming its 64 values by radix-2 butterflies; `twiddles`
// are `exp(-2πik/64)` for k under 32, conjugated for the inverse.
[RootSignature(DCF_ROOT_SIGNATURE)]
[numthreads(64, 1, 1)]
void dcf_fft(uint3 id : SV_DispatchThreadID)
{
    uint count = constants.x;
    if (id.x >= SIDE || id.y >= count) {
        return;
    }
    bool columns = constants.y != 0;
    bool inverse = constants.z != 0;
    uint base = id.y * CELLS;
    uint start = columns ? id.x : id.x * SIDE;
    uint stride = columns ? SIDE : 1;
    float2 points[SIDE];
    // In bit-reversed order, so the butterflies combine neighbours.
    for (uint i = 0; i < SIDE; i++) {
        points[reversebits(i) >> 26] = spectra[base + start + i * stride];
    }
    for (uint length = 2; length <= SIDE; length <<= 1) {
        uint half_length = length / 2;
        uint step = SIDE / length;
        for (uint at = 0; at < SIDE; at += length) {
            for (uint k = 0; k < half_length; k++) {
                float2 w = twiddles[k * step];
                if (inverse) {
                    w.y = -w.y;
                }
                float2 u = points[at + k];
                float2 v = times(points[at + k + half_length], w);
                points[at + k] = u + v;
                points[at + k + half_length] = u - v;
            }
        }
    }
    for (uint j = 0; j < SIDE; j++) {
        spectra[base + start + j * stride] = points[j];
    }
}

// constants: values in the batch — jobs times `CELLS` — and the
// regulariser's bits. A thread per value.
[RootSignature(DCF_ROOT_SIGNATURE)]
[numthreads(REDUCING, 1, 1)]
void dcf_correlate(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= constants.x) {
        return;
    }
    uint at = slots[i / CELLS] * CELLS + i % CELLS;
    float2 filter = numerator[at] / (denominator[at] + asfloat(constants.y));
    spectra[i] = times(filter, spectra[i]);
}

// constants: values in the batch and the rate's bits. A thread per value;
// no two jobs share a slot.
[RootSignature(DCF_ROOT_SIGNATURE)]
[numthreads(REDUCING, 1, 1)]
void dcf_learn(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= constants.x) {
        return;
    }
    float rate = asfloat(constants.y);
    uint cell = i % CELLS;
    uint at = slots[i / CELLS] * CELLS + cell;
    float2 f = spectra[i];
    float2 wanted = times(target[cell], float2(f.x, -f.y));
    numerator[at] = numerator[at] * (1.0 - rate) + wanted * rate;
    denominator[at] = denominator[at] * (1.0 - rate) + dot(f, f) * rate;
}

// Whether `a` is within the exclusion of `b`, the response wrapping round.
bool near(int a, int b)
{
    int d = (a - b) % int(SIDE);
    if (d < 0) {
        d += int(SIDE);
    }
    return min(d, int(SIDE) - d) <= PEAK_EXCLUSION;
}

// Where between `before` and `after` the top of a parabola through them and
// `top` is, from -0.5 to 0.5.
float parabola(float before, float top, float after)
{
    float curve = before - 2.0 * top + after;
    return abs(curve) < 1e-12 ? 0.0 : 0.5 * (before - after) / curve;
}

float response_at(uint base, int x, int y)
{
    uint wx = uint((x % int(SIDE) + int(SIDE)) % int(SIDE));
    uint wy = uint((y % int(SIDE) + int(SIDE)) % int(SIDE));
    return spectra[base + wy * SIDE + wx].x;
}

// A group per job: where its response peaks, to a fraction of a sample by a
// parabola through the peak and its neighbours either way, how high, and
// its peak-to-sidelobe ratio — `x, y, top, psr`.
[RootSignature(DCF_ROOT_SIGNATURE)]
[numthreads(REDUCING, 1, 1)]
void dcf_peak(uint3 group : SV_GroupID, uint t : SV_GroupIndex)
{
    uint base = group.x * CELLS;
    // The highest value, the first of equals.
    float best = -3.402823466e+38;
    uint cell = 0;
    for (uint i = t; i < CELLS; i += REDUCING) {
        float value = spectra[base + i].x;
        if (value > best) {
            best = value;
            cell = i;
        }
    }
    shared_values[t] = best;
    shared_cells[t] = cell;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = REDUCING / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            float other = shared_values[t + stride];
            uint other_cell = shared_cells[t + stride];
            if (other > shared_values[t]
                || (other == shared_values[t] && other_cell < shared_cells[t])) {
                shared_values[t] = other;
                shared_cells[t] = other_cell;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float top = shared_values[0];
    uint at = shared_cells[0];
    GroupMemoryBarrierWithGroupSync();
    int px = int(at % SIDE);
    int py = int(at / SIDE);

    float sum = 0.0;
    float squares = 0.0;
    float count = 0.0;
    for (uint j = t; j < CELLS; j += REDUCING) {
        int x = int(j % SIDE);
        int y = int(j / SIDE);
        if (near(x, px) && near(y, py)) {
            continue;
        }
        float value = spectra[base + j].x;
        sum += value;
        squares += value * value;
        count += 1.0;
    }
    sum = group_sum(sum, t);
    squares = group_sum(squares, t);
    count = group_sum(count, t);
    if (t != 0) {
        return;
    }
    float fx = float(px) + parabola(response_at(base, px - 1, py), top,
                                    response_at(base, px + 1, py));
    float fy = float(py) + parabola(response_at(base, px, py - 1), top,
                                    response_at(base, px, py + 1));
    float mean = sum / count;
    float deviation = sqrt(max(squares / count - mean * mean, 1e-12));
    peaks[group.x] = float4(fx, fy, top, (top - mean) / deviation);
}
