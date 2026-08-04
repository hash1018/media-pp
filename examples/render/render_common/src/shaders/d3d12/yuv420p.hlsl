#define FRAME_ROOT_SIGNATURE \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
    "DescriptorTable(SRV(t0, numDescriptors=3))," \
    "StaticSampler(s0," \
        "filter=FILTER_MIN_MAG_LINEAR_MIP_POINT," \
        "addressU=TEXTURE_ADDRESS_CLAMP," \
        "addressV=TEXTURE_ADDRESS_CLAMP," \
        "addressW=TEXTURE_ADDRESS_CLAMP)"

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Builds a screen-covering triangle without a vertex buffer.
[RootSignature(FRAME_ROOT_SIGNATURE)]
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

Texture2D<float> texture0 : register(t0);
Texture2D<float> texture1 : register(t1);
Texture2D<float> texture2 : register(t2);
SamplerState frame_sampler : register(s0);

// Converts the three YUV420P planes directly to RGB on the GPU.
// The defaults assume BT.601 limited range, typical for standard SDR video.
[RootSignature(FRAME_ROOT_SIGNATURE)]
float4 ps_yuv420p(VertexOutput input) : SV_Target
{
    float y = texture0.Sample(frame_sampler, input.uv).r;
    float u = texture1.Sample(frame_sampler, input.uv).r - 0.5;
    float v = texture2.Sample(frame_sampler, input.uv).r - 0.5;

    y = 1.16438356 * (y - 16.0 / 255.0);

    float3 rgb;
    rgb.r = y + 1.59602678 * v;
    rgb.g = y - 0.39176229 * u - 0.81296764 * v;
    rgb.b = y + 2.01723214 * u;

    return float4(saturate(rgb), 1.0);
}
