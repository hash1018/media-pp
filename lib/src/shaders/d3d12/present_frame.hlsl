// The shared vertex stage and root signature every frame shader in this
// renderer draws with. Each pixel stage lives in its own translation unit
// (present_nv12.hlsl, present_yuv420p.hlsl, present_bgra.hlsl) so its
// texture registers are declared exactly once.
//
// The table holds a frame's planes — up to three, YUV420P's, matching the
// three-entry SRV heap the renderer allocates; NV12 uses two, BGRA one. The
// twelve root constants at b0 are the three rows the YUV shaders convert
// Y'CbCr with, set per frame from the frame's own colour description; the
// eight at b1 the two rows the vertex stage reads each point shown from the
// stored picture by, so that a phone's portrait recording, stored on its
// side, is shown upright (Orientation::uv_rows).
#define FRAME_ROOT_SIGNATURE \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
    "DescriptorTable(SRV(t0, numDescriptors=3))," \
    "RootConstants(num32BitConstants=12, b0)," \
    "RootConstants(num32BitConstants=8, b1)," \
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

cbuffer Turn : register(b1)
{
    float4 u_row;
    float4 v_row;
};

// Builds a screen-covering triangle without a vertex buffer.
[RootSignature(FRAME_ROOT_SIGNATURE)]
VertexOutput vs_main(uint vertex_id : SV_VertexID)
{
    VertexOutput output;
    float3 shown = float3((vertex_id << 1) & 2, vertex_id & 2, 1.0);
    output.uv = float2(dot(u_row.xyz, shown), dot(v_row.xyz, shown));
    output.position = float4(
        shown.x * 2.0 - 1.0,
        1.0 - shown.y * 2.0,
        0.0,
        1.0
    );
    return output;
}
