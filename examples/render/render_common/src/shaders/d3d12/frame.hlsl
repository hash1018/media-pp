// The shared vertex stage and root signature every frame shader in this
// renderer draws with. The pixel stage lives in its own translation unit
// (nv12.hlsl) so its texture registers are declared exactly once.
//
// The table is exactly NV12's luma/chroma pair, matching the two-entry
// SRV heap the renderer allocates.
#define FRAME_ROOT_SIGNATURE \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
    "DescriptorTable(SRV(t0, numDescriptors=2))," \
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
