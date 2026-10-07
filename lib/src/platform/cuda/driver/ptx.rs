//! The kernels [`super::CudaDriver`] loads, as PTX source the driver
//! JIT-compiles — kept apart from the Rust that calls them, which reads as
//! Rust rather than as eight hundred lines of assembly.

/// The BGRA-to-NV12 conversion nothing else on this crate's CUDA path can do.
///
/// `scale_cuda` resizes but does not convert — FFmpeg 8.1 answers a BGRA
/// input with "Unsupported conversion: bgra -> semiplanar8" — and
/// `colorspace_cuda` only moves YUV between ranges. Without this kernel a
/// BGRA surface can reach [`crate::elements::CudaEncoder`], which ingests it
/// directly, and nothing else: [`crate::elements::CudaVideoCompositor`] and
/// [`crate::elements::CudaRenderer`] both work in NV12.
///
/// It also carries `key_bgra`, which is a BGRA reader like the rest of this
/// module rather than a converter: it writes one frame's alpha from each
/// pixel's distance to a key colour, leaving RGB alone. A module of its own
/// would mean a third JIT and a third unload for one kernel that reads
/// exactly the surfaces these already do. The same goes for `effect_bgra`,
/// which is every `VideoEffect`: a colour matrix, an exponent, an opacity
/// and a luma mask, evaluated as `EffectParams`' own docs define them.
///
/// Two entry points for the conversion rather than one: luma is a thread per pixel, chroma a
/// thread per 2x2 block, and splitting them keeps each a straight line of
/// loads, arithmetic, and one store. The colour maths is BT.709 limited
/// range, deliberately the same definition [`rgb_to_bt709_limited`](super::rgb_to_bt709_limited) uses for
/// compositor backgrounds so a converted capture and a filled background
/// agree, and written in the same operation order so a test can compare every
/// byte against that expression instead of a tolerance. `nv12_to_bgra` goes
/// the other way by whatever matrix and range it is handed — see
/// `YuvToBgra` — and brings BT.2020 primaries into BT.709's where told to.
pub(super) const CONVERT_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry bgra_to_luma(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<8>;
    .reg .b32   %r<24>;
    .reg .f32   %f<20>;
    .reg .b64   %rd<12>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        LUMA_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        LUMA_DONE;

    mul.lo.s32      %r13, %r12, %r2;
    shl.b32         %r14, %r8, 2;
    add.s32         %r15, %r13, %r14;
    cvt.u64.u32     %rd3, %r15;
    add.s64         %rd4, %rd2, %rd3;

    ld.global.u8    %rs1, [%rd4];
    ld.global.u8    %rs2, [%rd4+1];
    ld.global.u8    %rs3, [%rd4+2];

    cvt.u32.u16     %r16, %rs1;
    cvt.rn.f32.u32  %f1, %r16;
    cvt.u32.u16     %r17, %rs2;
    cvt.rn.f32.u32  %f2, %r17;
    cvt.u32.u16     %r18, %rs3;
    cvt.rn.f32.u32  %f3, %r18;

    mul.f32         %f4, %f3, 0f3E59B3D0;
    mul.f32         %f5, %f2, 0f3F371759;
    add.f32         %f6, %f4, %f5;
    mul.f32         %f7, %f1, 0f3D93DD98;
    add.f32         %f8, %f6, %f7;

    mul.f32         %f9, %f8, 0f435B0000;
    div.rn.f32      %f10, %f9, 0f437F0000;
    add.f32         %f11, %f10, 0f41800000;
    max.f32         %f12, %f11, 0f00000000;
    min.f32         %f13, %f12, 0f437F0000;
    add.f32         %f14, %f13, 0f3F000000;
    cvt.rzi.u32.f32 %r19, %f14;
    cvt.u16.u32     %rs4, %r19;

    mul.lo.s32      %r20, %r12, %r1;
    add.s32         %r21, %r20, %r8;
    cvt.u64.u32     %rd5, %r21;
    add.s64         %rd6, %rd1, %rd5;
    st.global.u8    [%rd6], %rs4;

LUMA_DONE:
    ret;
}

.visible .entry bgra_to_chroma(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 half_width,
    .param .u32 half_height
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<20>;
    .reg .b32   %r<48>;
    .reg .f32   %f<32>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [half_width];
    ld.param.u32    %r4, [half_height];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        CHROMA_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        CHROMA_DONE;

    shl.b32         %r13, %r12, 1;
    shl.b32         %r14, %r8, 1;
    mul.lo.s32      %r15, %r13, %r2;
    shl.b32         %r16, %r14, 2;
    add.s32         %r17, %r15, %r16;
    cvt.u64.u32     %rd3, %r17;
    add.s64         %rd4, %rd2, %rd3;
    cvt.u64.u32     %rd5, %r2;
    add.s64         %rd6, %rd4, %rd5;

    ld.global.u8    %rs1, [%rd4];
    ld.global.u8    %rs2, [%rd4+1];
    ld.global.u8    %rs3, [%rd4+2];
    ld.global.u8    %rs4, [%rd4+4];
    ld.global.u8    %rs5, [%rd4+5];
    ld.global.u8    %rs6, [%rd4+6];
    ld.global.u8    %rs7, [%rd6];
    ld.global.u8    %rs8, [%rd6+1];
    ld.global.u8    %rs9, [%rd6+2];
    ld.global.u8    %rs10, [%rd6+4];
    ld.global.u8    %rs11, [%rd6+5];
    ld.global.u8    %rs12, [%rd6+6];

    cvt.u32.u16     %r18, %rs1;
    cvt.u32.u16     %r19, %rs4;
    add.s32         %r20, %r18, %r19;
    cvt.u32.u16     %r21, %rs7;
    add.s32         %r22, %r20, %r21;
    cvt.u32.u16     %r23, %rs10;
    add.s32         %r24, %r22, %r23;
    cvt.rn.f32.u32  %f1, %r24;
    mul.f32         %f2, %f1, 0f3E800000;

    cvt.u32.u16     %r25, %rs2;
    cvt.u32.u16     %r26, %rs5;
    add.s32         %r27, %r25, %r26;
    cvt.u32.u16     %r28, %rs8;
    add.s32         %r29, %r27, %r28;
    cvt.u32.u16     %r30, %rs11;
    add.s32         %r31, %r29, %r30;
    cvt.rn.f32.u32  %f3, %r31;
    mul.f32         %f4, %f3, 0f3E800000;

    cvt.u32.u16     %r32, %rs3;
    cvt.u32.u16     %r33, %rs6;
    add.s32         %r34, %r32, %r33;
    cvt.u32.u16     %r35, %rs9;
    add.s32         %r36, %r34, %r35;
    cvt.u32.u16     %r37, %rs12;
    add.s32         %r38, %r36, %r37;
    cvt.rn.f32.u32  %f5, %r38;
    mul.f32         %f6, %f5, 0f3E800000;

    mul.f32         %f7, %f6, 0f3E59B3D0;
    mul.f32         %f8, %f4, 0f3F371759;
    add.f32         %f9, %f7, %f8;
    mul.f32         %f10, %f2, 0f3D93DD98;
    add.f32         %f11, %f9, %f10;

    sub.f32         %f12, %f2, %f11;
    div.rn.f32      %f13, %f12, 0f3FED844D;
    mul.f32         %f14, %f13, 0f43600000;
    div.rn.f32      %f15, %f14, 0f437F0000;
    add.f32         %f16, %f15, 0f43000000;
    max.f32         %f17, %f16, 0f00000000;
    min.f32         %f18, %f17, 0f437F0000;
    add.f32         %f19, %f18, 0f3F000000;
    cvt.rzi.u32.f32 %r39, %f19;
    cvt.u16.u32     %rs13, %r39;

    sub.f32         %f20, %f6, %f11;
    div.rn.f32      %f21, %f20, 0f3FC9930C;
    mul.f32         %f22, %f21, 0f43600000;
    div.rn.f32      %f23, %f22, 0f437F0000;
    add.f32         %f24, %f23, 0f43000000;
    max.f32         %f25, %f24, 0f00000000;
    min.f32         %f26, %f25, 0f437F0000;
    add.f32         %f27, %f26, 0f3F000000;
    cvt.rzi.u32.f32 %r40, %f27;
    cvt.u16.u32     %rs14, %r40;

    mul.lo.s32      %r41, %r12, %r1;
    shl.b32         %r42, %r8, 1;
    add.s32         %r43, %r41, %r42;
    cvt.u64.u32     %rd7, %r43;
    add.s64         %rd8, %rd1, %rd7;
    st.global.u8    [%rd8], %rs13;
    st.global.u8    [%rd8+1], %rs14;

CHROMA_DONE:
    ret;
}

.visible .entry extract_alpha(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<20>;
    .reg .b64   %rd<12>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        ALPHA_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        ALPHA_DONE;

    mul.lo.s32      %r13, %r12, %r2;
    shl.b32         %r14, %r8, 2;
    add.s32         %r15, %r13, %r14;
    cvt.u64.u32     %rd3, %r15;
    add.s64         %rd4, %rd2, %rd3;
    ld.global.u8    %rs1, [%rd4+3];

    mul.lo.s32      %r16, %r12, %r1;
    add.s32         %r17, %r16, %r8;
    cvt.u64.u32     %rd5, %r17;
    add.s64         %rd6, %rd1, %rd5;
    st.global.u8    [%rd6], %rs1;

ALPHA_DONE:
    ret;
}


.visible .entry extract_alpha_half(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 half_width,
    .param .u32 half_height
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<8>;
    .reg .b32   %r<30>;
    .reg .b64   %rd<14>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [half_width];
    ld.param.u32    %r4, [half_height];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        ALPHA_HALF_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        ALPHA_HALF_DONE;

    shl.b32         %r13, %r12, 1;
    shl.b32         %r14, %r8, 1;
    mul.lo.s32      %r15, %r13, %r2;
    shl.b32         %r16, %r14, 2;
    add.s32         %r17, %r15, %r16;
    cvt.u64.u32     %rd3, %r17;
    add.s64         %rd4, %rd2, %rd3;
    cvt.u64.u32     %rd5, %r2;
    add.s64         %rd6, %rd4, %rd5;

    ld.global.u8    %r18, [%rd4+3];
    ld.global.u8    %r19, [%rd4+7];
    ld.global.u8    %r20, [%rd6+3];
    ld.global.u8    %r21, [%rd6+7];

    add.s32         %r22, %r18, %r19;
    add.s32         %r23, %r22, %r20;
    add.s32         %r24, %r23, %r21;
    add.s32         %r25, %r24, 2;
    shr.u32         %r26, %r25, 2;

    mul.lo.s32      %r27, %r12, %r1;
    add.s32         %r28, %r27, %r8;
    cvt.u64.u32     %rd7, %r28;
    add.s64         %rd8, %rd1, %rd7;
    cvt.u16.u32     %rs1, %r26;
    st.global.u8    [%rd8], %rs1;

ALPHA_HALF_DONE:
    ret;
}

.visible .entry key_bgra(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .f32 key_b,
    .param .f32 key_g,
    .param .f32 key_r,
    .param .f32 band_low,
    .param .f32 inv_band_width
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<8>;
    .reg .b32   %r<24>;
    .reg .f32   %f<32>;
    .reg .b64   %rd<12>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.f32    %f1, [key_b];
    ld.param.f32    %f2, [key_g];
    ld.param.f32    %f3, [key_r];
    ld.param.f32    %f4, [band_low];
    ld.param.f32    %f5, [inv_band_width];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        KEY_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        KEY_DONE;

    mul.lo.s32      %r13, %r12, %r2;
    shl.b32         %r14, %r8, 2;
    add.s32         %r15, %r13, %r14;
    cvt.u64.u32     %rd3, %r15;
    add.s64         %rd4, %rd2, %rd3;

    ld.global.u8    %rs1, [%rd4];
    ld.global.u8    %rs2, [%rd4+1];
    ld.global.u8    %rs3, [%rd4+2];
    ld.global.u8    %rs5, [%rd4+3];

    cvt.u32.u16     %r16, %rs1;
    cvt.rn.f32.u32  %f6, %r16;
    cvt.u32.u16     %r17, %rs2;
    cvt.rn.f32.u32  %f7, %r17;
    cvt.u32.u16     %r18, %rs3;
    cvt.rn.f32.u32  %f8, %r18;
    cvt.u32.u16     %r22, %rs5;
    cvt.rn.f32.u32  %f28, %r22;

    div.rn.f32      %f9, %f6, 0f437F0000;
    sub.f32         %f10, %f9, %f1;
    div.rn.f32      %f11, %f7, 0f437F0000;
    sub.f32         %f12, %f11, %f2;
    div.rn.f32      %f13, %f8, 0f437F0000;
    sub.f32         %f14, %f13, %f3;

    mul.f32         %f15, %f10, %f10;
    mul.f32         %f16, %f12, %f12;
    mul.f32         %f17, %f14, %f14;
    add.f32         %f18, %f15, %f16;
    add.f32         %f19, %f18, %f17;
    sqrt.rn.f32     %f20, %f19;
    div.rn.f32      %f21, %f20, 0f3FDDB3D7;

    sub.f32         %f22, %f21, %f4;
    mul.f32         %f23, %f22, %f5;
    max.f32         %f24, %f23, 0f00000000;
    min.f32         %f25, %f24, 0f3F800000;

    mul.f32         %f26, %f25, %f28;
    add.f32         %f27, %f26, 0f3F000000;
    cvt.rzi.u32.f32 %r19, %f27;
    cvt.u16.u32     %rs4, %r19;

    mul.lo.s32      %r20, %r12, %r1;
    add.s32         %r21, %r20, %r14;
    cvt.u64.u32     %rd5, %r21;
    add.s64         %rd6, %rd1, %rd5;

    st.global.u8    [%rd6], %rs1;
    st.global.u8    [%rd6+1], %rs2;
    st.global.u8    [%rd6+2], %rs3;
    st.global.u8    [%rd6+3], %rs4;

KEY_DONE:
    ret;
}
.visible .entry nv12_to_bgra(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 luma,
    .param .u32 luma_pitch,
    .param .u64 chroma,
    .param .u32 chroma_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .f32 red_y,
    .param .f32 red_cb,
    .param .f32 red_cr,
    .param .f32 red_offset,
    .param .f32 green_y,
    .param .f32 green_cb,
    .param .f32 green_cr,
    .param .f32 green_offset,
    .param .f32 blue_y,
    .param .f32 blue_cb,
    .param .f32 blue_cr,
    .param .f32 blue_offset,
    .param .u32 gamut,
    .param .f32 gamut_rr,
    .param .f32 gamut_rg,
    .param .f32 gamut_rb,
    .param .f32 gamut_gr,
    .param .f32 gamut_gg,
    .param .f32 gamut_gb,
    .param .f32 gamut_br,
    .param .f32 gamut_bg,
    .param .f32 gamut_bb
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<12>;
    .reg .b32   %r<40>;
    .reg .f32   %f<80>;
    .reg .b64   %rd<20>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [luma];
    ld.param.u32    %r2, [luma_pitch];
    ld.param.u64    %rd3, [chroma];
    ld.param.u32    %r3, [chroma_pitch];
    ld.param.u32    %r4, [width];
    ld.param.u32    %r5, [height];

    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %ntid.x;
    mov.u32         %r8, %tid.x;
    mad.lo.s32      %r9, %r6, %r7, %r8;
    mov.u32         %r10, %ctaid.y;
    mov.u32         %r11, %ntid.y;
    mov.u32         %r12, %tid.y;
    mad.lo.s32      %r13, %r10, %r11, %r12;

    setp.ge.u32     %p1, %r9, %r4;
    @%p1 bra        NV12_BGRA_DONE;
    setp.ge.u32     %p2, %r13, %r5;
    @%p2 bra        NV12_BGRA_DONE;

    ld.param.f32    %f40, [red_y];
    ld.param.f32    %f41, [red_cb];
    ld.param.f32    %f42, [red_cr];
    ld.param.f32    %f43, [red_offset];
    ld.param.f32    %f44, [green_y];
    ld.param.f32    %f45, [green_cb];
    ld.param.f32    %f46, [green_cr];
    ld.param.f32    %f47, [green_offset];
    ld.param.f32    %f48, [blue_y];
    ld.param.f32    %f49, [blue_cb];
    ld.param.f32    %f50, [blue_cr];
    ld.param.f32    %f51, [blue_offset];
    ld.param.u32    %r30, [gamut];

    mad.lo.s32      %r14, %r13, %r2, %r9;
    cvt.u64.u32     %rd4, %r14;
    add.s64         %rd5, %rd2, %rd4;
    ld.global.u8    %rs1, [%rd5];

    shr.u32         %r15, %r13, 1;
    shr.u32         %r16, %r9, 1;
    shl.b32         %r17, %r16, 1;
    mad.lo.s32      %r18, %r15, %r3, %r17;
    cvt.u64.u32     %rd6, %r18;
    add.s64         %rd7, %rd3, %rd6;
    ld.global.u8    %rs2, [%rd7];
    ld.global.u8    %rs3, [%rd7+1];

    // Y', Cb and Cr as 0..1, then each of R', G' and B' as one row of
    // `yuv_to_rgb_rows`, in the order a test repeats with `mul_add`.
    cvt.u32.u16     %r19, %rs1;
    cvt.rn.f32.u32  %f1, %r19;
    mul.f32         %f1, %f1, 0f3B808081;
    cvt.u32.u16     %r20, %rs2;
    cvt.rn.f32.u32  %f2, %r20;
    mul.f32         %f2, %f2, 0f3B808081;
    cvt.u32.u16     %r21, %rs3;
    cvt.rn.f32.u32  %f3, %r21;
    mul.f32         %f3, %f3, 0f3B808081;

    fma.rn.f32      %f4, %f40, %f1, %f43;
    fma.rn.f32      %f4, %f41, %f2, %f4;
    fma.rn.f32      %f4, %f42, %f3, %f4;
    fma.rn.f32      %f5, %f44, %f1, %f47;
    fma.rn.f32      %f5, %f45, %f2, %f5;
    fma.rn.f32      %f5, %f46, %f3, %f5;
    fma.rn.f32      %f6, %f48, %f1, %f51;
    fma.rn.f32      %f6, %f49, %f2, %f6;
    fma.rn.f32      %f6, %f50, %f3, %f6;

    setp.eq.u32     %p3, %r30, 0;
    @%p3 bra        NV12_BGRA_GAMMA_DONE;

    // BT.2020 primaries into BT.709's: decode gamma 2.2, mix in linear
    // light, clip what BT.709 cannot hold, and encode again.
    ld.param.f32    %f52, [gamut_rr];
    ld.param.f32    %f53, [gamut_rg];
    ld.param.f32    %f54, [gamut_rb];
    ld.param.f32    %f55, [gamut_gr];
    ld.param.f32    %f56, [gamut_gg];
    ld.param.f32    %f57, [gamut_gb];
    ld.param.f32    %f58, [gamut_br];
    ld.param.f32    %f59, [gamut_bg];
    ld.param.f32    %f60, [gamut_bb];

    add.sat.f32     %f4, %f4, 0f00000000;
    add.sat.f32     %f5, %f5, 0f00000000;
    add.sat.f32     %f6, %f6, 0f00000000;
    lg2.approx.f32  %f7, %f4;
    mul.f32         %f7, %f7, 0f400CCCCD;
    ex2.approx.f32  %f4, %f7;
    lg2.approx.f32  %f7, %f5;
    mul.f32         %f7, %f7, 0f400CCCCD;
    ex2.approx.f32  %f5, %f7;
    lg2.approx.f32  %f7, %f6;
    mul.f32         %f7, %f7, 0f400CCCCD;
    ex2.approx.f32  %f6, %f7;

    mul.f32         %f7, %f52, %f4;
    fma.rn.f32      %f7, %f53, %f5, %f7;
    fma.rn.f32      %f7, %f54, %f6, %f7;
    mul.f32         %f8, %f55, %f4;
    fma.rn.f32      %f8, %f56, %f5, %f8;
    fma.rn.f32      %f8, %f57, %f6, %f8;
    mul.f32         %f9, %f58, %f4;
    fma.rn.f32      %f9, %f59, %f5, %f9;
    fma.rn.f32      %f9, %f60, %f6, %f9;

    add.sat.f32     %f7, %f7, 0f00000000;
    add.sat.f32     %f8, %f8, 0f00000000;
    add.sat.f32     %f9, %f9, 0f00000000;
    lg2.approx.f32  %f10, %f7;
    mul.f32         %f10, %f10, 0f3EE8BA2F;
    ex2.approx.f32  %f4, %f10;
    lg2.approx.f32  %f10, %f8;
    mul.f32         %f10, %f10, 0f3EE8BA2F;
    ex2.approx.f32  %f5, %f10;
    lg2.approx.f32  %f10, %f9;
    mul.f32         %f10, %f10, 0f3EE8BA2F;
    ex2.approx.f32  %f6, %f10;

NV12_BGRA_GAMMA_DONE:
    mul.f32         %f13, %f4, 0f437F0000;
    mul.f32         %f18, %f5, 0f437F0000;
    mul.f32         %f11, %f6, 0f437F0000;

    max.f32         %f19, %f11, 0f00000000;
    min.f32         %f20, %f19, 0f437F0000;
    add.f32         %f21, %f20, 0f3F000000;
    cvt.rzi.u32.f32 %r22, %f21;
    cvt.u16.u32     %rs4, %r22;

    max.f32         %f22, %f18, 0f00000000;
    min.f32         %f23, %f22, 0f437F0000;
    add.f32         %f24, %f23, 0f3F000000;
    cvt.rzi.u32.f32 %r23, %f24;
    cvt.u16.u32     %rs5, %r23;

    max.f32         %f25, %f13, 0f00000000;
    min.f32         %f26, %f25, 0f437F0000;
    add.f32         %f27, %f26, 0f3F000000;
    cvt.rzi.u32.f32 %r24, %f27;
    cvt.u16.u32     %rs6, %r24;

    mov.u16         %rs7, 255;

    shl.b32         %r25, %r9, 2;
    mad.lo.s32      %r26, %r13, %r1, %r25;
    cvt.u64.u32     %rd8, %r26;
    add.s64         %rd9, %rd1, %rd8;
    st.global.u8    [%rd9], %rs4;
    st.global.u8    [%rd9+1], %rs5;
    st.global.u8    [%rd9+2], %rs6;
    st.global.u8    [%rd9+3], %rs7;

NV12_BGRA_DONE:
    ret;
}
.visible .entry effect_bgra(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .f32 red_from_red,
    .param .f32 red_from_green,
    .param .f32 red_from_blue,
    .param .f32 red_offset,
    .param .f32 green_from_red,
    .param .f32 green_from_green,
    .param .f32 green_from_blue,
    .param .f32 green_offset,
    .param .f32 blue_from_red,
    .param .f32 blue_from_green,
    .param .f32 blue_from_blue,
    .param .f32 blue_offset,
    .param .f32 exponent,
    .param .f32 opacity,
    .param .f32 luma_low,
    .param .f32 luma_low_inv,
    .param .f32 luma_high,
    .param .f32 luma_high_inv
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<8>;
    .reg .b32   %r<24>;
    .reg .f32   %f<32>;
    .reg .b64   %rd<8>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.f32    %f1, [red_from_red];
    ld.param.f32    %f2, [red_from_green];
    ld.param.f32    %f3, [red_from_blue];
    ld.param.f32    %f4, [red_offset];
    ld.param.f32    %f5, [green_from_red];
    ld.param.f32    %f6, [green_from_green];
    ld.param.f32    %f7, [green_from_blue];
    ld.param.f32    %f8, [green_offset];
    ld.param.f32    %f9, [blue_from_red];
    ld.param.f32    %f10, [blue_from_green];
    ld.param.f32    %f11, [blue_from_blue];
    ld.param.f32    %f12, [blue_offset];
    ld.param.f32    %f13, [exponent];
    ld.param.f32    %f14, [opacity];
    ld.param.f32    %f15, [luma_low];
    ld.param.f32    %f16, [luma_low_inv];
    ld.param.f32    %f17, [luma_high];
    ld.param.f32    %f18, [luma_high_inv];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %ntid.x;
    mov.u32         %r7, %tid.x;
    mad.lo.s32      %r8, %r5, %r6, %r7;
    mov.u32         %r9, %ctaid.y;
    mov.u32         %r10, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r12, %r9, %r10, %r11;

    setp.ge.u32     %p1, %r8, %r3;
    @%p1 bra        EFFECT_DONE;
    setp.ge.u32     %p2, %r12, %r4;
    @%p2 bra        EFFECT_DONE;

    // The source pixel, bytes B, G, R, A, each scaled to 0..1.
    mul.lo.s32      %r13, %r12, %r2;
    shl.b32         %r14, %r8, 2;
    add.s32         %r15, %r13, %r14;
    cvt.u64.u32     %rd3, %r15;
    add.s64         %rd4, %rd2, %rd3;

    ld.global.u8    %rs1, [%rd4];
    ld.global.u8    %rs2, [%rd4+1];
    ld.global.u8    %rs3, [%rd4+2];
    ld.global.u8    %rs4, [%rd4+3];

    cvt.u32.u16     %r16, %rs1;
    cvt.rn.f32.u32  %f19, %r16;
    div.rn.f32      %f19, %f19, 0f437F0000;
    cvt.u32.u16     %r17, %rs2;
    cvt.rn.f32.u32  %f20, %r17;
    div.rn.f32      %f20, %f20, 0f437F0000;
    cvt.u32.u16     %r18, %rs3;
    cvt.rn.f32.u32  %f21, %r18;
    div.rn.f32      %f21, %f21, 0f437F0000;
    cvt.u32.u16     %r19, %rs4;
    cvt.rn.f32.u32  %f22, %r19;
    div.rn.f32      %f22, %f22, 0f437F0000;

    // BT.709 luma of the pixel as it arrived, saturated.
    mul.f32         %f23, %f21, 0f3E59B3D0;
    fma.rn.f32      %f23, %f20, 0f3F371759, %f23;
    fma.rn.f32      %f23, %f19, 0f3D93DD98, %f23;
    max.f32         %f23, %f23, 0f00000000;
    min.f32         %f23, %f23, 0f3F800000;

    // The luma mask: saturate((luma - low) * low_inv + 1)
    //              * saturate((high - luma) * high_inv + 1).
    sub.f32         %f24, %f23, %f15;
    fma.rn.f32      %f24, %f24, %f16, 0f3F800000;
    max.f32         %f24, %f24, 0f00000000;
    min.f32         %f24, %f24, 0f3F800000;
    sub.f32         %f25, %f17, %f23;
    fma.rn.f32      %f25, %f25, %f18, 0f3F800000;
    max.f32         %f25, %f25, 0f00000000;
    min.f32         %f25, %f25, 0f3F800000;
    mul.f32         %f26, %f24, %f25;

    // Each channel raised to the exponent, unless that is 1: x^e as
    // 2^(e * log2 x). log2 of 0 is -inf, and 2^-inf is 0, so black stays
    // black.
    setp.eq.f32     %p3, %f13, 0f3F800000;
    @%p3 bra        EFFECT_LINEAR;
    lg2.approx.f32  %f19, %f19;
    mul.f32         %f19, %f19, %f13;
    ex2.approx.f32  %f19, %f19;
    lg2.approx.f32  %f20, %f20;
    mul.f32         %f20, %f20, %f13;
    ex2.approx.f32  %f20, %f20;
    lg2.approx.f32  %f21, %f21;
    mul.f32         %f21, %f21, %f13;
    ex2.approx.f32  %f21, %f21;
EFFECT_LINEAR:

    // The colour matrix, one row per output channel, saturated.
    fma.rn.f32      %f27, %f1, %f21, %f4;
    fma.rn.f32      %f27, %f2, %f20, %f27;
    fma.rn.f32      %f27, %f3, %f19, %f27;
    max.f32         %f27, %f27, 0f00000000;
    min.f32         %f27, %f27, 0f3F800000;
    fma.rn.f32      %f28, %f5, %f21, %f8;
    fma.rn.f32      %f28, %f6, %f20, %f28;
    fma.rn.f32      %f28, %f7, %f19, %f28;
    max.f32         %f28, %f28, 0f00000000;
    min.f32         %f28, %f28, 0f3F800000;
    fma.rn.f32      %f29, %f9, %f21, %f12;
    fma.rn.f32      %f29, %f10, %f20, %f29;
    fma.rn.f32      %f29, %f11, %f19, %f29;
    max.f32         %f29, %f29, 0f00000000;
    min.f32         %f29, %f29, 0f3F800000;

    // Alpha times opacity times the mask, saturated.
    mul.f32         %f30, %f22, %f14;
    mul.f32         %f30, %f30, %f26;
    max.f32         %f30, %f30, 0f00000000;
    min.f32         %f30, %f30, 0f3F800000;

    // Back to bytes, nearest: x * 255 + 0.5, truncated.
    fma.rn.f32      %f31, %f27, 0f437F0000, 0f3F000000;
    cvt.rzi.u32.f32 %r20, %f31;
    fma.rn.f32      %f31, %f28, 0f437F0000, 0f3F000000;
    cvt.rzi.u32.f32 %r21, %f31;
    fma.rn.f32      %f31, %f29, 0f437F0000, 0f3F000000;
    cvt.rzi.u32.f32 %r22, %f31;
    fma.rn.f32      %f31, %f30, 0f437F0000, 0f3F000000;
    cvt.rzi.u32.f32 %r23, %f31;

    mul.lo.s32      %r13, %r12, %r1;
    add.s32         %r15, %r13, %r14;
    cvt.u64.u32     %rd5, %r15;
    add.s64         %rd6, %rd1, %rd5;

    cvt.u16.u32     %rs5, %r22;
    st.global.u8    [%rd6], %rs5;
    cvt.u16.u32     %rs6, %r21;
    st.global.u8    [%rd6+1], %rs6;
    cvt.u16.u32     %rs7, %r20;
    st.global.u8    [%rd6+2], %rs7;
    cvt.u16.u32     %rs5, %r23;
    st.global.u8    [%rd6+3], %rs5;

EFFECT_DONE:
    ret;
}
// `core/tone_map.rs`'s definition, one step to a block: see `ToneMap` there
// for why each is what it is. Powers are `ex2(y * lg2(x))`, which is zero
// for a zero `x`, as every power here wants.
.func (.reg .f32 nits) pq_to_nits (.reg .f32 signal)
{
    .reg .f32   %a<10>;

    max.f32         %a1, signal, 0f00000000;
    lg2.approx.f32  %a2, %a1;
    mul.f32         %a2, %a2, 0f3C4FCDAC;
    ex2.approx.f32  %a3, %a2;
    sub.f32         %a4, %a3, 0f3F560000;
    max.f32         %a4, %a4, 0f00000000;
    mul.f32         %a5, %a3, 0f41958000;
    mov.f32         %a6, 0f4196D000;
    sub.f32         %a5, %a6, %a5;
    div.rn.f32      %a7, %a4, %a5;
    lg2.approx.f32  %a8, %a7;
    mul.f32         %a8, %a8, 0f40C8E06B;
    ex2.approx.f32  %a9, %a8;
    mul.f32         nits, %a9, 0f461C4000;
    ret;
}

.func (.reg .f32 signal) nits_to_pq (.reg .f32 nits)
{
    .reg .f32   %b<10>;

    mul.f32         %b1, nits, 0f38D1B717;
    max.f32         %b1, %b1, 0f00000000;
    lg2.approx.f32  %b2, %b1;
    mul.f32         %b2, %b2, 0f3E232000;
    ex2.approx.f32  %b3, %b2;
    mul.f32         %b4, %b3, 0f4196D000;
    add.f32         %b4, %b4, 0f3F560000;
    mul.f32         %b5, %b3, 0f41958000;
    add.f32         %b5, %b5, 0f3F800000;
    div.rn.f32      %b6, %b4, %b5;
    lg2.approx.f32  %b7, %b6;
    mul.f32         %b7, %b7, 0f429DB000;
    ex2.approx.f32  signal, %b7;
    ret;
}

.func (.reg .f32 scene) hlg_to_scene (.reg .f32 signal)
{
    .reg .pred  %q;
    .reg .f32   %c<6>;

    setp.le.f32     %q, signal, 0f3F000000;
    @%q bra         HLG_LOW;
    sub.f32         %c1, signal, 0f3F0F564F;
    mul.f32         %c1, %c1, 0f40B2F029;
    mul.f32         %c1, %c1, 0f3FB8AA3B;
    ex2.approx.f32  %c2, %c1;
    add.f32         %c2, %c2, 0f3E91C020;
    mul.f32         scene, %c2, 0f3DAAAAAB;
    ret;
HLG_LOW:
    mul.f32         %c3, signal, signal;
    mul.f32         scene, %c3, 0f3EAAAAAB;
    ret;
}

.func (.reg .f32 brought) eetf (.reg .f32 x, .reg .f32 knee, .reg .f32 target)
{
    .reg .pred  %q;
    .reg .f32   %d<12>;

    setp.lt.f32     %q, x, knee;
    @%q bra         EETF_PASS;
    mov.f32         %d0, 0f3F800000;
    sub.f32         %d1, %d0, knee;
    sub.f32         %d2, x, knee;
    div.rn.f32      %d3, %d2, %d1;
    mul.f32         %d4, %d3, %d3;
    mul.f32         %d5, %d4, %d3;
    // 2t^3 - 3t^2 + 1, t^3 - 2t^2 + t, -2t^3 + 3t^2
    mul.f32         %d6, %d5, 0f40000000;
    mul.f32         %d7, %d4, 0f40400000;
    sub.f32         %d6, %d6, %d7;
    add.f32         %d6, %d6, 0f3F800000;
    mul.f32         %d8, %d4, 0f40000000;
    sub.f32         %d8, %d5, %d8;
    add.f32         %d8, %d8, %d3;
    mul.f32         %d9, %d5, 0f40000000;
    sub.f32         %d9, %d7, %d9;
    mul.f32         %d10, %d6, knee;
    fma.rn.f32      %d10, %d8, %d1, %d10;
    fma.rn.f32      brought, %d9, target, %d10;
    ret;
EETF_PASS:
    mov.f32         brought, x;
    ret;
}

.visible .entry hdr_to_bgra(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 luma,
    .param .u32 luma_pitch,
    .param .u64 chroma,
    .param .u32 chroma_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 wide,
    .param .f32 red_y,
    .param .f32 red_cb,
    .param .f32 red_cr,
    .param .f32 red_offset,
    .param .f32 green_y,
    .param .f32 green_cb,
    .param .f32 green_cr,
    .param .f32 green_offset,
    .param .f32 blue_y,
    .param .f32 blue_cb,
    .param .f32 blue_cr,
    .param .f32 blue_offset,
    .param .u32 transfer,
    .param .f32 source_peak_pq,
    .param .f32 target_peak,
    .param .f32 knee,
    .param .f32 gamut_rr,
    .param .f32 gamut_rg,
    .param .f32 gamut_rb,
    .param .f32 gamut_gr,
    .param .f32 gamut_gg,
    .param .f32 gamut_gb,
    .param .f32 gamut_br,
    .param .f32 gamut_bg,
    .param .f32 gamut_bb
)
{
    .reg .pred  %p<8>;
    .reg .b16   %rs<12>;
    .reg .b32   %r<48>;
    .reg .f32   %f<96>;
    .reg .b64   %rd<24>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [luma];
    ld.param.u32    %r2, [luma_pitch];
    ld.param.u64    %rd3, [chroma];
    ld.param.u32    %r3, [chroma_pitch];
    ld.param.u32    %r4, [width];
    ld.param.u32    %r5, [height];
    ld.param.u32    %r6, [wide];

    mov.u32         %r7, %ctaid.x;
    mov.u32         %r8, %ntid.x;
    mov.u32         %r10, %tid.x;
    mad.lo.s32      %r9, %r7, %r8, %r10;
    mov.u32         %r11, %ctaid.y;
    mov.u32         %r12, %ntid.y;
    mov.u32         %r14, %tid.y;
    mad.lo.s32      %r13, %r11, %r12, %r14;

    setp.ge.u32     %p1, %r9, %r4;
    @%p1 bra        HDR_DONE;
    setp.ge.u32     %p2, %r13, %r5;
    @%p2 bra        HDR_DONE;

    // A sample is `1 << wide` bytes: 2 for P010, 1 for NV12. Chroma is the
    // pair for the 2x2 block this pixel is in.
    shl.b32         %r15, %r9, %r6;
    mad.lo.s32      %r16, %r13, %r2, %r15;
    cvt.u64.u32     %rd4, %r16;
    add.s64         %rd5, %rd2, %rd4;
    shr.u32         %r17, %r13, 1;
    shr.u32         %r18, %r9, 1;
    add.u32         %r19, %r6, 1;
    shl.b32         %r20, %r18, %r19;
    mad.lo.s32      %r21, %r17, %r3, %r20;
    cvt.u64.u32     %rd6, %r21;
    add.s64         %rd7, %rd3, %rd6;

    setp.ne.u32     %p3, %r6, 0;
    @%p3 bra        HDR_WIDE;
    ld.global.u8    %rs1, [%rd5];
    ld.global.u8    %rs2, [%rd7];
    ld.global.u8    %rs3, [%rd7+1];
    cvt.u32.u16     %r22, %rs1;
    cvt.u32.u16     %r23, %rs2;
    cvt.u32.u16     %r24, %rs3;
    cvt.rn.f32.u32  %f1, %r22;
    cvt.rn.f32.u32  %f2, %r23;
    cvt.rn.f32.u32  %f3, %r24;
    mul.f32         %f1, %f1, 0f3B808081;
    mul.f32         %f2, %f2, 0f3B808081;
    mul.f32         %f3, %f3, 0f3B808081;
    bra             HDR_LOADED;
HDR_WIDE:
    ld.global.u16   %rs1, [%rd5];
    ld.global.u16   %rs2, [%rd7];
    ld.global.u16   %rs3, [%rd7+2];
    cvt.u32.u16     %r22, %rs1;
    cvt.u32.u16     %r23, %rs2;
    cvt.u32.u16     %r24, %rs3;
    cvt.rn.f32.u32  %f1, %r22;
    cvt.rn.f32.u32  %f2, %r23;
    cvt.rn.f32.u32  %f3, %r24;
    mul.f32         %f1, %f1, 0f37800080;
    mul.f32         %f2, %f2, 0f37800080;
    mul.f32         %f3, %f3, 0f37800080;
HDR_LOADED:

    ld.param.f32    %f40, [red_y];
    ld.param.f32    %f41, [red_cb];
    ld.param.f32    %f42, [red_cr];
    ld.param.f32    %f43, [red_offset];
    ld.param.f32    %f44, [green_y];
    ld.param.f32    %f45, [green_cb];
    ld.param.f32    %f46, [green_cr];
    ld.param.f32    %f47, [green_offset];
    ld.param.f32    %f48, [blue_y];
    ld.param.f32    %f49, [blue_cb];
    ld.param.f32    %f50, [blue_cr];
    ld.param.f32    %f51, [blue_offset];

    // 1. R'G'B', clipped to 0..1.
    fma.rn.f32      %f4, %f40, %f1, %f43;
    fma.rn.f32      %f4, %f41, %f2, %f4;
    fma.rn.f32      %f4, %f42, %f3, %f4;
    fma.rn.f32      %f5, %f44, %f1, %f47;
    fma.rn.f32      %f5, %f45, %f2, %f5;
    fma.rn.f32      %f5, %f46, %f3, %f5;
    fma.rn.f32      %f6, %f48, %f1, %f51;
    fma.rn.f32      %f6, %f49, %f2, %f6;
    fma.rn.f32      %f6, %f50, %f3, %f6;
    add.sat.f32     %f4, %f4, 0f00000000;
    add.sat.f32     %f5, %f5, 0f00000000;
    add.sat.f32     %f6, %f6, 0f00000000;

    // 2. Light, in nits.
    ld.param.u32    %r30, [transfer];
    setp.eq.u32     %p4, %r30, 1;
    @!%p4 bra       HDR_HLG;
    call            (%f7), pq_to_nits, (%f4);
    call            (%f8), pq_to_nits, (%f5);
    call            (%f9), pq_to_nits, (%f6);
    bra             HDR_NITS;
HDR_HLG:
    call            (%f10), hlg_to_scene, (%f4);
    call            (%f11), hlg_to_scene, (%f5);
    call            (%f12), hlg_to_scene, (%f6);
    mul.f32         %f13, %f10, 0f3E86809D;
    mul.f32         %f14, %f11, 0f3F2D9168;
    add.f32         %f13, %f13, %f14;
    mul.f32         %f15, %f12, 0f3D72E48F;
    add.f32         %f13, %f13, %f15;
    max.f32         %f13, %f13, 0f358637BD;
    lg2.approx.f32  %f16, %f13;
    mul.f32         %f16, %f16, 0f3E4CCCCD;
    ex2.approx.f32  %f17, %f16;
    mul.f32         %f17, %f17, 0f447A0000;
    mul.f32         %f7, %f10, %f17;
    mul.f32         %f8, %f11, %f17;
    mul.f32         %f9, %f12, %f17;
HDR_NITS:

    // 3. BT.2020's primaries into BT.709's, negatives clipped.
    ld.param.f32    %f52, [gamut_rr];
    ld.param.f32    %f53, [gamut_rg];
    ld.param.f32    %f54, [gamut_rb];
    ld.param.f32    %f55, [gamut_gr];
    ld.param.f32    %f56, [gamut_gg];
    ld.param.f32    %f57, [gamut_gb];
    ld.param.f32    %f58, [gamut_br];
    ld.param.f32    %f59, [gamut_bg];
    ld.param.f32    %f60, [gamut_bb];
    mul.f32         %f18, %f52, %f7;
    fma.rn.f32      %f18, %f53, %f8, %f18;
    fma.rn.f32      %f18, %f54, %f9, %f18;
    max.f32         %f18, %f18, 0f00000000;
    mul.f32         %f19, %f55, %f7;
    fma.rn.f32      %f19, %f56, %f8, %f19;
    fma.rn.f32      %f19, %f57, %f9, %f19;
    max.f32         %f19, %f19, 0f00000000;
    mul.f32         %f20, %f58, %f7;
    fma.rn.f32      %f20, %f59, %f8, %f20;
    fma.rn.f32      %f20, %f60, %f9, %f20;
    max.f32         %f20, %f20, 0f00000000;

    // 4. The EETF on the largest channel, the three scaled by it.
    max.f32         %f21, %f18, %f19;
    max.f32         %f21, %f21, %f20;
    mov.f32         %f22, 0f3F800000;
    ld.param.f32    %f61, [source_peak_pq];
    ld.param.f32    %f62, [target_peak];
    ld.param.f32    %f63, [knee];
    setp.le.f32     %p5, %f21, 0f00000000;
    @%p5 bra        HDR_SCALED;
    setp.ge.f32     %p6, %f63, 0f3F800000;
    @%p6 bra        HDR_SCALED;
    call            (%f23), nits_to_pq, (%f21);
    div.rn.f32      %f24, %f23, %f61;
    min.f32         %f24, %f24, 0f3F800000;
    call            (%f25), eetf, (%f24, %f63, %f62);
    mul.f32         %f26, %f25, %f61;
    call            (%f27), pq_to_nits, (%f26);
    div.rn.f32      %f22, %f27, %f21;
HDR_SCALED:

    // 5. 203 nits is 1.0, clipped, gamma 2.2, to bytes.
    mul.f32         %f28, %f18, %f22;
    mul.f32         %f29, %f19, %f22;
    mul.f32         %f30, %f20, %f22;
    mul.f32         %f28, %f28, 0f3BA16B31;
    mul.f32         %f29, %f29, 0f3BA16B31;
    mul.f32         %f30, %f30, 0f3BA16B31;
    add.sat.f32     %f28, %f28, 0f00000000;
    add.sat.f32     %f29, %f29, 0f00000000;
    add.sat.f32     %f30, %f30, 0f00000000;
    lg2.approx.f32  %f31, %f28;
    mul.f32         %f31, %f31, 0f3EE8BA2E;
    ex2.approx.f32  %f28, %f31;
    lg2.approx.f32  %f31, %f29;
    mul.f32         %f31, %f31, 0f3EE8BA2E;
    ex2.approx.f32  %f29, %f31;
    lg2.approx.f32  %f31, %f30;
    mul.f32         %f31, %f31, 0f3EE8BA2E;
    ex2.approx.f32  %f30, %f31;
    mul.f32         %f28, %f28, 0f437F0000;
    mul.f32         %f29, %f29, 0f437F0000;
    mul.f32         %f30, %f30, 0f437F0000;
    add.f32         %f28, %f28, 0f3F000000;
    add.f32         %f29, %f29, 0f3F000000;
    add.f32         %f30, %f30, 0f3F000000;
    cvt.rzi.u32.f32 %r31, %f28;
    cvt.rzi.u32.f32 %r32, %f29;
    cvt.rzi.u32.f32 %r33, %f30;
    min.u32         %r31, %r31, 255;
    min.u32         %r32, %r32, 255;
    min.u32         %r33, %r33, 255;
    cvt.u16.u32     %rs4, %r31;
    cvt.u16.u32     %rs5, %r32;
    cvt.u16.u32     %rs6, %r33;
    mov.u16         %rs7, 255;

    shl.b32         %r34, %r9, 2;
    mad.lo.s32      %r35, %r13, %r1, %r34;
    cvt.u64.u32     %rd8, %r35;
    add.s64         %rd9, %rd1, %rd8;
    st.global.u8    [%rd9], %rs6;
    st.global.u8    [%rd9+1], %rs5;
    st.global.u8    [%rd9+2], %rs4;
    st.global.u8    [%rd9+3], %rs7;

HDR_DONE:
    ret;
}
"#;

/// The one kernel this crate runs, as PTX the driver JIT-compiles at load.
///
/// # Why PTX rather than CUDA C
///
/// Compiling CUDA C needs `nvcc` or NVRTC, both of which ship with the CUDA
/// *toolkit* — a build requirement this crate deliberately does not impose,
/// since everything else it does needs only the driver. PTX is the driver's
/// own input format, so a kernel written here is a plain string constant:
/// nothing to compile, nothing to generate, nothing to check in. `.target
/// sm_50` is a floor, not a pin — the JIT recompiles it for whatever GPU is
/// actually present.
///
/// # What it computes
///
/// `dst = (src * alpha + dst * (255 - alpha) + 127) / 255`, one byte at a
/// time over a 2D region. Every term is non-negative, so the rounding is
/// symmetric and the division is unsigned — the reference implementation in
/// this module's tests is the same expression, and they are compared pixel
/// for pixel.
///
/// One kernel covers both NV12 planes. Luma is a byte per pixel; chroma is
/// interleaved `(U, V)` bytes, and blending each byte independently is
/// exactly right for both, so the chroma pass is the same call with the
/// plane's own byte width and half the rows.
///
/// `blend_masked` is the same mix with a per-pixel alpha and a constant
/// color, which is what a text layer needs: glyph coverage varies, the color
/// does not. Two things differ between its passes rather than one — chroma
/// alternates `(U, V)` by byte parity, hence `value_even`/`value_odd`, and
/// its mask is half resolution, hence `mask_shift`.
pub(super) const BLEND_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry blend_plane(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 alpha
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<32>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.u32    %r5, [alpha];

    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %ntid.x;
    mov.u32         %r8, %tid.x;
    mad.lo.s32      %r9, %r6, %r7, %r8;
    mov.u32         %r10, %ctaid.y;
    mov.u32         %r11, %ntid.y;
    mov.u32         %r12, %tid.y;
    mad.lo.s32      %r13, %r10, %r11, %r12;

    setp.ge.u32     %p1, %r9, %r3;
    @%p1 bra        DONE;
    setp.ge.u32     %p2, %r13, %r4;
    @%p2 bra        DONE;

    mad.lo.s32      %r14, %r13, %r1, %r9;
    cvt.u64.u32     %rd3, %r14;
    add.s64         %rd4, %rd1, %rd3;
    mad.lo.s32      %r15, %r13, %r2, %r9;
    cvt.u64.u32     %rd5, %r15;
    add.s64         %rd6, %rd2, %rd5;

    ld.global.u8    %r16, [%rd4];
    ld.global.u8    %r17, [%rd6];
    mul.lo.s32      %r18, %r17, %r5;
    sub.s32         %r19, 255, %r5;
    mul.lo.s32      %r20, %r16, %r19;
    add.s32         %r21, %r18, %r20;
    add.s32         %r22, %r21, 127;
    div.u32         %r23, %r22, 255;

    cvt.u16.u32     %rs1, %r23;
    st.global.u8    [%rd4], %rs1;
DONE:
    ret;
}

.visible .entry blend_masked(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 mask,
    .param .u32 mask_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 value_even,
    .param .u32 value_odd,
    .param .u32 opacity,
    .param .u32 mask_shift
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<40>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [mask];
    ld.param.u32    %r2, [mask_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.u32    %r5, [value_even];
    ld.param.u32    %r6, [value_odd];
    ld.param.u32    %r7, [opacity];
    ld.param.u32    %r31, [mask_shift];

    mov.u32         %r8, %ctaid.x;
    mov.u32         %r9, %ntid.x;
    mov.u32         %r10, %tid.x;
    mad.lo.s32      %r11, %r8, %r9, %r10;
    mov.u32         %r12, %ctaid.y;
    mov.u32         %r13, %ntid.y;
    mov.u32         %r14, %tid.y;
    mad.lo.s32      %r15, %r12, %r13, %r14;

    setp.ge.u32     %p1, %r11, %r3;
    @%p1 bra        MDONE;
    setp.ge.u32     %p2, %r15, %r4;
    @%p2 bra        MDONE;

    mad.lo.s32      %r16, %r15, %r1, %r11;
    cvt.u64.u32     %rd3, %r16;
    add.s64         %rd4, %rd1, %rd3;

    shr.u32         %r32, %r11, %r31;
    mad.lo.s32      %r17, %r15, %r2, %r32;
    cvt.u64.u32     %rd5, %r17;
    add.s64         %rd6, %rd2, %rd5;

    ld.global.u8    %r18, [%rd4];
    ld.global.u8    %r19, [%rd6];

    mul.lo.s32      %r20, %r19, %r7;
    add.s32         %r21, %r20, 127;
    div.u32         %r22, %r21, 255;

    and.b32         %r23, %r11, 1;
    setp.eq.u32     %p3, %r23, 0;
    selp.b32        %r24, %r5, %r6, %p3;

    mul.lo.s32      %r25, %r24, %r22;
    sub.s32         %r26, 255, %r22;
    mul.lo.s32      %r27, %r18, %r26;
    add.s32         %r28, %r25, %r27;
    add.s32         %r29, %r28, 127;
    div.u32         %r30, %r29, 255;

    cvt.u16.u32     %rs2, %r30;
    st.global.u8    [%rd4], %rs2;
MDONE:
    ret;
}

.visible .entry blend_plane_masked(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u64 mask,
    .param .u32 mask_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 opacity,
    .param .u32 mask_shift
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<40>;
    .reg .b64   %rd<20>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd7, [src];
    ld.param.u32    %r33, [src_pitch];
    ld.param.u64    %rd2, [mask];
    ld.param.u32    %r2, [mask_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.u32    %r7, [opacity];
    ld.param.u32    %r31, [mask_shift];

    mov.u32         %r8, %ctaid.x;
    mov.u32         %r9, %ntid.x;
    mov.u32         %r10, %tid.x;
    mad.lo.s32      %r11, %r8, %r9, %r10;
    mov.u32         %r12, %ctaid.y;
    mov.u32         %r13, %ntid.y;
    mov.u32         %r14, %tid.y;
    mad.lo.s32      %r15, %r12, %r13, %r14;

    setp.ge.u32     %p1, %r11, %r3;
    @%p1 bra        PMDONE;
    setp.ge.u32     %p2, %r15, %r4;
    @%p2 bra        PMDONE;

    mad.lo.s32      %r16, %r15, %r1, %r11;
    cvt.u64.u32     %rd3, %r16;
    add.s64         %rd4, %rd1, %rd3;

    mad.lo.s32      %r34, %r15, %r33, %r11;
    cvt.u64.u32     %rd8, %r34;
    add.s64         %rd9, %rd7, %rd8;

    shr.u32         %r32, %r11, %r31;
    mad.lo.s32      %r17, %r15, %r2, %r32;
    cvt.u64.u32     %rd5, %r17;
    add.s64         %rd6, %rd2, %rd5;

    ld.global.u8    %r18, [%rd4];
    ld.global.u8    %r19, [%rd6];
    ld.global.u8    %r24, [%rd9];

    mul.lo.s32      %r20, %r19, %r7;
    add.s32         %r21, %r20, 127;
    div.u32         %r22, %r21, 255;

    mul.lo.s32      %r25, %r24, %r22;
    sub.s32         %r26, 255, %r22;
    mul.lo.s32      %r27, %r18, %r26;
    add.s32         %r28, %r25, %r27;
    add.s32         %r29, %r28, 127;
    div.u32         %r30, %r29, 255;

    cvt.u16.u32     %rs2, %r30;
    st.global.u8    [%rd4], %rs2;
PMDONE:
    ret;
}



.visible .entry blend_bgra(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 opacity
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<8>;
    .reg .b32   %r<48>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [src];
    ld.param.u32    %r2, [src_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.u32    %r5, [opacity];

    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %ntid.x;
    mov.u32         %r8, %tid.x;
    mad.lo.s32      %r9, %r6, %r7, %r8;
    mov.u32         %r10, %ctaid.y;
    mov.u32         %r11, %ntid.y;
    mov.u32         %r12, %tid.y;
    mad.lo.s32      %r13, %r10, %r11, %r12;

    setp.ge.u32     %p1, %r9, %r3;
    @%p1 bra        BBDONE;
    setp.ge.u32     %p2, %r13, %r4;
    @%p2 bra        BBDONE;

    mul.lo.s32      %r14, %r9, 4;
    mad.lo.s32      %r15, %r13, %r1, %r14;
    cvt.u64.u32     %rd3, %r15;
    add.s64         %rd4, %rd1, %rd3;
    mad.lo.s32      %r16, %r13, %r2, %r14;
    cvt.u64.u32     %rd5, %r16;
    add.s64         %rd6, %rd2, %rd5;

    // The source's own alpha, scaled by the layer's opacity.
    ld.global.u8    %r17, [%rd6+3];
    mul.lo.s32      %r18, %r17, %r5;
    add.s32         %r19, %r18, 127;
    div.u32         %r20, %r19, 255;
    sub.s32         %r21, 255, %r20;

    // Blue, green and red: source over destination.
    ld.global.u8    %r22, [%rd6];
    ld.global.u8    %r23, [%rd4];
    mul.lo.s32      %r24, %r22, %r20;
    mul.lo.s32      %r25, %r23, %r21;
    add.s32         %r26, %r24, %r25;
    add.s32         %r27, %r26, 127;
    div.u32         %r28, %r27, 255;
    cvt.u16.u32     %rs1, %r28;
    st.global.u8    [%rd4], %rs1;

    ld.global.u8    %r29, [%rd6+1];
    ld.global.u8    %r30, [%rd4+1];
    mul.lo.s32      %r31, %r29, %r20;
    mul.lo.s32      %r32, %r30, %r21;
    add.s32         %r33, %r31, %r32;
    add.s32         %r34, %r33, 127;
    div.u32         %r35, %r34, 255;
    cvt.u16.u32     %rs2, %r35;
    st.global.u8    [%rd4+1], %rs2;

    ld.global.u8    %r36, [%rd6+2];
    ld.global.u8    %r37, [%rd4+2];
    mul.lo.s32      %r38, %r36, %r20;
    mul.lo.s32      %r39, %r37, %r21;
    add.s32         %r40, %r38, %r39;
    add.s32         %r41, %r40, 127;
    div.u32         %r42, %r41, 255;
    cvt.u16.u32     %rs3, %r42;
    st.global.u8    [%rd4+2], %rs3;

    // And the alpha it leaves behind: src + dst * (1 - src).
    ld.global.u8    %r43, [%rd4+3];
    mul.lo.s32      %r44, %r43, %r21;
    add.s32         %r45, %r44, 127;
    div.u32         %r46, %r45, 255;
    add.s32         %r47, %r46, %r20;
    cvt.u16.u32     %rs4, %r47;
    st.global.u8    [%rd4+3], %rs4;
BBDONE:
    ret;
}

.visible .entry blend_mask_bgra(
    .param .u64 dst,
    .param .u32 dst_pitch,
    .param .u64 mask,
    .param .u32 mask_pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 color,
    .param .u32 opacity
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<8>;
    .reg .b32   %r<52>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [dst];
    ld.param.u32    %r1, [dst_pitch];
    ld.param.u64    %rd2, [mask];
    ld.param.u32    %r2, [mask_pitch];
    ld.param.u32    %r3, [width];
    ld.param.u32    %r4, [height];
    ld.param.u32    %r5, [color];
    ld.param.u32    %r6, [opacity];

    mov.u32         %r7, %ctaid.x;
    mov.u32         %r8, %ntid.x;
    mov.u32         %r9, %tid.x;
    mad.lo.s32      %r10, %r7, %r8, %r9;
    mov.u32         %r11, %ctaid.y;
    mov.u32         %r12, %ntid.y;
    mov.u32         %r13, %tid.y;
    mad.lo.s32      %r14, %r11, %r12, %r13;

    setp.ge.u32     %p1, %r10, %r3;
    @%p1 bra        BMDONE;
    setp.ge.u32     %p2, %r14, %r4;
    @%p2 bra        BMDONE;

    mul.lo.s32      %r15, %r10, 4;
    mad.lo.s32      %r16, %r14, %r1, %r15;
    cvt.u64.u32     %rd3, %r16;
    add.s64         %rd4, %rd1, %rd3;
    mad.lo.s32      %r17, %r14, %r2, %r10;
    cvt.u64.u32     %rd5, %r17;
    add.s64         %rd6, %rd2, %rd5;

    // The glyph's coverage here, scaled by the layer's opacity.
    ld.global.u8    %r18, [%rd6];
    mul.lo.s32      %r19, %r18, %r6;
    add.s32         %r20, %r19, 127;
    div.u32         %r21, %r20, 255;
    sub.s32         %r22, 255, %r21;

    // One colour for every covered pixel, so its bytes come from the
    // parameter rather than from a surface.
    and.b32         %r23, %r5, 255;
    ld.global.u8    %r24, [%rd4];
    mul.lo.s32      %r25, %r23, %r21;
    mul.lo.s32      %r26, %r24, %r22;
    add.s32         %r27, %r25, %r26;
    add.s32         %r28, %r27, 127;
    div.u32         %r29, %r28, 255;
    cvt.u16.u32     %rs1, %r29;
    st.global.u8    [%rd4], %rs1;

    bfe.u32         %r30, %r5, 8, 8;
    ld.global.u8    %r31, [%rd4+1];
    mul.lo.s32      %r32, %r30, %r21;
    mul.lo.s32      %r33, %r31, %r22;
    add.s32         %r34, %r32, %r33;
    add.s32         %r35, %r34, 127;
    div.u32         %r36, %r35, 255;
    cvt.u16.u32     %rs2, %r36;
    st.global.u8    [%rd4+1], %rs2;

    bfe.u32         %r37, %r5, 16, 8;
    ld.global.u8    %r38, [%rd4+2];
    mul.lo.s32      %r39, %r37, %r21;
    mul.lo.s32      %r40, %r38, %r22;
    add.s32         %r41, %r39, %r40;
    add.s32         %r42, %r41, 127;
    div.u32         %r43, %r42, 255;
    cvt.u16.u32     %rs3, %r43;
    st.global.u8    [%rd4+2], %rs3;

    ld.global.u8    %r44, [%rd4+3];
    mul.lo.s32      %r45, %r44, %r22;
    add.s32         %r46, %r45, 127;
    div.u32         %r47, %r46, 255;
    add.s32         %r48, %r47, %r21;
    cvt.u16.u32     %rs4, %r48;
    st.global.u8    [%rd4+3], %rs4;
BMDONE:
    ret;
}
"#;

/// What [`crate::elements::CudaOrtDetector`] runs to fit a picture into a
/// detector's input: each thread one pixel of the model's `width` by
/// `height` input, writing it to three float planes — R, then G, then B,
/// each 0 to 1, as a YOLO model reads them.
///
/// The picture sits scaled inside the input at `offset`, `scaled` in size,
/// proportions kept; a pixel outside it is the grey Ultralytics trains on,
/// 114 of 255 (`0f3EE4E4E5`). Inside, each pixel is the mean of the source
/// pixels it covers — three by three where a 1080p picture is shrunk to
/// 640 — or, enlarging, of the one it falls in. The source sample at its
/// centre alone, one pixel in nine, was tried first: on a picture whose
/// bottle YOLOv10n scored about 0.6 from swscale's bilinear fitting it
/// scored 0.18, and the same inputs through the CPU's session scored the
/// same, so it was the fitting. The mean gives 0.57 for it, and every
/// object of that picture, of YOLOv10n and YOLO11n alike, scores within a
/// tenth of the CPU's — what is left is swscale's filter against a plain
/// mean. An NV12 pixel is averaged as Y', Cb and Cr — each source pixel
/// with the chroma sample it shares — and made RGB by the three rows
/// `fit_nv12` is handed, the same `(Y', Cb, Cr, 1)` rows `nv12_to_bgra`
/// reads; a BGRA one is averaged as it is.
///
/// Both read the picture the way it is shown: `src_w` and `src_h` are its
/// size turned, and each pixel of it turned is read from the stored pixel
/// the six sampling numbers name — `x_of_x * x + x_of_y * y + x_0`, and so
/// for y — which for a picture shown as stored are the pixel itself.
///
/// `best_class` reads what a YOLOv8 or YOLO11 model made of a batch where
/// it is, on the device — `rows` (four, then a score per class) by `boxes`
/// floats a picture — and writes each box as six floats: its centre, width
/// and height, and its best class's score and number. `selp` on a strict
/// `setp.gt` keeps the first of equal scores, as the CPU's reading does. A
/// thread a box, so neighbouring threads read neighbouring floats of each
/// row; what is copied down is six floats a box rather than every class's.
///
/// Kept apart from [`CONVERT_PTX`], and loaded by the detector alone: every
/// other CUDA element would otherwise pay for its JIT at construction.
#[cfg(feature = "ort-cuda")]
pub(super) const FIT_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry fit_nv12(
    .param .u64 dst,
    .param .u32 model_w,
    .param .u32 model_h,
    .param .u32 offset_x,
    .param .u32 offset_y,
    .param .u32 scaled_w,
    .param .u32 scaled_h,
    .param .u64 luma,
    .param .u32 luma_pitch,
    .param .u64 chroma,
    .param .u32 chroma_pitch,
    .param .u32 src_w,
    .param .u32 src_h,
    .param .s32 x_of_x,
    .param .s32 x_of_y,
    .param .s32 x_0,
    .param .s32 y_of_x,
    .param .s32 y_of_y,
    .param .s32 y_0,
    .param .f32 red_y,
    .param .f32 red_cb,
    .param .f32 red_cr,
    .param .f32 red_offset,
    .param .f32 green_y,
    .param .f32 green_cb,
    .param .f32 green_cr,
    .param .f32 green_offset,
    .param .f32 blue_y,
    .param .f32 blue_cb,
    .param .f32 blue_cr,
    .param .f32 blue_offset
)
{
    .reg .pred  %p<6>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<48>;
    .reg .f32   %f<32>;
    .reg .b64   %rd<16>;

    ld.param.u32    %r1, [model_w];
    ld.param.u32    %r2, [model_h];

    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %ntid.x;
    mov.u32         %r8, %tid.x;
    mad.lo.s32      %r9, %r6, %r7, %r8;
    mov.u32         %r10, %ctaid.y;
    mov.u32         %r11, %ntid.y;
    mov.u32         %r12, %tid.y;
    mad.lo.s32      %r13, %r10, %r11, %r12;

    setp.ge.u32     %p1, %r9, %r1;
    @%p1 bra        FIT_NV12_DONE;
    setp.ge.u32     %p2, %r13, %r2;
    @%p2 bra        FIT_NV12_DONE;

    // The margin's grey, unless the pixel is inside the picture.
    mov.f32         %f4, 0f3EE4E4E5;
    mov.f32         %f5, 0f3EE4E4E5;
    mov.f32         %f6, 0f3EE4E4E5;

    // dx, dy into the scaled picture; a pixel before its corner wraps round
    // to a huge value, which the one comparison each also refuses.
    ld.param.u32    %r3, [offset_x];
    ld.param.u32    %r4, [offset_y];
    ld.param.u32    %r5, [scaled_w];
    ld.param.u32    %r14, [scaled_h];
    sub.s32         %r15, %r9, %r3;
    sub.s32         %r16, %r13, %r4;
    setp.ge.u32     %p3, %r15, %r5;
    @%p3 bra        FIT_NV12_STORE;
    setp.ge.u32     %p4, %r16, %r14;
    @%p4 bra        FIT_NV12_STORE;

    // The source pixels the output pixel covers, [x0, x1) by [y0, y1):
    // x0 = d * src / scaled, x1 = (d + 1) * src / scaled rounded up, at
    // least one pixel and none past the edge. Averaged, so that a picture
    // shrunk to a third is read from every pixel rather than one in nine.
    ld.param.u32    %r17, [src_w];
    ld.param.u32    %r18, [src_h];
    mul.lo.s32      %r19, %r15, %r17;
    div.u32         %r21, %r19, %r5;
    add.s32         %r19, %r19, %r17;
    add.s32         %r19, %r19, %r5;
    sub.s32         %r19, %r19, 1;
    div.u32         %r36, %r19, %r5;
    min.u32         %r36, %r36, %r17;
    add.s32         %r20, %r21, 1;
    max.u32         %r36, %r36, %r20;
    mul.lo.s32      %r22, %r16, %r18;
    div.u32         %r24, %r22, %r14;
    add.s32         %r22, %r22, %r18;
    add.s32         %r22, %r22, %r14;
    sub.s32         %r22, %r22, 1;
    div.u32         %r37, %r22, %r14;
    min.u32         %r37, %r37, %r18;
    add.s32         %r23, %r24, 1;
    max.u32         %r37, %r37, %r23;

    ld.param.u64    %rd1, [luma];
    ld.param.u32    %r25, [luma_pitch];
    ld.param.u64    %rd2, [chroma];
    ld.param.u32    %r26, [chroma_pitch];

    // Where each pixel of the picture as shown is stored: x and y of it,
    // each one of the shown x, the shown y, and a start.
    ld.param.s32    %r40, [x_of_x];
    ld.param.s32    %r41, [x_of_y];
    ld.param.s32    %r42, [x_0];
    ld.param.s32    %r43, [y_of_x];
    ld.param.s32    %r44, [y_of_y];
    ld.param.s32    %r45, [y_0];

    // Sums of Y', Cb and Cr, each source pixel taking the chroma sample it
    // shares with three others.
    mov.f32         %f1, 0f00000000;
    mov.f32         %f2, 0f00000000;
    mov.f32         %f3, 0f00000000;
    mov.u32         %r38, 0;
    mov.u32         %r30, %r24;
FIT_NV12_ROW:
    mov.u32         %r31, %r21;
FIT_NV12_PIXEL:
    // The stored pixel (%r46, %r47) of shown pixel (%r31, %r30).
    mul.lo.s32      %r46, %r40, %r31;
    mad.lo.s32      %r46, %r41, %r30, %r46;
    add.s32         %r46, %r46, %r42;
    mul.lo.s32      %r47, %r43, %r31;
    mad.lo.s32      %r47, %r44, %r30, %r47;
    add.s32         %r47, %r47, %r45;
    mul.lo.s32      %r27, %r47, %r25;
    shr.u32         %r28, %r47, 1;
    mul.lo.s32      %r29, %r28, %r26;
    add.s32         %r32, %r27, %r46;
    cvt.u64.u32     %rd3, %r32;
    add.s64         %rd4, %rd1, %rd3;
    ld.global.u8    %rs1, [%rd4];
    shr.u32         %r33, %r46, 1;
    shl.b32         %r33, %r33, 1;
    add.s32         %r33, %r29, %r33;
    cvt.u64.u32     %rd5, %r33;
    add.s64         %rd6, %rd2, %rd5;
    ld.global.u8    %rs2, [%rd6];
    ld.global.u8    %rs3, [%rd6+1];
    cvt.u32.u16     %r34, %rs1;
    cvt.rn.f32.u32  %f7, %r34;
    add.f32         %f1, %f1, %f7;
    cvt.u32.u16     %r34, %rs2;
    cvt.rn.f32.u32  %f7, %r34;
    add.f32         %f2, %f2, %f7;
    cvt.u32.u16     %r34, %rs3;
    cvt.rn.f32.u32  %f7, %r34;
    add.f32         %f3, %f3, %f7;
    add.s32         %r38, %r38, 1;
    add.s32         %r31, %r31, 1;
    setp.lt.u32     %p5, %r31, %r36;
    @%p5 bra        FIT_NV12_PIXEL;
    add.s32         %r30, %r30, 1;
    setp.lt.u32     %p5, %r30, %r37;
    @%p5 bra        FIT_NV12_ROW;

    // The means, 0 to 1: each sum over 255 times the count.
    cvt.rn.f32.u32  %f8, %r38;
    mul.f32         %f8, %f8, 0f437F0000;
    rcp.rn.f32      %f8, %f8;
    mul.f32         %f1, %f1, %f8;
    mul.f32         %f2, %f2, %f8;
    mul.f32         %f3, %f3, %f8;

    ld.param.f32    %f10, [red_y];
    ld.param.f32    %f11, [red_cb];
    ld.param.f32    %f12, [red_cr];
    ld.param.f32    %f13, [red_offset];
    ld.param.f32    %f14, [green_y];
    ld.param.f32    %f15, [green_cb];
    ld.param.f32    %f16, [green_cr];
    ld.param.f32    %f17, [green_offset];
    ld.param.f32    %f18, [blue_y];
    ld.param.f32    %f19, [blue_cb];
    ld.param.f32    %f20, [blue_cr];
    ld.param.f32    %f21, [blue_offset];

    fma.rn.f32      %f4, %f10, %f1, %f13;
    fma.rn.f32      %f4, %f11, %f2, %f4;
    fma.rn.f32      %f4, %f12, %f3, %f4;
    fma.rn.f32      %f5, %f14, %f1, %f17;
    fma.rn.f32      %f5, %f15, %f2, %f5;
    fma.rn.f32      %f5, %f16, %f3, %f5;
    fma.rn.f32      %f6, %f18, %f1, %f21;
    fma.rn.f32      %f6, %f19, %f2, %f6;
    fma.rn.f32      %f6, %f20, %f3, %f6;
    add.sat.f32     %f4, %f4, 0f00000000;
    add.sat.f32     %f5, %f5, 0f00000000;
    add.sat.f32     %f6, %f6, 0f00000000;

FIT_NV12_STORE:
    // R at y * model_w + x, G a plane later, B two.
    ld.param.u64    %rd7, [dst];
    mad.lo.s32      %r34, %r13, %r1, %r9;
    mul.wide.u32    %rd8, %r34, 4;
    add.s64         %rd9, %rd7, %rd8;
    mul.lo.s32      %r35, %r1, %r2;
    mul.wide.u32    %rd10, %r35, 4;
    st.global.f32   [%rd9], %f4;
    add.s64         %rd9, %rd9, %rd10;
    st.global.f32   [%rd9], %f5;
    add.s64         %rd9, %rd9, %rd10;
    st.global.f32   [%rd9], %f6;

FIT_NV12_DONE:
    ret;
}

.visible .entry fit_bgra(
    .param .u64 dst,
    .param .u32 model_w,
    .param .u32 model_h,
    .param .u32 offset_x,
    .param .u32 offset_y,
    .param .u32 scaled_w,
    .param .u32 scaled_h,
    .param .u64 src,
    .param .u32 src_pitch,
    .param .u32 src_w,
    .param .u32 src_h,
    .param .s32 x_of_x,
    .param .s32 x_of_y,
    .param .s32 x_0,
    .param .s32 y_of_x,
    .param .s32 y_of_y,
    .param .s32 y_0
)
{
    .reg .pred  %p<6>;
    .reg .b16   %rs<4>;
    .reg .b32   %r<48>;
    .reg .f32   %f<8>;
    .reg .b64   %rd<16>;

    ld.param.u32    %r1, [model_w];
    ld.param.u32    %r2, [model_h];

    mov.u32         %r6, %ctaid.x;
    mov.u32         %r7, %ntid.x;
    mov.u32         %r8, %tid.x;
    mad.lo.s32      %r9, %r6, %r7, %r8;
    mov.u32         %r10, %ctaid.y;
    mov.u32         %r11, %ntid.y;
    mov.u32         %r12, %tid.y;
    mad.lo.s32      %r13, %r10, %r11, %r12;

    setp.ge.u32     %p1, %r9, %r1;
    @%p1 bra        FIT_BGRA_DONE;
    setp.ge.u32     %p2, %r13, %r2;
    @%p2 bra        FIT_BGRA_DONE;

    mov.f32         %f4, 0f3EE4E4E5;
    mov.f32         %f5, 0f3EE4E4E5;
    mov.f32         %f6, 0f3EE4E4E5;

    ld.param.u32    %r3, [offset_x];
    ld.param.u32    %r4, [offset_y];
    ld.param.u32    %r5, [scaled_w];
    ld.param.u32    %r14, [scaled_h];
    sub.s32         %r15, %r9, %r3;
    sub.s32         %r16, %r13, %r4;
    setp.ge.u32     %p3, %r15, %r5;
    @%p3 bra        FIT_BGRA_STORE;
    setp.ge.u32     %p4, %r16, %r14;
    @%p4 bra        FIT_BGRA_STORE;

    // The source pixels the output pixel covers, averaged, as `fit_nv12`
    // finds them.
    ld.param.u32    %r17, [src_w];
    ld.param.u32    %r18, [src_h];
    mul.lo.s32      %r19, %r15, %r17;
    div.u32         %r21, %r19, %r5;
    add.s32         %r19, %r19, %r17;
    add.s32         %r19, %r19, %r5;
    sub.s32         %r19, %r19, 1;
    div.u32         %r36, %r19, %r5;
    min.u32         %r36, %r36, %r17;
    add.s32         %r20, %r21, 1;
    max.u32         %r36, %r36, %r20;
    mul.lo.s32      %r22, %r16, %r18;
    div.u32         %r24, %r22, %r14;
    add.s32         %r22, %r22, %r18;
    add.s32         %r22, %r22, %r14;
    sub.s32         %r22, %r22, 1;
    div.u32         %r37, %r22, %r14;
    min.u32         %r37, %r37, %r18;
    add.s32         %r23, %r24, 1;
    max.u32         %r37, %r37, %r23;

    ld.param.u64    %rd1, [src];
    ld.param.u32    %r25, [src_pitch];
    ld.param.s32    %r40, [x_of_x];
    ld.param.s32    %r41, [x_of_y];
    ld.param.s32    %r42, [x_0];
    ld.param.s32    %r43, [y_of_x];
    ld.param.s32    %r44, [y_of_y];
    ld.param.s32    %r45, [y_0];

    // B, G, R in memory; summed as R, G, B.
    mov.f32         %f4, 0f00000000;
    mov.f32         %f5, 0f00000000;
    mov.f32         %f6, 0f00000000;
    mov.u32         %r38, 0;
    mov.u32         %r30, %r24;
FIT_BGRA_ROW:
    mov.u32         %r31, %r21;
FIT_BGRA_PIXEL:
    // The stored pixel (%r46, %r47) of shown pixel (%r31, %r30), as
    // `fit_nv12` finds it.
    mul.lo.s32      %r46, %r40, %r31;
    mad.lo.s32      %r46, %r41, %r30, %r46;
    add.s32         %r46, %r46, %r42;
    mul.lo.s32      %r47, %r43, %r31;
    mad.lo.s32      %r47, %r44, %r30, %r47;
    add.s32         %r47, %r47, %r45;
    mul.lo.s32      %r27, %r47, %r25;
    shl.b32         %r26, %r46, 2;
    add.s32         %r26, %r27, %r26;
    cvt.u64.u32     %rd2, %r26;
    add.s64         %rd3, %rd1, %rd2;
    ld.global.u8    %rs1, [%rd3];
    ld.global.u8    %rs2, [%rd3+1];
    ld.global.u8    %rs3, [%rd3+2];
    cvt.u32.u16     %r28, %rs3;
    cvt.rn.f32.u32  %f7, %r28;
    add.f32         %f4, %f4, %f7;
    cvt.u32.u16     %r28, %rs2;
    cvt.rn.f32.u32  %f7, %r28;
    add.f32         %f5, %f5, %f7;
    cvt.u32.u16     %r28, %rs1;
    cvt.rn.f32.u32  %f7, %r28;
    add.f32         %f6, %f6, %f7;
    add.s32         %r38, %r38, 1;
    add.s32         %r31, %r31, 1;
    setp.lt.u32     %p5, %r31, %r36;
    @%p5 bra        FIT_BGRA_PIXEL;
    add.s32         %r30, %r30, 1;
    setp.lt.u32     %p5, %r30, %r37;
    @%p5 bra        FIT_BGRA_ROW;

    cvt.rn.f32.u32  %f1, %r38;
    mul.f32         %f1, %f1, 0f437F0000;
    rcp.rn.f32      %f1, %f1;
    mul.f32         %f4, %f4, %f1;
    mul.f32         %f5, %f5, %f1;
    mul.f32         %f6, %f6, %f1;

FIT_BGRA_STORE:
    ld.param.u64    %rd7, [dst];
    mad.lo.s32      %r34, %r13, %r1, %r9;
    mul.wide.u32    %rd8, %r34, 4;
    add.s64         %rd9, %rd7, %rd8;
    mul.lo.s32      %r35, %r1, %r2;
    mul.wide.u32    %rd10, %r35, 4;
    st.global.f32   [%rd9], %f4;
    add.s64         %rd9, %rd9, %rd10;
    st.global.f32   [%rd9], %f5;
    add.s64         %rd9, %rd9, %rd10;
    st.global.f32   [%rd9], %f6;

FIT_BGRA_DONE:
    ret;
}

// Each float of `total`, in planes of `plane` floats, R G B over and over,
// becomes `x * scale + bias` by its plane's channel: a model's own
// normalisation, ImageNet's mean and deviation, applied to inputs the fits
// above wrote 0 to 1. A thread per float, a 16 by 16 block read as 256 in
// a row. ASCII only, as everything ptxas reads must be.
.visible .entry scale_planes(
    .param .u64 data,
    .param .u32 plane,
    .param .u32 total,
    .param .f32 scale_r,
    .param .f32 scale_g,
    .param .f32 scale_b,
    .param .f32 bias_r,
    .param .f32 bias_g,
    .param .f32 bias_b
)
{
    .reg .pred  %p<4>;
    .reg .b32   %r<12>;
    .reg .f32   %f<12>;
    .reg .b64   %rd<4>;

    ld.param.u64    %rd1, [data];
    ld.param.u32    %r1, [plane];
    ld.param.u32    %r2, [total];

    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %tid.y;
    mov.u32         %r5, %tid.x;
    shl.b32         %r6, %r3, 8;
    shl.b32         %r7, %r4, 4;
    add.s32         %r8, %r6, %r7;
    add.s32         %r9, %r8, %r5;

    setp.ge.u32     %p1, %r9, %r2;
    @%p1 bra        SCALE_DONE;

    div.u32         %r10, %r9, %r1;
    rem.u32         %r11, %r10, 3;

    ld.param.f32    %f1, [scale_r];
    ld.param.f32    %f2, [scale_g];
    ld.param.f32    %f3, [scale_b];
    ld.param.f32    %f4, [bias_r];
    ld.param.f32    %f5, [bias_g];
    ld.param.f32    %f6, [bias_b];
    setp.eq.u32     %p2, %r11, 1;
    setp.eq.u32     %p3, %r11, 2;
    selp.f32        %f7, %f2, %f1, %p2;
    selp.f32        %f7, %f3, %f7, %p3;
    selp.f32        %f8, %f5, %f4, %p2;
    selp.f32        %f8, %f6, %f8, %p3;

    mul.wide.u32    %rd2, %r9, 4;
    add.s64         %rd3, %rd1, %rd2;
    ld.global.f32   %f9, [%rd3];
    fma.rn.f32      %f10, %f9, %f7, %f8;
    st.global.f32   [%rd3], %f10;

SCALE_DONE:
    ret;
}

.visible .entry best_class(
    .param .u64 src,
    .param .u64 dst,
    .param .u32 rows,
    .param .u32 boxes,
    .param .u32 total
)
{
    .reg .pred  %p<4>;
    .reg .b32   %r<20>;
    .reg .f32   %f<8>;
    .reg .b64   %rd<8>;

    ld.param.u64    %rd1, [src];
    ld.param.u64    %rd2, [dst];
    ld.param.u32    %r1, [rows];
    ld.param.u32    %r2, [boxes];
    ld.param.u32    %r3, [total];

    mov.u32         %r4, %ctaid.x;
    mov.u32         %r5, %tid.y;
    mov.u32         %r6, %tid.x;
    shl.b32         %r7, %r4, 8;
    shl.b32         %r8, %r5, 4;
    add.s32         %r9, %r7, %r8;
    add.s32         %r9, %r9, %r6;

    setp.ge.u32     %p1, %r9, %r3;
    @%p1 bra        BEST_DONE;

    div.u32         %r10, %r9, %r2;
    rem.u32         %r11, %r9, %r2;
    mul.lo.u32      %r12, %r1, %r2;
    mul.wide.u32    %rd3, %r10, %r12;
    cvt.u64.u32     %rd4, %r11;
    add.s64         %rd3, %rd3, %rd4;
    shl.b64         %rd3, %rd3, 2;
    add.s64         %rd3, %rd1, %rd3;
    mul.wide.u32    %rd5, %r2, 4;

    ld.global.f32   %f1, [%rd3];
    add.s64         %rd6, %rd3, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.s64         %rd6, %rd6, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.s64         %rd6, %rd6, %rd5;
    ld.global.f32   %f4, [%rd6];

    mov.f32         %f5, 0fFF800000;
    mov.u32         %r13, 0;
    mov.u32         %r14, 0;
    sub.u32         %r15, %r1, 4;
BEST_LOOP:
    setp.ge.u32     %p2, %r14, %r15;
    @%p2 bra        BEST_WRITE;
    add.s64         %rd6, %rd6, %rd5;
    ld.global.f32   %f6, [%rd6];
    setp.gt.f32     %p3, %f6, %f5;
    selp.f32        %f5, %f6, %f5, %p3;
    selp.b32        %r13, %r14, %r13, %p3;
    add.u32         %r14, %r14, 1;
    bra             BEST_LOOP;

BEST_WRITE:
    cvt.rn.f32.u32  %f7, %r13;
    mul.wide.u32    %rd7, %r9, 24;
    add.s64         %rd7, %rd2, %rd7;
    st.global.f32   [%rd7], %f1;
    st.global.f32   [%rd7+4], %f2;
    st.global.f32   [%rd7+8], %f3;
    st.global.f32   [%rd7+12], %f4;
    st.global.f32   [%rd7+16], %f5;
    st.global.f32   [%rd7+20], %f7;

BEST_DONE:
    ret;
}
"#;

/// What `ObjectTracker` runs, with `cuda-visual-tracking`, to follow
/// objects by how they look on CUDA pictures — the GPU half of
/// `vision::track::dcf`, which the CPU runs for every other picture:
///
/// - `dcf_sample` samples each job's 64 by 64 neighbourhood of the
///   picture's brightness, between pixels, edges held, its logarithm;
/// - `dcf_normalize` makes each zero in mean and one in energy and fades
///   it out by the Hann window, its sums made in shared memory;
/// - between them and the next, cuFFT transforms the batch;
/// - `dcf_correlate` multiplies each spectrum by its object's filter,
///   numerator over denominator plus the regulariser;
/// - after cuFFT transforms the responses back, `dcf_peak` finds each
///   one's peak, to a fraction of a sample, and its peak-to-sidelobe ratio;
/// - `dcf_learn` takes a spectrum into its object's filter at a rate.
///
/// The logarithm is the hardware's approximate `lg2`, so the samples agree
/// with the CPU's to a few parts in a million, not exactly.
#[cfg(feature = "cuda-visual-tracking")]
pub(super) const DCF_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64
// Samples, for each job, a 64 by 64 neighbourhood of a picture's
// brightness: job j's four floats are its centre and size in pixels, and
// sample (u, v) is the picture between its pixels at the centre of that
// cell, edges held, its logarithm stored as the real part of a complex.
// `bgra` says whether the picture is BGRA, its brightness by BT.709's
// weights, or a plane of bytes. A thread per sample, a block per 16 by 16
// of them, a grid of 4 by 4 by jobs.
.visible .entry dcf_sample(
    .param .u64 src,
    .param .u32 pitch,
    .param .u32 src_w,
    .param .u32 src_h,
    .param .u32 bgra,
    .param .u64 jobs,
    .param .u64 out
)
{
    .reg .pred  %p<4>;
    .reg .b16   %rs<16>;
    .reg .b32   %r<32>;
    .reg .f32   %f<64>;
    .reg .b64   %rd<24>;

    ld.param.u64    %rd1, [src];
    ld.param.u32    %r1, [pitch];
    ld.param.u32    %r2, [src_w];
    ld.param.u32    %r3, [src_h];
    ld.param.u32    %r4, [bgra];
    ld.param.u64    %rd2, [jobs];
    ld.param.u64    %rd3, [out];

    mov.u32         %r5, %ctaid.x;
    mov.u32         %r6, %tid.x;
    shl.b32         %r5, %r5, 4;
    add.s32         %r7, %r5, %r6;
    mov.u32         %r8, %ctaid.y;
    mov.u32         %r9, %tid.y;
    shl.b32         %r8, %r8, 4;
    add.s32         %r10, %r8, %r9;
    mov.u32         %r11, %ctaid.z;

    mul.wide.u32    %rd4, %r11, 16;
    add.s64         %rd5, %rd2, %rd4;
    ld.global.f32   %f1, [%rd5];
    ld.global.f32   %f2, [%rd5+4];
    ld.global.f32   %f3, [%rd5+8];
    ld.global.f32   %f4, [%rd5+12];

    cvt.rn.f32.u32  %f5, %r7;
    add.f32         %f5, %f5, 0f3F000000;
    mul.f32         %f5, %f5, 0f3C800000;
    sub.f32         %f5, %f5, 0f3F000000;
    fma.rn.f32      %f6, %f5, %f3, %f1;
    cvt.rn.f32.u32  %f7, %r10;
    add.f32         %f7, %f7, 0f3F000000;
    mul.f32         %f7, %f7, 0f3C800000;
    sub.f32         %f7, %f7, 0f3F000000;
    fma.rn.f32      %f8, %f7, %f4, %f2;

    sub.f32         %f9, %f6, 0f3F000000;
    cvt.rn.f32.u32  %f10, %r2;
    sub.f32         %f10, %f10, 0f3F800000;
    max.f32         %f9, %f9, 0f00000000;
    min.f32         %f9, %f9, %f10;
    sub.f32         %f11, %f8, 0f3F000000;
    cvt.rn.f32.u32  %f12, %r3;
    sub.f32         %f12, %f12, 0f3F800000;
    max.f32         %f11, %f11, 0f00000000;
    min.f32         %f11, %f11, %f12;

    cvt.rmi.f32.f32 %f13, %f9;
    cvt.rzi.u32.f32 %r12, %f13;
    sub.f32         %f14, %f9, %f13;
    cvt.rmi.f32.f32 %f15, %f11;
    cvt.rzi.u32.f32 %r13, %f15;
    sub.f32         %f16, %f11, %f15;

    sub.s32         %r14, %r2, 1;
    add.s32         %r15, %r12, 1;
    min.u32         %r15, %r15, %r14;
    sub.s32         %r16, %r3, 1;
    add.s32         %r17, %r13, 1;
    min.u32         %r17, %r17, %r16;

    mul.wide.u32    %rd6, %r13, %r1;
    add.s64         %rd7, %rd1, %rd6;
    mul.wide.u32    %rd8, %r17, %r1;
    add.s64         %rd9, %rd1, %rd8;

    setp.ne.u32     %p1, %r4, 0;
    @%p1 bra        SAMPLE_BGRA;

    cvt.u64.u32     %rd10, %r12;
    cvt.u64.u32     %rd11, %r15;
    add.s64         %rd12, %rd7, %rd10;
    ld.global.u8    %rs1, [%rd12];
    cvt.rn.f32.u16  %f20, %rs1;
    add.s64         %rd13, %rd7, %rd11;
    ld.global.u8    %rs2, [%rd13];
    cvt.rn.f32.u16  %f21, %rs2;
    add.s64         %rd14, %rd9, %rd10;
    ld.global.u8    %rs3, [%rd14];
    cvt.rn.f32.u16  %f22, %rs3;
    add.s64         %rd15, %rd9, %rd11;
    ld.global.u8    %rs4, [%rd15];
    cvt.rn.f32.u16  %f23, %rs4;
    bra             SAMPLE_MIX;

SAMPLE_BGRA:
    shl.b32         %r18, %r12, 2;
    shl.b32         %r19, %r15, 2;
    cvt.u64.u32     %rd10, %r18;
    cvt.u64.u32     %rd11, %r19;
    add.s64         %rd12, %rd7, %rd10;
    ld.global.u8    %rs5, [%rd12];
    ld.global.u8    %rs6, [%rd12+1];
    ld.global.u8    %rs7, [%rd12+2];
    cvt.rn.f32.u16  %f40, %rs5;
    cvt.rn.f32.u16  %f41, %rs6;
    cvt.rn.f32.u16  %f42, %rs7;
    mul.f32         %f20, %f42, 0f3E59B3D0;
    fma.rn.f32      %f20, %f41, 0f3F371759, %f20;
    fma.rn.f32      %f20, %f40, 0f3D93DD98, %f20;
    add.s64         %rd13, %rd7, %rd11;
    ld.global.u8    %rs5, [%rd13];
    ld.global.u8    %rs6, [%rd13+1];
    ld.global.u8    %rs7, [%rd13+2];
    cvt.rn.f32.u16  %f43, %rs5;
    cvt.rn.f32.u16  %f44, %rs6;
    cvt.rn.f32.u16  %f45, %rs7;
    mul.f32         %f21, %f45, 0f3E59B3D0;
    fma.rn.f32      %f21, %f44, 0f3F371759, %f21;
    fma.rn.f32      %f21, %f43, 0f3D93DD98, %f21;
    add.s64         %rd14, %rd9, %rd10;
    ld.global.u8    %rs5, [%rd14];
    ld.global.u8    %rs6, [%rd14+1];
    ld.global.u8    %rs7, [%rd14+2];
    cvt.rn.f32.u16  %f46, %rs5;
    cvt.rn.f32.u16  %f47, %rs6;
    cvt.rn.f32.u16  %f48, %rs7;
    mul.f32         %f22, %f48, 0f3E59B3D0;
    fma.rn.f32      %f22, %f47, 0f3F371759, %f22;
    fma.rn.f32      %f22, %f46, 0f3D93DD98, %f22;
    add.s64         %rd15, %rd9, %rd11;
    ld.global.u8    %rs5, [%rd15];
    ld.global.u8    %rs6, [%rd15+1];
    ld.global.u8    %rs7, [%rd15+2];
    cvt.rn.f32.u16  %f49, %rs5;
    cvt.rn.f32.u16  %f50, %rs6;
    cvt.rn.f32.u16  %f51, %rs7;
    mul.f32         %f23, %f51, 0f3E59B3D0;
    fma.rn.f32      %f23, %f50, 0f3F371759, %f23;
    fma.rn.f32      %f23, %f49, 0f3D93DD98, %f23;

SAMPLE_MIX:
    sub.f32         %f24, %f21, %f20;
    fma.rn.f32      %f25, %f24, %f14, %f20;
    sub.f32         %f26, %f23, %f22;
    fma.rn.f32      %f27, %f26, %f14, %f22;
    sub.f32         %f28, %f27, %f25;
    fma.rn.f32      %f29, %f28, %f16, %f25;
    add.f32         %f30, %f29, 0f3F800000;
    lg2.approx.f32  %f31, %f30;
    mul.f32         %f32, %f31, 0f3F317218;

    shl.b32         %r20, %r10, 6;
    add.s32         %r21, %r20, %r7;
    shl.b32         %r22, %r11, 12;
    add.s32         %r23, %r22, %r21;
    mul.wide.u32    %rd16, %r23, 8;
    add.s64         %rd17, %rd3, %rd16;
    st.global.f32   [%rd17], %f32;
    st.global.f32   [%rd17+4], 0f00000000;
    ret;
}

// Makes each job's samples zero in mean and one in energy, then fades
// them out toward the edges by `window`: a block of 256 per job, each
// thread sixteen samples, the sums made in shared memory.
.visible .entry dcf_normalize(
    .param .u64 data,
    .param .u64 window
)
{
    .shared .align 4 .f32 s_sum[256];
    .shared .align 4 .f32 s_sq[256];
    .reg .pred  %p<4>;
    .reg .b32   %r<24>;
    .reg .f32   %f<24>;
    .reg .b64   %rd<12>;

    ld.param.u64    %rd1, [data];
    ld.param.u64    %rd2, [window];
    mov.u32         %r1, %ctaid.x;
    mov.u32         %r3, %tid.y;
    mov.u32         %r4, %tid.x;
    shl.b32         %r3, %r3, 4;
    add.s32         %r2, %r3, %r4;
    mul.wide.u32    %rd3, %r1, 32768;
    add.s64         %rd4, %rd1, %rd3;

    mov.f32         %f1, 0f00000000;
    mov.f32         %f2, 0f00000000;
    add.s32         %r5, %r2, 0;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 256;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 512;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 768;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 1024;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 1280;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 1536;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 1792;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 2048;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 2304;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 2560;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 2816;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 3072;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 3328;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 3584;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    add.s32         %r5, %r2, 3840;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    add.f32         %f1, %f1, %f3;
    fma.rn.f32      %f2, %f3, %f3, %f2;
    mov.u32         %r6, s_sum;
    mov.u32         %r7, s_sq;
    shl.b32         %r8, %r2, 2;
    add.s32         %r9, %r6, %r8;
    add.s32         %r10, %r7, %r8;
    st.shared.f32   [%r9], %f1;
    st.shared.f32   [%r10], %f2;
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 128;
    @!%p1 bra       NORM_0;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+512];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+512];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_0:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 64;
    @!%p1 bra       NORM_1;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+256];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+256];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_1:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 32;
    @!%p1 bra       NORM_2;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+128];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+128];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_2:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 16;
    @!%p1 bra       NORM_3;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+64];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+64];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_3:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 8;
    @!%p1 bra       NORM_4;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+32];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+32];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_4:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 4;
    @!%p1 bra       NORM_5;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+16];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+16];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_5:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 2;
    @!%p1 bra       NORM_6;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+8];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+8];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_6:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 1;
    @!%p1 bra       NORM_7;
    ld.shared.f32   %f4, [%r9];
    ld.shared.f32   %f5, [%r9+4];
    add.f32         %f4, %f4, %f5;
    st.shared.f32   [%r9], %f4;
    ld.shared.f32   %f6, [%r10];
    ld.shared.f32   %f7, [%r10+4];
    add.f32         %f6, %f6, %f7;
    st.shared.f32   [%r10], %f6;
NORM_7:
    bar.sync        0;
    ld.shared.f32   %f8, [%r6];
    ld.shared.f32   %f9, [%r7];
    mul.f32         %f10, %f8, 0f39800000;
    mul.f32         %f11, %f8, %f10;
    sub.f32         %f12, %f9, %f11;
    max.f32         %f12, %f12, 0f00000000;
    sqrt.rn.f32     %f13, %f12;
    setp.gt.f32     %p2, %f13, 0f358637BD;
    rcp.rn.f32      %f14, %f13;
    selp.f32        %f14, %f14, 0f00000000, %p2;
    add.s32         %r5, %r2, 0;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 256;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 512;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 768;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 1024;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 1280;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 1536;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 1792;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 2048;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 2304;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 2560;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 2816;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 3072;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 3328;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 3584;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    add.s32         %r5, %r2, 3840;
    mul.wide.u32    %rd5, %r5, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f3, [%rd6];
    mul.wide.u32    %rd7, %r5, 4;
    add.s64         %rd8, %rd2, %rd7;
    ld.global.f32   %f15, [%rd8];
    sub.f32         %f3, %f3, %f10;
    mul.f32         %f3, %f3, %f14;
    mul.f32         %f3, %f3, %f15;
    st.global.f32   [%rd6], %f3;
    ret;
}

// Multiplies each job's spectrum by its object's filter, numerator over
// denominator plus `lambda`: the correlation, once transformed back. A
// thread per frequency, 256 to a block read in a row.
.visible .entry dcf_correlate(
    .param .u64 spectra,
    .param .u64 slots,
    .param .u64 num,
    .param .u64 den,
    .param .u32 total,
    .param .f32 lambda
)
{
    .reg .pred  %p<4>;
    .reg .b32   %r<24>;
    .reg .f32   %f<24>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [spectra];
    ld.param.u64    %rd2, [slots];
    ld.param.u64    %rd3, [num];
    ld.param.u64    %rd4, [den];
    ld.param.u32    %r1, [total];
    ld.param.f32    %f1, [lambda];

    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %tid.y;
    mov.u32         %r4, %tid.x;
    shl.b32         %r2, %r2, 8;
    shl.b32         %r3, %r3, 4;
    add.s32         %r5, %r2, %r3;
    add.s32         %r5, %r5, %r4;
    setp.ge.u32     %p1, %r5, %r1;
    @%p1 bra        CORR_DONE;

    shr.u32         %r6, %r5, 12;
    and.b32         %r7, %r5, 4095;
    mul.wide.u32    %rd5, %r6, 4;
    add.s64         %rd6, %rd2, %rd5;
    ld.global.u32   %r8, [%rd6];
    shl.b32         %r9, %r8, 12;
    add.s32         %r9, %r9, %r7;

    mul.wide.u32    %rd7, %r5, 8;
    add.s64         %rd8, %rd1, %rd7;
    ld.global.f32   %f2, [%rd8];
    ld.global.f32   %f3, [%rd8+4];
    mul.wide.u32    %rd9, %r9, 8;
    add.s64         %rd10, %rd3, %rd9;
    ld.global.f32   %f4, [%rd10];
    ld.global.f32   %f5, [%rd10+4];
    mul.wide.u32    %rd11, %r9, 4;
    add.s64         %rd12, %rd4, %rd11;
    ld.global.f32   %f6, [%rd12];

    add.f32         %f7, %f6, %f1;
    rcp.rn.f32      %f8, %f7;
    mul.f32         %f9, %f4, %f8;
    mul.f32         %f10, %f5, %f8;
    mul.f32         %f11, %f9, %f2;
    mul.f32         %f12, %f10, %f3;
    sub.f32         %f13, %f11, %f12;
    mul.f32         %f14, %f9, %f3;
    fma.rn.f32      %f15, %f10, %f2, %f14;
    st.global.f32   [%rd8], %f13;
    st.global.f32   [%rd8+4], %f15;

CORR_DONE:
    ret;
}

// Finds each job's response peak and says how it stands out: a block of
// 256 per job finds the highest sample, then the mean and deviation of
// the rest beyond five samples of it either way, round the edges; thread 0
// writes the peak to a fraction of a sample by a parabola through its
// neighbours, its height, and its peak-to-sidelobe ratio.
.visible .entry dcf_peak(
    .param .u64 data,
    .param .u64 out
)
{
    .shared .align 4 .f32 s_val[256];
    .shared .align 4 .u32 s_idx[256];
    .shared .align 4 .f32 s_sum[256];
    .shared .align 4 .f32 s_sq[256];
    .shared .align 4 .f32 s_cnt[256];
    .reg .pred  %p<8>;
    .reg .b32   %r<48>;
    .reg .f32   %f<48>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [data];
    ld.param.u64    %rd2, [out];
    mov.u32         %r1, %ctaid.x;
    mov.u32         %r3, %tid.y;
    mov.u32         %r4, %tid.x;
    shl.b32         %r3, %r3, 4;
    add.s32         %r2, %r3, %r4;
    mul.wide.u32    %rd3, %r1, 32768;
    add.s64         %rd4, %rd1, %rd3;

    mov.f32         %f1, 0fFF800000;
    mov.u32         %r5, 0;
    add.s32         %r6, %r2, 0;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 256;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 512;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 768;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 1024;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 1280;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 1536;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 1792;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 2048;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 2304;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 2560;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 2816;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 3072;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 3328;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 3584;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    add.s32         %r6, %r2, 3840;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    setp.gt.f32     %p2, %f2, %f1;
    selp.f32        %f1, %f2, %f1, %p2;
    selp.b32        %r5, %r6, %r5, %p2;
    mov.u32         %r7, s_val;
    mov.u32         %r8, s_idx;
    shl.b32         %r9, %r2, 2;
    add.s32         %r10, %r7, %r9;
    add.s32         %r11, %r8, %r9;
    st.shared.f32   [%r10], %f1;
    st.shared.u32   [%r11], %r5;
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 128;
    @!%p1 bra       MAX_0;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+512];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+512];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_0:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 64;
    @!%p1 bra       MAX_1;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+256];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+256];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_1:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 32;
    @!%p1 bra       MAX_2;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+128];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+128];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_2:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 16;
    @!%p1 bra       MAX_3;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+64];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+64];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_3:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 8;
    @!%p1 bra       MAX_4;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+32];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+32];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_4:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 4;
    @!%p1 bra       MAX_5;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+16];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+16];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_5:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 2;
    @!%p1 bra       MAX_6;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+8];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+8];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_6:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 1;
    @!%p1 bra       MAX_7;
    ld.shared.f32   %f3, [%r10];
    ld.shared.f32   %f4, [%r10+4];
    ld.shared.u32   %r12, [%r11];
    ld.shared.u32   %r13, [%r11+4];
    setp.gt.f32     %p3, %f4, %f3;
    selp.f32        %f3, %f4, %f3, %p3;
    selp.b32        %r12, %r13, %r12, %p3;
    st.shared.f32   [%r10], %f3;
    st.shared.u32   [%r11], %r12;
MAX_7:
    bar.sync        0;
    ld.shared.f32   %f5, [%r7];
    ld.shared.u32   %r14, [%r8];
    and.b32         %r15, %r14, 63;
    shr.u32         %r16, %r14, 6;

    mov.f32         %f6, 0f00000000;
    mov.f32         %f7, 0f00000000;
    mov.f32         %f8, 0f00000000;
    add.s32         %r6, %r2, 0;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_0;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_0:
    add.s32         %r6, %r2, 256;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_1;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_1:
    add.s32         %r6, %r2, 512;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_2;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_2:
    add.s32         %r6, %r2, 768;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_3;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_3:
    add.s32         %r6, %r2, 1024;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_4;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_4:
    add.s32         %r6, %r2, 1280;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_5;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_5:
    add.s32         %r6, %r2, 1536;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_6;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_6:
    add.s32         %r6, %r2, 1792;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_7;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_7:
    add.s32         %r6, %r2, 2048;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_8;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_8:
    add.s32         %r6, %r2, 2304;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_9;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_9:
    add.s32         %r6, %r2, 2560;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_10;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_10:
    add.s32         %r6, %r2, 2816;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_11;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_11:
    add.s32         %r6, %r2, 3072;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_12;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_12:
    add.s32         %r6, %r2, 3328;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_13;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_13:
    add.s32         %r6, %r2, 3584;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_14;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_14:
    add.s32         %r6, %r2, 3840;
    and.b32         %r17, %r6, 63;
    shr.u32         %r18, %r6, 6;
    sub.s32         %r19, %r17, %r15;
    and.b32         %r19, %r19, 63;
    mov.u32         %r20, 64;
    sub.s32         %r20, %r20, %r19;
    min.u32         %r19, %r19, %r20;
    sub.s32         %r21, %r18, %r16;
    and.b32         %r21, %r21, 63;
    mov.u32         %r22, 64;
    sub.s32         %r22, %r22, %r21;
    min.u32         %r21, %r21, %r22;
    setp.le.u32     %p4, %r19, 5;
    setp.le.u32     %p5, %r21, 5;
    and.pred        %p6, %p4, %p5;
    @%p6 bra        SIDE_15;
    mul.wide.u32    %rd5, %r6, 8;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.f32   %f2, [%rd6];
    add.f32         %f6, %f6, %f2;
    fma.rn.f32      %f7, %f2, %f2, %f7;
    add.f32         %f8, %f8, 0f3F800000;
SIDE_15:
    mov.u32         %r23, s_sum;
    mov.u32         %r24, s_sq;
    mov.u32         %r25, s_cnt;
    add.s32         %r26, %r23, %r9;
    add.s32         %r27, %r24, %r9;
    add.s32         %r28, %r25, %r9;
    st.shared.f32   [%r26], %f6;
    st.shared.f32   [%r27], %f7;
    st.shared.f32   [%r28], %f8;
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 128;
    @!%p1 bra       SUM_0;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+512];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+512];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+512];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_0:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 64;
    @!%p1 bra       SUM_1;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+256];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+256];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+256];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_1:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 32;
    @!%p1 bra       SUM_2;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+128];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+128];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+128];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_2:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 16;
    @!%p1 bra       SUM_3;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+64];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+64];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+64];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_3:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 8;
    @!%p1 bra       SUM_4;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+32];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+32];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+32];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_4:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 4;
    @!%p1 bra       SUM_5;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+16];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+16];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+16];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_5:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 2;
    @!%p1 bra       SUM_6;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+8];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+8];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+8];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_6:
    bar.sync        0;
    setp.lt.u32     %p1, %r2, 1;
    @!%p1 bra       SUM_7;
    ld.shared.f32   %f9, [%r26];
    ld.shared.f32   %f10, [%r26+4];
    add.f32         %f9, %f9, %f10;
    st.shared.f32   [%r26], %f9;
    ld.shared.f32   %f11, [%r27];
    ld.shared.f32   %f12, [%r27+4];
    add.f32         %f11, %f11, %f12;
    st.shared.f32   [%r27], %f11;
    ld.shared.f32   %f13, [%r28];
    ld.shared.f32   %f14, [%r28+4];
    add.f32         %f13, %f13, %f14;
    st.shared.f32   [%r28], %f13;
SUM_7:
    bar.sync        0;
    setp.ne.u32     %p7, %r2, 0;
    @%p7 bra        PEAK_DONE;

    sub.s32         %r29, %r15, 1;
    and.b32         %r29, %r29, 63;
    add.s32         %r30, %r15, 1;
    and.b32         %r30, %r30, 63;
    sub.s32         %r31, %r16, 1;
    and.b32         %r31, %r31, 63;
    add.s32         %r32, %r16, 1;
    and.b32         %r32, %r32, 63;
    shl.b32         %r33, %r16, 6;
    add.s32         %r33, %r33, %r29;
    mul.wide.u32    %rd7, %r33, 8;
    add.s64         %rd8, %rd4, %rd7;
    ld.global.f32   %f15, [%rd8];
    shl.b32         %r33, %r16, 6;
    add.s32         %r33, %r33, %r30;
    mul.wide.u32    %rd7, %r33, 8;
    add.s64         %rd8, %rd4, %rd7;
    ld.global.f32   %f16, [%rd8];
    shl.b32         %r33, %r31, 6;
    add.s32         %r33, %r33, %r15;
    mul.wide.u32    %rd7, %r33, 8;
    add.s64         %rd8, %rd4, %rd7;
    ld.global.f32   %f17, [%rd8];
    shl.b32         %r33, %r32, 6;
    add.s32         %r33, %r33, %r15;
    mul.wide.u32    %rd7, %r33, 8;
    add.s64         %rd8, %rd4, %rd7;
    ld.global.f32   %f18, [%rd8];
    // x: 0.5 (before - after) / (before - 2 top + after), 0 where flat.
    add.f32         %f19, %f5, %f5;
    sub.f32         %f20, %f15, %f19;
    add.f32         %f20, %f20, %f16;
    sub.f32         %f21, %f15, %f16;
    mul.f32         %f21, %f21, 0f3F000000;
    div.rn.f32      %f22, %f21, %f20;
    abs.f32         %f23, %f20;
    setp.lt.f32     %p2, %f23, 0f2B8CBCCC;
    selp.f32        %f22, 0f00000000, %f22, %p2;
    cvt.rn.f32.u32  %f24, %r15;
    add.f32         %f24, %f24, %f22;

    sub.f32         %f25, %f17, %f19;
    add.f32         %f25, %f25, %f18;
    sub.f32         %f26, %f17, %f18;
    mul.f32         %f26, %f26, 0f3F000000;
    div.rn.f32      %f27, %f26, %f25;
    abs.f32         %f28, %f25;
    setp.lt.f32     %p3, %f28, 0f2B8CBCCC;
    selp.f32        %f27, 0f00000000, %f27, %p3;
    cvt.rn.f32.u32  %f29, %r16;
    add.f32         %f29, %f29, %f27;

    ld.shared.f32   %f30, [%r23];
    ld.shared.f32   %f31, [%r24];
    ld.shared.f32   %f32, [%r25];
    div.rn.f32      %f33, %f30, %f32;
    div.rn.f32      %f34, %f31, %f32;
    mul.f32         %f35, %f33, %f33;
    sub.f32         %f36, %f34, %f35;
    max.f32         %f36, %f36, 0f2B8CBCCC;
    sqrt.rn.f32     %f37, %f36;
    sub.f32         %f38, %f5, %f33;
    div.rn.f32      %f39, %f38, %f37;

    mul.wide.u32    %rd9, %r1, 16;
    add.s64         %rd10, %rd2, %rd9;
    st.global.f32   [%rd10], %f24;
    st.global.f32   [%rd10+4], %f29;
    st.global.f32   [%rd10+8], %f5;
    st.global.f32   [%rd10+12], %f39;

PEAK_DONE:
    ret;
}

// Takes each job's spectrum into its object's filter at `rate`: the
// numerator toward the wanted response times the spectrum's conjugate,
// the denominator toward the spectrum's energy. A thread per frequency.
.visible .entry dcf_learn(
    .param .u64 spectra,
    .param .u64 slots,
    .param .u64 num,
    .param .u64 den,
    .param .u64 target,
    .param .u32 total,
    .param .f32 rate
)
{
    .reg .pred  %p<4>;
    .reg .b32   %r<24>;
    .reg .f32   %f<32>;
    .reg .b64   %rd<20>;

    ld.param.u64    %rd1, [spectra];
    ld.param.u64    %rd2, [slots];
    ld.param.u64    %rd3, [num];
    ld.param.u64    %rd4, [den];
    ld.param.u64    %rd5, [target];
    ld.param.u32    %r1, [total];
    ld.param.f32    %f1, [rate];

    mov.u32         %r2, %ctaid.x;
    mov.u32         %r3, %tid.y;
    mov.u32         %r4, %tid.x;
    shl.b32         %r2, %r2, 8;
    shl.b32         %r3, %r3, 4;
    add.s32         %r5, %r2, %r3;
    add.s32         %r5, %r5, %r4;
    setp.ge.u32     %p1, %r5, %r1;
    @%p1 bra        LEARN_DONE;

    shr.u32         %r6, %r5, 12;
    and.b32         %r7, %r5, 4095;
    mul.wide.u32    %rd6, %r6, 4;
    add.s64         %rd7, %rd2, %rd6;
    ld.global.u32   %r8, [%rd7];
    shl.b32         %r9, %r8, 12;
    add.s32         %r9, %r9, %r7;

    mul.wide.u32    %rd8, %r5, 8;
    add.s64         %rd9, %rd1, %rd8;
    ld.global.f32   %f2, [%rd9];
    ld.global.f32   %f3, [%rd9+4];
    mul.wide.u32    %rd10, %r7, 8;
    add.s64         %rd11, %rd5, %rd10;
    ld.global.f32   %f4, [%rd11];
    ld.global.f32   %f5, [%rd11+4];

    mul.f32         %f6, %f4, %f2;
    fma.rn.f32      %f6, %f5, %f3, %f6;
    mul.f32         %f7, %f5, %f2;
    mul.f32         %f8, %f4, %f3;
    sub.f32         %f7, %f7, %f8;
    mul.f32         %f9, %f2, %f2;
    fma.rn.f32      %f9, %f3, %f3, %f9;

    sub.f32         %f10, 0f3F800000, %f1;
    mul.wide.u32    %rd12, %r9, 8;
    add.s64         %rd13, %rd3, %rd12;
    ld.global.f32   %f11, [%rd13];
    ld.global.f32   %f12, [%rd13+4];
    mul.f32         %f11, %f11, %f10;
    fma.rn.f32      %f11, %f6, %f1, %f11;
    mul.f32         %f12, %f12, %f10;
    fma.rn.f32      %f12, %f7, %f1, %f12;
    st.global.f32   [%rd13], %f11;
    st.global.f32   [%rd13+4], %f12;
    mul.wide.u32    %rd14, %r9, 4;
    add.s64         %rd15, %rd4, %rd14;
    ld.global.f32   %f13, [%rd15];
    mul.f32         %f13, %f13, %f10;
    fma.rn.f32      %f13, %f9, %f1, %f13;
    st.global.f32   [%rd15], %f13;

LEARN_DONE:
    ret;
}
"#;

/// What [`crate::elements::CudaDetectionOverlay`] runs to hide a box: the
/// box, on one plane of a picture — `width` by `height` samples of
/// `channels` bytes each, from `plane`, rows `pitch` apart — cut into
/// `cells_x` by `cells_y` cells as near equal as whole samples allow, cell
/// `c` of `n` along a side `length` long spanning `[c * length / n,
/// (c + 1) * length / n)`.
///
/// - `cell_means`: a thread a cell, summing each channel of its samples row
///   by row into `means` — `cells_y` rows of `cells_x` cells of `channels`
///   floats — each sum times the reciprocal of the count.
/// - `cell_paint`: a thread a sample, writing over it its cell's mean, or
///   with `smooth` the means of the four cells around it blended by where
///   it sits between their centres — with `ellipse`, only a sample whose
///   centre is inside the ellipse touching the plane's four sides. The cell that holds sample `x` is
///   `((x + 1) * n - 1) / length`, and the sample sits at
///   `(x + 0.5) * n / length - 0.5` measured in cells.
///
/// Every multiply, add and division is rounded as written (`.rn`), and
/// each byte rounded to nearest even (`cvt.rni`), so that the CPU's
/// overlay, computing the same in `f32` in the same order, writes the same
/// bytes.
///
/// Kept apart from [`BLEND_PTX`] and loaded by an overlay that hides,
/// alone: every other CUDA element would otherwise pay for its JIT.
pub(super) const REDACT_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry cell_means(
    .param .u64 plane,
    .param .u32 pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 channels,
    .param .u32 cells_x,
    .param .u32 cells_y,
    .param .u64 means
)
{
    .reg .pred  %p<8>;
    .reg .b16   %rs<2>;
    .reg .b32   %r<32>;
    .reg .f32   %f<8>;
    .reg .b64   %rd<12>;

    ld.param.u64    %rd1, [plane];
    ld.param.u32    %r1, [pitch];
    ld.param.u32    %r2, [width];
    ld.param.u32    %r3, [height];
    ld.param.u32    %r4, [channels];
    ld.param.u32    %r5, [cells_x];
    ld.param.u32    %r6, [cells_y];
    ld.param.u64    %rd2, [means];

    // The cell: a 16 by 16 block read as 256 threads in a row.
    mov.u32         %r7, %ctaid.x;
    mov.u32         %r8, %tid.y;
    mov.u32         %r9, %tid.x;
    shl.b32         %r7, %r7, 8;
    shl.b32         %r8, %r8, 4;
    add.s32         %r10, %r7, %r8;
    add.s32         %r10, %r10, %r9;
    mul.lo.u32      %r11, %r5, %r6;
    setp.ge.u32     %p1, %r10, %r11;
    @%p1 bra        MEANS_DONE;

    rem.u32         %r12, %r10, %r5;
    div.u32         %r13, %r10, %r5;
    mul.lo.u32      %r14, %r12, %r2;
    div.u32         %r14, %r14, %r5;
    add.u32         %r15, %r12, 1;
    mul.lo.u32      %r15, %r15, %r2;
    div.u32         %r15, %r15, %r5;
    mul.lo.u32      %r16, %r13, %r3;
    div.u32         %r16, %r16, %r6;
    add.u32         %r17, %r13, 1;
    mul.lo.u32      %r17, %r17, %r3;
    div.u32         %r17, %r17, %r6;

    mov.f32         %f1, 0f00000000;
    mov.f32         %f2, 0f00000000;
    mov.f32         %f3, 0f00000000;
    mov.f32         %f4, 0f00000000;
    setp.gt.u32     %p2, %r4, 1;
    setp.gt.u32     %p3, %r4, 2;
    setp.gt.u32     %p4, %r4, 3;

    mov.u32         %r18, %r16;
MEANS_ROW:
    setp.ge.u32     %p5, %r18, %r17;
    @%p5 bra        MEANS_SUMMED;
    mul.wide.u32    %rd3, %r18, %r1;
    add.s64         %rd4, %rd1, %rd3;
    mov.u32         %r19, %r14;
MEANS_SAMPLE:
    setp.ge.u32     %p6, %r19, %r15;
    @%p6 bra        MEANS_NEXT_ROW;
    mul.lo.u32      %r20, %r19, %r4;
    cvt.u64.u32     %rd5, %r20;
    add.s64         %rd6, %rd4, %rd5;
    ld.global.u8    %rs1, [%rd6];
    cvt.u32.u16     %r21, %rs1;
    cvt.rn.f32.u32  %f5, %r21;
    add.rn.f32      %f1, %f1, %f5;
    @!%p2 bra       MEANS_STEP;
    ld.global.u8    %rs1, [%rd6+1];
    cvt.u32.u16     %r21, %rs1;
    cvt.rn.f32.u32  %f5, %r21;
    add.rn.f32      %f2, %f2, %f5;
    @!%p3 bra       MEANS_STEP;
    ld.global.u8    %rs1, [%rd6+2];
    cvt.u32.u16     %r21, %rs1;
    cvt.rn.f32.u32  %f5, %r21;
    add.rn.f32      %f3, %f3, %f5;
    @!%p4 bra       MEANS_STEP;
    ld.global.u8    %rs1, [%rd6+3];
    cvt.u32.u16     %r21, %rs1;
    cvt.rn.f32.u32  %f5, %r21;
    add.rn.f32      %f4, %f4, %f5;
MEANS_STEP:
    add.u32         %r19, %r19, 1;
    bra             MEANS_SAMPLE;
MEANS_NEXT_ROW:
    add.u32         %r18, %r18, 1;
    bra             MEANS_ROW;

MEANS_SUMMED:
    sub.u32         %r22, %r15, %r14;
    sub.u32         %r23, %r17, %r16;
    mul.lo.u32      %r22, %r22, %r23;
    cvt.rn.f32.u32  %f6, %r22;
    rcp.rn.f32      %f6, %f6;
    mul.lo.u32      %r24, %r10, %r4;
    mul.wide.u32    %rd7, %r24, 4;
    add.s64         %rd8, %rd2, %rd7;
    mul.rn.f32      %f1, %f1, %f6;
    st.global.f32   [%rd8], %f1;
    @!%p2 bra       MEANS_DONE;
    mul.rn.f32      %f2, %f2, %f6;
    st.global.f32   [%rd8+4], %f2;
    @!%p3 bra       MEANS_DONE;
    mul.rn.f32      %f3, %f3, %f6;
    st.global.f32   [%rd8+8], %f3;
    @!%p4 bra       MEANS_DONE;
    mul.rn.f32      %f4, %f4, %f6;
    st.global.f32   [%rd8+12], %f4;

MEANS_DONE:
    ret;
}

.visible .entry cell_paint(
    .param .u64 plane,
    .param .u32 pitch,
    .param .u32 width,
    .param .u32 height,
    .param .u32 channels,
    .param .u32 cells_x,
    .param .u32 cells_y,
    .param .u64 means,
    .param .u32 smooth,
    .param .u32 ellipse
)
{
    .reg .pred  %p<8>;
    .reg .b16   %rs<2>;
    .reg .b32   %r<40>;
    .reg .f32   %f<32>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd1, [plane];
    ld.param.u32    %r1, [pitch];
    ld.param.u32    %r2, [width];
    ld.param.u32    %r3, [height];
    ld.param.u32    %r4, [channels];
    ld.param.u32    %r5, [cells_x];
    ld.param.u32    %r6, [cells_y];
    ld.param.u64    %rd2, [means];
    ld.param.u32    %r7, [smooth];

    mov.u32         %r8, %ctaid.x;
    mov.u32         %r9, %ntid.x;
    mov.u32         %r11, %tid.x;
    mad.lo.s32      %r10, %r8, %r9, %r11;
    mov.u32         %r8, %ctaid.y;
    mov.u32         %r9, %ntid.y;
    mov.u32         %r11, %tid.y;
    mad.lo.s32      %r13, %r8, %r9, %r11;
    setp.ge.u32     %p1, %r10, %r2;
    @%p1 bra        PAINT_DONE;
    setp.ge.u32     %p1, %r13, %r3;
    @%p1 bra        PAINT_DONE;

    // Cut to the ellipse inside the plane where asked: a sample whose
    // centre is outside it is left as it is. dx = (x + 0.5) * 2 / width - 1,
    // and dy so down, each step rounded as the CPU's overlay rounds it.
    ld.param.u32    %r31, [ellipse];
    setp.eq.u32     %p4, %r31, 0;
    @%p4 bra        PAINT_INSIDE;
    cvt.rn.f32.u32  %f27, %r10;
    add.rn.f32      %f27, %f27, 0f3F000000;
    mul.rn.f32      %f27, %f27, 0f40000000;
    cvt.rn.f32.u32  %f28, %r2;
    div.rn.f32      %f27, %f27, %f28;
    sub.rn.f32      %f27, %f27, 0f3F800000;
    cvt.rn.f32.u32  %f29, %r13;
    add.rn.f32      %f29, %f29, 0f3F000000;
    mul.rn.f32      %f29, %f29, 0f40000000;
    cvt.rn.f32.u32  %f28, %r3;
    div.rn.f32      %f29, %f29, %f28;
    sub.rn.f32      %f29, %f29, 0f3F800000;
    mul.rn.f32      %f27, %f27, %f27;
    mul.rn.f32      %f29, %f29, %f29;
    add.rn.f32      %f27, %f27, %f29;
    setp.gt.f32     %p4, %f27, 0f3F800000;
    @%p4 bra        PAINT_DONE;
PAINT_INSIDE:

    // The sample: y * pitch + x * channels bytes into the plane.
    mul.wide.u32    %rd3, %r13, %r1;
    mul.lo.u32      %r14, %r10, %r4;
    cvt.u64.u32     %rd4, %r14;
    add.s64         %rd3, %rd3, %rd4;
    add.s64         %rd5, %rd1, %rd3;

    setp.ne.u32     %p2, %r7, 0;
    @%p2 bra        PAINT_SMOOTH;

    add.u32         %r15, %r10, 1;
    mul.lo.u32      %r15, %r15, %r5;
    sub.u32         %r15, %r15, 1;
    div.u32         %r15, %r15, %r2;
    add.u32         %r16, %r13, 1;
    mul.lo.u32      %r16, %r16, %r6;
    sub.u32         %r16, %r16, 1;
    div.u32         %r16, %r16, %r3;
    mad.lo.u32      %r17, %r16, %r5, %r15;
    mul.lo.u32      %r17, %r17, %r4;
    mul.wide.u32    %rd6, %r17, 4;
    add.s64         %rd7, %rd2, %rd6;
    mov.u32         %r18, 0;
PAINT_FLAT:
    setp.ge.u32     %p3, %r18, %r4;
    @%p3 bra        PAINT_DONE;
    mul.wide.u32    %rd8, %r18, 4;
    add.s64         %rd9, %rd7, %rd8;
    ld.global.f32   %f1, [%rd9];
    cvt.rni.u16.f32 %rs1, %f1;
    cvt.u64.u32     %rd10, %r18;
    add.s64         %rd11, %rd5, %rd10;
    st.global.u8    [%rd11], %rs1;
    add.u32         %r18, %r18, 1;
    bra             PAINT_FLAT;

PAINT_SMOOTH:
    // Where the sample sits among the cells' centres, across: c0 and c1
    // the cells either side, t how far it is from c0's centre to c1's.
    cvt.rn.f32.u32  %f2, %r10;
    add.rn.f32      %f2, %f2, 0f3F000000;
    cvt.rn.f32.u32  %f3, %r5;
    mul.rn.f32      %f2, %f2, %f3;
    cvt.rn.f32.u32  %f4, %r2;
    div.rn.f32      %f2, %f2, %f4;
    sub.rn.f32      %f2, %f2, 0f3F000000;
    sub.u32         %r20, %r5, 1;
    cvt.rn.f32.u32  %f5, %r20;
    max.f32         %f2, %f2, 0f00000000;
    min.f32         %f2, %f2, %f5;
    cvt.rmi.u32.f32 %r21, %f2;
    cvt.rn.f32.u32  %f6, %r21;
    sub.rn.f32      %f7, %f2, %f6;
    add.u32         %r22, %r21, 1;
    min.u32         %r22, %r22, %r20;

    // And down: r0, r1 and s.
    cvt.rn.f32.u32  %f8, %r13;
    add.rn.f32      %f8, %f8, 0f3F000000;
    cvt.rn.f32.u32  %f9, %r6;
    mul.rn.f32      %f8, %f8, %f9;
    cvt.rn.f32.u32  %f10, %r3;
    div.rn.f32      %f8, %f8, %f10;
    sub.rn.f32      %f8, %f8, 0f3F000000;
    sub.u32         %r23, %r6, 1;
    cvt.rn.f32.u32  %f11, %r23;
    max.f32         %f8, %f8, 0f00000000;
    min.f32         %f8, %f8, %f11;
    cvt.rmi.u32.f32 %r24, %f8;
    cvt.rn.f32.u32  %f12, %r24;
    sub.rn.f32      %f13, %f8, %f12;
    add.u32         %r25, %r24, 1;
    min.u32         %r25, %r25, %r23;

    // The four cells' first floats: (r0, c0), (r0, c1), (r1, c0), (r1, c1).
    mad.lo.u32      %r26, %r24, %r5, %r21;
    mul.lo.u32      %r26, %r26, %r4;
    mad.lo.u32      %r27, %r24, %r5, %r22;
    mul.lo.u32      %r27, %r27, %r4;
    mad.lo.u32      %r28, %r25, %r5, %r21;
    mul.lo.u32      %r28, %r28, %r4;
    mad.lo.u32      %r29, %r25, %r5, %r22;
    mul.lo.u32      %r29, %r29, %r4;
    mov.u32         %r18, 0;
PAINT_BLEND:
    setp.ge.u32     %p3, %r18, %r4;
    @%p3 bra        PAINT_DONE;
    add.u32         %r30, %r26, %r18;
    mul.wide.u32    %rd8, %r30, 4;
    add.s64         %rd9, %rd2, %rd8;
    ld.global.f32   %f20, [%rd9];
    add.u32         %r30, %r27, %r18;
    mul.wide.u32    %rd8, %r30, 4;
    add.s64         %rd9, %rd2, %rd8;
    ld.global.f32   %f21, [%rd9];
    add.u32         %r30, %r28, %r18;
    mul.wide.u32    %rd8, %r30, 4;
    add.s64         %rd9, %rd2, %rd8;
    ld.global.f32   %f22, [%rd9];
    add.u32         %r30, %r29, %r18;
    mul.wide.u32    %rd8, %r30, 4;
    add.s64         %rd9, %rd2, %rd8;
    ld.global.f32   %f23, [%rd9];
    sub.rn.f32      %f24, %f21, %f20;
    mul.rn.f32      %f24, %f24, %f7;
    add.rn.f32      %f24, %f20, %f24;
    sub.rn.f32      %f25, %f23, %f22;
    mul.rn.f32      %f25, %f25, %f7;
    add.rn.f32      %f25, %f22, %f25;
    sub.rn.f32      %f26, %f25, %f24;
    mul.rn.f32      %f26, %f26, %f13;
    add.rn.f32      %f26, %f24, %f26;
    cvt.rni.u16.f32 %rs1, %f26;
    cvt.u64.u32     %rd10, %r18;
    add.s64         %rd11, %rd5, %rd10;
    st.global.u8    [%rd11], %rs1;
    add.u32         %r18, %r18, 1;
    bra             PAINT_BLEND;

PAINT_DONE:
    ret;
}
"#;

#[cfg(test)]
mod tests {
    /// ptxas refuses a module with a character past ASCII anywhere in it —
    /// a `·` in a comment included — and the refusal comes only when the
    /// driver JIT-compiles it, on a machine with a GPU, as "a PTX JIT
    /// compilation failed". Checked here instead, everywhere.
    #[test]
    fn every_module_is_ascii() {
        #[allow(unused_mut)]
        let mut modules = vec![
            ("CONVERT_PTX", super::CONVERT_PTX),
            ("BLEND_PTX", super::BLEND_PTX),
            ("REDACT_PTX", super::REDACT_PTX),
        ];
        #[cfg(feature = "ort-cuda")]
        modules.push(("FIT_PTX", super::FIT_PTX));
        #[cfg(feature = "cuda-visual-tracking")]
        modules.push(("DCF_PTX", super::DCF_PTX));
        for (name, module) in modules {
            for (number, line) in module.lines().enumerate() {
                assert!(line.is_ascii(), "{name}, line {}: {line}", number + 1);
            }
        }
    }
}
