//! The kernels [`super::CudaDriver`] loads, as PTX source the driver
//! JIT-compiles ??kept apart from the Rust that calls them, which reads as
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
