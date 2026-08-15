use std::{ffi::c_void, sync::Arc};

use crate::pp_log::{PpLog, pp_error, pp_info};
use ffmpeg_next::{self as ffmpeg, ffi};
use thiserror::Error as ThisError;
use windows::{
    Win32::Graphics::Direct3D11::{D3D11_BIND_SHADER_RESOURCE, ID3D11Device, ID3D11Texture2D},
    core::Interface,
};

use crate::{
    buffer::MediaBuffer,
    control::ControlMsg,
    element::{Element, ElementType, Sink, Source, element_pp_log},
    pad::SrcPad,
    pool::UnboundObjectPool,
};

use crate::elements::filter::is_codec_drain_boundary;

/// Mirrors of the D3D11VA-specific structs from FFmpeg's
/// `libavutil/hwcontext_d3d11va.h` (verified against the actual header
/// linked on this machine, `C:\ffmpeg-lib\include\libavutil\hwcontext_d3d11va.h`
/// — not from memory; see [`crate::elements::D3d12vaDecoder`]'s own D3D12
/// mirrors for the sibling case and why this hand-mirroring is needed at
/// all: `ffmpeg-sys-next` only binds the type-agnostic
/// `AVHWDeviceContext`/`AVHWFramesContext`). COM pointers are kept as
/// `*mut c_void` so this struct's layout depends only on pointer size, not
/// on `windows-rs`'s own representation. If a future FFmpeg version
/// changes this header's layout, these silently go stale — there's no
/// compile-time check tying them together.
///
/// D3D11VA's actual per-frame representation is notably **not** shaped
/// like D3D12's `AVD3D12VAFrame` (a wrapper struct pointed to by
/// `AVFrame.data[0]`): per `AVD3D11FrameDescriptor`'s own doc comment in
/// the header, a normal (non-custom-pool) D3D11VA frame stores the
/// `ID3D11Texture2D*` directly in `AVFrame.data[0]` and the array-texture
/// slice index directly in `AVFrame.data[1]` (cast from `intptr_t`) — no
/// wrapper struct dereference needed at all, and no fence/sync info here
/// either (see [`crate::elements::D3d11Renderer`]'s own docs on why: this
/// crate only ever uses a single shared `ID3D11Device`+context, so there's
/// nothing to synchronize explicitly).
#[repr(C)]
struct AVD3D11VADeviceContext {
    device: *mut c_void,
    device_context: *mut c_void,
    video_device: *mut c_void,
    video_context: *mut c_void,
    lock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    unlock: Option<unsafe extern "C" fn(lock_ctx: *mut c_void)>,
    lock_ctx: *mut c_void,
}

/// `AVHWFramesContext.hwctx` once cast for `AV_HWDEVICE_TYPE_D3D11VA` — used
/// **only** by [`configure_hw_frames_ctx`] to OR in `D3D11_BIND_SHADER_RESOURCE`
/// onto whatever `bind_flags` libavcodec's own `avcodec_get_hw_frames_parameters`
/// already set (typically just `D3D11_BIND_DECODER`, enough for the decode
/// API itself but not for `CreateShaderResourceView`, which is what
/// [`crate::elements::D3d11Renderer`] needs to draw a decoded frame at all).
/// This is deliberately a much narrower use of this struct than an earlier,
/// abandoned version of this module made: that version built an entire
/// `AVHWFramesContext`+`AVD3D11VAFramesContext` from scratch by hand and
/// corrupted memory badly enough to trip `/GS` (see [`wrap_d3d11_texture`]'s
/// own docs on that history). Here, every field except this one `bind_flags`
/// write is populated by FFmpeg's own C code (`avcodec_get_hw_frames_parameters`
/// for the base `AVHWFramesContext`, and its own D3D11VA-specific internals
/// for whatever it already put in `hwctx`) — this function only ever reads
/// or ORs into one already-initialized `u32`, never constructs this struct
/// itself.
#[repr(C)]
struct AVD3D11VAFramesContext {
    texture: *mut c_void,
    bind_flags: u32,
    misc_flags: u32,
    texture_infos: *mut c_void,
}

/// Errors specific to `D3d11Decoder`. Converts into the crate-wide
/// `Error` via `?` (see [`crate::error::Error`]).
#[derive(Debug, ThisError)]
pub enum D3d11vaDecoderError {
    #[error("unsupported media type: {0:?} (D3D11VA decode is video-only)")]
    UnsupportedMediaType(ffmpeg::media::Type),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg::Error),

    #[error("failed to create D3D11VA hw device context (code {0})")]
    HwDeviceInit(i32),

    #[error(
        "decoder did not select the D3D11VA pixel format — hardware decode \
         unavailable for this stream/GPU/driver"
    )]
    HwAccelUnavailable,
}

/// Decodes one video stream's `Packet`s into GPU-resident `Video` frames
/// via D3D11VA hardware acceleration — the D3D11 sibling of
/// [`crate::elements::D3d12vaDecoder`], for a pipeline built entirely on
/// one shared `ID3D11Device` (see [`crate::elements::D3d11Renderer`]'s own
/// docs on why that means no explicit fence/sync is needed anywhere in
/// this stack, unlike the D3D12 side). A `Filter`, same shape as
/// `SwDecoder`/`D3d12vaDecoder`.
pub struct D3d11Decoder {
    pp_log: PpLog,
    name: Arc<str>,
    decoder: ffmpeg::decoder::Video,
    hw_device_ctx: *mut ffi::AVBufferRef,
    pad: SrcPad,
    /// Reused across every decoded frame — see [`UnboundObjectPool`]'s
    /// docs; same reasoning as `D3d12vaDecoder`'s own `pool` field (the
    /// GPU texture itself is already pooled by ffmpeg's own hw frames
    /// context, this only reuses the small CPU-side `AVFrame` wrapper).
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: `hw_device_ctx` is a heap-allocated FFmpeg buffer with no
// thread affinity; `decoder`'s own `Send` already covers the rest.
// `&mut self` on every method that touches it rules out concurrent
// access from multiple threads — same reasoning as `D3d12vaDecoder`.
unsafe impl Send for D3d11Decoder {}

impl D3d11Decoder {
    /// `device` must outlive this decoder (and, transitively, every frame
    /// it produces that's still alive downstream), and must be the same
    /// `ID3D11Device` every other D3D11 element in this pipeline shares —
    /// see [`crate::elements::D3d11Renderer`]'s own docs on why this whole
    /// stack requires exactly one shared device/context, not just a
    /// same-adapter one.
    ///
    /// `extra_hw_frames` sets `AVCodecContext.extra_hw_frames` — how many
    /// *additional* decode surfaces to allocate beyond what the codec's own
    /// reference-frame count strictly requires. Unlike
    /// [`crate::elements::D3d12vaDecoder`] (no equivalent parameter needed),
    /// D3D11VA's decode surface pool is a **fixed-size** texture array,
    /// sized once at `av_hwframe_ctx_init()` time and never grown — every
    /// decoded frame still alive downstream (sitting in a
    /// [`crate::queue::Queue`], held by a slow renderer, ...) keeps one pool
    /// slot occupied, and once the pool runs out, decode itself starts
    /// failing (`AVERROR(ENOMEM)`, "Static surface pool size exceeded" in
    /// the log) instead of just blocking. Pass at least as many extra frames
    /// as the deepest queue/buffer this decoder's output can pile up in
    /// (e.g. match a downstream `ChainBuilder::queue` depth) — too few
    /// reproduces exactly that failure under real playback, not just in
    /// theory.
    pub fn new(
        name: impl Into<String>,
        params: ffmpeg::codec::Parameters,
        device: &ID3D11Device,
        extra_hw_frames: i32,
    ) -> Result<Self, D3d11vaDecoderError> {
        let name: Arc<str> = name.into().into();
        let pp_log = element_pp_log(ElementType::D3d11Decoder, &name, None);

        let hw_device_ctx =
            unsafe { create_hw_device_ctx(device) }.map_err(D3d11vaDecoderError::HwDeviceInit)?;

        let mut context = match ffmpeg::codec::context::Context::from_parameters(params) {
            Ok(context) => context,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                return Err(error.into());
            }
        };
        if context.medium() != ffmpeg::media::Type::Video {
            unsafe { free_buffer(hw_device_ctx) };
            return Err(D3d11vaDecoderError::UnsupportedMediaType(context.medium()));
        }
        unsafe {
            let ctx_ptr = context.as_mut_ptr();
            (*ctx_ptr).hw_device_ctx = ffi::av_buffer_ref(hw_device_ctx);
            (*ctx_ptr).get_format = Some(get_format);
            (*ctx_ptr).extra_hw_frames = extra_hw_frames;
        }

        let decoder = match context.decoder().video() {
            Ok(decoder) => decoder,
            Err(error) => {
                unsafe { free_buffer(hw_device_ctx) };
                return Err(error.into());
            }
        };

        let pad = SrcPad::new(format!("{name}_src"));
        let pool = UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, |_| {});
        pp_info!(pp_log: &pp_log, "opened: codec={:?}", decoder.id());
        Ok(Self {
            name,
            pp_log,
            decoder,
            hw_device_ctx,
            pad,
            pool,
        })
    }

    fn drain(&mut self) -> crate::error::Result<()> {
        let mut frame = self.pool.get();
        loop {
            match self.decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    if frame.format() != ffmpeg::format::Pixel::D3D11 {
                        pp_error!(self, "decoder did not select the D3D11VA pixel format");
                        return Err(D3d11vaDecoderError::HwAccelUnavailable.into());
                    }
                    self.pad.push(MediaBuffer::Video(Arc::new(frame)))?;
                    frame = self.pool.get();
                }
                Err(error) if is_codec_drain_boundary(&error) => break,
                Err(error) => return Err(D3d11vaDecoderError::from(error).into()),
            }
        }
        Ok(())
    }
}

impl Element for D3d11Decoder {
    fn name(&self) -> Arc<str> {
        self.name.clone()
    }

    fn element_type(&self) -> ElementType {
        ElementType::D3d11Decoder
    }

    fn pp_log(&self) -> &PpLog {
        &self.pp_log
    }

    fn pp_log_mut(&mut self) -> &mut PpLog {
        &mut self.pp_log
    }
}

impl Source for D3d11Decoder {
    fn src_pads(&mut self) -> &mut [SrcPad] {
        std::slice::from_mut(&mut self.pad)
    }
}

impl Sink for D3d11Decoder {
    fn consume(&mut self, buf: MediaBuffer) -> crate::error::Result<()> {
        match buf {
            MediaBuffer::Packet(packet) => {
                self.decoder
                    .send_packet(&*packet)
                    .inspect_err(|error| pp_error!(self, "send_packet failed: {error}"))
                    .map_err(D3d11vaDecoderError::from)?;
                self.drain()
            }
            MediaBuffer::Eos => {
                self.decoder
                    .send_eof()
                    .inspect_err(|error| pp_error!(self, "send_eof failed: {error}"))
                    .map_err(D3d11vaDecoderError::from)?;
                self.drain()?;
                self.pad.push(MediaBuffer::Eos)
            }
            other => {
                let _ = other;
                Ok(())
            }
        }
    }

    fn control(&mut self, msg: ControlMsg) -> crate::error::Result<()> {
        // Same reasoning as `D3d12vaDecoder::control`: nothing to do on
        // `Stop` (the hw device context is freed in `Drop`), flush
        // reference-frame state on `Seek`.
        if let ControlMsg::Seek(_) = msg {
            self.decoder.flush();
        }
        self.pad.control(msg)
    }
}

impl Drop for D3d11Decoder {
    fn drop(&mut self) {
        pp_info!(self, "dropped: freeing hw_device_ctx");
        unsafe { free_buffer(self.hw_device_ctx) };
    }
}

/// Shared with [`crate::elements::D3d11Upload`] and
/// [`crate::elements::DxgiCaptureSource`]'s GPU capture mode — every
/// producer of a `Pixel::D3D11` frame in this crate goes through this same
/// device-context setup, which is what lets [`d3d11va_texture`] read any
/// of their frames identically. Returns a raw `AVERROR` code rather than
/// `D3d11vaDecoderError` so it doesn't tie those callers to this module's
/// own error type; each call site maps it into its own error variant.
pub(crate) unsafe fn create_hw_device_ctx(
    device: &ID3D11Device,
) -> Result<*mut ffi::AVBufferRef, i32> {
    unsafe {
        let buf = ffi::av_hwdevice_ctx_alloc(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA);
        if buf.is_null() {
            return Err(-1);
        }

        let hw_device_ctx = (*buf).data as *mut ffi::AVHWDeviceContext;
        let d3d11_ctx = (*hw_device_ctx).hwctx as *mut AVD3D11VADeviceContext;
        // AddRef'd, not borrowed — unlike the D3D12 sibling
        // (`elements::filter::decoder::d3d12va_decoder`'s `create_hw_device_ctx`),
        // whose comment this one used to copy verbatim. Verified against
        // the real `libavutil/hwcontext_d3d11va.c` (n8.0): `d3d11va_device_uninit`
        // unconditionally calls `ID3D11Device_Release(device_hwctx->device)`
        // when the hw device context is freed — i.e. it assumes ownership
        // of whatever's in this field, not a borrow. `d3d12va_device_uninit`
        // does the opposite (never releases the caller-provided device),
        // which is what the old comment's claim was actually true for.
        // Handing over a bare `device.as_raw()` here (no `AddRef`) meant
        // FFmpeg's own `Release()` decremented a reference count `device`'s
        // own Rust-side ownership still believed it held — a genuine
        // one-too-many `Release()` that didn't crash immediately, only
        // later, once every other reference (including this crate's own
        // final `ID3D11Device` drop) had also released, at which point the
        // COM object was already gone (`STATUS_ACCESS_VIOLATION`/
        // `STATUS_BREAKPOINT` from the debug heap, depending on the run).
        // `.clone()` here gives FFmpeg its own independent, genuinely
        // owned reference to release — `device` is the header's one
        // mandatory field; `device_context`/`video_device`/`video_context`
        // are all derived from it on init if left unset.
        (*d3d11_ctx).device = device.clone().into_raw();

        let result = ffi::av_hwdevice_ctx_init(buf);
        if result < 0 {
            free_buffer(buf);
            return Err(result);
        }

        Ok(buf)
    }
}

/// Shared with [`crate::elements::D3d11Upload`]/[`crate::elements::DxgiCaptureSource`]
/// — see [`create_hw_device_ctx`]'s own docs.
pub(crate) unsafe fn free_buffer(mut buf: *mut ffi::AVBufferRef) {
    unsafe { ffi::av_buffer_unref(&mut buf) };
}

unsafe extern "C" fn release_d3d11_texture(_opaque: *mut c_void, data: *mut u8) {
    // `data` is the exact raw COM pointer `wrap_d3d11_texture` handed off
    // via `Interface::into_raw` — reconstructing it here and letting it
    // drop is what actually releases the texture, exactly once, whenever
    // FFmpeg's own refcounting (`av_frame_unref`/the last
    // `Arc<UnboundObjectPoolRef<..>>` clone dropping) determines nothing
    // references this buffer anymore.
    unsafe {
        drop(ID3D11Texture2D::from_raw(data as *mut c_void));
    }
}

/// Wraps `texture` (a COM reference this function takes ownership of) as a
/// `Pixel::D3D11` `ffmpeg::frame::Video`, *without* going through FFmpeg's
/// own `AVHWDeviceContext`/`AVHWFramesContext` machinery at all —
/// `av_hwframe_ctx_init`'s real D3D11VA implementation
/// (`libavutil/hwcontext_d3d11va.c`, verified against the exact `n8.0`
/// source matching this project's linked FFmpeg) turned out to be unsafe
/// to drive from a hand-mirrored `AVD3D11VAFramesContext*` — an earlier
/// version of this function did exactly that and corrupted memory badly
/// enough to trip `/GS` (`STATUS_STACK_BUFFER_OVERRUN`), for a reason not
/// fully root-caused. [`crate::elements::D3d11Upload`] and
/// [`crate::elements::DxgiCaptureSource`]'s GPU capture mode build their
/// own `ID3D11Texture2D` directly via plain `windows-rs` calls instead
/// (safe, ordinary D3D11 API usage, no struct-layout guessing) and use
/// this function only to make the *result* look like a normal
/// `Pixel::D3D11` frame to the rest of this crate (in particular
/// [`d3d11va_texture`], which reads any `Pixel::D3D11` frame the same way
/// regardless of how it was produced).
///
/// Uses `av_buffer_create` — a plain, stable, generically-documented
/// libavutil API, not a hand-mirrored struct — to give the resulting
/// `AVFrame` real reference-counted ownership of `texture`: FFmpeg calls
/// [`release_d3d11_texture`] (which drops the COM reference) exactly once,
/// once the last reference — including any downstream
/// `Arc<UnboundObjectPoolRef<..>>` clone — is gone. [`D3d11Decoder`] doesn't
/// need this: real hardware decode still goes through
/// `av_hwframe_get_buffer`/libavcodec's own D3D11VA hwaccel internally
/// allocating its frames context (see [`configure_hw_frames_ctx`] for the
/// one thing this crate *does* still need to customize about it — bind
/// flags, not the struct layout that corrupted memory here).
pub(crate) fn wrap_d3d11_texture(
    texture: ID3D11Texture2D,
    width: u32,
    height: u32,
) -> ffmpeg::frame::Video {
    let mut frame = ffmpeg::frame::Video::empty();
    let raw = texture.into_raw();
    unsafe {
        let ptr = frame.as_mut_ptr();
        (*ptr).format = ffi::AVPixelFormat::AV_PIX_FMT_D3D11 as i32;
        (*ptr).width = width as i32;
        (*ptr).height = height as i32;
        // Per `AVD3D11FrameDescriptor`'s own doc comment (see this
        // module's top-of-file docs): the texture pointer goes directly
        // in `data[0]`, the array-texture slice index directly in
        // `data[1]` — `0` here since this is never an array texture.
        (*ptr).data[0] = raw as *mut u8;
        (*ptr).data[1] = std::ptr::null_mut();
        let buf = ffi::av_buffer_create(
            raw as *mut u8,
            std::mem::size_of::<*mut c_void>(),
            Some(release_d3d11_texture),
            std::ptr::null_mut(),
            0,
        );
        if buf.is_null() {
            // `av_buffer_create`'s own allocation failed — nothing will
            // ever call `release_d3d11_texture` for the COM reference
            // `into_raw()` already handed off above, and `data[0]` must
            // not keep pointing at it (whatever reads this frame next
            // would treat it as a still-live texture). Release it here
            // ourselves instead of leaking it or leaving a dangling
            // pointer behind, then fail loudly: this is an allocation
            // failure with no sane per-frame recovery, not a condition
            // worth threading a `Result` through every one of this
            // function's callers for.
            (*ptr).data[0] = std::ptr::null_mut();
            drop(ID3D11Texture2D::from_raw(raw));
            panic!("av_buffer_create failed to allocate a D3D11 texture buffer wrapper");
        }
        (*ptr).buf[0] = buf;
    }
    frame
}

/// Builds and initializes `avctx->hw_frames_ctx` for D3D11VA decode, with
/// `D3D11_BIND_SHADER_RESOURCE` added on top of whatever libavcodec's own
/// `avcodec_get_hw_frames_parameters` already sets — see
/// [`AVD3D11VAFramesContext`]'s own docs on why that's the one thing this
/// crate needs to customize. Must run from inside `get_format` (FFmpeg's
/// own documented contract for this API — see `avcodec_get_hw_frames_parameters`'s
/// doc comment) with the exact same `avctx` `get_format` received.
///
/// Without this, decode still succeeds and produces valid `Pixel::D3D11`
/// frames (this is what an earlier version of this function — which just
/// picked the format and returned, leaving `hw_frames_ctx` to libavcodec's
/// own automatic fallback path — actually did), but
/// `ID3D11Device::CreateShaderResourceView` fails on the resulting textures:
/// libavcodec's own default `bind_flags` only cover what the D3D11 video
/// decode API itself needs (`D3D11_BIND_DECODER`), not sampling from a
/// pixel shader.
unsafe fn configure_hw_frames_ctx(ctx: *mut ffi::AVCodecContext) -> Result<(), i32> {
    unsafe {
        let mut frames_ref: *mut ffi::AVBufferRef = std::ptr::null_mut();
        let result = ffi::avcodec_get_hw_frames_parameters(
            ctx,
            (*ctx).hw_device_ctx,
            ffi::AVPixelFormat::AV_PIX_FMT_D3D11,
            &mut frames_ref,
        );
        if result < 0 {
            return Err(result);
        }

        let frames_ctx = (*frames_ref).data as *mut ffi::AVHWFramesContext;
        let d3d11_frames = (*frames_ctx).hwctx as *mut AVD3D11VAFramesContext;
        (*d3d11_frames).bind_flags |= D3D11_BIND_SHADER_RESOURCE.0 as u32;

        let result = ffi::av_hwframe_ctx_init(frames_ref);
        if result < 0 {
            ffi::av_buffer_unref(&mut frames_ref);
            return Err(result);
        }

        // Transfers ownership of `frames_ref` to `avctx` — matches
        // `avcodec_get_hw_frames_parameters`'s own documented contract
        // ("the user's responsibility to ... set
        // AVCodecContext.hw_frames_ctx to it").
        (*ctx).hw_frames_ctx = frames_ref;
        Ok(())
    }
}

unsafe extern "C" fn get_format(
    ctx: *mut ffi::AVCodecContext,
    mut fmt: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe {
        while *fmt != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *fmt == ffi::AVPixelFormat::AV_PIX_FMT_D3D11 {
                if configure_hw_frames_ctx(ctx).is_ok() {
                    return ffi::AVPixelFormat::AV_PIX_FMT_D3D11;
                }
                // Configuration failed (e.g. this GPU/driver doesn't
                // support sampling decoded D3D11VA surfaces) — fall
                // through to whatever other format the decoder offers,
                // same as if D3D11 had never been in `fmt`'s list at all.
                fmt = fmt.add(1);
                continue;
            }
            fmt = fmt.add(1);
        }
        ffi::AVPixelFormat::AV_PIX_FMT_NONE
    }
}

/// Extracts `(texture, array_index)` from a `Pixel::D3D11` frame — `None`
/// if `frame` isn't actually a D3D11VA frame. Unlike
/// [`crate::elements::D3d12vaDecoder`]'s `d3d12va_texture` (which
/// dereferences a wrapper struct pointed to by `data[0]`), a normal
/// D3D11VA frame stores the `ID3D11Texture2D*` directly in `data[0]` and
/// the array-texture slice index directly in `data[1]` — see this
/// module's own doc comment on `AVD3D11VADeviceContext` for why. `texture`
/// is a borrowed raw `ID3D11Texture2D*` — still owned by `frame`'s own hw
/// frame pool reference; the caller must `AddRef` (e.g. via
/// `Interface::from_raw_borrowed(..).clone()`) before holding onto it
/// independently of `frame`.
pub(crate) fn d3d11va_texture(frame: &ffmpeg::frame::Video) -> Option<(*mut c_void, isize)> {
    if frame.format() != ffmpeg::format::Pixel::D3D11 {
        return None;
    }
    unsafe {
        let ptr = frame.as_ptr();
        Some(((*ptr).data[0] as *mut c_void, (*ptr).data[1] as isize))
    }
}

#[cfg(test)]
mod tests {
    use windows::Win32::Graphics::{
        Direct3D::D3D_DRIVER_TYPE_HARDWARE,
        Direct3D11::{D3D11_SDK_VERSION, D3D11CreateDevice},
    };

    use super::*;

    /// Regression test for a real crash (`STATUS_ACCESS_VIOLATION`/
    /// `STATUS_BREAKPOINT`, depending on the run) found via
    /// `d3d11_decode_render`: `create_hw_device_ctx` used to hand FFmpeg a
    /// bare, non-`AddRef`'d `ID3D11Device*` — but
    /// `libavutil/hwcontext_d3d11va.c`'s `d3d11va_device_uninit`
    /// unconditionally `Release()`s whatever's in that field, unlike the
    /// D3D12 sibling (which never releases the caller-provided device).
    /// That one extra `Release()` didn't crash immediately — only later,
    /// once every *other* reference (including this test's own final
    /// `device` drop) had also released and the COM object was already
    /// gone. A real pipeline run masked exactly how late "later" was
    /// (looked like a clean exit until the process teardown itself);
    /// decoding a full file end-to-end and then dropping everything, with
    /// nothing downstream holding frames open, is what actually reproduces
    /// it deterministically.
    #[test]
    fn decodes_a_full_file_and_tears_down_cleanly() {
        let mut device = None;
        let result = unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                Default::default(),
                Default::default(),
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                None,
            )
        };
        let Ok(()) = result else {
            eprintln!("skipping: D3D11CreateDevice failed on this machine: {result:?}");
            return;
        };
        let device = device.expect("D3D11CreateDevice succeeded without producing a device");

        let Some(path) = crate::test_support::try_test_video() else {
            return;
        };
        let mut input = ffmpeg::format::input(&path).expect("failed to open test video");
        let video_stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .expect("no video stream");
        let video_index = video_stream.index();
        let params = video_stream.parameters();

        let mut decoder = D3d11Decoder::new("test-decoder", params, &device, 32)
            .expect("failed to open D3D11VA decoder");

        for (stream, packet) in input.packets() {
            if stream.index() != video_index {
                continue;
            }
            decoder
                .consume(MediaBuffer::Packet(Arc::new(packet)))
                .expect("consume(Packet) failed");
        }
        decoder
            .consume(MediaBuffer::Eos)
            .expect("consume(Eos) failed");

        // The crash this test guards against only ever showed up here,
        // on this final drop — see this test's own docs.
        drop(decoder);
        drop(input);
        drop(device);
    }
}
