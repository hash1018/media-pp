//! The `buffer -> scale_cuda -> buffersink` graph behind
//! [`crate::elements::CudaScaler`], factored out because
//! [`crate::elements::CudaVideoCompositor`] needs the same thing per layer:
//! resizing a CUDA surface without bringing it back to the CPU.
//!
//! The two callers differ in what drives the output size. `CudaScaler` fixes
//! it at construction; the compositor recomputes it from each layer's
//! rectangle, so it can change at any time. Both are handled the same way —
//! the graph records what it was configured for and rebuilds when any of it
//! stops matching.

use ffmpeg_next::{self as ffmpeg, ffi};

use crate::{
    buffer::release_picture, elements::filter::is_codec_drain_boundary, pool::UnboundObjectPool,
};

use super::cuda_scaler::{CudaScalerError, CudaScalerInterp};

/// One configured graph, plus what it was configured *for*, so a changed
/// input or output can be detected.
struct GraphState {
    /// Owns the filter contexts below; freeing it frees them, so it must
    /// outlive them — which it does, all three living in this struct.
    _graph: ffmpeg::filter::Graph,
    source: ffmpeg::filter::Context,
    sink: ffmpeg::filter::Context,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    /// The exact input pool this graph was configured against. A same-sized
    /// frame from a *different* pool — an upstream decoder rebuilt after a
    /// seek, say — still needs a rebuild, and comparing sizes alone would
    /// miss it.
    ///
    /// The pool itself, not a reference to it: `av_buffer_ref` allocates a
    /// fresh `AVBufferRef` per reference, so every frame in a pool carries a
    /// different one to the same context. Comparing the references finds a
    /// difference on every frame and rebuilds a graph that already matched.
    input_frames_ctx: *const ffi::AVHWFramesContext,
    /// The colorimetry the link was configured with. libavfilter reports a
    /// frame whose tags differ from its link as "video frame properties
    /// changing on the fly" and keeps the link's own values, so a change here
    /// is a rebuild like any other.
    input_color_space: ffi::AVColorSpace,
    input_color_range: ffi::AVColorRange,
}

/// The size to scale *through*, when scaling straight to the requested one
/// would come out blank. `None` when the direct scale is safe, which is
/// almost always.
///
/// # The defect this exists for
///
/// `scale_cuda`'s interpolating kernels produce an entirely zero surface when
/// **exactly one** output dimension equals the corresponding input one.
/// Measured against FFmpeg n8.1.2 on this machine, scaling a 1920x1080 frame:
///
/// ```text
/// -> 640x1080   every byte zero      (height unchanged)
/// -> 1920x720   every byte zero      (width unchanged)
/// -> 1919x1080  every byte zero      (height unchanged)
/// -> 640x1079   correct              (neither unchanged)
/// -> 1920x1080  correct              (both unchanged: the filter's own
///                                     passthrough, which does not scale)
/// ```
///
/// It is `scale_cuda`, not this crate: the same sizes reproduce through the
/// `ffmpeg` command-line tool with no media-pp in the picture, on a graph of
/// `format=nv12,hwupload,scale_cuda=640:1080,hwdownload`. `passthrough=0` does
/// not avoid it and neither does `format=nv12`; only `interp_algo=nearest`
/// does, which is a different kernel and not a quality this crate will silently
/// drop to.
///
/// # Why a detour rather than a nudge
///
/// The obvious workaround — ask for a size one or two pixels off and use part
/// of the result — changes the picture: the scale factor is no longer the one
/// the caller asked for, and the difference is a silent crop or stretch. This
/// keeps the requested size exactly, at the cost of resampling twice for the
/// frames that would otherwise be blank.
///
/// The intermediate has to differ in **both** axes from both ends, or one of
/// the two passes lands on the defect again. Since the destination shares
/// exactly one axis with the source, `+2` on both of the destination's
/// dimensions satisfies that everywhere except where it happens to land on the
/// source's own size, which is why that case is stepped over. `+2` rather than
/// `+1` keeps the parity of an NV12 frame's dimensions.
///
/// # What it costs
///
/// A second resample, on the frames that hit the defect only. In the case this
/// was found in — dragging a layer whose scaled height passes exactly through
/// the capture's own — that is roughly one frame in six hundred, which is the
/// rate `obs-rs` reported the corruption at.
fn detour(input_width: u32, input_height: u32, width: u32, height: u32) -> Option<(u32, u32)> {
    let same_width = width == input_width;
    let same_height = height == input_height;
    // Both unchanged is the filter's passthrough, which is correct; neither
    // unchanged is the ordinary path. Only exactly one is broken.
    if same_width == same_height {
        return None;
    }
    let step = |wanted: u32, source: u32| {
        let mut via = wanted + 2;
        if via == source {
            via += 2;
        }
        via
    };
    Some((step(width, input_width), step(height, input_height)))
}

/// The pool an `AVBufferRef` refers to, rather than the reference itself.
///
/// Returns null for a null reference, which no live pool ever is, so a graph
/// compared against it rebuilds rather than matching by accident.
fn pool_of(frames_ctx: *mut ffi::AVBufferRef) -> *const ffi::AVHWFramesContext {
    if frames_ctx.is_null() {
        return std::ptr::null();
    }
    // SAFETY: a non-null hardware-frames `AVBufferRef` — which is what a
    // validated CUDA frame's `hw_frames_ctx` is — has `data` pointing at its
    // `AVHWFramesContext`, by FFmpeg's own definition. Only the pointer's
    // identity is taken; it is never dereferenced past this.
    unsafe { (*frames_ctx).data as *const ffi::AVHWFramesContext }
}

/// A lazily built `scale_cuda` graph. Construct it, then hand it frames; it
/// builds itself from the first one and rebuilds whenever the input or the
/// requested output stops matching what it was configured for.
pub(crate) struct CudaScaleGraph {
    interp: CudaScalerInterp,
    state: Option<GraphState>,
    /// Reuses only the small CPU-side `AVFrame` wrapper; the scaled surface
    /// itself comes from `scale_cuda`'s own output pool. Same split as
    /// [`crate::elements::CudaUpload`].
    pool: UnboundObjectPool<ffmpeg::frame::Video>,
}

// SAFETY: a filter graph and its contexts have no thread affinity of their
// own — ffmpeg-next already marks `filter::Graph` itself `Send` — and every
// method here takes `&mut self`, so no two threads can drive one graph at
// once.
unsafe impl Send for CudaScaleGraph {}

/// What one call produced. `scale_cuda` emits one frame per frame, but a
/// graph is allowed to hold or emit more than that, so callers drain a list
/// rather than assuming.
pub(crate) type ScaledFrames = Vec<crate::pool::UnboundObjectPoolRef<ffmpeg::frame::Video>>;

impl CudaScaleGraph {
    pub(crate) fn new(interp: CudaScalerInterp) -> Self {
        Self {
            interp,
            state: None,
            // `release_picture` rather than a no-op: a wrapper that held its
            // reference while sitting in the pool would keep a surface out of
            // `scale_cuda`'s own output pool for as long as it sat unused.
            pool: UnboundObjectPool::new(0, ffmpeg::frame::Video::empty, release_picture),
        }
    }

    /// Whether a graph is configured for exactly this input and output. The
    /// caller checks first so it can log the rebuild in its own voice.
    pub(crate) fn matches(
        &self,
        frame: &ffmpeg::frame::Video,
        frames_ctx: *mut ffi::AVBufferRef,
        width: u32,
        height: u32,
    ) -> bool {
        match &self.state {
            Some(state) => {
                state.input_width == frame.width()
                    && state.input_height == frame.height()
                    && state.output_width == width
                    && state.output_height == height
                    && std::ptr::eq(state.input_frames_ctx, pool_of(frames_ctx))
                    && state.input_color_space == frame_color_space(frame)
                    && state.input_color_range == frame_color_range(frame)
            }
            None => false,
        }
    }

    /// Scales one already-validated CUDA frame, rebuilding the graph first if
    /// it does not match. `frames_ctx` is the frame's own hardware frames
    /// context, which is what libavfilter has to be configured against.
    pub(crate) fn scale(
        &mut self,
        frame: &ffmpeg::frame::Video,
        frames_ctx: *mut ffi::AVBufferRef,
        width: u32,
        height: u32,
    ) -> Result<ScaledFrames, CudaScalerError> {
        if !self.matches(frame, frames_ctx, width, height) {
            self.build(frame, frames_ctx, width, height)?;
        }
        // SAFETY: `as_mut_ptr` on a `filter::Context` the `GraphState` above owns,
        // which the line before has just built or confirmed. The pointer is used
        // only until the push below returns, and nothing between the two touches
        // `self.state`.
        let source = unsafe {
            self.state
                .as_mut()
                .expect("built or confirmed matching above")
                .source
                .as_mut_ptr()
        };
        // `av_buffersrc_write_frame`, not `add_frame`: the latter takes over
        // the caller's reference and resets the frame, and this frame is
        // shared — downstream `Arc` clones and the upstream pool both still
        // expect it intact.
        // SAFETY: `source` is that graph's live buffersrc, and `frame` is a live
        // `frame::Video` this only lends: `av_buffersrc_write_frame` takes a
        // reference of its own rather than the caller's, which is exactly why it is
        // used here — see the comment above.
        let code = unsafe { ffi::av_buffersrc_write_frame(source, frame.as_ptr()) };
        if code < 0 {
            return Err(CudaScalerError::BufferSrcPush(code));
        }
        self.drain()
    }

    /// Signals end of input and returns whatever the graph still held.
    /// `scale_cuda` holds nothing back, but a graph that never sees EOF never
    /// reports it either.
    pub(crate) fn flush(&mut self) -> Result<ScaledFrames, CudaScalerError> {
        if self.state.is_none() {
            return Ok(Vec::new());
        }
        // SAFETY: as in `scale` — `as_mut_ptr` on a `filter::Context` owned by the
        // `GraphState` the check above confirmed is present.
        let source = unsafe {
            self.state
                .as_mut()
                .expect("checked above")
                .source
                .as_mut_ptr()
        };
        // SAFETY: `source` is a live buffersrc. A null frame is how this function
        // is defined to signal end of input, not a missing argument.
        let code = unsafe { ffi::av_buffersrc_add_frame_flags(source, std::ptr::null_mut(), 0) };
        if code < 0 {
            return Err(CudaScalerError::BufferSrcPush(code));
        }
        self.drain()
    }

    fn build(
        &mut self,
        frame: &ffmpeg::frame::Video,
        frames_ctx: *mut ffi::AVBufferRef,
        width: u32,
        height: u32,
    ) -> Result<(), CudaScalerError> {
        let mut graph = ffmpeg::filter::Graph::new();

        let buffer =
            ffmpeg::filter::find("buffer").ok_or(CudaScalerError::FilterNotFound("buffer"))?;
        let scale = ffmpeg::filter::find("scale_cuda")
            .ok_or(CudaScalerError::FilterNotFound("scale_cuda"))?;
        let buffersink = ffmpeg::filter::find("buffersink")
            .ok_or(CudaScalerError::FilterNotFound("buffersink"))?;

        // Allocated and initialized in two steps rather than through
        // `Graph::add`, which is `avfilter_graph_create_filter` and therefore
        // initializes the filter with its argument string immediately. Since
        // FFmpeg 7.1 (commit a7fe27f9, "lavfi/buffersrc: validate hw context
        // presence in video_init()") a hardware `pix_fmt` without an
        // `hw_frames_ctx` is rejected at *init*, not at graph-config time, so
        // the frames context has to be in place before the filter is
        // initialized. Everything the argument string used to carry travels in
        // the same `AVBufferSrcParameters` instead.
        // SAFETY: `graph` owns the filter this allocates and frees it on drop even
        // when uninitialized, as the comment below records, so the early returns
        // leak nothing. `buffer` is a live filter descriptor and the name is a
        // literal `CStr`; a failed allocation comes back null and is rejected rather
        // than used.
        let mut source = unsafe {
            let context = ffi::avfilter_graph_alloc_filter(
                graph.as_mut_ptr(),
                buffer.as_ptr(),
                c"in".as_ptr(),
            );
            if context.is_null() {
                return Err(CudaScalerError::BufferSrcAlloc);
            }
            // The graph now owns `context` and frees it on drop, even
            // uninitialized, so an early return below leaks nothing.
            let params = ffi::av_buffersrc_parameters_alloc();
            if params.is_null() {
                return Err(CudaScalerError::BufferSrcParamsAlloc);
            }
            (*params).format = ffi::AVPixelFormat::AV_PIX_FMT_CUDA as i32;
            (*params).width = frame.width() as i32;
            (*params).height = frame.height() as i32;
            // `scale_cuda` neither reorders nor retimes, and nothing between
            // the source and the sink rescales timestamps, so a nominal 1/1
            // time base carries each frame's own pts through numerically
            // unchanged. This graph is not the place that decides what a pts
            // *means*.
            (*params).time_base = ffi::AVRational { num: 1, den: 1 };
            (*params).sample_aspect_ratio = ffi::AVRational { num: 1, den: 1 };
            // `av_buffersrc_parameters_alloc` zeroes the struct, and zero is
            // `AVCOL_SPC_RGB` rather than "unset" — so these are always set
            // explicitly, and set to what the frames actually carry. Leaving
            // the link unspecified made libavfilter report every frame as
            // properties changing on the fly, once per frame, for the whole
            // run of any pipeline whose frames are tagged at all.
            (*params).color_space = frame_color_space(frame);
            (*params).color_range = frame_color_range(frame);
            // This is what tells libavfilter which CUDA pool the frames come
            // from; without it `scale_cuda` has no device to configure itself
            // on. libavfilter takes its own reference, so the frame's context
            // need not outlive this call.
            (*params).hw_frames_ctx = frames_ctx;
            let code = ffi::av_buffersrc_parameters_set(context, params);
            ffi::av_free(params.cast());
            if code < 0 {
                return Err(CudaScalerError::BufferSrcConfig(code));
            }
            let code = ffi::avfilter_init_str(context, std::ptr::null());
            if code < 0 {
                return Err(CudaScalerError::BufferSrcInit(code));
            }
            ffmpeg::filter::Context::wrap(context)
        };

        let mut sink = graph.add(&buffersink, "out", "")?;
        let algo = self.interp.algo();

        // One `scale_cuda`, unless this is the size that filter gets wrong —
        // see [`detour`], which is where the whole explanation lives.
        let mut last = match detour(frame.width(), frame.height(), width, height) {
            None => {
                let mut only = graph.add(
                    &scale,
                    "scale",
                    &format!("w={width}:h={height}:interp_algo={algo}"),
                )?;
                source.link(0, &mut only, 0);
                only
            }
            Some((via_width, via_height)) => {
                let mut first = graph.add(
                    &scale,
                    "scale_via",
                    &format!("w={via_width}:h={via_height}:interp_algo={algo}"),
                )?;
                let mut second = graph.add(
                    &scale,
                    "scale",
                    &format!("w={width}:h={height}:interp_algo={algo}"),
                )?;
                source.link(0, &mut first, 0);
                first.link(0, &mut second, 0);
                second
            }
        };

        last.link(0, &mut sink, 0);
        graph.validate()?;

        self.state = Some(GraphState {
            _graph: graph,
            source,
            sink,
            input_width: frame.width(),
            input_height: frame.height(),
            output_width: width,
            output_height: height,
            input_frames_ctx: pool_of(frames_ctx),
            input_color_space: frame_color_space(frame),
            input_color_range: frame_color_range(frame),
        });
        Ok(())
    }

    /// Pulls everything the graph is willing to produce right now.
    fn drain(&mut self) -> Result<ScaledFrames, CudaScalerError> {
        // SAFETY: `as_mut_ptr` on a `filter::Context` owned by the `GraphState`,
        // which is present because `drain` is only reached after a graph has been
        // built.
        let sink = unsafe {
            self.state
                .as_mut()
                .expect("only called with a configured graph")
                .sink
                .as_mut_ptr()
        };
        let mut scaled = Vec::new();
        loop {
            let mut output = self.pool.get();
            // SAFETY: `ptr` is the pooled frame's own `AVFrame`, and the unref before
            // the pull is what hands its previous surface back — see the comment beside
            // it. `sink` is the live buffersink read out just above.
            let code = unsafe {
                let ptr = output.as_mut_ptr();
                // The pool already released whatever this wrapper held when it
                // came back; this is what makes a wrapper that never went
                // anywhere — one a failed pull left behind — safe to reuse.
                ffi::av_frame_unref(ptr);
                ffi::av_buffersink_get_frame(sink, ptr)
            };
            if code < 0 {
                if is_codec_drain_boundary(&ffmpeg::Error::from(code)) {
                    return Ok(scaled);
                }
                return Err(CudaScalerError::BufferSinkPull(code));
            }
            scaled.push(output);
        }
    }
}

/// The frame's own colorimetry, as libavfilter's C fields rather than
/// `ffmpeg-next`'s enums, which is what the buffersrc parameters take.
fn frame_color_space(frame: &ffmpeg::frame::Video) -> ffi::AVColorSpace {
    // SAFETY: `frame` is a live `frame::Video`, so `as_ptr` yields an
    // initialized `AVFrame` whose `colorspace` is a plain field in it.
    unsafe { (*frame.as_ptr()).colorspace }
}

fn frame_color_range(frame: &ffmpeg::frame::Video) -> ffi::AVColorRange {
    // SAFETY: as `frame_color_space` — a plain field of a live `AVFrame`.
    unsafe { (*frame.as_ptr()).color_range }
}
