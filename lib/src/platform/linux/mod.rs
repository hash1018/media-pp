#[cfg(any(
    feature = "pipewire-audio-capture",
    feature = "pipewire-audio-renderer"
))]
pub(crate) mod pipewire;

/// DMA-BUF -> CUDA import, for `PipeWireScreenCaptureSource`'s GPU capture
/// mode. Needs both features: the buffers come from the PipeWire capture and
/// land in a CUDA frame.
#[cfg(all(feature = "pipewire-screen-capture", feature = "cuda"))]
pub(crate) mod dmabuf_cuda;

/// Camera enumeration, for `V4l2CaptureSource`'s own picker.
#[cfg(feature = "v4l2-capture")]
pub(crate) mod v4l2;
